"""Generate standalone method diagnostics from frozen FYP outputs.

The script does not train or re-score models. It reads stored pair-metric CSVs
and, where available, frozen dense prediction caches to create method-specific
figures that are not mixed with cross-method comparison plots.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd


REGIONS = [
    ("root", "Root"),
    ("dorsum", "Dorsum"),
    ("tip", "Tip"),
    ("alar_left", "Alar L"),
    ("alar_right", "Alar R"),
]


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    section: str
    pair_metrics: str
    subunit_method: str
    dense_prediction: str | None = None
    spatial_title: str | None = None


METHODS = {
    "hybrid": MethodSpec(
        key="hybrid",
        label="Hybrid",
        section="M3 Hybrid diagnostics",
        pair_metrics=(
            "outputs/unified_strict/safe_fusion/strict_rescore/pair_metrics/"
            "identity_bootstrap_pair_metrics_global_hybrid.csv"
        ),
        subunit_method="hybrid_alpha_0.1",
        dense_prediction="outputs/unified_strict/safe_fusion/cache_test/global_hybrid.npy",
        spatial_title="Hybrid spatial diagnostics",
    ),
    "laplacian": MethodSpec(
        key="laplacian",
        label="Laplacian",
        section="M4 Geometry diagnostics",
        pair_metrics="outputs/unified_strict/main_table/pair_metrics/identity_bootstrap_pair_metrics_laplacian_handles.csv",
        subunit_method="laplacian",
    ),
    "arap": MethodSpec(
        key="arap",
        label="ARAP",
        section="M4 Geometry diagnostics",
        pair_metrics=(
            "outputs/unified_strict/safe_fusion/strict_rescore/pair_metrics/"
            "identity_bootstrap_pair_metrics_arap.csv"
        ),
        subunit_method="arap",
        dense_prediction="outputs/unified_strict/safe_fusion/cache_test/arap.npy",
        spatial_title="ARAP spatial diagnostics",
    ),
    "bilaplacian": MethodSpec(
        key="bilaplacian",
        label="Bi-Laplacian",
        section="M4 Geometry diagnostics",
        pair_metrics="outputs/unified_strict/main_table/pair_metrics/identity_bootstrap_pair_metrics_bilaplacian_handles.csv",
        subunit_method="bilaplacian",
    ),
    "safe_fusion": MethodSpec(
        key="safe_fusion",
        label="Safe Fusion",
        section="M6 Safe Fusion diagnostics",
        pair_metrics=(
            "outputs/unified_strict/safe_fusion/strict_rescore/pair_metrics/"
            "identity_bootstrap_pair_metrics_safe_fusion.csv"
        ),
        subunit_method="safe_fusion",
        dense_prediction="outputs/unified_strict/safe_fusion/evaluation/test_predictions/safe_fusion.npy",
        spatial_title="Safe Fusion spatial diagnostics",
    ),
}


def add_code_path(project: Path) -> None:
    code = project / "code"
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))


def split_counts(project: Path) -> tuple[int, int, int]:
    add_code_path(project)
    try:
        from data import load_rows

        rows, _ = load_rows(project / "data")
    except Exception:
        return 676, 70, 100
    train = sum(str(r.get("split")) in {"clean-prior train", "train pool"} for r in rows)
    val = sum(str(r.get("split")) == "clean-prior validation" for r in rows)
    test = sum(str(r.get("split")) == "main test" for r in rows)
    return int(train), int(val), int(test)


def load_pairs_json(path: Path) -> list[tuple[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("pairs", raw)
    pairs: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, dict):
            pairs.append((str(item["source_id"]), str(item["target_id"])))
        else:
            pairs.append((str(item[0]), str(item[1])))
    return pairs


def edge_index(faces: np.ndarray) -> np.ndarray:
    raw = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    raw.sort(axis=1)
    return np.unique(raw, axis=0).astype(np.int64)


def fold_mask(source: np.ndarray, edited: np.ndarray, faces: np.ndarray) -> np.ndarray:
    from safe_fusion import signed_fold_indicator

    det, _ = signed_fold_indicator(source, edited, faces)
    return det < 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 260,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
        }
    )


def meta_for(spec: MethodSpec, seed: int, split: str, counts: tuple[int, int, int], n_pairs: int, config: dict) -> dict:
    n_train, n_val, n_test = counts
    return {
        "seed": int(seed),
        "split": str(split),
        "n_train_ids": int(n_train),
        "n_val_ids": int(n_val),
        "n_test_ids": int(n_test),
        "n_pairs": int(n_pairs),
        "method": spec.key,
        "config": json.dumps(config, sort_keys=True, ensure_ascii=False),
    }


def metric_summary(df: pd.DataFrame, spec: MethodSpec, base_meta: dict) -> list[dict]:
    metrics = [
        "roi_rmse",
        "dorsum_rmse",
        "tip_rmse",
        "edge_strain_p95",
        "normal_flip_pct",
        "abs_flip_pct",
        "missed_flip_pct",
    ]
    rows = []
    for metric in metrics:
        if metric not in df:
            continue
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        rows.append(
            {
                **base_meta,
                "section": spec.section,
                "display_name": spec.label,
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "p05": float(values.quantile(0.05)),
                "p50": float(values.quantile(0.50)),
                "p95": float(values.quantile(0.95)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return rows


def plot_dashboard(df: pd.DataFrame, spec: MethodSpec, out_dir: Path) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.2), constrained_layout=True)
    fig.suptitle(f"{spec.label} diagnostics", fontsize=12, fontweight="semibold")
    panels = [
        ("roi_rmse", "ROI RMSE", "#2A9D8F"),
        ("normal_flip_pct", "Normal flips (%)", "#E76F51"),
        ("edge_strain_p95", "Edge strain p95", "#5E81AC"),
    ]
    for ax, (metric, title, color) in zip(axes.flat[:3], panels):
        values = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy()
        ax.hist(values, bins=42, color=color, alpha=0.82, edgecolor="white", linewidth=0.35)
        mean = float(np.mean(values))
        p95 = float(np.percentile(values, 95))
        ax.axvline(mean, color="black", linewidth=1.2, label=f"mean {mean:.3g}")
        ax.axvline(p95, color="#D55E00", linewidth=1.1, linestyle="--", label=f"p95 {p95:.3g}")
        ax.set_title(title)
        ax.set_ylabel("Pairs")
        ax.legend(frameon=False)

    ax = axes.flat[3]
    x = pd.to_numeric(df["roi_rmse"], errors="coerce")
    y = pd.to_numeric(df["normal_flip_pct"], errors="coerce")
    c = pd.to_numeric(df["edge_strain_p95"], errors="coerce")
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    sc = ax.scatter(x[finite], y[finite], c=c[finite], s=6, alpha=0.34, cmap="viridis", linewidths=0)
    ax.set_title("Error-risk scatter")
    ax.set_xlabel("ROI RMSE")
    ax.set_ylabel("Normal flips (%)")
    fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02, label="strain p95")

    png = out_dir / f"fig_{spec.key}_standalone_dashboard.png"
    pdf = out_dir / f"fig_{spec.key}_standalone_dashboard.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_tail_profile(df: pd.DataFrame, spec: MethodSpec, out_dir: Path) -> tuple[Path, Path]:
    metrics = [
        ("roi_rmse", "ROI RMSE", "#2A9D8F"),
        ("normal_flip_pct", "Normal flips (%)", "#E76F51"),
        ("edge_strain_p95", "Edge strain p95", "#5E81AC"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.65), constrained_layout=True)
    fig.suptitle(f"{spec.label} tail profile", fontsize=12, fontweight="semibold")
    for ax, (metric, title, color) in zip(axes, metrics):
        values = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy()
        stats = [np.percentile(values, 50), np.percentile(values, 95), np.max(values)]
        bars = ax.bar(["median", "p95", "max"], stats, color=color, alpha=0.88, edgecolor="white", linewidth=0.6)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        for bar, val in zip(bars, stats):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3g}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    png = out_dir / f"fig_{spec.key}_tail_profile.png"
    pdf = out_dir / f"fig_{spec.key}_tail_profile.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def available_learning_scales(project: Path) -> list[int]:
    path = project / "results/learning_curve_refactor/learning_curve_master_table.csv"
    if path.exists():
        df = pd.read_csv(path)
        scales = sorted(int(x) for x in df["scale"].dropna().unique() if int(x) > 0)
        if scales:
            return scales
    return [30, 60, 120, 240, 480, 676]


def plot_learning_curve_or_reference(
    project: Path,
    df: pd.DataFrame,
    spec: MethodSpec,
    out_dir: Path,
    base_meta: dict,
) -> tuple[Path, Path, Path, str]:
    master_path = project / "results/learning_curve_refactor/learning_curve_master_table.csv"
    curve_rows: list[dict] = []
    curve_type = "non_learning_reference"

    if spec.key == "hybrid" and master_path.exists():
        master = pd.read_csv(master_path)
        learned = master[
            (master["curve"] == "primary")
            & (master["method"] == spec.key)
            & (master["metric"].isin(["roi_rmse", "normal_flip_pct"]))
        ].copy()
    else:
        learned = pd.DataFrame()

    if not learned.empty:
        curve_type = "learned_scale_curve"
        for _, row in learned.iterrows():
            curve_rows.append(
                {
                    **base_meta,
                    "section": spec.section,
                    "display_name": spec.label,
                    "curve_type": curve_type,
                    "scale": int(row["scale"]),
                    "metric": str(row["metric"]),
                    "mean": float(row["mean"]),
                    "ci95_low": float(row["ci95_low"]),
                    "ci95_high": float(row["ci95_high"]),
                    "n_chains": int(row["n_chains"]),
                    "source_table": str(master_path.relative_to(project)),
                }
            )
    else:
        scales = available_learning_scales(project)
        stats = {
            "roi_rmse": float(pd.to_numeric(df["roi_rmse"], errors="coerce").mean()),
            "normal_flip_pct": float(pd.to_numeric(df["normal_flip_pct"], errors="coerce").mean()),
        }
        for metric, value in stats.items():
            for scale in scales:
                curve_rows.append(
                    {
                        **base_meta,
                        "section": spec.section,
                        "display_name": spec.label,
                        "curve_type": curve_type,
                        "scale": int(scale),
                        "metric": metric,
                        "mean": value,
                        "ci95_low": value,
                        "ci95_high": value,
                        "n_chains": 0,
                        "source_table": base_meta["config"],
                    }
                )

    curve_df = pd.DataFrame(curve_rows)
    csv_path = out_dir / f"{spec.key}_learning_curve.csv"
    curve_df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85), constrained_layout=True)
    title_suffix = "learning curve" if curve_type == "learned_scale_curve" else "reference curve"
    fig.suptitle(f"{spec.label} {title_suffix}", fontsize=12, fontweight="semibold")
    for ax, metric, label, color in [
        (axes[0], "roi_rmse", "ROI RMSE", "#2A9D8F"),
        (axes[1], "normal_flip_pct", "Normal flips (%)", "#E76F51"),
    ]:
        sub = curve_df[curve_df["metric"] == metric].sort_values("scale")
        x = sub["scale"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        lo = sub["ci95_low"].to_numpy(dtype=float)
        hi = sub["ci95_high"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", color=color, linewidth=1.8)
        if curve_type == "learned_scale_curve":
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
        else:
            ax.text(
                0.03,
                0.92,
                "not train-scale dependent",
                transform=ax.transAxes,
                fontsize=7.5,
                color="#555555",
                ha="left",
                va="top",
            )
        ax.set_title(label)
        ax.set_xlabel("Training identities")
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in x], rotation=25)
    png = out_dir / f"fig_{spec.key}_learning_curve.png"
    pdf = out_dir / f"fig_{spec.key}_learning_curve.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf, csv_path, curve_type


def plot_subunit_profile(subunit_row: pd.Series, spec: MethodSpec, out_dir: Path) -> tuple[Path, Path] | None:
    if subunit_row is None or subunit_row.empty:
        return None
    labels = [label for _, label in REGIONS]
    rmse = [float(subunit_row[f"{key}_rmse"]) for key, _ in REGIONS]
    flips = [float(subunit_row[f"{key}_new_flip_pct"]) for key, _ in REGIONS]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    fig.suptitle(f"{spec.label} regional profile", fontsize=12, fontweight="semibold")
    axes[0].bar(labels, rmse, color="#2A9D8F", edgecolor="white", linewidth=0.6)
    axes[0].set_title("Regional RMSE")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=22)
    axes[1].bar(labels, flips, color="#E76F51", edgecolor="white", linewidth=0.6)
    axes[1].set_title("Regional normal flips")
    axes[1].set_ylabel("% pairs")
    axes[1].tick_params(axis="x", rotation=22)

    png = out_dir / f"fig_{spec.key}_regional_profile.png"
    pdf = out_dir / f"fig_{spec.key}_regional_profile.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def finite_percentile_range(values: np.ndarray, low: float, high: float) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.nanpercentile(finite, low))
    hi = float(np.nanpercentile(finite, high))
    if hi <= lo:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def positive_vmax(values: np.ndarray, high: float) -> float:
    positive = np.asarray(values, dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if positive.size == 0:
        return 1e-6
    return max(float(np.nanpercentile(positive, high)), 1e-6)


def plot_spatial(
    project: Path,
    spec: MethodSpec,
    pred_path: Path,
    out_dir: Path,
    seed: int,
    split: str,
    counts: tuple[int, int, int],
    max_pairs: int,
    chunk_size: int,
) -> tuple[Path, Path, Path, Path, dict]:
    add_code_path(project)
    from data import load_rows

    rows, by_id = load_rows(project / "data")
    pairs_path = project / "outputs/unified_strict/safe_fusion/cache_test/pairs.json"
    pairs = load_pairs_json(pairs_path)
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    pred = np.load(pred_path, mmap_mode="r", allow_pickle=False)
    target_delta_path = project / "outputs/unified_strict/safe_fusion/cache_test/target_delta.npy"
    target_delta = np.load(target_delta_path, mmap_mode="r", allow_pickle=False)

    template = next(iter(by_id.values()))
    landmarks = np.asarray(template["landmarks"], dtype=np.int64)
    faces = np.asarray(template["faces"], dtype=np.int64)
    n_vertices = int(template["vertices"].shape[0])
    edges = edge_index(faces)
    edge0, edge1 = edges[:, 0], edges[:, 1]

    n_pairs = len(pairs)
    err_sq_sum = np.zeros(n_vertices, dtype=np.float64)
    source_sum = np.zeros((n_vertices, 3), dtype=np.float64)
    face_new_flip_count = np.zeros(len(faces), dtype=np.float64)
    edge_strain_samples = np.empty((n_pairs, len(edges)), dtype=np.float32)

    t0 = time.perf_counter()
    for start in range(0, n_pairs, chunk_size):
        stop = min(n_pairs, start + chunk_size)
        chunk_pairs = pairs[start:stop]
        src = np.stack([by_id[s]["vertices"] for s, _ in chunk_pairs]).astype(np.float64)
        true_delta = np.asarray(target_delta[start:stop], dtype=np.float64).reshape(stop - start, n_vertices, 3)
        tgt = src + true_delta
        pred_delta = np.asarray(pred[start:stop], dtype=np.float64).reshape(stop - start, n_vertices, 3)

        pred_delta = pred_delta.copy()
        pred_delta[:, landmarks] = true_delta[:, landmarks]
        edited = src + pred_delta

        diff = pred_delta - true_delta
        err_sq_sum += np.sum(diff * diff, axis=2).sum(axis=0)
        source_sum += src.sum(axis=0)

        src_len = np.linalg.norm(src[:, edge0] - src[:, edge1], axis=2)
        edited_len = np.linalg.norm(edited[:, edge0] - edited[:, edge1], axis=2)
        edge_strain_samples[start:stop] = (
            np.abs(edited_len - src_len) / np.maximum(src_len, 1e-12)
        ).astype(np.float32)

        for local in range(stop - start):
            pred_fold = fold_mask(src[local], edited[local], faces)
            target_fold = fold_mask(src[local], tgt[local], faces)
            face_new_flip_count += pred_fold & ~target_fold
        print(f"[standalone-spatial] {spec.key}: {stop}/{n_pairs}", flush=True)

    per_vertex_rmse = np.sqrt(err_sq_sum / float(n_pairs))
    per_vertex_rmse[landmarks] = np.nan
    face_new_flip_pct = face_new_flip_count / float(n_pairs) * 100.0
    edge_strain_p95 = np.percentile(edge_strain_samples, 95, axis=0)

    vertex_strain_sum = np.zeros(n_vertices, dtype=np.float64)
    vertex_strain_count = np.zeros(n_vertices, dtype=np.float64)
    np.add.at(vertex_strain_sum, edge0, edge_strain_p95)
    np.add.at(vertex_strain_sum, edge1, edge_strain_p95)
    np.add.at(vertex_strain_count, edge0, 1.0)
    np.add.at(vertex_strain_count, edge1, 1.0)
    per_vertex_strain_p95 = vertex_strain_sum / np.maximum(vertex_strain_count, 1.0)
    per_vertex_strain_p95[landmarks] = np.nan

    xyz = source_sum / float(n_pairs)
    landmark_xy = xyz[landmarks]
    face_xy = xyz[faces, :2]

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5), constrained_layout=True)
    fig.suptitle(spec.spatial_title or f"{spec.label} spatial diagnostics", fontsize=12, fontweight="semibold")

    rmse_norm = colors.Normalize(*finite_percentile_range(per_vertex_rmse, 2, 98), clip=True)
    sc0 = axes[0].scatter(xyz[:, 0], xyz[:, 1], c=per_vertex_rmse, cmap="magma", norm=rmse_norm, s=3, linewidths=0)
    axes[0].scatter(landmark_xy[:, 0], landmark_xy[:, 1], s=28, c="cyan", edgecolors="black", linewidths=0.4)
    axes[0].set_title("Free-vertex RMSE")
    fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.02, label="RMSE")

    flip_norm = colors.PowerNorm(gamma=0.45, vmin=0.0, vmax=positive_vmax(face_new_flip_pct, 99), clip=True)
    axes[1].add_collection(PolyCollection(face_xy, facecolors="#eeeeee", edgecolors="none", linewidths=0.0, zorder=0))
    positive_faces = face_new_flip_pct > 0
    cmap_flip = plt.get_cmap("inferno").copy()
    if np.any(positive_faces):
        axes[1].add_collection(
            PolyCollection(
                face_xy[positive_faces],
                array=face_new_flip_pct[positive_faces],
                cmap=cmap_flip,
                norm=flip_norm,
                edgecolors="none",
                linewidths=0.0,
                zorder=1,
            )
        )
    mappable = ScalarMappable(norm=flip_norm, cmap=cmap_flip)
    mappable.set_array(face_new_flip_pct[positive_faces] if np.any(positive_faces) else np.asarray([0.0]))
    axes[1].scatter(landmark_xy[:, 0], landmark_xy[:, 1], s=28, c="cyan", edgecolors="black", linewidths=0.4)
    axes[1].set_title("Normal flips")
    fig.colorbar(mappable, ax=axes[1], fraction=0.046, pad=0.02, label="% pairs")

    strain_norm = colors.Normalize(*finite_percentile_range(per_vertex_strain_p95, 2, 98), clip=True)
    sc2 = axes[2].scatter(
        xyz[:, 0],
        xyz[:, 1],
        c=per_vertex_strain_p95,
        cmap="viridis",
        norm=strain_norm,
        s=3,
        linewidths=0,
    )
    axes[2].scatter(landmark_xy[:, 0], landmark_xy[:, 1], s=28, c="cyan", edgecolors="black", linewidths=0.4)
    axes[2].set_title("Edge strain p95")
    fig.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.02, label="relative strain")

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    png = out_dir / f"fig_{spec.key}_spatial_diagnostics.png"
    pdf = out_dir / f"fig_{spec.key}_spatial_diagnostics.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    base_meta = meta_for(
        spec,
        seed,
        split,
        counts,
        n_pairs,
        {
            "pair_metrics": spec.pair_metrics,
            "dense_prediction": str(pred_path.relative_to(project)),
            "pairs_json": str(pairs_path.relative_to(project)),
            "target_delta": str(target_delta_path.relative_to(project)),
            "strict": "landmarks hard-fixed before metrics; RMSE excludes landmark vertices",
        },
    )
    vertex_rows = [
        {
            **base_meta,
            "section": spec.section,
            "local_vertex": int(i),
            "x": float(xyz[i, 0]),
            "y": float(xyz[i, 1]),
            "z": float(xyz[i, 2]),
            "is_landmark": int(i in set(landmarks.tolist())),
            "free_vertex_rmse": "" if np.isnan(per_vertex_rmse[i]) else float(per_vertex_rmse[i]),
            "incident_edge_strain_p95": "" if np.isnan(per_vertex_strain_p95[i]) else float(per_vertex_strain_p95[i]),
        }
        for i in range(n_vertices)
    ]
    face_rows = [
        {
            **base_meta,
            "section": spec.section,
            "face": int(i),
            "v0": int(face[0]),
            "v1": int(face[1]),
            "v2": int(face[2]),
            "normal_flip_frequency_pct": float(face_new_flip_pct[i]),
        }
        for i, face in enumerate(faces)
    ]
    vertex_csv = out_dir / f"{spec.key}_spatial_vertex_metrics.csv"
    face_csv = out_dir / f"{spec.key}_spatial_face_metrics.csv"
    write_csv(vertex_csv, vertex_rows)
    write_csv(face_csv, face_rows)
    info = {
        **base_meta,
        "section": spec.section,
        "status": "done",
        "png": str(png),
        "pdf": str(pdf),
        "vertex_csv": str(vertex_csv),
        "face_csv": str(face_csv),
        "n_vertices": int(n_vertices),
        "n_landmarks": int(len(landmarks)),
        "n_faces": int(len(faces)),
        "n_edges": int(len(edges)),
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    return png, pdf, vertex_csv, face_csv, info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("/content/drive/MyDrive/FYP final"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=["hybrid", "laplacian", "arap", "bilaplacian", "safe_fusion"])
    parser.add_argument("--spatial", action="store_true", help="Also render spatial heatmaps when dense caches exist.")
    parser.add_argument("--max-pairs", type=int, default=0, help="Debug only. 0 means all available pairs.")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--split", default="main test")
    args = parser.parse_args()

    setup_style()
    project = args.project.expanduser().resolve()
    out_root = (args.out or project / "results/standalone_method_diagnostics").expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    counts = split_counts(project)
    subunit_path = project / "results/subunit_metrics_all_methods/subunit_rmse_flip_ALL_METHODS_FINAL.csv"
    subunits = pd.read_csv(subunit_path) if subunit_path.exists() else pd.DataFrame()

    all_summary_rows: list[dict] = []
    manifest: dict = {"project": str(project), "out": str(out_root), "methods": {}, "notes": []}
    for key in args.methods:
        if key not in METHODS:
            raise KeyError(f"Unknown method {key}; available={sorted(METHODS)}")
        spec = METHODS[key]
        method_out = out_root / spec.key
        method_out.mkdir(parents=True, exist_ok=True)
        pair_path = project / spec.pair_metrics
        if not pair_path.exists():
            manifest["methods"][key] = {"status": "missing_pair_metrics", "pair_metrics": str(pair_path)}
            continue
        df = pd.read_csv(pair_path)
        if args.max_pairs > 0:
            df = df.head(args.max_pairs).copy()
        base_meta = meta_for(
            spec,
            args.seed,
            args.split,
            counts,
            len(df),
            {
                "pair_metrics": spec.pair_metrics,
                "subunit_metrics": str(subunit_path.relative_to(project)) if subunit_path.exists() else None,
                "dense_prediction": spec.dense_prediction,
            },
        )
        summary_rows = metric_summary(df, spec, base_meta)
        all_summary_rows.extend(summary_rows)
        write_csv(method_out / f"{spec.key}_metric_summary.csv", summary_rows)
        dashboard_png, dashboard_pdf = plot_dashboard(df, spec, method_out)

        subunit_outputs = None
        if not subunits.empty:
            match = subunits[subunits["method"] == spec.subunit_method]
            if not match.empty:
                row = match.iloc[0]
                subunit_outputs = plot_subunit_profile(row, spec, method_out)
                subunit_csv = method_out / f"{spec.key}_subunit_profile.csv"
                source_payload = {
                    f"source_{name}": value
                    for name, value in row.to_dict().items()
                    if name in {"method", "config", "protocol", "split", "source", "self_check"}
                }
                metric_payload = {
                    name: value
                    for name, value in row.to_dict().items()
                    if name not in {"method", "config", "protocol", "split", "source", "self_check"}
                }
                subunit_payload = [
                    {
                        **base_meta,
                        "section": spec.section,
                        **source_payload,
                        **metric_payload,
                    }
                ]
                write_csv(subunit_csv, subunit_payload)
            else:
                subunit_csv = None
        else:
            subunit_csv = None

        method_record = {
            **base_meta,
            "section": spec.section,
            "display_name": spec.label,
            "pair_metrics": str(pair_path),
            "dashboard_png": str(dashboard_png),
            "dashboard_pdf": str(dashboard_pdf),
            "tail_profile_png": None,
            "tail_profile_pdf": None,
            "learning_curve_png": None,
            "learning_curve_pdf": None,
            "learning_curve_csv": None,
            "learning_curve_type": None,
            "subunit_profile_png": str(subunit_outputs[0]) if subunit_outputs else None,
            "subunit_profile_pdf": str(subunit_outputs[1]) if subunit_outputs else None,
            "subunit_profile_csv": str(subunit_csv) if subunit_outputs else None,
            "spatial": None,
        }
        tail_png, tail_pdf = plot_tail_profile(df, spec, method_out)
        method_record["tail_profile_png"] = str(tail_png)
        method_record["tail_profile_pdf"] = str(tail_pdf)
        lc_png, lc_pdf, lc_csv, lc_type = plot_learning_curve_or_reference(project, df, spec, method_out, base_meta)
        method_record["learning_curve_png"] = str(lc_png)
        method_record["learning_curve_pdf"] = str(lc_pdf)
        method_record["learning_curve_csv"] = str(lc_csv)
        method_record["learning_curve_type"] = str(lc_type)

        if args.spatial and spec.dense_prediction:
            dense_path = project / spec.dense_prediction
            if dense_path.exists():
                png, pdf, vertex_csv, face_csv, spatial_info = plot_spatial(
                    project,
                    spec,
                    dense_path,
                    method_out,
                    args.seed,
                    args.split,
                    counts,
                    args.max_pairs,
                    args.chunk_size,
                )
                method_record["spatial"] = {
                    "png": str(png),
                    "pdf": str(pdf),
                    "vertex_csv": str(vertex_csv),
                    "face_csv": str(face_csv),
                    "summary": spatial_info,
                }
            else:
                method_record["spatial"] = {"status": "missing_dense_prediction", "path": str(dense_path)}
        elif args.spatial:
            method_record["spatial"] = {
                "status": "not_available",
                "reason": "No frozen dense prediction cache was found for this method.",
            }

        manifest["methods"][key] = method_record
        (method_out / f"{spec.key}_manifest.json").write_text(
            json.dumps(method_record, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    write_csv(out_root / "standalone_method_metric_summary.csv", all_summary_rows)
    (out_root / "standalone_method_diagnostics_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Standalone Method Diagnostics",
        "",
        "These figures are method-specific. Cross-method comparison figures should be placed in M7, not in the individual M3/M4/M6 method sections.",
        "",
        "| Method | Section | Dashboard | Regional profile | Tail profile | Learning curve | Spatial heatmap |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, record in manifest["methods"].items():
        if "status" in record and record["status"].startswith("missing"):
            lines.append(f"| {key} | - | missing pair metrics | - | - |")
            continue
        spatial = record.get("spatial")
        if isinstance(spatial, dict) and spatial.get("png"):
            spatial_cell = spatial["png"]
        elif isinstance(spatial, dict):
            spatial_cell = spatial.get("status", "")
        else:
            spatial_cell = ""
        lines.append(
            "| {method} | {section} | `{dash}` | `{region}` | `{tail}` | `{learning}` | `{spatial}` |".format(
                method=record["display_name"],
                section=record["section"],
                dash=Path(record["dashboard_png"]).relative_to(project).as_posix(),
                region=Path(record["subunit_profile_png"]).relative_to(project).as_posix()
                if record.get("subunit_profile_png")
                else "",
                tail=Path(record["tail_profile_png"]).relative_to(project).as_posix()
                if record.get("tail_profile_png")
                else "",
                learning=Path(record["learning_curve_png"]).relative_to(project).as_posix()
                if record.get("learning_curve_png")
                else "",
                spatial=Path(spatial_cell).relative_to(project).as_posix()
                if spatial_cell and str(spatial_cell).startswith(str(project))
                else spatial_cell,
            )
        )
    (out_root / "STANDALONE_METHOD_FIGURE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
