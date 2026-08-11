from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .data import edge_index, face_normals, load_rows, rmse_vertices
from .repro import atomic_write_csv


LOWER_IS_BETTER = {
    "roi_rmse": True,
    "landmark_rmse": True,
    "dorsum_rmse": True,
    "tip_rmse": True,
    "edge_strain_p95": True,
    "normal_flip_pct": True,
}


def metric_rows_for_method(
    by_id: dict[str, dict],
    pairs: list[tuple[str, str]],
    pred_flat: np.ndarray,
) -> list[dict[str, float | str]]:
    template = next(iter(by_id.values()))
    landmarks = template["landmarks"]
    subunits = template["subunits"]
    faces = template["faces"]
    edges = edge_index(faces)
    rows: list[dict[str, float | str]] = []
    for i, (src_id, tgt_id) in enumerate(pairs):
        src = by_id[src_id]["vertices"]
        tgt = by_id[tgt_id]["vertices"]
        pred_delta = np.asarray(pred_flat[i], dtype=np.float64).reshape(-1, 3)
        true_delta = tgt - src
        pred_vertices = src + pred_delta

        e0 = np.linalg.norm(src[edges[:, 0]] - src[edges[:, 1]], axis=1)
        e1 = np.linalg.norm(pred_vertices[edges[:, 0]] - pred_vertices[edges[:, 1]], axis=1)
        strain = np.abs(e1 - e0) / np.maximum(e0, 1e-12)
        n0 = face_normals(src, faces)
        n1 = face_normals(pred_vertices, faces)

        rows.append(
            {
                "pair_index": float(i),
                "source_id": str(src_id),
                "target_id": str(tgt_id),
                "roi_rmse": rmse_vertices(pred_delta, true_delta),
                "landmark_rmse": rmse_vertices(pred_delta, true_delta, landmarks),
                "dorsum_rmse": rmse_vertices(pred_delta, true_delta, subunits["dorsum"]),
                "tip_rmse": rmse_vertices(pred_delta, true_delta, subunits["tip"]),
                "edge_strain_p95": float(np.percentile(strain, 95)),
                "normal_flip_pct": float(np.mean(np.sum(n0 * n1, axis=1) < 0.0) * 100.0),
            }
        )
    return rows


def identity_level_means(rows: list[dict[str, float | str]], metric: str, cluster_key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row[cluster_key]), []).append(float(row[metric]))
    return {sid: float(np.mean(vals)) for sid, vals in grouped.items()}


def paired_identity_bootstrap(
    baseline_rows: list[dict[str, float | str]],
    method_rows: list[dict[str, float | str]],
    method_name: str,
    baseline_name: str,
    metric: str,
    cluster_key: str,
    n_boot: int,
    seed: int,
) -> dict[str, float | str]:
    base = identity_level_means(baseline_rows, metric, cluster_key)
    meth = identity_level_means(method_rows, metric, cluster_key)
    ids = sorted(set(base) & set(meth), key=lambda x: int(x) if x.isdigit() else x)
    diffs = np.asarray([meth[sid] - base[sid] for sid in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    n = len(diffs)
    for i in range(n_boot):
        boot[i] = float(np.mean(diffs[rng.integers(0, n, size=n)]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    observed = float(np.mean(diffs))
    lower_better = LOWER_IS_BETTER[metric]
    if lower_better:
        prob_improved = float(np.mean(boot < 0.0))
        interpretation = "improved" if hi < 0.0 else "not_significant" if lo <= 0.0 <= hi else "worse"
    else:
        prob_improved = float(np.mean(boot > 0.0))
        interpretation = "improved" if lo > 0.0 else "not_significant" if lo <= 0.0 <= hi else "worse"
    return {
        "method": method_name,
        "baseline": baseline_name,
        "cluster_key": cluster_key,
        "metric": metric,
        "n_cluster_identities": float(n),
        "baseline_cluster_mean": float(np.mean([base[sid] for sid in ids])),
        "method_cluster_mean": float(np.mean([meth[sid] for sid in ids])),
        "mean_diff_method_minus_baseline": observed,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "probability_improved_bootstrap": prob_improved,
        "interpretation": interpretation,
    }


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    atomic_write_csv(path, rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="../data")
    parser.add_argument("--predictions", default="../results/neural_field_predictions_cvae_ew0p1_lw0.npz")
    parser.add_argument("--out", default="../results")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2031)
    args = parser.parse_args()

    repo = Path(args.repo)
    out_dir = Path(args.out)
    _, by_id = load_rows(repo)
    pred = np.load(args.predictions, allow_pickle=True)
    pairs = [(str(a), str(b)) for a, b in pred["test_pairs"].tolist()]

    methods: dict[str, np.ndarray] = {}
    if "ridge_cond_pred" in pred:
        methods["ridge_sourcepca"] = np.asarray(pred["ridge_cond_pred"])
    if "cvae_pred" in pred:
        methods["cvae_only"] = np.asarray(pred["cvae_pred"])
    if "hybrid_selected_pred" in pred:
        alpha = float(np.asarray(pred["selected_alpha"]).reshape(-1)[0]) if "selected_alpha" in pred else float("nan")
        methods[f"hybrid_alpha_{alpha:g}"] = np.asarray(pred["hybrid_selected_pred"])
    for key in pred.files:
        if not key.endswith("_pred") or key in {"ridge_cond_pred", "cvae_pred", "hybrid_selected_pred"}:
            continue
        methods[key.removesuffix("_pred")] = np.asarray(pred[key])
    if "ridge_sourcepca" not in methods:
        raise KeyError("Predictions file must include ridge_cond_pred for the default ridge_sourcepca baseline.")

    per_method_rows = {
        name: metric_rows_for_method(by_id, pairs, arr)
        for name, arr in methods.items()
    }
    for name, rows in per_method_rows.items():
        write_csv(out_dir / f"identity_bootstrap_pair_metrics_{name}.csv", rows)

    metrics = list(LOWER_IS_BETTER)
    baseline_name = "ridge_sourcepca"
    summary_rows: list[dict[str, float | str]] = []
    for method_name, rows in per_method_rows.items():
        if method_name == baseline_name:
            continue
        for metric in metrics:
            summary_rows.append(
                paired_identity_bootstrap(
                    per_method_rows[baseline_name],
                    rows,
                    method_name,
                    baseline_name,
                    metric,
                    "source_id",
                    args.n_boot,
                    args.seed + len(summary_rows),
                )
            )

    write_csv(out_dir / "identity_bootstrap_summary.csv", summary_rows)
    report = {
        "predictions": str(Path(args.predictions)),
        "bootstrap_unit": "source identity",
        "n_boot": args.n_boot,
        "note": "Differences are method minus baseline. For all reported metrics, negative means the method is better/lower than ridge_sourcepca.",
        "summary": summary_rows,
    }
    (out_dir / "identity_bootstrap_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    sensitivity_rows: list[dict[str, float | str]] = []
    for cluster_key in ["source_id", "target_id"]:
        for method_name, rows in per_method_rows.items():
            if method_name == baseline_name:
                continue
            for metric in metrics:
                sensitivity_rows.append(
                    paired_identity_bootstrap(
                        per_method_rows[baseline_name],
                        rows,
                        method_name,
                        baseline_name,
                        metric,
                        cluster_key,
                        args.n_boot,
                        args.seed + 1000 + len(sensitivity_rows),
                    )
                )
    write_csv(out_dir / "identity_bootstrap_sensitivity_summary.csv", sensitivity_rows)
    sensitivity = {
        "predictions": str(Path(args.predictions)),
        "bootstrap_units": ["source_id", "target_id"],
        "n_boot": args.n_boot,
        "note": "Sensitivity analysis repeats paired bootstrap after aggregating by source identity and by target identity separately.",
        "summary": sensitivity_rows,
    }
    (out_dir / "identity_bootstrap_sensitivity_summary.json").write_text(json.dumps(sensitivity, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
