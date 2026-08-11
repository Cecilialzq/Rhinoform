from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.collections import PolyCollection
from matplotlib.cm import ScalarMappable


METHODS = ["rbsr", "cvae", "laplacian", "bilaplacian", "arap"]
METHOD_LABEL = {
    "rbsr": "RB-SR",
    "cvae": "CVAE-only",
    "laplacian": "Laplacian",
    "bilaplacian": "Bi-Laplacian",
    "arap": "ARAP",
}
METHOD_COLOR = {
    "rbsr": "#0F4D92",
    "cvae": "#606060",
    "laplacian": "#7884B4",
    "bilaplacian": "#B4C0E4",
    "arap": "#7C6CCF",
}
SUBUNITS = ["root", "dorsum", "tip", "alar_left", "alar_right"]
SUBUNIT_LABEL = {
    "root": "Root",
    "dorsum": "Dorsum",
    "tip": "Tip",
    "alar_left": "Alar L",
    "alar_right": "Alar R",
}


def apply_eval_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "legend.fontsize": 7.2,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
        }
    )


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.04, color: str = "black") -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=color,
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_output_dirs(root: Path | None = None) -> tuple[Path, Path]:
    out = root or project_root() / "results" / "eval_figures_redraw"
    svg = out / "svg"
    cache = out / "cache"
    out.mkdir(parents=True, exist_ok=True)
    svg.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    return out, svg


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> list[Path]:
    svg_dir = out_dir / "svg"
    paths = [out_dir / f"{name}.pdf", svg_dir / f"{name}.svg", out_dir / f"{name}.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1])
    fig.savefig(paths[2], dpi=300)
    plt.close(fig)
    return paths


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_manifest_rows(repo: Path) -> dict[str, dict]:
    manifest = json.loads((repo / "manifest.json").read_text(encoding="utf-8"))
    return {str(row["subject_id"]): row for row in manifest["rows"]}


def load_identity(repo: Path, subject_id: str) -> dict:
    rows = load_manifest_rows(repo)
    row = rows[str(subject_id)]
    npz_path = Path(row["npz_path"])
    if not npz_path.is_absolute():
        npz_path = repo / npz_path
    data = np.load(npz_path, allow_pickle=False)
    return {
        "subject_id": str(subject_id),
        "vertices": np.asarray(data["vertices"], dtype=np.float64),
        "faces": np.asarray(data["faces"], dtype=np.int64),
        "landmarks": np.asarray(data["local_landmarks"], dtype=np.int64),
        "subunits": {
            "root": np.asarray(data["subunit_root"], dtype=np.int64),
            "dorsum": np.asarray(data["subunit_dorsum"], dtype=np.int64),
            "tip": np.asarray(data["subunit_tip"], dtype=np.int64),
            "alar_left": np.asarray(data["subunit_alar_left"], dtype=np.int64),
            "alar_right": np.asarray(data["subunit_alar_right"], dtype=np.int64),
        },
    }


def load_identities(repo: Path, ids: Iterable[str]) -> dict[str, dict]:
    return {str(subject_id): load_identity(repo, str(subject_id)) for subject_id in sorted(set(ids), key=lambda x: int(x))}


def load_pairs(path: Path) -> list[tuple[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [(str(a), str(b)) for a, b in raw]


def pair_index(pairs: list[tuple[str, str]], source_id: str, target_id: str) -> int:
    target = (str(source_id), str(target_id))
    for index, pair in enumerate(pairs):
        if pair == target:
            return index
    raise ValueError(f"Pair {source_id}->{target_id} is not present in frozen pair order")


class NpyRowReader:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.f = self.path.open("rb")
        version = np.lib.format.read_magic(self.f)
        shape, fortran_order, dtype = np.lib.format._read_array_header(self.f, version)
        if fortran_order:
            raise ValueError(f"Fortran-order .npy is not supported: {path}")
        if len(shape) < 2:
            raise ValueError(f"Expected at least a 2D array: {path} shape={shape}")
        self.shape = tuple(int(x) for x in shape)
        self.dtype = np.dtype(dtype)
        self.offset = self.f.tell()
        self.row_shape = self.shape[1:]
        self.row_values = int(np.prod(self.row_shape))
        self.row_bytes = self.row_values * self.dtype.itemsize

    def row(self, index: int) -> np.ndarray:
        if index < 0 or index >= self.shape[0]:
            raise IndexError(index)
        self.f.seek(self.offset + int(index) * self.row_bytes)
        raw = self.f.read(self.row_bytes)
        if len(raw) != self.row_bytes:
            raise IOError(f"Short read from {self.path} at row {index}")
        return np.frombuffer(raw, dtype=self.dtype).copy().reshape(self.row_shape)

    def close(self) -> None:
        self.f.close()


@dataclass
class EvalPaths:
    root: Path
    data: Path
    cache_test: Path
    geometric_npz: Path
    rbsr_pair_metrics: Path
    main_means: Path
    rbsr_means: Path
    bootstrap: Path
    qualitative_cases: Path
    subunit_csv: Path
    demo_case: Path
    head_json: Path
    gate_map: Path


def default_paths(root: Path | None = None) -> EvalPaths:
    base = root or project_root()
    return EvalPaths(
        root=base,
        data=base / "data",
        cache_test=base / "outputs" / "unified_strict" / "safe_fusion" / "cache_test",
        geometric_npz=base / "results" / "geometric_tuning" / "geometric_validation_tuned_predictions.npz",
        rbsr_pair_metrics=base / "outputs" / "unified_strict" / "rbsr_gate" / "pair_metrics_rbsr_test.csv",
        main_means=base / "outputs" / "unified_strict" / "main_table" / "main_table_strict_means.csv",
        rbsr_means=base
        / "outputs"
        / "unified_strict"
        / "safe_fusion"
        / "strict_rescore"
        / "safe_fusion_strict_means.csv",
        bootstrap=base / "outputs" / "unified_strict" / "main_table" / "main_table_strict_bootstrap.csv",
        qualitative_cases=base / "outputs" / "unified_strict" / "qualitative_cases" / "qualitative_cases.csv",
        subunit_csv=base / "results" / "subunit_metrics_all_methods" / "subunit_rmse_flip_ALL_METHODS_FINAL.csv",
        demo_case=base / "demo" / "public" / "demo_case.json",
        head_json=base / "demo" / "public" / "head.json",
        gate_map=base / "outputs" / "unified_strict" / "rbsr_gate" / "gate_map_test.npz",
    )


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    denom = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(denom, 1e-12)


def edge_index(faces: np.ndarray) -> np.ndarray:
    edges = set()
    for a, b, c in np.asarray(faces, dtype=np.int64):
        for u, v in ((a, b), (b, c), (c, a)):
            u, v = int(u), int(v)
            if u > v:
                u, v = v, u
            edges.add((u, v))
    return np.asarray(sorted(edges), dtype=np.int64)


def new_flip_faces(source: np.ndarray, pred: np.ndarray, target: np.ndarray, faces: np.ndarray) -> np.ndarray:
    src_n = face_normals(source, faces)
    pred_n = face_normals(pred, faces)
    tgt_n = face_normals(target, faces)
    pred_flip = np.sum(src_n * pred_n, axis=1) < 0.0
    target_flip = np.sum(src_n * tgt_n, axis=1) < 0.0
    return pred_flip & ~target_flip


def vertex_from_face_indicator(faces: np.ndarray, face_mask: np.ndarray, n_vertices: int) -> np.ndarray:
    out = np.zeros(n_vertices, dtype=np.float64)
    if np.any(face_mask):
        out[np.unique(faces[face_mask].reshape(-1))] = 1.0
    return out


def vertex_edge_strain(source: np.ndarray, pred: np.ndarray, faces: np.ndarray) -> np.ndarray:
    edges = edge_index(faces)
    before = np.linalg.norm(source[edges[:, 0]] - source[edges[:, 1]], axis=1)
    after = np.linalg.norm(pred[edges[:, 0]] - pred[edges[:, 1]], axis=1)
    strain = np.abs(after - before) / np.maximum(before, 1e-12)
    out = np.zeros(len(source), dtype=np.float64)
    np.maximum.at(out, edges[:, 0], strain)
    np.maximum.at(out, edges[:, 1], strain)
    return out


def projection_xy(vertices: np.ndarray, view: str = "front") -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(vertices, dtype=np.float64)
    if view == "front":
        xy = v[:, [0, 1]]
        depth = v[:, 2]
    elif view == "lateral":
        xy = v[:, [2, 1]]
        depth = -v[:, 0]
    elif view == "oblique":
        theta = math.radians(38.0)
        x = math.cos(theta) * v[:, 0] - math.sin(theta) * v[:, 2]
        z = math.sin(theta) * v[:, 0] + math.cos(theta) * v[:, 2]
        xy = np.column_stack([x, v[:, 1]])
        depth = z
    else:
        raise ValueError(view)
    return xy, depth


def style_axis_equal(ax, xy: np.ndarray, pad: float = 0.045) -> None:
    mins = np.nanmin(xy, axis=0)
    maxs = np.nanmax(xy, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    margin = span * pad
    ax.set_xlim(mins[0] - margin[0], maxs[0] + margin[0])
    ax.set_ylim(mins[1] - margin[1], maxs[1] + margin[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_vertex_cloud(
    ax,
    vertices: np.ndarray,
    values: np.ndarray | None = None,
    *,
    view: str = "front",
    cmap: str = "viridis",
    norm=None,
    s: float = 4.0,
    alpha: float = 0.9,
    base_color: str = "#9AA7B2",
    limits_vertices: np.ndarray | None = None,
):
    xy, depth = projection_xy(vertices, view)
    order = np.argsort(depth)
    if values is None:
        sc = ax.scatter(xy[order, 0], xy[order, 1], s=s, c=base_color, alpha=alpha, linewidths=0, rasterized=True)
    else:
        sc = ax.scatter(
            xy[order, 0],
            xy[order, 1],
            s=s,
            c=np.asarray(values)[order],
            cmap=cmap,
            norm=norm,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )
    style_axis_equal(ax, projection_xy(limits_vertices if limits_vertices is not None else vertices, view)[0])
    return sc


def draw_handles(ax, vertices: np.ndarray, landmark_idx: np.ndarray, *, view: str = "front") -> None:
    xy, _ = projection_xy(vertices, view)
    ax.scatter(xy[:, 0], xy[:, 1], s=2.0, c="#4777B3", alpha=0.22, linewidths=0, rasterized=True)
    ax.scatter(
        xy[landmark_idx, 0],
        xy[landmark_idx, 1],
        s=12,
        c="#C42034",
        edgecolors="white",
        linewidths=0.35,
        zorder=5,
        label="9 sparse nasal handles",
    )
    style_axis_equal(ax, xy)


def draw_mesh_projection(
    ax,
    vertices: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray | None = None,
    *,
    view: str = "front",
    cmap: str = "viridis",
    norm=None,
    neutral: bool = False,
):
    xy, depth = projection_xy(vertices, view)
    face_depth = depth[faces].mean(axis=1)
    order = np.argsort(face_depth)
    polys = xy[faces][order]
    if values is None:
        if neutral:
            val = face_depth[order]
            val = (val - np.nanmin(val)) / max(np.nanmax(val) - np.nanmin(val), 1e-12)
            facecolors = plt.get_cmap("Greys")(0.22 + 0.45 * val)
        else:
            facecolors = np.tile(np.array([0.72, 0.77, 0.82, 1.0]), (len(order), 1))
    else:
        fv = np.asarray(values)[faces].mean(axis=1)[order]
        facecolors = plt.get_cmap(cmap)(norm(fv) if norm is not None else fv)
    coll = PolyCollection(polys, facecolors=facecolors, edgecolors="none", rasterized=True)
    ax.add_collection(coll)
    style_axis_equal(ax, xy)
    return coll


def method_summary(paths: EvalPaths) -> dict[str, dict[str, float]]:
    main = {row["method"]: row for row in read_csv_rows(paths.main_means)}
    rbsr_rows = {row["method"]: row for row in read_csv_rows(paths.rbsr_means)}
    mapping = {
        "rbsr": ("rbsr", rbsr_rows),
        "cvae": ("cvae_only", main),
        "laplacian": ("laplacian_handles", main),
        "bilaplacian": ("bilaplacian_handles", main),
        "arap": ("arap_handles_iter3", main),
    }
    out = {}
    for method, (key, rows) in mapping.items():
        row = rows[key]
        out[method] = {
            "roi_rmse": float(row["roi_rmse"]),
            "normal_flip_pct": float(row["normal_flip_pct"]),
            "edge_strain_p95": float(row["edge_strain_p95"]),
            "dorsum_rmse": float(row["dorsum_rmse"]),
            "tip_rmse": float(row["tip_rmse"]),
        }
    return out


def external_ci_from_bootstrap(paths: EvalPaths) -> dict[str, dict[str, tuple[float, float]]]:
    main = {row["method"]: row for row in read_csv_rows(paths.main_means)}
    ridge = main["ridge_sourcepca"]
    mapping = {
        "cvae_only": "cvae",
        "laplacian_handles": "laplacian",
        "bilaplacian_handles": "bilaplacian",
        "arap_handles_iter3": "arap",
    }
    ci: dict[str, dict[str, tuple[float, float]]] = {}
    for row in read_csv_rows(paths.bootstrap):
        if row["baseline"] != "ridge_sourcepca" or row["method"] not in mapping:
            continue
        metric = row["metric"]
        if metric not in {"roi_rmse", "normal_flip_pct", "edge_strain_p95"}:
            continue
        base = float(ridge[metric])
        ci.setdefault(mapping[row["method"]], {})[metric] = (
            base + float(row["ci95_low"]),
            base + float(row["ci95_high"]),
        )
    return ci


def load_subunit_rows(paths: EvalPaths) -> dict[str, dict]:
    raw = {row["method"]: row for row in read_csv_rows(paths.subunit_csv)}
    return {
        "rbsr": raw["rbsr"],
        "cvae": raw["cvae"],
        "laplacian": raw["laplacian"],
        "bilaplacian": raw["bilaplacian"],
        "arap": raw["arap"],
    }


def selected_qualitative_case(paths: EvalPaths) -> tuple[str, str, str]:
    rows = read_csv_rows(paths.qualitative_cases)
    for row in rows:
        if row.get("selection_reason") == "median_roi":
            return str(row["source_id"]), str(row["target_id"]), "median_roi"
    row = rows[0]
    return str(row["source_id"]), str(row["target_id"]), row.get("selection_reason", "first")


def select_worst_cases(paths: EvalPaths, n_cases: int = 20) -> list[dict]:
    rows = read_csv_rows(paths.rbsr_pair_metrics)
    rows = sorted(
        rows,
        key=lambda r: (
            -float(r["normal_flip_pct"]),
            -float(r["edge_strain_p95"]),
            -float(r["roi_rmse"]),
        ),
    )
    return rows[:n_cases]


class PredictionStore:
    def __init__(self, paths: EvalPaths, out_dir: Path):
        self.paths = paths
        self.out_dir = out_dir
        self.cache_dir = out_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pairs = load_pairs(paths.cache_test / "pairs.json")
        self.readers: dict[str, NpyRowReader] = {}
        self.external_rows: dict[str, dict[int, np.ndarray]] = {}

    def close(self) -> None:
        for reader in self.readers.values():
            reader.close()
        self.readers.clear()

    def _reader(self, filename: str) -> NpyRowReader:
        path = self.paths.cache_test / filename
        key = str(path)
        if key not in self.readers:
            self.readers[key] = NpyRowReader(path)
        return self.readers[key]

    def pair_index(self, source_id: str, target_id: str) -> int:
        return pair_index(self.pairs, source_id, target_id)

    def pred_delta(self, method: str, index: int, shape: tuple[int, int]) -> np.ndarray:
        if method == "rbsr":
            return self._reader("rbsr.npy").row(index).reshape(shape)
        if method == "arap":
            return self._reader("arap.npy").row(index).reshape(shape)
        if method == "cvae":
            ridge = self._reader("ridge.npy").row(index).reshape(shape)
            global_pred = self._reader("global_hybrid.npy").row(index).reshape(shape)
            alpha = 0.1
            return ridge + (global_pred - ridge) / alpha
        if method in {"laplacian", "bilaplacian"}:
            return self._external_pred_delta(method, index, shape)
        raise KeyError(method)

    def _external_pred_delta(self, method: str, index: int, shape: tuple[int, int]) -> np.ndarray:
        if method in self.external_rows and index in self.external_rows[method]:
            return self.external_rows[method][index].reshape(shape)
        key = {
            "laplacian": "laplacian_validation_tuned_pred",
            "bilaplacian": "bilaplacian_validation_tuned_pred",
        }[method]
        cache = self.cache_dir / f"{method}_selected_rows.npy"
        meta = self.cache_dir / f"{method}_selected_rows.json"
        if cache.exists() and meta.exists():
            payload = json.loads(meta.read_text(encoding="utf-8"))
            rows = np.load(cache, allow_pickle=False)
            self.external_rows[method] = {int(i): rows[pos] for pos, i in enumerate(payload["indices"])}
            if index in self.external_rows[method]:
                return self.external_rows[method][index].reshape(shape)
        raise KeyError(f"{method} row {index} has not been extracted from {self.paths.geometric_npz}")

    def prepare_external_rows(self, methods: Iterable[str], indices: Iterable[int]) -> dict:
        need_methods = [m for m in methods if m in {"laplacian", "bilaplacian"}]
        indices = sorted(set(int(i) for i in indices))
        status = {}
        if not need_methods or not indices:
            return status
        for method in need_methods:
            key = {
                "laplacian": "laplacian_validation_tuned_pred",
                "bilaplacian": "bilaplacian_validation_tuned_pred",
            }[method]
            cache = self.cache_dir / f"{method}_selected_rows.npy"
            meta = self.cache_dir / f"{method}_selected_rows.json"
            try:
                if cache.exists() and meta.exists():
                    payload = json.loads(meta.read_text(encoding="utf-8"))
                    existing = set(int(i) for i in payload.get("indices", []))
                    if set(indices).issubset(existing):
                        rows = np.load(cache, allow_pickle=False)
                        self.external_rows[method] = {int(i): rows[pos] for pos, i in enumerate(payload["indices"])}
                        status[method] = {"status": "cached", "cache": str(cache), "indices": indices}
                        continue
                data = np.load(self.paths.geometric_npz, allow_pickle=True)
                arr = np.asarray(data[key], dtype=np.float32)
                selected = arr[indices]
                np.save(cache, selected)
                write_json(
                    meta,
                    {
                        "source_npz": str(self.paths.geometric_npz),
                        "source_key": key,
                        "indices": indices,
                        "shape": list(selected.shape),
                        "note": "Rows copied exactly from frozen geometric prediction npz for figure rendering.",
                    },
                )
                self.external_rows[method] = {int(i): selected[pos] for pos, i in enumerate(indices)}
                status[method] = {"status": "extracted", "cache": str(cache), "indices": indices}
            except Exception as exc:  # noqa: BLE001
                status[method] = {"status": "missing", "error": repr(exc), "source_npz": str(self.paths.geometric_npz)}
        return status


def method_vertices(
    store: PredictionStore,
    by_id: dict[str, dict],
    source_id: str,
    target_id: str,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = store.pair_index(source_id, target_id)
    src = np.asarray(by_id[source_id]["vertices"], dtype=np.float64)
    tgt = np.asarray(by_id[target_id]["vertices"], dtype=np.float64)
    pred_delta = store.pred_delta(method, index, src.shape)
    landmarks = np.asarray(by_id[source_id]["landmarks"], dtype=np.int64)
    true_delta = tgt - src
    pred_delta = np.asarray(pred_delta, dtype=np.float64).copy()
    pred_delta[landmarks] = true_delta[landmarks]
    return src, tgt, src + pred_delta


def diagnostic_values(source: np.ndarray, target: np.ndarray, pred: np.ndarray, faces: np.ndarray) -> dict[str, np.ndarray]:
    pred_delta = pred - source
    true_delta = target - source
    error = np.linalg.norm(pred_delta - true_delta, axis=1)
    displacement = np.linalg.norm(pred_delta, axis=1)
    signed_z = pred_delta[:, 2] - true_delta[:, 2]
    new_flip = vertex_from_face_indicator(faces, new_flip_faces(source, pred, target, faces), len(source))
    strain = vertex_edge_strain(source, pred, faces)
    return {
        "error": error,
        "displacement": displacement,
        "signed_z": signed_z,
        "new_flip": new_flip,
        "edge_strain": strain,
    }


def subunit_vertex_labels(template: dict) -> np.ndarray:
    labels = np.full(len(template["vertices"]), -1, dtype=np.int64)
    for idx, name in enumerate(SUBUNITS):
        labels[np.asarray(template["subunits"][name], dtype=np.int64)] = idx
    return labels


def subunit_boundary_vertices(faces: np.ndarray, labels: np.ndarray) -> np.ndarray:
    edges = edge_index(faces)
    mask = labels[edges[:, 0]] != labels[edges[:, 1]]
    mask &= (labels[edges[:, 0]] >= 0) | (labels[edges[:, 1]] >= 0)
    return np.unique(edges[mask].reshape(-1))


def overlay_subunit_boundaries(ax, vertices: np.ndarray, faces: np.ndarray, labels: np.ndarray, *, view: str = "front") -> None:
    b = subunit_boundary_vertices(faces, labels)
    if b.size == 0:
        return
    xy, _ = projection_xy(vertices, view)
    ax.scatter(xy[b, 0], xy[b, 1], s=1.6, c="black", alpha=0.35, linewidths=0, rasterized=True)


def full_face_from_demo(paths: EvalPaths) -> dict:
    head = json.loads(paths.head_json.read_text(encoding="utf-8"))
    demo = json.loads(paths.demo_case.read_text(encoding="utf-8"))
    full_source = np.asarray(head["vertices"], dtype=np.float64).reshape(-1, 3)
    full_faces = np.asarray(head["faces"], dtype=np.int64).reshape(-1, 3)
    roi_indices = np.asarray(head["roi_indices"], dtype=np.int64)
    roi_source = np.asarray(demo["geometry"]["source"], dtype=np.float64)
    roi_output = np.asarray(demo["methods"]["rbsr"]["vertices"], dtype=np.float64)
    output = full_source.copy()
    output[roi_indices] = roi_output
    collar = head.get("collar", {})
    if collar:
        v = np.asarray(collar["v"], dtype=np.int64)
        anchor = np.asarray(collar["anchor"], dtype=np.int64)
        w = np.asarray(collar["w"], dtype=np.float64)[:, None]
        disp = roi_output[anchor] - roi_source[anchor]
        output[v] = full_source[v] + w * disp
    return {
        "head": head,
        "demo": demo,
        "source": full_source,
        "output": output,
        "faces": full_faces,
        "roi_indices": roi_indices,
        "roi_source": roi_source,
        "roi_output": roi_output,
        "collar_used": bool(collar),
    }


def figure_caption_notes() -> dict[str, str]:
    return {
        "fig_5_1_source_target_landmark_displacement": (
            "Sparse controls are shown on a real source-to-target identity pair. "
            "Red points are the nine sparse nasal handles and blue points are original mesh vertices. "
            "Displacement is target landmark position minus source landmark position; arrow lengths are scaled only for visibility."
        ),
        "fig_frontier_hero": (
            "Accuracy-regularity operating points for RB-SR and the four external baselines. "
            "Lower ROI RMSE and lower normal-flip percentage are better."
        ),
        "fig_spatial_error_evidence": (
            "Vertex-level diagnostic projections for one representative frozen test pair; colors are computed from frozen predictions."
        ),
        "fig_risk_spatial_distribution": (
            "Risk diagnostics show that fold-over and strain are spatially concentrated rather than uniformly distributed."
        ),
        "fig_subunit_triptych": (
            "Sub-unit RMSE, sub-unit normal-flip risk, and RB-SR minus the strongest displayed external accuracy baseline."
        ),
        "fig_demo_full_face_before_after": (
            "Full-face demo preview uses the existing full-head JSON and collar blend; this stored demo_case includes a frozen target for the reported metrics."
        ),
        "fig_method_qualitative_hero": (
            "Qualitative translation of the benchmark on one representative frozen pair; quantitative conclusions remain the frozen aggregate benchmark."
        ),
        "fig_visual_audit_grid": (
            "Qualitative visual audit of RB-SR worst-case samples selected by normal-flip percentage, then edge strain and ROI RMSE."
        ),
    }
