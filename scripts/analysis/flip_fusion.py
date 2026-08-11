"""Low-cost, no-training diagnosis of nasal mesh flip metrics and safe fusion.

The experiment calibrates the legacy source-normal reversal metric, localizes
failure causes, measures the effect of exact-handle projection, and compares
uniform versus local attenuation when adding a learned proposal to ARAP.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import torch

from rhinoform.baselines import arap_predict_vectorised
from rhinoform.data import array_sha256, edge_index, face_normals, ridge_predict, rmse_vertices
from rhinoform.geometry import graph_edges, solve_linear_handle_baseline, uniform_laplacian
from rhinoform.repro import atomic_write_json
from rhinoform.train import NeuralFieldCVAE, face_vertex_normals, predict_field
from rhinoform.train_rbsr_gate import SpatialRiskGate, build_rbf_projection, predict_gate_batches


SUBUNITS = ("root", "dorsum", "tip", "alar_left", "alar_right")


def load_manifest(repo: Path) -> tuple[list[dict], dict[str, Path]]:
    rows = json.loads((repo / "manifest.json").read_text(encoding="utf-8"))["rows"]
    paths = {}
    for row in rows:
        path = Path(row["npz_path"])
        paths[str(row["subject_id"])] = path if path.is_absolute() else repo / path
    return rows, paths


def load_template(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    return {
        "vertices": np.asarray(data["vertices"], dtype=np.float64),
        "faces": np.asarray(data["faces"], dtype=np.int64),
        "landmarks": np.asarray(data["local_landmarks"], dtype=np.int64),
        "subunits": {name: np.asarray(data[f"subunit_{name}"], dtype=np.int64) for name in SUBUNITS},
    }


def load_vertices(paths: dict[str, Path], ids: list[str]) -> dict[str, np.ndarray]:
    return {
        sid: np.asarray(np.load(paths[sid], allow_pickle=False)["vertices"], dtype=np.float64)
        for sid in ids
    }


def select_pairs(metric_csv: Path, n_pairs: int, seed: int) -> list[tuple[str, str]]:
    with metric_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: float(row["normal_flip_pct"]))
    rng = np.random.default_rng(seed)
    third = max(1, n_pairs // 3)
    bins = [rows[: len(rows) // 3], rows[len(rows) // 3 : 2 * len(rows) // 3], rows[2 * len(rows) // 3 :]]
    selected = []
    for group in bins:
        take = min(third, len(group))
        idx = rng.choice(len(group), size=take, replace=False)
        selected.extend((str(group[int(i)]["source_id"]), str(group[int(i)]["target_id"])) for i in idx)
    if len(selected) < n_pairs:
        used = set(selected)
        remaining = [(str(r["source_id"]), str(r["target_id"])) for r in rows if (str(r["source_id"]), str(r["target_id"])) not in used]
        selected.extend(remaining[: n_pairs - len(selected)])
    return selected[:n_pairs]


def selected_npz_rows(npz_path: Path, member: str, indices: np.ndarray) -> np.ndarray:
    member_name = f"{member}.npy"
    with zipfile.ZipFile(npz_path) as archive:
        info = archive.getinfo(member_name)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{npz_path}:{member_name} is compressed and cannot be directly memory-mapped")
        with npz_path.open("rb") as handle:
            handle.seek(info.header_offset)
            local_header = handle.read(30)
            fields = struct.unpack("<IHHHHHIIIHH", local_header)
            if fields[0] != 0x04034B50:
                raise ValueError(f"Invalid ZIP local header for {member_name}")
            npy_offset = info.header_offset + 30 + fields[9] + fields[10]
            handle.seek(npy_offset)
            version = np.lib.format.read_magic(handle)
            shape, fortran_order, dtype = np.lib.format._read_array_header(handle, version)
            data_offset = handle.tell()
    array = np.memmap(
        npz_path,
        dtype=dtype,
        mode="r",
        offset=data_offset,
        shape=shape,
        order="F" if fortran_order else "C",
    )
    result = np.asarray(array[indices], dtype=np.float32).copy()
    del array
    return result


def static_vertex_features(
    rows: list[dict], paths: dict[str, Path], train_ids: list[str], template: dict, use_subunits: bool
) -> np.ndarray:
    total = np.zeros(3, dtype=np.float64)
    count = 0
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for sid in train_ids:
        vertices = np.asarray(np.load(paths[sid], allow_pickle=False)["vertices"], dtype=np.float64)
        total += vertices.sum(axis=0)
        count += len(vertices)
        lo = np.minimum(lo, vertices.min(axis=0))
        hi = np.maximum(hi, vertices.max(axis=0))
    center = total[None, :] / count
    scale = max(float(np.linalg.norm(hi - lo)), 1e-6)
    base = template["vertices"]
    parts = [(base - center) / scale, face_vertex_normals(base, template["faces"])]
    if use_subunits:
        one_hot = np.zeros((len(base), len(SUBUNITS)), dtype=np.float64)
        for i, name in enumerate(SUBUNITS):
            one_hot[template["subunits"][name], i] = 1.0
        parts.append(one_hot)
    distances = np.linalg.norm(base[:, None, :] - base[template["landmarks"]][None, :, :], axis=2)
    distances /= np.maximum(np.percentile(distances, 95, axis=0, keepdims=True), 1e-6)
    parts.append(distances)
    return np.concatenate(parts, axis=1).astype(np.float32)


def pair_inputs(
    pairs: list[tuple[str, str]], vertices: dict[str, np.ndarray], landmarks: np.ndarray, base: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = np.stack([vertices[s] for s, _ in pairs]).astype(np.float32)
    target = np.stack([vertices[t] for _, t in pairs]).astype(np.float32)
    delta = target - source
    controls = delta[:, landmarks]
    source_flat = source.reshape(len(pairs), -1)
    source_code = (source_flat - base["source_pca"]["mean"]) @ base["source_pca"]["components"].T
    condition = np.concatenate([controls.reshape(len(pairs), -1), source_code], axis=1)
    condition = (condition - base["cond_mean"]) / base["cond_std"]
    return condition.astype(np.float32), source, delta, controls.astype(np.float32)


def rbsr_predictions(
    base: dict,
    rbsr: dict,
    features: np.ndarray,
    condition: np.ndarray,
    source: np.ndarray,
    ridge: np.ndarray,
    cvae: np.ndarray,
    controls: np.ndarray,
    landmarks: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cpu")
    args = rbsr["args"]
    gate = SpatialRiskGate(
        int(rbsr["gate_vertex_dim"]), int(rbsr["gate_cond_dim"]), int(args["hidden"]), float(base["selected_alpha"])
    ).to(device)
    gate.load_state_dict(rbsr["gate_state_dict"])
    gate.eval()
    projection = None
    if rbsr.get("projection_basis") is not None:
        projection = torch.as_tensor(rbsr["projection_basis"], dtype=torch.float32, device=device)
    return predict_gate_batches(
        gate,
        torch.as_tensor(features, dtype=torch.float32, device=device),
        condition,
        source,
        ridge,
        cvae,
        torch.as_tensor(rbsr["center"], dtype=torch.float32, device=device),
        float(rbsr["scale"]),
        torch.as_tensor(landmarks, dtype=torch.long, device=device),
        controls,
        projection,
        batch_size,
    )


def cvae_predictions(base: dict, features: np.ndarray, condition: np.ndarray) -> np.ndarray:
    args = base["args"]
    model = NeuralFieldCVAE(
        features.shape[1],
        condition.shape[1],
        int(np.asarray(base["obs_train_mean"]).shape[1]),
        latent_dim=int(args["latent_dim"]),
        hidden=int(args["hidden"]),
    )
    model.load_state_dict(base["cvae_state_dict"])
    model.eval()
    prediction = predict_field(model, features, condition, is_cvae=True).astype(np.float32)
    del model
    return prediction


def triangle_fields(source: np.ndarray, edited: np.ndarray, faces: np.ndarray) -> dict[str, np.ndarray]:
    src_tri = source[faces]
    out_tri = edited[faces]
    src_cross = np.cross(src_tri[:, 1] - src_tri[:, 0], src_tri[:, 2] - src_tri[:, 0])
    out_cross = np.cross(out_tri[:, 1] - out_tri[:, 0], out_tri[:, 2] - out_tri[:, 0])
    src_norm = np.linalg.norm(src_cross, axis=1)
    out_norm = np.linalg.norm(out_cross, axis=1)
    dot = np.sum(src_cross * out_cross, axis=1)
    area_ratio = out_norm / np.maximum(src_norm, 1e-12)
    cosine = dot / np.maximum(src_norm * out_norm, 1e-12)
    return {"legacy_flip": cosine < 0.0, "cosine": cosine, "area_ratio": area_ratio}


def triangle_quality(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    lengths2 = (
        np.sum((tri[:, 1] - tri[:, 0]) ** 2, axis=1)
        + np.sum((tri[:, 2] - tri[:, 1]) ** 2, axis=1)
        + np.sum((tri[:, 0] - tri[:, 2]) ** 2, axis=1)
    )
    twice_area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return 2.0 * np.sqrt(3.0) * twice_area / np.maximum(lengths2, 1e-12)


def face_gradient(field: np.ndarray, faces: np.ndarray, source: np.ndarray) -> np.ndarray:
    tri = field[faces]
    src = source[faces]
    numerator = np.maximum.reduce(
        [np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1), np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1), np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)]
    )
    denominator = np.maximum.reduce(
        [np.linalg.norm(src[:, 1] - src[:, 0], axis=1), np.linalg.norm(src[:, 2] - src[:, 1], axis=1), np.linalg.norm(src[:, 0] - src[:, 2], axis=1)]
    )
    return numerator / np.maximum(denominator, 1e-12)


def path_min_area_ratio(source: np.ndarray, delta: np.ndarray, faces: np.ndarray, steps: int = 21) -> float:
    minimum = np.inf
    for alpha in np.linspace(0.0, 1.0, steps):
        minimum = min(minimum, float(triangle_fields(source, source + alpha * delta, faces)["area_ratio"].min()))
    return minimum


def method_metrics(source: np.ndarray, target_delta: np.ndarray, pred: np.ndarray, faces: np.ndarray, edges: np.ndarray) -> dict:
    edited = source + pred
    fields = triangle_fields(source, edited, faces)
    e0 = np.linalg.norm(source[edges[:, 0]] - source[edges[:, 1]], axis=1)
    e1 = np.linalg.norm(edited[edges[:, 0]] - edited[edges[:, 1]], axis=1)
    strain = np.abs(e1 - e0) / np.maximum(e0, 1e-12)
    return {
        "roi_rmse": rmse_vertices(pred, target_delta),
        "legacy_flip_pct": float(100.0 * fields["legacy_flip"].mean()),
        "legacy_flip_count": int(fields["legacy_flip"].sum()),
        "area_ratio_min": float(fields["area_ratio"].min()),
        "area_ratio_p01": float(np.percentile(fields["area_ratio"], 1)),
        "severe_area_count": int(np.sum(fields["area_ratio"] < 0.05)),
        "edge_strain_p95": float(np.percentile(strain, 95)),
        "path_min_area_ratio": path_min_area_ratio(source, pred, faces),
    }


def aggregate(rows: list[dict], method: str) -> dict:
    selected = [row for row in rows if row["method"] == method]
    keys = [
        key
        for key in selected[0]
        if key not in {"method", "source_id", "target_id"}
        and all(row[key] != "" for row in selected)
    ]
    return {key: float(np.mean([float(row[key]) for row in selected])) for key in keys}


def legacy_target_calibration(ids: list[str], vertices: dict[str, np.ndarray], faces: np.ndarray) -> dict:
    normals = {sid: face_normals(vertices[sid], faces) for sid in ids}
    values = []
    any_count = 0
    for source_id in ids:
        for target_id in ids:
            if source_id == target_id:
                continue
            value = float(100.0 * np.mean(np.sum(normals[source_id] * normals[target_id], axis=1) < 0.0))
            values.append(value)
            any_count += int(value > 0.0)
    arr = np.asarray(values)
    return {
        "n_pairs": int(len(arr)),
        "mean_pct": float(arr.mean()),
        "median_pct": float(np.median(arr)),
        "p95_pct": float(np.percentile(arr, 95)),
        "max_pct": float(arr.max()),
        "pairs_with_any_pct": float(100.0 * any_count / len(arr)),
    }


def subunit_face_masks(faces: np.ndarray, subunits: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    masks = {}
    for name, indices in subunits.items():
        vertex_mask = np.zeros(int(faces.max()) + 1, dtype=bool)
        vertex_mask[indices] = True
        masks[name] = np.any(vertex_mask[faces], axis=1)
    return masks


def cause_analysis(
    pairs: list[tuple[str, str]], vertices: dict[str, np.ndarray], predictions: np.ndarray, ridge: np.ndarray, faces: np.ndarray, subunits: dict[str, np.ndarray]
) -> dict:
    quality_all = []
    gradient_all = []
    area_all = []
    flips_all = []
    masks = subunit_face_masks(faces, subunits)
    subunit_flip = {name: 0 for name in SUBUNITS}
    total_flips = 0
    for i, (source_id, _) in enumerate(pairs):
        source = vertices[source_id]
        edited = source + predictions[i]
        fields = triangle_fields(source, edited, faces)
        flips = fields["legacy_flip"]
        quality_all.append(triangle_quality(source, faces))
        gradient_all.append(face_gradient(predictions[i] - ridge[i], faces, source))
        area_all.append(fields["area_ratio"])
        flips_all.append(flips)
        total_flips += int(flips.sum())
        for name in SUBUNITS:
            subunit_flip[name] += int(np.sum(flips & masks[name]))
    quality = np.concatenate(quality_all)
    gradient = np.concatenate(gradient_all)
    area = np.concatenate(area_all)
    flips = np.concatenate(flips_all)
    def point_biserial(values: np.ndarray) -> float:
        return float(np.corrcoef(values, flips.astype(np.float64))[0, 1]) if flips.any() and (~flips).any() else float("nan")
    q10 = float(np.percentile(quality, 10))
    g90 = float(np.percentile(gradient, 90))
    return {
        "n_face_observations": int(len(flips)),
        "flip_observations": int(flips.sum()),
        "correlation_flip_source_quality": point_biserial(quality),
        "correlation_flip_neural_residual_gradient": point_biserial(gradient),
        "correlation_flip_area_ratio": point_biserial(area),
        "flip_rate_lowest_quality_decile": float(np.mean(flips[quality <= q10])),
        "flip_rate_other_quality": float(np.mean(flips[quality > q10])),
        "flip_rate_highest_gradient_decile": float(np.mean(flips[gradient >= g90])),
        "flip_rate_other_gradient": float(np.mean(flips[gradient < g90])),
        "flips_touching_subunit_fraction": {name: float(value / max(total_flips, 1)) for name, value in subunit_flip.items()},
    }


def project_exact_handles(prediction: np.ndarray, controls: np.ndarray, basis: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    error = controls - prediction[:, landmarks]
    correction = np.einsum("vk,bkd->bvd", basis, error)
    projected = prediction + correction
    projected[:, landmarks] = controls
    return projected, correction


def safety_signature(source: np.ndarray, pred: np.ndarray, base_pred: np.ndarray, faces: np.ndarray) -> tuple[int, int]:
    fields = triangle_fields(source, source + pred, faces)
    base = triangle_fields(source, source + base_pred, faces)
    new_legacy = int(np.sum(fields["legacy_flip"] & ~base["legacy_flip"]))
    severe_excess = max(0, int(np.sum(fields["area_ratio"] < 0.05)) - int(np.sum(base["area_ratio"] < 0.05)))
    return new_legacy, severe_excess


def handle_preserving_residual(base_pred: np.ndarray, proposal: np.ndarray, basis: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    residual = proposal - base_pred
    correction = np.einsum("vk,kd->vd", basis, -residual[landmarks])
    residual = residual + correction
    residual[landmarks] = 0.0
    return residual


def uniform_safe_fusion(source: np.ndarray, base_pred: np.ndarray, residual: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, float]:
    for alpha in np.linspace(1.0, 0.0, 51):
        candidate = base_pred + float(alpha) * residual
        if safety_signature(source, candidate, base_pred, faces) == (0, 0):
            return candidate, float(alpha)
    return base_pred.copy(), 0.0


def local_safe_fusion(
    source: np.ndarray,
    base_pred: np.ndarray,
    residual: np.ndarray,
    faces: np.ndarray,
    landmarks: np.ndarray,
    smooth: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    weights = np.ones(len(source), dtype=np.float64)
    weights[landmarks] = 0.0
    base_fields = triangle_fields(source, source + base_pred, faces)
    edges = graph_edges(faces)
    edge0, edge1 = edges[:, 0], edges[:, 1]
    for iteration in range(40):
        candidate = base_pred + weights[:, None] * residual
        fields = triangle_fields(source, source + candidate, faces)
        unsafe = (fields["legacy_flip"] & ~base_fields["legacy_flip"]) | (
            (fields["area_ratio"] < 0.05) & ~(base_fields["area_ratio"] < 0.05)
        )
        if not np.any(unsafe):
            return candidate, weights, iteration
        bad_vertices = np.unique(faces[unsafe].reshape(-1))
        weights[bad_vertices] *= 0.75
        if smooth:
            for _ in range(2):
                total = weights.copy()
                count = np.ones(len(weights), dtype=np.float64)
                np.add.at(total, edge0, weights[edge1])
                np.add.at(total, edge1, weights[edge0])
                np.add.at(count, edge0, 1.0)
                np.add.at(count, edge1, 1.0)
                weights = np.minimum(weights, total / count)
        weights[landmarks] = 0.0
    return base_pred + weights[:, None] * residual, weights, 40


def fusion_analysis(
    pairs: list[tuple[str, str]], vertices: dict[str, np.ndarray], target_delta: np.ndarray, arap: np.ndarray, proposal: np.ndarray,
    faces: np.ndarray, landmarks: np.ndarray, basis: np.ndarray
) -> tuple[list[dict], dict]:
    rows = []
    for i, (source_id, target_id) in enumerate(pairs):
        source = vertices[source_id]
        residual = handle_preserving_residual(arap[i], proposal[i], basis, landmarks)
        uniform, alpha = uniform_safe_fusion(source, arap[i], residual, faces)
        local, weights, iterations = local_safe_fusion(source, arap[i], residual, faces, landmarks, smooth=False)
        local_smooth, smooth_weights, smooth_iterations = local_safe_fusion(
            source, arap[i], residual, faces, landmarks, smooth=True
        )
        denom = max(float(np.linalg.norm(residual)), 1e-12)
        for method, pred, retention in (
            ("arap_base", arap[i], 0.0),
            ("uniform", uniform, float(np.linalg.norm(uniform - arap[i]) / denom)),
            ("local", local, float(np.linalg.norm(local - arap[i]) / denom)),
            ("local_smooth", local_smooth, float(np.linalg.norm(local_smooth - arap[i]) / denom)),
            ("full_proposal", arap[i] + residual, 1.0),
        ):
            metrics = method_metrics(source, target_delta[i], pred, faces, edge_index(faces))
            rows.append({
                "source_id": source_id, "target_id": target_id, "method": method,
                "residual_retention": retention, "uniform_alpha": alpha if method == "uniform" else "",
                "local_mean_weight": float(weights.mean()) if method == "local" else "",
                "local_iterations": iterations if method == "local" else "", **metrics,
                "smooth_mean_weight": float(smooth_weights.mean()) if method == "local_smooth" else "",
                "smooth_iterations": smooth_iterations if method == "local_smooth" else "",
            })
    summary = {
        method: aggregate(rows, method)
        for method in ("arap_base", "uniform", "local", "local_smooth", "full_proposal")
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("data"))
    parser.add_argument("--base-package", type=Path, default=Path("runs/required_primary_chain0_scale676/neural_field_model_package_cvae_ew0p1_lw0.pt"))
    parser.add_argument("--prediction-npz", type=Path, default=Path("runs/required_primary_chain0_scale676/neural_field_predictions_cvae_ew0p1_lw0.npz"))
    parser.add_argument("--arap-npz", type=Path, default=Path("results/geometric_tuning/geometric_validation_tuned_predictions.npz"))
    parser.add_argument("--rbsr-package", type=Path, default=Path("outputs/rbsr_gate_chain0_scale676_warm_strict/rbsr_gate_model.pt"))
    parser.add_argument("--metric-csv", type=Path, default=Path("outputs/rbsr_gate_chain0_scale676_warm_strict_test/pair_metrics_rbsr_test.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/flip_fusion_diagnosis"))
    parser.add_argument("--n-pairs", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows, paths = load_manifest(args.repo)
    base = torch.load(args.base_package, map_location="cpu", weights_only=False)
    rbsr = torch.load(args.rbsr_package, map_location="cpu", weights_only=False)
    train_ids = [str(x) for x in base["train_ids"]]
    template = load_template(paths[train_ids[0]])
    feature_template = base.get("feature_template_vertices")
    feature_template_sha256 = base.get("feature_template_sha256")
    if feature_template is None or not feature_template_sha256:
        raise RuntimeError("Base package predates the train-only feature template fix")
    if array_sha256(feature_template) != feature_template_sha256:
        raise RuntimeError("Base package feature template hash mismatch")
    template["vertices"] = np.asarray(feature_template, dtype=np.float64)
    faces = template["faces"]
    edges = edge_index(faces)
    landmarks = template["landmarks"]
    all_pairs = [(str(a), str(b)) for a, b in base["test_pairs"]]
    pair_to_index = {pair: i for i, pair in enumerate(all_pairs)}
    pairs = select_pairs(args.metric_csv, args.n_pairs, args.seed)
    indices = np.asarray([pair_to_index[pair] for pair in pairs], dtype=np.int64)
    test_ids = sorted({value for pair in all_pairs for value in pair}, key=int)
    vertices = load_vertices(paths, test_ids)

    calibration = legacy_target_calibration(test_ids, vertices, faces)
    features = static_vertex_features(rows, paths, train_ids, template, bool(base["use_subunit_features"]))
    condition, source, target_delta, controls = pair_inputs(pairs, vertices, landmarks, base)
    ridge = ridge_predict(condition, base["ridge_cond"]).astype(np.float32).reshape(len(pairs), -1, 3)
    cvae = cvae_predictions(base, features, condition).reshape(len(pairs), -1, 3)
    laplacian = uniform_laplacian(len(template["vertices"]), faces)
    arap_init = solve_linear_handle_baseline(controls, landmarks, len(template["vertices"]), laplacian, 100000.0, 1e-8)
    arap = arap_predict_vectorised(
        [np.asarray(item, dtype=np.float64) for item in source],
        controls,
        landmarks,
        faces,
        laplacian,
        arap_init,
        100000.0,
        1e-8,
        3,
    ).reshape(len(pairs), -1, 3).astype(np.float32)
    rbsr_pred, gates = rbsr_predictions(base, rbsr, features, condition, source, ridge, cvae, controls, landmarks, args.batch_size)
    rbsr_pred = np.asarray(rbsr_pred, dtype=np.float32).reshape(len(pairs), -1, 3)
    global_pred = ridge + float(base["selected_alpha"]) * (cvae - ridge)
    del cvae
    gc.collect()

    metric_rows = []
    methods = {"target": target_delta, "ridge": ridge, "global": global_pred, "rbsr": rbsr_pred, "arap": arap}
    for i, (source_id, target_id) in enumerate(pairs):
        for method, prediction in methods.items():
            metric_rows.append({"source_id": source_id, "target_id": target_id, "method": method, **method_metrics(source[i], target_delta[i], prediction[i], faces, edges)})
    method_summary = {method: aggregate(metric_rows, method) for method in methods}
    cause = cause_analysis(pairs, vertices, rbsr_pred, ridge, faces, template["subunits"])

    basis = build_rbf_projection(template["vertices"], landmarks).astype(np.float64)
    projected, correction = project_exact_handles(rbsr_pred.astype(np.float64), controls.astype(np.float64), basis, landmarks)
    projection_rows = []
    for i, (source_id, target_id) in enumerate(pairs):
        before = method_metrics(source[i], target_delta[i], rbsr_pred[i], faces, edges)
        after = method_metrics(source[i], target_delta[i], projected[i], faces, edges)
        projection_rows.append({
            "source_id": source_id, "target_id": target_id,
            **{f"before_{key}": value for key, value in before.items()},
            **{f"after_{key}": value for key, value in after.items()},
            "correction_rms": float(np.sqrt(np.mean(np.sum(correction[i] ** 2, axis=1)))),
            "correction_face_gradient_p95": float(np.percentile(face_gradient(correction[i], faces, source[i]), 95)),
        })
    projection_summary = {
        key: float(np.mean([float(row[key]) for row in projection_rows]))
        for key in projection_rows[0]
        if key not in {"source_id", "target_id"}
    }

    fusion_rows, fusion_summary = fusion_analysis(pairs, vertices, target_delta, arap, rbsr_pred, faces, landmarks, basis)
    write_csv(args.out / "method_metrics.csv", metric_rows)
    write_csv(args.out / "projection_metrics.csv", projection_rows)
    write_csv(args.out / "fusion_metrics.csv", fusion_rows)
    np.savez_compressed(
        args.out / "diagnostic_sample.npz",
        pairs=np.asarray(pairs, dtype=object),
        gate_mean=np.mean(gates, axis=1).astype(np.float32),
        source_quality=triangle_quality(template["vertices"], faces).astype(np.float32),
    )
    report = {
        "status": "no_training_low_cost_diagnosis",
        "n_sample_pairs": len(pairs),
        "legacy_metric_definition": "source-to-output face-normal dot product < 0; not a proof of local non-injectivity",
        "self_intersection_checked": False,
        "target_legacy_calibration_all_test_pairs": calibration,
        "sample_method_summary": method_summary,
        "cause_analysis_rbsr": cause,
        "exact_handle_rbf_projection": projection_summary,
        "arap_plus_handle_preserving_rbsr_residual": fusion_summary,
        "thresholds": {"severe_area_ratio": 0.05, "path_steps": 21, "fusion_safety": "no new legacy reversals or severe-area faces relative to ARAP"},
        "artifacts": {"method_metrics": "method_metrics.csv", "projection_metrics": "projection_metrics.csv", "fusion_metrics": "fusion_metrics.csv"},
    }
    atomic_write_json(args.out / "diagnosis.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
