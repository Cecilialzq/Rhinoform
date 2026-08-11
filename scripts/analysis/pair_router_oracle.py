from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from rhinoform.data import array_sha256, edge_index, ridge_predict
from rhinoform.repro import atomic_write_json
from rhinoform.train import NeuralFieldCVAE, face_vertex_normals, pair_conditions, predict_field, subunit_one_hot


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_selected_rows(repo: Path, identity_ids: set[str]) -> dict[str, dict]:
    manifest = json.loads((repo / "manifest.json").read_text(encoding="utf-8"))
    by_id: dict[str, dict] = {}
    for row in manifest["rows"]:
        subject_id = str(row["subject_id"])
        if subject_id not in identity_ids:
            continue
        npz_path = Path(row["npz_path"])
        if not npz_path.is_absolute():
            npz_path = repo / npz_path
        data = np.load(npz_path, allow_pickle=False)
        by_id[subject_id] = {
            "subject_id": subject_id,
            "split": str(row["split"]),
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
    missing = identity_ids - set(by_id)
    if missing:
        raise ValueError(f"Missing requested identities: {sorted(missing)}")
    return by_id


def frozen_vertex_features(
    template: dict,
    center: np.ndarray,
    scale: float,
    use_subunit_features: bool,
) -> np.ndarray:
    base = np.asarray(template["vertices"], dtype=np.float64)
    faces = np.asarray(template["faces"], dtype=np.int64)
    landmarks = np.asarray(template["landmarks"], dtype=np.int64)
    coords = (base - np.asarray(center, dtype=np.float64).reshape(1, 3)) / float(scale)
    normals = face_vertex_normals(base, faces)
    landmark_vertices = base[landmarks]
    distance = np.linalg.norm(base[:, None, :] - landmark_vertices[None, :, :], axis=2)
    distance = distance / np.maximum(np.percentile(distance, 95, axis=0, keepdims=True), 1e-6)
    parts = [coords, normals]
    if use_subunit_features:
        parts.append(subunit_one_hot(template))
    parts.append(distance)
    return np.concatenate(parts, axis=1).astype(np.float32)


def exact_pair_metrics(
    source: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    faces: np.ndarray,
    edges: np.ndarray,
    batch_size: int,
) -> dict[str, np.ndarray]:
    n_pairs = len(source)
    roi = np.empty(n_pairs, dtype=np.float64)
    strain = np.empty(n_pairs, dtype=np.float64)
    flips = np.empty(n_pairs, dtype=np.float64)
    for start in range(0, n_pairs, batch_size):
        stop = min(n_pairs, start + batch_size)
        src = np.asarray(source[start:stop], dtype=np.float64)
        truth = np.asarray(target[start:stop], dtype=np.float64).reshape(stop - start, -1, 3)
        pred = np.asarray(prediction[start:stop], dtype=np.float64).reshape(stop - start, -1, 3)
        roi[start:stop] = np.sqrt(np.mean(np.sum((pred - truth) ** 2, axis=2), axis=1))

        src_edge = src[:, edges[:, 0]] - src[:, edges[:, 1]]
        edit_edge = src_edge + pred[:, edges[:, 0]] - pred[:, edges[:, 1]]
        src_length = np.linalg.norm(src_edge, axis=2)
        edit_length = np.linalg.norm(edit_edge, axis=2)
        edge_strain = np.abs(edit_length - src_length) / np.maximum(src_length, 1e-8)
        strain[start:stop] = np.percentile(edge_strain, 95, axis=1)

        src_tri = src[:, faces]
        edit_tri = src_tri + pred[:, faces]
        src_normal = np.cross(src_tri[:, :, 1] - src_tri[:, :, 0], src_tri[:, :, 2] - src_tri[:, :, 0])
        edit_normal = np.cross(
            edit_tri[:, :, 1] - edit_tri[:, :, 0],
            edit_tri[:, :, 2] - edit_tri[:, :, 0],
        )
        dot = np.sum(src_normal * edit_normal, axis=2)
        flips[start:stop] = 100.0 * np.mean(dot < 0.0, axis=1)
    return {"roi_rmse": roi, "edge_strain_p95": strain, "normal_flip_pct": flips}


def choose_oracle(
    metrics: dict[str, np.ndarray],
    alphas: np.ndarray,
    ridge_metrics: dict[str, np.ndarray],
    strain_tolerance: float,
    flip_tolerance_pp: float,
) -> dict[str, np.ndarray]:
    roi = metrics["roi_rmse"]
    feasible = (
        metrics["edge_strain_p95"]
        <= ridge_metrics["edge_strain_p95"][None, :] * (1.0 + strain_tolerance)
    ) & (
        metrics["normal_flip_pct"]
        <= ridge_metrics["normal_flip_pct"][None, :] + flip_tolerance_pp
    )
    feasible[0] = True  # alpha=0 is the ridge anchor and must remain available.
    penalised = np.where(feasible, roi, np.inf)
    selected = np.argmin(penalised, axis=0)
    pair_index = np.arange(roi.shape[1])
    return {
        "selected_index": selected,
        "selected_alpha": alphas[selected],
        "roi_rmse": metrics["roi_rmse"][selected, pair_index],
        "edge_strain_p95": metrics["edge_strain_p95"][selected, pair_index],
        "normal_flip_pct": metrics["normal_flip_pct"][selected, pair_index],
        "feasible_fraction_by_alpha": np.mean(feasible, axis=1),
    }


def summary(name: str, values: dict[str, np.ndarray]) -> dict[str, float | str]:
    row: dict[str, float | str] = {"method": name}
    for metric in ("roi_rmse", "edge_strain_p95", "normal_flip_pct"):
        row[metric] = float(np.mean(values[metric]))
    if "selected_alpha" in values:
        alpha = values["selected_alpha"]
        row.update(
            selected_alpha_mean=float(np.mean(alpha)),
            selected_alpha_p50=float(np.percentile(alpha, 50)),
            selected_alpha_p90=float(np.percentile(alpha, 90)),
            selected_nonzero_fraction=float(np.mean(alpha > 0.0)),
        )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Validation-only pair-router oracle analysis.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model-package", required=True)
    parser.add_argument("--static-package", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--strain-tolerance", type=float, default=0.02)
    parser.add_argument("--flip-tolerance-pp", type=float, default=0.02)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    package = torch.load(args.model_package, map_location="cpu", weights_only=False)
    static_package = torch.load(args.static_package, map_location="cpu", weights_only=False)
    val_pairs = [(str(a), str(b)) for a, b in package["val_pairs"]]
    val_ids = {identity for pair in val_pairs for identity in pair}
    by_id = load_selected_rows(Path(args.repo), val_ids)
    template = next(iter(by_id.values()))
    feature_template = package.get("feature_template_vertices")
    feature_template_sha256 = package.get("feature_template_sha256")
    if feature_template is None or not feature_template_sha256:
        raise RuntimeError("Model package predates the train-only feature template fix")
    if array_sha256(feature_template) != feature_template_sha256:
        raise RuntimeError("Model package feature template hash mismatch")
    template = {**template, "vertices": np.asarray(feature_template, dtype=np.float64)}
    faces = np.asarray(template["faces"], dtype=np.int64)
    edges = edge_index(faces)

    vertex_features = frozen_vertex_features(
        template,
        np.asarray(static_package["center"]),
        float(static_package["scale"]),
        bool(package["use_subunit_features"]),
    )
    cond, source, target, _, _, _, _ = pair_conditions(
        by_id,
        val_pairs,
        package["source_pca"],
        package["cond_mean"],
        package["cond_std"],
    )
    ridge = ridge_predict(cond, package["ridge_cond"]).astype(np.float32)

    device = resolve_device(args.device)
    model_args = package["args"]
    obs_dim = int(np.asarray(package["obs_train_mean"]).shape[1])
    cvae = NeuralFieldCVAE(
        vertex_features.shape[1],
        cond.shape[1],
        obs_dim,
        latent_dim=int(model_args["latent_dim"]),
        hidden=int(model_args["hidden"]),
    )
    cvae.load_state_dict(package["cvae_state_dict"])
    cvae.to(device).eval()
    cvae_prediction = predict_field(cvae, vertex_features, cond, is_cvae=True).astype(np.float32)

    alphas = np.arange(0.0, 1.0 + args.alpha_step * 0.5, args.alpha_step, dtype=np.float64)
    metric_grid = {
        "roi_rmse": np.empty((len(alphas), len(val_pairs)), dtype=np.float64),
        "edge_strain_p95": np.empty((len(alphas), len(val_pairs)), dtype=np.float64),
        "normal_flip_pct": np.empty((len(alphas), len(val_pairs)), dtype=np.float64),
    }
    residual = cvae_prediction - ridge
    for alpha_index, alpha in enumerate(alphas):
        prediction = ridge + float(alpha) * residual
        one = exact_pair_metrics(source, target, prediction, faces, edges, args.batch_size)
        for metric in metric_grid:
            metric_grid[metric][alpha_index] = one[metric]

    ridge_metrics = {metric: values[0] for metric, values in metric_grid.items()}
    selected_global_alpha = float(package["selected_alpha"])
    global_index = int(np.argmin(np.abs(alphas - selected_global_alpha)))
    global_metrics = {metric: values[global_index] for metric, values in metric_grid.items()}

    unconstrained_index = np.argmin(metric_grid["roi_rmse"], axis=0)
    pair_index = np.arange(len(val_pairs))
    unconstrained = {
        "selected_alpha": alphas[unconstrained_index],
        **{
            metric: values[unconstrained_index, pair_index]
            for metric, values in metric_grid.items()
        },
    }
    constrained = choose_oracle(
        metric_grid,
        alphas,
        ridge_metrics,
        args.strain_tolerance,
        args.flip_tolerance_pp,
    )
    rows = [
        summary("ridge", ridge_metrics),
        summary(f"global_alpha_{alphas[global_index]:g}", global_metrics),
        summary("pair_alpha_oracle_unconstrained", unconstrained),
        summary("pair_alpha_oracle_ridge_constrained", constrained),
    ]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        out_dir / "pair_oracle.npz",
        alphas=alphas,
        pairs=np.asarray(val_pairs, dtype=object),
        roi_rmse=metric_grid["roi_rmse"],
        edge_strain_p95=metric_grid["edge_strain_p95"],
        normal_flip_pct=metric_grid["normal_flip_pct"],
        unconstrained_alpha=unconstrained["selected_alpha"],
        constrained_alpha=constrained["selected_alpha"],
    )
    report = {
        "scope": "validation_only",
        "n_pairs": len(val_pairs),
        "alpha_grid": alphas.tolist(),
        "selected_global_alpha": selected_global_alpha,
        "constraint": {
            "reference": "per-pair ridge",
            "strain_relative_tolerance": args.strain_tolerance,
            "flip_absolute_tolerance_pp": args.flip_tolerance_pp,
        },
        "summary": rows,
        "feasible_fraction_by_alpha": constrained["feasible_fraction_by_alpha"].tolist(),
    }
    atomic_write_json(out_dir / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
