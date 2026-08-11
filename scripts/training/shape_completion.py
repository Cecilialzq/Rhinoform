from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from rhinoform.data import edge_index
from rhinoform.repro import atomic_write_json, sha256_file
from scripts.training.residual_basis_expert import load_selected_rows


def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    centered = x - x_mean
    weight = np.linalg.solve(
        centered.T @ centered + float(lam) * np.eye(x.shape[1]),
        centered.T @ (y - y_mean),
    )
    return weight, x_mean, y_mean


def predict(model: tuple[np.ndarray, np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    weight, x_mean, y_mean = model
    return (x - x_mean) @ weight + y_mean


def shape_rmse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    diff = prediction.reshape(len(prediction), -1, 3) - target.reshape(len(target), -1, 3)
    return np.sqrt(np.mean(np.sum(diff * diff, axis=2), axis=1))


def select_lambda_cv(
    x: np.ndarray,
    y: np.ndarray,
    lambdas: list[float],
    folds: int,
    seed: int,
) -> tuple[float, list[dict[str, float]]]:
    order = np.random.default_rng(seed).permutation(len(x))
    fold_ids = np.array_split(order, folds)
    rows: list[dict[str, float]] = []
    for lam in lambdas:
        scores = []
        for held_out in fold_ids:
            keep = np.setdiff1d(order, held_out, assume_unique=True)
            model = fit_ridge(x[keep], y[keep], lam)
            scores.append(float(shape_rmse(predict(model, x[held_out]), y[held_out]).mean()))
        rows.append({"lambda": float(lam), "cv_roi_rmse": float(np.mean(scores))})
    best = min(rows, key=lambda row: row["cv_roi_rmse"])
    return float(best["lambda"]), rows


def exact_pair_metrics(
    by_id: dict[str, dict],
    pairs: list[tuple[str, str]],
    predicted_shapes: dict[str, np.ndarray],
    batch_size: int,
) -> dict[str, float]:
    template = next(iter(by_id.values()))
    landmarks = template["landmarks"]
    faces = template["faces"]
    edges = edge_index(faces)
    values = {"roi_rmse": [], "landmark_rmse": [], "edge_strain_p95": [], "normal_flip_pct": []}
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        source = np.stack([by_id[src]["vertices"] for src, _ in batch], axis=0)
        target = np.stack([by_id[tgt]["vertices"] for _, tgt in batch], axis=0)
        edited = np.stack([predicted_shapes[tgt] for _, tgt in batch], axis=0)
        error = edited - target
        values["roi_rmse"].append(np.sqrt(np.mean(np.sum(error * error, axis=2), axis=1)))
        landmark_error = error[:, landmarks]
        values["landmark_rmse"].append(
            np.sqrt(np.mean(np.sum(landmark_error * landmark_error, axis=2), axis=1))
        )

        source_edge = source[:, edges[:, 0]] - source[:, edges[:, 1]]
        edited_edge = edited[:, edges[:, 0]] - edited[:, edges[:, 1]]
        source_length = np.linalg.norm(source_edge, axis=2)
        strain = np.abs(np.linalg.norm(edited_edge, axis=2) - source_length) / np.maximum(source_length, 1e-8)
        values["edge_strain_p95"].append(np.percentile(strain, 95, axis=1))

        source_tri = source[:, faces]
        edited_tri = edited[:, faces]
        source_normal = np.cross(
            source_tri[:, :, 1] - source_tri[:, :, 0], source_tri[:, :, 2] - source_tri[:, :, 0]
        )
        edited_normal = np.cross(
            edited_tri[:, :, 1] - edited_tri[:, :, 0], edited_tri[:, :, 2] - edited_tri[:, :, 0]
        )
        values["normal_flip_pct"].append(
            100.0 * np.mean(np.sum(source_normal * edited_normal, axis=2) < 0.0, axis=1)
        )
    return {key: float(np.mean(np.concatenate(parts))) for key, parts in values.items()}


def read_validation_baselines(package: dict) -> list[dict[str, str]]:
    path = Path(str(package.get("validation_operating_point_candidates", "")))
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Target landmark to registered 3D shape completion anchor.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base-model-package", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lambdas", default="0.000001,0.0001,0.01,0.1,1,10,100")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    package = torch.load(args.base_model_package, map_location="cpu", weights_only=False)
    train_ids = [str(value) for value in package["train_ids"]]
    val_ids = [str(value) for value in package["val_ids"]]
    val_pairs = [(str(a), str(b)) for a, b in package["val_pairs"]]
    by_id = load_selected_rows(Path(args.repo), set(train_ids) | set(val_ids), workers=args.workers)
    landmarks = next(iter(by_id.values()))["landmarks"]
    train_x = np.stack([by_id[sid]["vertices"][landmarks].reshape(-1) for sid in train_ids]).astype(np.float64)
    train_y = np.stack([by_id[sid]["vertices"].reshape(-1) for sid in train_ids]).astype(np.float64)
    val_x = np.stack([by_id[sid]["vertices"][landmarks].reshape(-1) for sid in val_ids]).astype(np.float64)
    val_y = np.stack([by_id[sid]["vertices"].reshape(-1) for sid in val_ids]).astype(np.float64)
    lambdas = [float(value) for value in args.lambdas.split(",") if value.strip()]
    selected_lambda, cv_rows = select_lambda_cv(train_x, train_y, lambdas, args.cv_folds, args.seed)
    print(f"selected lambda={selected_lambda:g} from train-only identity CV", flush=True)
    model = fit_ridge(train_x, train_y, selected_lambda)
    val_prediction = predict(model, val_x).astype(np.float32)
    identity_errors = shape_rmse(val_prediction, val_y)
    predicted_shapes = {
        subject_id: prediction.reshape(-1, 3)
        for subject_id, prediction in zip(val_ids, val_prediction)
    }
    metrics = exact_pair_metrics(by_id, val_pairs, predicted_shapes, args.batch_size)
    baselines = read_validation_baselines(package)
    ridge_row = next((row for row in baselines if row.get("label") == "ridge_anchor"), None)
    global_row = next((row for row in baselines if row.get("label") == "hybrid_alpha_0.1"), None)
    comparisons = {}
    if ridge_row is not None:
        comparisons["relative_roi_improvement_vs_ridge_pct"] = 100.0 * (
            float(ridge_row["roi_rmse"]) - metrics["roi_rmse"]
        ) / float(ridge_row["roi_rmse"])
    if global_row is not None:
        comparisons["relative_roi_improvement_vs_global_hybrid_pct"] = 100.0 * (
            float(global_row["roi_rmse"]) - metrics["roi_rmse"]
        ) / float(global_row["roi_rmse"])

    model_payload = {
        "method": "target_landmark_shape_completion_anchor",
        "status": "train_cv_selected_validation_evaluated_no_test_access",
        "args": vars(args),
        "base_model_package": str(Path(args.base_model_package)),
        "base_model_package_sha256": sha256_file(Path(args.base_model_package)),
        "selected_lambda": selected_lambda,
        "weight": model[0],
        "x_mean": model[1],
        "y_mean": model[2],
        "landmarks": landmarks,
        "train_ids": train_ids,
        "val_ids": val_ids,
    }
    model_path = out_dir / "target_shape_completion.pt"
    torch.save(model_payload, model_path)
    report = {
        "method": model_payload["method"],
        "status": model_payload["status"],
        "selected_lambda": selected_lambda,
        "train_identity_cv": cv_rows,
        "validation": {
            **metrics,
            "identity_rmse_median": float(np.median(identity_errors)),
            "identity_rmse_p90": float(np.percentile(identity_errors, 90)),
            **comparisons,
        },
        "baseline_rows": [row for row in (ridge_row, global_row) if row is not None],
        "model_package": str(model_path),
        "model_package_sha256": sha256_file(model_path),
    }
    atomic_write_json(out_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
