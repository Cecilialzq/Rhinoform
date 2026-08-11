from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from rhinoform.data import load_rows, ridge_predict
from rhinoform.stats import metric_rows_for_method
from rhinoform.train import NeuralFieldCVAE, assert_feature_template_package, build_static_vertex_features, pair_conditions, predict_field
from rhinoform.train_rbsr_gate import SpatialRiskGate, predict_gate_batches, resolve_device


def read_metric_rows(metrics_path: Path) -> list[dict[str, str]]:
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_representative_pair(metrics_path: Path, comparison_path: Path) -> tuple[tuple[str, str], float]:
    rows = read_metric_rows(metrics_path)
    comparison_rows = read_metric_rows(comparison_path)
    comparison = {
        (str(row["source_id"]), str(row["target_id"])): float(row["roi_rmse"])
        for row in comparison_rows
    }
    deltas = np.asarray(
        [float(row["roi_rmse"]) - comparison[(str(row["source_id"]), str(row["target_id"]))] for row in rows]
    )
    mean_delta = float(np.mean(deltas))
    index = int(np.argmin(np.abs(deltas - mean_delta)))
    row = rows[index]
    return (str(row["source_id"]), str(row["target_id"])), mean_delta


def compact(values: np.ndarray, decimals: int = 6) -> list:
    return np.round(np.asarray(values), decimals=decimals).tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export one deterministic frozen RBSR case for the communication demo.")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-model-package", type=Path, required=True)
    parser.add_argument("--rbsr-package", type=Path, required=True)
    parser.add_argument("--pair-metrics", type=Path, required=True)
    parser.add_argument("--comparison-metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    pair, mean_roi_delta = select_representative_pair(args.pair_metrics, args.comparison_metrics)
    device = resolve_device(args.device)
    _, by_id = load_rows(args.repo)
    base = torch.load(args.base_model_package, map_location="cpu", weights_only=False)
    rbsr = torch.load(args.rbsr_package, map_location="cpu", weights_only=False)
    train_ids = [str(value) for value in base["train_ids"]]
    vertex_features_np, static = build_static_vertex_features(
        by_id, train_ids, use_subunit_features=bool(base["use_subunit_features"])
    )
    assert_feature_template_package(base, static, "Base package")
    condition, source, target, controls, _, _, _ = pair_conditions(
        by_id, [pair], base["source_pca"], base["cond_mean"], base["cond_std"]
    )
    ridge = ridge_predict(condition, base["ridge_cond"]).astype(np.float32)

    base_args = base["args"]
    obs_dim = int(np.asarray(base["obs_train_mean"]).shape[1])
    cvae = NeuralFieldCVAE(
        vertex_features_np.shape[1],
        condition.shape[1],
        obs_dim,
        latent_dim=int(base_args["latent_dim"]),
        hidden=int(base_args["hidden"]),
    ).to(device)
    cvae.load_state_dict(base["cvae_state_dict"])
    cvae.eval()
    cvae_prediction = predict_field(cvae, vertex_features_np, condition, is_cvae=True).astype(np.float32)

    rbsr_args = rbsr["args"]
    gate_model = SpatialRiskGate(
        int(rbsr["gate_vertex_dim"]),
        int(rbsr["gate_cond_dim"]),
        int(rbsr_args["hidden"]),
        float(base["selected_alpha"]),
    ).to(device)
    gate_model.load_state_dict(rbsr["gate_state_dict"])
    gate_model.eval()
    vertex_features = torch.as_tensor(vertex_features_np, dtype=torch.float32, device=device)
    center = torch.as_tensor(np.asarray(rbsr["center"]), dtype=torch.float32, device=device)
    landmarks_np = np.asarray(next(iter(by_id.values()))["landmarks"], dtype=np.int64)
    landmarks = torch.as_tensor(landmarks_np, dtype=torch.long, device=device)
    projection_basis = None
    if rbsr.get("projection_basis") is not None:
        projection_basis = torch.as_tensor(rbsr["projection_basis"], dtype=torch.float32, device=device)
    rbsr_prediction, gate_values = predict_gate_batches(
        gate_model,
        vertex_features,
        condition,
        source,
        ridge,
        cvae_prediction,
        center,
        float(rbsr["scale"]),
        landmarks,
        controls,
        projection_basis,
        batch_size=1,
    )
    global_prediction = ridge + float(base["selected_alpha"]) * (cvae_prediction - ridge)

    methods = {
        "ridge": ridge,
        "global": global_prediction,
        "rbsr": rbsr_prediction,
    }
    metrics = {
        name: {
            key: float(value)
            for key, value in metric_rows_for_method(by_id, [pair], prediction)[0].items()
            if key not in {"method", "source_id", "target_id"}
        }
        for name, prediction in methods.items()
    }
    source_vertices = np.asarray(source[0], dtype=np.float32)
    target_delta = np.asarray(target[0], dtype=np.float32).reshape(source_vertices.shape)
    target_vertices = source_vertices + target_delta
    source_row = by_id[pair[0]]
    payload = {
        "case": {
            "selection": "deterministic pair closest to the mean RBSR-vs-global ROI effect",
            "source_id": pair[0],
            "target_id": pair[1],
            "selected_alpha": float(base["selected_alpha"]),
            "method_status": "validation-selected; one-shot frozen test case",
            "aggregate_rbsr_minus_global_roi": mean_roi_delta,
        },
        "disclaimer": "Illustrative research communication prototype; not diagnosis, treatment planning, outcome prediction or a guarantee.",
        "geometry": {
            "source": compact(source_vertices),
            "target": compact(target_vertices),
            "faces": np.asarray(source_row["faces"], dtype=np.int32).tolist(),
            "landmarks": landmarks_np.tolist(),
            "controls": compact(controls[0]),
            "gate": compact(gate_values[0], decimals=5),
        },
        "methods": {
            name: {
                "vertices": compact(source_vertices + np.asarray(prediction[0]).reshape(source_vertices.shape)),
                "metrics": metrics[name],
            }
            for name, prediction in methods.items()
        },
        "metric_notes": {
            "roi_rmse": "Dataset-coordinate reconstruction error; FaceScape provides no official millimetre interpretation.",
            "normal_flip_pct": "Percentage of ROI faces whose orientation reverses relative to the source.",
            "edge_strain_p95": "95th percentile relative edge-length change.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "pair": pair, "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
