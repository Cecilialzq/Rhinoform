from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch

from rhinoform.data import edge_index, load_rows, ridge_predict
from rhinoform.stats import metric_rows_for_method
from rhinoform.train import NeuralFieldCVAE, assert_feature_template_package, build_static_vertex_features, pair_conditions, predict_field


def smooth_vertex_signal(
    signal: np.ndarray,
    faces: np.ndarray,
    iterations: int,
    strength: float,
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if iterations <= 0 or strength <= 0.0:
        return np.clip(signal, 0.0, 1.0).astype(np.float32)
    edges = edge_index(faces)
    degree = np.zeros(signal.shape[0], dtype=np.float64)
    np.add.at(degree, edges[:, 0], 1.0)
    np.add.at(degree, edges[:, 1], 1.0)
    degree = np.maximum(degree, 1.0)
    value = signal.copy()
    for _ in range(iterations):
        neighbor_sum = np.zeros_like(value)
        np.add.at(neighbor_sum, edges[:, 0], value[edges[:, 1]])
        np.add.at(neighbor_sum, edges[:, 1], value[edges[:, 0]])
        neighbor_mean = neighbor_sum / degree
        value = (1.0 - strength) * value + strength * neighbor_mean
    return np.clip(value, 0.0, 1.0).astype(np.float32)


def per_vertex_mse(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    pred = np.asarray(prediction).reshape(len(prediction), -1, 3)
    truth = np.asarray(target).reshape(len(target), -1, 3)
    return np.mean(np.sum((pred - truth) ** 2, axis=2), axis=0)


def per_vertex_mse_from_pairs(
    prediction: np.ndarray,
    by_id: dict[str, dict],
    pairs: list[tuple[str, str]],
) -> np.ndarray:
    pred = np.asarray(prediction).reshape(len(prediction), -1, 3)
    total = np.zeros(pred.shape[1], dtype=np.float64)
    for index, (source_id, target_id) in enumerate(pairs):
        truth = by_id[target_id]["vertices"] - by_id[source_id]["vertices"]
        total += np.sum((np.asarray(pred[index], dtype=np.float64) - truth) ** 2, axis=1)
    return total / max(1, len(pairs))


def mix(ridge: np.ndarray, cvae: np.ndarray, gate: np.ndarray) -> np.ndarray:
    gate3 = np.asarray(gate, dtype=np.float32).reshape(1, -1, 1)
    r = np.asarray(ridge, dtype=np.float32).reshape(len(ridge), -1, 3)
    c = np.asarray(cvae, dtype=np.float32).reshape(len(cvae), -1, 3)
    return (r + gate3 * (c - r)).reshape(len(ridge), -1)


def oracle_mix(ridge: np.ndarray, cvae: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(ridge, dtype=np.float32).reshape(len(ridge), -1, 3)
    c = np.asarray(cvae, dtype=np.float32).reshape(len(cvae), -1, 3)
    truth = np.asarray(target, dtype=np.float32).reshape(len(target), -1, 3)
    choose_cvae = np.sum((c - truth) ** 2, axis=2) < np.sum((r - truth) ** 2, axis=2)
    pred = np.where(choose_cvae[:, :, None], c, r)
    return pred.reshape(len(ridge), -1), choose_cvae.astype(np.float32)


def summarise_metrics(
    by_id: dict[str, dict],
    pairs: list[tuple[str, str]],
    methods: dict[str, np.ndarray],
) -> list[dict[str, float | str]]:
    summaries: list[dict[str, float | str]] = []
    for name, prediction in methods.items():
        rows = metric_rows_for_method(by_id, pairs, prediction)
        metric_names = [
            "roi_rmse",
            "landmark_rmse",
            "dorsum_rmse",
            "tip_rmse",
            "edge_strain_p95",
            "normal_flip_pct",
        ]
        summary: dict[str, float | str] = {"method": name, "n_pairs": float(len(rows))}
        for metric in metric_names:
            summary[metric] = float(np.mean([float(row[metric]) for row in rows]))
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-0 diagnostic for spatial ridge/CVAE mixing. The fixed gates are learned only "
            "from validation errors. The validation oracle is an upper bound, not a deployable method."
        )
    )
    parser.add_argument("--repo", required=True, help="Authoritative prepared FaceScape data directory.")
    parser.add_argument("--model-package", required=True)
    parser.add_argument("--test-predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smooth-iterations", type=int, default=3)
    parser.add_argument("--smooth-strength", type=float, default=0.5)
    parser.add_argument("--skip-test", action="store_true", help="Run validation diagnostics only.")
    args = parser.parse_args()

    if not 0.0 <= args.smooth_strength <= 1.0:
        raise ValueError("--smooth-strength must be in [0, 1]")

    repo = Path(args.repo)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false")

    print("Loading prepared meshes and frozen model package", flush=True)
    _, by_id = load_rows(repo)
    package = torch.load(args.model_package, map_location="cpu", weights_only=False)
    train_ids = [str(value) for value in package["train_ids"]]
    val_pairs = [(str(a), str(b)) for a, b in package["val_pairs"]]
    template = next(iter(by_id.values()))

    vertex_features, static = build_static_vertex_features(
        by_id,
        train_ids,
        use_subunit_features=bool(package["use_subunit_features"]),
    )
    assert_feature_template_package(package, static, "Model package")
    cond_val, _, delta_val, _, _, _, _ = pair_conditions(
        by_id,
        val_pairs,
        package["source_pca"],
        package["cond_mean"],
        package["cond_std"],
    )
    model_args = package["args"]
    obs_dim = int(np.asarray(package["obs_train_mean"]).shape[1])
    cvae = NeuralFieldCVAE(
        vertex_features.shape[1],
        cond_val.shape[1],
        obs_dim,
        latent_dim=int(model_args["latent_dim"]),
        hidden=int(model_args["hidden"]),
    )
    cvae.load_state_dict(package["cvae_state_dict"])
    cvae.to(device)

    print(f"Predicting {len(val_pairs)} validation pairs on {device}", flush=True)
    cvae_val = predict_field(cvae, vertex_features, cond_val, is_cvae=True).astype(np.float32)
    ridge_val = ridge_predict(cond_val, package["ridge_cond"]).astype(np.float32)
    selected_alpha = float(package["selected_alpha"])
    global_val = ((1.0 - selected_alpha) * ridge_val + selected_alpha * cvae_val).astype(np.float32)

    ridge_vertex_mse = per_vertex_mse(ridge_val, delta_val)
    cvae_vertex_mse = per_vertex_mse(cvae_val, delta_val)
    advantage = ridge_vertex_mse - cvae_vertex_mse
    hard_gate = (advantage > 0.0).astype(np.float32)
    smooth_gate = smooth_vertex_signal(
        hard_gate,
        template["faces"],
        args.smooth_iterations,
        args.smooth_strength,
    )
    scale = float(np.median(np.abs(advantage)))
    scale = max(scale, 1e-12)
    soft_gate = (1.0 / (1.0 + np.exp(-np.clip(advantage / scale, -30.0, 30.0)))).astype(np.float32)
    soft_smooth_gate = smooth_vertex_signal(
        soft_gate,
        template["faces"],
        args.smooth_iterations,
        args.smooth_strength,
    )

    oracle_val, oracle_choice = oracle_mix(ridge_val, cvae_val, delta_val)
    validation_methods = {
        "ridge_anchor": ridge_val,
        f"global_hybrid_alpha_{selected_alpha:g}": global_val,
        "fixed_hard_spatial_gate": mix(ridge_val, cvae_val, hard_gate),
        "fixed_smoothed_hard_gate": mix(ridge_val, cvae_val, smooth_gate),
        "fixed_smoothed_soft_gate": mix(ridge_val, cvae_val, soft_smooth_gate),
        "validation_oracle_upper_bound": oracle_val,
    }
    print("Computing exact validation metrics", flush=True)
    validation_summary = summarise_metrics(by_id, val_pairs, validation_methods)
    write_csv(out_dir / "validation_spatial_gate_summary.csv", validation_summary)

    report: dict[str, object] = {
        "purpose": "Stage-0 diagnostic only; not a final method result.",
        "selection_boundary": "All fixed spatial gates are derived from validation per-vertex error only.",
        "oracle_warning": "validation_oracle_upper_bound uses ground truth per pair and cannot be deployed or tested as a selected method.",
        "model_package": str(Path(args.model_package)),
        "test_predictions": str(Path(args.test_predictions)),
        "selected_global_alpha": selected_alpha,
        "smoothing": {"iterations": args.smooth_iterations, "strength": args.smooth_strength},
        "gate_diagnostics": {
            "hard_gate_cvae_vertex_fraction": float(np.mean(hard_gate)),
            "smoothed_hard_gate_mean": float(np.mean(smooth_gate)),
            "smoothed_soft_gate_mean": float(np.mean(soft_smooth_gate)),
            "validation_oracle_cvae_decision_fraction": float(np.mean(oracle_choice)),
            "validation_advantage_median_absolute_scale": scale,
        },
        "validation": validation_summary,
    }

    np.savez_compressed(
        out_dir / "validation_derived_spatial_gates.npz",
        hard_gate=hard_gate,
        smoothed_hard_gate=smooth_gate,
        soft_gate=soft_gate,
        smoothed_soft_gate=soft_smooth_gate,
        validation_ridge_minus_cvae_vertex_mse=advantage.astype(np.float32),
    )

    if not args.skip_test:
        del validation_methods, oracle_val, oracle_choice, ridge_val, cvae_val, global_val, delta_val
        cvae.to("cpu")
        del cvae
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        print("Loading frozen test predictions for one-shot transfer evaluation", flush=True)
        test_npz = np.load(args.test_predictions, allow_pickle=True)
        test_pairs = [(str(a), str(b)) for a, b in test_npz["test_pairs"].tolist()]
        expected_test_pairs = [(str(a), str(b)) for a, b in package["test_pairs"]]
        if test_pairs != expected_test_pairs:
            raise ValueError("Test prediction pair order does not match the frozen model package")
        ridge_test = np.asarray(test_npz["ridge_cond_pred"], dtype=np.float32)
        cvae_test = np.asarray(test_npz["cvae_pred"], dtype=np.float32)
        print("Computing exact frozen-test metrics", flush=True)
        test_summary: list[dict[str, float | str]] = []
        test_summary.extend(summarise_metrics(by_id, test_pairs, {"ridge_anchor": ridge_test}))
        gate_specs = [
            (f"global_hybrid_alpha_{selected_alpha:g}", np.full(len(hard_gate), selected_alpha, dtype=np.float32)),
            ("fixed_hard_spatial_gate", hard_gate),
            ("fixed_smoothed_hard_gate", smooth_gate),
            ("fixed_smoothed_soft_gate", soft_smooth_gate),
        ]
        for method_name, gate in gate_specs:
            prediction = mix(ridge_test, cvae_test, gate)
            test_summary.extend(summarise_metrics(by_id, test_pairs, {method_name: prediction}))
            del prediction
            gc.collect()
        write_csv(out_dir / "test_spatial_gate_summary.csv", test_summary)
        test_ridge_mse = per_vertex_mse_from_pairs(ridge_test, by_id, test_pairs)
        test_cvae_mse = per_vertex_mse_from_pairs(cvae_test, by_id, test_pairs)
        test_advantage = test_ridge_mse - test_cvae_mse
        correlation = float(np.corrcoef(advantage, test_advantage)[0, 1])
        report["test"] = test_summary
        report["transfer_diagnostics"] = {
            "validation_test_vertex_advantage_correlation": correlation,
            "test_fraction_vertices_favouring_cvae": float(np.mean(test_advantage > 0.0)),
        }

    (out_dir / "spatial_residual_feasibility.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
