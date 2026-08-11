from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from rhinoform.data import load_rows, ridge_predict
from rhinoform.repro import atomic_write_json, sha256_file
from rhinoform.strict_protocol_patch import strict_metric_rows as metric_rows_for_method
from rhinoform.train import NeuralFieldCVAE, assert_feature_template_package, build_static_vertex_features, pair_conditions, predict_field
from rhinoform.train_rbsr_gate import SpatialRiskGate, predict_gate_batches, resolve_device


def parse_models(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Model must use label=path syntax: {value!r}")
        label, path = value.split("=", 1)
        result[label.strip()] = Path(path.strip())
    return result


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate multiple RBSR gate ablations with shared frozen predictions.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base-model-package", required=True)
    parser.add_argument("--model", action="append", required=True, help="Repeat label=RBSR_PACKAGE_PATH.")
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    models = parse_models(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    base = torch.load(args.base_model_package, map_location="cpu", weights_only=False)
    pairs_key = "val_pairs" if args.split == "validation" else "test_pairs"
    pairs = [(str(a), str(b)) for a, b in base[pairs_key]]
    train_ids = [str(value) for value in base["train_ids"]]
    required_ids = set(train_ids) | {value for pair in pairs for value in pair}
    _, by_id = load_rows(Path(args.repo), allowed_ids=required_ids)
    vertex_features_np, static = build_static_vertex_features(
        by_id, train_ids, use_subunit_features=bool(base["use_subunit_features"])
    )
    assert_feature_template_package(base, static, "Base package")
    condition, source, _, controls, _, _, _ = pair_conditions(
        by_id, pairs, base["source_pca"], base["cond_mean"], base["cond_std"]
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
    )
    cvae.load_state_dict(base["cvae_state_dict"])
    cvae.to(device).eval()
    print(f"Caching shared CVAE predictions for {len(pairs)} {args.split} pairs", flush=True)
    cvae_prediction = predict_field(cvae, vertex_features_np, condition, is_cvae=True).astype(np.float32)
    cvae.to("cpu")
    del cvae
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    vertex_features = torch.as_tensor(vertex_features_np, dtype=torch.float32, device=device)
    landmarks = torch.as_tensor(next(iter(by_id.values()))["landmarks"], dtype=torch.long, device=device)
    summary_rows: list[dict[str, float | str]] = []
    reports: dict[str, object] = {}
    for label, package_path in models.items():
        print(f"Evaluating {label}", flush=True)
        package = torch.load(package_path, map_location="cpu", weights_only=False)
        package_args = package["args"]
        gate = SpatialRiskGate(
            int(package["gate_vertex_dim"]),
            int(package["gate_cond_dim"]),
            int(package_args["hidden"]),
            float(base["selected_alpha"]),
        ).to(device)
        gate.load_state_dict(package["gate_state_dict"])
        center = torch.as_tensor(np.asarray(package["center"]), dtype=torch.float32, device=device)
        projection_basis = None
        if package.get("projection_basis") is not None:
            projection_basis = torch.as_tensor(package["projection_basis"], dtype=torch.float32, device=device)
        prediction, gate_values = predict_gate_batches(
            gate,
            vertex_features,
            condition,
            source,
            ridge,
            cvae_prediction,
            center,
            float(package["scale"]),
            landmarks,
            controls,
            projection_basis,
            args.batch_size,
        )
        rows = metric_rows_for_method(by_id, pairs, prediction)
        metric_names = ["roi_rmse", "landmark_rmse", "dorsum_rmse", "tip_rmse", "edge_strain_p95", "normal_flip_pct"]
        summary: dict[str, float | str] = {"label": label, "n_pairs": float(len(rows))}
        for metric in metric_names:
            summary[metric] = float(np.mean([float(row[metric]) for row in rows]))
        summary["gate_mean"] = float(np.mean(gate_values))
        summary["gate_p95"] = float(np.percentile(gate_values, 95))
        summary["gate_active_fraction_0p5"] = float(np.mean(gate_values >= 0.5))
        summary_rows.append(summary)
        write_csv(out_dir / f"pair_metrics_{label}_{args.split}.csv", rows)
        reports[label] = {
            "package": str(package_path),
            "package_sha256": sha256_file(package_path),
            "training_mode": package_args["mode"],
            "best_validation": package["best_validation"],
            "summary": summary,
        }
        del gate, prediction, gate_values

    write_csv(out_dir / f"rbsr_ablation_summary_{args.split}.csv", summary_rows)
    report = {
        "split": args.split,
        "n_pairs": len(pairs),
        "selection_boundary": "Each checkpoint was selected using validation only; the test pass is one-shot.",
        "models": reports,
    }
    atomic_write_json(out_dir / f"rbsr_ablation_summary_{args.split}.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
