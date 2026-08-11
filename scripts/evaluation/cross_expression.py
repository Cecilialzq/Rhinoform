from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rhinoform.data import load_rows, ridge_predict
from rhinoform.repro import atomic_write_json, sha256_file
from rhinoform.stats import metric_rows_for_method, write_csv
from rhinoform.train import NeuralFieldCVAE, assert_feature_template_package, build_static_vertex_features, pair_conditions, predict_field
from rhinoform.train_rbsr_gate import SpatialRiskGate, predict_gate_batches, resolve_device


METRICS = (
    "roi_rmse",
    "landmark_rmse",
    "dorsum_rmse",
    "tip_rmse",
    "edge_strain_p95",
    "normal_flip_pct",
)


def metric_summary(rows: list[dict[str, float | str]]) -> dict[str, float]:
    return {name: float(np.mean([float(row[name]) for row in rows])) for name in METRICS}


def load_expression_rows(
    expression_root: Path,
    neutral_by_id: dict[str, dict],
) -> tuple[dict[str, dict[str, dict]], dict]:
    manifest_path = expression_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    grouped: dict[str, dict[str, dict]] = {}
    for row in manifest["rows"]:
        subject_id = str(row["subject_id"])
        if subject_id not in neutral_by_id:
            continue
        npz_path = Path(row["npz_path"])
        if not npz_path.is_absolute():
            npz_path = expression_root / npz_path
        data = np.load(npz_path, allow_pickle=False)
        neutral = neutral_by_id[subject_id]
        vertices = np.asarray(data["vertices"], dtype=np.float64)
        if vertices.shape != neutral["vertices"].shape:
            raise ValueError(
                f"{npz_path}: expression shape {vertices.shape} != neutral shape "
                f"{neutral['vertices'].shape}"
            )
        grouped.setdefault(str(row["expression"]), {})[subject_id] = {
            **neutral,
            "vertices": vertices,
            "expression": str(row["expression"]),
        }
    return grouped, manifest


def build_cvae(base: dict, vertex_dim: int, condition_dim: int, device: torch.device) -> NeuralFieldCVAE:
    base_args = base["args"]
    obs_dim = int(np.asarray(base["obs_train_mean"]).shape[1])
    model = NeuralFieldCVAE(
        vertex_dim,
        condition_dim,
        obs_dim,
        latent_dim=int(base_args["latent_dim"]),
        hidden=int(base_args["hidden"]),
    )
    model.load_state_dict(base["cvae_state_dict"])
    return model.to(device).eval()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen neutral-trained methods on expression-preserving test pairs. "
            "No model selection or tuning is performed."
        )
    )
    parser.add_argument("--neutral-repo", type=Path, required=True)
    parser.add_argument("--expression-repo", type=Path, required=True)
    parser.add_argument("--base-model-package", type=Path, required=True)
    parser.add_argument("--rbsr-package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--complete-case",
        action="store_true",
        help="Evaluate the intersection of frozen test identities available in every expression.",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    _, neutral_by_id = load_rows(args.neutral_repo)
    expressions, expression_manifest = load_expression_rows(args.expression_repo, neutral_by_id)
    base = torch.load(args.base_model_package, map_location="cpu", weights_only=False)
    rbsr = torch.load(args.rbsr_package, map_location="cpu", weights_only=False)
    frozen_pairs = [(str(a), str(b)) for a, b in base["test_pairs"]]
    pairs = list(frozen_pairs)
    required_ids = {value for pair in pairs for value in pair}
    complete_ids = set(required_ids)
    for expression_rows in expressions.values():
        complete_ids &= set(expression_rows)
    if args.complete_case:
        pairs = [pair for pair in frozen_pairs if pair[0] in complete_ids and pair[1] in complete_ids]
        required_ids = set(complete_ids)
    elif complete_ids != required_ids:
        missing = sorted(required_ids - complete_ids, key=int)
        raise ValueError(
            f"Expression cache is incomplete for {len(missing)} frozen test identities: {missing}. "
            "Use --complete-case for an availability-defined subset."
        )
    expressions = {"neutral": {subject_id: neutral_by_id[subject_id] for subject_id in required_ids}, **expressions}

    train_ids = [str(value) for value in base["train_ids"]]
    vertex_features_np, static = build_static_vertex_features(
        neutral_by_id,
        train_ids,
        use_subunit_features=bool(base["use_subunit_features"]),
    )
    assert_feature_template_package(base, static, "Base package")
    vertex_features = torch.as_tensor(vertex_features_np, dtype=torch.float32, device=device)
    center = torch.as_tensor(np.asarray(rbsr["center"]), dtype=torch.float32, device=device)
    landmarks = torch.as_tensor(
        next(iter(neutral_by_id.values()))["landmarks"], dtype=torch.long, device=device
    )
    projection_basis = None
    if rbsr.get("projection_basis") is not None:
        projection_basis = torch.as_tensor(
            rbsr["projection_basis"], dtype=torch.float32, device=device
        )

    summaries: dict[str, dict[str, dict[str, float]]] = {}
    gate_summaries: dict[str, dict[str, float]] = {}
    for expression in sorted(expressions):
        by_id = expressions[expression]
        missing = sorted(required_ids - set(by_id), key=lambda value: int(value))
        if missing:
            raise ValueError(f"{expression}: missing {len(missing)} frozen test identities: {missing}")

        print(f"Evaluating {expression} on {len(pairs)} frozen test pairs", flush=True)
        condition, source, _, controls, _, _, _ = pair_conditions(
            by_id,
            pairs,
            base["source_pca"],
            base["cond_mean"],
            base["cond_std"],
        )
        ridge = ridge_predict(condition, base["ridge_cond"]).astype(np.float32)
        cvae = build_cvae(base, vertex_features_np.shape[1], condition.shape[1], device)
        cvae_prediction = predict_field(
            cvae, vertex_features_np, condition, is_cvae=True
        ).astype(np.float32)
        cvae.to("cpu")
        del cvae
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        alpha = float(base["selected_alpha"])
        global_hybrid = ((1.0 - alpha) * ridge + alpha * cvae_prediction).astype(np.float32)
        gate_args = rbsr["args"]
        gate = SpatialRiskGate(
            int(rbsr["gate_vertex_dim"]),
            int(rbsr["gate_cond_dim"]),
            int(gate_args["hidden"]),
            alpha,
        ).to(device)
        gate.load_state_dict(rbsr["gate_state_dict"])
        gate.eval()
        rbsr_prediction, gate_values = predict_gate_batches(
            gate,
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
            args.batch_size,
        )
        gate.to("cpu")
        del gate

        methods = {
            "ridge": ridge,
            "cvae": cvae_prediction,
            "global_hybrid": global_hybrid,
            "rbsr": rbsr_prediction,
        }
        summaries[expression] = {}
        for method, prediction in methods.items():
            rows = metric_rows_for_method(by_id, pairs, prediction)
            for row in rows:
                row["expression"] = expression
                row["method"] = method
            write_csv(args.out / f"pair_metrics_{expression}_{method}.csv", rows)
            summaries[expression][method] = metric_summary(rows)
        gate_summaries[expression] = {
            "mean": float(np.mean(gate_values)),
            "p95": float(np.percentile(gate_values, 95)),
            "active_fraction_0p5": float(np.mean(gate_values >= 0.5)),
        }

    report = {
        "protocol": "expression_preserving_identity_transfer",
        "selection_boundary": (
            "All models and hyperparameters were frozen on neutral validation data before this "
            "one-shot cross-expression evaluation."
        ),
        "source_and_target_expression": "same unseen expression",
        "n_pairs_per_expression": len(pairs),
        "frozen_test_identity_count": len({value for pair in frozen_pairs for value in pair}),
        "evaluated_identity_count": len(required_ids),
        "excluded_identity_ids": sorted(
            {value for pair in frozen_pairs for value in pair} - required_ids, key=int
        ),
        "complete_case_selection": bool(args.complete_case),
        "expressions": sorted(expressions),
        "summaries": summaries,
        "gate_summaries": gate_summaries,
        "base_model_package_sha256": sha256_file(args.base_model_package),
        "rbsr_package_sha256": sha256_file(args.rbsr_package),
        "expression_manifest_sha256": sha256_file(args.expression_repo / "manifest.json"),
        "expression_source_manifest_sha256": expression_manifest.get("neutral_manifest_sha256"),
    }
    atomic_write_json(args.out / "cross_expression_evaluation.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
