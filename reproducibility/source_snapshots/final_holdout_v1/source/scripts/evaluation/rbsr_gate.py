from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from rhinoform.confirmation import validate_all_models_freeze
from rhinoform.data import load_rows, ridge_predict
from rhinoform.rbsr_calibration import (
    fuse_calibrated_residual,
    validate_projection_test_unlock,
    validate_test_unlock,
)
from rhinoform.repro import (
    atomic_savez_compressed,
    atomic_write_json,
    sha256_file,
    sha256_json,
    valid_sha256_sidecar,
    validate_torch_artifact,
    write_sha256_sidecar,
)
from rhinoform.safe_fusion import project_residual_no_new_ridge_folds
from rhinoform.stats import write_csv
from rhinoform.strict_protocol_patch import strict_metric_rows as metric_rows_for_method
from rhinoform.train import NeuralFieldCVAE, assert_feature_template_package, build_static_vertex_features, pair_conditions, predict_field
from rhinoform.train_rbsr_gate import (
    SpatialRiskGate,
    predict_gate_batches,
    resolve_device,
    validate_gate_package_training_contract,
)


def metric_summary(rows: list[dict[str, float | str]]) -> dict[str, float]:
    names = [
        "roi_rmse", "landmark_rmse", "dorsum_rmse", "tip_rmse", "edge_strain_p95",
        "normal_flip_pct", "abs_flip_pct", "missed_flip_pct", "target_flip_pct",
    ]
    return {name: float(np.mean([float(row[name]) for row in rows])) for name in names}


def report_relative(path: Path, report_root: Path) -> str:
    return path.resolve().relative_to(report_root.resolve()).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a validation-selected RBSR gate checkpoint.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base-model-package", required=True)
    parser.add_argument("--rbsr-package", required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dense", action="store_true")
    parser.add_argument("--chunk-pairs", type=int, default=320)
    parser.add_argument(
        "--smoke-max-pairs",
        type=int,
        default=0,
        help="Validation-only implementation smoke limiter; any nonzero run is non-reportable.",
    )
    parser.add_argument(
        "--calibration-freeze",
        default="",
        help=(
            "Strict validation calibration freeze. For test evaluation it must prove a "
            "Ridge-beating validation point and match the exact base/gate hashes."
        ),
    )
    parser.add_argument(
        "--projection-freeze",
        default="",
        help="Validation-selected hard Ridge-fold projection freeze for certified inference.",
    )
    parser.add_argument(
        "--confirmation-policy",
        default="",
        help="Frozen internal final-rerun holdout policy; required by the one-shot protocol.",
    )
    parser.add_argument(
        "--test-access-receipt",
        default="",
        help="Atomic one-shot/resume receipt written before any confirmation test mesh is loaded.",
    )
    parser.add_argument(
        "--all-models-freeze",
        default="",
        help="Exact pre-test freeze for every matched RB-SR/LAMM model artifact.",
    )
    parser.add_argument(
        "--save-base-comparators",
        action="store_true",
        help="Also strict-score the clean Ridge, CVAE and validation-selected Hybrid predictions.",
    )
    args = parser.parse_args()
    if args.chunk_pairs < 1:
        raise ValueError("--chunk-pairs must be positive")
    if args.smoke_max_pairs < 0:
        raise ValueError("--smoke-max-pairs cannot be negative")
    if args.smoke_max_pairs and args.split != "validation":
        raise ValueError("--smoke-max-pairs is forbidden for test evaluation")
    if args.calibration_freeze and args.projection_freeze:
        raise ValueError("Choose calibration or certified hard projection, not both")

    device = resolve_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.split} data and packages on {device}", flush=True)
    if not validate_torch_artifact(
        Path(args.base_model_package),
        required_keys=("args", "cvae_state_dict", "feature_template_sha256", "ridge_cond"),
    ):
        raise RuntimeError("Base package or SHA-256 sidecar is invalid")
    if not validate_torch_artifact(
        Path(args.rbsr_package),
        required_keys=(
            "gate_state_dict", "base_model_package_sha256", "best_validation",
            "deployment_status", "reconstruction_objective", "exact_handle_contract",
        ),
    ):
        raise RuntimeError("RB-SR package or SHA-256 sidecar is invalid")
    base = torch.load(args.base_model_package, map_location="cpu", weights_only=False)
    rbsr = torch.load(args.rbsr_package, map_location="cpu", weights_only=False)
    if rbsr.get("base_model_package_sha256") != sha256_file(Path(args.base_model_package)):
        raise ValueError("RB-SR package is not chained to the supplied base package")
    gate_training_contract = validate_gate_package_training_contract(rbsr, Path(args.rbsr_package))
    base_hash = sha256_file(Path(args.base_model_package))
    rbsr_hash = sha256_file(Path(args.rbsr_package))
    calibration_freeze_path: Path | None = None
    calibration_freeze_hash: str | None = None
    calibration_selected: dict[str, object] | None = None
    projection_freeze_path: Path | None = None
    projection_freeze_hash: str | None = None
    projection_selected: dict[str, object] | None = None
    confirmation_policy_path: Path | None = None
    confirmation_policy_hash: str | None = None
    all_models_freeze_path: Path | None = None
    all_models_freeze_hash: str | None = None
    access_receipt_path: Path | None = None
    if args.calibration_freeze:
        calibration_freeze_path = Path(args.calibration_freeze)
        if not valid_sha256_sidecar(calibration_freeze_path):
            raise RuntimeError("Calibration freeze or SHA-256 sidecar is invalid")
        calibration_freeze = json.loads(calibration_freeze_path.read_text(encoding="utf-8"))
        calibration_selected = validate_test_unlock(
            calibration_freeze,
            base_hash=base_hash,
            rbsr_hash=rbsr_hash,
        )
        calibration_freeze_hash = sha256_file(calibration_freeze_path)
        if rbsr.get("projection_basis") is not None or rbsr.get("args", {}).get("projection") != "none":
            raise ValueError("Strict calibrated inference requires a projection=none gate package")
    elif args.split == "test" and not args.projection_freeze:
        raise ValueError(
            "Fail closed: test inference requires a validation-frozen calibration or certified hard projection"
        )
    if args.projection_freeze:
        projection_freeze_path = Path(args.projection_freeze)
        if not valid_sha256_sidecar(projection_freeze_path):
            raise RuntimeError("Projection freeze or SHA-256 sidecar is invalid")
        projection_freeze = json.loads(projection_freeze_path.read_text(encoding="utf-8"))
        projection_selected = validate_projection_test_unlock(
            projection_freeze,
            base_hash=base_hash,
            rbsr_hash=rbsr_hash,
            projection_signature=str(projection_freeze.get("projection_signature", "")),
        )
        projection_freeze_hash = sha256_file(projection_freeze_path)
        implementation_hashes = dict(projection_freeze.get("implementation_hashes", {}))
        implementation_root = Path(__file__).resolve().parents[2]
        if implementation_hashes.get("projection_kernel") != sha256_file(
            implementation_root / "rhinoform/safe_fusion.py"
        ):
            raise ValueError("Projection kernel changed after validation freeze")
        if implementation_hashes.get("strict_scorer") != sha256_file(
            implementation_root / "rhinoform/strict_protocol_patch.py"
        ):
            raise ValueError("Strict scorer changed after validation freeze")
        if rbsr.get("projection_basis") is not None or rbsr.get("args", {}).get("projection") != "none":
            raise ValueError("Certified hard projection requires a projection=none gate package")

    if args.confirmation_policy:
        if args.split != "test" or projection_selected is None:
            raise ValueError("Confirmation policy is valid only for certified one-shot test inference")
        confirmation_policy_path = Path(args.confirmation_policy)
        if not valid_sha256_sidecar(confirmation_policy_path):
            raise RuntimeError("Confirmation policy or SHA-256 sidecar is invalid")
        policy = json.loads(confirmation_policy_path.read_text(encoding="utf-8"))
        if policy.get("status") != "FROZEN_INTERNAL_FINAL_RERUN_HOLDOUT_POLICY":
            raise ValueError("Confirmation policy is not frozen")
        if policy.get("test_access") is not False or int(policy.get("test_access_count", -1)) != 0:
            raise ValueError("Confirmation policy does not describe a locked final-rerun holdout")
        if base.get("split_manifest_sha256") != policy.get("split_manifest_sha256"):
            raise ValueError("Base package/confirmation split hash mismatch")
        if base.get("train_pair_manifest_sha256") != policy.get("train_pair_manifest_sha256"):
            raise ValueError("Base package/confirmation train-pair hash mismatch")
        frozen_config = dict(policy["frozen_rbsr_configuration"])
        checks = {
            "source_pca_dim": int(base["args"]["source_pca_dim"]),
            "delta_pca_dim": int(base["args"]["delta_pca_dim"]),
            "ridge_lambda": float(base["args"]["ridge_lambda"]),
            "hard_projection_attenuation": float(projection_selected["attenuation"]),
            "hard_projection_smoothing_steps": int(projection_selected["smoothing_steps"]),
            "hard_projection_max_iterations": int(projection_selected["max_iterations"]),
            "hard_projection_uniform_steps": int(projection_selected["uniform_steps"]),
            "gate_projection": str(rbsr["args"]["projection"]),
            "gate_reconstruction_objective": str(rbsr["reconstruction_objective"]),
            "gate_exact_handle_contract": str(rbsr["exact_handle_contract"]),
        }
        for key, actual in checks.items():
            if actual != frozen_config.get(key):
                raise ValueError(f"Confirmation frozen configuration mismatch: {key}")
        confirmation_policy_hash = sha256_file(confirmation_policy_path)
        if not args.all_models_freeze:
            raise ValueError("Confirmation test requires --all-models-freeze")
        all_models_freeze_path = Path(args.all_models_freeze)
        if not valid_sha256_sidecar(all_models_freeze_path):
            raise RuntimeError("All-model freeze or SHA-256 sidecar is invalid")
        all_models_freeze = json.loads(all_models_freeze_path.read_text(encoding="utf-8"))
        validate_all_models_freeze(
            all_models_freeze,
            expected={
                "confirmation_policy_sha256": confirmation_policy_hash,
                "split_manifest_sha256": str(policy["split_manifest_sha256"]),
                "rbsr_base_sha256": base_hash,
                "rbsr_gate_sha256": rbsr_hash,
                "rbsr_projection_freeze_sha256": str(projection_freeze_hash),
            },
            expected_implementations={
                "rhinoform/confirmation.py": sha256_file(
                    implementation_root / "rhinoform/confirmation.py"
                ),
                "rhinoform/safe_fusion.py": sha256_file(
                    implementation_root / "rhinoform/safe_fusion.py"
                ),
                "rhinoform/strict_protocol_patch.py": sha256_file(
                    implementation_root / "rhinoform/strict_protocol_patch.py"
                ),
                "rhinoform/train_rbsr_gate.py": sha256_file(
                    implementation_root / "rhinoform/train_rbsr_gate.py"
                ),
                "rhinoform/rbsr_calibration.py": sha256_file(
                    implementation_root / "rhinoform/rbsr_calibration.py"
                ),
                "scripts/evaluation/rbsr_gate.py": sha256_file(Path(__file__)),
            },
        )
        all_models_freeze_hash = sha256_file(all_models_freeze_path)
        if not args.test_access_receipt:
            raise ValueError("Confirmation test requires --test-access-receipt")
        access_receipt_path = Path(args.test_access_receipt)
        receipt_chain = {
            "confirmation_policy_sha256": confirmation_policy_hash,
            "base_model_package_sha256": base_hash,
            "rbsr_package_sha256": rbsr_hash,
            "projection_freeze_sha256": projection_freeze_hash,
            "all_models_freeze_sha256": all_models_freeze_hash,
        }
        if access_receipt_path.is_file():
            if not valid_sha256_sidecar(access_receipt_path):
                raise RuntimeError("Existing test-access receipt or sidecar is invalid")
            receipt = json.loads(access_receipt_path.read_text(encoding="utf-8"))
            if any(receipt.get(key) != value for key, value in receipt_chain.items()):
                raise ValueError("Test-access receipt belongs to a different frozen hash chain")
            if receipt.get("status") not in {"IN_PROGRESS_RESUMABLE", "COMPLETE"}:
                raise ValueError("Unrecognised test-access receipt status")
            print(f"Confirmation test exact-hash resume: {receipt['status']}", flush=True)
        else:
            atomic_write_json(
                access_receipt_path,
                {
                    "status": "IN_PROGRESS_RESUMABLE",
                    "test_access_count": 1,
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    **receipt_chain,
                },
            )
            write_sha256_sidecar(access_receipt_path)
            print("Confirmation test access receipt persisted before loading test meshes", flush=True)
    pairs_key = "val_pairs" if args.split == "validation" else "test_pairs"
    pairs = [(str(a), str(b)) for a, b in base[pairs_key]]
    if args.smoke_max_pairs:
        pairs = pairs[: args.smoke_max_pairs]
        print(f"NON-REPORTABLE validation smoke subset: {len(pairs)} pairs", flush=True)
    train_ids = [str(value) for value in base["train_ids"]]
    required_ids = set(train_ids) | {value for pair in pairs for value in pair}
    _, by_id = load_rows(Path(args.repo), allowed_ids=required_ids)
    vertex_features_np, static = build_static_vertex_features(
        by_id, train_ids, use_subunit_features=bool(base["use_subunit_features"])
    )
    assert_feature_template_package(base, static, "Base package")
    if rbsr.get("feature_template_schema") != base.get("feature_template_schema"):
        raise ValueError("RB-SR gate/base feature-template schemas differ")
    if rbsr.get("feature_template_sha256") != base.get("feature_template_sha256"):
        raise ValueError("RB-SR gate/base feature-template hashes differ")
    condition, source, target, controls, _, _, _ = pair_conditions(
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
    rbsr_args = rbsr["args"]
    gate = SpatialRiskGate(
        int(rbsr["gate_vertex_dim"]),
        int(rbsr["gate_cond_dim"]),
        int(rbsr_args["hidden"]),
        float(base["selected_alpha"]),
    ).to(device)
    gate.load_state_dict(rbsr["gate_state_dict"])
    vertex_features = torch.as_tensor(vertex_features_np, dtype=torch.float32, device=device)
    center = torch.as_tensor(np.asarray(rbsr["center"]), dtype=torch.float32, device=device)
    template = next(iter(by_id.values()))
    faces_np = np.asarray(template["faces"], dtype=np.int64)
    handles_np = np.asarray(template["landmarks"], dtype=np.int64)
    landmarks = torch.as_tensor(handles_np, dtype=torch.long, device=device)
    projection_basis = None
    if rbsr.get("projection_basis") is not None:
        projection_basis = torch.as_tensor(rbsr["projection_basis"], dtype=torch.float32, device=device)
    chunk_signature = sha256_json({
        "schema": "rbsr_strict_chunks_v1",
        "split": args.split,
        "base_model_package_sha256": base_hash,
        "rbsr_package_sha256": rbsr_hash,
        "gate_training_contract": gate_training_contract,
        "calibration_freeze_sha256": calibration_freeze_hash,
        "calibration_selected": calibration_selected,
        "projection_freeze_sha256": projection_freeze_hash,
        "projection_selected": projection_selected,
        "confirmation_policy_sha256": confirmation_policy_hash,
        "all_models_freeze_sha256": all_models_freeze_hash,
        "pairs": pairs,
        "save_base_comparators": bool(args.save_base_comparators),
    })
    chunk_root = out_dir / f"{args.split}_chunks" / chunk_signature[:16]
    chunk_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    comparator_row_sets: dict[str, list[dict[str, float | str]]] = {
        "ridge_sourcepca_clean": [],
        "cvae_clean": [],
        "hybrid_validation_selected_clean": [],
    } if args.save_base_comparators else {}
    prediction_parts: list[np.ndarray] = []
    gate_parts: list[np.ndarray] = []
    cvae_parts: list[np.ndarray] = []
    hybrid_parts: list[np.ndarray] = []
    projection_diagnostics: list[dict[str, float | str]] = []

    for start in range(0, len(pairs), args.chunk_pairs):
        stop = min(start + args.chunk_pairs, len(pairs))
        stem = f"chunk_{start:05d}_{stop:05d}"
        array_path = chunk_root / f"{stem}.npz"
        metric_path = chunk_root / f"{stem}_rbsr.csv"
        meta_path = chunk_root / f"{stem}.json"
        projection_diagnostic_path = chunk_root / f"{stem}_projection.csv"
        comparator_paths = {
            label: chunk_root / f"{stem}_{label}.csv"
            for label in comparator_row_sets
        }
        resumed = False
        if meta_path.is_file() and valid_sha256_sidecar(array_path) and valid_sha256_sidecar(metric_path):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                resumed = (
                    meta.get("signature") == chunk_signature
                    and int(meta.get("start", -1)) == start
                    and int(meta.get("stop", -1)) == stop
                    and all(valid_sha256_sidecar(path) for path in comparator_paths.values())
                    and (
                        projection_selected is None
                        or valid_sha256_sidecar(projection_diagnostic_path)
                    )
                )
            except (OSError, ValueError, json.JSONDecodeError):
                resumed = False
        if resumed:
            print(f"RBSR STRICT {args.split} resume from Drive {stop}/{len(pairs)}: {stem}", flush=True)
            with np.load(array_path, allow_pickle=True) as saved:
                saved_pairs = [(str(a), str(b)) for a, b in saved["pairs"].tolist()]
                if saved_pairs != pairs[start:stop]:
                    raise ValueError(f"Chunk pair mismatch: {array_path}")
                cvae_chunk = np.asarray(saved["cvae_pred"], dtype=np.float32)
                hybrid_chunk = np.asarray(saved["global_hybrid_pred"], dtype=np.float32)
                prediction_chunk = np.asarray(saved["rbsr_pred"], dtype=np.float32)
                gate_chunk = np.asarray(saved["gate"], dtype=np.float32)
            with metric_path.open(newline="", encoding="utf-8") as handle:
                chunk_rows = list(csv.DictReader(handle))
            if projection_selected is not None:
                with projection_diagnostic_path.open(newline="", encoding="utf-8") as handle:
                    projection_diagnostics.extend(list(csv.DictReader(handle)))
            for label, path in comparator_paths.items():
                with path.open(newline="", encoding="utf-8") as handle:
                    comparator_row_sets[label].extend(list(csv.DictReader(handle)))
        else:
            print(f"RBSR STRICT {args.split} live {start + 1}-{stop}/{len(pairs)}", flush=True)
            cvae_chunk = predict_field(
                cvae, vertex_features_np, condition[start:stop], is_cvae=True
            ).astype(np.float32)
            prediction_chunk, gate_chunk = predict_gate_batches(
                gate,
                vertex_features,
                condition[start:stop],
                source[start:stop],
                ridge[start:stop],
                cvae_chunk,
                center,
                float(rbsr["scale"]),
                landmarks,
                controls[start:stop],
                projection_basis,
                args.batch_size,
            )
            if calibration_selected is not None:
                prediction_chunk, gate_chunk = fuse_calibrated_residual(
                    ridge[start:stop],
                    cvae_chunk,
                    gate_chunk,
                    logit_offset=float(calibration_selected.get("logit_offset") or 0.0),
                    force_zero_gate=bool(calibration_selected.get("force_zero_gate", False)),
                )
            if projection_selected is not None:
                ridge_fixed = ridge[start:stop].reshape(stop - start, -1, 3).copy()
                ridge_fixed[:, handles_np] = controls[start:stop].reshape(stop - start, -1, 3)
                cvae_vertices = cvae_chunk.reshape(stop - start, -1, 3)
                gated_residual = gate_chunk[..., None] * (cvae_vertices - ridge_fixed)
                gated_residual[:, handles_np] = 0.0
                projected = []
                chunk_projection_diagnostics: list[dict[str, float | str]] = []
                for local_index, (source_vertices, ridge_delta, residual) in enumerate(
                    zip(source[start:stop], ridge_fixed, gated_residual)
                ):
                    result = project_residual_no_new_ridge_folds(
                        source_vertices,
                        ridge_delta,
                        residual,
                        faces_np,
                        handles_np,
                        attenuation=float(projection_selected["attenuation"]),
                        smoothing_steps=int(projection_selected["smoothing_steps"]),
                        max_iterations=int(projection_selected["max_iterations"]),
                        uniform_steps=int(projection_selected["uniform_steps"]),
                    )
                    if not result.certified or result.new_vs_ridge_fold_count != 0:
                        raise AssertionError("Certified Ridge-fold projection failed")
                    projected.append(result.delta)
                    chunk_projection_diagnostics.append(
                        {
                            "pair_index": float(start + local_index),
                            "source_id": pairs[start + local_index][0],
                            "target_id": pairs[start + local_index][1],
                            "status": result.status,
                            "retention": result.retention,
                            "iterations": result.iterations,
                            "ridge_fold_count": result.ridge_fold_count,
                            "projected_fold_count": result.projected_fold_count,
                            "new_vs_ridge_fold_count": result.new_vs_ridge_fold_count,
                        }
                    )
                prediction_chunk = np.asarray(projected, dtype=np.float32).reshape(stop - start, -1)
                write_csv(projection_diagnostic_path, chunk_projection_diagnostics)
                projection_diagnostics.extend(chunk_projection_diagnostics)
            hybrid_chunk = ridge[start:stop] + float(base["selected_alpha"]) * (
                cvae_chunk - ridge[start:stop]
            )
            chunk_rows = metric_rows_for_method(by_id, pairs[start:stop], prediction_chunk)
            if projection_selected is not None:
                ridge_certificate_rows = metric_rows_for_method(
                    by_id, pairs[start:stop], ridge_fixed.reshape(stop - start, -1)
                )
                for candidate_row, ridge_row in zip(chunk_rows, ridge_certificate_rows):
                    if float(candidate_row["normal_flip_pct"]) > float(ridge_row["normal_flip_pct"]) + 1e-12:
                        raise AssertionError("Certified test new flip exceeded matched Ridge")
            for offset, row in enumerate(chunk_rows):
                row["pair_index"] = float(start + offset)
            write_csv(metric_path, chunk_rows)
            if args.save_base_comparators:
                comparator_predictions = {
                    "ridge_sourcepca_clean": ridge[start:stop],
                    "cvae_clean": cvae_chunk,
                    "hybrid_validation_selected_clean": hybrid_chunk,
                }
                for label, values in comparator_predictions.items():
                    comparator_rows = metric_rows_for_method(by_id, pairs[start:stop], values)
                    for offset, row in enumerate(comparator_rows):
                        row["pair_index"] = float(start + offset)
                    write_csv(comparator_paths[label], comparator_rows)
                    comparator_row_sets[label].extend(comparator_rows)
            atomic_savez_compressed(
                array_path,
                pairs=np.asarray(pairs[start:stop], dtype=object),
                cvae_pred=cvae_chunk,
                global_hybrid_pred=hybrid_chunk.astype(np.float32),
                rbsr_pred=prediction_chunk.astype(np.float32),
                gate=gate_chunk.astype(np.float32),
            )
            atomic_write_json(meta_path, {
                "signature": chunk_signature,
                "start": start,
                "stop": stop,
                "array_sha256": sha256_file(array_path),
                "rbsr_metrics_sha256": sha256_file(metric_path),
                "comparator_sha256": {
                    label: sha256_file(path) for label, path in comparator_paths.items()
                },
                "projection_diagnostics_sha256": (
                    None
                    if projection_selected is None
                    else sha256_file(projection_diagnostic_path)
                ),
            })
            print(f"RBSR STRICT {args.split} persisted to Drive {stop}/{len(pairs)}", flush=True)
        if len(chunk_rows) != stop - start:
            raise ValueError(f"Metric row count mismatch for {stem}")
        rows.extend(chunk_rows)
        if args.save_dense:
            cvae_parts.append(cvae_chunk)
            hybrid_parts.append(hybrid_chunk)
            prediction_parts.append(prediction_chunk)
        gate_parts.append(gate_chunk)

    gate_values = np.concatenate(gate_parts, axis=0)
    cvae.to("cpu")
    gate.to("cpu")
    del cvae, gate
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    write_csv(out_dir / f"pair_metrics_rbsr_{args.split}.csv", rows)
    summary = metric_summary(rows)
    comparator_reports: dict[str, dict[str, object]] = {}
    if args.save_base_comparators:
        for label, comparator_rows in comparator_row_sets.items():
            comparator_path = out_dir / f"pair_metrics_{label}_{args.split}.csv"
            write_csv(comparator_path, comparator_rows)
            comparator_reports[label] = {
                "summary": metric_summary(comparator_rows),
                "pair_metrics": report_relative(comparator_path, out_dir),
                "pair_metrics_sha256": sha256_file(comparator_path),
            }
    vertex_gate_mean = np.mean(gate_values, axis=0).astype(np.float32)
    vertex_gate_std = np.std(gate_values, axis=0).astype(np.float32)
    atomic_savez_compressed(
        out_dir / f"gate_map_{args.split}.npz",
        vertex_gate_mean=vertex_gate_mean,
        vertex_gate_std=vertex_gate_std,
        pair_gate_mean=np.mean(gate_values, axis=1).astype(np.float32),
        pairs=np.asarray(pairs, dtype=object),
    )
    dense_path = None
    if args.save_dense:
        cvae_prediction = np.concatenate(cvae_parts, axis=0)
        global_hybrid = np.concatenate(hybrid_parts, axis=0)
        prediction = np.concatenate(prediction_parts, axis=0)
        dense_path = out_dir / f"rbsr_predictions_{args.split}.npz"
        atomic_savez_compressed(
            dense_path,
            pairs=np.asarray(pairs, dtype=object),
            ridge_pred=ridge.astype(np.float32),
            cvae_pred=cvae_prediction.astype(np.float32),
            global_hybrid_pred=global_hybrid.astype(np.float32),
            rbsr_pred=prediction.astype(np.float32),
            gate=gate_values.astype(np.float32),
            selected_alpha=np.asarray([float(base["selected_alpha"])], dtype=np.float32),
        )
    projection_diagnostics_path = None
    projection_certificate = None
    if projection_selected is not None:
        if len(projection_diagnostics) != len(pairs):
            raise ValueError("Incomplete certified projection diagnostics")
        projection_diagnostics_path = out_dir / f"projection_diagnostics_{args.split}.csv"
        write_csv(projection_diagnostics_path, projection_diagnostics)
        projection_certificate = {
            "certificate_rate": float(
                np.mean([
                    int(float(row["new_vs_ridge_fold_count"])) == 0
                    for row in projection_diagnostics
                ])
            ),
            "mean_retention": float(np.mean([float(row["retention"]) for row in projection_diagnostics])),
            "mean_iterations": float(np.mean([float(row["iterations"]) for row in projection_diagnostics])),
            "diagnostics": report_relative(projection_diagnostics_path, out_dir),
            "diagnostics_sha256": sha256_file(projection_diagnostics_path),
        }
        if projection_certificate["certificate_rate"] != 1.0:
            raise AssertionError("Final hard-projection certificate rate is not exactly one")
    report = {
        "method": "rbsr_gate_prototype",
        "split": args.split,
        "selection_status": (
            "non_reportable_validation_implementation_smoke"
            if args.smoke_max_pairs
            else (
            "validation_only_checkpoint_assessment"
            if args.split == "validation"
            else (
                "one_shot_test_after_frozen_certified_ridge_fold_projection"
                if projection_selected is not None
                else (
                    "one_shot_test_after_strict_ridge_beating_validation_calibration"
                    if calibration_selected is not None
                    else "historical_uncalibrated_one_shot_test"
                )
            )
            )
        ),
        "n_pairs": len(pairs),
        "summary": summary,
        "gate": {
            "mean": float(np.mean(gate_values)),
            "p95": float(np.percentile(gate_values, 95)),
            "active_fraction_0p5": float(np.mean(gate_values >= 0.5)),
            "validation_selected_checkpoint": rbsr["best_validation"],
        },
        "base_model_package_sha256": base_hash,
        "rbsr_package_sha256": rbsr_hash,
        "calibration_freeze": None if calibration_freeze_path is None else str(calibration_freeze_path),
        "calibration_freeze_sha256": calibration_freeze_hash,
        "calibration_selected": calibration_selected,
        "projection_freeze": (
            None
            if projection_freeze_path is None
            else projection_freeze_path.name
        ),
        "projection_freeze_sha256": projection_freeze_hash,
        "projection_selected": projection_selected,
        "projection_certificate": projection_certificate,
        "confirmation_policy": (
            None
            if confirmation_policy_path is None
            else confirmation_policy_path.name
        ),
        "confirmation_policy_sha256": confirmation_policy_hash,
        "all_models_freeze": (
            None if all_models_freeze_path is None else all_models_freeze_path.name
        ),
        "all_models_freeze_sha256": all_models_freeze_hash,
        "chunking": {
            "chunk_pairs": args.chunk_pairs,
            "chunk_signature": chunk_signature,
            "chunk_root": report_relative(chunk_root, out_dir),
            "completed_chunks": len(range(0, len(pairs), args.chunk_pairs)),
            "completed_pairs": len(rows),
        },
        "artifact_paths_relative_to": "this_report_directory",
        "pair_metrics": report_relative(out_dir / f"pair_metrics_rbsr_{args.split}.csv", out_dir),
        "pair_metrics_sha256": sha256_file(out_dir / f"pair_metrics_rbsr_{args.split}.csv"),
        "gate_map": report_relative(out_dir / f"gate_map_{args.split}.npz", out_dir),
        "gate_map_sha256": sha256_file(out_dir / f"gate_map_{args.split}.npz"),
        "dense_predictions": None if dense_path is None else report_relative(dense_path, out_dir),
        "dense_predictions_sha256": None if dense_path is None else sha256_file(dense_path),
        "clean_base_comparators": comparator_reports,
    }
    report_path = out_dir / f"rbsr_evaluation_{args.split}.json"
    atomic_write_json(report_path, report)
    write_sha256_sidecar(report_path)
    if access_receipt_path is not None:
        receipt = json.loads(access_receipt_path.read_text(encoding="utf-8"))
        receipt.update(
            {
                "status": "COMPLETE",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "test_access_count": 1,
                "test_report": report_path.name,
                "test_report_sha256": sha256_file(report_path),
            }
        )
        atomic_write_json(access_receipt_path, receipt)
        write_sha256_sidecar(access_receipt_path)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
