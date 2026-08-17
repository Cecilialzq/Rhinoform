"""Validation-only hard projection of RB-SR residuals into Ridge's fold set."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from rhinoform.data import load_rows, ridge_predict
from rhinoform.repro import (
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
    sha256_json,
    valid_sha256_sidecar,
    validate_torch_artifact,
    write_sha256_sidecar,
)
from rhinoform.safe_fusion import project_residual_no_new_ridge_folds, signed_fold_indicator
from rhinoform.strict_protocol_patch import strict_metric_rows
from rhinoform.train import pair_conditions
from rhinoform.train_rbsr_gate import validate_gate_package_training_contract


METRICS = (
    "roi_rmse", "landmark_rmse", "dorsum_rmse", "tip_rmse", "edge_strain_p95",
    "normal_flip_pct", "abs_flip_pct", "missed_flip_pct", "target_flip_pct",
)


def parse_floats(raw: str) -> list[float]:
    values = sorted(set(float(value.strip()) for value in raw.split(",") if value.strip()))
    if not values:
        raise ValueError("At least one attenuation is required")
    return values


def config_label(attenuation: float) -> str:
    return f"attenuation_{attenuation:g}".replace(".", "p")


def mean_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    return {metric: float(np.mean([float(row[metric]) for row in rows])) for metric in METRICS}


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def report_relative(path: Path, report_root: Path) -> str:
    """Store artifact references portably across Drive mounts and local hosts."""
    return path.resolve().relative_to(report_root.resolve()).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-model-package", type=Path, required=True)
    parser.add_argument("--rbsr-package", type=Path, required=True)
    parser.add_argument("--source-validation-report", type=Path, required=True)
    parser.add_argument("--source-chunk-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--attenuations", default="0.5,0.75,0.9")
    parser.add_argument("--smoothing-steps", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=64)
    parser.add_argument("--uniform-steps", type=int, default=1001)
    args = parser.parse_args()
    attenuations = parse_floats(args.attenuations)
    configs = [
        {
            "label": config_label(value),
            "attenuation": value,
            "smoothing_steps": args.smoothing_steps,
            "max_iterations": args.max_iterations,
            "uniform_steps": args.uniform_steps,
        }
        for value in attenuations
    ]

    if not validate_torch_artifact(
        args.base_model_package,
        required_keys=("args", "cvae_state_dict", "feature_template_sha256", "ridge_cond"),
        repair_sidecar=False,
    ):
        raise RuntimeError("Base package or sidecar is invalid")
    if not validate_torch_artifact(
        args.rbsr_package,
        required_keys=(
            "gate_state_dict", "base_model_package_sha256", "best_validation",
            "deployment_status", "reconstruction_objective", "exact_handle_contract",
        ),
        repair_sidecar=False,
    ):
        raise RuntimeError("RB-SR package or sidecar is invalid")
    base_hash = sha256_file(args.base_model_package)
    gate_hash = sha256_file(args.rbsr_package)
    base = torch.load(args.base_model_package, map_location="cpu", weights_only=False)
    gate_package = torch.load(args.rbsr_package, map_location="cpu", weights_only=False)
    gate_training_contract = validate_gate_package_training_contract(gate_package, args.rbsr_package)
    if gate_package.get("base_model_package_sha256") != base_hash:
        raise ValueError("RB-SR package/base hash mismatch")
    if gate_package.get("projection_basis") is not None or gate_package.get("args", {}).get("projection") != "none":
        raise ValueError("Hard Ridge-fold projection requires the clean projection=none gate")

    source_report = json.loads(args.source_validation_report.read_text(encoding="utf-8"))
    if source_report.get("split") != "validation" or int(source_report.get("n_pairs", 0)) != 4830:
        raise ValueError("A complete 4,830-pair validation cache is required")
    if source_report.get("base_model_package_sha256") != base_hash or source_report.get("rbsr_package_sha256") != gate_hash:
        raise ValueError("Source validation report hash chain does not match")
    source_signature = str(source_report["chunking"]["chunk_signature"])
    if not args.source_chunk_root.is_dir():
        raise FileNotFoundError(args.source_chunk_root)

    pairs = [(str(a), str(b)) for a, b in base["val_pairs"]]
    required_ids = {value for pair in pairs for value in pair}
    _, by_id = load_rows(args.repo, allowed_ids=required_ids)
    template = next(iter(by_id.values()))
    faces = np.asarray(template["faces"], dtype=np.int64)
    handles = np.asarray(template["landmarks"], dtype=np.int64)
    labels = [str(config["label"]) for config in configs]
    signature = sha256_json(
        {
            "schema": "rbsr_hard_ridge_fold_projection_v1",
            "base_model_package_sha256": base_hash,
            "rbsr_package_sha256": gate_hash,
            "source_validation_report_sha256": sha256_file(args.source_validation_report),
            "source_chunk_signature": source_signature,
            "configs": configs,
            "pairs": pairs,
            "certificate": "projected_fold_set subset_of matched_ridge_fold_set per pair",
        }
    )
    args.out.mkdir(parents=True, exist_ok=True)
    chunk_root = args.out / "projection_chunks" / signature[:16]
    chunk_root.mkdir(parents=True, exist_ok=True)
    row_sets: dict[str, list[dict[str, object]]] = {label: [] for label in labels}
    ridge_rows: list[dict[str, object]] = []
    diagnostic_sets: dict[str, list[dict[str, object]]] = {label: [] for label in labels}
    completed_pairs = 0

    for source_array_path in sorted(args.source_chunk_root.glob("chunk_*.npz")):
        stem = source_array_path.stem
        source_meta_path = args.source_chunk_root / f"{stem}.json"
        if not valid_sha256_sidecar(source_array_path) or not source_meta_path.is_file():
            raise RuntimeError(f"Invalid source chunk: {source_array_path}")
        source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
        if source_meta.get("signature") != source_signature:
            raise ValueError(f"Source signature mismatch: {source_array_path}")
        start, stop = int(source_meta["start"]), int(source_meta["stop"])
        if start != completed_pairs:
            raise ValueError("Source chunks are not contiguous")
        metric_paths = {label: chunk_root / f"{stem}_{label}.csv" for label in labels}
        diagnostic_paths = {label: chunk_root / f"{stem}_{label}_projection.csv" for label in labels}
        ridge_path = chunk_root / f"{stem}_ridge.csv"
        meta_path = chunk_root / f"{stem}.json"
        output_paths = list(metric_paths.values()) + list(diagnostic_paths.values()) + [ridge_path]
        resumed = False
        if meta_path.is_file() and all(valid_sha256_sidecar(path) for path in output_paths):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            resumed = meta.get("signature") == signature and int(meta["start"]) == start and int(meta["stop"]) == stop
        if resumed:
            print(f"RIDGE-FOLD PROJECTION resume {stop}/{len(pairs)}: {stem}", flush=True)
            ridge_rows.extend(read_csv(ridge_path))
            for label in labels:
                row_sets[label].extend(read_csv(metric_paths[label]))
                diagnostic_sets[label].extend(read_csv(diagnostic_paths[label]))
            completed_pairs = stop
            continue

        print(f"RIDGE-FOLD PROJECTION live {start + 1}-{stop}/{len(pairs)}", flush=True)
        with np.load(source_array_path, allow_pickle=True) as saved:
            saved_pairs = [(str(a), str(b)) for a, b in saved["pairs"].tolist()]
            cvae_prediction = np.asarray(saved["cvae_pred"], dtype=np.float32)
            learned_gate = np.asarray(saved["gate"], dtype=np.float32)
        if saved_pairs != pairs[start:stop]:
            raise ValueError("Cached pairs do not match base validation pairs")
        condition, source, _, controls, _, _, _ = pair_conditions(
            by_id, saved_pairs, base["source_pca"], base["cond_mean"], base["cond_std"]
        )
        ridge = ridge_predict(condition, base["ridge_cond"]).astype(np.float32).reshape(len(saved_pairs), -1, 3)
        cvae = cvae_prediction.reshape(len(saved_pairs), -1, 3)
        ridge[:, handles] = controls.reshape(len(saved_pairs), -1, 3)
        gated_residual = learned_gate[..., None] * (cvae - ridge)
        gated_residual[:, handles] = 0.0
        ridge_chunk_rows = strict_metric_rows(by_id, saved_pairs, ridge.reshape(len(saved_pairs), -1))
        for offset, row in enumerate(ridge_chunk_rows):
            row["pair_index"] = float(start + offset)
        atomic_write_csv(ridge_path, ridge_chunk_rows)
        ridge_rows.extend(ridge_chunk_rows)

        for config in configs:
            label = str(config["label"])
            predictions = []
            diagnostics = []
            for local_index, (source_vertices, ridge_delta, residual) in enumerate(zip(source, ridge, gated_residual)):
                if float(config["attenuation"]) == 0.0:
                    # First-class zero residual: do not pass through numerical
                    # projection arithmetic.  The fallback is bitwise Ridge.
                    delta = ridge_delta.copy()
                    ridge_fold_count = int(
                        np.sum(signed_fold_indicator(source_vertices, source_vertices + ridge_delta, faces)[0] < 0.0)
                    )
                    diagnostic = {
                        "pair_index": float(start + local_index),
                        "source_id": saved_pairs[local_index][0],
                        "target_id": saved_pairs[local_index][1],
                        "status": "exact_zero_residual_ridge_identity",
                        "retention": 0.0,
                        "iterations": 0,
                        "ridge_fold_count": ridge_fold_count,
                        "projected_fold_count": ridge_fold_count,
                        "new_vs_ridge_fold_count": 0,
                    }
                else:
                    result = project_residual_no_new_ridge_folds(
                        source_vertices,
                        ridge_delta,
                        residual,
                        faces,
                        handles,
                        attenuation=float(config["attenuation"]),
                        smoothing_steps=int(config["smoothing_steps"]),
                        max_iterations=int(config["max_iterations"]),
                        uniform_steps=int(config["uniform_steps"]),
                    )
                    if not result.certified or result.new_vs_ridge_fold_count != 0:
                        raise AssertionError("Hard Ridge-fold certificate failed")
                    delta = result.delta
                    diagnostic = {
                        "pair_index": float(start + local_index),
                        "source_id": saved_pairs[local_index][0],
                        "target_id": saved_pairs[local_index][1],
                        "status": result.status,
                        "retention": result.retention,
                        "iterations": result.iterations,
                        "ridge_fold_count": result.ridge_fold_count,
                        "projected_fold_count": result.projected_fold_count,
                        "new_vs_ridge_fold_count": result.new_vs_ridge_fold_count,
                    }
                predictions.append(delta)
                diagnostics.append(diagnostic)
            rows = strict_metric_rows(by_id, saved_pairs, np.asarray(predictions).reshape(len(saved_pairs), -1))
            for offset, row in enumerate(rows):
                row["pair_index"] = float(start + offset)
                if float(row["normal_flip_pct"]) > float(ridge_chunk_rows[offset]["normal_flip_pct"]) + 1e-12:
                    raise AssertionError("Strict target-relative new flip exceeded Ridge despite fold-subset certificate")
                if float(config["attenuation"]) == 0.0 and any(
                    abs(float(row[metric]) - float(ridge_chunk_rows[offset][metric])) > 1e-12
                    for metric in METRICS
                ):
                    raise AssertionError("Exact zero-residual candidate does not reproduce Ridge metrics")
            atomic_write_csv(metric_paths[label], rows)
            atomic_write_csv(diagnostic_paths[label], diagnostics)
            row_sets[label].extend(rows)
            diagnostic_sets[label].extend(diagnostics)
        atomic_write_json(
            meta_path,
            {
                "signature": signature,
                "start": start,
                "stop": stop,
                "source_array_sha256": sha256_file(source_array_path),
                "outputs_sha256": {path.name: sha256_file(path) for path in output_paths},
            },
        )
        write_sha256_sidecar(meta_path)
        completed_pairs = stop
        print(f"RIDGE-FOLD PROJECTION persisted {stop}/{len(pairs)}", flush=True)

    if completed_pairs != len(pairs) or len(ridge_rows) != len(pairs):
        raise ValueError("Incomplete validation projection")
    pair_dir = args.out / "pair_metrics"
    pair_dir.mkdir(parents=True, exist_ok=True)
    ridge_path = pair_dir / "pair_metrics_ridge_validation.csv"
    atomic_write_csv(ridge_path, ridge_rows)
    ridge_summary = {
        "label": "ridge_anchor",
        **mean_metrics(ridge_rows),
        "pair_metrics": report_relative(ridge_path, args.out),
        "pair_metrics_sha256": sha256_file(ridge_path),
    }
    candidates = []
    for config in configs:
        label = str(config["label"])
        metric_path = pair_dir / f"pair_metrics_{label}_validation.csv"
        diagnostic_path = pair_dir / f"projection_diagnostics_{label}_validation.csv"
        atomic_write_csv(metric_path, row_sets[label])
        atomic_write_csv(diagnostic_path, diagnostic_sets[label])
        status_counts: dict[str, int] = {}
        for row in diagnostic_sets[label]:
            status_counts[str(row["status"])] = status_counts.get(str(row["status"]), 0) + 1
        candidates.append(
            {
                **config,
                **mean_metrics(row_sets[label]),
                "mean_retention": float(np.mean([float(row["retention"]) for row in diagnostic_sets[label]])),
                "mean_iterations": float(np.mean([float(row["iterations"]) for row in diagnostic_sets[label]])),
                "certificate_rate": float(np.mean([int(row["new_vs_ridge_fold_count"]) == 0 for row in diagnostic_sets[label]])),
                "status_counts": status_counts,
                "pair_metrics": report_relative(metric_path, args.out),
                "pair_metrics_sha256": sha256_file(metric_path),
                "projection_diagnostics": report_relative(diagnostic_path, args.out),
                "projection_diagnostics_sha256": sha256_file(diagnostic_path),
            }
        )
    zero_candidates = [row for row in candidates if float(row["attenuation"]) == 0.0]
    if len(zero_candidates) != 1:
        raise ValueError("Validation projection must contain exactly one explicit zero-residual Ridge fallback")
    zero_candidate = zero_candidates[0]
    zero_identity_passed = all(
        abs(float(zero_candidate[metric]) - float(ridge_summary[metric])) <= 1e-12
        for metric in METRICS
    ) and zero_candidate["status_counts"] == {"exact_zero_residual_ridge_identity": len(pairs)}
    if not zero_identity_passed:
        raise AssertionError("Zero-residual Ridge identity audit failed")
    summary_path = args.out / "rbsr_ridge_fold_projection_validation.csv"
    atomic_write_csv(summary_path, [{k: v for k, v in row.items() if k != "status_counts"} for row in candidates] + [ridge_summary])
    report = {
        "status": "VALIDATION_HARD_RIDGE_FOLD_PROJECTION_COMPLETE_NO_TEST_ACCESS",
        "split": "validation",
        "test_access": False,
        "n_pairs": len(pairs),
        "certificate": "For every pair, projected_fold_set is a subset of matched_ridge_fold_set after exact handles.",
        "base_model_package_sha256": base_hash,
        "rbsr_package_sha256": gate_hash,
        "gate_training_contract": gate_training_contract,
        "source_validation_report_sha256": sha256_file(args.source_validation_report),
        "projection_signature": signature,
        "implementation_hashes": {
            "projection_kernel": sha256_file(Path(__file__).resolve().parents[2] / "rhinoform/safe_fusion.py"),
            "strict_scorer": sha256_file(Path(__file__).resolve().parents[2] / "rhinoform/strict_protocol_patch.py"),
            "evaluation_wrapper": sha256_file(Path(__file__).resolve()),
        },
        "ridge_reference": ridge_summary,
        "zero_gate_ridge_identity": {
            "passed": True,
            "label": zero_candidate["label"],
            "attenuation": 0.0,
            "pair_metrics_sha256": zero_candidate["pair_metrics_sha256"],
            "identity_tolerance": 1e-12,
        },
        "candidates": candidates,
        "artifact_paths_relative_to": "this_report_directory",
        "summary_csv": report_relative(summary_path, args.out),
        "summary_csv_sha256": sha256_file(summary_path),
        "chunk_root": report_relative(chunk_root, args.out),
    }
    report_path = args.out / "rbsr_ridge_fold_projection_validation.json"
    atomic_write_json(report_path, report)
    write_sha256_sidecar(report_path)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
