"""Validation-only logit calibration for a frozen projection-free RB-SR gate.

This command cannot open the test split. It reuses the frozen CVAE prediction
and learned gate arrays from a completed strict validation evaluation, then
scores ``sigmoid(logit(g) - beta)`` candidates plus an exact zero-gate Ridge
identity. All chunks and final pair-level tables are atomically persisted.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from rhinoform.data import load_rows, ridge_predict
from rhinoform.rbsr_calibration import enforce_exact_controls, fuse_calibrated_residual
from rhinoform.repro import (
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
    sha256_json,
    valid_sha256_sidecar,
    validate_torch_artifact,
    write_sha256_sidecar,
)
from rhinoform.strict_protocol_patch import strict_metric_rows
from rhinoform.train import pair_conditions


METRIC_NAMES = (
    "roi_rmse",
    "landmark_rmse",
    "dorsum_rmse",
    "tip_rmse",
    "edge_strain_p95",
    "normal_flip_pct",
    "abs_flip_pct",
    "missed_flip_pct",
    "target_flip_pct",
)


def parse_offsets(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("Logit offsets must be a non-empty list of finite non-negative values")
    return sorted(set(values))


def offset_label(value: float) -> str:
    token = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return f"beta_{token}"


def metric_summary(rows: list[dict[str, object]]) -> dict[str, float]:
    return {
        name: float(np.mean([float(row[name]) for row in rows]))
        for name in METRIC_NAMES
    }


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def resolve_source_chunk_root(report: dict[str, object], override: str) -> Path:
    if override:
        return Path(override)
    chunking = report.get("chunking")
    if not isinstance(chunking, dict) or not chunking.get("chunk_root"):
        raise ValueError("Source validation report has no chunk_root")
    return Path(str(chunking["chunk_root"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-model-package", type=Path, required=True)
    parser.add_argument("--rbsr-package", type=Path, required=True)
    parser.add_argument("--source-validation-report", type=Path, required=True)
    parser.add_argument("--source-chunk-root", default="")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--logit-offsets",
        default="0,0.5,1,1.5,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.5,5,6,8",
    )
    args = parser.parse_args()
    offsets = parse_offsets(args.logit_offsets)
    labels = [offset_label(value) for value in offsets] + ["zero_gate"]

    if not validate_torch_artifact(
        args.base_model_package,
        required_keys=("args", "feature_template_sha256", "ridge_cond"),
        repair_sidecar=False,
    ):
        raise RuntimeError("Base model package or sidecar is invalid")
    if not validate_torch_artifact(
        args.rbsr_package,
        required_keys=("gate_state_dict", "base_model_package_sha256", "best_validation"),
        repair_sidecar=False,
    ):
        raise RuntimeError("RB-SR package or sidecar is invalid")
    base_hash = sha256_file(args.base_model_package)
    gate_hash = sha256_file(args.rbsr_package)
    base = torch.load(args.base_model_package, map_location="cpu", weights_only=False)
    gate_package = torch.load(args.rbsr_package, map_location="cpu", weights_only=False)
    if gate_package.get("base_model_package_sha256") != base_hash:
        raise ValueError("RB-SR package is not chained to the supplied base package")
    if gate_package.get("projection_basis") is not None:
        raise ValueError("Strict calibration requires projection=none so zero gate is the Ridge identity")
    if gate_package.get("args", {}).get("projection") != "none":
        raise ValueError("RB-SR package metadata does not declare projection=none")

    source_report = json.loads(args.source_validation_report.read_text(encoding="utf-8"))
    if source_report.get("split") != "validation" or int(source_report.get("n_pairs", 0)) != 4830:
        raise ValueError("Calibration requires a complete 4,830-pair validation evaluation")
    if source_report.get("base_model_package_sha256") != base_hash:
        raise ValueError("Source validation/base package hash mismatch")
    if source_report.get("rbsr_package_sha256") != gate_hash:
        raise ValueError("Source validation/gate package hash mismatch")
    source_chunk_root = resolve_source_chunk_root(source_report, args.source_chunk_root)
    if not source_chunk_root.is_dir():
        raise FileNotFoundError(source_chunk_root)
    source_signature = str(source_report["chunking"]["chunk_signature"])

    pairs = [(str(a), str(b)) for a, b in base["val_pairs"]]
    if len(pairs) != 4830:
        raise ValueError("Base package validation pair count is not 4,830")
    # Calibration derives the Ridge condition from validation source geometry;
    # it never rebuilds train-only vertex features or the CVAE. Loading the 70
    # validation identities is therefore sufficient and avoids staging all 676
    # training meshes again.
    required_ids = {value for pair in pairs for value in pair}
    _, by_id = load_rows(args.repo, allowed_ids=required_ids)
    landmark_indices = np.asarray(next(iter(by_id.values()))["landmarks"], dtype=np.int64)

    args.out.mkdir(parents=True, exist_ok=True)
    signature = sha256_json(
        {
            "schema": "rbsr_projection_none_logit_calibration_exact_controls_v2",
            "base_model_package_sha256": base_hash,
            "rbsr_package_sha256": gate_hash,
            "source_validation_report_sha256": sha256_file(args.source_validation_report),
            "source_chunk_signature": source_signature,
            "logit_offsets": offsets,
            "pairs": pairs,
            "strict_rule": (
                "target controls hard-overwritten after calibration; selection requires the "
                "external frozen reference-dominance rule"
            ),
        }
    )
    chunk_root = args.out / "calibration_chunks" / signature[:16]
    chunk_root.mkdir(parents=True, exist_ok=True)
    row_sets: dict[str, list[dict[str, object]]] = {label: [] for label in labels}
    gate_sums = {label: 0.0 for label in labels}
    gate_counts = {label: 0 for label in labels}
    gate_active = {label: 0 for label in labels}
    zero_identity_chunks = 0

    source_arrays = sorted(source_chunk_root.glob("chunk_*.npz"))
    if not source_arrays:
        raise FileNotFoundError(f"No cached validation chunks in {source_chunk_root}")
    completed_pairs = 0
    for array_path in source_arrays:
        stem = array_path.stem
        source_meta_path = source_chunk_root / f"{stem}.json"
        if not valid_sha256_sidecar(array_path) or not source_meta_path.is_file():
            raise RuntimeError(f"Invalid source validation chunk: {array_path}")
        source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
        if source_meta.get("signature") != source_signature:
            raise ValueError(f"Source chunk signature mismatch: {array_path}")
        start = int(source_meta["start"])
        stop = int(source_meta["stop"])
        if start != completed_pairs or pairs[start:stop] == []:
            raise ValueError(f"Non-contiguous source validation chunk: {array_path}")
        candidate_paths = {
            label: chunk_root / f"{stem}_{label}.csv"
            for label in labels
        }
        out_meta_path = chunk_root / f"{stem}.json"
        resumed = False
        if out_meta_path.is_file() and all(valid_sha256_sidecar(path) for path in candidate_paths.values()):
            try:
                out_meta = json.loads(out_meta_path.read_text(encoding="utf-8"))
                resumed = (
                    out_meta.get("signature") == signature
                    and int(out_meta.get("start", -1)) == start
                    and int(out_meta.get("stop", -1)) == stop
                )
            except (OSError, ValueError, json.JSONDecodeError):
                resumed = False
        if resumed:
            print(f"RB-SR CALIBRATION resume from Drive {stop}/{len(pairs)}: {stem}", flush=True)
            for label, path in candidate_paths.items():
                row_sets[label].extend(read_csv(path))
                gate_record = out_meta["gate_stats"][label]
                gate_sums[label] += float(gate_record["sum"])
                gate_counts[label] += int(gate_record["count"])
                gate_active[label] += int(gate_record["active_0p5"])
            if bool(out_meta.get("zero_gate_identity_exact", False)):
                zero_identity_chunks += 1
        else:
            print(f"RB-SR CALIBRATION live {start + 1}-{stop}/{len(pairs)}", flush=True)
            with np.load(array_path, allow_pickle=True) as saved:
                saved_pairs = [(str(a), str(b)) for a, b in saved["pairs"].tolist()]
                proposer_key = "proposer_pred" if "proposer_pred" in saved.files else "cvae_pred"
                cvae_prediction = np.asarray(saved[proposer_key], dtype=np.float32)
                learned_gate = np.asarray(saved["gate"], dtype=np.float32)
                source_rbsr_prediction = np.asarray(saved["rbsr_pred"], dtype=np.float32)
            if saved_pairs != pairs[start:stop]:
                raise ValueError(f"Source cached pairs do not match the base package: {array_path}")
            condition, _, _, controls, _, _, _ = pair_conditions(
                by_id,
                saved_pairs,
                base["source_pca"],
                base["cond_mean"],
                base["cond_std"],
            )
            ridge_prediction = ridge_predict(condition, base["ridge_cond"]).astype(np.float32)
            ridge_fixed = enforce_exact_controls(ridge_prediction, controls, landmark_indices)
            chunk_gate_stats: dict[str, dict[str, float | int]] = {}
            for value in offsets:
                label = offset_label(value)
                prediction, calibrated_gate = fuse_calibrated_residual(
                    ridge_prediction,
                    cvae_prediction,
                    learned_gate,
                    logit_offset=value,
                )
                prediction = enforce_exact_controls(prediction, controls, landmark_indices)
                if value == 0.0 and not np.allclose(
                    prediction, source_rbsr_prediction, rtol=1e-5, atol=1e-6
                ):
                    raise ValueError("beta=0 does not reproduce the frozen projection-free RB-SR prediction")
                rows = strict_metric_rows(by_id, saved_pairs, prediction)
                for offset, row in enumerate(rows):
                    row["pair_index"] = float(start + offset)
                atomic_write_csv(candidate_paths[label], rows)
                row_sets[label].extend(rows)
                chunk_gate_stats[label] = {
                    "sum": float(np.sum(calibrated_gate, dtype=np.float64)),
                    "count": int(calibrated_gate.size),
                    "active_0p5": int(np.sum(calibrated_gate >= 0.5)),
                }
            zero_prediction, zero_gate = fuse_calibrated_residual(
                ridge_prediction,
                cvae_prediction,
                learned_gate,
                force_zero_gate=True,
            )
            zero_prediction = enforce_exact_controls(zero_prediction, controls, landmark_indices)
            if not np.array_equal(zero_prediction, ridge_fixed):
                raise AssertionError("zero_gate prediction is not bitwise identical to Ridge")
            zero_rows = strict_metric_rows(by_id, saved_pairs, zero_prediction)
            for offset, row in enumerate(zero_rows):
                row["pair_index"] = float(start + offset)
            atomic_write_csv(candidate_paths["zero_gate"], zero_rows)
            row_sets["zero_gate"].extend(zero_rows)
            chunk_gate_stats["zero_gate"] = {
                "sum": 0.0,
                "count": int(zero_gate.size),
                "active_0p5": 0,
            }
            for label, record in chunk_gate_stats.items():
                gate_sums[label] += float(record["sum"])
                gate_counts[label] += int(record["count"])
                gate_active[label] += int(record["active_0p5"])
            zero_identity_chunks += 1
            atomic_write_json(
                out_meta_path,
                {
                    "signature": signature,
                    "source_array_sha256": sha256_file(array_path),
                    "start": start,
                    "stop": stop,
                    "zero_gate_identity_exact": True,
                    "gate_stats": chunk_gate_stats,
                    "pair_metric_sha256": {
                        label: sha256_file(path) for label, path in candidate_paths.items()
                    },
                },
            )
            write_sha256_sidecar(out_meta_path)
            print(f"RB-SR CALIBRATION persisted to Drive {stop}/{len(pairs)}", flush=True)
        completed_pairs = stop

    if completed_pairs != len(pairs) or any(len(rows) != len(pairs) for rows in row_sets.values()):
        raise ValueError("Calibration chunks do not cover the complete validation split")
    if zero_identity_chunks != len(source_arrays):
        raise AssertionError("The exact zero-gate Ridge identity was not verified for every chunk")

    pair_dir = args.out / "pair_metrics"
    pair_dir.mkdir(parents=True, exist_ok=True)
    candidate_records: list[dict[str, object]] = []
    for value in offsets:
        label = offset_label(value)
        path = pair_dir / f"pair_metrics_{label}_validation.csv"
        atomic_write_csv(path, row_sets[label])
        candidate_records.append(
            {
                "label": label,
                "logit_offset": value,
                "force_zero_gate": False,
                **metric_summary(row_sets[label]),
                "gate_mean": gate_sums[label] / gate_counts[label],
                "gate_active_fraction_0p5": gate_active[label] / gate_counts[label],
                "pair_metrics": str(path),
                "pair_metrics_sha256": sha256_file(path),
            }
        )
    ridge_path = pair_dir / "pair_metrics_zero_gate_ridge_validation.csv"
    atomic_write_csv(ridge_path, row_sets["zero_gate"])
    ridge_reference = {
        "label": "zero_gate",
        "logit_offset": None,
        "force_zero_gate": True,
        **metric_summary(row_sets["zero_gate"]),
        "gate_mean": 0.0,
        "gate_active_fraction_0p5": 0.0,
        "pair_metrics": str(ridge_path),
        "pair_metrics_sha256": sha256_file(ridge_path),
    }
    summary_path = args.out / "rbsr_logit_calibration_validation.csv"
    atomic_write_csv(summary_path, candidate_records + [ridge_reference])
    report = {
        "status": "VALIDATION_CALIBRATION_COMPLETE_NO_TEST_ACCESS",
        "split": "validation",
        "test_access": False,
        "selection_rule": "roi_rmse < matched Ridge and normal_flip_pct <= matched Ridge; no ratio tolerance",
        "n_pairs": len(pairs),
        "base_model_package": str(args.base_model_package),
        "base_model_package_sha256": base_hash,
        "rbsr_package": str(args.rbsr_package),
        "rbsr_package_sha256": gate_hash,
        "source_validation_report": str(args.source_validation_report),
        "source_validation_report_sha256": sha256_file(args.source_validation_report),
        "source_chunk_signature": source_signature,
        "calibration_signature": signature,
        "zero_gate_identity_exact": True,
        "exact_handle_contract": "target_controls_hard_overwrite_after_every_calibration_v1",
        "ridge_reference": ridge_reference,
        "candidates": candidate_records,
        "summary_csv": str(summary_path),
        "summary_csv_sha256": sha256_file(summary_path),
        "chunk_root": str(chunk_root),
    }
    report_path = args.out / "rbsr_logit_calibration_validation.json"
    atomic_write_json(report_path, report)
    write_sha256_sidecar(report_path)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
