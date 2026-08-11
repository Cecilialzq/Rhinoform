"""Freeze a hard-certified Ridge-fold projection using validation only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rhinoform.rbsr_calibration import select_certified_projection
from rhinoform.repro import (
    atomic_write_json,
    sha256_file,
    valid_sha256_sidecar,
    write_sha256_sidecar,
)


def resolve_report_artifact(report_path: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else report_path.parent / path


def portable_reference(path: Path, anchor: Path) -> str:
    try:
        return path.resolve().relative_to(anchor.resolve()).as_posix()
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not valid_sha256_sidecar(args.projection_report):
        raise RuntimeError("Projection report or SHA-256 sidecar is invalid")
    report = json.loads(args.projection_report.read_text(encoding="utf-8"))
    if report.get("status") != "VALIDATION_HARD_RIDGE_FOLD_PROJECTION_COMPLETE_NO_TEST_ACCESS":
        raise ValueError("Selector requires a completed validation-only hard-projection report")
    if report.get("split") != "validation" or int(report.get("n_pairs", 0)) != 4830:
        raise ValueError("Selector requires the complete 4,830-pair validation split")
    if report.get("test_access") is not False:
        raise ValueError("Projection report accessed test before selection")
    zero_identity = report.get("zero_gate_ridge_identity")
    if not isinstance(zero_identity, dict) or zero_identity.get("passed") is not True:
        raise ValueError("Projection report lacks the exact zero-residual Ridge identity audit")

    ridge_reference = dict(report["ridge_reference"])
    ridge_path = resolve_report_artifact(args.projection_report, ridge_reference["pair_metrics"])
    if not valid_sha256_sidecar(ridge_path):
        raise RuntimeError("Projected-run Ridge metrics or sidecar are invalid")
    if sha256_file(ridge_path) != ridge_reference["pair_metrics_sha256"]:
        raise ValueError("Projected-run Ridge metrics hash mismatch")

    candidates = [dict(row) for row in report["candidates"]]
    if not candidates:
        raise ValueError("Projection report has no candidates")
    for candidate in candidates:
        metric_path = resolve_report_artifact(args.projection_report, candidate["pair_metrics"])
        diagnostic_path = resolve_report_artifact(args.projection_report, candidate["projection_diagnostics"])
        if not valid_sha256_sidecar(metric_path) or not valid_sha256_sidecar(diagnostic_path):
            raise RuntimeError(f"Candidate artifacts or sidecars are invalid: {candidate['label']}")
        if sha256_file(metric_path) != candidate["pair_metrics_sha256"]:
            raise ValueError(f"Candidate metrics hash mismatch: {candidate['label']}")
        if sha256_file(diagnostic_path) != candidate["projection_diagnostics_sha256"]:
            raise ValueError(f"Candidate diagnostics hash mismatch: {candidate['label']}")

    selected, assessed = select_certified_projection(candidates, ridge_reference)
    common = {
        "selection_policy": (
            "Minimise full-validation ROI RMSE among candidates with roi_rmse < matched Ridge, "
            "target-relative new flip <= matched Ridge, and a per-pair projected-fold-set "
            "subset certificate rate of exactly 1.0. No test access during selection."
        ),
        "selection_split": "validation",
        "n_validation_pairs": 4830,
        "base_model_package_sha256": report["base_model_package_sha256"],
        "rbsr_package_sha256": report["rbsr_package_sha256"],
        "projection_signature": report["projection_signature"],
        "implementation_hashes": report["implementation_hashes"],
        "artifact_paths_relative_to": "this_freeze_directory",
        "projection_report": portable_reference(args.projection_report, args.out.parent),
        "projection_report_sha256": sha256_file(args.projection_report),
        "ridge_reference": ridge_reference,
        "zero_gate_ridge_identity": zero_identity,
        "candidate_count": len(candidates),
        "feasible_count": sum(bool(row["strictly_beats_ridge"]) for row in assessed),
        "candidates": assessed,
    }
    if selected is None:
        freeze = {
            "status": "NO_FEASIBLE_CERTIFIED_RIDGE_FOLD_PROJECTION",
            "test_access": False,
            "selected": None,
            **common,
        }
    else:
        freeze = {
            "status": "FROZEN_CERTIFIED_RIDGE_FOLD_PROJECTION",
            "test_access": True,
            "test_access_note": (
                "Only the exact base/gate/projection hash chain is unlocked for one-shot evaluation."
            ),
            "selected": selected,
            **common,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out, freeze)
    write_sha256_sidecar(args.out)
    print(json.dumps(freeze, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
