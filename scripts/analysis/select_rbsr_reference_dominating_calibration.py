"""Freeze an RB-SR calibration that dominates a matched validation reference.

The reference is normally the frozen LAMM validation evaluation.  Test metrics
are deliberately not accepted by this selector.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rhinoform.rbsr_calibration import select_reference_dominating_operating_point
from rhinoform.repro import (
    atomic_write_json,
    sha256_file,
    valid_sha256_sidecar,
    write_sha256_sidecar,
)


def reference_metrics(payload: dict[str, object]) -> dict[str, object]:
    values = payload.get("means", payload.get("summary"))
    if not isinstance(values, dict):
        raise ValueError("Reference summary must contain a means or summary object")
    return dict(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--reference-validation-summary", type=Path, required=True)
    parser.add_argument("--relative-margin", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.calibration_report, args.reference_validation_summary):
        if not valid_sha256_sidecar(path):
            raise RuntimeError(f"Artifact or SHA-256 sidecar is invalid: {path}")
    report = json.loads(args.calibration_report.read_text(encoding="utf-8"))
    reference_payload = json.loads(args.reference_validation_summary.read_text(encoding="utf-8"))
    if report.get("status") != "VALIDATION_CALIBRATION_COMPLETE_NO_TEST_ACCESS":
        raise ValueError("Selector requires completed validation-only RB-SR calibration")
    if report.get("split") != "validation" or report.get("test_access") is not False:
        raise ValueError("RB-SR calibration is not validation-only")
    if int(report.get("n_pairs", 0)) != 4830:
        raise ValueError("RB-SR calibration must cover all 4,830 validation pairs")
    if reference_payload.get("split") != "validation":
        raise ValueError("Reference is not a validation evaluation")
    if int(reference_payload.get("n_pairs", 0)) != 4830:
        raise ValueError("Reference must cover all 4,830 validation pairs")
    reference = reference_metrics(reference_payload)
    candidates = [dict(row) for row in report.get("candidates", [])]
    if not candidates:
        raise ValueError("Calibration report has no candidates")
    for candidate in candidates:
        pair_path = Path(str(candidate["pair_metrics"]))
        if not valid_sha256_sidecar(pair_path):
            raise RuntimeError(f"Candidate pair metrics or sidecar is invalid: {pair_path}")

    selected, assessed = select_reference_dominating_operating_point(
        candidates,
        reference,
        relative_margin=args.relative_margin,
    )
    common = {
        "selection_policy": (
            "Maximise the minimum relative validation improvement over the frozen matched "
            "reference across ROI RMSE, target-relative new flip, and edge-strain p95 among "
            "candidates satisfying the declared margin."
        ),
        "selection_split": "validation",
        "n_validation_pairs": 4830,
        "relative_margin": float(args.relative_margin),
        "base_model_package_sha256": report["base_model_package_sha256"],
        "rbsr_package_sha256": report["rbsr_package_sha256"],
        "calibration_signature": report["calibration_signature"],
        "calibration_report": str(args.calibration_report),
        "calibration_report_sha256": sha256_file(args.calibration_report),
        "reference_validation_summary": str(args.reference_validation_summary),
        "reference_validation_summary_sha256": sha256_file(args.reference_validation_summary),
        "reference": reference,
        "candidate_count": len(candidates),
        "feasible_count": sum(bool(row["dominates_reference_core"]) for row in assessed),
        "candidates": assessed,
    }
    if selected is None:
        freeze = {
            "status": "NO_REFERENCE_DOMINATING_VALIDATION_CALIBRATION",
            "test_access": False,
            "selected": None,
            **common,
        }
    else:
        freeze = {
            "status": "FROZEN_REFERENCE_DOMINATING_VALIDATION_CALIBRATION",
            "test_access": True,
            "test_access_note": "Only this exact base/gate/calibration hash chain is unlocked.",
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
