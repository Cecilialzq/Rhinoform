"""Freeze a strictly Ridge-beating RB-SR calibration using validation only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rhinoform.rbsr_calibration import ridge_reference_matches, select_strict_operating_point
from rhinoform.repro import (
    atomic_write_json,
    sha256_file,
    valid_sha256_sidecar,
    write_sha256_sidecar,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--ridge-reference-csv", type=Path, required=True)
    parser.add_argument("--reference-label", default="ridge_anchor")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not valid_sha256_sidecar(args.calibration_report):
        raise RuntimeError("Calibration report or SHA-256 sidecar is invalid")
    report = json.loads(args.calibration_report.read_text(encoding="utf-8"))
    if report.get("status") != "VALIDATION_CALIBRATION_COMPLETE_NO_TEST_ACCESS":
        raise ValueError("Selector requires a completed validation-only calibration report")
    if report.get("split") != "validation" or int(report.get("n_pairs", 0)) != 4830:
        raise ValueError("Selector requires the complete 4,830-pair validation split")
    if report.get("test_access") is not False:
        raise ValueError("Calibration report accessed or unlocked test before selection")
    if report.get("zero_gate_identity_exact") is not True:
        raise ValueError("Calibration did not prove the exact zero-gate Ridge identity")

    ridge_reference = dict(report["ridge_reference"])
    if not valid_sha256_sidecar(Path(str(ridge_reference["pair_metrics"]))):
        raise RuntimeError("Calibration Ridge pair metrics or sidecar is invalid")
    external_rows = read_csv(args.ridge_reference_csv)
    external_ridge = next(
        (row for row in external_rows if row.get("label") == args.reference_label),
        None,
    )
    if external_ridge is None:
        raise KeyError(f"Reference label {args.reference_label!r} not found")
    if not ridge_reference_matches(ridge_reference, external_ridge):
        raise ValueError(
            "zero_gate does not reproduce the frozen matched Ridge within 1e-8 "
            "float-serialisation tolerance"
        )

    candidates = [dict(row) for row in report["candidates"]]
    if not candidates:
        raise ValueError("Calibration report has no non-zero candidates")
    for candidate in candidates:
        pair_path = Path(str(candidate["pair_metrics"]))
        if not valid_sha256_sidecar(pair_path):
            raise RuntimeError(f"Candidate pair metrics or sidecar is invalid: {pair_path}")
    selected, assessed = select_strict_operating_point(candidates, ridge_reference)
    common = {
        "selection_policy": (
            "Minimise strict validation ROI RMSE among candidates satisfying "
            "roi_rmse < matched Ridge and target-relative new flip <= matched Ridge; "
            "no ratio tolerance and no test access during selection."
        ),
        "selection_split": "validation",
        "n_validation_pairs": 4830,
        "base_model_package_sha256": report["base_model_package_sha256"],
        "rbsr_package_sha256": report["rbsr_package_sha256"],
        "calibration_signature": report["calibration_signature"],
        "calibration_report": str(args.calibration_report),
        "calibration_report_sha256": sha256_file(args.calibration_report),
        "ridge_reference_csv": str(args.ridge_reference_csv),
        "ridge_reference_csv_sha256": sha256_file(args.ridge_reference_csv),
        "ridge_reference": ridge_reference,
        "candidate_count": len(candidates),
        "feasible_count": sum(bool(row["strictly_beats_ridge"]) for row in assessed),
        "candidates": assessed,
    }
    if selected is None:
        freeze = {
            "status": "NO_FEASIBLE_STRICT_CALIBRATION",
            "test_access": False,
            "selected": None,
            **common,
        }
    else:
        freeze = {
            "status": "FROZEN_STRICT_VALIDATION_CALIBRATION",
            "test_access": True,
            "test_access_note": (
                "One-shot test evaluation is unlocked only for this exact base/gate/calibration hash chain."
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
