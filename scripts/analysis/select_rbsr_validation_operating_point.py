"""Freeze one RB-SR gate package using exact strict validation metrics only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rhinoform.repro import atomic_write_json, sha256_file


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--safety-reference-csv", type=Path, required=True)
    parser.add_argument("--reference-label", default="ridge_anchor")
    parser.add_argument("--max-flip-ratio", type=float, default=1.05)
    parser.add_argument("--max-strain-ratio", type=float, default=1.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.max_flip_ratio < 1.0 or args.max_strain_ratio < 1.0:
        raise ValueError("Safety ratios must be >= 1")
    candidates = read_rows(args.validation_summary)
    if not candidates:
        raise ValueError("No RB-SR validation candidates")
    reference_rows = read_rows(args.safety_reference_csv)
    reference = next(
        (row for row in reference_rows if row.get("label") == args.reference_label),
        None,
    )
    if reference is None:
        raise KeyError(f"Reference label {args.reference_label!r} not found")
    flip_limit = float(reference["normal_flip_pct"]) * args.max_flip_ratio
    strain_limit = float(reference["edge_strain_p95"]) * args.max_strain_ratio
    for row in candidates:
        row["within_frozen_safety_envelope"] = (
            float(row["normal_flip_pct"]) <= flip_limit
            and float(row["edge_strain_p95"]) <= strain_limit
        )
    feasible = [row for row in candidates if row["within_frozen_safety_envelope"]]
    if not feasible:
        raise RuntimeError(
            "No RB-SR candidate satisfies the frozen validation safety envelope; "
            "do not open test metrics."
        )
    selected = min(
        feasible,
        key=lambda row: (
            float(row["roi_rmse"]),
            float(row["normal_flip_pct"]),
            float(row["edge_strain_p95"]),
            float(row["gate_mean"]),
            row["label"],
        ),
    )

    evaluation = json.loads(args.validation_report.read_text(encoding="utf-8"))
    if evaluation.get("split") != "validation":
        raise ValueError("Operating-point selection requires a validation report")
    model_record = evaluation["models"][selected["label"]]
    package_path = Path(model_record["package"])
    if not package_path.is_file():
        raise FileNotFoundError(package_path)
    package_hash = sha256_file(package_path)
    if package_hash != model_record["package_sha256"]:
        raise ValueError("Selected package hash does not match the validation report")

    report = {
        "status": "FROZEN_VALIDATION_SELECTION",
        "test_access": "This selector consumes validation summaries only.",
        "selection_policy": (
            "Minimise strict validation free-ROI RMSE within the frozen new-flip and "
            "edge-strain envelope; ties prefer lower new-flip, lower strain and lower gate mean."
        ),
        "safety_reference": {
            "csv": str(args.safety_reference_csv),
            "label": args.reference_label,
            "normal_flip_pct": float(reference["normal_flip_pct"]),
            "edge_strain_p95": float(reference["edge_strain_p95"]),
            "max_flip_ratio": args.max_flip_ratio,
            "max_strain_ratio": args.max_strain_ratio,
            "normal_flip_pct_limit": flip_limit,
            "edge_strain_p95_limit": strain_limit,
        },
        "selected": {
            **selected,
            "package": str(package_path),
            "package_sha256": package_hash,
        },
        "candidate_count": len(candidates),
        "feasible_count": len(feasible),
        "inputs_sha256": {
            "validation_summary": sha256_file(args.validation_summary),
            "validation_report": sha256_file(args.validation_report),
            "safety_reference_csv": sha256_file(args.safety_reference_csv),
        },
    }
    atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
