"""Aggregate chunk-resumable per-candidate RB-SR validation evaluations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rhinoform.repro import (
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
    valid_sha256_sidecar,
    validate_torch_artifact,
)


def parse_mapping(values: list[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use name=path syntax: {value!r}")
        name, raw_path = value.split("=", 1)
        result[name.strip()] = Path(raw_path.strip())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="Repeat label=gate_package.pt")
    parser.add_argument("--evaluation", action="append", required=True, help="Repeat label=evaluation.json")
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    models = parse_mapping(args.model, "--model")
    evaluations = parse_mapping(args.evaluation, "--evaluation")
    if set(models) != set(evaluations):
        raise ValueError("Model and evaluation labels differ")

    summary_rows = []
    reports = {}
    for label in models:
        package_path = models[label]
        report_path = evaluations[label]
        if not validate_torch_artifact(
            package_path,
            required_keys=("gate_state_dict", "base_model_package_sha256", "best_validation"),
        ):
            raise RuntimeError(f"Gate package or sidecar invalid: {package_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("split") != args.split or int(report.get("n_pairs", 0)) != 4830:
            raise ValueError(f"Incomplete/non-validation evaluation: {report_path}")
        if report.get("rbsr_package_sha256") != sha256_file(package_path):
            raise ValueError(f"Evaluation/package hash mismatch for {label}")
        pair_path = Path(report["pair_metrics"])
        if not valid_sha256_sidecar(pair_path):
            raise RuntimeError(f"Pair CSV or sidecar invalid: {pair_path}")
        gate_summary = report["gate"]
        row = {
            "label": label,
            "n_pairs": int(report["n_pairs"]),
            **report["summary"],
            "gate_mean": float(gate_summary["mean"]),
            "gate_p95": float(gate_summary["p95"]),
            "gate_active_fraction_0p5": float(gate_summary["active_fraction_0p5"]),
        }
        summary_rows.append(row)
        reports[label] = {
            "package": str(package_path),
            "package_sha256": sha256_file(package_path),
            "evaluation": str(report_path),
            "evaluation_sha256": sha256_file(report_path),
            "best_validation": gate_summary["validation_selected_checkpoint"],
            "summary": row,
        }

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "rbsr_ablation_summary_validation.csv"
    json_path = args.out / "rbsr_ablation_summary_validation.json"
    atomic_write_csv(csv_path, summary_rows)
    atomic_write_json(json_path, {
        "split": args.split,
        "n_pairs": 4830,
        "selection_boundary": "Each checkpoint was evaluated on validation only using chunk-resumable strict scoring.",
        "models": reports,
    })
    print(f"VALIDATION AGGREGATE persisted to Drive: {len(summary_rows)} candidates", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
