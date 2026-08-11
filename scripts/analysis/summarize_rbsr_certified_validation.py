"""Assemble report-ready validation evidence from certified RB-SR freezes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rhinoform.repro import (
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
    valid_sha256_sidecar,
    write_sha256_sidecar,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-freeze", type=Path, required=True)
    parser.add_argument("--capacity-freeze", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    specs = [
        ("canonical_clean_pca16", "original_canonical", args.canonical_freeze),
        ("capacity_extension", "posthoc_representation_capacity_extension", args.capacity_freeze),
    ]
    rows = []
    evidence = []
    for track, role, path in specs:
        if not valid_sha256_sidecar(path):
            raise RuntimeError(f"Freeze or sidecar is invalid: {path}")
        freeze = json.loads(path.read_text(encoding="utf-8"))
        if freeze.get("status") != "FROZEN_CERTIFIED_RIDGE_FOLD_PROJECTION":
            raise ValueError(f"Track is not certified: {track}")
        selected = dict(freeze["selected"])
        ridge = dict(freeze["ridge_reference"])
        ridge_rmse = float(ridge["roi_rmse"])
        candidate_rmse = float(selected["roi_rmse"])
        ridge_flip = float(ridge["normal_flip_pct"])
        candidate_flip = float(selected["normal_flip_pct"])
        row = {
            "track": track,
            "role": role,
            "split": "validation",
            "n_pairs": int(freeze["n_validation_pairs"]),
            "ridge_roi_rmse": ridge_rmse,
            "rbsr_roi_rmse": candidate_rmse,
            "rmse_absolute_improvement": ridge_rmse - candidate_rmse,
            "rmse_relative_improvement_pct": 100.0 * (ridge_rmse - candidate_rmse) / ridge_rmse,
            "ridge_new_flip_pct": ridge_flip,
            "rbsr_new_flip_pct": candidate_flip,
            "new_flip_absolute_change_pp": candidate_flip - ridge_flip,
            "new_flip_relative_reduction_pct": 100.0 * (ridge_flip - candidate_flip) / ridge_flip,
            "ridge_edge_strain_p95": float(ridge["edge_strain_p95"]),
            "rbsr_edge_strain_p95": float(selected["edge_strain_p95"]),
            "hard_certificate_rate": float(selected["certificate_rate"]),
            "mean_residual_retention": float(selected["mean_retention"]),
            "attenuation": float(selected["attenuation"]),
            "test_access": False,
        }
        rows.append(row)
        evidence.append(
            {
                "track": track,
                "freeze": str(path),
                "freeze_sha256": sha256_file(path),
                "base_model_package_sha256": freeze["base_model_package_sha256"],
                "rbsr_package_sha256": freeze["rbsr_package_sha256"],
                "projection_signature": freeze["projection_signature"],
                "implementation_hashes": freeze["implementation_hashes"],
            }
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "certified_projection_validation_summary.csv"
    json_path = args.out_dir / "certified_projection_validation_summary.json"
    atomic_write_csv(csv_path, rows)
    atomic_write_json(
        json_path,
        {
            "status": "VALIDATION_ONLY_CERTIFIED_DEVELOPMENT_EVIDENCE",
            "test_access": False,
            "claim_boundary": (
                "Report as validation-only method-development evidence. The final headline awaits the "
                "separately frozen internal blind retrain confirmation."
            ),
            "rows": rows,
            "evidence": evidence,
            "summary_csv": csv_path.name,
            "summary_csv_sha256": sha256_file(csv_path),
        },
    )
    write_sha256_sidecar(json_path)
    print(json.dumps({"status": "PASS", "rows": rows}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
