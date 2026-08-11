"""Fail-closed audit that RB-SR, LAMM and frozen baselines share one protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rhinoform.repro import (
    atomic_write_json,
    sha256_file,
    valid_sha256_sidecar,
    validate_torch_artifact,
)


STRICT_COLUMNS = {
    "pair_index", "source_id", "target_id", "roi_rmse", "landmark_rmse",
    "dorsum_rmse", "tip_rmse", "edge_strain_p95", "normal_flip_pct",
    "abs_flip_pct", "missed_flip_pct", "target_flip_pct", "n_new_flip_faces", "n_faces",
}
TRUTH_COLUMNS = ("target_flip_pct", "n_faces")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def pair_order(rows: list[dict[str, str]]) -> list[tuple[str, str]]:
    return [(row["source_id"], row["target_id"]) for row in rows]


def pair_order_sha256(pairs: list[tuple[str, str]]) -> str:
    payload = "\n".join(f"{source},{target}" for source, target in pairs).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--rbsr-pair-csv", type=Path, required=True)
    parser.add_argument("--rbsr-base-package", type=Path, required=True)
    parser.add_argument("--rbsr-gate-package", type=Path, required=True)
    parser.add_argument("--lamm-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root
    source_paths = {
        "manifest": root / "data/manifest.json",
        "split": root / "splits/facescape_847/split_manifest.json",
        "roi": root / "roi/vertices.json",
        "subunits": root / "roi/subunits.json",
    }
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    provenance = json.loads((args.lamm_out / "run_provenance.json").read_text(encoding="utf-8"))
    suite = json.loads((args.lamm_out / "lamm_full_evaluation_suite.json").read_text(encoding="utf-8"))
    audit = json.loads((args.lamm_out / "protocol_audit.json").read_text(encoding="utf-8"))
    if provenance.get("status") != "COMPLETE" or suite.get("status") != "PASS" or audit.get("status") != "PASS":
        raise RuntimeError("LAMM primary run is not COMPLETE/PASS")
    if provenance["source_hashes"] != source_hashes or audit["source_hashes"] != source_hashes:
        raise ValueError("RBSR repository inputs do not hash-match the completed LAMM run")

    protocol = provenance["protocol"]
    expected_protocol = {
        "train_identities": 676,
        "validation_identities": 70,
        "test_identities": 100,
        "test_pairs": 9900,
        "roi_vertices": 3934,
        "free_vertices": 3925,
        "landmarks": 9,
    }
    for key, expected in expected_protocol.items():
        if int(protocol[key]) != expected:
            raise ValueError(f"LAMM protocol mismatch for {key}: {protocol[key]} != {expected}")

    if not validate_torch_artifact(
        args.rbsr_base_package,
        required_keys=("args", "cvae_state_dict", "feature_template_sha256", "ridge_cond"),
    ):
        raise RuntimeError("RB-SR base package or SHA-256 sidecar is invalid")
    if not validate_torch_artifact(
        args.rbsr_gate_package,
        required_keys=("gate_state_dict", "base_model_package_sha256", "best_validation"),
    ):
        raise RuntimeError("RB-SR gate package or SHA-256 sidecar is invalid")
    if not valid_sha256_sidecar(args.rbsr_pair_csv):
        raise RuntimeError("RB-SR pair CSV or SHA-256 sidecar is invalid")
    base = torch.load(args.rbsr_base_package, map_location="cpu", weights_only=False)
    gate = torch.load(args.rbsr_gate_package, map_location="cpu", weights_only=False)
    if base.get("feature_template_schema") != "mean_neutral_roi_of_training_identities_v1":
        raise ValueError("RB-SR base package lacks the clean train-only template schema")
    if not base.get("feature_template_sha256"):
        raise ValueError("RB-SR base package lacks a train-only template hash")
    if gate.get("feature_template_sha256") != base["feature_template_sha256"]:
        raise ValueError("RB-SR gate/base template hashes differ")
    if gate.get("base_model_package_sha256") != sha256_file(args.rbsr_base_package):
        raise ValueError("RB-SR gate is not chained to the supplied base package")
    if not bool(base.get("args", {}).get("defer_test_evaluation", False)):
        raise ValueError("RB-SR base was not trained behind the deferred-test boundary")

    split = json.loads(source_paths["split"].read_text(encoding="utf-8"))
    train_ids = [str(value) for value in split["train_pool_ids"]]
    val_ids = [str(value) for value in split["val_ids"]]
    test_ids = [str(value) for value in split["test_ids"]]
    if set(map(str, base["train_ids"])) != set(train_ids):
        raise ValueError("RB-SR training identities differ from the frozen train pool")
    if set(map(str, base["val_ids"])) != set(val_ids) or set(map(str, base["test_ids"])) != set(test_ids):
        raise ValueError("RB-SR held-out identities differ from the frozen split")
    expected_pairs = [(source, target) for source in test_ids for target in test_ids if source != target]
    if [(str(a), str(b)) for a, b in base["test_pairs"]] != expected_pairs:
        raise ValueError("RB-SR package test-pair order differs from the frozen nested order")

    lamm_csv = args.lamm_out / "identity_bootstrap_pair_metrics_lamm.csv"
    lamm_fields, lamm_rows = read_csv(lamm_csv)
    rbsr_fields, rbsr_rows = read_csv(args.rbsr_pair_csv)
    if not STRICT_COLUMNS.issubset(lamm_fields) or not STRICT_COLUMNS.issubset(rbsr_fields):
        raise ValueError("LAMM or RB-SR pair CSV lacks strict transparency columns")
    if len(lamm_rows) != 9900 or len(rbsr_rows) != 9900:
        raise ValueError("LAMM and RB-SR must each contain exactly 9,900 test pairs")
    if pair_order(lamm_rows) != expected_pairs or pair_order(rbsr_rows) != expected_pairs:
        raise ValueError("LAMM/RB-SR pair order does not equal the frozen ordered test pairs")
    if max(abs(float(row["landmark_rmse"])) for row in rbsr_rows) > 1e-8:
        raise ValueError("RB-SR controls were not hard-fixed before strict scoring")
    for column in TRUTH_COLUMNS:
        left = np.asarray([float(row[column]) for row in lamm_rows])
        right = np.asarray([float(row[column]) for row in rbsr_rows])
        if not np.array_equal(left, right):
            raise ValueError(f"LAMM/RB-SR truth column mismatch: {column}")

    clean_rbsr_family = []
    for path in sorted(args.rbsr_pair_csv.parent.glob("pair_metrics_*_test.csv")):
        fields, rows = read_csv(path)
        if not STRICT_COLUMNS.issubset(fields) or pair_order(rows) != expected_pairs:
            raise ValueError(f"Clean RB-SR-family comparator is not protocol matched: {path}")
        for column in TRUTH_COLUMNS:
            reference = np.asarray([float(row[column]) for row in lamm_rows])
            candidate = np.asarray([float(row[column]) for row in rows])
            if not np.array_equal(reference, candidate):
                raise ValueError(f"Clean RB-SR-family truth mismatch in {column}: {path}")
        clean_rbsr_family.append({"path": str(path), "sha256": sha256_file(path), "n_pairs": len(rows)})

    comparison_dir = args.lamm_out / "direct_comparison_pair_metrics"
    baseline_records = []
    for path in sorted(comparison_dir.glob("identity_bootstrap_pair_metrics_*.csv")):
        fields, rows = read_csv(path)
        if not STRICT_COLUMNS.issubset(fields):
            raise ValueError(f"Frozen baseline lacks strict columns: {path}")
        if pair_order(rows) != expected_pairs:
            raise ValueError(f"Frozen baseline pair order mismatch: {path}")
        for column in TRUTH_COLUMNS:
            reference = np.asarray([float(row[column]) for row in lamm_rows])
            candidate = np.asarray([float(row[column]) for row in rows])
            if not np.array_equal(reference, candidate):
                raise ValueError(f"Frozen baseline truth mismatch in {column}: {path}")
        method = path.stem.removeprefix("identity_bootstrap_pair_metrics_")
        evidence_status = (
            "historical_training_provenance_evaluation_protocol_matched"
            if method in {"cvae_only", "hybrid_alpha_0.1"}
            else "verified_evaluation_protocol_matched"
        )
        baseline_records.append({
            "method": method,
            "path": str(path),
            "sha256": sha256_file(path),
            "n_pairs": len(rows),
            "evidence_status": evidence_status,
        })
    expected_baselines = {
        "ridge_sourcepca",
        "hybrid_alpha_0.1",
        "cvae_only",
        "laplacian_handles",
        "bilaplacian_handles",
        "arap_handles_iter3",
        "lamm",
    }
    found_baselines = {record["method"] for record in baseline_records}
    missing_baselines = sorted(expected_baselines - found_baselines)
    if missing_baselines:
        raise FileNotFoundError(
            f"Frozen direct-comparison baseline family is incomplete: {missing_baselines}"
        )

    report = {
        "status": "PASS",
        "conclusion": (
            "Clean RB-SR, completed LAMM and every supplied frozen baseline use the same "
            "676/70/100 split, 9,900 ordered pairs, 3,934-vertex ROI, nine hard-fixed controls, "
            "3,925-free-vertex RMSE and target-relative new-flip truth columns."
        ),
        "provenance_qualification": (
            "Protocol comparability does not repair historical CVAE/Hybrid training-template "
            "provenance; those rows remain historical until separately clean-retrained."
        ),
        "source_hashes": source_hashes,
        "protocol": expected_protocol,
        "pair_order_sha256": pair_order_sha256(expected_pairs),
        "rbsr": {
            "pair_csv": str(args.rbsr_pair_csv),
            "pair_csv_sha256": sha256_file(args.rbsr_pair_csv),
            "base_package_sha256": sha256_file(args.rbsr_base_package),
            "gate_package_sha256": sha256_file(args.rbsr_gate_package),
            "feature_template_schema": base["feature_template_schema"],
            "feature_template_sha256": base["feature_template_sha256"],
            "test_was_deferred_during_training": True,
            "strict_family_pair_csvs": clean_rbsr_family,
        },
        "lamm": {
            "pair_csv": str(lamm_csv),
            "pair_csv_sha256": sha256_file(lamm_csv),
            "checkpoint_sha256": provenance["checkpoints"]["manipulation"],
            "official_commit": provenance["lamm_commit"],
        },
        "frozen_baselines": baseline_records,
        "required_frozen_baselines": sorted(expected_baselines),
    }
    atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
