"""Single path-independent entry point for Rhinoform release verification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rhinoform.reproduction import ReproductionPaths, validate_processed_facescape
from rhinoform.repro import valid_sha256_sidecar
from tools.replay_release import replay_release

FROZEN_INPUTS = (
    "docs/final_tables/FINAL_PAPER_TABLES.json",
    "results/rbsr_final_rerun_holdout_v1/protocol/FINAL_RERUN_HOLDOUT_POLICY_FREEZE.json",
    "results/rbsr_final_rerun_holdout_v1/protocol/final_rerun_holdout_split_manifest.json",
    "results/rbsr_final_rerun_holdout_v1/protocol/final_rerun_holdout_train_pairs.json",
    "results/rbsr_final_rerun_holdout_v1/direct_paired_statistics/paired_primary_rbsr_vs_ridge.json",
    "results/rbsr_final_rerun_holdout_v1/direct_paired_statistics/pair_metrics/identity_bootstrap_pair_metrics_certified_rbsr.csv",
    "results/rbsr_final_rerun_holdout_v1/direct_paired_statistics/pair_metrics/identity_bootstrap_pair_metrics_ridge.csv",
    "results/supplemental_rbsr_lamm_dominance_search_v1/release_evidence/analysis/rbsr_vs_lamm_core_paired_statistics.json",
    "results/supplemental_rbsr_lamm_dominance_search_v1/release_evidence/pair_metrics/identity_bootstrap_pair_metrics_lamm.csv",
    "results/supplemental_rbsr_lamm_dominance_search_v1/release_evidence/pair_metrics/pair_metrics_rbsr_optimized_test.csv",
)


def resolve_paths(args: argparse.Namespace) -> ReproductionPaths:
    return ReproductionPaths.resolve(
        config=args.config,
        data_root=args.data_root,
    )


def verify_frozen_inputs(paths: ReproductionPaths) -> list[str]:
    verified = []
    for relative in FROZEN_INPUTS:
        path = paths.repo_root / relative
        if not valid_sha256_sidecar(path):
            raise RuntimeError(f"Frozen input or SHA-256 sidecar is invalid: {path}")
        verified.append(relative)
    return verified


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "command",
        choices=("show-paths", "verify", "replay", "preflight"),
    )
    result.add_argument("--config", type=Path)
    result.add_argument("--data-root", type=Path)
    result.add_argument(
        "--full-data-hash",
        action="store_true",
        help="Hash all 846 processed meshes instead of checking manifest and presence only.",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    paths = resolve_paths(args)
    report: dict[str, object] = {"paths": paths.public_dict()}
    if args.command in {"verify", "preflight"}:
        report["frozen_inputs"] = verify_frozen_inputs(paths)
    if args.command == "replay":
        report["release_statistics"] = replay_release()
    if args.command == "preflight":
        report["processed_facescape"] = validate_processed_facescape(
            paths, full_hash=args.full_data_hash
        )
    report["status"] = "PASS"
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
