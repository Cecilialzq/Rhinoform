"""Freeze a fresh internal identity holdout before any confirmation evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rhinoform.confirmation import blind_confirmation_partition, validate_frozen_classical_configuration
from rhinoform.repro import atomic_write_json, sha256_file, valid_sha256_sidecar, write_sha256_sidecar
from rhinoform.sampling import coverage_balanced_ordered_pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-split-manifest", type=Path, required=True)
    parser.add_argument("--development-freeze", type=Path, required=True)
    parser.add_argument("--geometric-development-selection", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260809)
    parser.add_argument("--model-seed", type=int, default=20260609)
    parser.add_argument("--val-size", type=int, default=70)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--train-pair-budget", type=int, default=870)
    args = parser.parse_args()

    if not args.parent_split_manifest.is_file():
        raise FileNotFoundError(args.parent_split_manifest)
    if not valid_sha256_sidecar(args.development_freeze):
        raise RuntimeError("Development projection freeze or SHA-256 sidecar is invalid")
    if not args.geometric_development_selection.is_file():
        raise FileNotFoundError(args.geometric_development_selection)
    parent = json.loads(args.parent_split_manifest.read_text(encoding="utf-8"))
    development = json.loads(args.development_freeze.read_text(encoding="utf-8"))
    if development.get("status") != "FROZEN_CERTIFIED_RIDGE_FOLD_PROJECTION":
        raise ValueError("A certified development operating point must be frozen first")
    selected = dict(development["selected"])
    if selected.get("hard_certificate_complete") is not True:
        raise ValueError("Development operating point lacks a complete hard certificate")
    geometric = json.loads(args.geometric_development_selection.read_text(encoding="utf-8"))
    classical_selected = dict(geometric.get("selected", {}))
    if set(classical_selected) != {"laplacian", "bilaplacian", "arap"}:
        raise ValueError("Geometric development selection lacks the three declared baselines")
    validate_frozen_classical_configuration(classical_selected)

    partition = blind_confirmation_partition(
        parent,
        seed=args.split_seed,
        val_size=args.val_size,
        test_size=args.test_size,
    )
    partition.update(
        {
            "status": "FROZEN_BEFORE_CONFIRMATION_TRAINING",
            "parent_split_manifest_sha256": sha256_file(args.parent_split_manifest),
            "parent_test_status": "previously_observed_and_excluded_from_confirmation",
            "test_access": False,
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.out_dir / "final_rerun_holdout_split_manifest.json"
    atomic_write_json(split_path, partition)
    write_sha256_sidecar(split_path)

    train_ids = [str(value) for value in partition["train_pool_ids"]]
    train_pairs = coverage_balanced_ordered_pairs(train_ids, args.train_pair_budget, args.split_seed)
    pair_manifest = {
        "status": "FROZEN_BEFORE_CONFIRMATION_TRAINING",
        "label": "final_rerun_holdout_train_pairs",
        "seed": args.split_seed,
        "budget": args.train_pair_budget,
        "n_identities": len(train_ids),
        "n_pairs": len(train_pairs),
        "source_coverage": len({source for source, _ in train_pairs}),
        "target_coverage": len({target for _, target in train_pairs}),
        "split_manifest_sha256": sha256_file(split_path),
        "ids": train_ids,
        "pairs": [
            {"source_id": source, "target_id": target}
            for source, target in train_pairs
        ],
    }
    if pair_manifest["source_coverage"] != len(train_ids) or pair_manifest["target_coverage"] != len(train_ids):
        raise AssertionError("Frozen training pairs do not cover every training identity in both roles")
    pair_path = args.out_dir / "final_rerun_holdout_train_pairs.json"
    atomic_write_json(pair_path, pair_manifest)
    write_sha256_sidecar(pair_path)

    policy = {
        "status": "FROZEN_INTERNAL_FINAL_RERUN_HOLDOUT_POLICY",
        "scope": (
            "Retrospective internal identity-disjoint final-rerun holdout. These identities and some "
            "pair outcomes were available during earlier project development, so this is not a "
            "historically untouched or external confirmation. They are excluded from every input to "
            "the final retraining and the frozen holdout is not reopened until all models are frozen."
        ),
        "claim_boundary": "internal held-out-from-final-retrain evidence; not historically untouched and not external",
        "historical_exposure": {
            "identities_previously_used_in_development": True,
            "some_pair_outcomes_may_have_been_seen_in_prior_training_or_validation": True,
            "independent_external_confirmation": False,
        },
        "test_access": False,
        "test_access_count": 0,
        "split_manifest": split_path.name,
        "split_manifest_sha256": sha256_file(split_path),
        "train_pair_manifest": pair_path.name,
        "train_pair_manifest_sha256": sha256_file(pair_path),
        "parent_split_manifest_sha256": sha256_file(args.parent_split_manifest),
        "development_projection_freeze_sha256": sha256_file(args.development_freeze),
        "development_geometric_selection_sha256": sha256_file(args.geometric_development_selection),
        "model_seed": args.model_seed,
        "split_seed": args.split_seed,
        "frozen_rbsr_configuration": {
            "source_pca_dim": 64,
            "delta_pca_dim": 16,
            "ridge_lambda": 300.0,
            "use_subunit_features": False,
            "base_epochs": 180,
            "gate_projection": "none",
            "gate_epochs": 80,
            "gate_orientation_budget_multiplier": 1.0,
            "gate_strain_budget_multiplier": 0.8,
            "hard_projection_attenuation": float(selected["attenuation"]),
            "hard_projection_smoothing_steps": int(selected["smoothing_steps"]),
            "hard_projection_max_iterations": int(selected["max_iterations"]),
            "hard_projection_uniform_steps": int(selected["uniform_steps"]),
        },
        "frozen_classical_configuration": {
            method: {
                "handle_weight": float(config["handle_weight"]),
                "system_ridge": float(config.get("system_ridge") or 1e-8),
                "arap_iter": (
                    int(config["arap_iter"])
                    if method == "arap"
                    else None
                ),
                "selection_source": "development-validation-frozen-before-final-rerun",
            }
            for method, config in classical_selected.items()
        },
        "matched_comparison_contract": {
            "same_split": True,
            "same_train_identity_pool": True,
            "rbsr_train_pair_manifest_frozen": True,
            "method_native_training_pairing_documented": True,
            "same_9_hard_fixed_controls": True,
            "same_3925_free_vertex_rmse": True,
            "same_target_relative_new_flip": True,
            "retrain_ridge_rbsr_and_lamm": True,
            "reevaluate_classical_baselines": True,
        },
        "primary_hypothesis": (
            "On the 9,900 pairs held out from the final retraining, certified RB-SR has mean free-ROI "
            "RMSE strictly below its matched Ridge anchor, target-relative new flip no greater than "
            "Ridge, and a projected-fold-set subset certificate rate of exactly 1.0."
        ),
        "statistical_analysis_plan": {
            "primary_family": "certified RB-SR versus matched Ridge across the six frozen metrics",
            "primary_tests": "source-identity paired two-sided Wilcoxon with Holm correction over six metrics",
            "uncertainty": "10,000 source-identity and target-identity cluster bootstrap replicates",
            "secondary_family": "all reportable methods versus matched Ridge across the same six metrics",
            "secondary_direct_comparison": "certified RB-SR versus LAMM across the same six metrics",
            "degenerate_rule": "all-zero or non-finite Wilcoxon is p=1 and never significant",
        },
        "failure_policy": "Fail closed; do not replace, tune, or rerun the frozen test operating point.",
    }
    policy_path = args.out_dir / "FINAL_RERUN_HOLDOUT_POLICY_FREEZE.json"
    atomic_write_json(policy_path, policy)
    write_sha256_sidecar(policy_path)
    print(json.dumps(policy, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
