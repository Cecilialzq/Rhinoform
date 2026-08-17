from __future__ import annotations

import csv
import json
from pathlib import Path

from rhinoform.repro import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
    write_sha256_sidecar,
)


METRICS = (
    "roi_rmse",
    "landmark_rmse",
    "dorsum_rmse",
    "tip_rmse",
    "edge_strain_p95",
    "normal_flip_pct",
    "abs_flip_pct",
    "missed_flip_pct",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summary_row(method: str, evidence_role: str, summary: dict) -> dict:
    return {
        "method": method,
        "evidence_role": evidence_role,
        **{metric: float(summary[metric]) for metric in METRICS},
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    atomic_write_csv(path, rows)
    write_sha256_sidecar(path)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    labels = {
        "method": "Method",
        "roi_rmse": "ROI RMSE",
        "normal_flip_pct": "New flip %",
        "edge_strain_p95": "Edge strain p95",
        "abs_flip_pct": "Absolute flip %",
        "missed_flip_pct": "Missed flip %",
    }
    lines = [
        "| " + " | ".join(labels.get(column, column) for column in columns) + " |",
        "|" + "|".join("---" if column == "method" else "---:" for column in columns) + "|",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(str(value) if column == "method" else f"{float(value):.6f}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    final = root / "results/rbsr_final_rerun_holdout_v1"
    supplemental = root / "results/supplemental_rbsr_lamm_dominance_search_v1/release_evidence"
    out = root / "docs/final_tables"
    out.mkdir(parents=True, exist_ok=True)

    final_test_path = final / "rbsr/seed20260609/one_shot_test/rbsr_evaluation_test.json"
    lamm_path = final / "lamm/seed20260609/lamm_test_summary.json"
    classical_path = final / "classical/classical_confirmation_test_summary.json"
    validation_path = final / "rbsr/seed20260609/validation_raw/rbsr_evaluation_validation.json"
    projection_path = final / "rbsr/seed20260609/certified_projection_validation/RBSR_CERTIFIED_PROJECTION_FREEZE.json"
    primary_stats_path = final / "direct_paired_statistics/paired_primary_rbsr_vs_ridge.json"
    dominance_path = supplemental / "analysis/RBSR_VS_LAMM_CORE_DOMINANCE_VERDICT.json"
    dominance_stats_path = supplemental / "analysis/rbsr_vs_lamm_core_paired_statistics.json"

    source_paths = (
        final_test_path,
        lamm_path,
        classical_path,
        validation_path,
        projection_path,
        primary_stats_path,
        dominance_path,
        dominance_stats_path,
    )
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    final_test = read_json(final_test_path)
    lamm = read_json(lamm_path)
    classical = read_json(classical_path)
    validation = read_json(validation_path)
    projection = read_json(projection_path)
    primary_stats = read_json(primary_stats_path)
    dominance = read_json(dominance_path)
    dominance_stats = read_json(dominance_stats_path)

    main_rows = [summary_row("Certified RB-SR", "primary_internal_final_rerun", final_test["summary"])]
    comparator_names = {
        "ridge_sourcepca_clean": "Ridge",
        "cvae_clean": "CVAE",
        "hybrid_validation_selected_clean": "Hybrid",
    }
    for key in ("ridge_sourcepca_clean", "cvae_clean", "hybrid_validation_selected_clean"):
        main_rows.append(
            summary_row(
                comparator_names[key],
                "matched_internal_final_rerun_baseline",
                final_test["clean_base_comparators"][key]["summary"],
            )
        )
    main_rows.append(summary_row("LAMM", "matched_internal_final_rerun_external_baseline", lamm["means"]))
    for key, label in (("laplacian", "Laplacian"), ("bilaplacian", "bi-Laplacian"), ("arap", "ARAP")):
        main_rows.append(
            summary_row(label, "matched_internal_final_rerun_classical_baseline", classical["methods"][key]["summary"])
        )
    write_csv(out / "main_results_9900_pairs.csv", main_rows)

    validation_comparators = validation["clean_base_comparators"]
    ablation_rows = [
        {
            "method": "Ridge anchor",
            "component_state": "anchor_only_zero_residual",
            "roi_rmse": validation_comparators["ridge_sourcepca_clean"]["summary"]["roi_rmse"],
            "normal_flip_pct": validation_comparators["ridge_sourcepca_clean"]["summary"]["normal_flip_pct"],
            "edge_strain_p95": validation_comparators["ridge_sourcepca_clean"]["summary"]["edge_strain_p95"],
            "certificate_rate": 1.0,
            "selection_split": "validation",
        },
        {
            "method": "CVAE proposer",
            "component_state": "neural_proposer_without_rbsr_safety",
            "roi_rmse": validation_comparators["cvae_clean"]["summary"]["roi_rmse"],
            "normal_flip_pct": validation_comparators["cvae_clean"]["summary"]["normal_flip_pct"],
            "edge_strain_p95": validation_comparators["cvae_clean"]["summary"]["edge_strain_p95"],
            "certificate_rate": "",
            "selection_split": "validation",
        },
        {
            "method": "Fixed Hybrid",
            "component_state": "validation_selected_fixed_blend",
            "roi_rmse": validation_comparators["hybrid_validation_selected_clean"]["summary"]["roi_rmse"],
            "normal_flip_pct": validation_comparators["hybrid_validation_selected_clean"]["summary"]["normal_flip_pct"],
            "edge_strain_p95": validation_comparators["hybrid_validation_selected_clean"]["summary"]["edge_strain_p95"],
            "certificate_rate": "",
            "selection_split": "validation",
        },
        {
            "method": "Raw RB-SR gate",
            "component_state": "learned_gate_before_hard_certificate",
            "roi_rmse": validation["summary"]["roi_rmse"],
            "normal_flip_pct": validation["summary"]["normal_flip_pct"],
            "edge_strain_p95": validation["summary"]["edge_strain_p95"],
            "certificate_rate": "",
            "selection_split": "validation",
        },
        {
            "method": "Certified RB-SR",
            "component_state": "learned_gate_plus_ridge_fold_projection_attenuation_0p75",
            "roi_rmse": projection["selected"]["roi_rmse"],
            "normal_flip_pct": projection["selected"]["normal_flip_pct"],
            "edge_strain_p95": projection["selected"]["edge_strain_p95"],
            "certificate_rate": projection["selected"]["certificate_rate"],
            "selection_split": "validation",
        },
    ]
    write_csv(out / "component_ablation_validation_4830_pairs.csv", ablation_rows)

    primary_rows = [
        {
            "comparison": "Certified RB-SR minus Ridge",
            "metric": row["metric"],
            "mean_difference": row["mean_difference_method_minus_baseline"],
            "holm_p_two_sided": row["holm_p_two_sided"],
            "source_ci95_low": row["source_ci95_low"],
            "source_ci95_high": row["source_ci95_high"],
            "final_judgement": row["final_judgement"],
        }
        for row in primary_stats["rows"]
    ]
    write_csv(out / "primary_rbsr_vs_ridge_statistics.csv", primary_rows)

    supplemental_rows = []
    stats_by_metric = {row["metric"]: row for row in dominance_stats["rows"]}
    for metric in ("roi_rmse", "normal_flip_pct", "edge_strain_p95"):
        values = dominance["metrics"][metric]
        stats = stats_by_metric[metric]
        supplemental_rows.append(
            {
                "metric": metric,
                "calibrated_rbsr": values["rbsr"],
                "frozen_lamm": values["lamm"],
                "rbsr_minus_lamm": values["delta"],
                "relative_improvement_pct": -100.0 * values["delta"] / values["lamm"],
                "pair_win_rate": stats["pair_win_rate"],
                "holm_robust_p_two_sided": stats["holm_robust_p_two_sided"],
                "source_ci95_low": stats["source_ci95_low"],
                "source_ci95_high": stats["source_ci95_high"],
                "target_ci95_low": stats["target_ci95_low"],
                "target_ci95_high": stats["target_ci95_high"],
                "final_judgement": stats["final_judgement"],
            }
        )
    write_csv(out / "supplementary_rbsr_vs_lamm_9900_pairs.csv", supplemental_rows)

    summary = {
        "status": "FINAL_PAPER_TABLES_GENERATED_FROM_FROZEN_EVIDENCE",
        "paper_experiment_families": [
            {
                "role": "primary",
                "name": "certified RB-SR internal held-out final-rerun confirmation",
                "claim_boundary": "internal identity-disjoint held-out-from-final-retrain evidence; not historically untouched and not external",
            },
            {
                "role": "supplementary",
                "name": "PCA-128 calibrated RB-SR versus frozen LAMM",
                "claim_boundary": dominance["claim_boundary"],
            },
        ],
        "main_method_count": len(main_rows),
        "main_test_pairs": final_test["n_pairs"],
        "validation_ablation_pairs": projection["n_validation_pairs"],
        "supplementary_test_pairs": dominance["matched_ordered_test_pairs"],
        "source_sha256": {str(path.relative_to(root)): sha256_file(path) for path in source_paths},
        "outputs": [
            "main_results_9900_pairs.csv",
            "component_ablation_validation_4830_pairs.csv",
            "primary_rbsr_vs_ridge_statistics.csv",
            "supplementary_rbsr_vs_lamm_9900_pairs.csv",
        ],
    }
    summary_path = out / "FINAL_PAPER_TABLES.json"
    atomic_write_json(summary_path, summary)
    write_sha256_sidecar(summary_path)

    readme = f"""# Final paper tables

These tables are derived without retraining from the two frozen experiment families retained by the submission repository.

## Main internal final-rerun results

{markdown_table(main_rows, ['method', 'roi_rmse', 'normal_flip_pct', 'edge_strain_p95', 'abs_flip_pct', 'missed_flip_pct'])}

The main table is the clean internal identity-disjoint final rerun over the same 9,900 ordered test pairs. It is not an external or historically untouched confirmation.

## Validation-only component ablation

{markdown_table(ablation_rows, ['method', 'roi_rmse', 'normal_flip_pct', 'edge_strain_p95'])}

This component table is validation-only (4,830 ordered pairs). It shows the accuracy--safety role of the learned residual and the frozen ridge-fold certificate without reopening or retuning on test.

## Supplementary post-hoc LAMM comparison

The PCA-128 calibrated RB-SR result is supplementary post-hoc matched evidence on the same 9,900 ordered pairs. It must not be merged with the PCA-64 certified primary model or described as a blind confirmatory headline.
"""
    readme_path = out / "README.md"
    atomic_write_text(readme_path, readme)
    write_sha256_sidecar(readme_path)
    print(json.dumps({"out": str(out), "main_methods": len(main_rows), "status": summary["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
