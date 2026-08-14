#!/usr/bin/env python3
"""Map each final-report image to the repository evidence supporting it."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = Path(__file__).with_name("figure_asset_inventory.csv")
OUT = Path(__file__).with_name("figure_evidence_chain.csv")

SYSTEM = "demo/src/components/Studio3D.jsx;demo/src/engine/certified.js;demo/public/bundle/manifest.json"
METHOD = "rhinoform/train.py;rhinoform/cvae.py;rhinoform/train_rbsr_gate.py;scripts/evaluation/rbsr_ridge_fold_projection.py"
PROTOCOL = "splits/facescape_847/split_manifest.json;results/rbsr_final_rerun_holdout_v1/protocol/FINAL_RERUN_HOLDOUT_POLICY_FREEZE.json;scripts/evaluation/direct_paired_statistics.py"
MAIN = "docs/final_tables/main_results_9900_pairs.csv;docs/final_tables/primary_rbsr_vs_ridge_statistics.csv;docs/final_tables/supplementary_rbsr_vs_lamm_9900_pairs.csv"
COMPONENT = "docs/final_tables/component_ablation_validation_4830_pairs.csv;results/rbsr_final_rerun_holdout_v1/rbsr/seed20260609/base/validation_operating_point_candidates.csv"
GATE = "results/rbsr_final_rerun_holdout_v1/rbsr/seed20260609/gate/rbsr_gate_training.json;docs/final_tables/component_ablation_validation_4830_pairs.csv"
CERT = "results/rbsr_final_rerun_holdout_v1/rbsr/seed20260609/certified_projection_validation/rbsr_ridge_fold_projection_validation.csv;results/rbsr_final_rerun_holdout_v1/rbsr/seed20260609/certified_projection_validation/RBSR_CERTIFIED_PROJECTION_FREEZE.json"
SUBUNIT = "results/rbsr_final_rerun_holdout_v1/posthoc_subunit_analysis_v1/subunit_means_all_methods.csv;results/rbsr_final_rerun_holdout_v1/posthoc_subunit_analysis_v1/region_topology_audit.json"
SPATIAL = "results/rbsr_final_rerun_holdout_v1/posthoc_subunit_analysis_v1/spatial;results/rbsr_final_rerun_holdout_v1/posthoc_subunit_analysis_v1/figures/FIGURE_EVIDENCE.json"
QUAL = "results/rbsr_final_rerun_holdout_v1/posthoc_subunit_analysis_v1/qualitative/QUALITATIVE_CASE_SELECTION_FREEZE.json;results/rbsr_final_rerun_holdout_v1/posthoc_subunit_analysis_v1/qualitative/QUALITATIVE_CASES_EVIDENCE.json"
NOISE = "results/rbsr_final_rerun_holdout_v1/posthoc_subunit_analysis_v1/noise/noise_robustness_summary.csv;results/rbsr_final_rerun_holdout_v1/posthoc_subunit_analysis_v1/noise/STRICT_NOISE_ROBUSTNESS_EVIDENCE.json"
PERSONAL = "results/rbsr_final_rerun_holdout_v1/posthoc_subunit_analysis_v1/personalization/fixed_control_personalization_demo.npz.sha256.json;results/rbsr_final_rerun_holdout_v1/posthoc_subunit_analysis_v1/personalization/PERSONALIZATION_ABLATION_EVIDENCE.json"


def evidence_for(path: str) -> tuple[str, str, str]:
    name = Path(path).name
    if path == "imperial-logo.pdf":
        return "document template branding", "not a scientific claim", "template asset outside repository"
    if path.startswith("figures/classic/"):
        return "prior-work illustration", "bibliography entry and publisher record cited in the surrounding text", "external source; not a repository-generated result"
    if name in {"fig_1_1_clinical_gap.pdf", "fig_2_1_task_taxonomy.pdf"}:
        return "literature synthesis diagram", "bibliography_identifier_audit.csv and the cited works in Chapters 1--2", "citation-backed; exact composite absent from repository"
    if "fig_3_" in name:
        return "prototype/system design", SYSTEM, "implementation-backed; exact composite absent from repository"
    if name == "fig_4_1_task_geometry.pdf":
        return "task geometry", "demo/public/bundle/head_mean.json;data/manifest.json;splits/facescape_847/split_manifest.json", "data/topology-backed; exact composite absent from repository"
    if name in {"fig_4_2_rbsr_framework.pdf", "fig_4_3_learning_architecture.pdf"}:
        return "method architecture", METHOD, "implementation-backed; exact composite absent from repository"
    if name == "fig_4_4_stage1_anchor_evidence.pdf":
        return "stage-1 validation", COMPONENT, "numeric source committed; exact composite absent from repository"
    if name == "fig_4_5_stage2_residual_evidence.pdf":
        return "stage-2 validation sweep", COMPONENT, "numeric source committed; exact composite absent from repository"
    if name == "fig_4_6_stage3_gate_evidence.pdf":
        return "stage-3 gate validation", GATE, "history/summary committed; exact composite absent from repository"
    if name == "fig_4_7_stage3_spatial_mechanism.pdf":
        return "stage-3 spatial mechanism", GATE + ";" + SPATIAL, "aggregate/spatial sources committed; exact composite absent from repository"
    if name == "fig_4_8_stage4_certificate_evidence.pdf":
        return "stage-4 certificate validation", CERT, "numeric/protocol sources committed; exact composite absent from repository"
    if name == "fig_5_1_evaluation_protocol.pdf":
        return "evaluation protocol", PROTOCOL, "protocol-backed; exact composite absent from repository"
    if "eval_00_" in name:
        return "primary paired effects", MAIN, "numeric source committed and replayed; exact composite absent from repository"
    if "eval_01_" in name:
        return "operating-point comparison", MAIN, "numeric source committed and replayed; exact composite absent from repository"
    if "eval_02_" in name or name == "fig_a_3_subunit_strain.pdf":
        return "sub-unit analysis", SUBUNIT, "numeric/topology source committed; exact composite absent from repository"
    if "eval_03_" in name or "eval_04_" in name:
        return "spatial/tail analysis", SPATIAL, "NPZ/CSV and elementary figure evidence committed; exact composite absent from repository"
    if "eval_05_" in name:
        return "noise robustness", NOISE, "numeric/protocol source committed; exact composite absent from repository"
    if "eval_06_" in name or name == "fig_a_1_qualitative_all.pdf":
        return "deterministic qualitative cases", QUAL, "case freeze and per-case arrays committed; exact composite absent from repository"
    if "eval_07_" in name:
        return "fixed-control personalisation", PERSONAL, "evidence and digest committed; exact composite absent from repository"
    if name == "fig_a_6_learning_curve.pdf":
        return "training learning curve", "results/rbsr_final_rerun_holdout_v1/rbsr/seed20260609/gate/rbsr_gate_training.json", "training history committed; exact composite absent from repository"
    raise RuntimeError(f"unmapped figure: {path}")


def main() -> None:
    rows = []
    with INVENTORY.open(newline="", encoding="utf-8") as stream:
        inventory = list(csv.DictReader(stream))
    for asset in inventory:
        claim_scope, sources, status = evidence_for(asset["report_relative_path"])
        for source in sources.split(";"):
            if source and "external source" not in status and "template asset" not in status and not source.startswith("bibliography_"):
                candidate = ROOT / source
                if not candidate.exists():
                    raise FileNotFoundError(f"missing evidence source for {asset['report_relative_path']}: {source}")
        rows.append(
            {
                "report_relative_path": asset["report_relative_path"],
                "claim_scope": claim_scope,
                "repo_evidence_sources": sources,
                "evidence_status": status,
                "report_asset_sha256": asset["sha256"],
            }
        )
    fields = list(rows[0])
    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    exact = sum("exact composite absent" not in row["evidence_status"] for row in rows)
    print(f"mapped={len(rows)} exact_or_external_template={exact} project_composites_absent={len(rows)-exact}")


if __name__ == "__main__":
    main()
