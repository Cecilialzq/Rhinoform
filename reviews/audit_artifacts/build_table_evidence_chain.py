#!/usr/bin/env python3
"""Trace every final-report table to implementation or committed evidence."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEX = Path("/Users/lizequan/Desktop/FYP_revised/最终版report.tex")
OUT = Path(__file__).with_name("table_evidence_chain.csv")

MAP = {
    "tab:intro-reqs": ("requirements synthesis", "demo/src/App.jsx;demo/src/components/Studio.jsx", "implementation-backed synthesis"),
    "tab:rel-terms": ("terminology", "rhinoform/train.py;scripts/evaluation/rbsr_ridge_fold_projection.py;demo/src/engine/admission.js", "code-backed definitions"),
    "tab:rel-competitor": ("external-tool positioning", "bibliography_identifier_audit.csv", "external literature/vendor evidence; not a repo result"),
    "tab:rel-synthesis": ("related-work synthesis", "bibliography_identifier_audit.csv", "external literature synthesis; not a repo result"),
    "tab:proto-reqs": ("prototype requirements", "demo/src/App.jsx;demo/src/components/Studio.jsx;docs/final_tables/main_results_9900_pairs.csv", "implementation and result-backed"),
    "tab:proto-semantic-map": ("semantic control map", "demo/public/bundle/ridge_browser.json;demo/src/engine/geometry.js", "released-bundle and implementation-backed"),
    "tab:architecture-tensor-contract": ("tensor contract", "demo/public/bundle/ridge_browser.json;results/rbsr_final_rerun_holdout_v1/rbsr/seed20260609/base/neural_field_metadata_cvae_ew0p1_lw0.json;results/rbsr_final_rerun_holdout_v1/rbsr/seed20260609/gate/rbsr_gate_training.json", "package/implementation-backed"),
    "tab:algorithm-rbsr": ("method algorithm", "rhinoform/train.py;rhinoform/cvae.py;rhinoform/train_rbsr_gate.py;scripts/evaluation/rbsr_ridge_fold_projection.py", "implementation-backed"),
    "tab:architecture-ridge-anchor": ("Ridge architecture", "rhinoform/train.py;demo/public/bundle/ridge_browser.json", "implementation/bundle-backed"),
    "tab:architecture-cvae-detailed": ("CVAE architecture", "rhinoform/cvae.py;results/rbsr_final_rerun_holdout_v1/rbsr/seed20260609/base/neural_field_metadata_cvae_ew0p1_lw0.json", "implementation/package-backed"),
    "tab:architecture-rbsr-gate": ("gate architecture", "rhinoform/train_rbsr_gate.py;results/rbsr_final_rerun_holdout_v1/rbsr/seed20260609/gate/rbsr_gate_training.json", "implementation/package-backed"),
    "tab:method-rationale": ("method taxonomy", "rhinoform/baselines.py;rhinoform/cvae.py;scripts/evaluation/rbsr_ridge_fold_projection.py;results/rbsr_final_rerun_holdout_v1/lamm/seed20260609/protocol_audit.json", "implementation/protocol-backed"),
    "tab:evaluation-metrics-definitions": ("metric definitions", "rhinoform/geometry.py;rhinoform/confirmation.py;scripts/evaluation/direct_paired_statistics.py", "scorer-backed"),
    "tab:evaluation-statistical-protocol": ("statistical protocol", "rhinoform/stats.py;scripts/evaluation/direct_paired_statistics.py;results/rbsr_final_rerun_holdout_v1/protocol/FINAL_RERUN_HOLDOUT_POLICY_FREEZE.json", "code/protocol-backed"),
    "tab:m7-main": ("main benchmark", "docs/final_tables/main_results_9900_pairs.csv", "committed summary; replayed from pair records"),
    "tab:pca128": ("post-hoc capacity study", "docs/final_tables/supplementary_rbsr_vs_lamm_9900_pairs.csv", "committed summary; replayed from pair records"),
    "tab:m7-subunit-rmse": ("sub-unit accuracy", "results/rbsr_final_rerun_holdout_v1/posthoc_subunit_analysis_v1/subunit_means_all_methods.csv", "committed post-hoc summary"),
    "tab:m7-subunit-flip": ("sub-unit flips", "results/rbsr_final_rerun_holdout_v1/posthoc_subunit_analysis_v1/subunit_means_all_methods.csv", "committed post-hoc summary"),
    "tab:eval-noise-unified": ("noise robustness", "results/rbsr_final_rerun_holdout_v1/posthoc_subunit_analysis_v1/noise/noise_robustness_summary.csv;results/rbsr_final_rerun_holdout_v1/posthoc_subunit_analysis_v1/noise/STRICT_NOISE_ROBUSTNESS_EVIDENCE.json", "committed post-hoc summary/protocol"),
    "tab:eval-failure-cases": ("failure modes", "results/rbsr_final_rerun_holdout_v1/posthoc_subunit_analysis_v1/qualitative/QUALITATIVE_CASES_EVIDENCE.json;results/rbsr_final_rerun_holdout_v1/posthoc_subunit_analysis_v1/noise/STRICT_NOISE_ROBUSTNESS_EVIDENCE.json;demo/src/engine/admission.js", "result and implementation-backed synthesis"),
    "tab:eval-metric-map": ("benchmark/runtime metric roles", "rhinoform/geometry.py;demo/src/engine/geometry.js;demo/src/engine/admission.js;demo/src/engine/certified.js", "scorer/runtime implementation-backed"),
    "tab:eval-compute": ("compute and release constants", "results/rbsr_final_rerun_holdout_v1/rbsr/seed20260609/base/neural_field_metadata_cvae_ew0p1_lw0.json;results/rbsr_final_rerun_holdout_v1/rbsr/seed20260609/gate/rbsr_gate_training.json;demo/public/bundle/ridge_browser.json;splits/facescape_847/split_manifest.json", "committed metadata/bundle-backed"),
    "tab:architecture-parameter-values": ("architecture/training constants", "results/rbsr_final_rerun_holdout_v1/rbsr/seed20260609/base/neural_field_metadata_cvae_ew0p1_lw0.json;results/rbsr_final_rerun_holdout_v1/rbsr/seed20260609/gate/rbsr_gate_training.json;results/rbsr_final_rerun_holdout_v1/rbsr/seed20260609/certified_projection_validation/RBSR_CERTIFIED_PROJECTION_FREEZE.json;splits/facescape_847/split_manifest.json", "committed package/protocol-backed"),
}


def clean(value: str) -> str:
    return " ".join(re.sub(r"\\[A-Za-z]+|[{}~$]", " ", value).split())


def main() -> None:
    source = TEX.read_text(encoding="utf-8")
    rows = []
    blocks = re.findall(r"\\begin\{table\}.*?\\end\{table\}", source, re.S)
    for block in blocks:
        labels = re.findall(r"\\label\{([^}]+)\}", block)
        if len(labels) != 1:
            raise RuntimeError(f"table must have exactly one label: {labels}")
        label = labels[0]
        caption = re.search(r"\\caption(?:\[[^]]*\])?\{(.*?)\}\s*\\label", block, re.S)
        if label not in MAP:
            raise RuntimeError(f"unmapped table: {label}")
        scope, sources, status = MAP[label]
        if not sources.startswith("bibliography_"):
            for item in sources.split(";"):
                if not (ROOT / item).exists():
                    raise FileNotFoundError(f"missing evidence for {label}: {item}")
        rows.append({
            "table_label": label,
            "caption": clean(caption.group(1)) if caption else "",
            "claim_scope": scope,
            "repo_evidence_sources": sources,
            "evidence_status": status,
        })
    if set(MAP) != {row["table_label"] for row in rows}:
        raise RuntimeError("mapping contains labels not present in report")
    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"tables={len(rows)} mapped={len(rows)}")


if __name__ == "__main__":
    main()
