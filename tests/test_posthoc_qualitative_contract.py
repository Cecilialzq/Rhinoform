from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluation/posthoc_qualitative_cases.py"
NOTEBOOK = ROOT / "notebooks/Rhinoform_final_rerun_posthoc_subunits_colab.ipynb"


def test_qualitative_selection_and_evidence_contract():
    text = SCRIPT.read_text(encoding="utf-8")
    for value in (
        "median_case",
        "small_control",
        "large_control",
        "high_certificate_iterations",
        "largest_rbsr_gain_over_ridge",
        "worst_rbsr_failure_vs_best_comparator",
        "QUALITATIVE_CASE_SELECTION_FREEZE.json",
        "QUALITATIVE_CASES_EVIDENCE.json",
        "new_flip_faces",
        "vertex_error",
        "retention",
        "iterations",
        "edge_strain",
        "geometrically natural/regular",
        "must not be called clinically preferred",
    ):
        assert value in text
    assert "random" not in text.lower()
    assert "FYP final" not in text
    assert "dense_results" not in text


def test_notebook_runs_and_renders_qualitative_cases():
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in data["cells"])
    for value in (
        "RUN_QUALITATIVE_CASES = True",
        "posthoc_qualitative_cases.py",
        "worst real failure",
        "shared error scale",
        "new-flip",
        "QUALITATIVE_CASES_EVIDENCE.json",
    ):
        assert value in text
