from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluation/posthoc_personalization_ablation.py"
RESUME_ERRATUM = ROOT / "scripts/evaluation/posthoc_personalization_resume_erratum.py"
NOTEBOOK = ROOT / "notebooks/Rhinoform_final_rerun_posthoc_subunits_colab.ipynb"


def test_personalization_ablation_contract():
    text = SCRIPT.read_text(encoding="utf-8")
    for value in (
        "controls_only_ridge",
        "source_pca_ridge",
        "mean_source_code",
        "shuffled_source_code",
        "ridge_ctrl",
        "4830",
        "validation",
        "fixed_cyclic_derangement",
        "primary_rmse_family_size",
        "secondary_geometry_family_size",
        "source_ci95_low",
        "target_ci95_low",
        "operating_point_reselected",
        "fixed_control_personalization_demo.npz",
        "same control vector across four deterministic source identities",
        "no_ground_truth_accuracy_claim",
    ):
        assert value in text
    assert "test_pairs" not in text
    assert "FYP final" not in text


def test_notebook_runs_personalization_ablation_and_limits_claims():
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in data["cells"])
    for value in (
        "RUN_PERSONALIZATION_ABLATION = True",
        "posthoc_personalization_resume_erratum.py",
        "controls-only Ridge",
        "source-PCA Ridge",
        "mean-source-code",
        "shuffled-source-code",
        "geometrically natural/regular",
        "must not be called clinically preferred",
        "strain overlay",
        "landmark edit",
    ):
        assert value in text


def test_personalization_resume_uses_metric_specific_canonical_tolerances():
    from scripts.evaluation.posthoc_personalization_resume_erratum import (
        CANONICAL_REPLAY_ATOL,
        canonical_metric_close,
    )

    assert CANONICAL_REPLAY_ATOL == {
        "roi_rmse": 1e-7,
        "normal_flip_pct": 1e-12,
        "edge_strain_p95": 1e-5,
    }
    assert canonical_metric_close("roi_rmse", 0.9 + 5e-8, 0.9)
    assert not canonical_metric_close("roi_rmse", 0.9 + 3e-6, 0.9)
    assert canonical_metric_close("normal_flip_pct", 0.0, 0.0)
    assert not canonical_metric_close("normal_flip_pct", 1e-8, 0.0)
    assert canonical_metric_close("edge_strain_p95", 0.3 + 5e-6, 0.3)
    assert not canonical_metric_close("edge_strain_p95", 0.3 + 2e-5, 0.3)

    frozen_text = SCRIPT.read_bytes()
    import hashlib

    assert hashlib.sha256(frozen_text).hexdigest() == (
        "02519255bbdcd1c709490995ffc905e50e3eddc2c7c9ca751904973c580c6b77"
    )
    erratum_text = RESUME_ERRATUM.read_text(encoding="utf-8")
    assert "audit_complete_canonical_replay" in erratum_text
    assert "PRE_AGGREGATION_CANONICAL_REPLAY_TOLERANCE_ERRATUM" in erratum_text


def test_complete_canonical_replay_audit_accepts_only_bounded_strain_drift(tmp_path):
    import pytest

    from rhinoform.repro import (
        atomic_write_csv,
        atomic_write_json,
        valid_sha256_sidecar,
        write_sha256_sidecar,
    )
    from scripts.evaluation.posthoc_personalization_resume_erratum import (
        audit_complete_canonical_replay,
    )

    canonical = []
    observed = []
    for index in range(4830):
        row = {
            "source_id": str(index // 69),
            "target_id": str(index % 69),
            "roi_rmse": "0.9",
            "normal_flip_pct": "0.0",
            "edge_strain_p95": "0.3",
        }
        canonical.append(row)
        observed.append({
            **row,
            "edge_strain_p95": "0.300005" if index == 1200 else "0.3",
            "validation_pair_index": index,
            "variant": "source_pca_ridge",
        })

    canonical_path = tmp_path / "canonical.csv"
    chunk_path = tmp_path / "personalization/chunks/chunk_00000_04830_source_pca_ridge.csv"
    erratum_path = tmp_path / "personalization/erratum.json"
    atomic_write_csv(canonical_path, canonical)
    atomic_write_csv(chunk_path, observed)
    atomic_write_json(erratum_path, {"status": "TEST"})
    for path in (canonical_path, chunk_path, erratum_path):
        write_sha256_sidecar(path)

    audit_path = audit_complete_canonical_replay(tmp_path, canonical_path, erratum_path)
    assert valid_sha256_sidecar(audit_path)

    observed[1200]["edge_strain_p95"] = "0.30002"
    atomic_write_csv(chunk_path, observed)
    with pytest.raises(AssertionError, match="edge_strain_p95"):
        audit_complete_canonical_replay(tmp_path, canonical_path, erratum_path)
