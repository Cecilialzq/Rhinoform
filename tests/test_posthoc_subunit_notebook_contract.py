from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/"notebooks/Rhinoform_final_rerun_posthoc_subunits_colab.ipynb"


def test_notebook_contract():
    data=json.loads(NOTEBOOK.read_text()); text="\n".join("".join(c.get("source",[])) for c in data["cells"])
    for value in (
        "RUN_POSTHOC_SUBUNITS = True", "RUN_NOISE_ROBUSTNESS = True", "secondary post-hoc", "legacy three-vertex majority vote",
        "both endpoints", "175", "25", "posthoc_strict_noise_resume_erratum.py",
        "target-relative new flip", "old absolute-flip/nonzero-landmark", "spatial_diagnostics",
        "flip-frequency", "dpi=300", ".pdf", ".png",
    ): assert value in text
    assert "landmark RMSE" in text and "target flip" in text and "descriptive" in text
    for value in ("4,830", "0.25", "0.5", "1.0 mm", "36", "Certified RB-SR, Ridge, LAMM and ARAP"):
        assert value in text
    assert "2,000" not in text
    assert "8×3×9" not in text
    assert "RESULTS=CANONICAL_REPO/'results/rbsr_final_rerun_holdout_v1'" in text
    assert "OUT=RESULTS/'posthoc_subunit_analysis_v1'" in text
    assert "RHINOFORM_FACESCAPE_DATA_ROOT" in text
    assert "Path('/content/drive/MyDrive/FYP final/data')" in text
    assert "LEGACY_LAMM_ARTIFACTS=Path('/content/drive/MyDrive/FYP final/results/rbsr_final_rerun_holdout_v1/lamm/seed20260609')" in text
    assert "RHINOFORM_LAMM_ARTIFACT_ROOT" in text
    assert "LAMM artifact source:" in text
    assert "posthoc_preflight.py" in text
    assert "REPO/'tests/test_posthoc_preflight.py'" in text
    assert "POSTHOC PREFLIGHT PASS" in text
    assert "for directory in ('rhinoform','scripts','experiments','roi','tests','tools','notebooks')" in text
    assert "REPO/'notebooks/Rhinoform_final_rerun_posthoc_subunits_colab.ipynb'" in text
    assert "REPO/'data/manifest.json'" in text
    assert "shutil.copy2(CANONICAL_REPO/'data/manifest.json',REPO/'data/manifest.json')" in text
    assert "requirements-lamm-inference.txt" in text
    assert "LAMM_RUNTIME=Path('/content/rhinoform_lamm_inference_runtime')" in text
    assert "'--no-deps','--target',LAMM_RUNTIME" in text
    assert "from models import LAMM" in text
    assert "LAMM dependency bootstrap PASS" in text
    assert "def run_live(cmd, *, env=None):" in text
    assert "lamm_probe_env={**LIVE_ENV}" in text
    assert "run_live([" in text and "],env=lamm_probe_env)" in text
    assert "LIVE_ENV['PYTHONPATH']=str(LAMM_ROOT)" not in text
    assert "LAMM_ROOT/'requirements.txt'" not in text
    assert "torch==2.0.1" not in text and "numpy==1.23.4" not in text
    assert "kernel and subprocess import directly from runtime overlay" in text
    assert "Path(rhinoform.__file__).resolve()" in text
    assert "if not (REPO/'rhinoform/repro.py').is_file():" not in text
    # Author convenience: FaceScape plus the exact LAMM artifact mount used by
    # the successful final rerun. Outputs still go only to the GitHub-ready repo.
    assert text.count("/content/drive/MyDrive/FYP final") == 2
    assert "DENSE_RESULTS" not in text
    assert "--dense-results" not in text
    assert "rhinoform.train" not in text and "train_autoencoder" not in text
    for stage in ("audit","neural","classical","lamm","aggregate"):
        assert f"run_stage('{stage}')" in text


def test_all_posthoc_code_cells_compile():
    data = json.loads(NOTEBOOK.read_text())
    for index, cell in enumerate(data["cells"]):
        if cell.get("cell_type") == "code":
            compile(
                "".join(cell.get("source", [])),
                f"{NOTEBOOK.name}:cell-{index}",
                "exec",
            )


def test_posthoc_runner_has_no_private_archive_dependency():
    text=(ROOT/"scripts/evaluation/posthoc_subunit_analysis.py").read_text()
    assert "dense_results" not in text
    assert "--dense-results" not in text


def test_lamm_inference_requirements_are_minimal_and_pinned():
    lines = {
        line.strip()
        for line in (ROOT / "requirements-lamm-inference.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert lines == {
        "trimesh==3.22.3",
        "PyYAML==6.0.2",
        "einops==0.6.1",
        "timm==0.9.2",
    }
    assert not any(line.lower().startswith(("torch", "torchvision", "numpy")) for line in lines)
