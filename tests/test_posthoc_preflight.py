from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.evaluation.posthoc_preflight import (
    KNOWN_POST_FREEZE_IMPLEMENTATION_HASHES,
    discover_facescape_data_root,
    validate_implementation_lineage,
)


def _fake_dataset(root: Path, count: int = 846) -> Path:
    (root / "meshes").mkdir(parents=True)
    (root / "manifest.json").write_text('{"schema": "test"}\n')
    for index in range(count):
        (root / "meshes" / f"{index:04d}.npz").touch()
    return root


def test_discovers_first_complete_manifest_matched_dataset(tmp_path: Path):
    incomplete = _fake_dataset(tmp_path / "incomplete", count=845)
    complete = _fake_dataset(tmp_path / "complete")
    manifest_sha = __import__("hashlib").sha256(
        (complete / "manifest.json").read_bytes()
    ).hexdigest()
    assert discover_facescape_data_root(
        [incomplete, complete], expected_manifest_sha256=manifest_sha
    ) == complete.resolve()


def test_rejects_missing_or_wrong_dataset(tmp_path: Path):
    wrong = _fake_dataset(tmp_path / "wrong")
    with pytest.raises(RuntimeError, match="846 processed FaceScape meshes"):
        discover_facescape_data_root(
            [tmp_path / "missing", wrong], expected_manifest_sha256="0" * 64
        )


def test_preflight_script_is_read_only_and_checks_full_stack():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/evaluation/posthoc_preflight.py").read_text()
    for value in (
        "COMPLETE", "test_access_count", "implementation_hashes",
        "prepare_neural", "prepare_lamm", "torch.cuda.is_available",
        "9900", "4830", "POSTHOC PREFLIGHT PASS",
        "zero-noise/canonical replay", "KNOWN_POST_FREEZE_IMPLEMENTATION_HASHES",
        "LAMM_RUNTIME_MODULES", "trimesh", "yaml", "einops", "timm",
    ):
        assert value in text
    for forbidden in ("atomic_write", "write_sha256_sidecar", "unlink(", "rmtree("):
        assert forbidden not in text
    protocol = (root / "scripts/evaluation/posthoc_subunit_analysis.py").read_text()
    assert '"original_final_implementation_hashes"' in protocol
    assert '"rhinoform/rbsr_calibration.py"' in protocol
    assert '"rhinoform/train_rbsr_gate.py"' in protocol
    assert '"requirements-lamm-inference.txt"' in protocol
    noise = (root / "scripts/evaluation/posthoc_strict_noise_robustness.py").read_text()
    for implementation in (protocol, noise):
        assert "relocate_comparable_model_config" in implementation
        assert "require_frozen_result_artifact" in implementation


def test_preflight_forces_its_repository_ahead_of_external_scripts_package(
    tmp_path: Path,
):
    root = Path(__file__).resolve().parents[1]
    external = tmp_path / "external_checkout"
    (external / "scripts").mkdir(parents=True)
    (external / "scripts/__init__.py").write_text("ORIGIN = 'external'\n")
    probe = (
        "import runpy,sys; "
        f"runpy.run_path({str(root / 'scripts/evaluation/posthoc_preflight.py')!r}, "
        "run_name='_posthoc_preflight_probe'); "
        "import scripts; print(scripts.__file__)"
    )
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(external), str(root))),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        cwd=tmp_path,
    )
    assert Path(completed.stdout.strip()).resolve() == (root / "scripts/__init__.py").resolve()


def test_implementation_lineage_accepts_only_exact_known_dual_chain():
    frozen = {"unchanged.py": "a", **{name: "old" for name in KNOWN_POST_FREEZE_IMPLEMENTATION_HASHES}}
    actual = {"unchanged.py": "a", **KNOWN_POST_FREEZE_IMPLEMENTATION_HASHES}
    assert set(validate_implementation_lineage(frozen, actual)) == set(
        KNOWN_POST_FREEZE_IMPLEMENTATION_HASHES
    )
    with pytest.raises(RuntimeError, match="Unexpected implementation lineage"):
        validate_implementation_lineage(frozen, {**actual, "unchanged.py": "new"})
    first = next(iter(KNOWN_POST_FREEZE_IMPLEMENTATION_HASHES))
    with pytest.raises(RuntimeError, match="Unexpected implementation lineage"):
        validate_implementation_lineage(frozen, {**actual, first: "unapproved"})
