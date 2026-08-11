from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rhinoform.reproduction import ReproductionPaths, validate_processed_facescape
from tools.audit_source_snapshots import audit_snapshot


def test_path_precedence_and_relative_resolution(tmp_path: Path) -> None:
    repo=tmp_path/"repo";repo.mkdir()
    config=repo/"local.toml"
    config.write_text('[paths]\ndata_root="from_toml"\nartifact_root="assets"\n')
    paths=ReproductionPaths.resolve(
        config=config,
        data_root=Path("from_cli"),
        environ={"RHINOFORM_DATA_ROOT":"from_env"},
        repo_root=repo,
    )
    assert paths.data_root == (repo/"from_cli").resolve()
    assert paths.artifact_root == (repo/"assets").resolve()
    assert paths.output_root == (repo/"reproduction_output").resolve()


def test_release_overlay_is_used_without_source_edits(tmp_path: Path) -> None:
    repo=tmp_path/"repo";overlay=tmp_path/"assets";repo.mkdir();overlay.mkdir()
    artifact=overlay/"results/example.json";artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    paths=ReproductionPaths.resolve(repo_root=repo,artifact_root=overlay,environ={})
    assert paths.locate("results/example.json") == artifact
    with pytest.raises(ValueError):
        paths.locate("../private.txt")


def test_processed_dataset_is_manifest_bound(tmp_path: Path) -> None:
    repo=tmp_path/"repo";data=tmp_path/"licensed";(repo/"data").mkdir(parents=True)
    mesh=data/"meshes/1_neutral.npz";mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"mesh")
    mesh_hash=hashlib.sha256(b"mesh").hexdigest()
    manifest={"n_rows":1,"rows":[{"npz_path":"meshes/1_neutral.npz","sha256":mesh_hash}]}
    payload=json.dumps(manifest,sort_keys=True)
    (repo/"data/manifest.json").write_text(payload)
    (data/"manifest.json").write_text(payload)
    paths=ReproductionPaths.resolve(repo_root=repo,data_root=data,environ={})
    assert validate_processed_facescape(paths,full_hash=True)["status"] == "PASS"


def test_all_repository_source_snapshots_are_exact() -> None:
    root=Path(__file__).resolve().parents[1]/"reproducibility/source_snapshots"
    reports=[audit_snapshot(path) for path in sorted(root.iterdir()) if path.is_dir()]
    assert {row["snapshot"] for row in reports} == {
        "final_holdout_v1","supplemental_dominance_v1","posthoc_subunits_v1"
    }
