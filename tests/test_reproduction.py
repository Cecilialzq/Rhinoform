from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from rhinoform.reproduction import ReproductionPaths, validate_processed_facescape
from rhinoform.source_runtime import (
    materialize_source_runtime,
    run_runtime_module,
    verify_runtime_import_origins,
)
from tools.audit_source_snapshots import audit_snapshot
from tools.reproduce import parser


def _write_sidecar(path: Path) -> None:
    payload = path.read_bytes()
    sidecar = {
        "path": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    path.with_name(path.name + ".sha256.json").write_text(
        json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
    )


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_materialized_runtime_uses_tagged_base_and_archived_source(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "rhinoform").mkdir(parents=True)
    (repo / "rhinoform/__init__.py").write_text("", encoding="utf-8")
    (repo / "rhinoform/data.py").write_text("VERSION='tagged-base'\n", encoding="utf-8")
    (repo / "rhinoform/rbsr_calibration.py").write_text(
        "VERSION='root-at-tag'\n", encoding="utf-8"
    )
    archived = (
        repo
        / "reproducibility/source_snapshots/final_holdout_v1/source/rhinoform/rbsr_calibration.py"
    )
    archived.parent.mkdir(parents=True)
    archived.write_text(
        "from rhinoform import data\nVERSION='archived-v1:' + data.VERSION\n",
        encoding="utf-8",
    )
    payload = archived.read_bytes()
    manifest = archived.parents[2] / "SOURCE_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "rhinoform_source_snapshot_v1",
                "status": "COMPLETE_HASH_VERIFIED_SOURCE_SNAPSHOT",
                "expected_git_tag": "final-holdout-v1",
                "files": [
                    {
                        "path": "rhinoform/rbsr_calibration.py",
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_sidecar(manifest)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "reviewer@example.org")
    _git(repo, "config", "user.name", "Reviewer")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "tagged source base")
    _git(repo, "tag", "-a", "final-holdout-v1", "-m", "frozen source")

    # A later root edit must never leak into the historical runtime.
    (repo / "rhinoform/data.py").write_text("VERSION='current-root'\n", encoding="utf-8")
    destination = tmp_path / "runtime"
    report = materialize_source_runtime(
        repo_root=repo,
        snapshot_name="final_holdout_v1",
        destination=destination,
    )

    assert report["status"] == "PASS_ISOLATED_SOURCE_RUNTIME"
    assert (destination / "rhinoform/rbsr_calibration.py").read_text() == (
        "from rhinoform import data\nVERSION='archived-v1:' + data.VERSION\n"
    )
    assert (destination / "rhinoform/data.py").read_text() == "VERSION='tagged-base'\n"
    imports = verify_runtime_import_origins(
        runtime_root=destination,
        modules={"rhinoform.rbsr_calibration": hashlib.sha256(payload).hexdigest()},
    )
    assert imports["status"] == "PASS_ISOLATED_IMPORTS"
    assert imports["modules"]["rhinoform.rbsr_calibration"]["origin"].startswith(
        str(destination)
    )


def test_materialize_cli_defaults_to_final_holdout() -> None:
    args = parser().parse_args(["materialize"])
    assert args.experiment == "final-holdout"
    assert args.runtime_root is None
    assert args.source_variant is None

    runtime_args = parser().parse_args(
        [
            "runtime-exec",
            "--runtime-root",
            "/tmp/runtime",
            "--module",
            "rhinoform.train",
            "--module-args",
            "--help",
        ]
    )
    assert runtime_args.module == "rhinoform.train"
    assert runtime_args.runtime_arguments == ["--help"]


def test_runtime_module_execution_cannot_import_live_checkout(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    package = runtime / "rhinoform"
    package.mkdir(parents=True)
    (runtime / "RUNTIME_SOURCE_LOCK.json").write_text("{}\n", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "data.py").write_text("VERSION='isolated'\n", encoding="utf-8")
    (package / "entry.py").write_text(
        "from pathlib import Path\n"
        "from rhinoform import data\n"
        "import sys\n"
        "Path(sys.argv[1]).write_text(data.VERSION)\n",
        encoding="utf-8",
    )
    output = tmp_path / "result.txt"
    completed = run_runtime_module(
        runtime_root=runtime,
        module="rhinoform.entry",
        arguments=[str(output)],
        capture_output=True,
    )
    assert completed.returncode == 0
    assert output.read_text(encoding="utf-8") == "isolated"


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
