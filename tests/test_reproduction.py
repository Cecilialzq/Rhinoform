from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rhinoform.reproduction import ReproductionPaths, validate_processed_facescape
from tools.reproduce import parser, verify_frozen_inputs


def test_public_cli_contains_only_portable_commands() -> None:
    for command in ("show-paths", "verify", "replay", "preflight"):
        assert parser().parse_args([command]).command == command


def test_path_precedence_and_relative_resolution(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "local.toml"
    config.write_text('[paths]\ndata_root="from_toml"\n', encoding="utf-8")
    paths = ReproductionPaths.resolve(
        config=config,
        data_root=Path("from_cli"),
        environ={"RHINOFORM_DATA_ROOT": "from_env"},
        repo_root=repo,
    )
    assert paths.data_root == (repo / "from_cli").resolve()
    assert paths.public_dict() == {
        "repo_root": str(repo.resolve()),
        "data_root": str((repo / "from_cli").resolve()),
    }


def test_processed_dataset_is_manifest_bound(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    data = tmp_path / "licensed"
    (repo / "data").mkdir(parents=True)
    mesh = data / "meshes/1_neutral.npz"
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"mesh")
    mesh_hash = hashlib.sha256(b"mesh").hexdigest()
    manifest = {
        "n_rows": 1,
        "rows": [{"npz_path": "meshes/1_neutral.npz", "sha256": mesh_hash}],
    }
    payload = json.dumps(manifest, sort_keys=True)
    (repo / "data/manifest.json").write_text(payload, encoding="utf-8")
    (data / "manifest.json").write_text(payload, encoding="utf-8")
    paths = ReproductionPaths.resolve(repo_root=repo, data_root=data, environ={})
    assert validate_processed_facescape(paths, full_hash=True)["status"] == "PASS"


def test_current_frozen_inputs_have_valid_sidecars() -> None:
    paths = ReproductionPaths.resolve(environ={})
    assert len(verify_frozen_inputs(paths)) == 10
