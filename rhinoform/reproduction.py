"""Path-independent configuration and preflight helpers for public reproduction."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


ENVIRONMENT_KEYS = {
    "data_root": "RHINOFORM_DATA_ROOT",
    "artifact_root": "RHINOFORM_ARTIFACT_ROOT",
    "output_root": "RHINOFORM_OUTPUT_ROOT",
    "lamm_root": "RHINOFORM_LAMM_ROOT",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Reproduction configuration not found: {config_path}")
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)
    paths = payload.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("The reproduction TOML must contain a [paths] table")
    return paths


def _resolve_path(value: str | os.PathLike[str] | None, *, base: Path) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


@dataclass(frozen=True)
class ReproductionPaths:
    """All machine-dependent paths used by the public reproduction wrapper."""

    repo_root: Path
    data_root: Path | None
    artifact_root: Path
    output_root: Path
    lamm_root: Path | None

    @classmethod
    def resolve(
        cls,
        *,
        config: Path | None = None,
        data_root: Path | None = None,
        artifact_root: Path | None = None,
        output_root: Path | None = None,
        lamm_root: Path | None = None,
        environ: dict[str, str] | None = None,
        repo_root: Path | None = None,
    ) -> "ReproductionPaths":
        root = Path(repo_root or repository_root()).resolve()
        table = _load_config(config)
        env = os.environ if environ is None else environ

        def choose(name: str, override: Path | None, default: str | None) -> Path | None:
            value: str | os.PathLike[str] | None = override
            if value is None:
                value = env.get(ENVIRONMENT_KEYS[name])
            if value is None:
                value = table.get(name)
            if value is None:
                value = default
            return _resolve_path(value, base=root)

        resolved_artifacts = choose("artifact_root", artifact_root, ".")
        resolved_output = choose("output_root", output_root, "reproduction_output")
        assert resolved_artifacts is not None and resolved_output is not None
        return cls(
            repo_root=root,
            data_root=choose("data_root", data_root, None),
            artifact_root=resolved_artifacts,
            output_root=resolved_output,
            lamm_root=choose("lamm_root", lamm_root, None),
        )

    def locate(self, relative: str | Path) -> Path:
        """Resolve a repository-relative artifact from Git or a release overlay."""
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Unsafe repository-relative path: {relative}")
        repository_candidate = self.repo_root / rel
        if repository_candidate.is_file():
            return repository_candidate
        overlay_candidate = self.artifact_root / rel
        if overlay_candidate.is_file():
            return overlay_candidate
        raise FileNotFoundError(
            f"Required artifact is absent from both the Git checkout and release overlay: {rel}\n"
            f"  checkout: {repository_candidate}\n  overlay: {overlay_candidate}"
        )

    def public_dict(self) -> dict[str, str | None]:
        return {
            "repo_root": str(self.repo_root),
            "data_root": None if self.data_root is None else str(self.data_root),
            "artifact_root": str(self.artifact_root),
            "output_root": str(self.output_root),
            "lamm_root": None if self.lamm_root is None else str(self.lamm_root),
        }


def validate_processed_facescape(
    paths: ReproductionPaths, *, full_hash: bool = False
) -> dict[str, Any]:
    """Validate a separately licensed processed FaceScape directory."""
    if paths.data_root is None:
        raise RuntimeError(
            "FaceScape is not distributed with Rhinoform. Set --data-root, "
            "RHINOFORM_DATA_ROOT, or [paths].data_root in reproduction.local.toml."
        )
    canonical_manifest = paths.repo_root / "data/manifest.json"
    supplied_manifest = paths.data_root / "manifest.json"
    if not canonical_manifest.is_file() or not supplied_manifest.is_file():
        raise FileNotFoundError(
            f"Both manifests are required: {canonical_manifest} and {supplied_manifest}"
        )
    canonical_hash = sha256_file(canonical_manifest)
    supplied_hash = sha256_file(supplied_manifest)
    if supplied_hash != canonical_hash:
        raise RuntimeError(
            "Processed FaceScape manifest mismatch. Use the repository preprocessing "
            "contract and do not edit split or mesh metadata."
        )
    manifest = json.loads(canonical_manifest.read_text(encoding="utf-8"))
    rows = manifest.get("rows", [])
    if len(rows) != int(manifest.get("n_rows", -1)):
        raise RuntimeError("Canonical data manifest has an inconsistent row count")
    missing: list[str] = []
    mismatched: list[str] = []
    for row in rows:
        relative = Path(str(row["npz_path"]))
        mesh = paths.data_root / relative
        if not mesh.is_file():
            missing.append(relative.as_posix())
        elif full_hash and sha256_file(mesh) != str(row["sha256"]):
            mismatched.append(relative.as_posix())
    if missing or mismatched:
        raise RuntimeError(
            f"Processed FaceScape validation failed: missing={missing[:10]} "
            f"mismatched={mismatched[:10]}"
        )
    return {
        "status": "PASS",
        "manifest_sha256": canonical_hash,
        "mesh_count": len(rows),
        "full_mesh_hash_check": bool(full_hash),
        "data_root": str(paths.data_root),
    }


def validate_release_assets(paths: ReproductionPaths) -> dict[str, Any]:
    """Validate the separately downloaded checkpoint bundle by relative path."""
    manifest_path = paths.repo_root / "reproducibility/RELEASE_ASSET_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked: list[dict[str, Any]] = []
    for row in manifest["artifacts"]:
        artifact = paths.locate(row["path"])
        actual_bytes = artifact.stat().st_size
        actual_hash = sha256_file(artifact)
        if actual_bytes != int(row["bytes"]) or actual_hash != row["sha256"]:
            raise RuntimeError(
                f"Release artifact mismatch: {artifact}\n"
                f"expected bytes={row['bytes']} sha256={row['sha256']}\n"
                f"actual bytes={actual_bytes} sha256={actual_hash}"
            )
        checked.append({"path": row["path"], "bytes": actual_bytes})
    return {
        "status": "PASS",
        "manifest_sha256": sha256_file(manifest_path),
        "artifacts_checked": len(checked),
        "total_bytes": sum(row["bytes"] for row in checked),
        "artifact_root": str(paths.artifact_root),
    }
