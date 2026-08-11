"""Materialise a historical source snapshot into an isolated runtime tree."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any


RUNTIME_DIRECTORIES = ("rhinoform", "scripts", "experiments", "configs")
RUNTIME_ROOT_FILES = (
    "pyproject.toml",
    "requirements.txt",
    "requirements-ci.txt",
    "requirements-replay.txt",
    "requirements-lamm-inference.txt",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_manifest(snapshot: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = snapshot / "SOURCE_MANIFEST.json"
    sidecar_path = manifest_path.with_name(manifest_path.name + ".sha256.json")
    if not manifest_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"Source manifest or sidecar is missing: {manifest_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if (
        sidecar.get("path") != manifest_path.name
        or int(sidecar.get("bytes", -1)) != manifest_path.stat().st_size
        or sidecar.get("sha256") != _sha256_file(manifest_path)
    ):
        raise RuntimeError(f"Invalid source manifest sidecar: {sidecar_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "rhinoform_source_snapshot_v1":
        raise RuntimeError(f"Unsupported source manifest schema: {manifest_path}")
    return manifest_path, manifest


def _snapshot_rows(manifest: dict[str, Any], variant: str | None) -> tuple[str, list[dict[str, Any]]]:
    variants = manifest.get("variants")
    if variants is None:
        if variant is not None:
            raise ValueError("This source snapshot does not define variants")
        return "source", list(manifest["files"])
    if variant not in variants:
        raise ValueError(f"Choose one source variant from: {sorted(variants)}")
    directory = "source_pre_erratum" if variant == "pre_erratum" else "source_corrected"
    return directory, list(variants[variant]["files"])


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _assert_tag_binds_manifest(repo: Path, manifest_path: Path, tag: str) -> str:
    _git_output(repo, "rev-parse", "--verify", f"refs/tags/{tag}^{{tag}}")
    relative = manifest_path.relative_to(repo).as_posix()
    tagged = subprocess.check_output(["git", "-C", str(repo), "show", f"{tag}:{relative}"])
    if tagged != manifest_path.read_bytes():
        raise RuntimeError(f"Annotated tag {tag} does not bind the selected source manifest")
    return _git_output(repo, "rev-parse", f"{tag}^{{commit}}")


def _archive_runtime_base(repo: Path, tag: str, destination: Path) -> None:
    top_level = set(_git_output(repo, "ls-tree", "--name-only", tag).splitlines())
    selected = [name for name in RUNTIME_DIRECTORIES if name in top_level]
    selected.extend(name for name in RUNTIME_ROOT_FILES if name in top_level)
    if "rhinoform" not in selected:
        raise RuntimeError(f"Tagged source base {tag} does not contain rhinoform/")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="rhinoform-source-runtime-", suffix=".tar", dir=destination.parent, delete=False
    ) as handle:
        archive_path = Path(handle.name)
    try:
        with archive_path.open("wb") as handle:
            subprocess.run(
                ["git", "-C", str(repo), "archive", "--format=tar", tag, "--", *selected],
                check=True,
                stdout=handle,
            )
        destination.mkdir(parents=True, exist_ok=False)
        root = destination.resolve()
        with tarfile.open(archive_path, mode="r") as archive:
            for member in archive.getmembers():
                target = (destination / member.name).resolve()
                if target != root and root not in target.parents:
                    raise RuntimeError(f"Unsafe path in tagged source archive: {member.name}")
            if sys.version_info >= (3, 12):
                archive.extractall(destination, filter="data")
            else:  # pragma: no cover - compatibility path for the frozen Python 3.10 runtime
                archive.extractall(destination)
    finally:
        archive_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def materialize_source_runtime(
    *,
    repo_root: Path,
    snapshot_name: str,
    destination: Path,
    variant: str | None = None,
) -> dict[str, Any]:
    """Build a clean tagged runtime and overlay the byte-frozen source files.

    The destination must not already contain files. The live repository is never
    edited, imported, or used as an execution root.
    """
    repo = Path(repo_root).resolve()
    destination = Path(destination).resolve()
    snapshot = repo / "reproducibility/source_snapshots" / snapshot_name
    manifest_path, manifest = _validated_manifest(snapshot)
    source_directory, rows = _snapshot_rows(manifest, variant)
    tag = str(manifest["expected_git_tag"])
    base_commit = _assert_tag_binds_manifest(repo, manifest_path, tag)

    if destination == repo or repo in destination.parents:
        allowed_output = repo / "reproduction_output"
        if destination != allowed_output and allowed_output not in destination.parents:
            raise ValueError("Runtime destination inside the repository must be under reproduction_output/")
    if destination.exists():
        if any(destination.iterdir()):
            raise FileExistsError(f"Runtime destination is not empty: {destination}")
        destination.rmdir()

    _archive_runtime_base(repo, tag, destination)
    checked: list[dict[str, Any]] = []
    for row in rows:
        relative = Path(str(row["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe source snapshot path: {relative}")
        source = snapshot / source_directory / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size != int(row["bytes"]) or _sha256_file(source) != row["sha256"]:
            raise RuntimeError(f"Archived source mismatch: {source}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if target.stat().st_size != int(row["bytes"]) or _sha256_file(target) != row["sha256"]:
            raise RuntimeError(f"Materialized source mismatch: {target}")
        checked.append(
            {"path": relative.as_posix(), "bytes": int(row["bytes"]), "sha256": row["sha256"]}
        )

    report: dict[str, Any] = {
        "status": "PASS_ISOLATED_SOURCE_RUNTIME",
        "snapshot": snapshot_name,
        "variant": variant,
        "tag": tag,
        "base_commit": base_commit,
        "source_manifest_sha256": _sha256_file(manifest_path),
        "archived_files_checked": len(checked),
        "archived_files": checked,
        "runtime_root": str(destination),
        "execution_policy": "run_only_from_this_runtime; never import from the live checkout",
    }
    _atomic_write_json(destination / "RUNTIME_SOURCE_LOCK.json", report)
    return report


def verify_runtime_import_origins(
    *,
    runtime_root: Path,
    modules: dict[str, str],
    python_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Import selected modules in isolated Python and reject source-tree leakage."""
    root = Path(runtime_root).resolve()
    if not (root / "RUNTIME_SOURCE_LOCK.json").is_file():
        raise FileNotFoundError(f"Runtime source lock is missing: {root}")
    request = {"root": str(root), "modules": modules}
    probe = r'''
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

request = json.loads(sys.argv[1])
root = Path(request["root"]).resolve()
os.chdir(root)
sys.path.insert(0, str(root))
checked = {}
for name, expected_hash in request["modules"].items():
    module = importlib.import_module(name)
    origin = Path(module.__file__).resolve()
    if origin != root and root not in origin.parents:
        raise RuntimeError(f"Historical module escaped isolated runtime: {name}: {origin}")
    actual_hash = hashlib.sha256(origin.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Historical module hash mismatch: {name}: {actual_hash} != {expected_hash}"
        )
    checked[name] = {"origin": str(origin), "sha256": actual_hash}

external = {}
for name, module in sorted(sys.modules.items()):
    if not name.startswith(("rhinoform", "scripts", "experiments")):
        continue
    filename = getattr(module, "__file__", None)
    if filename is None:
        continue
    origin = Path(filename).resolve()
    if origin != root and root not in origin.parents:
        external[name] = str(origin)
if external:
    raise RuntimeError(f"Repository modules imported outside isolated runtime: {external}")
print("RHINOFORM_IMPORT_REPORT=" + json.dumps({"modules": checked}, sort_keys=True))
'''
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [
            str(python_executable or sys.executable),
            "-I",
            "-c",
            probe,
            json.dumps(request, sort_keys=True),
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Isolated runtime import probe failed\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    prefix = "RHINOFORM_IMPORT_REPORT="
    lines = [line for line in completed.stdout.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise RuntimeError(f"Isolated import probe did not emit one report: {completed.stdout}")
    report = json.loads(lines[0][len(prefix) :])
    report["status"] = "PASS_ISOLATED_IMPORTS"
    report["runtime_root"] = str(root)
    return report


def run_runtime_module(
    *,
    runtime_root: Path,
    module: str,
    arguments: list[str] | tuple[str, ...] = (),
    python_executable: str | Path | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Execute a module with the isolated runtime as its only project source.

    Python's isolated mode ignores user site-packages and ``PYTHONPATH``. The
    launcher adds exactly the materialized runtime, then rejects any Rhinoform
    repository module that resolves outside it before returning.
    """
    root = Path(runtime_root).resolve()
    if not (root / "RUNTIME_SOURCE_LOCK.json").is_file():
        raise FileNotFoundError(f"Runtime source lock is missing: {root}")
    if not module or any(part in {"", ".", ".."} for part in module.split(".")):
        raise ValueError(f"Invalid Python module name: {module!r}")
    launcher = r'''
import os
from pathlib import Path
import runpy
import sys

root = Path(sys.argv[1]).resolve()
module_name = sys.argv[2]
module_arguments = sys.argv[3:]
os.chdir(root)
sys.path.insert(0, str(root))
sys.argv = [module_name, *module_arguments]
exit_code = 0
try:
    runpy.run_module(module_name, run_name="__main__", alter_sys=True)
except SystemExit as error:
    if error.code is None:
        exit_code = 0
    elif isinstance(error.code, int):
        exit_code = error.code
    else:
        print(error.code, file=sys.stderr)
        exit_code = 1

external = {}
for name, loaded in sorted(sys.modules.items()):
    if not name.startswith(("rhinoform", "scripts", "experiments")):
        continue
    filename = getattr(loaded, "__file__", None)
    if filename is None:
        continue
    origin = Path(filename).resolve()
    if origin != root and root not in origin.parents:
        external[name] = str(origin)
if external:
    raise RuntimeError(f"Repository modules imported outside isolated runtime: {external}")
raise SystemExit(exit_code)
'''
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    return subprocess.run(
        [
            str(python_executable or sys.executable),
            "-I",
            "-c",
            launcher,
            str(root),
            module,
            *[str(value) for value in arguments],
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=capture_output,
        check=False,
    )
