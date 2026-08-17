from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SEED_REGISTRY = {
    "global_seed": 20260609,
    "split_seed": 20260609,
    "training_seed": 20260609,
    "bootstrap_seed": 20260609,
    "noise_seed": 20260609,
    "figure_case_seed": 20260609,
    "chain_seeds": {
        "0": 20260609,
        "1": 20260610,
        "2": 20260611,
        "3": 20260612,
        "4": 20260613,
    },
}


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except Exception:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker_factory(training_seed: int):
    def seed_worker(worker_id: int) -> None:
        worker_seed = int(training_seed) + int(worker_id)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_sha256_sidecar(path: Path) -> Path:
    sidecar = path.with_suffix(path.suffix + ".sha256.json")
    atomic_write_json(sidecar, {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    })
    return sidecar


def valid_sha256_sidecar(path: Path) -> bool:
    sidecar = path.with_suffix(path.suffix + ".sha256.json")
    if not path.is_file() or not sidecar.is_file():
        return False
    try:
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        return (
            record.get("path") == path.name
            and int(record["bytes"]) == path.stat().st_size
            and str(record["sha256"]) == sha256_file(path)
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def validate_torch_artifact(
    path: Path,
    required_keys: tuple[str, ...] = (),
    repair_sidecar: bool = True,
) -> bool:
    """Validate an existing Torch file and optionally create/repair its hash sidecar."""
    import torch

    if not path.is_file():
        return False
    sidecar_valid = valid_sha256_sidecar(path)
    if sidecar_valid and not required_keys:
        return True
    try:
        probe = torch.load(path, map_location="cpu", weights_only=False)
        if required_keys:
            if not isinstance(probe, dict) or any(key not in probe for key in required_keys):
                return False
        del probe
        if not sidecar_valid and repair_sidecar:
            write_sha256_sidecar(path)
        return sidecar_valid or repair_sidecar
    except Exception:
        return False


def atomic_torch_save(path: Path, payload: Any, required_keys: tuple[str, ...] = ()) -> None:
    """Atomically serialise, fsync, read back and hash a Torch artefact."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        probe = torch.load(tmp, map_location="cpu", weights_only=False)
        if required_keys:
            if not isinstance(probe, dict):
                raise ValueError(f"Expected a mapping in Torch artefact {tmp}")
            missing = [key for key in required_keys if key not in probe]
            if missing:
                raise ValueError(f"Incomplete Torch artefact {tmp}; missing keys={missing}")
        del probe
        os.replace(tmp, path)
        write_sha256_sidecar(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_savez_compressed(path: Path, **arrays: np.ndarray) -> None:
    """Atomically write and read-validate a compressed NumPy archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        with np.load(tmp, allow_pickle=True) as probe:
            if set(probe.files) != set(arrays):
                raise ValueError(f"Incomplete NumPy archive {tmp}")
        os.replace(tmp, path)
        write_sha256_sidecar(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        with tmp.open(newline="", encoding="utf-8") as handle:
            if len(list(csv.DictReader(handle))) != len(rows):
                raise ValueError(f"CSV read-back row count mismatch: {tmp}")
        os.replace(tmp, path)
        write_sha256_sidecar(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_copy_file(source: Path, destination: Path) -> None:
    """Copy a file through a same-directory temporary, then hash the committed copy."""
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with source.open("rb") as src, tmp.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        if sha256_file(tmp) != sha256_file(source):
            raise ValueError(f"Copied file hash mismatch: {source} -> {tmp}")
        os.replace(tmp, destination)
        write_sha256_sidecar(destination)
    finally:
        if tmp.exists():
            tmp.unlink()


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        import torch

        state["torch_cpu"] = torch.get_rng_state()
        state["torch_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    except Exception:
        pass
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    try:
        import torch

        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and state.get("torch_cuda"):
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except Exception:
        pass


def git_commit_hash(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "not-a-git-repository"


def environment_metadata() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["cuda_available"] = bool(torch.cuda.is_available())
        meta["cuda"] = torch.version.cuda
        meta["cudnn"] = torch.backends.cudnn.version()
    except Exception as exc:
        meta["torch"] = f"unavailable: {exc}"
    for pkg in ("scipy", "matplotlib", "jupyter"):
        try:
            import importlib.metadata as importlib_metadata

            meta[pkg] = importlib_metadata.version(pkg)
        except Exception as exc:
            meta[pkg] = f"unavailable: {exc}"
    return meta


def artifact_metadata(
    *,
    repo_root: Path,
    command_args: argparse.Namespace | dict[str, Any],
    config_path: Path | None = None,
    input_manifest_path: Path | None = None,
    split_manifest_path: Path | None = None,
    seed: int | None = None,
    data_root_identifier: str = "not-recorded",
    output_schema_version: str = "2026-06-09",
) -> dict[str, Any]:
    args_dict = vars(command_args) if isinstance(command_args, argparse.Namespace) else dict(command_args)
    resolved_config = {
        "source": "command_args",
        "command_args": args_dict,
    }
    meta: dict[str, Any] = {
        "git_commit": git_commit_hash(repo_root),
        "command_args": args_dict,
        "resolved_config": resolved_config,
        "resolved_config_sha256": sha256_json(resolved_config),
        "seed": seed,
        "environment": environment_metadata(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_root_identifier": data_root_identifier,
        "output_schema_version": output_schema_version,
    }
    requirements_path = repo_root / "requirements.txt"
    meta["requirements_path"] = "requirements.txt" if requirements_path.exists() else None
    meta["requirements_sha256"] = sha256_file(requirements_path) if requirements_path.exists() else None
    for key, path in (
        ("config", config_path),
        ("input_manifest", input_manifest_path),
        ("split_manifest", split_manifest_path),
    ):
        if path is None:
            if key == "config":
                meta[f"{key}_path"] = "<command_args>"
                meta[f"{key}_sha256"] = meta["resolved_config_sha256"]
            else:
                meta[f"{key}_path"] = None
                meta[f"{key}_sha256"] = None
        else:
            p = Path(path)
            meta[f"{key}_path"] = str(p)
            meta[f"{key}_sha256"] = sha256_file(p) if p.exists() else None
    meta["metadata_sha256"] = sha256_json({k: v for k, v in meta.items() if k != "metadata_sha256"})
    return meta


def write_seed_registry(path: Path) -> None:
    atomic_write_json(path, SEED_REGISTRY)


def append_run_report(path: Path, section: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(section.rstrip() + "\n\n")
