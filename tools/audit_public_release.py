"""Fail closed on common public-repository release mistakes."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_GITHUB_BLOB_BYTES = 100_000_000
REQUIRED = (
    "README.md",
    "LICENSE_PENDING.md",
    "CITATION.cff",
    "THIRD_PARTY.md",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "requirements.txt",
    "requirements-replay.txt",
    "configs/reproduction.example.toml",
    "data/manifest.json",
    "reproducibility/RELEASE_ASSET_MANIFEST.json",
    "reproducibility/RELEASE_ARCHIVE.json",
    ".github/workflows/ci.yml",
)
FORBIDDEN_SUFFIXES = {".pt", ".npy", ".npz"}


def tracked_files(root: Path) -> list[Path]:
    output = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "--cached", "-z"],
        stderr=subprocess.STDOUT,
    )
    return [root / item.decode("utf-8") for item in output.split(b"\0") if item]


def privacy_hits_in_public_surface(root: Path) -> list[str]:
    pattern = r'/Users/[A-Za-z0-9._-]+/|[A-Za-z0-9._%+-]+@(gmail|icloud|outlook)\.[A-Za-z]{2,}|"(userId|displayName)"[[:space:]]*:'
    targets = [
        "README.md",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "THIRD_PARTY.md",
        ".github",
        "configs",
        "data/README.md",
        "docs",
        "notebooks",
        "reproducibility/README.md",
    ]
    completed = subprocess.run(
        [
            "rg",
            "-I",
            "-n",
            pattern,
            "--",
            *targets,
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(f"privacy audit failed: {completed.stderr}")
    return completed.stdout.splitlines()


def audit_public_release(root: Path = ROOT) -> dict[str, object]:
    root = root.resolve()
    missing = [relative for relative in REQUIRED if not (root / relative).is_file()]
    if missing:
        raise RuntimeError(f"Missing public-release files: {missing}")

    files = tracked_files(root)
    sizes = {
        str(path.relative_to(root)): path.stat().st_size
        for path in files
        if path.is_file()
    }
    oversized = [relative for relative, size in sizes.items() if size >= MAX_GITHUB_BLOB_BYTES]
    if oversized:
        raise RuntimeError(f"Files exceed GitHub's ordinary blob limit: {oversized}")

    binary_payloads = [
        str(path.relative_to(root)) for path in files if path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    if binary_payloads:
        raise RuntimeError(
            "Model/array payloads must be release assets, not Git blobs: "
            f"{binary_payloads}"
        )

    privacy_hits = privacy_hits_in_public_surface(root)
    if privacy_hits:
        raise RuntimeError(f"Personal machine/account metadata remains tracked: {privacy_hits}")

    return {
        "status": "PASS_PUBLIC_RELEASE_AUDIT",
        "tracked_files": len(files),
        "largest_tracked_blob_bytes": max(sizes.values(), default=0),
        "required_files": len(REQUIRED),
        "ordinary_git_model_payloads": 0,
        "privacy_hits": 0,
    }


def main() -> int:
    print(json.dumps(audit_public_release(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
