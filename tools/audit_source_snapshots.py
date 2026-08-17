"""Verify every archived source byte and, optionally, its annotated Git tag."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_sidecar(path: Path) -> bool:
    sidecar = path.with_name(path.name + ".sha256.json")
    if not path.is_file() or not sidecar.is_file():
        return False
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    return (
        record.get("path") == path.name
        and int(record.get("bytes", -1)) == path.stat().st_size
        and record.get("sha256") == sha256_file(path)
    )


def _rows(manifest: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    if "variants" in manifest:
        return [
            ("source_pre_erratum", manifest["variants"]["pre_erratum"]["files"]),
            ("source_corrected", manifest["variants"]["corrected"]["files"]),
        ]
    rows = list(manifest["files"])
    rows.extend(manifest.get("supporting_files_not_in_original_protocol_hash_map", []))
    return [("source", rows)]


def audit_snapshot(snapshot: Path) -> dict[str, Any]:
    manifest_path = snapshot / "SOURCE_MANIFEST.json"
    if not valid_sidecar(manifest_path):
        raise RuntimeError(f"Invalid source manifest or sidecar: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = 0
    for directory, rows in _rows(manifest):
        for row in rows:
            path = snapshot / directory / row["path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = sha256_file(path)
            if actual != row["sha256"] or path.stat().st_size != int(row["bytes"]):
                raise RuntimeError(
                    f"Source snapshot mismatch: {path} expected={row['sha256']} actual={actual}"
                )
            checked += 1
    return {
        "snapshot": snapshot.name,
        "status": "PASS",
        "files_checked": checked,
        "manifest_sha256": sha256_file(manifest_path),
        "expected_git_tag": manifest["expected_git_tag"],
    }


def audit_tag(repo: Path, snapshot: Path, tag: str) -> str:
    subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", f"refs/tags/{tag}^{{tag}}"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    relative = snapshot.relative_to(repo) / "SOURCE_MANIFEST.json"
    tagged = subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{tag}:{relative.as_posix()}"]
    )
    live = (snapshot / "SOURCE_MANIFEST.json").read_bytes()
    if tagged != live:
        raise RuntimeError(f"Tag {tag} does not bind the current source manifest")
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", f"{tag}^{{commit}}"], text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-tags", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    reports = []
    for snapshot in sorted((root / "reproducibility/source_snapshots").iterdir()):
        if not snapshot.is_dir():
            continue
        report = audit_snapshot(snapshot)
        if args.require_tags:
            report["git_commit"] = audit_tag(
                root, snapshot, str(report["expected_git_tag"])
            )
        reports.append(report)
    print(json.dumps({"status": "PASS", "snapshots": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
