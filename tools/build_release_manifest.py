"""Build a release manifest from a real Git commit; refuse dirty/uncommitted state."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    commit = run("git", "-C", str(root), "rev-parse", "HEAD")
    if run("git", "-C", str(root), "status", "--porcelain"):
        raise SystemExit("Refusing to freeze a dirty worktree")
    tracked = run("git", "-C", str(root), "ls-files").splitlines()
    artefacts = []
    for relative in tracked:
        path = root / relative
        if not path.is_file() or relative == "RELEASE_MANIFEST.json":
            continue
        data = path.read_bytes()
        artefacts.append({"path": relative, "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest()})
    output = {"repository": "Rhinoform", "git_commit": commit,
              "artifact_count": len(artefacts), "artifacts": artefacts,
              "exclusions": ["licensed FaceScape meshes", "dense prediction caches",
                             "legacy pre-strict checkpoints", "pending experiment outputs"]}
    (root / "RELEASE_MANIFEST.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(artefacts)} artefacts at commit {commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

