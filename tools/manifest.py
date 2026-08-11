from __future__ import annotations

import argparse
import json
from pathlib import Path

from rhinoform.repro import SEED_REGISTRY, atomic_write_json, artifact_metadata, sha256_file, write_seed_registry


DEFAULT_PATTERNS = [
    "code/**/*.py",
    "code/**/*.md",
    "docs/**/*.md",
    "notebooks/**/*.ipynb",
    "GOAL_experimental_layer.md",
    "CODEX_GOAL_PROMPT.md",
    "RUN_REPORT.md",
    "requirements.txt",
    "execution_plan.json",
    "progress_ledger.json",
    "seed_registry.json",
    "REQUIRED_TIER_FREEZE_AUDIT.json",
    "results/**/*.csv",
    "results/**/*.json",
    "figures/**/*.png",
    "figures/**/*.pdf",
    "splits/**/*.json",
    "runs/**/metadata.json",
    "runs/**/predictions/*.csv",
    "runs/**/predictions/*.npz",
    "frozen_artifacts/**/*.csv",
    "frozen_artifacts/**/*.json",
    "frozen_artifacts/**/*.npz",
    "frozen_artifacts/**/*.png",
    "frozen_artifacts/**/*.pdf",
]


def collect(root: Path, patterns: list[str]) -> list[dict]:
    seen: set[Path] = set()
    rows = []
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                rows.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="reproducibility_manifest.json")
    parser.add_argument("--pattern", action="append", default=[])
    parser.add_argument("--split-manifest", default="")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    patterns = args.pattern if args.pattern else DEFAULT_PATTERNS
    artifacts = collect(root, patterns)
    write_seed_registry(root / "seed_registry.json")
    manifest = {
        "source": "manifest.py",
        "seed_registry": SEED_REGISTRY,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "metadata": artifact_metadata(
            repo_root=root,
            command_args=args,
            input_manifest_path=root / "data" / "manifest.json",
            split_manifest_path=Path(args.split_manifest) if args.split_manifest else None,
            seed=SEED_REGISTRY["global_seed"],
            data_root_identifier="frozen_artifact_tree",
        ),
    }
    atomic_write_json(root / args.out, manifest)
    print(json.dumps({"out": args.out, "artifact_count": len(artifacts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
