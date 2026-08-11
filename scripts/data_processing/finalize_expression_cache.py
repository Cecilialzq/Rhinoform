from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from rhinoform.repro import atomic_write_json, sha256_file


def source_archive(subject_id: str) -> str:
    value = int(subject_id)
    start = ((value - 1) // 100) * 100 + 1
    end = min(start + 99, 847)
    return f"facescape_trainset_{start:03d}_{end:03d}.zip"


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize a manifest from already streamed expression ROI caches.")
    parser.add_argument("--neutral-repo", type=Path, required=True)
    parser.add_argument("--expression-root", type=Path, required=True)
    parser.add_argument("--split", default="main test")
    parser.add_argument("--expected-expression", action="append", default=[])
    args = parser.parse_args()

    expected_expressions = args.expected_expression or [
        "smile",
        "anger",
        "lip_puckerer",
        "lip_funneler",
        "cheek_blowing",
        "brow_raiser",
    ]
    neutral_manifest_path = args.neutral_repo / "manifest.json"
    neutral_manifest = json.loads(neutral_manifest_path.read_text(encoding="utf-8"))
    neutral_rows = {
        str(row["subject_id"]): row
        for row in neutral_manifest["rows"]
        if str(row["split"]) == args.split
    }
    mesh_dir = args.expression_root / "meshes"
    grouped: dict[str, set[str]] = defaultdict(set)
    rows: list[dict] = []
    for path in sorted(mesh_dir.glob("*.npz")):
        subject_id, expression = path.stem.split("_", 1)
        if subject_id not in neutral_rows or expression not in expected_expressions:
            continue
        with np.load(path, allow_pickle=False) as data:
            roi_count = int(np.asarray(data["vertices"]).shape[0])
        grouped[subject_id].add(expression)
        rows.append(
            {
                "npz_path": str(path.relative_to(args.expression_root)),
                "subject_id": subject_id,
                "expression": expression,
                "split": args.split,
                "roi_count": roi_count,
                "vertex_count": int(neutral_manifest["topology_vertex_count"]),
                "sha256": sha256_file(path),
                "source_zip": source_archive(subject_id),
                "source_member": "not_recovered_from_cache_finalization",
                "source_crc": None,
            }
        )

    expected_ids = set(neutral_rows)
    complete_ids = sorted(
        [subject_id for subject_id in expected_ids if grouped[subject_id] >= set(expected_expressions)],
        key=int,
    )
    incomplete = {
        subject_id: sorted(set(expected_expressions) - grouped[subject_id])
        for subject_id in sorted(expected_ids, key=int)
        if grouped[subject_id] < set(expected_expressions)
    }
    rows.sort(key=lambda row: (int(row["subject_id"]), row["expression"]))
    counts = Counter(row["expression"] for row in rows)
    manifest = {
        "source": "finalize_expression_cache.py",
        "selection_status": "complete_case_cache_recovery",
        "selection_rule": (
            "Use frozen main-test identities with all six requested expression ROI caches; "
            "selection depends only on cache availability, never model outcomes."
        ),
        "protocol_deviation": (
            "The Google Drive File Provider could not hydrate the remaining members of the "
            "201-300 archive during the evaluation window."
        ),
        "neutral_manifest": str(neutral_manifest_path),
        "neutral_manifest_sha256": sha256_file(neutral_manifest_path),
        "requested_split": args.split,
        "expressions": expected_expressions,
        "expected_subject_count": len(expected_ids),
        "complete_case_subject_count": len(complete_ids),
        "complete_case_subject_ids": complete_ids,
        "incomplete_or_missing": incomplete,
        "expression_counts": dict(sorted(counts.items())),
        "topology_vertex_count": int(neutral_manifest["topology_vertex_count"]),
        "n_rows": len(rows),
        "rows": rows,
    }
    args.expression_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.expression_root / "manifest.json", manifest)
    done = {
        "decision": "pass_with_protocol_deviation",
        "manifest_sha256": sha256_file(args.expression_root / "manifest.json"),
        "expected_subject_count": len(expected_ids),
        "complete_case_subject_count": len(complete_ids),
        "complete_case_pair_count": len(complete_ids) * (len(complete_ids) - 1),
        "expression_counts": dict(sorted(counts.items())),
        "incomplete_or_missing": incomplete,
    }
    atomic_write_json(args.expression_root / "EXPRESSIONS_DONE.json", done)
    print(json.dumps(done, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
