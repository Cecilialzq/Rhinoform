from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rhinoform.repro import SEED_REGISTRY, artifact_metadata, atomic_write_json


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def choose(rows: list[dict[str, str]], seed: int, n: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    if not rows:
        return []
    numeric = []
    for i, row in enumerate(rows):
        numeric.append(
            {
                **row,
                "_row_index": i,
                "_roi": float(row.get("roi_rmse", "nan")),
                "_flip": float(row.get("normal_flip_pct", "nan")),
            }
        )
    valid = [r for r in numeric if np.isfinite(r["_roi"]) and np.isfinite(r["_flip"])]
    if not valid:
        return []
    valid_sorted_roi = sorted(valid, key=lambda r: r["_roi"])
    valid_sorted_flip = sorted(valid, key=lambda r: r["_flip"])
    pools = {
        "median_roi": valid_sorted_roi[max(0, len(valid_sorted_roi) // 2 - 10) : min(len(valid_sorted_roi), len(valid_sorted_roi) // 2 + 11)],
        "high_roi": valid_sorted_roi[int(0.9 * (len(valid_sorted_roi) - 1)) :],
        "high_flip": valid_sorted_flip[int(0.9 * (len(valid_sorted_flip) - 1)) :],
    }
    selected = []
    seen: set[tuple[str, str]] = set()
    for reason, pool in pools.items():
        if not pool:
            continue
        order = rng.permutation(len(pool))
        for idx in order:
            row = pool[int(idx)]
            key = (str(row["source_id"]), str(row["target_id"]))
            if key in seen:
                continue
            seen.add(key)
            selected.append(
                {
                    "source_id": key[0],
                    "target_id": key[1],
                    "selection_reason": reason,
                    "roi_rmse": row["_roi"],
                    "normal_flip_pct": row["_flip"],
                    "source_row_index": row["_row_index"],
                    "seed": seed,
                }
            )
            break
    remaining = [r for r in valid if (str(r["source_id"]), str(r["target_id"])) not in seen]
    while len(selected) < n and remaining:
        idx = int(rng.integers(0, len(remaining)))
        row = remaining.pop(idx)
        key = (str(row["source_id"]), str(row["target_id"]))
        selected.append(
            {
                "source_id": key[0],
                "target_id": key[1],
                "selection_reason": "seeded_fill",
                "roi_rmse": row["_roi"],
                "normal_flip_pct": row["_flip"],
                "source_row_index": row["_row_index"],
                "seed": seed,
            }
        )
    return selected[:n]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-metrics", required=True)
    parser.add_argument("--out", default="frozen_artifacts/qualitative_cases")
    parser.add_argument("--n-cases", type=int, default=6)
    parser.add_argument("--seed", type=int, default=SEED_REGISTRY["figure_case_seed"])
    parser.add_argument("--split-manifest", default="")
    args = parser.parse_args()

    rows = read_csv(Path(args.pair_metrics))
    selected = choose(rows, args.seed, args.n_cases)
    out_dir = Path(args.out)
    write_csv(out_dir / "qualitative_cases.csv", selected)
    report = {
        "pair_metrics": args.pair_metrics,
        "seed": args.seed,
        "n_cases": args.n_cases,
        "selected": selected,
        "metadata": artifact_metadata(
            repo_root=Path(__file__).resolve().parents[1],
            command_args=args,
            split_manifest_path=Path(args.split_manifest) if args.split_manifest else None,
            seed=args.seed,
            data_root_identifier="frozen_pair_metrics",
        ),
    }
    atomic_write_json(out_dir / "qualitative_cases_manifest.json", report)
    print(json.dumps({"out": str(out_dir), "selected": len(selected), "seed": args.seed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

