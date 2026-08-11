from __future__ import annotations

import argparse
import json
from pathlib import Path

from rhinoform.repro import SEED_REGISTRY, atomic_write_json, sha256_file


PRIMARY_SCALES = [30, 60, 120, 240, 480, 676]
SECONDARY_REQUIRED_SCALES = [30, 120, 480, 676]
EXPECTED_REQUIRED_PAIR_COUNT = 23


def file_hash_or_none(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


def dataset_ready(repo: Path, split_manifest: Path | None = None) -> tuple[bool, str]:
    manifest = repo / "manifest.json"
    done_path = repo / "DATASET_DONE.json"
    if not repo.exists() or not manifest.exists():
        return False, "blocked_missing_dataset_manifest"
    if not done_path.exists():
        return False, "blocked_missing_dataset_done"
    try:
        done = json.loads(done_path.read_text(encoding="utf-8"))
    except Exception:
        return False, "blocked_invalid_dataset_done"
    if done.get("decision") != "pass":
        return False, "blocked_dataset_done_not_pass"
    if split_manifest is not None and split_manifest.exists():
        if done.get("split_manifest_sha256") != sha256_file(split_manifest):
            return False, "blocked_dataset_done_split_manifest_mismatch"
    try:
        manifest_obj = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False, "blocked_invalid_dataset_manifest"
    if int(done.get("n_rows", -1)) != int(manifest_obj.get("n_rows", -2)):
        return False, "blocked_dataset_done_manifest_row_mismatch"
    if done.get("failed"):
        return False, "blocked_dataset_done_contains_failures"
    return True, "pending"


def split_ready(split_manifest: Path) -> tuple[bool, str]:
    if not split_manifest.exists():
        return False, "blocked_missing_split_manifest"
    try:
        obj = json.loads(split_manifest.read_text(encoding="utf-8"))
    except Exception:
        return False, "blocked_invalid_split_manifest"
    if int(obj.get("expected_usable", -1)) != 846 or int(obj.get("usable_count", -1)) != 846:
        return False, "blocked_split_manifest_usable_count_mismatch"
    sizes = obj.get("sizes", {})
    if {k: int(sizes.get(k, -1)) for k in ("test", "val", "train_pool")} != {"test": 100, "val": 70, "train_pool": 676}:
        return False, "blocked_split_manifest_size_mismatch"
    qc = obj.get("identity_qc", {})
    if int(qc.get("passed_count", -1)) != 846:
        return False, "blocked_split_manifest_qc_mismatch"
    counts = obj.get("pair_manifest_counts", {})
    if int(counts.get("required", -1)) != EXPECTED_REQUIRED_PAIR_COUNT:
        return False, "blocked_required_pair_manifest_count_mismatch"
    if len(obj.get("required_pair_manifests", [])) != EXPECTED_REQUIRED_PAIR_COUNT:
        return False, "blocked_required_pair_manifest_list_mismatch"
    return True, "pending"


def run_id(curve: str, chain: int, scale: int) -> str:
    return f"required_{curve}_chain{chain}_scale{scale}"


def run_record(
    *,
    curve: str,
    chain: int,
    scale: int,
    pair_manifest: Path,
    repo: Path,
    runs_dir: Path,
    split_manifest: Path,
    dense_cache: bool,
) -> dict:
    rid = run_id(curve, chain, scale)
    out_dir = runs_dir / rid
    status_path = out_dir / "status.json"
    status = "pending"
    ready, blocked_status = dataset_ready(repo, split_manifest)
    if not ready:
        status = blocked_status
    else:
        split_ok, split_status = split_ready(split_manifest)
        if not split_ok:
            status = split_status
        elif not pair_manifest.exists():
            status = "blocked_missing_pair_manifest"
        elif status_path.exists():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8")).get("state", "pending")
            except Exception:
                status = "pending"
    return {
        "run_id": rid,
        "tier": "required",
        "curve": curve,
        "chain": chain,
        "scale": scale,
        "seed": int(SEED_REGISTRY["chain_seeds"][str(chain)]),
        "train_pair_budget": 870 if curve == "primary" else min(scale * (scale - 1), 8700),
        "pair_manifest": str(pair_manifest),
        "pair_manifest_sha256": file_hash_or_none(pair_manifest),
        "train_ids_json": "",
        "val_ids_json": "",
        "test_ids_json": "",
        "out_dir": str(out_dir),
        "checkpoint_dir": str(out_dir / "checkpoints"),
        "status": status,
        "dense_prediction_npz_required": dense_cache,
        "command": [
            "python",
            "rhinoform/train.py",
            "--repo",
            str(repo),
            "--out",
            str(out_dir),
            "--model-kind",
            "cvae",
            "--no-use-subunit-features",
            "--train-pairs-json",
            str(pair_manifest),
            "--checkpoint-dir",
            str(out_dir / "checkpoints"),
            "--checkpoint-every",
            "10",
            "--seed",
            str(SEED_REGISTRY["chain_seeds"][str(chain)]),
            "--split-manifest",
            str(split_manifest),
        ]
        + (["--save-dense-predictions"] if dense_cache else ["--no-save-dense-predictions"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", default="splits/facescape_847")
    parser.add_argument("--repo", default="data")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", default="execution_plan.json")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    repo = Path(args.repo)
    runs_dir = Path(args.runs_dir)
    split_manifest = split_dir / "split_manifest.json"
    pair_dir = split_dir / "pairs"

    runs = []
    anchor_pair = pair_dir / "primary_anchor_scale30.json"
    runs.append(
        run_record(
            curve="primary",
            chain=0,
            scale=30,
            pair_manifest=anchor_pair,
            repo=repo,
            runs_dir=runs_dir,
            split_manifest=split_manifest,
            dense_cache=True,
        )
    )
    runs[-1]["run_id"] = "required_primary_anchor_scale30"
    runs[-1]["chain"] = "anchor"
    runs[-1]["seed"] = int(SEED_REGISTRY["training_seed"])
    runs[-1]["command"] = [
        "python",
        "rhinoform/train.py",
        "--repo",
        str(repo),
        "--out",
        str(runs_dir / "required_primary_anchor_scale30"),
        "--model-kind",
        "cvae",
        "--no-use-subunit-features",
        "--train-pairs-json",
        str(anchor_pair),
        "--checkpoint-dir",
        str(runs_dir / "required_primary_anchor_scale30" / "checkpoints"),
        "--checkpoint-every",
        "10",
        "--seed",
        str(SEED_REGISTRY["training_seed"]),
        "--split-manifest",
        str(split_manifest),
        "--save-dense-predictions",
    ]
    runs[-1]["out_dir"] = str(runs_dir / "required_primary_anchor_scale30")
    runs[-1]["checkpoint_dir"] = str(runs_dir / "required_primary_anchor_scale30" / "checkpoints")
    anchor_status_path = runs_dir / "required_primary_anchor_scale30" / "status.json"
    ready, blocked_status = dataset_ready(repo, split_manifest)
    if not ready:
        runs[-1]["status"] = blocked_status
    else:
        split_ok, split_status = split_ready(split_manifest)
        if not split_ok:
            runs[-1]["status"] = split_status
        elif not anchor_pair.exists():
            runs[-1]["status"] = "blocked_missing_pair_manifest"
        elif anchor_status_path.exists():
            try:
                runs[-1]["status"] = json.loads(anchor_status_path.read_text(encoding="utf-8")).get("state", "pending")
            except Exception:
                runs[-1]["status"] = "pending"
        else:
            runs[-1]["status"] = "pending"
    for chain in range(3):
        for scale in PRIMARY_SCALES:
            dense = scale in {30, 676}
            runs.append(
                run_record(
                    curve="primary",
                    chain=chain,
                    scale=scale,
                    pair_manifest=pair_dir / f"primary_chain_{chain}_scale{scale}.json",
                    repo=repo,
                    runs_dir=runs_dir,
                    split_manifest=split_manifest,
                    dense_cache=dense,
                )
            )
    for scale in SECONDARY_REQUIRED_SCALES:
        runs.append(
            run_record(
                curve="secondary",
                chain=0,
                scale=scale,
                pair_manifest=pair_dir / f"secondary_chain_0_scale{scale}.json",
                repo=repo,
                runs_dir=runs_dir,
                split_manifest=split_manifest,
                dense_cache=scale == 676,
            )
        )

    blocked = [r for r in runs if str(r["status"]).startswith("blocked")]
    plan = {
        "source": "plan.py",
        "scope": "Required tier from GOAL_experimental_layer.md §9.0",
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": file_hash_or_none(split_manifest),
        "dataset_manifest": str(repo / "manifest.json"),
        "dataset_done": str(repo / "DATASET_DONE.json"),
        "run_count": len(runs),
        "blocked_count": len(blocked),
        "compute_estimate_gpu_hours": {"primary": 14, "secondary": 24, "total_training": 38},
        "dense_cache_policy": "per-pair CSV for every run; dense prediction NPZ for final/manuscript-bound and sensitivity-map runs",
        "final_model_run_id": "required_primary_chain0_scale676",
        "final_model_policy": "top-scale primary chain 0, seed 20260609; used for Required-tier noise robustness and dependence table unless a later frozen manifest explicitly selects another top-scale run",
        "runs": runs,
    }
    atomic_write_json(Path(args.out), plan)
    print(json.dumps({"out": args.out, "run_count": len(runs), "blocked_count": len(blocked)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
