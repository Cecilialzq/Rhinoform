"""Build the 847-identity split and nested-chain manifest required by GOAL §A.

This script reads the licensed FaceScape neutral registered meshes from either
flat ``<id>.obj`` files or nested ``<id>/models_reg/1_neutral.obj`` folders,
performs the pre-split QC gate, fits a split-design PCA on cropped ROI vertices,
then writes a manifest with fixed test/val/train-pool IDs, canonical and random
nested chains, and fixed pair manifests for primary/secondary learning curves.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.data_processing.build_data import parse_obj
from rhinoform.repro import SEED_REGISTRY, atomic_write_json, sha256_file, write_seed_registry
from rhinoform.sampling import coverage_balanced_ordered_pairs, identity_sort_key


SCALES = [30, 60, 120, 240, 480, 676]
SECONDARY_REQUIRED_SCALES = [30, 120, 480, 676]
REQUIRED_PRIMARY_CHAINS = {"chain_0", "chain_1", "chain_2"}
OPTIONAL_PRIMARY_CHAINS = {f"chain_{i}" for i in range(5)}
OPTIONAL_SECONDARY_CHAINS = {f"chain_{i}" for i in range(5)}


def tier_for_pair_manifest(chain_name: str, curve: str, scale: int) -> str | None:
    if chain_name == "anchor":
        return "required" if curve == "primary" and scale == 30 else None
    if curve == "primary" and chain_name in REQUIRED_PRIMARY_CHAINS and scale in SCALES:
        return "required"
    if curve == "secondary" and chain_name == "chain_0" and scale in SECONDARY_REQUIRED_SCALES:
        return "required"
    if curve == "primary" and chain_name in OPTIONAL_PRIMARY_CHAINS and scale in SCALES:
        return "optional"
    if curve == "secondary" and chain_name in OPTIONAL_SECONDARY_CHAINS and scale in SCALES:
        return "optional"
    return None


def resolve_obj(obj_root: Path, sid: str) -> Path | None:
    candidates = [
        obj_root / sid / "models_reg" / "1_neutral.obj",
        obj_root / f"{sid}.obj",
        obj_root / f"{int(sid):03d}" / "models_reg" / "1_neutral.obj" if sid.isdigit() else obj_root / sid,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def roi_mesh_hash(vertices: np.ndarray) -> str:
    arr = np.ascontiguousarray(vertices.astype(np.float32))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def roi_degeneracy_reasons(vertices: np.ndarray, eps: float = 1e-8) -> list[str]:
    reasons = []
    if vertices.size == 0:
        return ["empty_roi_vertices"]
    extent = np.ptp(vertices, axis=0)
    if not bool(np.all(extent > eps)):
        reasons.append("degenerate_roi_extent")
    unique_count = np.unique(vertices.astype(np.float32), axis=0).shape[0]
    if unique_count < max(3, int(0.95 * len(vertices))):
        reasons.append("excess_duplicate_roi_vertices")
    return reasons


def legacy_splits(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {"test": [], "val": [], "train": []}
    man = json.loads(path.read_text(encoding="utf-8"))
    out = {"test": [], "val": [], "train": []}
    for row in man.get("rows", []):
        split = str(row.get("split", ""))
        sid = str(row.get("subject_id"))
        if split == "main test":
            out["test"].append(sid)
        elif split == "clean-prior validation":
            out["val"].append(sid)
        elif split == "clean-prior train":
            out["train"].append(sid)
    return {k: sorted(v, key=identity_sort_key) for k, v in out.items()}


def quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.int64)
    for j in range(x.shape[1]):
        qs = np.quantile(x[:, j], np.linspace(0, 1, n_bins + 1)[1:-1])
        out[:, j] = np.searchsorted(qs, x[:, j], side="right")
    return out


def stratified_take(
    candidates: list[str],
    strata: dict[str, tuple[int, int, int]],
    n: int,
    rng: np.random.Generator,
) -> list[str]:
    if n <= 0:
        return []
    by_stratum: dict[tuple[int, int, int], list[str]] = defaultdict(list)
    for sid in candidates:
        by_stratum[strata[sid]].append(sid)
    for ids in by_stratum.values():
        rng.shuffle(ids)

    selected: list[str] = []
    total = len(candidates)
    targets = {
        key: int(np.floor(len(ids) / max(1, total) * n))
        for key, ids in by_stratum.items()
    }
    for key, quota in targets.items():
        take = min(quota, len(by_stratum[key]), n - len(selected))
        selected.extend(by_stratum[key][:take])
        by_stratum[key] = by_stratum[key][take:]

    while len(selected) < n:
        available = [(key, ids) for key, ids in by_stratum.items() if ids]
        if not available:
            break
        # Prefer strata with the largest remaining pool to keep distributions close.
        max_len = max(len(ids) for _, ids in available)
        keys = [key for key, ids in available if len(ids) == max_len]
        key = keys[int(rng.integers(0, len(keys)))]
        selected.append(by_stratum[key].pop())
    return sorted(selected, key=identity_sort_key)


def nested_chain(
    pool: list[str],
    strata: dict[str, tuple[int, int, int]],
    seed: int,
    anchor: list[str] | None = None,
) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    selected = sorted(anchor or [], key=identity_sort_key)
    remaining = [sid for sid in pool if sid not in set(selected)]
    out: dict[str, list[str]] = {}
    for scale in SCALES:
        need = scale - len(selected)
        if need > 0:
            add = stratified_take(remaining, strata, need, rng)
            selected.extend(add)
            selected = sorted(selected, key=identity_sort_key)
            remaining = [sid for sid in remaining if sid not in set(add)]
        out[str(scale)] = selected[:scale]
    return out


def write_pair_files(out_dir: Path, chains: dict[str, dict[str, list[str]]]) -> list[dict]:
    pair_dir = out_dir / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for chain_name, chain in chains.items():
        chain_seed = SEED_REGISTRY["chain_seeds"].get(chain_name.split("_")[-1], SEED_REGISTRY["split_seed"])
        for scale_s, ids in chain.items():
            scale = int(scale_s)
            for curve, budget in (("primary", 870), ("secondary", min(scale * (scale - 1), 8700))):
                tier = tier_for_pair_manifest(chain_name, curve, scale)
                if tier is None:
                    continue
                pairs = coverage_balanced_ordered_pairs(ids, budget, int(chain_seed) + scale + (100000 if curve == "secondary" else 0))
                all_pairs_count = scale * (scale - 1)
                feasible_role_coverage = budget >= scale and scale > 1
                source_coverage = len({s for s, _ in pairs})
                target_coverage = len({t for _, t in pairs})
                if budget >= all_pairs_count and len(pairs) != all_pairs_count:
                    raise RuntimeError(f"{curve} {chain_name} scale{scale}: expected all {all_pairs_count} ordered pairs, got {len(pairs)}")
                if feasible_role_coverage and (source_coverage != scale or target_coverage != scale):
                    raise RuntimeError(
                        f"{curve} {chain_name} scale{scale}: coverage-balanced sampling failed "
                        f"(source={source_coverage}, target={target_coverage}, scale={scale})"
                    )
                path = pair_dir / f"{curve}_{chain_name}_scale{scale}.json"
                obj = {
                    "curve": curve,
                    "chain": chain_name,
                    "tier": tier,
                    "required_tier": tier == "required",
                    "scale": scale,
                    "seed": int(chain_seed) + scale + (100000 if curve == "secondary" else 0),
                    "budget": budget,
                    "n_pairs": len(pairs),
                    "all_ordered_pairs": all_pairs_count,
                    "all_ordered_pairs_used": len(pairs) == all_pairs_count,
                    "source_coverage": source_coverage,
                    "target_coverage": target_coverage,
                    "coverage_policy": "coverage-balanced ordered pairs; every identity appears at least once as source and target whenever feasible",
                    "pair_budget_policy": "primary fixed at 870; secondary fixed at min(N*(N-1), 8700)",
                    "ids": ids,
                    "pairs": [{"source_id": s, "target_id": t} for s, t in pairs],
                }
                atomic_write_json(path, obj)
                records.append({"path": str(path), "sha256": sha256_file(path), **{k: obj[k] for k in ("curve", "chain", "tier", "scale", "n_pairs", "source_coverage", "target_coverage")}})
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj-root", type=Path, required=True)
    parser.add_argument("--roi", type=Path, default=Path("roi/vertices.json"))
    parser.add_argument("--legacy-manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("splits/facescape_847"))
    parser.add_argument("--max-id", type=int, default=847, help="Highest FaceScape identity index to scan (ID space).")
    parser.add_argument("--expected-usable", type=int, default=846, help="Required count of QC-passing identities. FaceScape id 832 ships no mesh OBJ (textures/dpmaps only), so the usable count is 846.")
    parser.add_argument("--train-pool-size", type=int, default=676, help="train pool = expected_usable - 100 test - 70 val = 676.")
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--n-bins", type=int, default=4)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    roi_json = json.loads(args.roi.read_text(encoding="utf-8"))
    roi = np.asarray(roi_json["roi_indices"], dtype=np.int64)
    nverts_full = int(roi_json["vertex_count"])
    landmarks = [int(roi_json["landmarks_27_35"][str(k)]) for k in range(27, 36)]

    usable: list[str] = []
    failed: list[dict] = []
    qc_rows: list[dict] = []
    descriptors = []
    roi_hash_owner: dict[str, str] = {}
    for i in range(1, args.max_id + 1):
        sid = str(i)
        path = resolve_obj(args.obj_root, sid)
        if path is None:
            failed.append({"subject_id": sid, "reason": "missing_neutral_obj"})
            continue
        try:
            vertices, _ = parse_obj(path, want_faces=False)
            roi_vertices = vertices[roi]
            mesh_hash = roi_mesh_hash(roi_vertices)
            checks = [
                ("vertex_count", vertices.shape[0] == nverts_full),
                ("finite_vertices", bool(np.isfinite(vertices).all())),
                ("roi_crop_count", roi_vertices.shape[0] == 3934),
                ("landmarks_present", all(0 <= lm < vertices.shape[0] for lm in landmarks)),
            ]
            bad = [name for name, ok in checks if not ok]
            bad.extend(roi_degeneracy_reasons(roi_vertices))
            duplicate_of = roi_hash_owner.get(mesh_hash)
            if duplicate_of is not None:
                bad.append("duplicate_roi_mesh")
            if bad:
                failed.append({"subject_id": sid, "path": str(path), "reason": ",".join(bad), "duplicate_of": duplicate_of})
                continue
            usable.append(sid)
            roi_hash_owner[mesh_hash] = sid
            descriptors.append(roi_vertices.reshape(-1).astype(np.float64))
            qc_rows.append(
                {
                    "subject_id": sid,
                    "path": str(path),
                    "source_obj_sha256": sha256_file(path),
                    "roi_mesh_sha256_float32": mesh_hash,
                    "vertex_count": int(vertices.shape[0]),
                    "roi_count": int(roi_vertices.shape[0]),
                    "roi_extent": [float(x) for x in np.ptp(roi_vertices, axis=0)],
                    "landmarks_present": True,
                    "decision": "pass",
                }
            )
        except Exception as exc:
            failed.append({"subject_id": sid, "path": str(path), "reason": repr(exc)})

    if len(usable) != args.expected_usable:
        report = {
            "decision": "stop",
            "expected_usable": args.expected_usable,
            "max_id_scanned": args.max_id,
            "usable_count": len(usable),
            "failed_count": len(failed),
            "failed": failed,
        }
        atomic_write_json(args.out / "identity_qc_report.json", report)
        (args.out / "identity_qc_report.md").write_text(
            f"# Identity QC Report\n\nExpected {args.expected_usable} usable identities (scanning ids 1..{args.max_id}), found {len(usable)}. Split sizes were not rescaled.\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2))
        return 2

    mat = np.stack(descriptors, axis=0)
    centered = mat - mat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ vt[:3].T
    bins = quantile_bins(scores, args.n_bins)
    strata = {sid: tuple(int(x) for x in bins[j]) for j, sid in enumerate(usable)}

    legacy = legacy_splits(args.legacy_manifest)
    fixed_test = [sid for sid in legacy["test"] if sid in usable]
    fixed_val = [sid for sid in legacy["val"] if sid in usable]
    fixed_train = [sid for sid in legacy["train"] if sid in usable]
    if len(fixed_test) != 50 or len(fixed_val) != 10 or len(fixed_train) != 30:
        raise SystemExit("Legacy manifest must provide original 50 test, 10 validation and 30 train identities.")

    rng = np.random.default_rng(args.seed)
    remaining = [sid for sid in usable if sid not in set(fixed_test + fixed_val + fixed_train)]
    extra_test = stratified_take(remaining, strata, 100 - len(fixed_test), rng)
    remaining = [sid for sid in remaining if sid not in set(extra_test)]
    extra_val = stratified_take(remaining, strata, 70 - len(fixed_val), rng)
    remaining = [sid for sid in remaining if sid not in set(extra_val)]

    test_ids = sorted(fixed_test + extra_test, key=identity_sort_key)
    val_ids = sorted(fixed_val + extra_val, key=identity_sort_key)
    train_pool = sorted([sid for sid in usable if sid not in set(test_ids + val_ids)], key=identity_sort_key)
    if len(test_ids) != 100 or len(val_ids) != 70 or len(train_pool) != args.train_pool_size:
        raise SystemExit("Internal split-size error; refusing to write manifest.")

    chains: dict[str, dict[str, list[str]]] = {
        "anchor": nested_chain(train_pool, strata, args.seed, anchor=fixed_train)
    }
    for chain_idx in range(5):
        seed = int(SEED_REGISTRY["chain_seeds"][str(chain_idx)])
        chains[f"chain_{chain_idx}"] = nested_chain(train_pool, strata, seed, anchor=None)

    pair_records = write_pair_files(args.out, chains)
    required_pair_records = [r for r in pair_records if r["tier"] == "required"]
    required_by_key = {(r["curve"], r["chain"], int(r["scale"])) for r in required_pair_records}
    expected_required = {("primary", "anchor", 30)}
    expected_required.update(("primary", f"chain_{chain}", scale) for chain in range(3) for scale in SCALES)
    expected_required.update(("secondary", "chain_0", scale) for scale in SECONDARY_REQUIRED_SCALES)
    if required_by_key != expected_required:
        missing = sorted(expected_required - required_by_key)
        extra = sorted(required_by_key - expected_required)
        raise RuntimeError(f"Required pair-manifest set mismatch; missing={missing}, extra={extra}")
    rows = []
    for sid in test_ids:
        rows.append({"subject_id": sid, "split": "main test"})
    for sid in val_ids:
        rows.append({"subject_id": sid, "split": "clean-prior validation"})
    for sid in train_pool:
        rows.append({"subject_id": sid, "split": "train pool"})
    manifest = {
        "source": "build_splits.py",
        "seed_registry": SEED_REGISTRY,
        "split_seed": args.seed,
        "max_id": args.max_id,
        "expected_usable": args.expected_usable,
        "excluded_incomplete_ids": [str(r["subject_id"]) for r in failed if r.get("reason") == "missing_neutral_obj"],
        "usable_count": len(usable),
        "sizes": {"test": len(test_ids), "val": len(val_ids), "train_pool": len(train_pool)},
        "fixed_original": {"test": fixed_test, "val": fixed_val, "train": fixed_train},
        "stratification": {
            "descriptor": "first 3 PCA components of all-identity ROI vertex arrays; split-design metadata only",
            "n_bins_per_component": args.n_bins,
            "stratum_counts": {",".join(map(str, k)): v for k, v in Counter(strata.values()).items()},
        },
        "identity_qc": {
            "checks": [
                "full-head vertex count matches ROI metadata",
                "finite full-head vertices",
                "ROI crop count is 3934",
                "landmarks 27-35 present in full-head topology",
                "non-degenerate ROI coordinate extent",
                "no excessive duplicate ROI vertices",
                "no duplicate ROI mesh hashes across identities",
            ],
            "passed_count": len(qc_rows),
            "failed_count": len(failed),
            "rows": qc_rows,
        },
        "test_ids": test_ids,
        "val_ids": val_ids,
        "train_pool_ids": train_pool,
        "chains": chains,
        "pair_manifests": pair_records,
        "required_pair_manifests": required_pair_records,
        "pair_manifest_counts": {
            "required": len(required_pair_records),
            "optional": sum(1 for r in pair_records if r["tier"] == "optional"),
            "total": len(pair_records),
        },
        "rows": sorted(rows, key=lambda r: identity_sort_key(r["subject_id"])),
    }
    qc_report = {
        "decision": "pass",
        "expected_usable": args.expected_usable,
        "max_id_scanned": args.max_id,
        "usable_count": len(usable),
        "failed_count": len(failed),
        "failed": failed,
        "excluded_incomplete_ids": manifest["excluded_incomplete_ids"],
    }
    atomic_write_json(args.out / "identity_qc_report.json", qc_report)
    (args.out / "identity_qc_report.md").write_text(
        "# Identity QC Report\n\n"
        f"Decision: pass. Found {len(usable)} usable identities while scanning ids "
        f"1..{args.max_id}; excluded incomplete ids: "
        f"{', '.join(manifest['excluded_incomplete_ids']) or 'none'}.\n",
        encoding="utf-8",
    )
    write_seed_registry(args.out / "seed_registry.json")
    atomic_write_json(args.out / "split_manifest.json", manifest)
    print(json.dumps({"split_manifest": str(args.out / "split_manifest.json"), "sizes": manifest["sizes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
