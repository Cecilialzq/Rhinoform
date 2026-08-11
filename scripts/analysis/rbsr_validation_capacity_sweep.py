"""Validation-only Ridge capacity selection for the clean RB-SR extension.

This program deliberately never loads a test mesh. It selects the source-PCA
dimension and Ridge regularisation before the CVAE/residual proposer and RB-SR
gate are trained. The selected pair is therefore a frozen input to all later
test evaluation, not a choice made from the 9,900 test pairs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rhinoform.cvae import fit_truncated_pca, flatten_vertices
from rhinoform.data import assert_identity_disjoint, load_rows, pair_arrays, ridge_fit, ridge_predict
from rhinoform.repro import atomic_write_csv, atomic_write_json, sha256_file, sha256_json
from rhinoform.sampling import all_ordered_pairs, read_pairs
from rhinoform.strict_protocol_patch import strict_metric_rows
from rhinoform.train import pair_conditions


METRICS = ("roi_rmse", "normal_flip_pct", "abs_flip_pct", "edge_strain_p95")


def parse_numbers(value: str, cast) -> list:
    result = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("The sweep grid must not be empty")
    return result


def mean_metrics(rows: list[dict]) -> dict[str, float]:
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in METRICS
    }


def pareto_rows(rows: list[dict]) -> list[dict]:
    frontier: list[dict] = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            no_worse = all(float(other[key]) <= float(candidate[key]) for key in METRICS)
            strictly_better = any(float(other[key]) < float(candidate[key]) for key in METRICS)
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda row: (float(row["roi_rmse"]), float(row["normal_flip_pct"])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--train-pairs-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dimensions", default="16,32,64,96,128")
    parser.add_argument("--lambdas", default="30,100,300")
    parser.add_argument("--canonical-dimension", type=int, default=16)
    parser.add_argument("--canonical-lambda", type=float, default=100.0)
    parser.add_argument("--max-flip-ratio", type=float, default=1.05)
    parser.add_argument("--max-strain-ratio", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()

    dimensions = parse_numbers(args.dimensions, int)
    lambdas = parse_numbers(args.lambdas, float)
    if args.canonical_dimension not in dimensions or args.canonical_lambda not in lambdas:
        raise ValueError("The canonical dimension/lambda must be present in the sweep grid")
    if args.max_flip_ratio < 1.0 or args.max_strain_ratio < 1.0:
        raise ValueError("Safety ratios must be >= 1 so the canonical reference remains feasible")
    args.out.mkdir(parents=True, exist_ok=True)
    progress_path = args.out / "ridge_capacity_validation_progress.json"
    progress_signature = sha256_json({
        "schema": "rbsr_validation_capacity_v2",
        "reported_metrics": list(METRICS),
        "dimensions": dimensions,
        "lambdas": lambdas,
        "canonical_dimension": args.canonical_dimension,
        "canonical_lambda": args.canonical_lambda,
        "max_flip_ratio": args.max_flip_ratio,
        "max_strain_ratio": args.max_strain_ratio,
        "seed": args.seed,
        "data_manifest_sha256": sha256_file(args.repo / "manifest.json"),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "train_pairs_sha256": sha256_file(args.train_pairs_json),
    })
    progress = {"signature": progress_signature, "status": "IN_PROGRESS", "completed_rows": []}
    if progress_path.is_file():
        try:
            saved = json.loads(progress_path.read_text(encoding="utf-8"))
            if saved.get("signature") == progress_signature:
                progress = saved
        except (OSError, json.JSONDecodeError):
            pass
    completed = {
        (int(row["source_pca_dim"]), float(row["ridge_lambda"])): row
        for row in progress.get("completed_rows", [])
    }

    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    train_ids = [str(value) for value in split["train_pool_ids"]]
    val_ids = [str(value) for value in split["val_ids"]]
    test_ids = [str(value) for value in split["test_ids"]]
    assert_identity_disjoint({"train": train_ids, "validation": val_ids, "test": test_ids})
    if (len(train_ids), len(val_ids), len(test_ids)) != (676, 70, 100):
        raise ValueError("Expected the frozen 676/70/100 split")

    train_pairs = read_pairs(args.train_pairs_json)
    train_pair_ids = {value for pair in train_pairs for value in pair}
    if not train_pair_ids.issubset(set(train_ids)):
        raise ValueError("Training pair manifest contains a held-out identity")
    val_pairs = all_ordered_pairs(val_ids)
    if len(val_pairs) != 4830:
        raise ValueError("Expected all 4,830 ordered validation pairs")

    # Critical selection boundary: test IDs are known from the split manifest,
    # but their mesh files are never opened in this process.
    _, by_id = load_rows(args.repo, allowed_ids=set(train_ids) | set(val_ids))
    y_train = pair_arrays(by_id, train_pairs)[1]
    source_matrix = flatten_vertices(by_id, train_ids)
    for dimension in dimensions:
        pending_lambdas = [value for value in lambdas if (dimension, float(value)) not in completed]
        if not pending_lambdas:
            print(f"CAPACITY SWEEP resume from Drive: dimension={dimension} complete", flush=True)
            continue
        print(f"Fitting train-only source PCA dimension={dimension}", flush=True)
        source_pca = fit_truncated_pca(source_matrix, dimension)
        cond_train, _, _, _, _, cond_mean, cond_std = pair_conditions(by_id, train_pairs, source_pca)
        cond_val, _, _, _, _, _, _ = pair_conditions(by_id, val_pairs, source_pca, cond_mean, cond_std)
        for ridge_lambda in pending_lambdas:
            print(f"CAPACITY SWEEP live: dim={dimension} lambda={ridge_lambda:g}", flush=True)
            ridge = ridge_fit(cond_train, y_train, ridge_lambda)
            prediction = ridge_predict(cond_val, ridge)
            metrics = mean_metrics(strict_metric_rows(by_id, val_pairs, prediction))
            completed[(dimension, float(ridge_lambda))] = {
                "source_pca_dim": int(dimension),
                "ridge_lambda": float(ridge_lambda),
                **metrics,
                "n_train_ids": len(train_ids),
                "n_validation_ids": len(val_ids),
                "n_validation_pairs": len(val_pairs),
                "test_meshes_loaded": False,
            }
            progress["completed_rows"] = [
                completed[(declared_dimension, float(declared_lambda))]
                for declared_dimension in dimensions
                for declared_lambda in lambdas
                if (declared_dimension, float(declared_lambda)) in completed
            ]
            atomic_write_json(progress_path, progress)
            atomic_write_csv(args.out / "ridge_capacity_validation_grid_progress.csv", progress["completed_rows"])
            print(f"CAPACITY SWEEP persisted to Drive: dim={dimension} lambda={ridge_lambda:g}", flush=True)

    rows = [completed[(dimension, float(ridge_lambda))] for dimension in dimensions for ridge_lambda in lambdas]

    canonical = next(
        row for row in rows
        if int(row["source_pca_dim"]) == args.canonical_dimension
        and float(row["ridge_lambda"]) == args.canonical_lambda
    )
    flip_limit = float(canonical["normal_flip_pct"]) * args.max_flip_ratio
    strain_limit = float(canonical["edge_strain_p95"]) * args.max_strain_ratio
    for row in rows:
        row["within_canonical_safety_envelope"] = bool(
            float(row["normal_flip_pct"]) <= flip_limit
            and float(row["edge_strain_p95"]) <= strain_limit
        )
    feasible = [row for row in rows if row["within_canonical_safety_envelope"]]
    selected = min(
        feasible,
        key=lambda row: (
            float(row["roi_rmse"]),
            float(row["normal_flip_pct"]),
            float(row["edge_strain_p95"]),
            int(row["source_pca_dim"]),
        ),
    )
    frontier = pareto_rows(rows)

    atomic_write_csv(args.out / "ridge_capacity_validation_grid.csv", rows)
    atomic_write_csv(args.out / "ridge_capacity_validation_pareto.csv", frontier)
    report = {
        "status": "FROZEN_VALIDATION_SELECTION",
        "role": "post_hoc_representation_capacity_extension",
        "test_access": "No test mesh was loaded or scored by this selector.",
        "selection_policy": (
            "Minimise validation free-ROI RMSE among candidates whose target-relative new-flip "
            "and edge-strain p95 are each within the declared ratio of PCA16/lambda100; ties "
            "prefer lower new-flip, lower strain and lower dimension."
        ),
        "reported_safety_metrics": ["normal_flip_pct", "abs_flip_pct", "edge_strain_p95"],
        "safety_limits": {
            "canonical_source_pca_dim": args.canonical_dimension,
            "canonical_ridge_lambda": args.canonical_lambda,
            "max_flip_ratio": args.max_flip_ratio,
            "max_strain_ratio": args.max_strain_ratio,
            "normal_flip_pct": flip_limit,
            "edge_strain_p95": strain_limit,
        },
        "selected": selected,
        "grid": {"dimensions": dimensions, "lambdas": lambdas},
        "protocol": {
            "train_identities": len(train_ids),
            "validation_identities": len(val_ids),
            "test_identities_reserved": len(test_ids),
            "train_pairs": len(train_pairs),
            "validation_pairs": len(val_pairs),
            "seed": args.seed,
        },
        "input_sha256": {
            "data_manifest": sha256_file(args.repo / "manifest.json"),
            "split_manifest": sha256_file(args.split_manifest),
            "train_pair_manifest": sha256_file(args.train_pairs_json),
        },
        "pareto_rows": frontier,
    }
    atomic_write_json(args.out / "ridge_capacity_selection.json", report)
    progress["status"] = "COMPLETE"
    progress["selection_sha256"] = sha256_file(args.out / "ridge_capacity_selection.json")
    atomic_write_json(progress_path, progress)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
