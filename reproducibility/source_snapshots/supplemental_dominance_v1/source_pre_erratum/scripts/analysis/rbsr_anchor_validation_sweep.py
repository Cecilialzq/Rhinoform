"""Validation-only Ridge anchor sweep for the RB-SR dominance search.

The sweep reuses a train-only source-PCA basis, refits Ridge on a frozen larger
pair manifest, ranks the full dimension/lambda grid by strict free-ROI RMSE,
and strict-scores the predeclared top-k candidates for flip and strain.  It
never loads test identities.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from rhinoform.data import load_rows
from rhinoform.repro import (
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
    valid_sha256_sidecar,
    validate_torch_artifact,
    write_sha256_sidecar,
)
from rhinoform.sampling import read_pairs
from rhinoform.strict_protocol_patch import strict_metric_rows


CORE_METRICS = ("roi_rmse", "normal_flip_pct", "edge_strain_p95")


def parse_ints(raw: str) -> list[int]:
    values = sorted(set(int(value.strip()) for value in raw.split(",") if value.strip()))
    if not values or any(value < 1 for value in values):
        raise ValueError("dimensions must be positive")
    return values


def parse_floats(raw: str) -> list[float]:
    values = sorted(set(float(value.strip()) for value in raw.split(",") if value.strip()))
    if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("lambdas must be finite and positive")
    return values


def summary_values(payload: dict[str, object]) -> dict[str, float]:
    values = payload.get("means", payload.get("summary"))
    if not isinstance(values, dict):
        raise ValueError("Reference summary has no means/summary object")
    return {metric: float(values[metric]) for metric in CORE_METRICS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--train-pairs", type=Path, required=True)
    parser.add_argument("--source-pca-package", type=Path, required=True)
    parser.add_argument("--reference-validation-summary", type=Path, required=True)
    parser.add_argument("--dimensions", default="32,64,96,128")
    parser.add_argument("--lambdas", default="1,3,10,30,100,300")
    parser.add_argument("--strict-top-k", type=int, default=12)
    parser.add_argument("--relative-margin", type=float, default=0.0)
    parser.add_argument("--score-chunk-pairs", type=int, default=320)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dimensions = parse_ints(args.dimensions)
    lambdas = parse_floats(args.lambdas)
    if args.strict_top_k < 1 or args.score_chunk_pairs < 1:
        raise ValueError("strict-top-k and score-chunk-pairs must be positive")
    if not 0.0 <= args.relative_margin < 1.0:
        raise ValueError("relative-margin must lie in [0,1)")
    for path in (args.split_manifest, args.train_pairs, args.reference_validation_summary):
        if not valid_sha256_sidecar(path):
            raise RuntimeError(f"Artifact or SHA-256 sidecar is invalid: {path}")
    if not validate_torch_artifact(
        args.source_pca_package,
        required_keys=("source_pca", "train_ids", "val_ids", "feature_template_sha256"),
        repair_sidecar=False,
    ):
        raise RuntimeError("Source-PCA package or sidecar is invalid")

    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    train_ids = [str(value) for value in split["train_pool_ids"]]
    val_ids = [str(value) for value in split["val_ids"]]
    source_package = torch.load(args.source_pca_package, map_location="cpu", weights_only=False)
    if [str(value) for value in source_package["train_ids"]] != train_ids:
        raise ValueError("Source-PCA package training identities do not match the split")
    if [str(value) for value in source_package["val_ids"]] != val_ids:
        raise ValueError("Source-PCA package validation identities do not match the split")
    source_pca = source_package["source_pca"]
    components = np.asarray(source_pca["components"], dtype=np.float32)
    if components.shape[0] < max(dimensions):
        raise ValueError("Source-PCA package has too few components for the declared grid")
    pca_mean = np.asarray(source_pca["mean"], dtype=np.float32)

    train_pairs = [(str(a), str(b)) for a, b in read_pairs(args.train_pairs)]
    if not train_pairs:
        raise ValueError("Training pair manifest is empty")
    allowed_train = set(train_ids)
    if any(a not in allowed_train or b not in allowed_train or a == b for a, b in train_pairs):
        raise ValueError("Training pair manifest is not contained in the training identities")
    val_pairs = [(a, b) for a in val_ids for b in val_ids if a != b]
    if len(val_pairs) != 4830:
        raise AssertionError(len(val_pairs))
    print(f"Loading {len(train_ids) + len(val_ids)} train/validation identities", flush=True)
    _, by_id = load_rows(args.repo, allowed_ids=set(train_ids) | set(val_ids))
    template = next(iter(by_id.values()))
    landmarks = np.asarray(template["landmarks"], dtype=np.int64)
    max_dim = max(dimensions)
    max_components = components[:max_dim]
    codes = {
        subject_id: (by_id[subject_id]["vertices"].reshape(-1) - pca_mean) @ max_components.T
        for subject_id in train_ids + val_ids
    }

    def build_arrays(pairs: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        x = np.empty((len(pairs), len(landmarks) * 3 + max_dim), dtype=np.float32)
        y = np.empty((len(pairs), template["vertices"].size), dtype=np.float32)
        for index, (source_id, target_id) in enumerate(pairs):
            delta = by_id[target_id]["vertices"] - by_id[source_id]["vertices"]
            x[index, : len(landmarks) * 3] = delta[landmarks].reshape(-1)
            x[index, len(landmarks) * 3 :] = codes[source_id]
            y[index] = delta.reshape(-1)
        return x, y

    print(f"Building frozen train pairs={len(train_pairs)} and validation pairs={len(val_pairs)}", flush=True)
    x_train, y_train = build_arrays(train_pairs)
    x_val, y_val = build_arrays(val_pairs)
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = np.maximum(x_train.std(axis=0, keepdims=True), 1e-6)
    x_train = (x_train - x_mean) / x_std
    x_val = (x_val - x_mean) / x_std
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_train -= y_mean
    print("Computing shared Ridge normal equations", flush=True)
    xtx = x_train.T @ x_train
    xty = x_train.T @ y_train
    free = np.ones(template["vertices"].shape[0], dtype=bool)
    free[landmarks] = False

    candidates: list[dict[str, object]] = []
    weights: dict[tuple[int, float], np.ndarray] = {}
    for dimension in dimensions:
        width = len(landmarks) * 3 + dimension
        a = xtx[:width, :width].astype(np.float64)
        b = xty[:width].astype(np.float64)
        for ridge_lambda in lambdas:
            weight = np.linalg.solve(a + ridge_lambda * np.eye(width), b)
            total = 0.0
            for start in range(0, len(val_pairs), args.score_chunk_pairs):
                stop = min(start + args.score_chunk_pairs, len(val_pairs))
                prediction = x_val[start:stop, :width] @ weight + y_mean
                difference = (
                    prediction.reshape(-1, template["vertices"].shape[0], 3)[:, free]
                    - y_val[start:stop].reshape(-1, template["vertices"].shape[0], 3)[:, free]
                )
                total += float(
                    np.sqrt(np.mean(np.sum(difference * difference, axis=2), axis=1)).sum()
                )
            record = {
                "label": f"pca{dimension}_lambda{ridge_lambda:g}".replace(".", "p"),
                "source_pca_dim": dimension,
                "ridge_lambda": ridge_lambda,
                "roi_rmse": total / len(val_pairs),
                "strict_scored": False,
                "normal_flip_pct": "",
                "edge_strain_p95": "",
                "dominates_reference_core": "",
                "relative_improvement_vs_reference_roi_rmse": "",
                "relative_improvement_vs_reference_normal_flip_pct": "",
                "relative_improvement_vs_reference_edge_strain_p95": "",
                "minimum_core_relative_improvement": "",
            }
            candidates.append(record)
            weights[(dimension, ridge_lambda)] = weight
            print(
                f"ANCHOR SCREEN {record['label']} val_roi_rmse={record['roi_rmse']:.6f}",
                flush=True,
            )

    reference_payload = json.loads(args.reference_validation_summary.read_text(encoding="utf-8"))
    if reference_payload.get("split") != "validation" or int(reference_payload.get("n_pairs", 0)) != 4830:
        raise ValueError("Reference must be a complete validation evaluation")
    reference = summary_values(reference_payload)
    ranked = sorted(candidates, key=lambda row: (float(row["roi_rmse"]), str(row["label"])))
    to_score = ranked[: min(args.strict_top_k, len(ranked))]
    for record in to_score:
        dimension = int(record["source_pca_dim"])
        ridge_lambda = float(record["ridge_lambda"])
        width = len(landmarks) * 3 + dimension
        prediction = (x_val[:, :width] @ weights[(dimension, ridge_lambda)] + y_mean).astype(np.float32)
        rows: list[dict[str, object]] = []
        for start in range(0, len(val_pairs), args.score_chunk_pairs):
            stop = min(start + args.score_chunk_pairs, len(val_pairs))
            rows.extend(strict_metric_rows(by_id, val_pairs[start:stop], prediction[start:stop]))
            print(f"ANCHOR STRICT {record['label']} {stop}/{len(val_pairs)}", flush=True)
        record["roi_rmse"] = float(np.mean([float(row["roi_rmse"]) for row in rows]))
        record["normal_flip_pct"] = float(np.mean([float(row["normal_flip_pct"]) for row in rows]))
        record["edge_strain_p95"] = float(np.mean([float(row["edge_strain_p95"]) for row in rows]))
        record["strict_scored"] = True
        record["dominates_reference_core"] = all(
            float(record[metric]) < reference[metric] * (1.0 - args.relative_margin)
            for metric in CORE_METRICS
        )
        for metric in CORE_METRICS:
            record[f"relative_improvement_vs_reference_{metric}"] = (
                reference[metric] - float(record[metric])
            ) / reference[metric]
        record["minimum_core_relative_improvement"] = min(
            float(record[f"relative_improvement_vs_reference_{metric}"])
            for metric in CORE_METRICS
        )
        print(json.dumps(record), flush=True)

    feasible = [row for row in to_score if bool(row.get("dominates_reference_core", False))]
    selected = max(
        feasible,
        key=lambda row: (
            float(row["minimum_core_relative_improvement"]),
            -float(row["roi_rmse"]),
            -float(row["normal_flip_pct"]),
            -float(row["edge_strain_p95"]),
            str(row["label"]),
        ),
    ) if feasible else None
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "ridge_anchor_validation_sweep.csv"
    atomic_write_csv(summary_path, candidates)
    freeze = {
        "status": "FROZEN_VALIDATION_SELECTED_RBSR_ANCHOR" if selected else "NO_LAMM_DOMINATING_RIDGE_ANCHOR",
        "split": "validation",
        "test_access": False,
        "selection_policy": (
            "Maximise the minimum relative validation improvement over LAMM across ROI RMSE, "
            "new flip, and edge strain among candidates satisfying the declared margin."
        ),
        "n_validation_pairs": len(val_pairs),
        "n_train_pairs": len(train_pairs),
        "dimensions": dimensions,
        "lambdas": lambdas,
        "strict_top_k": args.strict_top_k,
        "relative_margin": args.relative_margin,
        "reference": reference,
        "selected": selected,
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "train_pair_manifest_sha256": sha256_file(args.train_pairs),
        "source_pca_package_sha256": sha256_file(args.source_pca_package),
        "reference_validation_summary_sha256": sha256_file(args.reference_validation_summary),
        "summary_csv": summary_path.name,
        "summary_csv_sha256": sha256_file(summary_path),
    }
    freeze_path = args.out / "RBSR_ANCHOR_SELECTION_FREEZE.json"
    atomic_write_json(freeze_path, freeze)
    write_sha256_sidecar(freeze_path)
    print(json.dumps(freeze, indent=2), flush=True)
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
