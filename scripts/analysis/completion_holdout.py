from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rhinoform.cvae import fit_truncated_pca
from rhinoform.data import edge_index, ridge_fit
from rhinoform.repro import atomic_write_json
from rhinoform.sampling import all_ordered_pairs, coverage_balanced_ordered_pairs
from scripts.training.residual_basis_expert import load_selected_rows
from scripts.training.shape_completion import fit_ridge, predict


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen internal holdout confirmation for target-shape completion.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base-model-package", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--holdout-size", type=int, default=80)
    parser.add_argument("--pair-budget", type=int, default=8700)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--completion-lambda", type=float, default=1.0)
    parser.add_argument("--ridge-lambda", type=float, default=100.0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = torch.load(args.base_model_package, map_location="cpu", weights_only=False)
    pool = np.asarray([str(value) for value in base["train_ids"]], dtype=object)
    order = np.random.default_rng(args.seed).permutation(len(pool))
    holdout_ids = sorted(pool[order[: args.holdout_size]].tolist(), key=int)
    train_ids = sorted(pool[order[args.holdout_size :]].tolist(), key=int)
    by_id = load_selected_rows(Path(args.repo), set(pool.tolist()), workers=32)
    template = next(iter(by_id.values()))
    landmarks = template["landmarks"]
    faces = template["faces"]
    edges = edge_index(faces)

    train_shapes = np.stack([by_id[sid]["vertices"].reshape(-1) for sid in train_ids]).astype(np.float64)
    source_pca = fit_truncated_pca(train_shapes, 16)
    source_codes = {
        sid: (by_id[sid]["vertices"].reshape(-1) - source_pca["mean"]) @ source_pca["components"].T
        for sid in pool
    }
    train_pairs = coverage_balanced_ordered_pairs(train_ids, args.pair_budget, args.seed)

    def raw_condition(pairs: list[tuple[str, str]]) -> np.ndarray:
        controls = np.stack(
            [(by_id[tgt]["vertices"] - by_id[src]["vertices"])[landmarks].reshape(-1) for src, tgt in pairs]
        )
        return np.concatenate([controls, np.stack([source_codes[src] for src, _ in pairs])], axis=1)

    x_train_raw = raw_condition(train_pairs)
    cond_mean = x_train_raw.mean(axis=0, keepdims=True)
    cond_std = np.maximum(x_train_raw.std(axis=0, keepdims=True), 1e-6)
    x_train = (x_train_raw - cond_mean) / cond_std
    y_train = np.stack(
        [(by_id[tgt]["vertices"] - by_id[src]["vertices"]).reshape(-1) for src, tgt in train_pairs]
    )
    ridge_model = ridge_fit(x_train, y_train, args.ridge_lambda)

    landmark_train = np.stack([by_id[sid]["vertices"][landmarks].reshape(-1) for sid in train_ids])
    completion_model = fit_ridge(landmark_train, train_shapes, args.completion_lambda)
    landmark_holdout = np.stack([by_id[sid]["vertices"][landmarks].reshape(-1) for sid in holdout_ids])
    completed = predict(completion_model, landmark_holdout).astype(np.float32).reshape(len(holdout_ids), -1, 3)
    completed_by_id = dict(zip(holdout_ids, completed))

    pairs = all_ordered_pairs(holdout_ids)
    x_eval = (raw_condition(pairs) - cond_mean) / cond_std
    pair_values = {"ridge": [[], [], []], "frozen_blend": [[], [], []]}
    for start in range(0, len(pairs), 64):
        batch = pairs[start : start + 64]
        stop = start + len(batch)
        source = np.stack([by_id[src]["vertices"] for src, _ in batch])
        target = np.stack([by_id[tgt]["vertices"] for _, tgt in batch])
        ridge = (x_eval[start:stop] @ ridge_model[0] + ridge_model[1]).reshape(len(batch), -1, 3)
        completion_delta = np.stack([completed_by_id[tgt] for _, tgt in batch]) - source
        for name, prediction in (
            ("ridge", ridge),
            ("frozen_blend", ridge + args.alpha * (completion_delta - ridge)),
        ):
            error = prediction - (target - source)
            pair_values[name][0].append(np.sqrt(np.mean(np.sum(error * error, axis=2), axis=1)))
            edited = source + prediction
            source_edge = source[:, edges[:, 0]] - source[:, edges[:, 1]]
            edited_edge = edited[:, edges[:, 0]] - edited[:, edges[:, 1]]
            source_length = np.linalg.norm(source_edge, axis=2)
            strain = np.abs(np.linalg.norm(edited_edge, axis=2) - source_length) / np.maximum(source_length, 1e-8)
            pair_values[name][1].append(np.percentile(strain, 95, axis=1))
            source_tri = source[:, faces]
            edited_tri = edited[:, faces]
            source_normal = np.cross(
                source_tri[:, :, 1] - source_tri[:, :, 0], source_tri[:, :, 2] - source_tri[:, :, 0]
            )
            edited_normal = np.cross(
                edited_tri[:, :, 1] - edited_tri[:, :, 0], edited_tri[:, :, 2] - edited_tri[:, :, 0]
            )
            pair_values[name][2].append(
                100.0 * np.mean(np.sum(source_normal * edited_normal, axis=2) < 0.0, axis=1)
            )

    arrays = {
        name: [np.concatenate(parts) for parts in metrics]
        for name, metrics in pair_values.items()
    }
    summaries = {
        name: {
            "roi_rmse": float(values[0].mean()),
            "edge_strain_p95": float(values[1].mean()),
            "normal_flip_pct": float(values[2].mean()),
        }
        for name, values in arrays.items()
    }
    improvement = arrays["ridge"][0] - arrays["frozen_blend"][0]
    source_means = np.asarray(
        [improvement[np.asarray([src == sid for src, _ in pairs])].mean() for sid in holdout_ids]
    )
    rng = np.random.default_rng(args.seed + 1)
    bootstrap = source_means[rng.integers(0, len(source_means), size=(20000, len(source_means)))].mean(axis=1)
    report = {
        "scope": "post-development_internal_holdout_retrained_no_original_validation_or_test",
        "frozen_hyperparameters": {"alpha": args.alpha, "completion_lambda": args.completion_lambda},
        "n_train_ids": len(train_ids),
        "n_holdout_ids": len(holdout_ids),
        "n_train_pairs": len(train_pairs),
        "n_holdout_pairs": len(pairs),
        "holdout_ids": holdout_ids,
        "metrics": summaries,
        "relative_roi_improvement_pct": float(
            100.0 * (summaries["ridge"]["roi_rmse"] - summaries["frozen_blend"]["roi_rmse"])
            / summaries["ridge"]["roi_rmse"]
        ),
        "pair_roi_win_fraction": float(np.mean(improvement > 0.0)),
        "source_identity_win_fraction": float(np.mean(source_means > 0.0)),
        "source_identity_bootstrap_absolute_improvement_95ci": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
    }
    atomic_write_json(out_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
