from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rhinoform.data import edge_index
from rhinoform.repro import atomic_write_json
from scripts.training.residual_basis_expert import load_selected_rows, source_codes


def main() -> int:
    parser = argparse.ArgumentParser(description="Validation-only ridge/completion risk Pareto analysis.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base-model-package", required=True)
    parser.add_argument("--completion-package", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = torch.load(args.base_model_package, map_location="cpu", weights_only=False)
    completion = torch.load(args.completion_package, map_location="cpu", weights_only=False)
    pairs = [(str(a), str(b)) for a, b in base["val_pairs"]]
    val_ids = sorted({identity for pair in pairs for identity in pair}, key=int)
    by_id = load_selected_rows(Path(args.repo), set(val_ids), workers=32)
    template = next(iter(by_id.values()))
    landmarks = template["landmarks"]
    faces = template["faces"]
    edges = edge_index(faces)

    landmark_shapes = np.stack([by_id[sid]["vertices"][landmarks].reshape(-1) for sid in val_ids])
    completed = (
        (landmark_shapes - completion["x_mean"]) @ completion["weight"] + completion["y_mean"]
    ).astype(np.float32).reshape(len(val_ids), -1, 3)
    completed_by_id = dict(zip(val_ids, completed))

    source_code = source_codes(by_id, val_ids, base["source_pca"])
    controls = np.stack(
        [(by_id[target]["vertices"] - by_id[source]["vertices"])[landmarks].reshape(-1) for source, target in pairs]
    )
    condition = np.concatenate(
        [controls, np.stack([source_code[source] for source, _ in pairs])], axis=1
    )
    condition = (condition - base["cond_mean"]) / base["cond_std"]
    ridge_w = np.asarray(base["ridge_cond"][0], dtype=np.float32)
    ridge_b = np.asarray(base["ridge_cond"][1], dtype=np.float32)
    alphas = np.arange(0.0, 1.0 + args.alpha_step * 0.5, args.alpha_step)
    pair_metrics = np.empty((len(alphas), len(pairs), 3), dtype=np.float32)

    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start : start + args.batch_size]
        stop = start + len(batch)
        source = np.stack([by_id[src]["vertices"] for src, _ in batch])
        target = np.stack([by_id[tgt]["vertices"] for _, tgt in batch])
        ridge = (condition[start:stop] @ ridge_w + ridge_b).reshape(len(batch), -1, 3)
        completion_delta = np.stack([completed_by_id[tgt] for _, tgt in batch]) - source
        for alpha_index, alpha in enumerate(alphas):
            prediction = ridge + float(alpha) * (completion_delta - ridge)
            error = prediction - (target - source)
            pair_metrics[alpha_index, start:stop, 0] = np.sqrt(
                np.mean(np.sum(error * error, axis=2), axis=1)
            )
            edited = source + prediction
            source_edge = source[:, edges[:, 0]] - source[:, edges[:, 1]]
            edited_edge = edited[:, edges[:, 0]] - edited[:, edges[:, 1]]
            source_length = np.linalg.norm(source_edge, axis=2)
            strain = np.abs(np.linalg.norm(edited_edge, axis=2) - source_length) / np.maximum(source_length, 1e-8)
            pair_metrics[alpha_index, start:stop, 1] = np.percentile(strain, 95, axis=1)
            source_tri = source[:, faces]
            edited_tri = edited[:, faces]
            source_normal = np.cross(
                source_tri[:, :, 1] - source_tri[:, :, 0], source_tri[:, :, 2] - source_tri[:, :, 0]
            )
            edited_normal = np.cross(
                edited_tri[:, :, 1] - edited_tri[:, :, 0], edited_tri[:, :, 2] - edited_tri[:, :, 0]
            )
            pair_metrics[alpha_index, start:stop, 2] = 100.0 * np.mean(
                np.sum(source_normal * edited_normal, axis=2) < 0.0, axis=1
            )

    means = pair_metrics.mean(axis=1)
    ridge_mean = means[0]
    feasible = (means[:, 1] <= ridge_mean[1]) & (means[:, 2] <= ridge_mean[2])
    selected = int(np.argmin(np.where(feasible, means[:, 0], np.inf)))
    selected_pair = pair_metrics[selected]
    ridge_pair = pair_metrics[0]
    rows = [
        {
            "alpha": float(alpha),
            "roi_rmse": float(mean[0]),
            "edge_strain_p95": float(mean[1]),
            "normal_flip_pct": float(mean[2]),
            "strict_ridge_mean_feasible": bool(ok),
        }
        for alpha, mean, ok in zip(alphas, means, feasible)
    ]
    report = {
        "scope": "validation_only_no_test_access",
        "selection_rule": "minimum mean ROI subject to mean strain and flip no worse than ridge",
        "selected": rows[selected],
        "relative_roi_improvement_vs_ridge_pct": float(
            100.0 * (ridge_mean[0] - means[selected, 0]) / ridge_mean[0]
        ),
        "pair_roi_win_fraction_vs_ridge": float(np.mean(selected_pair[:, 0] < ridge_pair[:, 0])),
        "curve": rows,
    }
    np.savez_compressed(
        out_dir / "pair_metrics.npz",
        pairs=np.asarray(pairs, dtype=object),
        alphas=alphas,
        roi_rmse=pair_metrics[:, :, 0],
        edge_strain_p95=pair_metrics[:, :, 1],
        normal_flip_pct=pair_metrics[:, :, 2],
        selected_index=np.asarray([selected]),
    )
    atomic_write_json(out_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
