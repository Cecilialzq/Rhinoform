"""Full-set classical deformation baselines with per-pair metrics.

Runs Laplacian-handle, bi-Laplacian-handle and (vectorised) ARAP-handle
baselines on every held-out test pair, under the *same* ROI topology, metric
set and pair ordering as the neural/hybrid pipeline. For each method it writes a
per-pair metrics CSV in the identity-bootstrap schema
(``pair_index,source_id,target_id,roi_rmse,landmark_rmse,dorsum_rmse,tip_rmse,
edge_strain_p95,normal_flip_pct``) so the existing identity-level bootstrap /
Holm machinery can consume classical baselines exactly like learned methods.

ARAP is fully vectorised (batched 3x3 SVD for the local-rotation fit, scatter-add
for the right-hand side), making the full 2450-pair run tractable in a single
process instead of the ~16 min pure-Python loop.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized

from .data import (
    edge_index,
    face_normals,
    fit_pca,
    load_rows,
    ordered_pairs,
    pair_arrays,
    pca_coeff,
    ridge_fit,
    ridge_predict,
    split_ids,
)
from .geometry import (
    graph_edges,
    selector_matrix,
    solve_linear_handle_baseline,
    uniform_laplacian,
)


def arap_predict_vectorised(
    sources: list[np.ndarray],
    controls: np.ndarray,
    landmarks: np.ndarray,
    faces: np.ndarray,
    laplacian: sparse.csr_matrix,
    init_pred: np.ndarray,
    handle_weight: float,
    ridge: float,
    n_iter: int,
    *,
    output: np.ndarray | None = None,
    start_index: int = 0,
    checkpoint_every: int = 0,
    checkpoint_callback=None,
) -> np.ndarray:
    """Vectorised ARAP with soft handle constraints. ``sources`` is the per-pair
    source vertex array (shared topology), ``controls`` are landmark deltas."""
    n = sources[0].shape[0]
    edges = graph_edges(faces)
    e0, e1 = edges[:, 0], edges[:, 1]
    pmat = selector_matrix(landmarks, n)
    a = laplacian + float(handle_weight) * (pmat.T @ pmat) + float(ridge) * sparse.eye(n, format="csr")
    solve = factorized(a.tocsc())
    pred = np.zeros((len(sources), n * 3), dtype=np.float64) if output is None else output
    if pred.shape != (len(sources), n * 3):
        raise ValueError(f"ARAP output has shape {pred.shape}, expected {(len(sources), n * 3)}")

    for k in range(int(start_index), len(sources)):
        src = sources[k]
        target_handles = src[landmarks] + controls[k]
        q = src + init_pred[k].reshape(n, 3)
        d_src = src[e0] - src[e1]  # (E,3) rest edge vectors
        for _ in range(int(n_iter)):
            d_q = q[e0] - q[e1]
            # per-edge outer products q_edge x p_edge -> accumulate covariance at both endpoints
            outer = d_q[:, :, None] * d_src[:, None, :]  # (E,3,3)
            cov = np.zeros((n, 3, 3), dtype=np.float64)
            np.add.at(cov, e0, outer)
            np.add.at(cov, e1, outer)
            u, _, vt = np.linalg.svd(cov)
            r = np.matmul(u, vt)
            det = np.linalg.det(r)
            flip = det < 0.0
            if np.any(flip):
                u[flip, :, -1] *= -1.0
                r[flip] = np.matmul(u[flip], vt[flip])
            # rhs: sum over edges of 0.5 (R_i + R_j) (p_i - p_j)
            term = 0.5 * np.matmul((r[e0] + r[e1]), d_src[:, :, None])[:, :, 0]  # (E,3)
            b = np.zeros((n, 3), dtype=np.float64)
            np.add.at(b, e0, term)
            np.add.at(b, e1, -term)
            b += float(handle_weight) * (pmat.T @ target_handles)
            for d in range(3):
                q[:, d] = solve(b[:, d])
        pred[k] = (q - src).reshape(-1)
        completed = k + 1
        if checkpoint_callback is not None and (
            completed == len(sources) or (checkpoint_every > 0 and completed % checkpoint_every == 0)
        ):
            checkpoint_callback(completed, pred)
    return pred


def per_pair_metrics(name: str, by_id: dict, pairs, pred_flat: np.ndarray, faces, edges, subunits, landmarks) -> list[dict]:
    rows = []
    fn0_cache: dict[str, np.ndarray] = {}
    e0_cache: dict[str, np.ndarray] = {}
    for i, (s, t) in enumerate(pairs):
        src = by_id[s]["vertices"]
        tgt = by_id[t]["vertices"]
        true_delta = (tgt - src).reshape(-1, 3)
        pred_delta = pred_flat[i].reshape(-1, 3)
        pred_v = src + pred_delta

        def rmse(idx=None):
            pd, td = pred_delta, true_delta
            if idx is not None:
                pd, td = pd[idx], td[idx]
            return float(np.sqrt(np.mean(np.sum((pd - td) ** 2, axis=1))))

        if s not in e0_cache:
            e0_cache[s] = np.linalg.norm(src[edges[:, 0]] - src[edges[:, 1]], axis=1)
            fn0_cache[s] = face_normals(src, faces)
        e0 = e0_cache[s]
        e1 = np.linalg.norm(pred_v[edges[:, 0]] - pred_v[edges[:, 1]], axis=1)
        strain = np.abs(e1 - e0) / np.maximum(e0, 1e-12)
        n1 = face_normals(pred_v, faces)
        flip = float(np.mean(np.sum(fn0_cache[s] * n1, axis=1) < 0.0) * 100.0)
        rows.append(
            {
                "pair_index": float(i),
                "source_id": s,
                "target_id": t,
                "roi_rmse": rmse(),
                "landmark_rmse": rmse(landmarks),
                "dorsum_rmse": rmse(subunits["dorsum"]),
                "tip_rmse": rmse(subunits["tip"]),
                "edge_strain_p95": float(np.percentile(strain, 95)),
                "normal_flip_pct": flip,
            }
        )
    return rows


def aggregate(name: str, rows: list[dict]) -> dict:
    out = {"method": name, "n_pairs": float(len(rows))}
    for k in ("roi_rmse", "landmark_rmse", "dorsum_rmse", "tip_rmse", "edge_strain_p95", "normal_flip_pct"):
        out[k] = float(np.mean([r[k] for r in rows]))
    return out


def write_pair_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="../data")
    parser.add_argument("--out", default="../results/classical")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--handle-weight", type=float, default=1000.0)
    parser.add_argument("--system-ridge", type=float, default=1e-8)
    parser.add_argument("--arap-iter", type=int, default=3)
    parser.add_argument("--source-pca-dim", type=int, default=16)
    parser.add_argument("--ridge-lambda", type=float, default=100.0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(args.repo)
    _, by_id = load_rows(repo)
    train_ids = split_ids(by_id, "clean-prior train")
    test_ids = split_ids(by_id, "main test")
    train_pairs = ordered_pairs(train_ids, None, args.seed)
    test_pairs = ordered_pairs(test_ids, None, args.seed)

    template = next(iter(by_id.values()))
    landmarks = template["landmarks"]
    faces = template["faces"]
    edges = edge_index(faces)
    subunits = template["subunits"]
    n = template["vertices"].shape[0]

    controls = np.stack([(by_id[t]["vertices"] - by_id[s]["vertices"])[landmarks] for s, t in test_pairs], axis=0)
    lap = uniform_laplacian(n, faces)

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    laplacian_pred = solve_linear_handle_baseline(controls, landmarks, n, lap, args.handle_weight, args.system_ridge)
    timings["laplacian"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    bilaplacian_pred = solve_linear_handle_baseline(controls, landmarks, n, lap @ lap, args.handle_weight, args.system_ridge)
    timings["bilaplacian"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    sources = [by_id[s]["vertices"] for s, _ in test_pairs]
    arap_pred = arap_predict_vectorised(
        sources, controls, landmarks, faces, lap,
        laplacian_pred, args.handle_weight, args.system_ridge, args.arap_iter,
    )
    timings["arap"] = time.perf_counter() - t0

    # ridge + source-PCA conditional baseline (kept for parity with prior table)
    src_pca = fit_pca(np.stack([by_id[i]["vertices"].reshape(-1) for i in train_ids]), args.source_pca_dim)

    def cond(pairs):
        ctrl, sf = [], []
        for s, t in pairs:
            ctrl.append((by_id[t]["vertices"] - by_id[s]["vertices"])[landmarks].reshape(-1))
            sf.append(by_id[s]["vertices"].reshape(-1))
        return np.concatenate([np.stack(ctrl), pca_coeff(np.stack(sf), src_pca)], axis=1)

    x_tr, x_te = cond(train_pairs), cond(test_pairs)
    mu, sd = x_tr.mean(0, keepdims=True), np.maximum(x_tr.std(0, keepdims=True), 1e-6)
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    _, y_tr = pair_arrays(by_id, train_pairs)
    ridge_pred = ridge_predict(x_te, ridge_fit(x_tr, y_tr, args.ridge_lambda))

    methods = {
        "laplacian_handles": laplacian_pred,
        "bilaplacian_handles": bilaplacian_pred,
        f"arap_handles_iter{args.arap_iter}": arap_pred,
        "ridge_sourcepca": ridge_pred,
    }
    summary = {"n_test_pairs": len(test_pairs), "timings_sec": timings, "results": []}
    arrays = {"test_pairs": np.asarray(test_pairs, dtype=object)}
    for name, pred in methods.items():
        rows = per_pair_metrics(name, by_id, test_pairs, pred, faces, edges, subunits, landmarks)
        write_pair_csv(out_dir / f"identity_bootstrap_pair_metrics_{name}.csv", rows)
        summary["results"].append(aggregate(name, rows))
        arrays[f"{name}_pred"] = np.asarray(pred, dtype=np.float32)

    np.savez_compressed(out_dir / "classical_deformation_predictions_all.npz", **arrays)
    (out_dir / "classical_deformation_summary_all.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
