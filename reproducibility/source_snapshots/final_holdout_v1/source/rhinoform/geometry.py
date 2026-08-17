from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized

from .data import (
    evaluate_method,
    fit_pca,
    load_rows,
    ordered_pairs,
    pair_arrays,
    pca_coeff,
    ridge_fit,
    ridge_predict,
    split_ids,
)


def graph_edges(faces: np.ndarray) -> np.ndarray:
    edges = set()
    for a, b, c in np.asarray(faces, dtype=np.int64):
        for u, v in ((a, b), (b, c), (c, a)):
            u, v = int(u), int(v)
            if u > v:
                u, v = v, u
            edges.add((u, v))
    return np.asarray(sorted(edges), dtype=np.int64)


def uniform_laplacian(n_vertices: int, faces: np.ndarray) -> sparse.csr_matrix:
    edges = graph_edges(faces)
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(row), dtype=np.float64)
    adj = sparse.coo_matrix((data, (row, col)), shape=(n_vertices, n_vertices)).tocsr()
    deg = np.asarray(adj.sum(axis=1)).reshape(-1)
    return sparse.diags(deg) - adj


def selector_matrix(indices: np.ndarray, n_vertices: int) -> sparse.csr_matrix:
    indices = np.asarray(indices, dtype=np.int64)
    rows = np.arange(len(indices), dtype=np.int64)
    data = np.ones(len(indices), dtype=np.float64)
    return sparse.coo_matrix((data, (rows, indices)), shape=(len(indices), n_vertices)).tocsr()


def solve_linear_handle_baseline(
    controls: np.ndarray,
    landmarks: np.ndarray,
    n_vertices: int,
    smooth_operator: sparse.csr_matrix,
    handle_weight: float,
    ridge: float,
) -> np.ndarray:
    p = selector_matrix(landmarks, n_vertices)
    a = smooth_operator.T @ smooth_operator + float(handle_weight) * (p.T @ p) + float(ridge) * sparse.eye(n_vertices, format="csr")
    solve = factorized(a.tocsc())
    pred = np.zeros((len(controls), n_vertices, 3), dtype=np.float64)
    rhs_handle = float(handle_weight) * p.T
    for i, ctrl in enumerate(controls):
        rhs = rhs_handle @ ctrl
        for d in range(3):
            pred[i, :, d] = solve(rhs[:, d])
    return pred.reshape(len(controls), -1)


def arap_predict_one(
    source: np.ndarray,
    control_delta: np.ndarray,
    landmarks: np.ndarray,
    edges: np.ndarray,
    laplacian: sparse.csr_matrix,
    handle_weight: float,
    ridge: float,
    n_iter: int,
    initial_delta: np.ndarray,
) -> np.ndarray:
    n = source.shape[0]
    pmat = selector_matrix(landmarks, n)
    a = laplacian + float(handle_weight) * (pmat.T @ pmat) + float(ridge) * sparse.eye(n, format="csr")
    solve = factorized(a.tocsc())
    target_handles = source[landmarks] + control_delta
    q = source + initial_delta.reshape(n, 3)
    neighbors = [[] for _ in range(n)]
    for i, j in edges:
        neighbors[int(i)].append(int(j))
        neighbors[int(j)].append(int(i))
    rotations = np.tile(np.eye(3)[None, :, :], (n, 1, 1))
    for _ in range(int(n_iter)):
        for i in range(n):
            cov = np.zeros((3, 3), dtype=np.float64)
            pi = source[i]
            qi = q[i]
            for j in neighbors[i]:
                cov += np.outer(qi - q[j], pi - source[j])
            u, _, vt = np.linalg.svd(cov)
            r = u @ vt
            if np.linalg.det(r) < 0.0:
                u[:, -1] *= -1.0
                r = u @ vt
            rotations[i] = r
        b = np.zeros((n, 3), dtype=np.float64)
        for i, j in edges:
            i, j = int(i), int(j)
            term = 0.5 * (rotations[i] + rotations[j]) @ (source[i] - source[j])
            b[i] += term
            b[j] -= term
        b += float(handle_weight) * (pmat.T @ target_handles)
        for d in range(3):
            q[:, d] = solve(b[:, d])
    return q - source


def arap_predict(
    by_id: dict[str, dict],
    pairs: list[tuple[str, str]],
    init_pred: np.ndarray,
    handle_weight: float,
    ridge: float,
    n_iter: int,
) -> np.ndarray:
    template = next(iter(by_id.values()))
    landmarks = template["landmarks"]
    faces = template["faces"]
    edges = graph_edges(faces)
    lap = uniform_laplacian(template["vertices"].shape[0], faces)
    pred = np.zeros_like(init_pred, dtype=np.float64)
    for k, (src_id, tgt_id) in enumerate(pairs):
        src = by_id[src_id]["vertices"]
        tgt = by_id[tgt_id]["vertices"]
        ctrl = (tgt - src)[landmarks]
        pred[k] = arap_predict_one(src, ctrl, landmarks, edges, lap, handle_weight, ridge, n_iter, init_pred[k])
    return pred.reshape(len(pairs), -1)


def flatten_vertices(by_id: dict[str, dict], ids: list[str]) -> np.ndarray:
    return np.stack([by_id[sid]["vertices"].reshape(-1) for sid in ids], axis=0)


def build_sourcepca_condition(by_id: dict[str, dict], pairs: list[tuple[str, str]], source_pca: dict[str, np.ndarray]) -> np.ndarray:
    landmarks = next(iter(by_id.values()))["landmarks"]
    ctrl = []
    src_flat = []
    for src_id, tgt_id in pairs:
        src = by_id[src_id]["vertices"]
        tgt = by_id[tgt_id]["vertices"]
        ctrl.append((tgt - src)[landmarks].reshape(-1))
        src_flat.append(src.reshape(-1))
    return np.concatenate([np.stack(ctrl), pca_coeff(np.stack(src_flat), source_pca)], axis=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="../data")
    parser.add_argument("--out", default="../results/classical")
    parser.add_argument("--max-pairs", type=int, default=0, help="0 means all held-out test pairs.")
    parser.add_argument("--source-pca-dim", type=int, default=16)
    parser.add_argument("--ridge-lambda", type=float, default=100.0)
    parser.add_argument("--handle-weight", type=float, default=1000.0)
    parser.add_argument("--system-ridge", type=float, default=1e-8)
    parser.add_argument("--include-arap", action="store_true")
    parser.add_argument("--arap-iter", type=int, default=3)
    parser.add_argument("--arap-warning-pair-threshold", type=int, default=200)
    parser.add_argument(
        "--confirm-slow-arap",
        action="store_true",
        help="Required when running the pure-Python ARAP baseline on more than --arap-warning-pair-threshold pairs.",
    )
    parser.add_argument("--seed", type=int, default=2028)
    args = parser.parse_args()

    if args.include_arap and not args.confirm_slow_arap:
        if not args.max_pairs or args.max_pairs <= 0:
            raise SystemExit(
                "Refusing to run pure-Python ARAP on the full held-out pair set without --confirm-slow-arap. "
                "Run Laplacian/bi-Laplacian first, set a small --max-pairs smoke test, or confirm the slow full ARAP run explicitly."
            )
        if args.max_pairs > args.arap_warning_pair_threshold:
            raise SystemExit(
                "Refusing to run pure-Python ARAP on "
                f"{args.max_pairs} requested pairs without --confirm-slow-arap. "
                "Lower --max-pairs, increase --arap-warning-pair-threshold for a controlled run, or confirm the slow ARAP run explicitly."
            )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(args.repo)
    _, by_id = load_rows(repo)
    train_ids = split_ids(by_id, "clean-prior train")
    test_ids = split_ids(by_id, "main test")
    train_pairs = ordered_pairs(train_ids, None, args.seed)
    test_pairs = ordered_pairs(test_ids, None, args.seed)
    if args.max_pairs and args.max_pairs > 0:
        test_pairs = test_pairs[: args.max_pairs]
    if args.include_arap and len(test_pairs) > args.arap_warning_pair_threshold and not args.confirm_slow_arap:
        raise SystemExit(
            "Refusing to run pure-Python ARAP on "
            f"{len(test_pairs)} pairs without --confirm-slow-arap. "
            "Run Laplacian/bi-Laplacian first, lower --max-pairs, or confirm the slow full ARAP run explicitly."
        )
    template = next(iter(by_id.values()))
    landmarks = template["landmarks"]
    n_vertices = template["vertices"].shape[0]
    faces = template["faces"]

    source_pca = fit_pca(flatten_vertices(by_id, train_ids), args.source_pca_dim)
    x_train = build_sourcepca_condition(by_id, train_pairs, source_pca)
    x_test = build_sourcepca_condition(by_id, test_pairs, source_pca)
    mean = x_train.mean(axis=0, keepdims=True)
    std = np.maximum(x_train.std(axis=0, keepdims=True), 1e-6)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    _, y_train = pair_arrays(by_id, train_pairs)
    ridge_model = ridge_fit(x_train, y_train, args.ridge_lambda)
    ridge_pred = ridge_predict(x_test, ridge_model)

    controls = np.stack([(by_id[t]["vertices"] - by_id[s]["vertices"])[landmarks] for s, t in test_pairs], axis=0)
    lap = uniform_laplacian(n_vertices, faces)
    started = time.perf_counter()
    laplacian_pred = solve_linear_handle_baseline(controls, landmarks, n_vertices, lap, args.handle_weight, args.system_ridge)
    bilap = lap @ lap
    bilaplacian_pred = solve_linear_handle_baseline(controls, landmarks, n_vertices, bilap, args.handle_weight, args.system_ridge)
    arap_pred = None
    if args.include_arap:
        arap_pred = arap_predict(by_id, test_pairs, laplacian_pred.reshape(len(test_pairs), n_vertices, 3), args.handle_weight, args.system_ridge, args.arap_iter)
    elapsed = time.perf_counter() - started

    results = [
        evaluate_method("ridge_sourcepca", by_id, test_pairs, ridge_pred),
        evaluate_method("laplacian_handles", by_id, test_pairs, laplacian_pred),
        evaluate_method("bilaplacian_handles", by_id, test_pairs, bilaplacian_pred),
    ]
    arrays: dict[str, np.ndarray] = {
        "test_pairs": np.asarray(test_pairs, dtype=object),
        "ridge_cond_pred": np.asarray(ridge_pred, dtype=np.float32),
        "laplacian_handles_pred": np.asarray(laplacian_pred, dtype=np.float32),
        "bilaplacian_handles_pred": np.asarray(bilaplacian_pred, dtype=np.float32),
    }
    if arap_pred is not None:
        results.append(evaluate_method(f"arap_handles_iter{args.arap_iter}", by_id, test_pairs, arap_pred))
        arrays[f"arap_handles_iter{args.arap_iter}_pred"] = np.asarray(arap_pred, dtype=np.float32)

    suffix = "all" if not args.max_pairs else f"first{args.max_pairs}"
    npz_path = out_dir / f"classical_deformation_predictions_{suffix}.npz"
    np.savez_compressed(npz_path, **arrays)
    report = {
        "prediction_file": str(npz_path),
        "elapsed_sec": elapsed,
        "args": vars(args),
        "n_test_pairs": len(test_pairs),
        "results": results,
    }
    (out_dir / f"classical_deformation_summary_{suffix}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
