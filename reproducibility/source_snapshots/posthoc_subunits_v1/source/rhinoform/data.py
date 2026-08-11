from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import svds

from .sampling import coverage_balanced_ordered_pairs


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    return hashlib.sha256(array.tobytes()).hexdigest()


def training_mean_geometry(by_id: dict[str, dict], train_ids: list[str]) -> np.ndarray:
    ids = [str(value) for value in train_ids]
    if not ids:
        raise ValueError("train_ids must not be empty")
    if len(ids) != len(set(ids)):
        raise ValueError("train_ids contains duplicates")
    missing = sorted(set(ids) - set(by_id))
    if missing:
        raise KeyError(f"Missing training identities: {missing}")
    first = np.asarray(by_id[ids[0]]["vertices"], dtype=np.float64)
    total = np.zeros_like(first, dtype=np.float64)
    for subject_id in ids:
        vertices = np.asarray(by_id[subject_id]["vertices"], dtype=np.float64)
        if vertices.shape != first.shape:
            raise ValueError(f"Training geometry shape mismatch for identity {subject_id}: {vertices.shape}")
        if not np.isfinite(vertices).all():
            raise ValueError(f"Training geometry contains non-finite values for identity {subject_id}")
        total += vertices
    return total / float(len(ids))


def assert_identity_disjoint(groups: dict[str, list[str]]) -> None:
    normalized = {name: set(map(str, values)) for name, values in groups.items()}
    names = list(normalized)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sorted(normalized[left] & normalized[right])
            if overlap:
                raise ValueError(f"Identity leakage between {left} and {right}: {overlap}")


def assert_pairs_within(pairs: list[tuple[str, str]], allowed_ids: list[str], label: str) -> None:
    allowed = set(map(str, allowed_ids))
    observed = {str(value) for pair in pairs for value in pair}
    outside = sorted(observed - allowed)
    if outside:
        raise ValueError(f"{label} pairs contain identities outside the declared split: {outside}")


def load_rows(
    repo: Path,
    allowed_ids: set[str] | list[str] | tuple[str, ...] | None = None,
) -> tuple[list[dict], dict[str, dict]]:
    manifest_path = repo / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = list(manifest["rows"])
    allowed = None if allowed_ids is None else {str(value) for value in allowed_ids}
    by_id: dict[str, dict] = {}
    for row in rows:
        subject_id = str(row["subject_id"])
        if allowed is not None and subject_id not in allowed:
            continue
        npz_path = Path(row["npz_path"])
        if not npz_path.is_absolute():
            npz_path = repo / npz_path
        data = np.load(npz_path, allow_pickle=False)
        by_id[subject_id] = {
            "subject_id": subject_id,
            "split": str(row["split"]),
            "vertices": np.asarray(data["vertices"], dtype=np.float64),
            "faces": np.asarray(data["faces"], dtype=np.int64),
            "landmarks": np.asarray(data["local_landmarks"], dtype=np.int64),
            "subunits": {
                "root": np.asarray(data["subunit_root"], dtype=np.int64),
                "dorsum": np.asarray(data["subunit_dorsum"], dtype=np.int64),
                "tip": np.asarray(data["subunit_tip"], dtype=np.int64),
                "alar_left": np.asarray(data["subunit_alar_left"], dtype=np.int64),
                "alar_right": np.asarray(data["subunit_alar_right"], dtype=np.int64),
            },
        }
    if allowed is not None:
        missing = sorted(allowed - set(by_id))
        if missing:
            raise KeyError(f"Requested identities are absent from the data manifest: {missing}")
        rows = [row for row in rows if str(row["subject_id"]) in allowed]
    return rows, by_id


def split_ids(by_id: dict[str, dict], split: str) -> list[str]:
    def key(s: str) -> int:
        try:
            return int(s)
        except ValueError:
            return 10**9

    return sorted([sid for sid, row in by_id.items() if row["split"] == split], key=key)


def ordered_pairs(ids: list[str], max_pairs: int | None, seed: int) -> list[tuple[str, str]]:
    pairs = [(a, b) for a in ids for b in ids if a != b]
    if max_pairs is not None and len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[int(i)] for i in idx]
    return pairs


def fixed_ordered_pairs(ids: list[str], budget: int | None, seed: int, coverage_balanced: bool) -> list[tuple[str, str]]:
    if budget is None or int(budget) <= 0:
        return ordered_pairs(ids, None, seed)
    if coverage_balanced:
        return coverage_balanced_ordered_pairs(ids, int(budget), seed)
    return ordered_pairs(ids, int(budget), seed)


def pair_arrays(by_id: dict[str, dict], pairs: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    landmarks = next(iter(by_id.values()))["landmarks"]
    n_vertices = next(iter(by_id.values()))["vertices"].shape[0]
    x = np.zeros((len(pairs), landmarks.size * 3), dtype=np.float64)
    y = np.zeros((len(pairs), n_vertices * 3), dtype=np.float64)
    for i, (src_id, tgt_id) in enumerate(pairs):
        src = by_id[src_id]["vertices"]
        tgt = by_id[tgt_id]["vertices"]
        delta = tgt - src
        x[i] = delta[landmarks].reshape(-1)
        y[i] = delta.reshape(-1)
    return x, y


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    xc = x - x_mean
    yc = y - y_mean
    xtx = xc.T @ xc
    reg = float(lam) * np.eye(xtx.shape[0], dtype=np.float64)
    w = np.linalg.solve(xtx + reg, xc.T @ yc)
    intercept = y_mean - x_mean @ w
    return w, intercept.reshape(-1)


def ridge_predict(x: np.ndarray, model: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    w, b = model
    return x @ w + b


def fit_pca(y: np.ndarray, max_components: int) -> dict[str, np.ndarray]:
    mean = y.mean(axis=0)
    yc = y - mean
    k = min(int(max_components), yc.shape[0] - 1, yc.shape[1] - 1)
    if k <= 0:
        raise ValueError("Not enough rows for PCA")
    _, singular, vt = svds(yc, k=k)
    order = np.argsort(singular)[::-1]
    return {"mean": mean, "components": vt[order], "singular": singular[order]}


def pca_coeff(y: np.ndarray, pca: dict[str, np.ndarray]) -> np.ndarray:
    return (y - pca["mean"]) @ pca["components"].T


def pca_recon(coeff: np.ndarray, pca: dict[str, np.ndarray]) -> np.ndarray:
    return pca["mean"] + coeff @ pca["components"]


def rbf_predict(source: np.ndarray, landmarks: np.ndarray, control_delta: np.ndarray) -> np.ndarray:
    ctrl_xyz = source[landmarks]
    diff_cc = ctrl_xyz[:, None, :] - ctrl_xyz[None, :, :]
    d_cc = np.linalg.norm(diff_cc, axis=-1)
    nonzero = d_cc[d_cc > 1e-12]
    sigma = float(np.median(nonzero)) if nonzero.size else float(np.linalg.norm(source.ptp(axis=0)) * 0.15)
    sigma = max(sigma, 1e-6)
    k_cc = np.exp(-(d_cc**2) / (2.0 * sigma * sigma)) + 1e-6 * np.eye(len(landmarks))
    alpha = np.linalg.solve(k_cc, control_delta)
    d_all = np.linalg.norm(source[:, None, :] - ctrl_xyz[None, :, :], axis=-1)
    k_all = np.exp(-(d_all**2) / (2.0 * sigma * sigma))
    return k_all @ alpha


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    denom = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(denom, 1e-12)


def edge_index(faces: np.ndarray) -> np.ndarray:
    edges = set()
    for a, b, c in faces:
        for i, j in ((a, b), (b, c), (c, a)):
            i, j = int(i), int(j)
            if i > j:
                i, j = j, i
            edges.add((i, j))
    return np.asarray(sorted(edges), dtype=np.int64)


def rmse_vertices(pred_delta: np.ndarray, true_delta: np.ndarray, idx: np.ndarray | None = None) -> float:
    pd = pred_delta.reshape(-1, 3)
    td = true_delta.reshape(-1, 3)
    if idx is not None:
        pd = pd[idx]
        td = td[idx]
    return float(np.sqrt(np.mean(np.sum((pd - td) ** 2, axis=1))))


def quality_metrics(source: np.ndarray, pred: np.ndarray, faces: np.ndarray, edges: np.ndarray) -> tuple[float, float]:
    e0 = np.linalg.norm(source[edges[:, 0]] - source[edges[:, 1]], axis=1)
    e1 = np.linalg.norm(pred[edges[:, 0]] - pred[edges[:, 1]], axis=1)
    strain = np.abs(e1 - e0) / np.maximum(e0, 1e-12)
    n0 = face_normals(source, faces)
    n1 = face_normals(pred, faces)
    flip = float(np.mean(np.sum(n0 * n1, axis=1) < 0.0) * 100.0)
    return float(np.percentile(strain, 95)), flip


def evaluate_method(
    name: str,
    by_id: dict[str, dict],
    pairs: list[tuple[str, str]],
    pred_flat: np.ndarray | None,
    use_rbf: bool = False,
) -> dict[str, float | str]:
    template = next(iter(by_id.values()))
    landmarks = template["landmarks"]
    faces = template["faces"]
    edges = edge_index(faces)
    subunits = template["subunits"]
    rows = []
    t0 = time.perf_counter()
    for i, (src_id, tgt_id) in enumerate(pairs):
        src = by_id[src_id]["vertices"]
        tgt = by_id[tgt_id]["vertices"]
        true_delta = (tgt - src).reshape(-1)
        if use_rbf:
            ctrl = (tgt - src)[landmarks]
            pred_delta = rbf_predict(src, landmarks, ctrl).reshape(-1)
        elif pred_flat is None:
            pred_delta = np.zeros_like(true_delta)
        else:
            pred_delta = pred_flat[i]
        pred_vertices = src + pred_delta.reshape(-1, 3)
        edge_p95, flip = quality_metrics(src, pred_vertices, faces, edges)
        one = {
            "roi_rmse": rmse_vertices(pred_delta, true_delta),
            "landmark_rmse": rmse_vertices(pred_delta, true_delta, landmarks),
            "root_rmse": rmse_vertices(pred_delta, true_delta, subunits["root"]),
            "dorsum_rmse": rmse_vertices(pred_delta, true_delta, subunits["dorsum"]),
            "tip_rmse": rmse_vertices(pred_delta, true_delta, subunits["tip"]),
            "alar_left_rmse": rmse_vertices(pred_delta, true_delta, subunits["alar_left"]),
            "alar_right_rmse": rmse_vertices(pred_delta, true_delta, subunits["alar_right"]),
            "edge_strain_p95": edge_p95,
            "normal_flip_pct": flip,
        }
        rows.append(one)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(pairs))
    out: dict[str, float | str] = {"method": name, "n_pairs": float(len(pairs)), "latency_ms_per_pair": float(elapsed_ms)}
    for key in rows[0]:
        vals = np.asarray([r[key] for r in rows], dtype=np.float64)
        out[key] = float(vals.mean())
    return out


def choose_ridge_lambda(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, lambdas: list[float]) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
    best_lam = lambdas[0]
    best_model = ridge_fit(x_train, y_train, best_lam)
    best = np.inf
    for lam in lambdas:
        model = ridge_fit(x_train, y_train, lam)
        pred = ridge_predict(x_val, model)
        score = float(np.sqrt(np.mean((pred - y_val) ** 2)))
        if score < best:
            best = score
            best_lam = lam
            best_model = model
    return best_lam, best_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="../data")
    parser.add_argument("--out", default="../results")
    parser.add_argument("--max-test-pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    repo = Path(args.repo)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, by_id = load_rows(repo)
    train_ids = split_ids(by_id, "clean-prior train")
    val_ids = split_ids(by_id, "clean-prior validation")
    test_ids = split_ids(by_id, "main test")
    train_pairs = ordered_pairs(train_ids, None, args.seed)
    val_pairs = ordered_pairs(val_ids, None, args.seed)
    test_pairs = ordered_pairs(test_ids, args.max_test_pairs, args.seed)
    print(
        f"loaded ids train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}; "
        f"pairs train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}",
        flush=True,
    )

    x_train, y_train = pair_arrays(by_id, train_pairs)
    x_val, y_val = pair_arrays(by_id, val_pairs)
    x_test, _ = pair_arrays(by_id, test_pairs)
    lambdas = [1e-6, 1e-4, 1e-2, 1.0, 100.0, 10000.0]

    print("fitting dense ridge control baseline", flush=True)
    ridge_lam, ridge_model = choose_ridge_lambda(x_train, y_train, x_val, y_val, lambdas)
    ridge_pred = ridge_predict(x_test, ridge_model)

    print("fitting PCA32 conditional baseline", flush=True)
    pca = fit_pca(y_train, max_components=32)
    coeff_train = pca_coeff(y_train, pca)
    coeff_val = pca_coeff(y_val, pca)
    pca_lam, pca_model = choose_ridge_lambda(x_train, coeff_train, x_val, coeff_val, lambdas)
    pca_pred = pca_recon(ridge_predict(x_test, pca_model), pca)

    print("evaluating methods", flush=True)
    results = [
        evaluate_method("source_copy", by_id, test_pairs, None),
        evaluate_method("rbf_controls", by_id, test_pairs, None, use_rbf=True),
        evaluate_method(f"ridge_dense_lambda_{ridge_lam:g}", by_id, test_pairs, ridge_pred),
        evaluate_method(f"pca32_control_lambda_{pca_lam:g}", by_id, test_pairs, pca_pred),
    ]

    summary = {
        "splits": {"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids},
        "n_pairs": {"train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs)},
        "results": results,
        "interpretation": {},
    }
    by_method = {r["method"]: r for r in results}
    source = by_method["source_copy"]["roi_rmse"]
    rbf = by_method["rbf_controls"]["roi_rmse"]
    ridge = by_method[f"ridge_dense_lambda_{ridge_lam:g}"]["roi_rmse"]
    pca_rmse = by_method[f"pca32_control_lambda_{pca_lam:g}"]["roi_rmse"]
    summary["interpretation"] = {
        "ridge_improvement_vs_source_pct": float((source - ridge) / source * 100.0),
        "ridge_improvement_vs_rbf_pct": float((rbf - ridge) / rbf * 100.0),
        "pca_improvement_vs_source_pct": float((source - pca_rmse) / source * 100.0),
        "pca_improvement_vs_rbf_pct": float((rbf - pca_rmse) / rbf * 100.0),
    }

    with (out_dir / "viability_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    (out_dir / "viability_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
