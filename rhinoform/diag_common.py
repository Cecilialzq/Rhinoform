"""Shared helpers for the negative-result diagnostic battery (Tests 1-4).

These four diagnostics defend the *credibility* of the negative result
("sparse-control conditioning removes the transferable nonlinear advantage")
rather than chase a positive result. They are deliberately model-free where
possible and reuse the frozen pipeline's exact data path so that nothing leaks:

* identity splits come from the data manifest (``train pool`` / ``main test`` /
  ``clean-prior validation``) and are identity-disjoint by construction;
* the conditional input ``cond`` is exactly ``[control-landmark deltas |
  source-PCA code]`` normalised by the train-fit ``cond_mean`` / ``cond_std`` --
  it never contains the target dense shape;
* the ridge anchor is the package's own ``ridge_cond`` (the same anchor the
  hybrid / RBSR gate build on), so the residual is the real conditional residual.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from .data import load_rows, ordered_pairs, ridge_predict, rmse_vertices, split_ids
from .train import pair_conditions

TRAIN_SPLIT = "train pool"
VAL_SPLIT = "clean-prior validation"
TEST_SPLIT = "main test"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON via a same-directory temporary file, then atomically replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def provenance(seed: int, extra: dict | None = None) -> dict:
    import scipy

    out = {
        "seed": seed,
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import torch

        out["torch"] = torch.__version__
    except Exception:  # pragma: no cover - torch always present in practice
        out["torch"] = None
    try:
        import sklearn

        out["sklearn"] = sklearn.__version__
    except Exception:  # pragma: no cover
        out["sklearn"] = None
    if extra:
        out.update(extra)
    return out


def manifest_splits(by_id: dict[str, dict]) -> tuple[list[str], list[str], list[str]]:
    """Identity-disjoint splits straight from the data manifest."""
    train_ids = split_ids(by_id, TRAIN_SPLIT)
    val_ids = split_ids(by_id, VAL_SPLIT)
    test_ids = split_ids(by_id, TEST_SPLIT)
    if not train_ids or not test_ids:
        raise RuntimeError(
            f"Empty split: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}. "
            "Check manifest split names."
        )
    overlap = (set(train_ids) & set(test_ids)) | (set(train_ids) & set(val_ids)) | (set(val_ids) & set(test_ids))
    if overlap:
        raise RuntimeError(f"Identity leakage across splits: {sorted(overlap)}")
    return train_ids, val_ids, test_ids


def identity_matrix(by_id: dict[str, dict], ids: list[str]) -> np.ndarray:
    """Stack centred-able per-identity ROI shape vectors, shape (n_ids, V*3)."""
    rows = [np.asarray(by_id[i]["vertices"], dtype=np.float64).reshape(-1) for i in ids]
    return np.stack(rows, axis=0)


def recon_rmse_per_identity(pred_flat: np.ndarray, true_flat: np.ndarray) -> np.ndarray:
    """Per-identity reconstruction RMSE (sqrt mean over vertices of squared L2)."""
    n = pred_flat.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = rmse_vertices(pred_flat[i], true_flat[i])
    return out


def build_conditions(by_id, pairs, base):
    """Legal conditional inputs for a set of pairs, using the package's train-fit stats."""
    cond, source, delta, ctrl, source_code, _, _ = pair_conditions(
        by_id, pairs, base["source_pca"], base["cond_mean"], base["cond_std"]
    )
    return cond, source, delta, ctrl, source_code


def ridge_residual(cond: np.ndarray, delta_flat: np.ndarray, base) -> tuple[np.ndarray, np.ndarray]:
    """Ridge anchor prediction and conditional residual (target - ridge), both flat (N, V*3)."""
    ridge_flat = ridge_predict(cond, base["ridge_cond"]).astype(np.float64)
    residual = delta_flat.astype(np.float64) - ridge_flat
    return ridge_flat, residual


def deterministic_pairs(ids: list[str], max_pairs: int, seed: int) -> list[tuple[str, str]]:
    return ordered_pairs([str(i) for i in ids], max_pairs if max_pairs and max_pairs > 0 else None, seed)


def identity_bootstrap_ci(
    per_pair: np.ndarray, source_ids: list[str], n_boot: int, seed: int, alpha: float = 0.05
) -> tuple[float, float, float]:
    per_pair = np.asarray(per_pair, dtype=np.float64)
    if len(per_pair) == 0:
        return float("nan"), float("nan"), float("nan")
    unique = sorted(set(source_ids))
    index_by_id: dict[str, list[int]] = {uid: [] for uid in unique}
    for i, sid in enumerate(source_ids):
        index_by_id[sid].append(i)
    rng = np.random.default_rng(seed)
    n = len(unique)
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        chosen = rng.integers(0, n, size=n)
        idx: list[int] = []
        for c in chosen:
            idx.extend(index_by_id[unique[c]])
        boot[b] = per_pair[idx].mean()
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return float(per_pair.mean()), lo, hi


def effective_rank(singular_values: np.ndarray) -> float:
    """Participation ratio (Σλ)²/Σλ² over eigenvalues λ = s²; a flat/high-freq
    spectrum gives a large effective rank, a low-rank spectrum a small one."""
    s = np.asarray(singular_values, dtype=np.float64)
    lam = s * s
    total = lam.sum()
    if total <= 0:
        return 0.0
    return float((total * total) / float((lam * lam).sum()))


def energy_fraction_for(singular_values: np.ndarray, fraction: float) -> int:
    """Smallest k such that the top-k components capture >= `fraction` of energy."""
    lam = np.asarray(singular_values, dtype=np.float64) ** 2
    total = lam.sum()
    if total <= 0:
        return 0
    cum = np.cumsum(lam) / total
    return int(np.searchsorted(cum, fraction) + 1)
