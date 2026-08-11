"""Test 2 -- residual learnability oracle (the strongest reviewer defense).

Question answered: *is the conditional residual that ridge leaves behind even
learnable from legal inputs, by ANY flexible model?*

We freeze the ridge anchor (the package's own ``ridge_cond``), take the residual
``target - ridge_prediction``, and try to predict it from LEGAL inputs only --
the conditioning vector ``cond`` = ``[control-landmark deltas | source-PCA
code]``. We never use the target dense shape. We hand the residual to several
flexible regressors (linear ridge as a floor, kNN, random forest, histogram
gradient boosting, MLP) and measure how much held-out residual energy they
recover on identity-disjoint test pairs.

Reading:

* If even the best oracle cannot reduce held-out residual energy (R^2 ~ 0 and
  no ROI-RMSE improvement): strong evidence the conditional residual is weakly
  generalisable -- this is a property of the problem, not a single-model
  failure (CVAE/RBSR). The negative result is credible.
* If an oracle clearly recovers residual energy: we must NOT write
  "unlearnable"; instead "our current residual-admission training failed to
  extract this signal", forcing a more conservative paper and triggering Test 3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rhinoform.diag_common import (
    atomic_write_json,
    build_conditions,
    deterministic_pairs,
    identity_bootstrap_ci,
    provenance,
    ridge_residual,
    sha256_file,
)
from rhinoform.data import load_rows


def fit_residual_pca(resid_train: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_r = resid_train.mean(axis=0, keepdims=True)
    centered = resid_train - mean_r
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(rank, vt.shape[0])
    return mean_r, vt[:k], singular[:k]


def r2_energy(resid_true: np.ndarray, resid_pred: np.ndarray) -> float:
    """Fraction of held-out residual energy recovered by the prediction."""
    num = float(np.sum((resid_true - resid_pred) ** 2))
    den = float(np.sum(resid_true**2))
    if den <= 0:
        return float("nan")
    return 1.0 - num / den


def per_pair_rmse(pred_flat: np.ndarray, true_flat: np.ndarray) -> np.ndarray:
    p = pred_flat.reshape(pred_flat.shape[0], -1, 3)
    t = true_flat.reshape(true_flat.shape[0], -1, 3)
    return np.sqrt(np.mean(np.sum((p - t) ** 2, axis=2), axis=1))


def build_regressors(names: list[str], rank: int, seed: int) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor

    catalogue = {
        "ridge_linear": lambda: Ridge(alpha=1.0),
        "knn": lambda: KNeighborsRegressor(n_neighbors=15, weights="distance"),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=seed
        ),
        "hist_gbm": lambda: MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed)
        ),
        "mlp": lambda: MLPRegressor(
            hidden_layer_sizes=(256, 256), activation="relu", max_iter=600,
            early_stopping=True, random_state=seed,
        ),
    }
    return {name: catalogue[name]() for name in names if name in catalogue}


def main() -> int:
    parser = argparse.ArgumentParser(description="Test 2: residual learnability oracle on the conditional ridge residual.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base-model-package", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-train-pairs", type=int, default=4000)
    parser.add_argument("--max-test-pairs", type=int, default=2000)
    parser.add_argument("--residual-rank", type=int, default=32)
    parser.add_argument(
        "--regressors", nargs="+",
        default=["ridge_linear", "knn", "random_forest", "hist_gbm", "mlp"],
    )
    parser.add_argument("--r2-threshold", type=float, default=0.05,
                        help="held-out residual R^2 below this counts as 'not learnable'.")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260615)
    args = parser.parse_args()

    print(f"[test2] loading repo + base package", flush=True)
    _, by_id = load_rows(Path(args.repo))
    base = torch.load(args.base_model_package, map_location="cpu", weights_only=False)
    if not base.get("feature_template_sha256"):
        print("[test2] WARNING: base package predates the leakage fix (no feature_template_sha256). "
              "Fine for a smoke test; use a clean package for the reported number.", flush=True)

    train_ids = [str(v) for v in base["train_ids"]]
    if base.get("test_pairs") is not None and len(base["test_pairs"]) > 0:
        test_pairs = [(str(a), str(b)) for a, b in base["test_pairs"]]
    else:
        test_ids = [str(v) for v in base["test_ids"]]
        test_pairs = deterministic_pairs(test_ids, args.max_test_pairs, args.seed)
    if args.max_test_pairs and len(test_pairs) > args.max_test_pairs:
        rng = np.random.default_rng(args.seed)
        sel = np.sort(rng.choice(len(test_pairs), size=args.max_test_pairs, replace=False))
        test_pairs = [test_pairs[i] for i in sel]
    train_pairs = deterministic_pairs(train_ids, args.max_train_pairs, args.seed)
    print(f"[test2] train_pairs={len(train_pairs)}  test_pairs={len(test_pairs)}", flush=True)

    cond_tr, _, delta_tr, _, _ = build_conditions(by_id, train_pairs, base)
    cond_te, _, delta_te, _, _ = build_conditions(by_id, test_pairs, base)
    delta_tr_flat = delta_tr.reshape(len(train_pairs), -1).astype(np.float64)
    delta_te_flat = delta_te.reshape(len(test_pairs), -1).astype(np.float64)

    ridge_tr, resid_tr = ridge_residual(cond_tr, delta_tr_flat, base)
    ridge_te, resid_te = ridge_residual(cond_te, delta_te_flat, base)

    # Reference energy: how much of the target the ridge anchor already explains.
    target_energy = float(np.sum(delta_te_flat**2))
    residual_energy = float(np.sum(resid_te**2))
    ridge_rmse_pp = per_pair_rmse(ridge_te, delta_te_flat)
    source_ids = [a for a, _ in test_pairs]

    print(f"[test2] fitting residual PCA (rank={args.residual_rank})", flush=True)
    mean_r, basis, singular = fit_residual_pca(resid_tr, args.residual_rank)
    scores_tr = (resid_tr - mean_r) @ basis.T
    # PCA ceiling on held-out: best this basis could do if scores were perfect.
    recon_ceiling = mean_r + ((resid_te - mean_r) @ basis.T) @ basis
    ceiling_r2 = r2_energy(resid_te, recon_ceiling)
    print(f"[test2] residual-PCA ceiling R^2 on held-out = {ceiling_r2:.4f}", flush=True)

    regressors = build_regressors(args.regressors, args.residual_rank, args.seed)
    per_model: dict[str, dict] = {}
    best_r2 = -np.inf
    best_name = None
    for name, model in regressors.items():
        print(f"[test2] fitting oracle regressor: {name}", flush=True)
        model.fit(cond_tr, scores_tr)
        scores_hat = np.asarray(model.predict(cond_te), dtype=np.float64)
        if scores_hat.ndim == 1:
            scores_hat = scores_hat.reshape(-1, 1)
        resid_pred = mean_r + scores_hat @ basis
        r2 = r2_energy(resid_te, resid_pred)
        oracle_total = ridge_te + resid_pred
        oracle_rmse_pp = per_pair_rmse(oracle_total, delta_te_flat)
        improvement_pp = ridge_rmse_pp - oracle_rmse_pp  # >0 => oracle improved over ridge
        imp_mean, imp_lo, imp_hi = identity_bootstrap_ci(improvement_pp, source_ids, args.n_boot, args.seed)
        per_model[name] = {
            "heldout_residual_r2": r2,
            "roi_rmse_ridge": float(ridge_rmse_pp.mean()),
            "roi_rmse_oracle": float(oracle_rmse_pp.mean()),
            "roi_rmse_improvement": {"mean": imp_mean, "ci95": [imp_lo, imp_hi]},
            "roi_rmse_improvement_pct": float(imp_mean / ridge_rmse_pp.mean() * 100.0),
        }
        print(f"[test2]   {name}: residual_R^2={r2:.4f}  ROI_RMSE {ridge_rmse_pp.mean():.4f}->{oracle_rmse_pp.mean():.4f} "
              f"(Δ={imp_mean:.4f}, CI[{imp_lo:.4f},{imp_hi:.4f}])", flush=True)
        if r2 > best_r2:
            best_r2 = r2
            best_name = name

    learnable = bool(best_r2 > args.r2_threshold)
    out = {
        "test": "test2_residual_learnability_oracle",
        "purpose": (
            "Can ANY flexible model predict the conditional ridge residual from legal inputs "
            "(controls + source PCA, never the target dense shape) on held-out identities?"
        ),
        "provenance": provenance(args.seed, {
            "repo": str(args.repo),
            "base_model_package": str(args.base_model_package),
            "base_model_package_sha256": sha256_file(Path(args.base_model_package)),
            "feature_template_sha256": base.get("feature_template_sha256"),
            "residual_rank": args.residual_rank,
            "max_train_pairs": args.max_train_pairs,
            "max_test_pairs": args.max_test_pairs,
        }),
        "n_train_pairs": len(train_pairs),
        "n_test_pairs": len(test_pairs),
        "legal_input_note": "oracle input = cond = [control-landmark deltas | source-PCA code]; target dense shape NEVER used.",
        "energy": {
            "ridge_explained_fraction_of_target": float(1.0 - residual_energy / target_energy),
            "residual_fraction_of_target": float(residual_energy / target_energy),
            "residual_pca_ceiling_r2_heldout": ceiling_r2,
        },
        "oracles": per_model,
        "best_oracle": {"name": best_name, "heldout_residual_r2": float(best_r2)},
        "verdict": (
            "RESIDUAL_PARTIALLY_LEARNABLE: an oracle recovers held-out residual energy "
            f"(best R^2={best_r2:.3f} > {args.r2_threshold}). Do NOT claim 'unlearnable'; write "
            "'current residual-admission training under-extracts this signal' and run Test 3."
            if learnable else
            "RESIDUAL_NOT_LEARNABLE: no oracle recovers held-out residual energy "
            f"(best R^2={best_r2:.3f} <= {args.r2_threshold}). The conditional residual is weakly "
            "generalisable -- a property of the problem, not a single-model failure. Negative result is credible."
        ),
    }
    out_path = Path(args.out)
    atomic_write_json(out_path, out)
    print(f"[test2] wrote {out_path}", flush=True)
    print(f"[test2] VERDICT: {out['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
