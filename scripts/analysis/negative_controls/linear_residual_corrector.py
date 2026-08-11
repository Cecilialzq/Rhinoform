"""Test 5 -- linear residual corrector as a candidate method, with geometry.

Test 2 showed the conditional ridge residual carries a weak-but-measurable
*linear* learnable signal: the ``ridge_linear`` oracle was the best, recovering
a little held-out residual energy and improving ROI RMSE (~1.057 -> ~1.008 mm).
That makes the linear residual corrector the only thing that actually moves
accuracy -- so it deserves to be evaluated as a real candidate method, INCLUDING
its geometric cost (foldover risk), not just RMSE.

This script promotes the same legal, train-fit linear oracle from Test 2 into a
deployable corrector ``pred = ridge_anchor + linear_residual(cond)`` and scores
it head-to-head against the ridge anchor on held-out TEST identities with:

* ``roi_rmse``        -- L2 accuracy vs the ground-truth displacement.
* ``abs_flip_pct``    -- legacy normal-flip (faces with det<0 vs source).
* ``new_flip_pct``    -- model-induced foldover beyond the true target (the
  corrected regularity metric; the baseline-relative budget).
* ``missed_flip_pct`` -- true target folds not reproduced.

Paired identity-level bootstrap CIs of gains (ridge - corrector; positive means
the corrector is better/lower) decide the verdict:

* ``GEOMETRICALLY_FREE_GAIN`` -- corrector improves RMSE (CI clearly >0) and does
  NOT raise new-flip -> supports "a linear anchor + lightweight residual
  correction improves accuracy; only learned/high-capacity residual admission
  raises geometric risk, hence shrink/reject is needed".
* ``GAIN_NOT_FREE``           -- corrector improves RMSE but new-flip clearly
  worsens -> even the weak learnable accuracy gain is not geometrically free.
* ``NO_GAIN``                 -- no clear RMSE improvement.

No leakage: input = ``cond`` = [control-landmark deltas | source-PCA code]; the
corrector is fit on TRAIN pairs and evaluated on identity-disjoint TEST pairs;
the ridge anchor is the package's own ``ridge_cond``. Methods are scored on their
native displacement fields (as-deployed), exactly like the ridge baseline.
"""
from __future__ import annotations

import argparse
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
from rhinoform.safe_fusion import new_and_missed_folds, signed_fold_indicator


def fold_mask(src: np.ndarray, edited: np.ndarray, faces: np.ndarray) -> np.ndarray:
    det, _ = signed_fold_indicator(src, edited, faces)
    return det < 0.0


def per_pair_rmse(pred_flat: np.ndarray, true_flat: np.ndarray) -> np.ndarray:
    p = pred_flat.reshape(pred_flat.shape[0], -1, 3)
    t = true_flat.reshape(true_flat.shape[0], -1, 3)
    return np.sqrt(np.mean(np.sum((p - t) ** 2, axis=2), axis=1))


def per_pair_flips(source_abs: np.ndarray, pred_flat: np.ndarray, target_flat: np.ndarray,
                   faces: np.ndarray) -> dict[str, np.ndarray]:
    n = source_abs.shape[0]
    pred3 = pred_flat.reshape(n, -1, 3)
    tgt3 = target_flat.reshape(n, -1, 3)
    abs_pp = np.empty(n)
    new_pp = np.empty(n)
    missed_pp = np.empty(n)
    intrinsic_pp = np.empty(n)
    for i in range(n):
        src = source_abs[i]
        pred_fold = fold_mask(src, src + pred3[i], faces)
        tgt_fold = fold_mask(src, src + tgt3[i], faces)
        new_fold, missed_fold = new_and_missed_folds(pred_fold, tgt_fold)
        abs_pp[i] = 100.0 * pred_fold.mean()
        intrinsic_pp[i] = 100.0 * tgt_fold.mean()
        new_pp[i] = 100.0 * new_fold.mean()
        missed_pp[i] = 100.0 * missed_fold.mean()
    return {"abs": abs_pp, "new": new_pp, "missed": missed_pp, "intrinsic": intrinsic_pp}


def summarise(per_pair: np.ndarray, source_ids: list[str], n_boot: int, seed: int) -> dict:
    mean, lo, hi = identity_bootstrap_ci(per_pair, source_ids, n_boot, seed)
    return {"mean": mean, "ci95": [lo, hi], "p95": float(np.percentile(per_pair, 95)),
            "max": float(per_pair.max())}


def fit_residual_pca(resid_train: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    mean_r = resid_train.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(resid_train - mean_r, full_matrices=False)
    return mean_r, vt[: min(rank, vt.shape[0])]


def main() -> int:
    parser = argparse.ArgumentParser(description="Test 5: linear residual corrector candidate + geometric risk.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base-model-package", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-train-pairs", type=int, default=4000)
    parser.add_argument("--max-test-pairs", type=int, default=2000)
    parser.add_argument("--residual-rank", type=int, default=32)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260615)
    args = parser.parse_args()

    from sklearn.linear_model import Ridge

    print("[test5] loading repo + base package", flush=True)
    _, by_id = load_rows(Path(args.repo))
    base = torch.load(args.base_model_package, map_location="cpu", weights_only=False)
    if not base.get("feature_template_sha256"):
        print("[test5] WARNING: legacy (pre-leakage-fix) base package; fine for smoke, use a clean one for reported numbers.", flush=True)
    train_ids = [str(v) for v in base["train_ids"]]
    faces = np.asarray(by_id[train_ids[0]]["faces"], dtype=np.int64)
    if base.get("test_pairs") is not None and len(base["test_pairs"]) > 0:
        test_pairs = [(str(a), str(b)) for a, b in base["test_pairs"]]
        if args.max_test_pairs and len(test_pairs) > args.max_test_pairs:
            rng = np.random.default_rng(args.seed)
            sel = np.sort(rng.choice(len(test_pairs), size=args.max_test_pairs, replace=False))
            test_pairs = [test_pairs[i] for i in sel]
    else:
        test_pairs = deterministic_pairs([str(v) for v in base["test_ids"]], args.max_test_pairs, args.seed)
    train_pairs = deterministic_pairs(train_ids, args.max_train_pairs, args.seed)
    source_ids = [a for a, _ in test_pairs]
    print(f"[test5] train_pairs={len(train_pairs)} test_pairs={len(test_pairs)}", flush=True)

    cond_tr, _, delta_tr, _, _ = build_conditions(by_id, train_pairs, base)
    cond_te, source_te, delta_te, _, _ = build_conditions(by_id, test_pairs, base)
    delta_tr_flat = delta_tr.reshape(len(train_pairs), -1).astype(np.float64)
    delta_te_flat = delta_te.reshape(len(test_pairs), -1).astype(np.float64)
    source_te = np.asarray(source_te, dtype=np.float64)

    ridge_tr, resid_tr = ridge_residual(cond_tr, delta_tr_flat, base)
    ridge_te, _ = ridge_residual(cond_te, delta_te_flat, base)

    # Fit the same legal linear residual oracle as Test 2's best (ridge_linear),
    # train-only, then deploy as a corrector on held-out test pairs.
    print(f"[test5] fitting linear residual corrector (residual PCA rank={args.residual_rank})", flush=True)
    mean_r, basis = fit_residual_pca(resid_tr, args.residual_rank)
    scores_tr = (resid_tr - mean_r) @ basis.T
    reg = Ridge(alpha=args.ridge_alpha)
    reg.fit(cond_tr, scores_tr)
    scores_te = np.asarray(reg.predict(cond_te), dtype=np.float64)
    if scores_te.ndim == 1:
        scores_te = scores_te.reshape(-1, 1)
    resid_pred_te = mean_r + scores_te @ basis
    corrected_te = ridge_te + resid_pred_te

    methods = {"ridge": ridge_te, "ridge_plus_linear_residual": corrected_te}
    results: dict[str, dict] = {}
    flips_cache: dict[str, dict] = {}
    for name, pred in methods.items():
        print(f"[test5] scoring {name}", flush=True)
        rmse_pp = per_pair_rmse(pred, delta_te_flat)
        flips = per_pair_flips(source_te, pred, delta_te_flat, faces)
        flips_cache[name] = flips
        results[name] = {
            "roi_rmse": summarise(rmse_pp, source_ids, args.n_boot, args.seed),
            "abs_flip_pct": summarise(flips["abs"], source_ids, args.n_boot, args.seed),
            "new_flip_pct": summarise(flips["new"], source_ids, args.n_boot, args.seed),
            "missed_flip_pct": summarise(flips["missed"], source_ids, args.n_boot, args.seed),
        }

    # Paired (ridge - corrector) gains; >0 => corrector better/lower.
    ridge_rmse = per_pair_rmse(methods["ridge"], delta_te_flat)
    corr_rmse = per_pair_rmse(methods["ridge_plus_linear_residual"], delta_te_flat)
    rmse_gain = ridge_rmse - corr_rmse  # >0 => corrector more accurate
    new_gain = flips_cache["ridge"]["new"] - flips_cache["ridge_plus_linear_residual"]["new"]  # >0 => fewer new flips
    abs_gain = flips_cache["ridge"]["abs"] - flips_cache["ridge_plus_linear_residual"]["abs"]
    paired = {
        "rmse_gain_ridge_minus_corrector": summarise(rmse_gain, source_ids, args.n_boot, args.seed),
        "new_flip_gain_ridge_minus_corrector": summarise(new_gain, source_ids, args.n_boot, args.seed),
        "abs_flip_gain_ridge_minus_corrector": summarise(abs_gain, source_ids, args.n_boot, args.seed),
    }

    rmse_ci = paired["rmse_gain_ridge_minus_corrector"]["ci95"]
    new_ci = paired["new_flip_gain_ridge_minus_corrector"]["ci95"]
    rmse_better = rmse_ci[0] > 0.0                 # accuracy gain CI clearly > 0
    newflip_worse = new_ci[1] < 0.0                # new-flip gain CI clearly < 0 (corrector adds folds)
    if not rmse_better:
        verdict = ("NO_GAIN: the linear residual corrector does not clearly improve ROI RMSE over ridge "
                   "(paired CI of the accuracy gain includes 0). The ridge anchor is already the operating point.")
    elif newflip_worse:
        verdict = ("GAIN_NOT_FREE: the corrector improves ROI RMSE but clearly raises model-induced new-flip "
                   "-> even the weak learnable accuracy gain is NOT geometrically free; residual admission must "
                   "be shrunk/rejected. Strengthens the 'linear prior + risk-managed residual' thesis.")
    else:
        verdict = ("GEOMETRICALLY_FREE_GAIN: the linear residual corrector improves ROI RMSE without clearly "
                   "raising new-flip -> a linear anchor + lightweight LINEAR residual correction is safe; the "
                   "geometric risk arises from learned/high-capacity residual admission (cf. Test 3), motivating shrink/reject.")

    out = {
        "test": "linear_residual_corrector",
        "purpose": ("Promote Test 2's best (linear) residual oracle to a candidate method and measure whether its "
                    "accuracy gain over the ridge anchor is geometrically free (new-flip)."),
        "provenance": provenance(args.seed, {
            "repo": str(args.repo),
            "base_model_package": str(args.base_model_package),
            "base_model_package_sha256": sha256_file(Path(args.base_model_package)),
            "feature_template_sha256": base.get("feature_template_sha256"),
            "residual_rank": args.residual_rank, "ridge_alpha": args.ridge_alpha,
            "max_train_pairs": args.max_train_pairs, "max_test_pairs": args.max_test_pairs,
        }),
        "n_train_pairs": len(train_pairs),
        "n_test_pairs": len(test_pairs),
        "legal_input_note": "corrector input = cond = [control-landmark deltas | source-PCA code]; fit train-only; native displacement fields scored.",
        "metric_note": ("abs_flip_pct = legacy normal-flip; new_flip_pct = model-induced foldover beyond the true "
                        "target (baseline-relative); missed_flip_pct = unreproduced target folds. CIs identity-level bootstrap."),
        "intrinsic_target_flip_pct": {
            "mean": float(flips_cache["ridge"]["intrinsic"].mean()),
            "p95": float(np.percentile(flips_cache["ridge"]["intrinsic"], 95)),
            "max": float(flips_cache["ridge"]["intrinsic"].max()),
        },
        "methods": results,
        "paired_gain_vs_ridge": paired,
        "verdict": verdict,
    }
    out_path = Path(args.out)
    atomic_write_json(out_path, out)
    print(f"[test5] wrote {out_path}", flush=True)
    print(f"[test5] ridge RMSE={results['ridge']['roi_rmse']['mean']:.4f} -> corrector "
          f"{results['ridge_plus_linear_residual']['roi_rmse']['mean']:.4f}; "
          f"new_flip {results['ridge']['new_flip_pct']['mean']:.3f}% -> "
          f"{results['ridge_plus_linear_residual']['new_flip_pct']['mean']:.3f}%", flush=True)
    print(f"[test5] VERDICT: {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
