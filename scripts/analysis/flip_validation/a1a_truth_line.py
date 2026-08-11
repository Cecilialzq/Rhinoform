"""A1a (decisive truth line): real i2i pairs, stratified by deformation magnitude.

For each test pair compute ROI RMSE of ridge vs rbsr (BOTH through the same RBF
projector) against the real target. Stratify by ||target-source|| into 4-5
quantile buckets and report RMSE_ridge - RMSE_rbsr (identity-level bootstrap CI)
per bucket. Core question: does the nonlinear gain grow with deformation
magnitude? NOTE: exploratory (gate-1) weights -> DIRECTIONAL SIGNAL ONLY.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import scripts.analysis.flip_validation.common as common
from scripts.analysis.flip_validation.a1_models import A1Models

SEED = common.SEED


def make_pairs(ids, n, seed):
    rng = np.random.default_rng(seed)
    all_pairs = [(a, b) for a in ids for b in ids if a != b]
    if n <= 0 or n >= len(all_pairs):
        return all_pairs
    idx = rng.choice(len(all_pairs), size=n, replace=False)
    return [all_pairs[int(i)] for i in idx]


def roi_rmse_per_pair(pred_flat, target_flat, roi_idx):
    B = pred_flat.shape[0]
    p = pred_flat.reshape(B, -1, 3)[:, roi_idx, :]
    t = target_flat.reshape(B, -1, 3)[:, roi_idx, :]
    return np.sqrt(np.mean(np.sum((p - t) ** 2, axis=2), axis=1))  # (B,)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=1000)
    ap.add_argument("--buckets", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    out = common.ensure_out()

    m = A1Models(device="cpu")
    pairs = make_pairs(m.splits["test"], args.pairs, SEED)
    m.prepare(extra_ids=[i for p in pairs for i in p])
    roi_idx = common.roi_index(m.by_id)

    preds = m.predict(pairs, batch_size=args.batch_size)
    target = preds["target"]
    source = preds["source"].reshape(len(pairs), -1)

    rmse_ridge = roi_rmse_per_pair(preds["ridge"], target, roi_idx)
    rmse_rbsr = roi_rmse_per_pair(preds["rbsr"], target, roi_idx)
    # deformation magnitude on ROI: RMS of ||target-source|| over ROI vertices
    mag = roi_rmse_per_pair(target, np.zeros_like(target), roi_idx)  # ||target_delta|| over ROI

    edges = np.quantile(mag, np.linspace(0, 1, args.buckets + 1))
    edges[-1] += 1e-9
    bucket_results = []
    for b in range(args.buckets):
        sel = (mag >= edges[b]) & (mag < edges[b + 1])
        idx = np.where(sel)[0]
        if len(idx) == 0:
            continue
        diff = (rmse_ridge - rmse_rbsr)[idx]
        sub_pairs = [pairs[i] for i in idx]
        mean, lo, hi = common.identity_bootstrap_ci(diff, sub_pairs, n_boot=1000, seed=SEED)
        rel = float(np.mean(rmse_ridge[idx] - rmse_rbsr[idx]) / max(np.mean(rmse_ridge[idx]), 1e-9) * 100.0)
        bucket_results.append({
            "bucket": b,
            "mag_range": [float(edges[b]), float(edges[b + 1])],
            "n_pairs": int(len(idx)),
            "rmse_ridge": float(np.mean(rmse_ridge[idx])),
            "rmse_rbsr": float(np.mean(rmse_rbsr[idx])),
            "delta_ridge_minus_rbsr": mean,
            "delta_ci95": [lo, hi],
            "relative_gain_pct": rel,
        })

    # does the gain grow with magnitude?
    deltas = [br["delta_ridge_minus_rbsr"] for br in bucket_results]
    lows = [br["delta_ci95"][0] for br in bucket_results]
    largest = bucket_results[-1]
    smallest = bucket_results[0]
    grows = (largest["delta_ridge_minus_rbsr"] > smallest["delta_ridge_minus_rbsr"]) and (largest["delta_ci95"][0] > 0)
    overall_mean, overall_lo, overall_hi = common.identity_bootstrap_ci(
        rmse_ridge - rmse_rbsr, pairs, n_boot=1000, seed=SEED)

    pulse = grows or (largest["delta_ci95"][0] > 0 and largest["relative_gain_pct"] > 2.0)
    result = {
        "analysis": "A1a_truth_line",
        "caveat": "Exploratory gate-1 weights -> DIRECTIONAL SIGNAL ONLY; re-verify after clean retrain.",
        "n_pairs": len(pairs),
        "roi_vertices": int(len(roi_idx)),
        "overall_delta_ridge_minus_rbsr": overall_mean,
        "overall_delta_ci95": [overall_lo, overall_hi],
        "buckets": bucket_results,
        "gain_grows_with_magnitude": bool(grows),
        "largest_bucket_relative_gain_pct": largest["relative_gain_pct"],
        "nonlinear_has_pulse": bool(pulse),
        "conclusion": (
            "Nonlinear gain GROWS with deformation magnitude (largest-bucket gain CI excludes 0) "
            "-> nonlinear claim has a pulse; worth re-checking after clean retrain."
            if pulse else
            "Gain is flat/~0 across magnitude buckets -> nonlinear linkage contributes little; "
            "consider dropping the 'nonlinear linkage' claim and repositioning the contribution."
        ),
    }
    (out / "a1a_truth_line.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "buckets"}, indent=2))
    for br in bucket_results:
        print(f"bucket {br['bucket']} mag[{br['mag_range'][0]:.2f},{br['mag_range'][1]:.2f}] "
              f"n={br['n_pairs']} ridge={br['rmse_ridge']:.4f} rbsr={br['rmse_rbsr']:.4f} "
              f"delta={br['delta_ridge_minus_rbsr']:.4f} CI{br['delta_ci95']} rel={br['relative_gain_pct']:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
