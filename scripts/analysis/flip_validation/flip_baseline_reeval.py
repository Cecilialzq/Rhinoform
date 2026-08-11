"""Baseline-corrected (i2i-intrinsic-removed) foldover re-evaluation.

Motivation
----------
The frozen ``normal_flip_pct`` metric compares each predicted ROI mesh's face
normals to the SOURCE mesh. But an identity->identity (i2i) transfer between two
different people *intrinsically* reverses some faces in the TRUE target geometry
itself -- those are not model defects, they are the real shape difference. So the
absolute flip rate conflates:

  * intrinsic flips already present in the true target deformation, and
  * model-induced NEW flips (the actual foldover artefacts we care about).

This is exactly the ``n_new`` lesson from Gate 2B (report new, not total),
applied to orientation reversal. The honest, baseline-relative quantity is

  new_flip(method) = mean over faces of [ method reverses orientation vs source
                                          AND true target does NOT reverse ].

sign(signed_fold_indicator.det) is identical to sign(normal_cosine) (Gate 2A),
so det < 0 reproduces the original flip definition exactly; we just additionally
subtract the intrinsic target flips.

ZERO RETRAIN: this only re-aggregates predictions from the existing exploratory
gate-1 weights through the same control-point projector used everywhere in A1.
All numbers are DIRECTIONAL (gate-1 pre-fix weights), to be re-checked after a
clean retrain. Output is written to local disk and packaged separately.
"""
from __future__ import annotations

import json
import time

import numpy as np

import scripts.analysis.flip_validation.common as common
from scripts.analysis.flip_validation.a1_models import A1Models
from rhinoform.safe_fusion import signed_fold_indicator
from rhinoform.sampling import coverage_balanced_ordered_pairs


def fold_mask(source_v: np.ndarray, edited_v: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-face orientation-reversal mask (det < 0), == the legacy normal-flip set."""
    det, _ = signed_fold_indicator(source_v, edited_v, faces)
    return det < 0.0


def main() -> int:
    out = common.ensure_out()
    prov = common.env_provenance()
    splits = common.load_split_ids()
    test_ids = splits["test"]

    budget = 1500
    pairs = coverage_balanced_ordered_pairs(test_ids, budget, common.SEED)
    print(f"[flip] test identities={len(test_ids)} pairs={len(pairs)}", flush=True)

    models = A1Models(device="cpu")
    models.prepare(test_ids)
    faces = models.faces
    n = models.n
    alpha = models.alpha

    # ---- intrinsic target flip on ALL 9900 ordered test pairs (model-free) ----
    all_pairs = [(a, b) for a in test_ids for b in test_ids if a != b]
    t0 = time.perf_counter()
    intrinsic_all = np.empty(len(all_pairs), dtype=np.float64)
    by_id = models.by_id
    for k, (a, b) in enumerate(all_pairs):
        sv = by_id[a]["vertices"]
        tv = by_id[b]["vertices"]
        intrinsic_all[k] = fold_mask(sv, tv, faces).mean() * 100.0
    print(f"[flip] intrinsic target flip over ALL {len(all_pairs)} pairs: "
          f"mean={intrinsic_all.mean():.4f}% p95={np.percentile(intrinsic_all,95):.4f}% "
          f"max={intrinsic_all.max():.4f}% ({time.perf_counter()-t0:.1f}s)", flush=True)

    # ---- model predictions on the coverage-balanced subset ----
    res = models.predict(pairs, batch_size=32)
    source = res["source"].reshape(len(pairs), n, 3).astype(np.float64)
    target_delta = res["target"].reshape(len(pairs), n, 3).astype(np.float64)
    ridge_delta = res["ridge"].reshape(len(pairs), n, 3).astype(np.float64)
    rbsr_delta = res["rbsr"].reshape(len(pairs), n, 3).astype(np.float64)
    cvae_delta = res["cvae_raw"].reshape(len(pairs), n, 3).astype(np.float64)
    # global hybrid (alpha-blend), pushed through the SAME projector for fairness
    hybrid_delta_raw = (ridge_delta + alpha * (cvae_delta - ridge_delta)).reshape(len(pairs), -1)
    hybrid_delta = models.project(hybrid_delta_raw.astype(np.float32), res["controls"]).reshape(len(pairs), n, 3)

    methods = {
        "ridge": ridge_delta,
        "hybrid": hybrid_delta,
        "rbsr": rbsr_delta,
        "cvae": cvae_delta,
    }

    P = len(pairs)
    target_v = source + target_delta
    target_masks = np.empty((P, faces.shape[0]), dtype=bool)
    for i in range(P):
        target_masks[i] = fold_mask(source[i], target_v[i], faces)
    target_flip = target_masks.mean(axis=1) * 100.0

    per_pair = {}
    for name, delta in methods.items():
        edited = source + delta
        abs_pp = np.empty(P)
        new_pp = np.empty(P)
        missed_pp = np.empty(P)
        for i in range(P):
            m = fold_mask(source[i], edited[i], faces)
            abs_pp[i] = m.mean() * 100.0
            new_pp[i] = (m & ~target_masks[i]).mean() * 100.0
            missed_pp[i] = (~m & target_masks[i]).mean() * 100.0
        per_pair[name] = {"abs": abs_pp, "new": new_pp, "missed": missed_pp}

    def stat(name, key):
        v = per_pair[name][key]
        mean, lo, hi = common.identity_bootstrap_ci(v, pairs, n_boot=1000)
        return {"mean": mean, "ci95": [lo, hi], "p95": float(np.percentile(v, 95)), "max": float(v.max())}

    def delta_stat(a, b, key):
        d = per_pair[a][key] - per_pair[b][key]
        mean, lo, hi = common.identity_bootstrap_ci(d, pairs, n_boot=1000)
        return {"mean": mean, "ci95": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0)}

    tgt_mean, tgt_lo, tgt_hi = common.identity_bootstrap_ci(target_flip, pairs, n_boot=1000)

    summary = {
        "provenance": prov,
        "caveat": "DIRECTIONAL: exploratory gate-1 (pre-fix) weights; re-check after a clean retrain.",
        "n_pairs_subset": P,
        "n_pairs_all": len(all_pairs),
        "metric_note": (
            "abs = legacy normal_flip_pct (det<0 vs source, == original metric). "
            "new = model-induced foldover = faces the method reverses vs source that the TRUE "
            "target does NOT reverse (the n_new analogue). missed = true target folds the method "
            "failed to reproduce."
        ),
        "intrinsic_target_flip_pct": {
            "subset": {"mean": tgt_mean, "ci95": [tgt_lo, tgt_hi],
                       "p95": float(np.percentile(target_flip, 95)), "max": float(target_flip.max())},
            "all_9900": {"mean": float(intrinsic_all.mean()),
                         "p95": float(np.percentile(intrinsic_all, 95)),
                         "max": float(intrinsic_all.max())},
        },
        "methods": {},
        "deltas_new_flip": {},
        "deltas_abs_flip": {},
    }
    for name in methods:
        v_abs = per_pair[name]["abs"].mean()
        v_new = per_pair[name]["new"].mean()
        summary["methods"][name] = {
            "abs_flip_pct": stat(name, "abs"),
            "new_flip_pct": stat(name, "new"),
            "missed_flip_pct": stat(name, "missed"),
            "intrinsic_fraction_of_abs": float(1.0 - v_new / v_abs) if v_abs > 0 else None,
        }
    for a, b in (("rbsr", "ridge"), ("hybrid", "ridge"), ("rbsr", "hybrid")):
        summary["deltas_new_flip"][f"{a}_minus_{b}"] = delta_stat(a, b, "new")
        summary["deltas_abs_flip"][f"{a}_minus_{b}"] = delta_stat(a, b, "abs")

    (out / "flip_baseline_reeval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n================ BASELINE-CORRECTED FLIP ================", flush=True)
    print(f"intrinsic target flip (subset): {tgt_mean:.4f}%  | all-9900: {intrinsic_all.mean():.4f}%")
    for name in methods:
        m = summary["methods"][name]
        print(f"  {name:7s} abs={m['abs_flip_pct']['mean']:.4f}%  "
              f"new={m['new_flip_pct']['mean']:.4f}%  "
              f"(intrinsic share of abs={m['intrinsic_fraction_of_abs']*100:.1f}%)")
    print("  --- deltas on NEW (model-induced) flip ---")
    for k, v in summary["deltas_new_flip"].items():
        print(f"  {k}: {v['mean']:+.5f}  CI{v['ci95']}  excl0={v['excludes_zero']}")
    print("  --- deltas on ABS (legacy) flip ---")
    for k, v in summary["deltas_abs_flip"].items():
        print(f"  {k}: {v['mean']:+.5f}  CI{v['ci95']}  excl0={v['excludes_zero']}")
    print("=========================================================", flush=True)
    print(f"[flip] wrote -> {out/'flip_baseline_reeval.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
