"""A1a alar-restricted re-aggregation: absolute accuracy delta in alar subregions.

Pure re-aggregation of the EXISTING A1a configuration (no retrain, no full alpha
sweep). A1a did not cache per-pair predictions, so we reproduce the IDENTICAL
deterministic A1a run (same seed=20260615, same exploratory gate-1 weights, same
1000 test pairs) and VERIFY the recomputed overall-ROI per-bucket delta matches
the saved a1a_truth_line.json to full precision (proving same data). We then
restrict the absolute delta RMSE_ridge - RMSE_rbsr to alar_left / alar_right /
alar-combined, using the SAME 5 magnitude buckets defined on the OVERALL ROI
||target-source|| (not re-bucketed on alar), with identity-level bootstrap CIs.
"""
from __future__ import annotations

import json

import numpy as np

import scripts.analysis.flip_validation.common as common
from scripts.analysis.flip_validation.a1_models import A1Models
from scripts.analysis.flip_validation.a1a_truth_line import make_pairs, roi_rmse_per_pair

SEED = common.SEED
OUT = common.ensure_out()


def trend(values):
    lo, hi = values[0], values[-1]
    if hi > 1.10 * max(lo, 1e-12):
        return "grows"
    if hi < 0.90 * max(lo, 1e-12):
        return "shrinks"
    return "flat"


def per_bucket(diff, mag, edges, pairs, label):
    rows, means = [], []
    for b in range(len(edges) - 1):
        sel = (mag >= edges[b]) & (mag < edges[b + 1])
        idx = np.where(sel)[0]
        if len(idx) == 0:
            continue
        sub_pairs = [pairs[i] for i in idx]
        mean, lo, hi = common.identity_bootstrap_ci(diff[idx], sub_pairs, n_boot=1000, seed=SEED)
        rows.append({"bucket": b, "mag_range": [float(edges[b]), float(edges[b + 1])],
                     "n_pairs": int(len(idx)), "abs_delta_ridge_minus_rbsr": mean, "abs_delta_ci95": [lo, hi]})
        means.append(mean)
    return rows, trend(means), means


def main() -> int:
    m = A1Models(device="cpu")
    pairs = make_pairs(m.splits["test"], 1000, SEED)
    m.prepare(extra_ids=[i for p in pairs for i in p])
    roi_idx = common.roi_index(m.by_id)
    subs = m.top["subunits"]
    alar_l = np.asarray(subs["alar_left"], dtype=np.int64)
    alar_r = np.asarray(subs["alar_right"], dtype=np.int64)
    alar_lr = np.unique(np.concatenate([alar_l, alar_r]))

    preds = m.predict(pairs, batch_size=16)
    target = preds["target"]

    # SAME magnitude definition + SAME 5-bucket quantile edges as original A1a
    mag = roi_rmse_per_pair(target, np.zeros_like(target), roi_idx)
    edges = np.quantile(mag, np.linspace(0, 1, 6))
    edges[-1] += 1e-9

    # --- reproduction check: overall-ROI per-bucket delta vs saved JSON ---
    rmse_ridge_roi = roi_rmse_per_pair(preds["ridge"], target, roi_idx)
    rmse_rbsr_roi = roi_rmse_per_pair(preds["rbsr"], target, roi_idx)
    diff_roi = rmse_ridge_roi - rmse_rbsr_roi
    roi_rows, _, _ = per_bucket(diff_roi, mag, edges, pairs, "roi")
    saved = json.loads((OUT / "a1a_truth_line.json").read_text(encoding="utf-8"))
    max_diff = 0.0
    for rr, sb in zip(roi_rows, saved["buckets"]):
        max_diff = max(max_diff, abs(rr["abs_delta_ridge_minus_rbsr"] - sb["delta_ridge_minus_rbsr"]))
        max_diff = max(max_diff, abs(rr["abs_delta_ci95"][0] - sb["delta_ci95"][0]))
        max_diff = max(max_diff, abs(rr["abs_delta_ci95"][1] - sb["delta_ci95"][1]))

    # --- alar-restricted absolute delta, same buckets ---
    out = {"left": {}, "right": {}, "combined": {}}
    trends = {}
    for name, idx_set in (("left", alar_l), ("right", alar_r), ("combined", alar_lr)):
        rr = roi_rmse_per_pair(preds["ridge"], target, idx_set)
        rb = roi_rmse_per_pair(preds["rbsr"], target, idx_set)
        rows, tr, _ = per_bucket(rr - rb, mag, edges, pairs, name)
        out[name] = rows
        trends[name] = tr

    combined_grows = trends["combined"] == "grows"
    any_alar_grows = any(t == "grows" for t in trends.values())

    result = {
        "metric": "absolute delta = RMSE_ridge - RMSE_rbsr restricted to alar subunits (subunit_alar_*)",
        "buckets_defined_on": "OVERALL ROI ||target-source|| quantiles (same edges as A1a; NOT re-bucketed on alar)",
        "reproduction_check_max_abs_diff_vs_saved_overall_roi": max_diff,
        "reproduction_is_bit_identical": bool(max_diff < 1e-9),
        "alar_left": out["left"],
        "alar_right": out["right"],
        "alar_combined": out["combined"],
        "trend_by_region": trends,
        "alar_combined_abs_delta_grows_with_magnitude": bool(combined_grows),
        "any_alar_abs_delta_grows_with_magnitude": bool(any_alar_grows),
    }

    if combined_grows or any_alar_grows:
        decision = "DOWNGRADE_TO_MILD_SUPPORTIVE"
        verdict = (
            "LOCKED = DOWNGRADE_TO_MILD_SUPPORTIVE. The absolute accuracy delta (RMSE_ridge - RMSE_rbsr) "
            "RESTRICTED to the alar subunits GROWS with deformation magnitude (same direction as the A1b alar "
            "divergence), so the alar divergence DOES convert into a measurable accuracy gain there. Claim: "
            "'the nonlinear tissue-linkage model delivers a mild-but-real, alar-localised accuracy gain that "
            "increases with edit magnitude' (localised claim; overall accuracy gain stays small ~3%, abs ~0.04 "
            "and shrinks with magnitude, so this is NOT a headline large-deformation claim)."
        )
    else:
        decision = "DELETE_GAIN_CLAIM_BEHAVIOURAL_ONLY"
        verdict = (
            "LOCKED = DELETE the accuracy-gain claim (behavioural observation only). The absolute accuracy delta "
            "restricted to the alar subunits is flat/shrinking across magnitude buckets (same as the overall ROI), "
            "so the A1b alar divergence is 'different, not better': the nonlinear model diverges more from linear "
            "under larger alar edits but this does NOT translate into a measurable accuracy gain. Claim: 'the "
            "nonlinear model's divergence from linear grows under large alar edits but yields no measurable "
            "accuracy benefit' -> do not claim a nonlinear accuracy advantage."
        )

    result["A1_final_decision"] = decision
    result["A1_final_verdict"] = verdict
    result["caveat"] = "DIRECTIONAL SIGNAL ONLY: exploratory gate-1 (pre-fix) weights; re-verify after clean retrain."

    # merge into master
    mp = OUT / "gate23a1_results.json"
    master = json.loads(mp.read_text(encoding="utf-8"))
    a1 = master["A1_nonlinear_vs_linear"]
    a1["a1a_alar_absolute_delta_by_bucket"] = result
    a1["A1_overall_decision"] = decision
    a1["A1_final_verdict"] = verdict
    master["GO_NO_GO_SUMMARY"]["A1_decision"] = decision
    master["GO_NO_GO_SUMMARY"]["A1_alar_accuracy_grows_with_magnitude"] = bool(combined_grows or any_alar_grows)
    mp.write_text(json.dumps(master, indent=2), encoding="utf-8")

    print("reproduction bit-identical:", result["reproduction_is_bit_identical"], "max_diff=", max_diff)
    for name in ("alar_left", "alar_right", "alar_combined"):
        print(f"\n== {name} (trend={trends[name.split('_')[1]] if name!='alar_combined' else trends['combined']}) ==")
        for r in result[name]:
            print(f"  bucket {r['bucket']} mag[{r['mag_range'][0]:.2f},{r['mag_range'][1]:.2f}] n={r['n_pairs']} "
                  f"absΔ={r['abs_delta_ridge_minus_rbsr']:.5f} CI[{r['abs_delta_ci95'][0]:.5f},{r['abs_delta_ci95'][1]:.5f}]")
    print("\nA1_final_decision:", decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
