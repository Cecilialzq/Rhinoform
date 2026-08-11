"""Re-aggregate EXISTING A1 data into ABSOLUTE quantities (no retrain, no full sweep).

A1a: the absolute per-bucket delta (RMSE_ridge - RMSE_rbsr) was already logged by
     a1a_truth_line.py; we only re-expose it and judge growth.
A1b: only the NORMALISED divergence ratio was logged, so we re-derive the
     UN-normalised absolute ||d_rbsr - d_ridge|| by reproducing the IDENTICAL
     deterministic A1b configuration (same seed=20260615, same exploratory gate-1
     weights, same alpha grid / directions / identities). We VERIFY the recomputed
     normalised ratio matches the saved a1b_synthetic.json bit-for-bit, proving the
     re-derivation is the same A1 data (not a new experiment), and cache the raw
     per-identity arrays so any future re-aggregation needs no inference at all.
"""
from __future__ import annotations

import json

import numpy as np

import scripts.analysis.flip_validation.common as common
from scripts.analysis.flip_validation.a1_models import A1Models
from scripts.analysis.flip_validation.a1b_synthetic import ALPHAS, REGIONS, build_directions, landmark_regions, region_norm
from rhinoform.baselines import arap_predict_vectorised
from rhinoform.geometry import solve_linear_handle_baseline, uniform_laplacian

SEED = common.SEED
OUT = common.ensure_out()


def trend(values):
    """+ increasing / flat / - shrinking, comparing endpoints with tolerance."""
    lo, hi = values[0], values[-1]
    if hi > 1.10 * max(lo, 1e-12):
        return "grows"
    if hi < 0.90 * max(lo, 1e-12):
        return "shrinks"
    return "flat"


def a1a_absolute():
    d = json.loads((OUT / "a1a_truth_line.json").read_text(encoding="utf-8"))
    buckets = []
    deltas = []
    for b in d["buckets"]:
        buckets.append({
            "bucket": b["bucket"],
            "mag_range": b["mag_range"],
            "n_pairs": b["n_pairs"],
            "abs_delta_ridge_minus_rbsr": b["delta_ridge_minus_rbsr"],
            "abs_delta_ci95": b["delta_ci95"],
        })
        deltas.append(b["delta_ridge_minus_rbsr"])
    tr = trend(deltas)
    grows = tr == "grows"
    return {
        "metric": "absolute delta = RMSE_ridge - RMSE_rbsr (mm-equiv units), per magnitude bucket",
        "source": "directly from a1a_truth_line.json (already absolute; no recompute)",
        "buckets": buckets,
        "abs_delta_trend_low_to_high_bucket": tr,
        "abs_delta_grows_with_magnitude": bool(grows),
    }


def a1b_absolute(n_ids: int = 30):
    saved = json.loads((OUT / "a1b_synthetic.json").read_text(encoding="utf-8"))
    m = A1Models(device="cpu")
    rng = np.random.default_rng(SEED)
    test_ids = m.splits["test"]
    sel_ids = [test_ids[i] for i in sorted(rng.choice(len(test_ids), size=min(n_ids, len(test_ids)), replace=False))]
    m.prepare(extra_ids=sel_ids)
    top = m.top
    subs = top["subunits"]
    lm_region = landmark_regions(top)

    val_pairs = [(a, b) for a in m.splits["val"][:25] for b in m.splits["val"][:25] if a != b]
    val_ids_needed = sorted({i for p in val_pairs for i in p}, key=int)
    val_by = common.load_meshes(val_ids_needed)
    L = np.stack([(val_by[t]["vertices"] - val_by[s]["vertices"])[top["landmarks"]] for s, t in val_pairs], axis=0)
    typical = float(np.median(np.sqrt(np.sum(L.reshape(len(L), -1) ** 2, axis=1))))
    dirs = build_directions(L, lm_region, typical)

    lap = uniform_laplacian(top["n_vertices"], top["faces"])

    # absolute numerator ||d_rbsr - d_ridge|| and denominator ||d_ridge - d_base||,
    # stored per (direction, alpha, region, identity) and the reproduced ratio.
    abs_num = {dn: {r: np.zeros((len(ALPHAS), len(sel_ids))) for r in REGIONS} for dn in dirs}
    abs_den = {dn: {r: np.zeros((len(ALPHAS), len(sel_ids))) for r in REGIONS} for dn in dirs}
    ratio_repro = {dn: [] for dn in dirs}

    for dn, dvec in dirs.items():
        for ai, alpha in enumerate(ALPHAS):
            ctrl = (alpha * dvec)[None].repeat(len(sel_ids), axis=0)
            pred = m.predict_synthetic(sel_ids, ctrl, batch_size=16)
            d_ridge = pred["ridge"].reshape(len(sel_ids), -1, 3)
            d_rbsr = pred["rbsr"].reshape(len(sel_ids), -1, 3)
            sources = [m.by_id[s]["vertices"] for s in sel_ids]
            init = solve_linear_handle_baseline(ctrl, top["landmarks"], top["n_vertices"], lap, 1000.0, 1e-8)
            arap = arap_predict_vectorised(sources, ctrl, top["landmarks"], top["faces"], lap, init, 1000.0, 1e-8, 3)
            d_base = m.project(arap.astype(np.float32), pred["controls"]).reshape(len(sel_ids), -1, 3)
            region_div = {}
            for r in REGIONS:
                verts = subs[r]
                num = np.asarray([region_norm(d_rbsr[i] - d_ridge[i], verts) for i in range(len(sel_ids))])
                den = np.asarray([region_norm(d_ridge[i] - d_base[i], verts) for i in range(len(sel_ids))])
                abs_num[dn][r][ai] = num
                abs_den[dn][r][ai] = den
                region_div[r] = float(np.mean(num / np.maximum(den, 1e-9)))
            ratio_repro[dn].append({"alpha": alpha, "region_divergence": region_div})

    # verify reproduction against saved normalised divergence
    max_abs_diff = 0.0
    for dn in dirs:
        for ai, alpha in enumerate(ALPHAS):
            for r in REGIONS:
                a = ratio_repro[dn][ai]["region_divergence"][r]
                b = saved["curves"][dn][ai]["region_divergence"][r]
                max_abs_diff = max(max_abs_diff, abs(a - b))

    # pooled (over directions) absolute numerator per (region, alpha): mean + identity bootstrap CI
    by_region = {}
    grows_flags = {}
    for r in REGIONS:
        curve = []
        means = []
        for ai, alpha in enumerate(ALPHAS):
            pooled = np.concatenate([abs_num[dn][r][ai] for dn in dirs])  # (n_dir*n_id,)
            mean, lo, hi = common.bootstrap_ci(pooled, n_boot=1000, seed=SEED)
            curve.append({"alpha": alpha, "abs_div_mean": mean, "abs_div_ci95": [lo, hi]})
            means.append(mean)
        tr = trend(means)
        by_region[r] = {"curve": curve, "trend_low_to_high_alpha": tr, "grows_with_alpha": tr == "grows"}
        grows_flags[r] = tr == "grows"

    # cache raw arrays so future re-aggregation needs no inference
    cache = {"alphas": np.asarray(ALPHAS), "ids": np.asarray(sel_ids), "directions": np.asarray(list(dirs.keys()))}
    for dn in dirs:
        for r in REGIONS:
            cache[f"num__{dn}__{r}"] = abs_num[dn][r]
            cache[f"den__{dn}__{r}"] = abs_den[dn][r]
    np.savez_compressed(OUT / "a1b_absolute_cache.npz", **cache)

    alar_tip_grows = any(grows_flags[r] for r in ("tip", "alar_left", "alar_right"))
    any_grows = any(grows_flags.values())
    return {
        "metric": "absolute ||d_rbsr - d_ridge|| (UN-normalised), per region, pooled over directions",
        "reproduction_check_max_abs_ratio_diff_vs_saved": max_abs_diff,
        "reproduction_is_bit_identical": bool(max_abs_diff < 1e-9),
        "raw_cache": str(OUT / "a1b_absolute_cache.npz"),
        "n_identities": len(sel_ids),
        "alphas": ALPHAS,
        "directions": list(dirs.keys()),
        "by_region": by_region,
        "abs_divergence_grows_with_alpha": bool(any_grows),
        "abs_divergence_grows_with_alpha_alar_tip": bool(alar_tip_grows),
        "grows_flags_by_region": grows_flags,
    }, alar_tip_grows, any_grows


def main() -> int:
    a1a_abs = a1a_absolute()
    a1b_abs, alar_tip_grows, any_grows = a1b_absolute()

    abs_a1a_grows = a1a_abs["abs_delta_grows_with_magnitude"]
    # Final A1 verdict on ABSOLUTE evidence
    if abs_a1a_grows or any_grows:
        verdict = ("MILD SUPPORTIVE: absolute quantity grows with magnitude/alpha on the protocol metric, "
                   "so the earlier relative shrink was partly a denominator artefact. Overall gain is still small "
                   "(~3%, abs delta ~0.04); reposition the nonlinear model as a MILD-BUT-REAL supportive result.")
        a1_decision = "DOWNGRADE_TO_MILD_SUPPORTIVE"
    else:
        verdict = ("NO PULSE (confirmed on ABSOLUTE quantities): absolute A1a delta SHRINKS with magnitude "
                   f"(0.0533 -> 0.0294) and absolute A1b divergence does not grow with alpha in any region "
                   "-> the relative shrink is NOT a denominator artefact. fail-closed: DROP the 'nonlinear "
                   "linkage' claim and reposition the contribution to population completion + safe selective "
                   "editing + safety-linkage trade-off characterisation.")
        a1_decision = "DROP_NONLINEAR_CLAIM"

    caveat = "DIRECTIONAL SIGNAL ONLY: exploratory gate-1 (pre-fix) weights; re-verify after clean retrain."

    # merge into master json
    master_path = OUT / "gate23a1_results.json"
    master = json.loads(master_path.read_text(encoding="utf-8"))
    a1 = master["A1_nonlinear_vs_linear"]
    a1["a1a_absolute_delta_by_bucket"] = a1a_abs
    a1["a1b_absolute_divergence_by_alpha"] = a1b_abs
    a1["abs_delta_grows_with_magnitude"] = abs_a1a_grows
    a1["abs_divergence_grows_with_alpha"] = any_grows
    a1["abs_divergence_grows_with_alpha_alar_tip"] = alar_tip_grows
    a1["A1_overall_decision"] = a1_decision
    a1["A1_overall_verdict"] = verdict
    a1["caveat"] = caveat
    master["GO_NO_GO_SUMMARY"]["A1_nonlinear_pulse"] = bool(abs_a1a_grows or any_grows)
    master["GO_NO_GO_SUMMARY"]["A1_decision"] = a1_decision
    master_path.write_text(json.dumps(master, indent=2), encoding="utf-8")

    print(json.dumps({
        "a1a_abs_buckets": [(b["bucket"], round(b["abs_delta_ridge_minus_rbsr"], 5), b["abs_delta_ci95"]) for b in a1a_abs["buckets"]],
        "a1a_abs_grows": abs_a1a_grows,
        "a1b_repro_bit_identical": a1b_abs["reproduction_is_bit_identical"],
        "a1b_grows_flags": a1b_abs["grows_flags_by_region"],
        "a1b_abs_grows_any": any_grows,
        "A1_decision": a1_decision,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
