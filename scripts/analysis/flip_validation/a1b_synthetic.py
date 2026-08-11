"""A1b (corroborating synthetic line): controlled edits, rbsr divergence from ridge.

Edit directions are taken from REAL landmark-displacement PCA (single-region +
multi-region), scaled by alpha in {0.1..1.25}. For each (identity, direction,
alpha) all three predictors (ridge / rbsr / ARAP geometric base) go through the
SAME RBF projector; we measure
  divergence = ||d_rbsr - d_ridge|| / max(||d_ridge - d_base||, eps)
per region. Core question: does divergence grow with alpha (esp. alar/tip)?
NOTE: exploratory gate-1 weights -> DIRECTIONAL SIGNAL ONLY.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import scripts.analysis.flip_validation.common as common
from scripts.analysis.flip_validation.a1_models import A1Models
from rhinoform.baselines import arap_predict_vectorised
from rhinoform.geometry import solve_linear_handle_baseline, uniform_laplacian

SEED = common.SEED
REGIONS = common.SUBUNITS
ALPHAS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25]
EPS = 1e-9


def landmark_regions(top):
    subs = top["subunits"]
    lm = top["landmarks"]
    lm_region = []
    for v in lm:
        owner = None
        for r in REGIONS:
            if int(v) in set(int(x) for x in subs[r]):
                owner = r
                break
        lm_region.append(owner)
    return lm_region


def build_directions(real_L, lm_region, typical):
    """real_L: (N,K,3) real landmark deltas. Return dict name -> (K,3) direction."""
    K = real_L.shape[1]
    flat = real_L.reshape(len(real_L), -1)
    flat_c = flat - flat.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(flat_c, full_matrices=False)
    dirs = {}
    for i in range(3):
        d = vt[i].reshape(K, 3)
        d = d / max(np.linalg.norm(d), 1e-9) * typical
        dirs[f"pca{i}_multiregion"] = d
    # single-region mean directions
    mean_disp = real_L.mean(axis=0)  # (K,3)
    for r in ("tip", "alar_left", "alar_right", "dorsum"):
        mask = np.asarray([reg == r for reg in lm_region])
        if not mask.any():
            continue
        d = np.zeros_like(mean_disp)
        d[mask] = mean_disp[mask]
        nrm = np.linalg.norm(d)
        if nrm < 1e-9:
            continue
        dirs[f"single_{r}"] = d / nrm * typical
    return dirs


def region_norm(delta, verts):
    return float(np.sqrt(np.sum(delta[verts] ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    out = common.ensure_out()

    m = A1Models(device="cpu")
    rng = np.random.default_rng(SEED)
    test_ids = m.splits["test"]
    sel_ids = [test_ids[i] for i in sorted(rng.choice(len(test_ids), size=min(args.ids, len(test_ids)), replace=False))]
    m.prepare(extra_ids=sel_ids)
    top = m.top
    subs = top["subunits"]
    lm_region = landmark_regions(top)

    # real landmark deltas (val) for PCA directions + typical magnitude
    val_pairs = [(a, b) for a in m.splits["val"][:25] for b in m.splits["val"][:25] if a != b]
    val_ids_needed = sorted({i for p in val_pairs for i in p}, key=int)
    val_by = common.load_meshes(val_ids_needed)
    L = np.stack([(val_by[t]["vertices"] - val_by[s]["vertices"])[top["landmarks"]] for s, t in val_pairs], axis=0)
    typical = float(np.median(np.sqrt(np.sum(L.reshape(len(L), -1) ** 2, axis=1))))
    dirs = build_directions(L, lm_region, typical)

    lap = uniform_laplacian(top["n_vertices"], top["faces"])

    curves = {}
    for dname, dvec in dirs.items():
        per_alpha = []
        for alpha in ALPHAS:
            ctrl = (alpha * dvec)[None].repeat(len(sel_ids), axis=0)  # (B,K,3)
            pred = m.predict_synthetic(sel_ids, ctrl, batch_size=args.batch_size)
            d_ridge = pred["ridge"].reshape(len(sel_ids), -1, 3)
            d_rbsr = pred["rbsr"].reshape(len(sel_ids), -1, 3)
            # ARAP geometric base + same projector
            sources = [m.by_id[s]["vertices"] for s in sel_ids]
            init = solve_linear_handle_baseline(ctrl, top["landmarks"], top["n_vertices"], lap, 1000.0, 1e-8)
            arap = arap_predict_vectorised(sources, ctrl, top["landmarks"], top["faces"], lap, init, 1000.0, 1e-8, 3)
            d_base = m.project(arap.astype(np.float32), pred["controls"]).reshape(len(sel_ids), -1, 3)
            # per-region divergence averaged over identities
            region_div = {}
            for r in REGIONS:
                verts = subs[r]
                num = np.asarray([region_norm(d_rbsr[i] - d_ridge[i], verts) for i in range(len(sel_ids))])
                den = np.asarray([region_norm(d_ridge[i] - d_base[i], verts) for i in range(len(sel_ids))])
                div = num / np.maximum(den, EPS)
                region_div[r] = float(np.mean(div))
            per_alpha.append({"alpha": alpha, "region_divergence": region_div})
            print(f"{dname} alpha={alpha}: " + " ".join(f"{r}={per_alpha[-1]['region_divergence'][r]:.3f}" for r in REGIONS), flush=True)
        curves[dname] = per_alpha

    # growth assessment: divergence at max alpha vs min alpha, per region
    growth = {}
    for r in REGIONS:
        grows_any = False
        ratios = []
        for dname, pa in curves.items():
            v_lo = pa[0]["region_divergence"][r]
            v_hi = pa[-1]["region_divergence"][r]
            ratios.append(v_hi / max(v_lo, 1e-9))
            if v_hi > 1.2 * max(v_lo, 1e-9) and v_hi > 0.05:
                grows_any = True
        growth[r] = {"grows_with_alpha": bool(grows_any), "median_hi_lo_ratio": float(np.median(ratios))}

    alar_tip_grows = any(growth[r]["grows_with_alpha"] for r in ("tip", "alar_left", "alar_right"))
    max_div = max(
        d["region_divergence"][r]
        for pa in curves.values() for d in pa for r in REGIONS
    )
    pulse = alar_tip_grows and max_div > 0.05

    result = {
        "analysis": "A1b_synthetic_line",
        "caveat": "Exploratory gate-1 weights -> DIRECTIONAL SIGNAL ONLY; re-verify after clean retrain.",
        "n_identities": len(sel_ids),
        "alphas": ALPHAS,
        "directions": list(dirs.keys()),
        "divergence_definition": "||d_rbsr - d_ridge|| / max(||d_ridge - d_base(ARAP)||, eps), per region",
        "curves": curves,
        "growth_by_region": growth,
        "alar_tip_divergence_grows": bool(alar_tip_grows),
        "max_divergence_observed": float(max_div),
        "nonlinear_has_pulse": bool(pulse),
        "conclusion": (
            "Divergence grows with alpha (notably in alar/tip) -> rbsr meaningfully departs from ridge "
            "under larger edits; nonlinear claim has a pulse, worth re-checking after clean retrain."
            if pulse else
            "Divergence stays ~flat/near-0 across alpha -> rbsr barely departs from ridge; "
            "nonlinear linkage adds little, consider dropping the claim."
        ),
    }
    (out / "a1b_synthetic.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "curves"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
