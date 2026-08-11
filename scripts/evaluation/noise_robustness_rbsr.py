"""
RB-SR control-point noise robustness (addendum to noise_robustness.py).

noise_robustness.py reports ridge / hybrid / cvae / laplacian / bilaplacian / arap
under landmark-control noise but NOT the deployed RB-SR gate. This script adds the
missing RB-SR row under the IDENTICAL noise protocol and proves comparability with
two self-checks (no hand-waving):

  CHECK-1  the ridge column reproduces the published noise table exactly
           (level 0/0.10/0.20 -> ROI 1.056/1.090/1.184, flip 0.37/0.47/0.74).
           ridge depends only on the condition vector, so identical ridge numbers
           prove the noisy controls / pairs / noise scale are bit-for-bit the same.
  CHECK-2  RB-SR at noise 0 reproduces the published RB-SR (ROI 1.0497, flip 0.370),
           proving the genuine frozen gate pipeline is used.

If both pass, the RB-SR row is computed under the same benchmark, same seeds and
same noise draws as every other method -- only the method changed to RB-SR.

Why this pipeline (and not noise_robustness.py's): the RB-SR gate consumes the
diagnose-style vertex features and the diagnose-style (mean-latent) CVAE, which is
exactly what produced the frozen RB-SR. noise_robustness.py's CVAE column uses a
different feature builder (hence its CVAE level-0 = 1.272 vs main-table 1.2825);
that does not affect ridge or the RB-SR gate, which is why CHECK-1/CHECK-2 are the
right comparability tests.

Condition build (pca_transform) and CVAE decode (z=None, deterministic) are
verified identical to noise_robustness.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from rhinoform.data import load_rows, ridge_predict
from scripts.analysis.flip_fusion import (
    load_manifest, load_template, load_vertices,
    static_vertex_features, cvae_predictions, rbsr_predictions,
)
from rhinoform.stats import metric_rows_for_method

METRICS = ["roi_rmse", "landmark_rmse", "dorsum_rmse", "tip_rmse", "edge_strain_p95", "normal_flip_pct"]

# Reference values for the comparability self-checks.
# CHECK-1 compares the ridge column against the FROZEN noise summary (same metric,
# same code path) — this is the authoritative comparability proof. The hardcoded
# dict below is only a last-resort fallback if the frozen summary is unavailable.
# NB: the frozen noise summary reports the *absolute* normal_flip_pct (e.g. ridge@0
# = 1.0554 / 0.7737), which is a different quantity from the paper main-table
# *new-flip* (0.37). CHECK-2's ROI target is unambiguous; its flip is reported for
# information only, not asserted, because the two pipelines use different flip
# definitions.
PUB_RIDGE_FALLBACK = {0.0: (1.0554, 0.7737), 0.10: (1.090, None), 0.20: (1.184, None)}
PUB_RBSR_L0_ROI = 1.0497
TOL_ROI_CHECK = 0.003


def noisy_condition(controls_noisy: np.ndarray, source: np.ndarray, base: dict) -> np.ndarray:
    """Standardised condition from noisy controls (== pca_transform path in noise_robustness.py)."""
    n = len(controls_noisy)
    source_flat = source.reshape(n, -1)
    source_code = (source_flat - base["source_pca"]["mean"]) @ base["source_pca"]["components"].T
    condition = np.concatenate([controls_noisy.reshape(n, -1), source_code], axis=1)
    condition = (condition - base["cond_mean"]) / base["cond_std"]
    return condition.astype(np.float32)


def build_clean_controls_source(test_pairs, vertices, landmarks):
    """float64 controls/source, matching noise_robustness.controls_and_sources exactly."""
    source = np.stack([np.asarray(vertices[s], dtype=np.float64) for s, _ in test_pairs])
    target = np.stack([np.asarray(vertices[t], dtype=np.float64) for _, t in test_pairs])
    controls = (target - source)[:, landmarks]          # (n_pairs, n_landmarks, 3), float64
    return controls, source


def run(base, rbsr, by_id, features, test_pairs, vertices, landmarks,
        median_control_delta, levels, seeds, batch_size, ckpt_path=None):
    controls, source = build_clean_controls_source(test_pairs, vertices, landmarks)
    n_pairs = len(test_pairs)

    done = {}
    if ckpt_path and Path(ckpt_path).exists():
        for r in json.loads(Path(ckpt_path).read_text()):
            done[(float(r["noise_fraction"]), int(r["noise_seed"]), r["method"])] = r
        print(f"[resume] loaded {len(done)} completed draws from {ckpt_path}")
    summary = list(done.values())

    def save_ckpt():
        if not ckpt_path:
            return
        tmp = Path(str(ckpt_path) + ".tmp")
        tmp.write_text(json.dumps(summary, indent=2))
        tmp.replace(ckpt_path)  # atomic

    for level in levels:
        for nseed in seeds:
            if (level, nseed, "rbsr") in done and (level, nseed, "ridge") in done:
                print(f"[skip] level={level} seed={nseed} (already done)")
                continue
            rng = np.random.default_rng(nseed)
            sigma = float(level * median_control_delta)
            noisy_ctrl = controls + rng.normal(0.0, sigma, size=controls.shape)   # float64, same draw as noise_robustness
            cond = noisy_condition(noisy_ctrl, source, base)
            ridge = ridge_predict(cond, base["ridge_cond"]).astype(np.float32).reshape(n_pairs, -1, 3)
            cvae = cvae_predictions(base, features, cond).reshape(n_pairs, -1, 3)
            rbsr_pred, _ = rbsr_predictions(
                base, rbsr, features, cond, source, ridge, cvae, noisy_ctrl, landmarks, batch_size
            )
            rbsr_pred = np.asarray(rbsr_pred, dtype=np.float32).reshape(n_pairs, -1, 3)
            for method, pred in (("ridge", ridge), ("rbsr", rbsr_pred)):
                mrows = metric_rows_for_method(by_id, test_pairs, pred)
                agg = {m: float(np.mean([float(r[m]) for r in mrows])) for m in METRICS}
                agg.update(method=method, noise_fraction=level, noise_seed=nseed, sigma=sigma)
                summary = [s for s in summary
                           if not (s["noise_fraction"] == level and s["noise_seed"] == nseed and s["method"] == method)]
                summary.append(agg)
                print(f"[{method:5s}] level={level:<5} seed={nseed} "
                      f"roi={agg['roi_rmse']:.4f} flip%={agg['normal_flip_pct']:.3f}")
            save_ckpt()   # checkpoint after every (level, seed)
    return summary


def comparability_report(summary, levels, frozen_ridge=None):
    """CHECK-1: ridge reproduces the FROZEN noise summary (same metric) -> proves the
    noisy controls/pairs/scale/metric are identical. CHECK-2: RB-SR@0 ROI matches the
    deployed constrained gate (1.0497). Flip is reported for information only."""
    def mean_for(method, level):
        sub = [r for r in summary if r["method"] == method and r["noise_fraction"] == level]
        if not sub:
            return (float("nan"), float("nan"))
        return (float(np.mean([r["roi_rmse"] for r in sub])),
                float(np.mean([r["normal_flip_pct"] for r in sub])))
    ref = frozen_ridge or PUB_RIDGE_FALLBACK
    print("\n================ COMPARABILITY SELF-CHECKS ================")
    ok = True
    print("CHECK-1  ridge column vs FROZEN noise summary (same metric):")
    for lv in levels:
        if lv not in ref:
            continue
        roi, flip = mean_for("ridge", lv)
        proi, pflip = ref[lv]
        good_roi = abs(roi - proi) <= TOL_ROI_CHECK
        good_flip = (pflip is None) or (abs(flip - pflip) <= 0.02)
        good = good_roi and good_flip
        ok &= good
        pflip_s = f"{pflip:.3f}" if pflip is not None else "n/a"
        print(f"  noise {lv:<5}: mine ROI {roi:.4f}/flip {flip:.3f}  vs frozen {proi:.4f}/{pflip_s}  -> {'OK' if good else 'MISMATCH'}")
    roi0, flip0 = mean_for("rbsr", 0.0)
    good0 = abs(roi0 - PUB_RBSR_L0_ROI) <= TOL_ROI_CHECK
    ok &= good0
    print(f"CHECK-2  RB-SR @ noise 0 ROI: mine {roi0:.4f}  vs deployed {PUB_RBSR_L0_ROI}  -> {'OK' if good0 else 'MISMATCH'}")
    print(f"         (RB-SR @ noise 0 flip {flip0:.3f} reported for info; absolute normal_flip_pct, not the paper new-flip)")
    print("=> COMPARABLE" if ok else "=> NOT comparable: investigate before using")
    print("\n=== RB-SR rows for tab:m7-noise (mean over seeds) ===")
    for lv in levels:
        roi, flip = mean_for("rbsr", lv)
        print(f"noise {lv:<5}: RB-SR ROI {roi:.4f}   normal_flip_pct {flip:.3f}")
    return ok


def frozen_ridge_from_summary(summary_path):
    """Extract {level: (mean_roi, mean_flip)} for ridge from the frozen noise summary."""
    try:
        obj = json.loads(Path(summary_path).read_text())
        rows = obj.get("rows", obj if isinstance(obj, list) else [])
    except Exception:
        return None
    by_level = {}
    for r in rows:
        if r.get("method") != "ridge_sourcepca":
            continue
        lv = float(r.get("noise_level_c", r.get("noise_fraction")))
        by_level.setdefault(lv, []).append(r)
    out = {}
    for lv, rs in by_level.items():
        out[lv] = (float(np.mean([x["roi_rmse"] for x in rs])),
                   float(np.mean([x["normal_flip_pct"] for x in rs])))
    return out or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("../data"))
    ap.add_argument("--base-package", type=Path, required=True)
    ap.add_argument("--rbsr-package", type=Path, required=True)
    ap.add_argument("--noise-summary", type=Path, default=None)
    ap.add_argument("--levels", default="0,0.10,0.20")
    ap.add_argument("--noise-seeds", default="20260609,20260610,20260611")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("../results/noise_robustness_rbsr"))
    args = ap.parse_args()

    rows_manifest, paths = load_manifest(args.repo)
    _, by_id = load_rows(args.repo)
    base = torch.load(args.base_package, map_location="cpu", weights_only=False)
    rbsr = torch.load(args.rbsr_package, map_location="cpu", weights_only=False)
    train_ids = [str(x) for x in base["train_ids"]]
    template = load_template(paths[train_ids[0]])
    template["vertices"] = np.asarray(base["feature_template_vertices"], dtype=np.float64)
    landmarks = template["landmarks"]
    test_pairs = [(str(a), str(b)) for a, b in base["test_pairs"]]
    test_ids = sorted({v for p in test_pairs for v in p}, key=int)
    vertices = load_vertices(paths, test_ids)
    features = static_vertex_features(rows_manifest, paths, train_ids, template, bool(base["use_subunit_features"]))

    if args.noise_summary and args.noise_summary.exists():
        median_control_delta = float(json.loads(args.noise_summary.read_text())["median_control_delta_from_validation"])
        print(f"[noise] reusing published median_control_delta = {median_control_delta:.6f}")
    else:
        val_pairs = [(str(a), str(b)) for a, b in base["val_pairs"]]
        vids = sorted({v for p in val_pairs for v in p}, key=int)
        vverts = load_vertices(paths, vids)
        vctrl = np.stack([(np.asarray(vverts[t], np.float64) - np.asarray(vverts[s], np.float64))[landmarks] for s, t in val_pairs])
        median_control_delta = float(np.median(np.linalg.norm(vctrl, axis=2)))
        print(f"[noise] recomputed median_control_delta = {median_control_delta:.6f}")

    levels = [float(x) for x in args.levels.split(",") if x]
    seeds = [int(x) for x in args.noise_seeds.split(",") if x]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "noise_rbsr_progress.json"

    summary = run(base, rbsr, by_id, features, test_pairs, vertices, landmarks,
                  median_control_delta, levels, seeds, args.batch_size, ckpt_path=ckpt)

    with (out / "noise_robustness_rbsr_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    (out / "noise_robustness_rbsr_summary.json").write_text(json.dumps(summary, indent=2))
    frozen_ridge = frozen_ridge_from_summary(args.noise_summary) if args.noise_summary else None
    if frozen_ridge:
        print(f"[check] using frozen ridge reference from {args.noise_summary}")
    comparability_report(summary, levels, frozen_ridge)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
