"""Compute per-subunit RMSE + per-subunit NEW flip for ALL methods, POST-HOC.

WHY THIS IS REPRODUCIBLE (no re-training):
  It reads the ALREADY-FROZEN dense test predictions and recomputes region metrics
  with the EXACT algorithm used to produce outputs/unified_strict/rbsr_region_diagnostic.json
  (face->subunit = majority vote of a face's 3 vertex region labels; new flip via
  safe_fusion.signed_fold_indicator, baseline-relative). Because it uses frozen
  predictions, the per-method OVERALL flip MUST match the canonical main table; the
  script asserts this as a SELF-CHECK. If a self-check fails, the environment/data
  differs and results must NOT be merged into the canonical table.

DATA DEPENDENCIES (verified present on Drive 2026-06-21):
  outputs/unified_strict/safe_fusion/cache_test/{ridge,rbsr,arap,global_hybrid,target_delta}.npy , pairs.json
  runs/required_primary_chain0_scale676/neural_field_predictions_cvae_ew0p1_lw0.npz  (keys: pair*, ridge*, cvae*)
  data/  (846 ROI npz, via data.load_rows)
  NOTE: cache_test/*.npy are DELTAS (pred displacement); pred_shape = source + delta.

HANDLED BY STEP 4: laplacian and bilaplacian are re-solved with the frozen main-table
  geometry settings, and safe_fusion is read from its frozen dense prediction. This script
  emits temporary PENDING rows only so the notebook can replace them after Step 4.

Run (Colab or any numpy env with the Drive mounted):
  python scripts/evaluation/region_flip.py --project "/content/drive/MyDrive/FYP final"
Only numpy is required (no torch/scipy needed for the post-hoc part).
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

REG = ["root", "dorsum", "tip", "alar_left", "alar_right"]
SELF_CHECK_TOL = 1e-6  # absolute percentage-point tolerance against frozen means


def default_repo(project: Path) -> Path:
    local = Path("/content/fyp_data")
    return local if (local / "manifest.json").exists() else project / "data"


def default_cache_test(project: Path) -> Path:
    local = Path("/content/ct")
    return local if (local / "pairs.json").exists() else project / "outputs/unified_strict/safe_fusion/cache_test"


def default_neural_pred_npz(project: Path, cache: Path) -> Path:
    local = cache / "neural_field_predictions_cvae_ew0p1_lw0.npz"
    return local if local.exists() else project / "runs/required_primary_chain0_scale676/neural_field_predictions_cvae_ew0p1_lw0.npz"


def frozen_mean(path: Path, method: str, metric: str) -> float:
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["method"] == method:
                return float(row[metric])
    raise KeyError(f"{method}.{metric} not found in {path}")


def expected_flips(project: Path) -> dict[str, float]:
    main = project / "outputs/unified_strict/main_table/main_table_strict_means.csv"
    safe = project / "outputs/unified_strict/safe_fusion/strict_rescore/safe_fusion_strict_means.csv"
    return {
        "ridge": frozen_mean(main, "ridge_sourcepca", "normal_flip_pct"),
        "cvae": frozen_mean(main, "cvae_only", "normal_flip_pct"),
        "hybrid_alpha_0.1": frozen_mean(main, "hybrid_alpha_0.1", "normal_flip_pct"),
        "rbsr": frozen_mean(safe, "rbsr", "normal_flip_pct"),
        "arap": frozen_mean(safe, "arap", "normal_flip_pct"),
        "global_hybrid": frozen_mean(safe, "global_hybrid", "normal_flip_pct"),
    }


def build_face_regions(tmpl):
    faces = np.asarray(tmpl["faces"], int)
    sub = tmpl["subunits"]
    lms = np.asarray(tmpl["landmarks"], int)
    vlabel = np.full(tmpl["vertices"].shape[0], -1, int)
    for ri, name in enumerate(REG):
        vlabel[np.asarray(sub[name], int)] = ri
    face_reg = np.array([
        np.bincount(vlabel[f][vlabel[f] >= 0], minlength=len(REG)).argmax()
        if (vlabel[f] >= 0).any() else -1
        for f in faces
    ])
    n_unassigned = int((face_reg < 0).sum())
    free = lambda idx: np.asarray(idx, int)[~np.isin(np.asarray(idx, int), lms)]
    REG_V = {name: free(sub[name]) for name in REG}
    return faces, face_reg, REG_V, n_unassigned


def region_metrics(by_id, pairs, pred_delta, faces, face_reg, REG_V, signed_fold):
    """STRICT region metrics. Hard-fix control landmarks before fold scoring."""
    N = len(pairs)
    src = np.stack([by_id[s]["vertices"] for s, _ in pairs])
    tgt = np.stack([by_id[t]["vertices"] for _, t in pairs])
    true = tgt - src
    P = np.asarray(pred_delta).reshape(N, -1, 3)
    landmarks = np.asarray(next(iter(by_id.values()))["landmarks"], int)
    rmse = {n: float(np.sqrt(np.mean(np.sum((P[:, i] - true[:, i]) ** 2, axis=2), axis=1)).mean())
            for n, i in REG_V.items()}
    new_flip = {n: 0.0 for n in REG}
    overall = 0.0
    for k, (s, t) in enumerate(pairs):
        so, ta = by_id[s]["vertices"], by_id[t]["vertices"]
        pk = np.asarray(P[k], dtype=np.float64).copy()
        pk[landmarks] = true[k, landmarks]
        pf = signed_fold(so, so + pk, faces)[0] < 0
        tf = signed_fold(so, ta, faces)[0] < 0
        nf = pf & ~tf
        overall += nf.mean() * 100.0
        for ri, n in enumerate(REG):
            m = face_reg == ri
            if m.any():
                new_flip[n] += nf[m].mean() * 100.0
    new_flip = {n: v / N for n, v in new_flip.items()}
    return rmse, new_flip, overall / N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--repo", default="", help="ROI data repo; defaults to /content/fyp_data when present, else PROJECT/data")
    ap.add_argument("--cache-test", default="", help="safe_fusion cache_test; defaults to /content/ct when present, else frozen Drive cache")
    ap.add_argument("--neural-pred-npz", default="", help="CVAE/hybrid dense prediction npz; defaults to /content/ct when present")
    ap.add_argument("--out", default="results/subunit_metrics_all_methods")
    ap.add_argument("--self-check-tol", type=float, default=SELF_CHECK_TOL)
    args = ap.parse_args()
    P = Path(args.project)
    repo = Path(args.repo) if args.repo else default_repo(P)
    cache = Path(args.cache_test) if args.cache_test else default_cache_test(P)
    neural_npz = Path(args.neural_pred_npz) if args.neural_pred_npz else default_neural_pred_npz(P, cache)
    expect = expected_flips(P)
    sys.path.insert(0, str(P / "code"))
    from rhinoform.data import load_rows
    from rhinoform.safe_fusion import signed_fold_indicator
    _, by_id = load_rows(repo)
    tmpl = next(iter(by_id.values()))
    faces, face_reg, REG_V, n_unassigned = build_face_regions(tmpl)
    print(f"face->subunit majority vote; faces={len(faces)} unassigned={n_unassigned}")

    pairs = [(str(a), str(b)) for a, b in json.loads((cache / "pairs.json").read_text())]
    print(f"repo={repo}")
    print(f"cache_test={cache}")
    print(f"test pairs={len(pairs)}")

    preds = {}
    # --- methods with frozen dense deltas in cache_test ---
    for m, fn in [("ridge", "ridge.npy"), ("rbsr", "rbsr.npy"),
                  ("arap", "arap.npy"), ("global_hybrid", "global_hybrid.npy")]:
        fp = cache / fn
        if fp.exists():
            preds[m] = np.load(fp, mmap_mode="r")
    # --- cvae + hybrid(alpha=0.1) from the chain0 prediction npz ---
    npz = neural_npz
    if npz.exists():
        d = np.load(npz, allow_pickle=True); files = list(d.files)
        pk = next(f for f in files if "pair" in f.lower())
        rk = next((f for f in files if "ridge" in f.lower() and "cond" in f.lower()),
                  next(f for f in files if "ridge" in f.lower()))
        ck = next(f for f in files if "cvae" in f.lower())
        npairs = [(str(a), str(b)) for a, b in np.asarray(d[pk]).tolist()]
        assert npairs == pairs, "npz pair order != cache_test pair order; refusing (would misalign)"
        rg = np.asarray(d[rk], np.float64).reshape(len(pairs), -1, 3)
        cv = np.asarray(d[ck], np.float64).reshape(len(pairs), -1, 3)
        preds["cvae"] = cv
        preds["hybrid_alpha_0.1"] = (1.0 - 0.1) * rg + 0.1 * cv

    rows = []
    for m, arr in preds.items():
        rmse, flip, overall = region_metrics(by_id, pairs, arr, faces, face_reg, REG_V, signed_fold_indicator)
        ok = (m not in expect) or abs(overall - expect[m]) <= float(args.self_check_tol)
        flag = "OK" if ok else "SELF-CHECK-FAIL"
        print(f"[{flag}] {m:16s} overall_flip={overall:.9f} (frozen {expect.get(m,'?')})")
        row = {"method": m, "protocol": "canonical_strict_posthoc", "n_test_ids": 100, "n_pairs": len(pairs),
               "overall_new_flip_pct": round(overall, 4), "self_check": flag}
        for n in REG: row[f"{n}_rmse"] = round(rmse[n], 4)
        for n in REG: row[f"{n}_new_flip_pct"] = round(flip[n], 4)
        rows.append(row)

    failures = [r for r in rows if r.get("self_check") == "SELF-CHECK-FAIL"]
    outd = P / args.out; outd.mkdir(parents=True, exist_ok=True)
    fail_path = outd / "SELF_CHECK_FAILED_posthoc.json"
    if fail_path.exists():
        fail_path.unlink()
    if failures:
        fail = {"status": "self_check_failed", "tolerance": float(args.self_check_tol),
                "expected": expect, "failures": failures}
        fail_path.write_text(json.dumps(fail, indent=2), encoding="utf-8")
        raise SystemExit(2)

    # methods handled by Step 4.
    for m in ["laplacian", "bilaplacian", "safe_fusion"]:
        row = {"method": m, "protocol": "canonical_strict_posthoc", "n_test_ids": 100, "n_pairs": len(pairs),
               "overall_new_flip_pct": "PENDING(step4)", "self_check": "N/A"}
        for n in REG: row[f"{n}_rmse"] = "PENDING"
        for n in REG: row[f"{n}_new_flip_pct"] = "PENDING"
        rows.append(row)

    keys = list(rows[0].keys())
    with open(outd / "subunit_rmse_flip_all_methods_POSTHOC.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    json.dump({"face_to_subunit": "majority vote of 3 vertex region labels; -1 unassigned",
               "n_unassigned_faces": n_unassigned, "self_check_tol_pct": SELF_CHECK_TOL,
               "rows": rows}, open(outd / "subunit_rmse_flip_all_methods_POSTHOC.json", "w"), indent=2, ensure_ascii=False)
    print("WROTE", outd / "subunit_rmse_flip_all_methods_POSTHOC.csv")
    print("NOTE: laplacian/bilaplacian/safe_fusion are PENDING here and must be replaced by Step 4 before final merge.")


if __name__ == "__main__":
    raise SystemExit(main())
