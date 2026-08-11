"""Step 4 finisher: geometric per-region flip (laplacian/bi-laplacian) + RBSR/gate qualitative maps.

Completes the per-region accuracy-risk table for ALL 7 MAIN methods, and emits qualitative maps.

PART A — geometric re-prediction (laplacian, bi-laplacian):
  These have NO saved dense test predictions, so we re-solve them on the 100-id test set
  (9900 pairs) with the same strict main-table geometry settings recorded in
  outputs/unified_strict/main_table/MAIN_TABLE_STRICT_DONE.json
  (geometry.solve_linear_handle_baseline; bi-laplacian operator = L@L).
  SELF-CHECK: overall new-flip must reproduce the canonical main table
  (laplacian 0.644, bi-laplacian 3.100). If not, weights/version differ -> flagged.
  Needs scipy (sparse solve). No torch / no GPU.

PART B — safe_fusion: use the frozen dense test prediction saved by the strict
safe-fusion evaluation and self-check against strict_rescore means.

PART C — qualitative maps (numpy + matplotlib, from frozen cache):
  RBSR per-vertex error, RBSR-minus-Ridge improvement, gate activation (gate_map_test.npz),
  residual-correction magnitude. Output PNGs to results/supplemental_qualitative_maps/.

Run:  python scripts/evaluation/region_flip_maps.py --project "/content/drive/MyDrive/FYP final"
"""
from __future__ import annotations
import argparse, json, sys, csv
from pathlib import Path
import numpy as np

REG = ["root", "dorsum", "tip", "alar_left", "alar_right"]
SELF_CHECK_TOL = 1e-6  # percentage-point tolerance against frozen strict means


def effective_flip_tol(user_tol: float, n_pairs: int, n_faces: int) -> float:
    """Allow one face-pair quantum because flip % is a discrete count over pairs*faces."""
    quantum = 100.0 / float(n_pairs * n_faces)
    return max(float(user_tol), quantum + 1e-9)


def default_repo(project: Path) -> Path:
    local = Path("/content/fyp_data")
    return local if (local / "manifest.json").exists() else project / "data"


def default_cache_test(project: Path) -> Path:
    local = Path("/content/ct")
    return local if (local / "pairs.json").exists() else project / "outputs/unified_strict/safe_fusion/cache_test"


def default_safe_fusion_pred(project: Path, cache: Path) -> Path:
    local = cache / "safe_fusion.npy"
    return local if local.exists() else project / "outputs/unified_strict/safe_fusion/evaluation/test_predictions/safe_fusion.npy"


def frozen_mean(path: Path, method: str, metric: str) -> float:
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["method"] == method:
                return float(row[metric])
    raise KeyError(f"{method}.{metric} not found in {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--repo", default="", help="ROI data repo; defaults to /content/fyp_data when present, else PROJECT/data")
    ap.add_argument("--cache-test", default="", help="safe_fusion cache_test; defaults to /content/ct when present, else frozen Drive cache")
    ap.add_argument("--safe-fusion-pred", default="", help="safe_fusion dense prediction .npy; defaults to /content/ct/safe_fusion.npy when present")
    ap.add_argument("--self-check-tol", type=float, default=SELF_CHECK_TOL)
    args = ap.parse_args()
    P = Path(args.project); sys.path.insert(0, str(P / "code"))
    repo = Path(args.repo) if args.repo else default_repo(P)
    cache = Path(args.cache_test) if args.cache_test else default_cache_test(P)
    if not (cache / "pairs.json").exists():
        fallback = default_cache_test(P)
        print(f"WARNING: cache_test missing pairs.json at {cache}; falling back to {fallback}", flush=True)
        cache = fallback
    safe_fusion_pred = Path(args.safe_fusion_pred) if args.safe_fusion_pred else default_safe_fusion_pred(P, cache)
    if not safe_fusion_pred.exists():
        fallback_pred = default_safe_fusion_pred(P, cache)
        print(f"WARNING: safe_fusion pred missing at {safe_fusion_pred}; falling back to {fallback_pred}", flush=True)
        safe_fusion_pred = fallback_pred
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from rhinoform.data import load_rows
    from rhinoform.safe_fusion import signed_fold_indicator
    from rhinoform.geometry import uniform_laplacian, solve_linear_handle_baseline
    from scripts.evaluation.region_flip import build_face_regions, region_metrics
    from scripts.evaluation.main_table import RIDGE_SYS

    _, by_id = load_rows(repo)
    tmpl = next(iter(by_id.values()))
    faces, face_reg, REG_V, n_unassigned = build_face_regions(tmpl)
    n = tmpl["vertices"].shape[0]; landmarks = np.asarray(tmpl["landmarks"], int)
    pairs = [(str(a), str(b)) for a, b in json.loads((cache / "pairs.json").read_text())]
    flip_tol = effective_flip_tol(float(args.self_check_tol), len(pairs), len(faces))
    print(f"repo={repo}")
    print(f"cache_test={cache}")
    print(f"pairs={len(pairs)} faces={len(faces)} unassigned={n_unassigned}")
    print(f"flip_self_check_tol={flip_tol:.9g} percentage points")
    main_done = json.loads((P / "outputs/unified_strict/main_table/MAIN_TABLE_STRICT_DONE.json").read_text())
    handle_weight = float(main_done["handle_weight"])
    main_means = P / "outputs/unified_strict/main_table/main_table_strict_means.csv"
    safe_means = P / "outputs/unified_strict/safe_fusion/strict_rescore/safe_fusion_strict_means.csv"
    expect = {
        "laplacian": frozen_mean(main_means, "laplacian_handles", "normal_flip_pct"),
        "bilaplacian": frozen_mean(main_means, "bilaplacian_handles", "normal_flip_pct"),
        "safe_fusion": frozen_mean(safe_means, "safe_fusion", "normal_flip_pct"),
    }

    rows = []

    # ---------- PART A: geometric re-prediction ----------
    controls = np.stack([(by_id[t]["vertices"] - by_id[s]["vertices"])[landmarks] for s, t in pairs])  # (N,9,3)
    lap = uniform_laplacian(n, faces)
    for name, op in [("laplacian", lap), ("bilaplacian", lap @ lap)]:
        print(f"solving {name} ...", flush=True)
        pred = solve_linear_handle_baseline(controls, landmarks, n, op, handle_weight, RIDGE_SYS)  # (N, V*3) delta
        rmse, flip, overall = region_metrics(by_id, pairs, pred, faces, face_reg, REG_V, signed_fold_indicator)
        exp = expect[name]; diff = abs(overall - exp); ok = diff <= flip_tol
        print(f"[{'OK' if ok else 'FAIL'}] {name} overall_flip={overall:.9f} "
              f"(frozen {exp:.9f}, diff {diff:.3g})")
        row = {"method": name, "protocol": "canonical_strict_posthoc", "source": "main_table_geometry_reprediction",
               "n_test_ids": 100, "n_pairs": len(pairs), "self_check": "OK" if ok else "SELF-CHECK-FAIL",
               "overall_new_flip_pct": round(overall, 4)}
        for r in REG: row[f"{r}_rmse"] = round(rmse[r], 4)
        for r in REG: row[f"{r}_new_flip_pct"] = round(flip[r], 4)
        rows.append(row)

    # ---------- PART B: safe_fusion ----------
    safe_pred = safe_fusion_pred
    if not safe_pred.exists():
        raise FileNotFoundError(f"safe_fusion dense prediction missing: {safe_pred}")
    pred = np.load(safe_pred, mmap_mode="r")
    rmse, flip, overall = region_metrics(by_id, pairs, pred, faces, face_reg, REG_V, signed_fold_indicator)
    exp = expect["safe_fusion"]; diff = abs(overall - exp); ok = diff <= flip_tol
    print(f"[{'OK' if ok else 'FAIL'}] safe_fusion overall_flip={overall:.9f} "
          f"(frozen {exp:.9f}, diff {diff:.3g})")
    row = {"method": "safe_fusion", "protocol": "canonical_strict_posthoc", "source": "frozen_safe_fusion_prediction",
           "n_test_ids": 100, "n_pairs": len(pairs), "self_check": "OK" if ok else "SELF-CHECK-FAIL",
           "overall_new_flip_pct": round(overall, 4)}
    for r in REG: row[f"{r}_rmse"] = round(rmse[r], 4)
    for r in REG: row[f"{r}_new_flip_pct"] = round(flip[r], 4)
    rows.append(row)

    od = P / "results/subunit_metrics_all_methods"; od.mkdir(parents=True, exist_ok=True)
    fail_path = od / "SELF_CHECK_FAILED_step4.json"
    if fail_path.exists():
        fail_path.unlink()
    failures = [r for r in rows if r.get("self_check") != "OK"]
    if failures:
        fail = {"status": "self_check_failed", "tolerance": flip_tol,
                "requested_tolerance": float(args.self_check_tol),
                "one_face_pair_quantum_pct": 100.0 / float(len(pairs) * len(faces)),
                "handle_weight": handle_weight, "ridge_sys": RIDGE_SYS, "expected": expect, "failures": failures}
        fail_path.write_text(json.dumps(fail, indent=2), encoding="utf-8")
        raise SystemExit(2)
    with open(od / "subunit_geometric_flip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    json.dump({"handle_weight": handle_weight, "ridge_sys": RIDGE_SYS, "expected": expect,
               "self_check_tolerance_pct": flip_tol,
               "one_face_pair_quantum_pct": 100.0 / float(len(pairs) * len(faces)),
               "rows": rows},
              open(od / "subunit_geometric_flip.json", "w"), indent=2, ensure_ascii=False)
    print("WROTE", od / "subunit_geometric_flip.csv")

    # ---------- PART C: qualitative maps ----------
    # These are useful figures but should not block the verified numeric table.
    try:
        mp = P / "results/supplemental_qualitative_maps"; mp.mkdir(parents=True, exist_ok=True)
        V = tmpl["vertices"]; X, Y = V[:, 0], V[:, 1]
        ridge = np.load(cache / "ridge.npy", mmap_mode="r"); rbsr = np.load(cache / "rbsr.npy", mmap_mode="r")
        # per-vertex mean error magnitude (chunked to bound memory)
        N = len(pairs); err_ridge = np.zeros(n); err_rbsr = np.zeros(n); resid = np.zeros(n)
        CH = 500
        for a in range(0, N, CH):
            b = min(a + CH, N)
            src = np.stack([by_id[s]["vertices"] for s, _ in pairs[a:b]])
            tgt = np.stack([by_id[t]["vertices"] for _, t in pairs[a:b]])
            tr = tgt - src
            rg = np.asarray(ridge[a:b]).reshape(b - a, n, 3).copy()
            rb = np.asarray(rbsr[a:b]).reshape(b - a, n, 3).copy()
            rg[:, landmarks] = tr[:, landmarks]
            rb[:, landmarks] = tr[:, landmarks]
            err_ridge += np.sum(np.linalg.norm(rg - tr, axis=2), axis=0)
            err_rbsr += np.sum(np.linalg.norm(rb - tr, axis=2), axis=0)
            resid += np.sum(np.linalg.norm(rb - rg, axis=2), axis=0)
        err_ridge /= N; err_rbsr /= N; resid /= N
        gm = np.load(P / "outputs/unified_strict/rbsr_gate/gate_map_test.npz", allow_pickle=True)
        gate = np.asarray(gm["vertex_gate_mean"])

        def scatter(c, title, fname, cmap="viridis"):
            fig, ax = plt.subplots(figsize=(3.4, 4.2)); s = ax.scatter(X, Y, c=c, s=2, cmap=cmap)
            ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=8); fig.colorbar(s, ax=ax, fraction=0.046)
            fig.savefig(mp / fname, dpi=150, bbox_inches="tight"); plt.close(fig); print("WROTE", mp / fname, flush=True)

        scatter(err_rbsr, "RBSR per-vertex error (canonical strict)", "fig_rbsr_per_vertex_error.png")
        scatter(err_ridge - err_rbsr, "RBSR improvement over Ridge (ridge_err - rbsr_err)", "fig_rbsr_improvement_over_ridge.png", cmap="RdBu")
        scatter(gate, "RBSR gate activation (vertex_gate_mean)", "fig_rbsr_gate_activation.png", cmap="magma")
        scatter(resid, "RBSR residual-correction magnitude", "fig_rbsr_residual_magnitude.png")
    except Exception as e:
        fail = {"status": "qualitative_maps_failed_nonfatal", "error_type": type(e).__name__, "error": str(e)}
        (P / "results/supplemental_qualitative_maps").mkdir(parents=True, exist_ok=True)
        (P / "results/supplemental_qualitative_maps/QUALITATIVE_MAPS_FAILED.json").write_text(
            json.dumps(fail, indent=2), encoding="utf-8")
        print(f"WARNING: qualitative maps failed non-fatally: {type(e).__name__}: {e}", flush=True)
    print("Step 4 done.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
