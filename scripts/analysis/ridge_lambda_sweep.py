"""SUPPLEMENTAL diagnostic: ridge lambda validation sweep (NOT part of the frozen main line).

WHY: the final pipeline hardcodes ridge lambda = 100.0 (train.py:627; implementation_constants.md).
This script answers the reviewer/viva question "why 100? is it near-optimal?" by sweeping lambda
ON VALIDATION ONLY and evaluating the test split only at lambda=100 and the validation-selected
lambda. It does NOT change the main table (which stays lambda=100.0); results go to
results/supplemental_ridge_lambda/.

PROTOCOL ALIGNMENT (strict, identity-disjoint):
  - frozen split manifest (test=100 / val=70 / train_pool=676)
  - train pairs = chain0 / scale676 / primary (870 ordered pairs) -- same as the main ridge
  - source-PCA dim = 16, fitted on TRAIN identities only (no leakage)
  - validation = all ordered pairs of the 70 val identities (70*69 = 4830)
  - test = all ordered pairs of the 100 test identities (100*99 = 9900), evaluated ONCE
  - metrics = strict free-ROI RMSE (9 landmark vertices excluded) and strict baseline-relative new flip

SELF-CHECK: re-evaluates lambda=100 on test; it MUST reproduce the frozen ridge ROI and
frozen ridge new-flip read from outputs/unified_strict/main_table/main_table_strict_means.csv
within tolerance. If not, the environment/code version diverges -> results are not mergeable.

NUMPY-ONLY for the ridge math, but it imports train.py helpers (which import torch); torch is
present in the current/Colab env. Heavy part = loading 846 meshes (copy to local/Colab scratch first).

Run:
  python scripts/analysis/ridge_lambda_sweep.py --project "/content/drive/MyDrive/FYP final"
"""
from __future__ import annotations
import argparse, json, sys, csv
from pathlib import Path
import os, tempfile
import numpy as np

LAMBDA_GRID = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
SELF_CHECK_TOL = 1e-6
FLIP_SELF_CHECK_TOL = 1e-6


def effective_flip_tol(user_tol: float, n_pairs: int, n_faces: int) -> float:
    """Allow one face-pair quantum because flip % is a discrete count over pairs*faces."""
    quantum = 100.0 / float(n_pairs * n_faces)
    return max(float(user_tol), quantum + 1e-9)


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def all_ordered(ids):
    ids = [str(i) for i in ids]
    return [(a, b) for a in ids for b in ids if a != b]


def free_rmse(by_id, pairs, pred_delta, landmarks):
    """Mean per-pair strict free-ROI RMSE, matching strict main-table aggregation."""
    N = len(pairs)
    src = np.stack([by_id[s]["vertices"] for s, _ in pairs])
    tgt = np.stack([by_id[t]["vertices"] for _, t in pairs])
    true = (tgt - src).reshape(N, -1, 3)
    P = np.asarray(pred_delta).reshape(N, -1, 3)
    V = P.shape[1]
    free = np.ones(V, bool); free[np.asarray(landmarks, int)] = False
    err = P[:, free] - true[:, free]
    per_pair = np.sqrt(np.mean(np.sum(err ** 2, axis=2), axis=1))
    return float(np.mean(per_pair))


def default_repo(project: Path) -> Path:
    local = Path("/content/fyp_data")
    return local if (local / "manifest.json").exists() else project / "data"


def frozen_main_value(project: Path, method: str, metric: str) -> float:
    import csv
    path = project / "outputs/unified_strict/main_table/main_table_strict_means.csv"
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["method"] == method:
                return float(row[metric])
    raise KeyError(f"{method}.{metric} not found in {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--repo", default="", help="ROI data repo; defaults to /content/fyp_data when present, else PROJECT/data")
    ap.add_argument("--self-check-tol", type=float, default=SELF_CHECK_TOL)
    args = ap.parse_args()
    P = Path(args.project)
    repo = Path(args.repo) if args.repo else default_repo(P)
    sys.path.insert(0, str(P / "code"))
    from rhinoform.data import load_rows
    from rhinoform import train as nfe  # fit_truncated_pca, pair_conditions, pair_arrays, ridge_fit, ridge_predict, flatten_vertices
    from rhinoform.sampling import read_pairs

    man = json.loads((P / "splits/facescape_847/split_manifest.json").read_text())
    test_ids = [str(x) for x in man["test_ids"]]
    val_ids = [str(x) for x in man["val_ids"]]
    train_ids = [str(x) for x in man["train_pool_ids"]]
    assert (len(test_ids), len(val_ids), len(train_ids)) == (100, 70, 676), "split size mismatch"
    train_pairs = read_pairs(P / "splits/facescape_847/pairs/primary_chain_0_scale676.json")
    val_pairs = all_ordered(val_ids)      # 4830
    test_pairs = all_ordered(test_ids)    # 9900
    assert len(val_pairs) == 4830 and len(test_pairs) == 9900

    _, by_id = load_rows(repo)
    landmarks = np.asarray(next(iter(by_id.values()))["landmarks"], int)

    source_pca = nfe.fit_truncated_pca(nfe.flatten_vertices(by_id, train_ids), 16)  # TRAIN-only
    cond_tr, _, _, _, _, mean, std = nfe.pair_conditions(by_id, train_pairs, source_pca)
    cond_val, _, _, _, _, _, _ = nfe.pair_conditions(by_id, val_pairs, source_pca, mean, std)
    cond_te, _, _, _, _, _, _ = nfe.pair_conditions(by_id, test_pairs, source_pca, mean, std)
    _, y_tr = nfe.pair_arrays(by_id, train_pairs)

    # flip helper (baseline-relative new flip; hard-fix control landmarks before fold scoring,
    # matching the strict main-table / region-diagnostic convention).
    from rhinoform.safe_fusion import signed_fold_indicator
    faces = np.asarray(next(iter(by_id.values()))["faces"], int)
    def flip_of(pred, pairs):
        pred = np.asarray(pred).reshape(len(pairs), -1, 3)
        tot = 0.0
        for k, (s, t) in enumerate(pairs):
            so = by_id[s]["vertices"]; ta = by_id[t]["vertices"]
            pk = pred[k].copy(); pk[landmarks] = (ta - so)[landmarks]
            pf = signed_fold_indicator(so, so + pk, faces)[0] < 0
            tf = signed_fold_indicator(so, ta, faces)[0] < 0
            tot += (pf & ~tf).mean() * 100.0
        return tot / len(pairs)

    # FULL sweep: per-lambda validation RMSE AND new-flip -> draws the accuracy<->flip frontier.
    curve = []
    for lam in LAMBDA_GRID:
        model = nfe.ridge_fit(cond_tr, y_tr, float(lam))
        pred_val = nfe.ridge_predict(cond_val, model)
        v_rmse = free_rmse(by_id, val_pairs, pred_val, landmarks)
        v_flip = flip_of(pred_val, val_pairs)
        curve.append({"lambda": lam, "val_free_rmse": round(v_rmse, 6), "val_new_flip_pct": round(v_flip, 4)})
        print(f"lambda={lam:>7}: val_free_rmse={v_rmse:.6f}  val_new_flip={v_flip:.4f}")
    best = min(curve, key=lambda r: r["val_free_rmse"])

    # test anchors (test touched only for lambda=100 self-check and the selected lambda)
    def test_roi(lam):
        return free_rmse(by_id, test_pairs, nfe.ridge_predict(cond_te, nfe.ridge_fit(cond_tr, y_tr, float(lam))), landmarks)
    def test_flip(lam):
        return flip_of(nfe.ridge_predict(cond_te, nfe.ridge_fit(cond_tr, y_tr, float(lam))), test_pairs)
    test_at_best = test_roi(best["lambda"]); test_at_100 = test_roi(100.0)
    flip_at_best = test_flip(best["lambda"]); flip_at_100 = test_flip(100.0)
    frozen_ridge_flip = frozen_main_value(P, "ridge_sourcepca", "normal_flip_pct")
    reference_ridge_roi = frozen_main_value(P, "ridge_sourcepca", "roi_rmse")
    diff = abs(test_at_100 - reference_ridge_roi)
    ok = diff <= float(args.self_check_tol)
    flip_tol = effective_flip_tol(FLIP_SELF_CHECK_TOL, len(test_pairs), len(faces))
    flip_diff = abs(flip_at_100 - frozen_ridge_flip)
    flip_check_ok = bool(flip_diff <= flip_tol)
    print(f"\n[SELF-CHECK] test@100 ROI={test_at_100:.9f} vs frozen {reference_ridge_roi:.9f} (diff {diff:.2g}) -> {'OK' if ok else 'DIVERGENT'}")
    print(f"[SELF-CHECK] test@100 flip={flip_at_100:.9f} vs frozen {frozen_ridge_flip:.9f} "
          f"(diff {flip_diff:.2g}, tol {flip_tol:.2g}) -> {'OK' if flip_check_ok else 'DIVERGENT'}")

    out = {"note": "SUPPLEMENTAL diagnostic; main line keeps fixed lambda=100.0",
           "protocol": "strict identity-disjoint; val=4830 pairs, test=9900 pairs; source_pca_dim=16 train-only",
           "repo": str(repo), "lambda_grid": LAMBDA_GRID, "validation_curve": curve,
           "selected_lambda_on_validation": best["lambda"], "selected_val_free_rmse": best["val_free_rmse"],
           "test_free_rmse_at_selected": round(test_at_best, 6), "test_free_rmse_at_lambda100": round(test_at_100, 6),
           "test_new_flip_at_selected": round(flip_at_best, 4), "test_new_flip_at_lambda100": round(flip_at_100, 4),
           "frozen_ridge_flip": float(frozen_ridge_flip), "flip_self_check_abs_diff": float(flip_diff),
           "flip_self_check_tolerance": float(flip_tol),
           "one_face_pair_quantum_pct": float(100.0 / float(len(test_pairs) * len(faces))),
           "flip_self_check_ok": bool(flip_check_ok),
           "reference_ridge_roi": float(reference_ridge_roi), "self_check_abs_diff": float(diff),
           "self_check_tolerance": float(args.self_check_tol), "self_check_lambda100_reproduces_reference": bool(ok),
           "interpretation": ("lambda is reported as an accuracy<->flip sensitivity/frontier knob. The main line keeps "
                              "lambda=100 as the frozen protocol setting; changing it would require rerunning downstream "
                              "hybrid, RBSR, and Safe Fusion evidence."),
           "verdict": ("lambda=100 is near-optimal" if abs(test_at_100 - test_at_best) < 0.003
                       else f"validation free-RMSE prefers lambda={best['lambda']}; but it trades flip; main line keeps 100")}
    od = P / "results/supplemental_ridge_lambda"; od.mkdir(parents=True, exist_ok=True)
    fail_path = od / "SELF_CHECK_FAILED.json"
    if fail_path.exists():
        fail_path.unlink()
    atomic_json(od / "ridge_lambda_selection.json", out)
    with open(od / "ridge_lambda_grid_validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lambda", "val_free_rmse", "val_new_flip_pct"]); w.writeheader(); [w.writerow(r) for r in curve]

    # ---- figures ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    lams = [r["lambda"] for r in curve]; rms = [r["val_free_rmse"] for r in curve]; fls = [r["val_new_flip_pct"] for r in curve]
    # fig 1: separate panels avoid a misleading dual-axis crossing.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.6), sharex=True)
    ax1.semilogx(lams, rms, "o-", color="#1f77b4", label="val free-ROI RMSE")
    ax2.semilogx(lams, fls, "s--", color="#d62728", label="val new-flip %")
    for ax in (ax1, ax2):
        ax.axvline(100, ls=":", c="gray")
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=8)
    ax1.set_ylabel("val free-ROI RMSE")
    ax2.set_ylabel("val new-flip %")
    ax2.set_xlabel("ridge lambda (log)")
    fig.suptitle("Ridge lambda sweep: accuracy vs flip (SUPPLEMENTAL; main line lambda=100)", fontsize=11)
    fig.tight_layout()
    fig.savefig(od / "fig_ridge_lambda_sweep_curves.png", dpi=150, bbox_inches="tight")
    fig.savefig(od / "fig_ridge_lambda_selection_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # fig 2: accuracy-flip frontier scatter
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.plot(rms, fls, "-", color="gray", alpha=0.5)
    sc = ax.scatter(rms, fls, c=np.log10(lams), cmap="viridis", s=60, zorder=3)
    for r in curve: ax.annotate(f"λ={r['lambda']:g}", (r["val_free_rmse"], r["val_new_flip_pct"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("val free-ROI RMSE (accuracy →)"); ax.set_ylabel("val new-flip % (risk →)")
    ax.set_title("Ridge lambda accuracy–flip frontier (validation)"); fig.colorbar(sc, label="log10(lambda)")
    fig.savefig(od / "fig_ridge_lambda_frontier.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("verdict:", out["verdict"]); print("WROTE", od)
    if not (ok and flip_check_ok):
        atomic_json(fail_path, out)
        raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
