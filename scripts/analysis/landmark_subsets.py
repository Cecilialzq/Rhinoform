"""Stage-1 feasibility audit: is there any landmark-SELECTION room? (numpy-only, NO training.)

Reads the Stage-0 export (scripts/analysis/identifiability.py) and evaluates ALL 2^9-1 = 511
non-empty landmark subsets under the affine/PCA target-shape prior, with NO retraining and NO neural
model. For each subset it computes the affine reconstruction risk

    residual(S) = tr(Sigma) - tr( Sigma S' (S Sigma S')^+ S Sigma )                 (lower = better)

i.e. the population shape variance that an optimal AFFINE estimator cannot recover from those landmark
coordinates. Sigma = Cov(x_t) is estimated from UNIQUE TRAINING identities only.

It then answers the frozen go/no-go question from PLAN_v3_1_stop_loss.md:
  best-vs-FPS improvement  < 3%  -> abandon active acquisition
                           3-8%  -> optimised static placement only
                           > 8%  and best subset differs across identities -> adaptive worth studying

Low-rank trick: with m (~676) identities and p = 3*n_roi (~11802) coordinates, work in the m-dim
right-singular basis. We only ever index the <=27 landmark coordinate rows, so nothing of size p^2 is
formed. Everything here is seconds-to-minutes.

Usage:
  python scripts/analysis/landmark_subsets.py --in outputs/identifiability_inputs \
      --out outputs/landmark_subset_audit --n-boot 200
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def shape_basis(X: np.ndarray):
    """Return (Wland_full, lam, mu) low-rank pieces.
    X: (m, p) unique-identity ROI vertex rows (flattened xyz). Centered SVD.
    lam: (k,) eigenvalues of Cov; Vt: (k, p) right singular vectors. We return Vt and lam; caller
    slices landmark rows. mu: (p,) mean shape.
    """
    mu = X.mean(axis=0)
    Xc = X - mu
    m = X.shape[0]
    # economy SVD: Xc = U (m x k) diag(s) Vt (k x p), k = m
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    lam = (s ** 2) / (m - 1)                      # eigenvalues of the shape covariance
    return Vt, lam, mu


def subset_residual(Wland: np.ndarray, lam: np.ndarray, total_var: float, rows: np.ndarray) -> float:
    """Affine LMMSE residual for the landmark-coordinate rows in `rows`.
    Wland: (n_land_coords, k) = (Vt[:, land_coord_rows]).T * sqrt(lam) ... but we pass Vt-slice scaled.
    Here Wland already equals  W[land_coord_rows, :]  with  W = Vt.T * sqrt(lam).
    """
    A = Wland[rows, :]                            # (3|subset|, k)
    AAt = A @ A.T                                 # (3|subset|, 3|subset|) small
    Q = (A * lam[None, :]) @ A.T                  # A diag(lam) A'
    explained = float(np.trace(np.linalg.pinv(AAt, rcond=1e-10) @ Q))
    return float(total_var - explained)


def fps_subset(coords: np.ndarray, k: int) -> tuple[int, ...]:
    """Farthest-point sampling of k of the n landmark template positions (deterministic: seed=first)."""
    n = coords.shape[0]
    if k >= n:
        return tuple(range(n))
    chosen = [0]
    d = np.linalg.norm(coords - coords[0], axis=1)
    while len(chosen) < k:
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(coords - coords[i], axis=1))
    return tuple(sorted(chosen))


def coord_rows(subset: tuple[int, ...]) -> np.ndarray:
    return np.array([3 * l + a for l in subset for a in (0, 1, 2)], dtype=np.int64)


def best_per_k(Wland: np.ndarray, lam: np.ndarray, total_var: float, n_land: int):
    """Return dict K -> (best_subset, best_residual, all_residuals_for_K)."""
    out = {}
    for k in range(1, n_land + 1):
        best, best_res, all_res = None, np.inf, []
        for subset in itertools.combinations(range(n_land), k):
            r = subset_residual(Wland, lam, total_var, coord_rows(subset))
            all_res.append((subset, r))
            if r < best_res:
                best, best_res = subset, r
        out[k] = (best, best_res, all_res)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="outputs/identifiability_inputs")
    ap.add_argument("--out", default="outputs/landmark_subset_audit")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260609)
    args = ap.parse_args()

    inp, out = Path(args.inp), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    X = np.load(inp / "train_roi_matrix.npy")                 # (m, p)
    land = np.load(inp / "landmark_indices.npy").astype(np.int64)
    n_land = int(land.size)
    print(f"[audit] X={X.shape}  landmarks={n_land}")

    Vt, lam, mu = shape_basis(X)
    total_var = float(lam.sum())
    land_coord_rows = np.array([3 * l + a for l in land for a in (0, 1, 2)], dtype=np.int64)
    # W[land_coord_rows, :] = (Vt[:, land_coord_rows]).T * sqrt(lam)
    Wland = (Vt[:, land_coord_rows].T) * np.sqrt(lam)[None, :]  # (3*n_land, k)
    # template landmark positions for FPS
    land_xyz = mu[land_coord_rows].reshape(n_land, 3)

    per_k = best_per_k(Wland, lam, total_var, n_land)

    # comparison table: best vs FPS vs random (mean) per K
    rng = np.random.default_rng(args.seed)
    table = []
    for k in range(1, n_land + 1):
        best_subset, best_res, all_res = per_k[k]
        res_arr = np.array([r for _, r in all_res])
        fps = fps_subset(land_xyz, k)
        fps_res = subset_residual(Wland, lam, total_var, coord_rows(fps))
        # random baseline: mean residual over up to 50 random subsets of size k
        combos = [s for s, _ in all_res]
        ridx = rng.choice(len(combos), size=min(50, len(combos)), replace=False)
        rand_res = float(np.mean([res_arr[i] for i in ridx]))
        # express as fraction of total variance recoverable; improvement best vs FPS
        impr_vs_fps = (fps_res - best_res) / max(fps_res, 1e-12)
        table.append({
            "K": k,
            "best_subset": list(best_subset),
            "best_residual": best_res,
            "fps_subset": list(fps),
            "fps_residual": fps_res,
            "random_mean_residual": rand_res,
            "subset_residual_std": float(res_arr.std()),
            "best_vs_fps_rel_improvement": float(impr_vs_fps),
            "best_recoverable_fraction": float(1.0 - best_res / total_var),
        })
        print(f"  K={k}: best={list(best_subset)} res={best_res:.4g} | FPS res={fps_res:.4g} "
              f"| best-vs-FPS={impr_vs_fps*100:+.1f}% | recover={100*(1-best_res/total_var):.1f}%")

    # identity-bootstrap: stability of the best subset per K
    m = X.shape[0]
    boot_best = {k: {} for k in range(1, n_land + 1)}
    for b in range(args.n_boot):
        idx = rng.integers(0, m, size=m)
        Vt_b, lam_b, mu_b = shape_basis(X[idx])
        tv_b = float(lam_b.sum())
        Wl_b = (Vt_b[:, land_coord_rows].T) * np.sqrt(lam_b)[None, :]
        for k in range(1, n_land + 1):
            best, bres = None, np.inf
            for subset in itertools.combinations(range(n_land), k):
                r = subset_residual(Wl_b, lam_b, tv_b, coord_rows(subset))
                if r < bres:
                    best, bres = subset, r
            boot_best[k][best] = boot_best[k].get(best, 0) + 1
    stability = {}
    for k in range(1, n_land + 1):
        counts = boot_best[k]
        top = max(counts.items(), key=lambda kv: kv[1])
        stability[k] = {
            "top_subset": list(top[0]),
            "top_fraction": top[1] / max(1, args.n_boot),
            "n_distinct_best_subsets": len(counts),
        }
        print(f"  [boot] K={k}: top subset {list(top[0])} stable in "
              f"{100*top[1]/max(1,args.n_boot):.0f}% of resamples, "
              f"{len(counts)} distinct winners")

    # ---- frozen go/no-go verdict (PLAN_v3_1) ----
    # use a small, practically-relevant budget (K=2..4) as the decision regime
    decide_ks = [k for k in (2, 3, 4) if k <= n_land]
    impr = np.mean([next(t for t in table if t["K"] == k)["best_vs_fps_rel_improvement"] for k in decide_ks])
    differs = np.mean([stability[k]["top_fraction"] < 0.5 for k in decide_ks]) > 0.5
    if impr < 0.03:
        verdict = "ABANDON active acquisition (best vs FPS < 3%): the 9 points have little selection room."
    elif impr <= 0.08:
        verdict = ("OPTIMISED STATIC PLACEMENT only (3-8%): report a fixed good subset; "
                   "an adaptive policy is NOT justified.")
    else:
        if differs:
            verdict = ("ADAPTIVE WORTH STUDYING (>8% AND best subset differs across identities): "
                       "train the arbitrary-subset model and study true adaptive selection.")
        else:
            verdict = ("OPTIMISED STATIC PLACEMENT (>8% but best subset is stable across identities): "
                       "a single fixed subset captures the gain; adaptive policy likely unnecessary.")
    print("\n================ GO/NO-GO VERDICT ================")
    print(f"decision regime K={decide_ks}  mean best-vs-FPS = {impr*100:.1f}%  "
          f"best-subset-unstable = {differs}")
    print(verdict)
    print("==================================================")

    json.dump({
        "total_variance": total_var,
        "tables": table,
        "stability": stability,
        "decision": {"regime_K": decide_ks, "mean_best_vs_fps": float(impr),
                     "best_subset_differs_across_identities": bool(differs), "verdict": verdict},
    }, open(out / "landmark_subset_audit.json", "w"), indent=2)
    print(f"[audit] wrote -> {out/'landmark_subset_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
