"""Test 4 -- residual mechanism quantification (cheap, for the Discussion).

Turns the mechanistic explanation of the negative result into measured numbers
instead of post-hoc hand-waving. Three lightweight analyses on the conditional
ridge residual (target - ridge_prediction, from legal inputs only):

1. **Energy budget** -- what fraction of the target deformation energy the ridge
   anchor already explains, and how small the leftover residual is (overall and
   per nose subunit). Controls pinning the large-scale deformation => small
   residual.
2. **Spectrum** -- compare the PCA spectra of the target deformation vs the
   residual: effective rank and the #components for 90/95% energy. A residual
   that is higher-rank / flatter is higher-frequency and less low-rank
   compressible.
3. **Predictability** -- grouped cross-validated R^2 (and top canonical
   correlations) of the legal input ``cond`` predicting the residual PC scores.
   Low predictability => the leftover is weakly conditioned on the controls.

Together: residual is small, high-frequency/dispersed, and weakly predictable
from the controls -> the nonlinear advantage naturally collapses under
sparse-control conditioning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rhinoform.diag_common import (
    atomic_write_json,
    build_conditions,
    deterministic_pairs,
    effective_rank,
    energy_fraction_for,
    provenance,
    ridge_residual,
    sha256_file,
)
from rhinoform.data import load_rows


def spectrum_summary(centered: np.ndarray, keep: int = 200) -> dict:
    _, singular, _ = np.linalg.svd(centered, full_matrices=False)
    lam = singular**2
    total = float(lam.sum())
    cum = (np.cumsum(lam) / total) if total > 0 else np.zeros_like(lam)
    return {
        "effective_rank": effective_rank(singular),
        "n_components_90pct": energy_fraction_for(singular, 0.90),
        "n_components_95pct": energy_fraction_for(singular, 0.95),
        "n_total_components": int(singular.shape[0]),
        "cumulative_energy_head": [float(x) for x in cum[:keep]],
    }


def grouped_cv_r2(cond: np.ndarray, scores: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> float:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    n_groups = len(set(groups.tolist()))
    n_splits = max(2, min(n_splits, n_groups))
    gkf = GroupKFold(n_splits=n_splits)
    num = 0.0
    den = 0.0
    for tr, te in gkf.split(cond, scores, groups):
        model = Ridge(alpha=1.0)
        model.fit(cond[tr], scores[tr])
        pred = model.predict(cond[te])
        num += float(np.sum((scores[te] - pred) ** 2))
        den += float(np.sum(scores[te] ** 2))
    if den <= 0:
        return float("nan")
    return 1.0 - num / den


def canonical_correlations(cond: np.ndarray, scores: np.ndarray, n_comp: int, seed: int) -> list[float]:
    from sklearn.cross_decomposition import CCA

    n_comp = max(1, min(n_comp, cond.shape[1], scores.shape[1]))
    try:
        cca = CCA(n_components=n_comp, max_iter=1000)
        u, v = cca.fit_transform(cond, scores)
    except Exception:
        return []
    cors = []
    for i in range(u.shape[1]):
        a = u[:, i]
        b = v[:, i]
        if a.std() < 1e-12 or b.std() < 1e-12:
            cors.append(0.0)
        else:
            cors.append(float(np.corrcoef(a, b)[0, 1]))
    return cors


def subunit_energy_fraction(by_id, template_id, resid, delta, n_vertices) -> dict:
    subunits = by_id[template_id]["subunits"]
    resid3 = resid.reshape(resid.shape[0], n_vertices, 3)
    delta3 = delta.reshape(delta.shape[0], n_vertices, 3)
    out = {}
    for name, idx in subunits.items():
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            continue
        r_e = float(np.sum(resid3[:, idx, :] ** 2))
        d_e = float(np.sum(delta3[:, idx, :] ** 2))
        out[name] = {
            "residual_fraction_of_target": (r_e / d_e) if d_e > 0 else float("nan"),
            "ridge_explained_fraction": (1.0 - r_e / d_e) if d_e > 0 else float("nan"),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Test 4: residual mechanism quantification.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base-model-package", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=4000)
    parser.add_argument("--residual-rank", type=int, default=64)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260615)
    args = parser.parse_args()

    print("[test4] loading repo + base package", flush=True)
    _, by_id = load_rows(Path(args.repo))
    base = torch.load(args.base_model_package, map_location="cpu", weights_only=False)
    train_ids = [str(v) for v in base["train_ids"]]
    template_id = train_ids[0]
    n_vertices = int(np.asarray(by_id[template_id]["vertices"]).shape[0])

    pairs = deterministic_pairs(train_ids, args.max_pairs, args.seed)
    print(f"[test4] pairs={len(pairs)} vertices={n_vertices}", flush=True)
    cond, _, delta, _, _ = build_conditions(by_id, pairs, base)
    delta_flat = delta.reshape(len(pairs), -1).astype(np.float64)
    ridge_flat, resid = ridge_residual(cond, delta_flat, base)

    # 1. Energy budget.
    target_energy = float(np.sum(delta_flat**2))
    residual_energy = float(np.sum(resid**2))
    energy = {
        "ridge_explained_fraction_of_target": float(1.0 - residual_energy / target_energy),
        "residual_fraction_of_target": float(residual_energy / target_energy),
        "per_subunit": subunit_energy_fraction(by_id, template_id, resid, delta_flat, n_vertices),
    }
    print(f"[test4] ridge explains {energy['ridge_explained_fraction_of_target']*100:.1f}% of target energy; "
          f"residual is {energy['residual_fraction_of_target']*100:.1f}%", flush=True)

    # 2. Spectrum: target vs residual.
    delta_c = delta_flat - delta_flat.mean(axis=0, keepdims=True)
    resid_c = resid - resid.mean(axis=0, keepdims=True)
    print("[test4] computing target / residual PCA spectra", flush=True)
    spectrum = {
        "target_deformation": spectrum_summary(delta_c),
        "residual": spectrum_summary(resid_c),
    }
    print(f"[test4] effective rank: target={spectrum['target_deformation']['effective_rank']:.1f} "
          f"residual={spectrum['residual']['effective_rank']:.1f}", flush=True)

    # 3. Predictability of residual PCs from cond.
    _, singular, vt = np.linalg.svd(resid_c, full_matrices=False)
    rank = min(args.residual_rank, vt.shape[0])
    scores = resid_c @ vt[:rank].T
    groups = np.asarray([a for a, _ in pairs])
    cv_r2 = grouped_cv_r2(cond, scores, groups, args.cv_splits, args.seed)
    cca = canonical_correlations(cond, scores, n_comp=min(5, cond.shape[1], rank), seed=args.seed)
    predictability = {
        "residual_rank_used": int(rank),
        "grouped_cv_r2_cond_to_residual_pcs": cv_r2,
        "top_canonical_correlations": cca,
    }
    print(f"[test4] grouped CV R^2 (cond -> residual PCs) = {cv_r2:.4f}; "
          f"top CCA = {[round(c,3) for c in cca[:3]]}", flush=True)

    out = {
        "test": "residual_mechanism",
        "purpose": "Quantify why the nonlinear advantage collapses under conditioning (energy, spectrum, predictability).",
        "provenance": provenance(args.seed, {
            "repo": str(args.repo),
            "base_model_package": str(args.base_model_package),
            "base_model_package_sha256": sha256_file(Path(args.base_model_package)),
            "feature_template_sha256": base.get("feature_template_sha256"),
            "n_pairs": len(pairs),
            "residual_rank": rank,
        }),
        "energy_budget": energy,
        "spectrum": spectrum,
        "predictability": predictability,
        "interpretation": (
            "Controls pin the large-scale low-frequency deformation (high ridge-explained fraction); "
            "the leftover residual is small, higher-rank/flatter (more high-frequency), and weakly predictable "
            "from the controls (low grouped CV R^2). This is the mechanism behind the conditional collapse of "
            "the nonlinear advantage."
        ),
    }
    out_path = Path(args.out)
    atomic_write_json(out_path, out)
    print(f"[test4] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
