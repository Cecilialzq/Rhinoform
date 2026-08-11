"""Exact-hybrid failure-case audit.

Unlike ``failure_quality_audit.py`` (which could only reconstruct the
ridge/source-PCA anchor and explicitly flagged that as a gap), this script
audits the *exact* validation-selected hybrid by consuming the saved per-pair
hybrid predictions (``neural_field_predictions_*.npz``). It produces:

  * a per-case CSV over all held-out pairs (roi/dorsum/tip/alar RMSE, edge
    strain p95, normal-flip %, flipped-face counts, subunit flip touches,
    ROI-boundary flip touches, deformation magnitude, dominant subunit);
  * the worst-N cases for figure selection;
  * quantitative failure-mode distributions (P2.2): flips by subunit, failure
    rate vs deformation-magnitude bins, ROI-boundary artifact share, hardest
    target identities; and
  * a hybrid-vs-anchor comparison on the shared worst-case set, so the audit
    shows whether the neural residual *adds* or *removes* artifacts.

All predictions are vertex deltas in ROI-local topology, aligned 1:1 with
``ordered_pairs(test_ids, None, seed)``; alignment is asserted at load time.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rhinoform.data import (
    edge_index,
    face_normals,
    load_rows,
    ordered_pairs,
    split_ids,
)


def boundary_vertices(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    """Open-boundary vertices of an ROI patch: endpoints of edges used by a
    single triangle."""
    from collections import Counter

    edge_count: Counter = Counter()
    for a, b, c in faces.astype(np.int64):
        for u, v in ((a, b), (b, c), (c, a)):
            u, v = int(u), int(v)
            if u > v:
                u, v = v, u
            edge_count[(u, v)] += 1
    bset: set[int] = set()
    for (u, v), cnt in edge_count.items():
        if cnt == 1:
            bset.add(u)
            bset.add(v)
    mask = np.zeros(n_vertices, dtype=bool)
    if bset:
        mask[np.fromiter(bset, dtype=np.int64)] = True
    return mask


def rmse(pred_delta: np.ndarray, true_delta: np.ndarray, idx: np.ndarray | None = None) -> float:
    pd = pred_delta.reshape(-1, 3)
    td = true_delta.reshape(-1, 3)
    if idx is not None:
        pd = pd[idx]
        td = td[idx]
    return float(np.sqrt(np.mean(np.sum((pd - td) ** 2, axis=1))))


def faces_touching(flipped_faces: np.ndarray, idx: np.ndarray) -> int:
    if len(flipped_faces) == 0:
        return 0
    s = set(int(v) for v in idx)
    return int(sum(any(int(v) in s for v in tri) for tri in flipped_faces))


def audit_method(
    name: str,
    by_id: dict,
    pairs: list[tuple[str, str]],
    pred: np.ndarray,
    faces: np.ndarray,
    edges: np.ndarray,
    subunits: dict[str, np.ndarray],
    boundary_idx: np.ndarray,
    landmarks: np.ndarray,
) -> list[dict]:
    rows = []
    n0_cache: dict[str, np.ndarray] = {}
    for i, (src_id, tgt_id) in enumerate(pairs):
        src = by_id[src_id]["vertices"]
        tgt = by_id[tgt_id]["vertices"]
        true_delta = tgt - src
        pred_delta = pred[i].reshape(-1, 3)
        pred_v = src + pred_delta

        if src_id not in n0_cache:
            n0_cache[src_id] = face_normals(src, faces)
        n0 = n0_cache[src_id]
        n1 = face_normals(pred_v, faces)
        flip_mask = np.sum(n0 * n1, axis=1) < 0.0
        flipped = faces[flip_mask]

        e0 = np.linalg.norm(src[edges[:, 0]] - src[edges[:, 1]], axis=1)
        e1 = np.linalg.norm(pred_v[edges[:, 0]] - pred_v[edges[:, 1]], axis=1)
        strain = np.abs(e1 - e0) / np.maximum(e0, 1e-12)

        # dominant subunit = subunit with largest mean target displacement
        sub_mag = {k: float(np.mean(np.linalg.norm(true_delta[v], axis=1))) for k, v in subunits.items()}
        dominant = max(sub_mag, key=sub_mag.get)

        row = {
            "method": name,
            "source_id": src_id,
            "target_id": tgt_id,
            "roi_rmse": rmse(pred_delta, true_delta),
            "dorsum_rmse": rmse(pred_delta, true_delta, subunits["dorsum"]),
            "tip_rmse": rmse(pred_delta, true_delta, subunits["tip"]),
            "alar_left_rmse": rmse(pred_delta, true_delta, subunits["alar_left"]),
            "alar_right_rmse": rmse(pred_delta, true_delta, subunits["alar_right"]),
            "edge_p95": float(np.percentile(strain, 95)),
            "normal_flip_pct": float(np.mean(flip_mask) * 100.0),
            "n_flipped_faces": int(flip_mask.sum()),
            "flip_touch_boundary": faces_touching(flipped, np.where(boundary_idx)[0]),
            "deform_magnitude": float(np.mean(np.linalg.norm(true_delta, axis=1))),
            "landmark_drive": float(np.mean(np.linalg.norm(true_delta[landmarks], axis=1))),
            "dominant_subunit": dominant,
        }
        for k, v in subunits.items():
            row[f"flip_touch_{k}"] = faces_touching(flipped, v)
        rows.append(row)
    return rows


def distributions(rows: list[dict], subunits: dict[str, np.ndarray]) -> dict:
    arr = lambda k: np.asarray([r[k] for r in rows], dtype=np.float64)
    flip = arr("normal_flip_pct")
    rmse_a = arr("roi_rmse")
    mag = arr("deform_magnitude")

    # failure-rate vs deformation-magnitude terciles
    q1, q2 = np.quantile(mag, [1 / 3, 2 / 3])
    bins = {"low": mag <= q1, "mid": (mag > q1) & (mag <= q2), "high": mag > q2}
    mag_bins = {
        b: {
            "n": int(m.sum()),
            "mean_roi_rmse": float(rmse_a[m].mean()),
            "mean_normal_flip_pct": float(flip[m].mean()),
            "p95_normal_flip_pct": float(np.percentile(flip[m], 95)),
        }
        for b, m in bins.items()
    }

    # flips by subunit (share of all flipped-face touches)
    sub_tot = {k: int(sum(r[f"flip_touch_{k}"] for r in rows)) for k in subunits}
    grand = sum(sub_tot.values()) or 1
    sub_share = {k: round(v / grand, 4) for k, v in sub_tot.items()}

    # ROI-boundary artifact share
    bnd = int(sum(r["flip_touch_boundary"] for r in rows))
    total_flip = int(sum(r["n_flipped_faces"] for r in rows)) or 1
    boundary_share = round(bnd / total_flip, 4)

    # failure by dominant subunit
    dom_stats: dict[str, dict] = {}
    for k in subunits:
        sel = [r for r in rows if r["dominant_subunit"] == k]
        if sel:
            dom_stats[k] = {
                "n": len(sel),
                "mean_normal_flip_pct": round(float(np.mean([r["normal_flip_pct"] for r in sel])), 4),
                "mean_roi_rmse": round(float(np.mean([r["roi_rmse"] for r in sel])), 4),
            }

    # hardest target identities (by mean normal_flip_pct)
    by_tgt: dict[str, list] = {}
    for r in rows:
        by_tgt.setdefault(r["target_id"], []).append(r["normal_flip_pct"])
    hardest = sorted(
        ({"target_id": t, "mean_normal_flip_pct": round(float(np.mean(v)), 4), "n": len(v)} for t, v in by_tgt.items()),
        key=lambda d: d["mean_normal_flip_pct"],
        reverse=True,
    )[:8]

    return {
        "failure_rate_by_deform_magnitude": mag_bins,
        "flip_share_by_subunit": sub_share,
        "roi_boundary_flip_share_of_all_flips": boundary_share,
        "failure_by_dominant_subunit": dom_stats,
        "hardest_target_identities": hardest,
        "corr_flip_vs_deform_magnitude": round(float(np.corrcoef(flip, mag)[0, 1]), 4),
        "corr_rmse_vs_deform_magnitude": round(float(np.corrcoef(rmse_a, mag)[0, 1]), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="../data")
    parser.add_argument(
        "--pred-npz",
        default="../results/neural_field_predictions_cvae_ew0p1_lw0.npz",
    )
    parser.add_argument("--out", default="../results")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--worst", type=int, default=12)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, by_id = load_rows(Path(args.repo))
    test_ids = split_ids(by_id, "main test")
    pairs = ordered_pairs(test_ids, None, args.seed)
    template = next(iter(by_id.values()))
    faces = template["faces"]
    edges = edge_index(faces)
    subunits = template["subunits"]
    landmarks = template["landmarks"]
    boundary_idx = boundary_vertices(faces, template["vertices"].shape[0])

    pf = np.load(args.pred_npz, allow_pickle=True)
    npz_pairs = [(str(a), str(b)) for a, b in pf["test_pairs"]]
    assert npz_pairs == [(str(a), str(b)) for a, b in pairs], "pred npz pair order mismatch"
    hybrid = pf["hybrid_selected_pred"]
    ridge = pf["ridge_cond_pred"]

    hybrid_rows = audit_method(
        "hybrid", by_id, pairs, hybrid, faces, edges, subunits, boundary_idx, landmarks
    )
    ridge_rows = audit_method(
        "ridge_anchor", by_id, pairs, ridge, faces, edges, subunits, boundary_idx, landmarks
    )

    worst = sorted(
        hybrid_rows, key=lambda r: (r["normal_flip_pct"], r["edge_p95"], r["roi_rmse"]), reverse=True
    )[: args.worst]

    # hybrid-vs-anchor on the shared worst-case set
    ridge_by_pair = {(r["source_id"], r["target_id"]): r for r in ridge_rows}
    paired = []
    for hr in worst:
        rr = ridge_by_pair[(hr["source_id"], hr["target_id"])]
        paired.append(
            {
                "source_id": hr["source_id"],
                "target_id": hr["target_id"],
                "hybrid_normal_flip_pct": round(hr["normal_flip_pct"], 4),
                "anchor_normal_flip_pct": round(rr["normal_flip_pct"], 4),
                "hybrid_roi_rmse": round(hr["roi_rmse"], 4),
                "anchor_roi_rmse": round(rr["roi_rmse"], 4),
            }
        )

    fields = list(hybrid_rows[0].keys())
    with (out_dir / "failure_quality_audit_exact_cases.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(hybrid_rows, key=lambda r: r["normal_flip_pct"], reverse=True))

    h_flip = np.asarray([r["normal_flip_pct"] for r in hybrid_rows])
    r_flip = np.asarray([r["normal_flip_pct"] for r in ridge_rows])
    report = {
        "method": "hybrid (validation-selected, exact saved predictions)",
        "n_pairs": len(pairs),
        "aggregate": {
            "mean_roi_rmse": round(float(np.mean([r["roi_rmse"] for r in hybrid_rows])), 4),
            "mean_normal_flip_pct": round(float(h_flip.mean()), 4),
            "p95_normal_flip_pct": round(float(np.percentile(h_flip, 95)), 4),
            "max_normal_flip_pct": round(float(h_flip.max()), 4),
            "frac_pairs_flip_gt_3pct": round(float(np.mean(h_flip > 3.0)), 4),
        },
        "anchor_vs_hybrid_mean_flip": {
            "anchor_mean_normal_flip_pct": round(float(r_flip.mean()), 4),
            "hybrid_mean_normal_flip_pct": round(float(h_flip.mean()), 4),
            "delta_hybrid_minus_anchor": round(float(h_flip.mean() - r_flip.mean()), 4),
        },
        "distributions": distributions(hybrid_rows, subunits),
        "worst_cases": worst,
        "worst_cases_hybrid_vs_anchor": paired,
    }
    (out_dir / "failure_quality_audit_exact_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in ("aggregate", "anchor_vs_hybrid_mean_flip", "distributions")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
