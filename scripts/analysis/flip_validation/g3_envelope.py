"""Gate 3 - population plausibility envelope (region-level joint nonconformity).

Calibrate a SINGLE joint nonconformity score tau on val_ids real deformations,
then verify on the DISJOINT test_ids that (1) real humans are accepted >=90% and
(2) near-distribution bad edits (hard negatives) are rejected at a clearly
non-zero rate. Calibration uses ONLY val; acceptance is measured ONLY on test.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import scripts.analysis.flip_validation.common as common

SEED = common.SEED
REGIONS = common.SUBUNITS


def region_topology(top: dict) -> dict:
    """Assign faces/edges to a region by majority/both-endpoint vertex membership."""
    faces = top["faces"]
    subs = top["subunits"]
    # edges
    from collections import defaultdict
    edge_set = set()
    for tri in faces:
        for u, v in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_set.add((int(min(u, v)), int(max(u, v))))
    edges = np.asarray(sorted(edge_set), dtype=np.int64)
    region = {}
    for r in REGIONS:
        vset = set(int(x) for x in subs[r])
        vmask = np.zeros(top["n_vertices"], dtype=bool)
        vmask[subs[r]] = True
        face_in = vmask[faces].sum(axis=1) >= 2          # >=2 of 3 vertices in region
        edge_in = vmask[edges[:, 0]] & vmask[edges[:, 1]]
        region[r] = {
            "verts": np.asarray(subs[r], dtype=np.int64),
            "faces": faces[face_in],
            "edges": edges[edge_in],
        }
    return region


def tri_area2(v, f):
    t = v[f]
    return np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)


def region_features(src: np.ndarray, d: np.ndarray, region: dict) -> np.ndarray:
    """Per-region 3-vector: [disp_rms, edge_strain_p95, neg_log_area_ratio_p01]."""
    edited = src + d
    feats = []
    for r in REGIONS:
        info = region[r]
        verts, faces, edges = info["verts"], info["faces"], info["edges"]
        disp_rms = float(np.sqrt(np.mean(np.sum(d[verts] ** 2, axis=1))))
        if len(edges):
            l0 = np.linalg.norm(src[edges[:, 0]] - src[edges[:, 1]], axis=1)
            l1 = np.linalg.norm(edited[edges[:, 0]] - edited[edges[:, 1]], axis=1)
            strain = np.abs(l1 - l0) / np.maximum(l0, 1e-9)
            strain_p95 = float(np.percentile(strain, 95))
        else:
            strain_p95 = 0.0
        if len(faces):
            a0 = tri_area2(src, faces)
            a1 = tri_area2(edited, faces)
            ratio = a1 / np.maximum(a0, 1e-12)
            ar_p01 = float(np.percentile(ratio, 1))
            neg_log_ar = float(-np.log(max(ar_p01, 1e-6)))
        else:
            neg_log_ar = 0.0
        feats.append([disp_rms, strain_p95, neg_log_ar])
    return np.asarray(feats, dtype=np.float64)  # (5,3)


def make_pairs(ids, n, seed):
    rng = np.random.default_rng(seed)
    pairs, seen = [], set()
    if n >= len(ids) * (len(ids) - 1):
        return [(a, b) for a in ids for b in ids if a != b]
    while len(pairs) < n:
        a, b = rng.choice(len(ids), size=2, replace=False)
        key = (ids[a], ids[b])
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def collect_features(by_id, region, pairs):
    feats = []
    for s, t in pairs:
        src = by_id[s]["vertices"]
        d = by_id[t]["vertices"] - src
        feats.append(region_features(src, d, region))
    return np.asarray(feats)  # (N,5,3)


class Envelope:
    """Per-region Mahalanobis; joint score s = max_r Mahalanobis_r."""

    def __init__(self, val_feats: np.ndarray, ridge: float = 1e-6):
        self.mu = val_feats.mean(axis=0)                       # (5,3)
        self.inv = []
        for r in range(val_feats.shape[1]):
            x = val_feats[:, r, :]
            cov = np.cov(x, rowvar=False) + ridge * np.eye(x.shape[1])
            self.inv.append(np.linalg.inv(cov))
        self.inv = np.asarray(self.inv)                        # (5,3,3)

    def score(self, feats: np.ndarray) -> np.ndarray:
        diff = feats - self.mu[None]                           # (N,5,3)
        m = np.einsum("nrd,rde,nre->nr", diff, self.inv, diff) # (N,5) squared Mahalanobis
        return np.sqrt(np.maximum(m, 0.0)).max(axis=1)         # (N,)


def build_hard_negatives(by_id, region, base_pairs, top, kind: str, n: int, seed: int):
    """Return list of (src_vertices, deformation) for near-distribution bad edits."""
    rng = np.random.default_rng(seed)
    subs = top["subunits"]
    out = []
    idx = 0
    while len(out) < n:
        s, t = base_pairs[idx % len(base_pairs)]
        idx += 1
        src = by_id[s]["vertices"]
        d = (by_id[t]["vertices"] - src).copy()
        if kind == "over_narrow_alar":
            # amplify alar displacement 2-3x along its dominant (narrowing) direction
            scale = rng.uniform(2.0, 3.0)
            for r in ("alar_left", "alar_right"):
                v = subs[r]
                mean_dir = d[v].mean(axis=0)
                nrm = np.linalg.norm(mean_dir)
                if nrm > 1e-9:
                    d[v] = d[v] + (scale - 1.0) * mean_dir[None]  # push further in narrowing dir
        elif kind == "isolated_tip_only":
            # only tip moves, all other regions forced to zero (breaks coupling)
            keep = np.zeros(top["n_vertices"], dtype=bool)
            keep[subs["tip"]] = True
            d[~keep] = 0.0
        elif kind == "smooth_broken_coupling":
            # zero out dorsum/alar response, keep tip, then Laplacian-smooth
            kill = np.zeros(top["n_vertices"], dtype=bool)
            for r in ("dorsum", "alar_left", "alar_right"):
                kill[subs[r]] = True
            d[kill] = 0.0
            d = _smooth(d, top["faces"], top["n_vertices"], iters=5)
        else:
            raise ValueError(kind)
        out.append((src, d))
    return out


_NBR_CACHE = {}


def _smooth(d, faces, n, iters=5):
    key = id(faces)
    if key not in _NBR_CACHE:
        nbr = [[] for _ in range(n)]
        for tri in faces:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            nbr[a] += [b, c]; nbr[b] += [a, c]; nbr[c] += [a, b]
        _NBR_CACHE[key] = [np.asarray(sorted(set(x)), dtype=np.int64) for x in nbr]
    nbr = _NBR_CACHE[key]
    out = d.copy()
    for _ in range(iters):
        nxt = out.copy()
        for i in range(n):
            if len(nbr[i]):
                nxt[i] = 0.5 * out[i] + 0.5 * out[nbr[i]].mean(axis=0)
        out = nxt
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-pairs", type=int, default=800)
    ap.add_argument("--test-pairs", type=int, default=800)
    ap.add_argument("--hard-neg", type=int, default=60)
    ap.add_argument("--quantile", type=float, default=95.0)
    args = ap.parse_args()
    out = common.ensure_out()
    splits = common.load_split_ids()

    need = sorted(set(splits["val"]) | set(splits["test"]), key=int)
    by_id = common.load_meshes(need)
    top = common.template_topology(by_id)
    region = region_topology(top)

    val_pairs = make_pairs(splits["val"], args.val_pairs, SEED)
    test_pairs = make_pairs(splits["test"], args.test_pairs, SEED + 1)
    hn_base = make_pairs(splits["test"], max(args.hard_neg, 60), SEED + 2)

    val_feats = collect_features(by_id, region, val_pairs)
    test_feats = collect_features(by_id, region, test_pairs)

    env = Envelope(val_feats)
    s_val = env.score(val_feats)
    tau = float(np.percentile(s_val, args.quantile))

    s_test = env.score(test_feats)
    accept_mask = s_test <= tau
    accept_rate = float(accept_mask.mean())
    accept_mean, accept_lo, accept_hi = common.identity_bootstrap_ci(
        accept_mask.astype(float), test_pairs, n_boot=1000, seed=SEED)

    # hard negatives
    hard = {}
    for kind in ("over_narrow_alar", "isolated_tip_only", "smooth_broken_coupling"):
        samples = build_hard_negatives(by_id, region, hn_base, top, kind, args.hard_neg, SEED + 7)
        feats = np.asarray([region_features(src, d, region) for src, d in samples])
        s_hn = env.score(feats)
        det = float(np.mean(s_hn > tau))
        hard[kind] = {"n": len(samples), "detection_rate": det,
                      "score_mean": float(s_hn.mean()), "score_p50": float(np.median(s_hn))}

    # tau / acceptance bootstrap CI (identity-level on val calibration)
    val_ids = sorted({i for p in val_pairs for i in p}, key=int)
    id_to_pairs = {i: [] for i in val_ids}
    for k, (a, b) in enumerate(val_pairs):
        id_to_pairs[a].append(k); id_to_pairs[b].append(k)
    rng = np.random.default_rng(SEED)
    taus = []
    for _ in range(1000):
        chosen = rng.integers(0, len(val_ids), len(val_ids))
        sel = []
        for c in chosen:
            sel.extend(id_to_pairs[val_ids[c]])
        if sel:
            taus.append(float(np.percentile(s_val[sel], args.quantile)))
    tau_lo, tau_hi = float(np.percentile(taus, 2.5)), float(np.percentile(taus, 97.5))

    accept_go = accept_lo >= 0.90 or accept_rate >= 0.90
    hn_go = all(h["detection_rate"] > 0.2 for h in hard.values()) and \
        np.mean([h["detection_rate"] for h in hard.values()]) > 0.5
    go = accept_go and hn_go

    result = {
        "gate": "3_population_envelope",
        "design": {
            "calibration_set": "val_ids real deformations (DISJOINT from test/train)",
            "evaluation_set": "test_ids real deformations (DISJOINT from val/train)",
            "score": "s = max_region Mahalanobis(region 3-feature vector); tau at val %.1f pct" % args.quantile,
            "features_per_region": ["disp_rms", "edge_strain_p95", "neg_log_area_ratio_p01"],
            "n_val_pairs": len(val_pairs), "n_test_pairs": len(test_pairs),
        },
        "tau": tau,
        "tau_bootstrap_ci95": [tau_lo, tau_hi],
        "heldout_human_acceptance_rate": accept_rate,
        "heldout_human_acceptance_ci95": [accept_lo, accept_hi],
        "hard_negative_detection": hard,
        "decision": "GO" if go else "NO-GO",
        "decision_reason": _reason(go, accept_rate, accept_go, hard, hn_go),
    }
    (out / "g3_envelope.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "design"}, indent=2))
    return 0


def _reason(go, acc, acc_go, hard, hn_go):
    if go:
        return ("Held-out human acceptance >=90% on DISJOINT test_ids and all three hard-negative classes "
                "detected at clearly non-zero rates -> envelope is a valid instrument; proceed to selective eval.")
    msgs = []
    if not acc_go:
        msgs.append(f"Held-out human acceptance {acc*100:.1f}% < 90% -> envelope too tight (repeats absolute-gate failure); "
                    "CLAIM 'population plausibility envelope' must be reset (features/quantile).")
    if not hn_go:
        weak = [k for k, h in hard.items() if h["detection_rate"] <= 0.2]
        msgs.append(f"Hard-negative detection near zero for {weak} -> envelope only catches extreme anomalies; "
                    "selective-system value NOT established; features must be reset.")
    return " ".join(msgs)


if __name__ == "__main__":
    raise SystemExit(main())
