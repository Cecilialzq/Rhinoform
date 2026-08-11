"""Gate 2A - signed foldover detector: synthetic battery + real-pair comparison.

Pass criterion (protocol): the synthetic battery must be 100% correct for all four
classes (rigid / uniform scaling / reflection / degenerate). Degeneracy uses the
RELATIVE threshold det <= eps * src_area2 (not an absolute eps).
"""
from __future__ import annotations

import json

import numpy as np

import scripts.analysis.flip_validation.common as common
from rhinoform.safe_fusion import signed_fold_indicator, triangle_distortion

EPS = 1e-12
SEED = common.SEED


def make_pairs(ids: list[str], n: int, seed: int) -> list[tuple[str, str]]:
    rng = np.random.default_rng(seed)
    pairs: list[tuple[str, str]] = []
    seen = set()
    while len(pairs) < n:
        a, b = rng.choice(len(ids), size=2, replace=False)
        key = (ids[a], ids[b])
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def rotation_matrix(axis: np.ndarray, deg: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    a = np.deg2rad(deg)
    k = np.asarray([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)


def reflect_point_across_line(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    t = np.dot(p - a, d) / max(np.dot(d, d), 1e-18)
    foot = a + t * d
    return 2.0 * foot - p


def synthetic_battery(source: np.ndarray, faces: np.ndarray, n_local: int = 30) -> dict:
    rng = np.random.default_rng(SEED)
    results: dict[str, dict] = {}

    # --- rigid: bounded rotation (editing regime) + translation -> all det>0 ---
    rigid_ok = True
    for s in range(8):
        rsub = np.random.default_rng(SEED + s)
        rot = rotation_matrix(rsub.standard_normal(3), rsub.uniform(5, 45))
        edited = source @ rot.T + rsub.standard_normal(3) * 10.0
        det, _ = signed_fold_indicator(source, edited, faces)
        rigid_ok = rigid_ok and bool(np.all(det > 0))
    results["rigid"] = {"all_positive": rigid_ok, "correct": rigid_ok, "n_trials": 8}

    # --- uniform scaling -> all det>0 ---
    scale_ok = True
    for sc in (0.25, 0.5, 1.5, 3.0):
        det, area2 = signed_fold_indicator(source, source * sc, faces)
        scale_ok = scale_ok and bool(np.all(det > 0))
    results["uniform_scaling"] = {"all_positive": scale_ok, "correct": scale_ok,
                                  "scales": [0.25, 0.5, 1.5, 3.0]}

    # choose >=n_local target faces spread across the mesh
    targets = np.linspace(0, len(faces) - 1, n_local).astype(int)
    targets = np.unique(targets)

    # --- reflection/fold (apex reflected across opposite edge) -> det<0, area preserved ---
    refl_detected = 0
    area_ratio_blind = 0
    for f in targets:
        i, j, k = (int(x) for x in faces[f])
        edited = source.copy()
        edited[k] = reflect_point_across_line(source[k], source[i], source[j])
        det, area2 = signed_fold_indicator(source, edited, faces)
        if det[f] < 0:
            refl_detected += 1
        ar = triangle_distortion(source, edited, faces).area_ratio[f]
        if abs(ar - 1.0) < 1e-6:  # area_ratio is blind to this true fold
            area_ratio_blind += 1
    results["reflection"] = {
        "n_target_faces": int(len(targets)),
        "n_detected_signed": int(refl_detected),
        "detection_rate": float(refl_detected / len(targets)),
        "n_area_ratio_blind": int(area_ratio_blind),
        "correct": refl_detected == len(targets),
    }

    # --- degenerate (apex collapsed to midpoint of opposite edge) -> det<=eps*area2 ---
    degen_detected = 0
    for f in targets:
        i, j, k = (int(x) for x in faces[f])
        edited = source.copy()
        edited[k] = 0.5 * (source[i] + source[j])
        det, area2 = signed_fold_indicator(source, edited, faces)
        if det[f] <= EPS * area2[f]:
            degen_detected += 1
    results["degenerate"] = {
        "n_target_faces": int(len(targets)),
        "n_detected_relative_eps": int(degen_detected),
        "detection_rate": float(degen_detected / len(targets)),
        "threshold": "det <= eps*src_area2",
        "correct": degen_detected == len(targets),
    }

    results["all_classes_100pct_correct"] = all(
        results[c]["correct"] for c in ("rigid", "uniform_scaling", "reflection", "degenerate")
    )
    return results


def real_pair_comparison(by_id: dict, faces: np.ndarray, pairs: list[tuple[str, str]]) -> dict:
    legacy = []        # mean(normal_cosine<0)*100  (legacy proxy)
    signed = []        # mean(det<=eps*src_area2)*100 (signed foldover)
    disagree = []      # faces where the two criteria disagree (%)
    near_degen = []    # faces with near-zero edited area (%)
    per_pair = []
    for s, t in pairs:
        src = by_id[s]["vertices"]
        tgt = by_id[t]["vertices"]
        dist = triangle_distortion(src, tgt, faces)
        det, area2 = signed_fold_indicator(src, tgt, faces)
        legacy_mask = dist.normal_cosine < 0.0
        signed_mask = det <= EPS * area2
        lg = float(np.mean(legacy_mask) * 100.0)
        sg = float(np.mean(signed_mask) * 100.0)
        dg = float(np.mean(legacy_mask != signed_mask) * 100.0)
        # near-degenerate edited faces (relative to source area)
        nd = float(np.mean(np.abs(det) <= 1e-6 * area2) * 100.0)
        legacy.append(lg)
        signed.append(sg)
        disagree.append(dg)
        near_degen.append(nd)
        per_pair.append({"source": s, "target": t, "legacy_normal_flip_pct": lg,
                         "signed_foldover_pct": sg, "disagreement_pct": dg})
    legacy = np.asarray(legacy)
    signed = np.asarray(signed)
    disagree = np.asarray(disagree)
    near_degen = np.asarray(near_degen)

    def stats(a):
        return {"mean": float(a.mean()), "p95": float(np.percentile(a, 95)), "max": float(a.max())}

    return {
        "n_pairs": len(pairs),
        "legacy_normal_flip_pct": stats(legacy),
        "signed_foldover_pct": stats(signed),
        "criteria_disagreement_pct": stats(disagree),
        "near_degenerate_face_pct": stats(near_degen),
        "sign_equivalent_on_real_data": bool(np.allclose(legacy, signed)),
        "relationship_note": (
            "det = normal_cosine * |edited_area2|, so sign(det) == sign(normal_cosine): "
            "the signed indicator equals the legacy normal proxy in SIGN on non-degenerate real "
            "faces. Its added value is (1) a hard, properly-wired signed certificate and (2) a "
            "size-relative degeneracy threshold det<=eps*src_area2 that the unsigned area_ratio / "
            "sigma criteria are blind to (validated 100% on the synthetic battery)."
        ),
        "per_pair": per_pair,
    }


def main() -> int:
    out = common.ensure_out()
    splits = common.load_split_ids()
    # one real neutral mesh as battery base + 30 real test pairs
    pair_ids = make_pairs(splits["test"], 30, SEED)
    needed = sorted({pair_ids[0][0]} | {i for p in pair_ids for i in p}, key=int)
    by_id = common.load_meshes(needed)
    top = common.template_topology(by_id)
    faces = top["faces"]
    base_mesh = by_id[needed[0]]["vertices"]

    battery = synthetic_battery(base_mesh, faces)
    real = real_pair_comparison(by_id, faces, pair_ids)

    go = battery["all_classes_100pct_correct"]
    result = {
        "gate": "2A_signed_foldover",
        "battery_base_identity": needed[0],
        "synthetic_battery": battery,
        "real_pair_comparison": real,
        "decision": "GO" if go else "NO-GO",
        "decision_reason": (
            "Synthetic battery 100% correct on all four classes; signed indicator is a valid instrument."
            if go else
            "Synthetic battery NOT 100% correct -> detector has a bug; CLAIM SHOULD BE FIXED before use."
        ),
    }
    (out / "g2a_signed_fold.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "real_pair_comparison"}, indent=2))
    print("real legacy vs signed:", real["legacy_normal_flip_pct"], real["signed_foldover_pct"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
