"""Gate 2B - self-intersection detector: synthetic pos/neg battery + real diagnosis.

Diagnoses the prior "100% self-intersection" finding by separating the source's
OWN pre-existing intersections (n_src, e.g. at the ROI cut boundary) from the
NEW intersections introduced by the edit (n_new). Also explicitly tests
adjacency exclusion (shared edge / shared vertex) and ROI-boundary false
positives.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import scripts.analysis.flip_validation.common as common
from rhinoform.baselines import arap_predict_vectorised
from rhinoform.geometry import graph_edges, solve_linear_handle_baseline, uniform_laplacian
from rhinoform.safe_fusion import count_self_intersections

SEED = common.SEED


def synthetic_battery() -> dict:
    res: dict[str, dict] = {}

    # positive: two non-adjacent crossing triangles (3 independent groups) -> count == 3
    base = np.asarray([[-1, -1, 0], [1, -1, 0], [0, 1, 0], [0, -0.5, -1], [0, -0.5, 1], [0, 0.8, 0]], float)
    verts = []
    faces = []
    for g in range(3):
        off = np.asarray([5.0 * g, 0.0, 0.0])
        verts.append(base + off)
        faces.append([[6 * g, 6 * g + 1, 6 * g + 2], [6 * g + 3, 6 * g + 4, 6 * g + 5]])
    verts = np.concatenate(verts, axis=0)
    faces = np.asarray(sum(faces, []), dtype=np.int64)
    pos = count_self_intersections(verts, faces)
    res["positive_three_crossings"] = {"count": int(pos), "expected": 3, "correct": pos == 3}

    # negative: separated triangles -> 0
    sep = verts.copy()
    sep[3::6, 0] += 50.0
    sep[4::6, 0] += 50.0
    sep[5::6, 0] += 50.0
    neg = count_self_intersections(sep, faces)
    res["negative_separated"] = {"count": int(neg), "expected": 0, "correct": neg == 0}

    # adjacency exclusion: two triangles sharing an EDGE (coplanar, touching) -> 0
    shared_edge_v = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    shared_edge_f = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)  # share edge (1,2)
    se = count_self_intersections(shared_edge_v, shared_edge_f)
    res["adjacency_shared_edge"] = {"count": int(se), "expected": 0, "correct": se == 0}

    # adjacency exclusion: two triangles sharing a single VERTEX -> 0
    shared_vtx_v = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], float)
    shared_vtx_f = np.asarray([[0, 1, 2], [0, 3, 4]], dtype=np.int64)  # share vertex 0
    sv = count_self_intersections(shared_vtx_v, shared_vtx_f)
    res["adjacency_shared_vertex"] = {"count": int(sv), "expected": 0, "correct": sv == 0}

    # pre-existing exclusion: source already intersects, edited still intersects, no NEW -> 0
    src = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.2, -1], [0.2, 0.2, 1], [1, 1, 0]], float)
    edited = src.copy()
    edited[3:] += np.asarray([0.01, 0.01, 0.0])  # still intersecting, no new pair
    faces2 = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    n_src_pre = count_self_intersections(src, faces2)
    n_new_pre = count_self_intersections(edited, faces2, baseline_vertices=src)
    res["preexisting_excluded"] = {
        "n_src": int(n_src_pre), "n_new": int(n_new_pre),
        "expected_n_new": 0, "correct": n_src_pre == 1 and n_new_pre == 0,
    }

    res["all_correct"] = all(v["correct"] for v in res.values())
    return res


def make_pairs(ids, n, seed):
    rng = np.random.default_rng(seed)
    pairs, seen = [], set()
    while len(pairs) < n:
        a, b = rng.choice(len(ids), size=2, replace=False)
        key = (ids[a], ids[b])
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def boundary_face_mask(faces: np.ndarray) -> np.ndarray:
    """Boundary faces touch at least one boundary edge (edge used by one face)."""
    from collections import Counter
    cnt = Counter()
    for tri in faces:
        for u, v in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            cnt[tuple(sorted((int(u), int(v))))] += 1
    boundary_edges = {e for e, c in cnt.items() if c == 1}
    mask = np.zeros(len(faces), dtype=bool)
    for fi, tri in enumerate(faces):
        for u, v in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            if tuple(sorted((int(u), int(v)))) in boundary_edges:
                mask[fi] = True
                break
    return mask


def real_diagnosis(n_pairs: int) -> dict:
    splits = common.load_split_ids()
    pairs = make_pairs(splits["test"], n_pairs, SEED)
    needed = sorted({i for p in pairs for i in p}, key=int)
    by_id = common.load_meshes(needed)
    top = common.template_topology(by_id)
    faces = top["faces"]
    landmarks = top["landmarks"]
    n = top["n_vertices"]

    # boundary-face diagnostics for the ROI mesh
    bmask = boundary_face_mask(faces)
    n_boundary_faces = int(bmask.sum())
    interior_faces = faces[~bmask]

    lap = uniform_laplacian(n, faces)
    sources = [by_id[s]["vertices"] for s, _ in pairs]
    controls = np.stack([(by_id[t]["vertices"] - by_id[s]["vertices"])[landmarks] for s, t in pairs], axis=0)
    init = solve_linear_handle_baseline(controls, landmarks, n, lap, 1000.0, 1e-8)
    arap = arap_predict_vectorised(
        sources, controls, landmarks, faces, lap, init, 1000.0, 1e-8, 3,
    ).reshape(len(pairs), n, 3)

    per_pair = []
    n_src_l, n_edit_l, n_new_l, n_new_int_l = [], [], [], []
    for k, (s, t) in enumerate(pairs):
        src = by_id[s]["vertices"]
        edited = src + arap[k]
        t0 = time.perf_counter()
        n_src = count_self_intersections(src, faces)
        n_edit = count_self_intersections(edited, faces)
        n_new = count_self_intersections(edited, faces, baseline_vertices=src)
        n_new_int = count_self_intersections(edited, interior_faces, baseline_vertices=src)
        dt = time.perf_counter() - t0
        n_src_l.append(n_src)
        n_edit_l.append(n_edit)
        n_new_l.append(n_new)
        n_new_int_l.append(n_new_int)
        per_pair.append({"source": s, "target": t, "n_src": int(n_src), "n_edit": int(n_edit),
                         "n_new": int(n_new), "n_new_interior": int(n_new_int), "sec": round(dt, 2)})
        print(f"pair {k+1}/{len(pairs)} {s}->{t}: n_src={n_src} n_edit={n_edit} "
              f"n_new={n_new} n_new_interior={n_new_int} ({dt:.1f}s)", flush=True)

    def stats(a):
        a = np.asarray(a, float)
        return {"mean": float(a.mean()), "p95": float(np.percentile(a, 95)), "max": float(a.max())}

    n_src_arr = np.asarray(n_src_l, float)
    n_new_arr = np.asarray(n_new_l, float)
    n_new_int_arr = np.asarray(n_new_int_l, float)
    src_high = n_src_arr.mean() > 1.0
    boundary_share = 1.0 - (n_new_int_arr.mean() / max(n_new_arr.mean(), 1e-9))
    boundary_dominated = boundary_share > 0.5
    parts = []
    if src_high:
        parts.append(f"The source ROI mesh ALREADY self-intersects (mean n_src={n_src_arr.mean():.0f}); "
                     "the prior '100% self-intersection' counted TOTAL intersections, not NEW ones -> must report only n_new.")
    if boundary_dominated:
        parts.append(f"NEW intersections are dominated by ROI-boundary faces "
                     f"(~{boundary_share*100:.0f}% disappear when boundary faces are excluded); "
                     "ROI-boundary triangles are the main false-positive source and must be excluded/segregated.")
    elif n_new_int_arr.mean() > 1.0:
        parts.append(f"Even excluding boundary faces, interior n_new remains high (mean={n_new_int_arr.mean():.0f}); "
                     "edits introduce genuine new interior self-intersections.")
    else:
        parts.append("After excluding boundary faces, interior n_new is ~0; new intersections are a boundary artefact.")
    root_cause = " ".join(parts)

    return {
        "n_pairs": len(pairs),
        "n_boundary_faces": n_boundary_faces,
        "n_total_faces": int(len(faces)),
        "boundary_share_of_new": float(boundary_share),
        "n_src": stats(n_src_l),
        "n_edit": stats(n_edit_l),
        "n_new": stats(n_new_l),
        "n_new_interior": stats(n_new_int_l),
        "root_cause": root_cause,
        "per_pair": per_pair,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=30)
    args = ap.parse_args()
    out = common.ensure_out()

    battery = synthetic_battery()
    diag = real_diagnosis(args.pairs)

    go = battery["all_correct"]
    result = {
        "gate": "2B_self_intersection",
        "synthetic_battery": battery,
        "real_diagnosis": diag,
        "decision": "GO" if go else "NO-GO",
        "decision_reason": (
            "Synthetic pos/neg/adjacency/pre-existing battery all correct; detector discriminates. "
            + diag["root_cause"]
        ) if go else (
            "Self-intersection detector FAILS synthetic discrimination -> per protocol, PERMANENTLY DROP "
            "the 'no self-intersection / global injectivity' claim; validity is limited to per-face signed foldover."
        ),
    }
    (out / "g2b_self_intersection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"battery": battery, "decision": result["decision"],
                      "n_src": diag["n_src"], "n_edit": diag["n_edit"], "n_new": diag["n_new"],
                      "root_cause": diag["root_cause"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
