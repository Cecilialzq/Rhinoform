from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np


REQUIRED_LANDMARK_IDS = [str(i) for i in range(27, 36)]
REQUIRED_SUBUNITS = ["root", "dorsum", "tip", "alar_left", "alar_right"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_obj_vertices_faces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                p = line.strip().split()
                vertices.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("f "):
                raw = line.strip().split()[1:]
                idx = [int(x.split("/")[0]) - 1 for x in raw]
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) > 3:
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def normalize_subunits(subunits: dict) -> dict[str, np.ndarray]:
    groups = subunits.get("groups", subunits)
    out: dict[str, np.ndarray] = {}
    for name, val in groups.items():
        if isinstance(val, dict):
            arr = val.get("vertex_indices", val.get("indices", []))
        else:
            arr = val
        if isinstance(arr, list):
            out[name] = np.asarray(arr, dtype=np.int64)
    return out


def connected_components(roi: set[int], faces: np.ndarray) -> list[int]:
    adj = {v: set() for v in roi}
    for tri in faces:
        a, b, c = [int(x) for x in tri]
        for u, v in ((a, b), (b, c), (c, a)):
            if u in roi and v in roi:
                adj[u].add(v)
                adj[v].add(u)
    seen: set[int] = set()
    sizes = []
    for start in roi:
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        size = 0
        while q:
            u = q.popleft()
            size += 1
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def boundary_vertices(roi: set[int], faces: np.ndarray) -> set[int]:
    out: set[int] = set()
    for tri in faces:
        a, b, c = [int(x) for x in tri]
        for u, v in ((a, b), (b, c), (c, a)):
            u_in = u in roi
            v_in = v in roi
            if u_in != v_in:
                if u_in:
                    out.add(u)
                if v_in:
                    out.add(v)
    return out


def add_check(checks: list[dict], name: str, passed: bool, severity: str, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "severity": severity, "detail": detail})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi", default="../roi/vertices.json")
    parser.add_argument("--subunits", default="../roi/subunits.json")
    # A licensed full-head FaceScape registered mesh (26317 vertices) is required
    # to validate the ROI against the topology; it is not committed. Override with
    # the path to any neutral registered mesh, e.g. data/meshes_fullhead/1_neutral.obj.
    parser.add_argument("--topology", default="../data/meshes_fullhead/1_neutral.obj")
    parser.add_argument("--out", default="../results/roi_report.json")
    parser.add_argument("--min-vertices", type=int, default=3053)
    parser.add_argument("--max-components", type=int, default=1)
    parser.add_argument("--max-dorsum-boundary-ratio", type=float, default=0.20)
    parser.add_argument("--warn-dorsum-boundary-ratio", type=float, default=0.12)
    args = parser.parse_args()

    roi_json = load_json(Path(args.roi))
    sub_json = load_json(Path(args.subunits))
    vertices, faces = load_obj_vertices_faces(Path(args.topology))
    roi_indices = np.asarray(roi_json.get("roi_indices", []), dtype=np.int64)
    roi_set = set(int(x) for x in roi_indices)
    subunits = normalize_subunits(sub_json)
    checks: list[dict] = []

    add_check(
        checks,
        "vertex_count_matches_topology",
        int(roi_json.get("vertex_count", len(vertices))) == len(vertices),
        "error",
        f"roi vertex_count={roi_json.get('vertex_count')} topology vertices={len(vertices)}",
    )
    add_check(
        checks,
        "roi_count_at_least_minimum",
        len(roi_indices) >= args.min_vertices,
        "error",
        f"roi_count={len(roi_indices)} min_vertices={args.min_vertices}",
    )
    add_check(
        checks,
        "roi_indices_unique",
        len(roi_set) == len(roi_indices),
        "error",
        f"unique={len(roi_set)} total={len(roi_indices)}",
    )
    add_check(
        checks,
        "roi_indices_in_topology_range",
        bool(len(roi_indices) and roi_indices.min() >= 0 and roi_indices.max() < len(vertices)),
        "error",
        f"min={int(roi_indices.min()) if len(roi_indices) else None} max={int(roi_indices.max()) if len(roi_indices) else None} topology={len(vertices)}",
    )

    landmarks = roi_json.get("landmarks_27_35", {})
    missing_landmarks = [lm for lm in REQUIRED_LANDMARK_IDS if str(landmarks.get(lm, -1)) == "-1" or int(landmarks.get(lm, -1)) not in roi_set]
    add_check(
        checks,
        "official_landmarks_27_35_inside_roi",
        not missing_landmarks,
        "error",
        f"missing_or_outside={missing_landmarks}",
    )

    empty_subunits = [name for name in REQUIRED_SUBUNITS if len(subunits.get(name, [])) == 0]
    add_check(checks, "required_subunits_nonempty", not empty_subunits, "error", f"empty={empty_subunits}")

    outside_by_subunit = {
        name: int(sum(int(v) not in roi_set for v in np.asarray(subunits.get(name, []), dtype=np.int64)))
        for name in REQUIRED_SUBUNITS
    }
    add_check(
        checks,
        "required_subunits_subset_of_roi",
        all(v == 0 for v in outside_by_subunit.values()),
        "error",
        f"outside_counts={outside_by_subunit}",
    )

    components = connected_components(roi_set, faces)
    add_check(
        checks,
        "roi_connected_components",
        len(components) <= args.max_components,
        "error",
        f"n_components={len(components)} largest={components[:5]} max_components={args.max_components}",
    )

    boundary = boundary_vertices(roi_set, faces)
    dorsum = set(int(x) for x in subunits.get("dorsum", []))
    dorsum_boundary = len(boundary & dorsum)
    dorsum_ratio = dorsum_boundary / max(1, len(dorsum))
    add_check(
        checks,
        "dorsum_boundary_ratio_below_fail_threshold",
        dorsum_ratio <= args.max_dorsum_boundary_ratio,
        "error",
        f"dorsum_boundary={dorsum_boundary} dorsum_count={len(dorsum)} ratio={dorsum_ratio:.4f} max={args.max_dorsum_boundary_ratio}",
    )
    add_check(
        checks,
        "dorsum_boundary_ratio_below_warning_threshold",
        dorsum_ratio <= args.warn_dorsum_boundary_ratio,
        "warning",
        f"dorsum_boundary={dorsum_boundary} dorsum_count={len(dorsum)} ratio={dorsum_ratio:.4f} warn={args.warn_dorsum_boundary_ratio}",
    )

    subunit_counts = {name: int(len(arr)) for name, arr in subunits.items()}
    error_failed = [c for c in checks if c["severity"] == "error" and not c["passed"]]
    warning_failed = [c for c in checks if c["severity"] == "warning" and not c["passed"]]
    report = {
        "decision": "fail" if error_failed else "pass_with_warnings" if warning_failed else "pass",
        "roi_path": str(Path(args.roi)),
        "subunits_path": str(Path(args.subunits)),
        "topology_path": str(Path(args.topology)),
        "roi_vertex_count": int(len(roi_indices)),
        "topology_vertex_count": int(len(vertices)),
        "subunit_counts": subunit_counts,
        "connected_components": components,
        "boundary_vertex_count": int(len(boundary)),
        "dorsum_boundary_count": int(dorsum_boundary),
        "dorsum_boundary_ratio": float(dorsum_ratio),
        "checks": checks,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if error_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
