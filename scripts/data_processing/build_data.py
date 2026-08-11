"""Build the cropped nasal ROI dataset from full-head FaceScape neutral meshes.

The operation is a pure crop: vertices == raw_obj[global_roi_indices], with no
coordinate transform. The final ROI has 3934 vertices.

Per-subject npz schema matches data.load_rows:
  vertices (N,3) float32, faces (F,3) int32, local_landmarks (9,) int32,
  subunit_{root,dorsum,tip,alar_left,alar_right} int32, plus roi_indices_full.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from rhinoform.repro import atomic_write_json, sha256_file


def parse_obj(fn: Path, want_faces: bool):
    vs = []
    fs = []
    with open(fn, "r", errors="ignore") as f:
        for ln in f:
            if ln.startswith("v "):
                p = ln.split()
                vs.append((float(p[1]), float(p[2]), float(p[3])))
            elif want_faces and ln.startswith("f "):
                p = ln.split()[1:]
                idx = [int(tok.split("/")[0]) - 1 for tok in p]
                # triangulate fan if quad+
                for k in range(1, len(idx) - 1):
                    fs.append((idx[0], idx[k], idx[k + 1]))
    V = np.asarray(vs, dtype=np.float64)
    F = np.asarray(fs, dtype=np.int64) if want_faces else None
    return V, F


def resolve_obj(obj_dir: Path, sid: str) -> Path:
    candidates = [
        obj_dir / sid / "models_reg" / "1_neutral.obj",
        obj_dir / f"{sid}.obj",
    ]
    if sid.isdigit():
        candidates.append(obj_dir / f"{int(sid):03d}" / "models_reg" / "1_neutral.obj")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"{sid}: neutral registered OBJ not found under {obj_dir}")


def split_subject_ids(split_manifest: Path) -> list[str]:
    man = json.loads(split_manifest.read_text(encoding="utf-8"))
    return [str(r["subject_id"]) for r in man.get("rows", [])]


def existing_dataset_done_valid(done_path: Path, out_dir: Path, split_manifest: Path) -> tuple[bool, str]:
    manifest_path = out_dir / "manifest.json"
    if not done_path.exists() or not manifest_path.exists():
        return False, "missing DATASET_DONE or manifest.json"
    try:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"invalid JSON: {exc!r}"
    split_sha = sha256_file(split_manifest)
    if done.get("decision") != "pass":
        return False, "DATASET_DONE decision is not pass"
    if done.get("split_manifest_sha256") != split_sha:
        return False, "DATASET_DONE split_manifest_sha256 does not match current split manifest"
    expected = len(split_subject_ids(split_manifest))
    if int(done.get("n_rows", -1)) != expected or int(manifest.get("n_rows", -1)) != expected:
        return False, f"row count mismatch for current split manifest: expected {expected}"
    if done.get("failed"):
        return False, "DATASET_DONE contains failed rows"
    roi_count = int(done.get("roi_vertex_count", -1))
    topology_vertex_count = int(done.get("topology_vertex_count", -1))
    for row in done.get("rows", []):
        if int(row.get("roi_count", -1)) != roi_count:
            return False, f"{row.get('subject_id')}: roi_count mismatch"
        if int(row.get("vertex_count", -1)) != topology_vertex_count:
            return False, f"{row.get('subject_id')}: vertex_count mismatch"
        if not row.get("landmarks_present"):
            return False, f"{row.get('subject_id')}: landmarks not present"
    return True, "valid"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj-dir", type=Path, required=True)
    parser.add_argument("--roi", type=Path, default=Path("roi/vertices.json"))
    parser.add_argument("--subunits", type=Path, default=Path("roi/subunits.json"))
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--dataset-done", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    done_path = args.dataset_done or (args.out / "DATASET_DONE.json")
    if done_path.exists() and not args.force:
        valid, reason = existing_dataset_done_valid(done_path, args.out, args.split_manifest)
        if valid:
            print(f"DATASET_DONE exists and matches current split manifest; skipping extraction: {done_path}")
            return 0
        print(f"DATASET_DONE exists but cannot be reused ({reason}); rebuilding extraction.")

    out_mesh = args.out / "meshes"
    nr = json.loads(args.roi.read_text())
    roi = np.asarray(nr["roi_indices"], dtype=np.int64)          # global indices (0-based)
    nverts_full = int(nr["vertex_count"])
    lms = nr["landmarks_27_35"]
    lm_global = np.asarray([lms[str(k)] for k in range(27, 36)], dtype=np.int64)
    lm_ids = np.arange(27, 36, dtype=np.int64)

    su = json.loads(args.subunits.read_text())["groups"]
    subunit_local = {k: np.asarray(su[k]["local_roi_indices"], dtype=np.int64)
                     for k in ("root", "dorsum", "tip", "alar_left", "alar_right")}

    # global -> local map over the ROI
    g2l = -np.ones(nverts_full, dtype=np.int64)
    g2l[roi] = np.arange(roi.size, dtype=np.int64)
    local_landmarks = g2l[lm_global]
    assert (local_landmarks >= 0).all(), "some landmark not in ROI"

    man = json.loads(args.split_manifest.read_text())
    if "rows" in man:
        split_by_id = {str(r["subject_id"]): str(r["split"]) for r in man["rows"]}
    else:
        split_by_id = {}

    # faces: shared registered topology -> parse once, restrict to ROI
    topology_sid = sorted(split_by_id, key=lambda s: int(s) if s.isdigit() else 10**9)[0]
    topology_path = resolve_obj(args.obj_dir, topology_sid)
    V_top, F_full = parse_obj(topology_path, want_faces=True)
    if V_top.shape[0] != nverts_full:
        raise SystemExit(f"topology vertex count mismatch: {V_top.shape[0]} != {nverts_full}")
    in_roi = np.zeros(nverts_full, dtype=bool); in_roi[roi] = True
    keep = in_roi[F_full[:, 0]] & in_roi[F_full[:, 1]] & in_roi[F_full[:, 2]]
    F_local = g2l[F_full[keep]].astype(np.int32)
    print(f"ROI {roi.size} verts, faces kept {F_local.shape[0]}/{F_full.shape[0]}")

    out_mesh.mkdir(parents=True, exist_ok=True)
    rows = []
    done_rows = []
    failed = []
    ids = sorted(split_by_id, key=lambda s: int(s))
    for sid in ids:
        try:
            obj_path = resolve_obj(args.obj_dir, sid)
            V, _ = parse_obj(obj_path, want_faces=False)
            assert V.shape[0] == nverts_full, f"{sid}: {V.shape[0]} != {nverts_full}"
            assert np.isfinite(V).all(), f"{sid}: non-finite vertex coordinate"
        except Exception as exc:
            failed.append({"subject_id": sid, "reason": repr(exc)})
            continue
        Vroi = V[roi].astype(np.float32)
        npz_path = out_mesh / f"{sid}_neutral.npz"
        np.savez(
            npz_path,
            vertices=Vroi,
            faces=F_local,
            local_landmarks=local_landmarks.astype(np.int32),
            landmark_ids=lm_ids.astype(np.int32),
            roi_indices_full=roi.astype(np.int32),
            subunit_root=subunit_local["root"].astype(np.int32),
            subunit_dorsum=subunit_local["dorsum"].astype(np.int32),
            subunit_tip=subunit_local["tip"].astype(np.int32),
            subunit_alar_left=subunit_local["alar_left"].astype(np.int32),
            subunit_alar_right=subunit_local["alar_right"].astype(np.int32),
        )
        digest = sha256_file(npz_path)
        rows.append({
            "npz_path": f"meshes/{sid}_neutral.npz",
            "subject_id": sid,
            "expression": "neutral",
            "split": split_by_id[sid],
            "roi_count": int(roi.size),
            "vertex_count": nverts_full,
            "sha256": digest,
        })
        done_rows.append({
            "subject_id": sid,
            "source_obj_sha256": sha256_file(obj_path),
            "npz_sha256": digest,
            "vertex_count": int(V.shape[0]),
            "roi_count": int(Vroi.shape[0]),
            "landmarks_present": bool((local_landmarks >= 0).all()),
        })

    manifest = {
        "source": "build_data.py",
        "dataset_name": "nasal_roi",
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "roi_vertex_count": int(roi.size),
        "topology_vertex_count": int(nverts_full),
        "n_rows": len(rows),
        "rows": rows,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    done = {
        "decision": "fail" if failed else "pass",
        "source": "build_data.py",
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "topology_source": str(topology_path),
        "topology_vertex_count": int(V_top.shape[0]),
        "topology_face_count": int(F_full.shape[0]),
        "roi_vertex_count": int(roi.size),
        "n_rows": len(rows),
        "failed": failed,
        "rows": done_rows,
    }
    atomic_write_json(done_path, done)
    from collections import Counter
    print("splits:", dict(Counter(r["split"] for r in rows)))
    print("wrote", len(rows), "meshes ->", args.out)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
