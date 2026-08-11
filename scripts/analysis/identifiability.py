"""Stage-0 export for the active-acquisition / identifiability analysis (READ-ONLY, NO TRAINING).

Run this ONCE on Colab/Mac where the data + the model package are available. It dumps to plain .npy
the only things the affine-risk / identifiability analysis needs, so the rest can be computed in numpy
with no torch and no dense caches:

  train_roi_matrix.npy   (n_train_ids, 3*n_roi)  each UNIQUE training identity's neutral ROI vertices,
                          flattened (x0,y0,z0,x1,...). This forms Sigma = Cov(x_t).
  train_ids.json         the ordered list of training identity ids (rows of the matrix).
  landmark_indices.npy   (K,) integer ROI-vertex indices of the candidate control landmarks (expect 9).
  roi_faces.npy          (F,3) ROI triangle connectivity (for later geometry-risk checks).
  export_meta.json       shapes, hashes, counts for provenance.

It performs NO training, NO prediction, and does NOT touch any target non-landmark coordinate beyond
building the population shape covariance from TRAINING identities only (which is permitted: the affine
prior is fit on the training split, never on val/test targets).

Usage (from the repo root, adjust paths to your environment):
  python scripts/analysis/identifiability.py \
      --repo data \
      --model-package <path to the frozen model package .pt> \
      --out outputs/identifiability_inputs

If --model-package is omitted, train ids are derived via split_ids(by_id, "train pool").
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_of(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="data")
    ap.add_argument("--model-package", default="", help="frozen .pt package to read train_ids from")
    ap.add_argument("--out", default="outputs/identifiability_inputs")
    ap.add_argument("--split-name", default="clean-prior train",
                    help="split_ids label used if no model package is given (data.py uses 'clean-prior train')")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # data.py is the project's loader (same API used by noise_robustness.py)
    from rhinoform.data import load_rows, split_ids  # noqa: E402

    repo = Path(args.repo)
    _, by_id = load_rows(repo)

    # --- determine the UNIQUE training identities (never val/test) ---
    train_ids = None
    if args.model_package:
        import torch  # only needed to read the package; no model is built or run
        pkg = torch.load(args.model_package, map_location="cpu", weights_only=False)
        if "train_ids" in pkg:
            train_ids = [str(x) for x in pkg["train_ids"]]
    if train_ids is None:
        train_ids = [str(x) for x in split_ids(by_id, args.split_name)]
    # keep only ids that actually have a mesh, preserve order, dedupe
    seen = set()
    train_ids = [i for i in train_ids if (i in by_id and i not in seen and not seen.add(i))]
    if not train_ids:
        raise SystemExit("no training identities resolved; check --model-package / --split-name")

    template = by_id[train_ids[0]]
    landmarks = np.asarray(template["landmarks"]).reshape(-1).astype(np.int64)
    faces = np.asarray(template["faces"]).astype(np.int64)
    n_roi = int(np.asarray(template["vertices"]).shape[0])

    print(f"[export] train identities: {len(train_ids)}")
    print(f"[export] ROI vertices: {n_roi}  | candidate landmarks (K): {landmarks.size}  "
          f"| faces: {faces.shape[0]}")
    if landmarks.size != 9:
        print(f"[export] NOTE: expected 9 control landmarks, found {landmarks.size}. "
              f"Confirm this is the control-landmark index set before proceeding.")

    # --- stack each unique training identity's neutral ROI vertices, flattened ---
    rows = []
    kept = []
    for tid in train_ids:
        v = np.asarray(by_id[tid]["vertices"], dtype=np.float64)
        if v.shape[0] != n_roi:
            print(f"[export] skip {tid}: ROI vertex count {v.shape[0]} != {n_roi}")
            continue
        if not np.isfinite(v).all():
            print(f"[export] skip {tid}: non-finite coordinates")
            continue
        rows.append(v.reshape(-1))
        kept.append(tid)
    X = np.stack(rows, axis=0)  # (n_train, 3*n_roi)
    print(f"[export] target-shape matrix X: {X.shape}  (rows = unique training identities)")

    np.save(out / "train_roi_matrix.npy", X)
    np.save(out / "landmark_indices.npy", landmarks)
    np.save(out / "roi_faces.npy", faces)
    (out / "train_ids.json").write_text(json.dumps(kept, indent=0), encoding="utf-8")
    meta = {
        "n_train_identities": len(kept),
        "n_roi_vertices": n_roi,
        "n_landmarks": int(landmarks.size),
        "n_faces": int(faces.shape[0]),
        "X_shape": list(X.shape),
        "sha256": {
            "train_roi_matrix": sha256_of(X),
            "landmark_indices": sha256_of(landmarks),
            "roi_faces": sha256_of(faces),
        },
        "note": "READ-ONLY export; training identities only; no val/test target used.",
    }
    (out / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[export] wrote -> {out}")
    print(json.dumps(meta["sha256"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
