"""Shared inference + metric utilities for the spatial-residual-gate (SRG)
novelty-upgrade experiments.

Scientific-integrity notes
--------------------------
* This module performs *inference only* using the frozen Required-tier model
  package (`neural_field_model_package_cvae_ew0p1_lw0.pt`). It does not train.
* It reproduces the frozen ridge + CVAE dense predictions on the fixed
  validation (70 ids) and test (100 ids) identity sets so a post-hoc spatial
  gate can be selected on validation and evaluated once on test.
* Ridge predictions are an exact function of the stored ridge model and the
  per-pair conditioning vector; they do not depend on the static vertex
  features and are therefore bit-for-bit reproducible.
* The CVAE deterministic (prior-mean) prediction depends on the static vertex
  feature `scale` normaliser, which in the frozen run was computed over the 676
  train-pool meshes. Those meshes are not all materialised locally, so `scale`
  is recomputed from the available mesh set and the reproduction is *verified*
  against the frozen aggregate metrics in `reproduce_check.py`. Any residual
  discrepancy is reported, not hidden.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ..data import edge_index, face_normals, load_rows  # noqa: E402
from ..train import NeuralFieldCVAE, pair_conditions, predict_field  # noqa: E402
from ..cvae import pca_transform  # noqa: E402

PACKAGE_DEFAULT = ROOT / "runs/required_primary_chain0_scale676/neural_field_model_package_cvae_ew0p1_lw0.pt"


def load_package(path: Path = PACKAGE_DEFAULT) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_available_rows(repo: Path) -> dict:
    """Like data.load_rows but only loads meshes present on disk (val+test).

    Preserves manifest row order so the feature template (first row, id 1) and
    shared topology match the frozen run.
    """
    import json
    manifest = json.loads((repo / "manifest.json").read_text(encoding="utf-8"))
    by_id: dict[str, dict] = {}
    for row in manifest["rows"]:
        npz_path = repo / row["npz_path"]
        try:
            data = np.load(npz_path, allow_pickle=False)
            _ = data["vertices"].shape
        except OSError:
            continue
        by_id[str(row["subject_id"])] = {
            "subject_id": str(row["subject_id"]),
            "split": str(row["split"]),
            "vertices": np.asarray(data["vertices"], dtype=np.float64),
            "faces": np.asarray(data["faces"], dtype=np.int64),
            "landmarks": np.asarray(data["local_landmarks"], dtype=np.int64),
            "subunits": {k: np.asarray(data[f"subunit_{k}"], dtype=np.int64)
                         for k in ("root", "dorsum", "tip", "alar_left", "alar_right")},
        }
    return by_id


def build_vertex_features(by_id: dict, train_ids: list[str], source_pca: dict,
                          use_subunit_features: bool, available_ids: list[str]) -> np.ndarray:
    """Reproduce train.build_static_vertex_features.

    center is recovered exactly from the source-PCA mean (mean over train
    identities of flattened vertices, i.e. the per-vertex mean shape); its mean
    over vertices equals the train vertex centroid used in the frozen run.
    scale (bbox diagonal over train vertices) is approximated from the meshes
    available locally; reproduce_check.py verifies the approximation.
    """
    template = next(iter(by_id.values()))
    faces = template["faces"]
    landmarks = template["landmarks"]
    base = template["vertices"]

    center = source_pca["mean"].reshape(-1, 3).mean(axis=0, keepdims=True)
    verts = np.concatenate([by_id[s]["vertices"] for s in available_ids], axis=0)
    scale = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    scale = max(scale, 1e-6)

    coords = (base - center) / scale
    normals = _vertex_normals(base, faces)
    lm = base[landmarks]
    dist = np.linalg.norm(base[:, None, :] - lm[None, :, :], axis=2)
    dist = dist / np.maximum(np.percentile(dist, 95, axis=0, keepdims=True), 1e-6)
    parts = [coords, normals]
    if use_subunit_features:
        parts.append(_subunit_one_hot(template))
    parts.append(dist)
    return np.concatenate(parts, axis=1).astype(np.float32)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float64)
    tri = faces.astype(np.int64)
    fn = np.cross(vertices[tri[:, 1]] - vertices[tri[:, 0]], vertices[tri[:, 2]] - vertices[tri[:, 0]])
    for j in range(3):
        np.add.at(normals, tri[:, j], fn)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def _subunit_one_hot(row: dict) -> np.ndarray:
    n = row["vertices"].shape[0]
    out = np.zeros((n, 5), dtype=np.float64)
    for i, name in enumerate(["root", "dorsum", "tip", "alar_left", "alar_right"]):
        out[row["subunits"][name], i] = 1.0
    return out


def regenerate_predictions(by_id: dict, pkg: dict, pairs: list, vertex_feat: np.ndarray, device=None):
    """Return (ridge_pred, cvae_pred, target_delta) each (P, V*3) / (P,V,3).

    ``device`` is optional and backward-compatible: when None the CVAE stays on
    CPU (original behaviour). Pass a torch device (e.g. 'cuda') to run the CVAE
    inference on GPU; ``predict_field`` follows the model's device automatically.
    """
    from ..data import ridge_predict
    source_pca = pkg["source_pca"]
    cond, _src, target_delta, _ctrl, _code, _m, _s = pair_conditions(
        by_id, pairs, source_pca, pkg["cond_mean"], pkg["cond_std"])
    ridge_pred = ridge_predict(cond.astype(np.float64), pkg["ridge_cond"]).astype(np.float32)

    cvae = NeuralFieldCVAE(vertex_feat.shape[1], cond.shape[1],
                           obs_dim=int(pkg["args"]["delta_pca_dim"]),
                           latent_dim=int(pkg["args"]["latent_dim"]),
                           hidden=int(pkg["args"]["hidden"]))
    cvae.load_state_dict(pkg["cvae_state_dict"])
    cvae.eval()
    if device is not None:
        cvae.to(device)
    cvae_pred = predict_field(cvae, vertex_feat, cond, is_cvae=True).astype(np.float32)
    return ridge_pred, cvae_pred, target_delta.reshape(len(pairs), -1).astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics (mirror code/baselines.per_pair_metrics exactly)
# ---------------------------------------------------------------------------
def per_pair_metric_rows(by_id, pairs, pred_flat, faces, edges, subunits, landmarks):
    rows = []
    fn0, e0c = {}, {}
    for i, (s, t) in enumerate(pairs):
        src = by_id[s]["vertices"]
        tgt = by_id[t]["vertices"]
        true_delta = (tgt - src).reshape(-1, 3)
        pred_delta = pred_flat[i].reshape(-1, 3)
        pred_v = src + pred_delta

        def rmse(idx=None):
            pd, td = pred_delta, true_delta
            if idx is not None:
                pd, td = pd[idx], td[idx]
            return float(np.sqrt(np.mean(np.sum((pd - td) ** 2, axis=1))))

        if s not in e0c:
            e0c[s] = np.linalg.norm(src[edges[:, 0]] - src[edges[:, 1]], axis=1)
            fn0[s] = face_normals(src, faces)
        e1 = np.linalg.norm(pred_v[edges[:, 0]] - pred_v[edges[:, 1]], axis=1)
        strain = np.abs(e1 - e0c[s]) / np.maximum(e0c[s], 1e-12)
        n1 = face_normals(pred_v, faces)
        flip = float(np.mean(np.sum(fn0[s] * n1, axis=1) < 0.0) * 100.0)
        rows.append({
            "pair_index": float(i), "source_id": s, "target_id": t,
            "roi_rmse": rmse(), "landmark_rmse": rmse(landmarks),
            "dorsum_rmse": rmse(subunits["dorsum"]), "tip_rmse": rmse(subunits["tip"]),
            "edge_strain_p95": float(np.percentile(strain, 95)), "normal_flip_pct": flip,
        })
    return rows


def aggregate(rows):
    keys = ["roi_rmse", "landmark_rmse", "dorsum_rmse", "tip_rmse", "edge_strain_p95", "normal_flip_pct"]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def topology(by_id):
    tmpl = next(iter(by_id.values()))
    return tmpl["faces"], edge_index(tmpl["faces"]), tmpl["subunits"], tmpl["landmarks"]
