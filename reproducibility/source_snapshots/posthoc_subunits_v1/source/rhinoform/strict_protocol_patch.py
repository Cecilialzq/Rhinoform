"""Unified STRICT evaluation protocol patch (Gen2 + strict).

Single source of truth for the frozen, paper-ready evaluation contract. Importing
this module and calling ``apply()`` monkeypatches the two per-pair scorers used
across the repo so that EVERY downstream evaluation script emits metrics under
one consistent protocol, without re-implementing each script:

  * ``stats.metric_rows_for_method``      (neural / RBSR / safe-fusion / noise / cross-expr)
  * ``baselines.per_pair_metrics``        (classical Laplacian / bi-Laplacian / ARAP)

The STRICT contract (matches strict_sparse_dense_diagnostics + Plan A):

  1. The 9 control landmarks are HARD-OVERWRITTEN with the ground-truth target
     delta before scoring, so the controls are exact by construction
     (landmark_rmse ~ 0, kept only as a sanity column).
  2. ``roi_rmse`` is the STRICT FREE RMSE over the ROI vertices EXCLUDING the 9
     landmark vertices (3925 of 3934 for the current ROI). Subunit RMSEs also
     exclude any landmark vertex.
  3. ``normal_flip_pct`` is the BASELINE-RELATIVE NEW flip percentage: faces the
     prediction reverses (signed Jacobian det < 0 in the source frame) that the
     TRUE i2i target does NOT reverse. Intrinsic cross-identity folds are not
     charged to the model.

The 6 canonical metric keys are preserved (so all bootstrap / Holm / table code
keeps working); only their MEANING is tightened. Extra transparency columns are
added: ``abs_flip_pct`` (legacy absolute), ``missed_flip_pct``, ``target_flip_pct``.

Usage (in a runner, BEFORE importing the target evaluation script):
    import strict_protocol_patch as strict
    strict.apply()
    # then run the existing evaluation script as __main__
"""
from __future__ import annotations

import numpy as np

from .data import edge_index, face_normals, rmse_vertices
from .safe_fusion import signed_fold_indicator


def _free_indices(n_vertices: int, landmarks: np.ndarray) -> np.ndarray:
    mask = np.ones(int(n_vertices), dtype=bool)
    mask[np.asarray(landmarks, dtype=np.int64)] = False
    return np.where(mask)[0]


def _subunit_free(indices: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    lms = np.asarray(landmarks, dtype=np.int64)
    return idx[~np.isin(idx, lms)]


def strict_metric_rows(by_id: dict, pairs, pred_flat: np.ndarray) -> list[dict]:
    """STRICT per-pair scorer. Drop-in replacement for stats.metric_rows_for_method."""
    template = next(iter(by_id.values()))
    landmarks = np.asarray(template["landmarks"], dtype=np.int64)
    faces = template["faces"]
    subunits = template["subunits"]
    edges = edge_index(faces)
    n_vertices = template["vertices"].shape[0]
    free_idx = _free_indices(n_vertices, landmarks)
    dorsum_free = _subunit_free(subunits["dorsum"], landmarks)
    tip_free = _subunit_free(subunits["tip"], landmarks)

    rows: list[dict] = []
    for i, (src_id, tgt_id) in enumerate(pairs):
        src = np.asarray(by_id[str(src_id)]["vertices"], dtype=np.float64)
        tgt = np.asarray(by_id[str(tgt_id)]["vertices"], dtype=np.float64)
        true_delta = tgt - src
        pred_delta = np.asarray(pred_flat[i], dtype=np.float64).reshape(-1, 3).copy()

        # (1) hard-fix the 9 controls to ground truth
        pred_delta[landmarks] = true_delta[landmarks]
        pred_vertices = src + pred_delta

        # (2) strict free RMSE (exclude landmark vertices)
        roi_free_rmse = rmse_vertices(pred_delta, true_delta, free_idx)
        landmark_rmse = rmse_vertices(pred_delta, true_delta, landmarks)  # ~0 sanity

        # edge strain p95 (on the controls-fixed mesh)
        e0 = np.linalg.norm(src[edges[:, 0]] - src[edges[:, 1]], axis=1)
        e1 = np.linalg.norm(pred_vertices[edges[:, 0]] - pred_vertices[edges[:, 1]], axis=1)
        strain = np.abs(e1 - e0) / np.maximum(e0, 1e-12)

        # (3) baseline-relative NEW normal flips
        det_pred, _ = signed_fold_indicator(src, pred_vertices, faces)
        det_tgt, _ = signed_fold_indicator(src, tgt, faces)
        pred_fold = det_pred < 0.0
        target_fold = det_tgt < 0.0
        new_fold = pred_fold & ~target_fold
        missed_fold = target_fold & ~pred_fold
        n_faces = float(len(faces))

        rows.append(
            {
                "pair_index": float(i),
                "source_id": str(src_id),
                "target_id": str(tgt_id),
                "roi_rmse": float(roi_free_rmse),                 # STRICT FREE RMSE
                "landmark_rmse": float(landmark_rmse),            # ~0 sanity check
                "dorsum_rmse": float(rmse_vertices(pred_delta, true_delta, dorsum_free)),
                "tip_rmse": float(rmse_vertices(pred_delta, true_delta, tip_free)),
                "edge_strain_p95": float(np.percentile(strain, 95)),
                "normal_flip_pct": float(np.mean(new_fold) * 100.0),   # NEW flip (baseline-relative)
                # transparency extras (downstream ignores unknown columns):
                "abs_flip_pct": float(np.mean(pred_fold) * 100.0),
                "missed_flip_pct": float(np.mean(missed_fold) * 100.0),
                "target_flip_pct": float(np.mean(target_fold) * 100.0),
                "n_new_flip_faces": float(np.sum(new_fold)),
                "n_faces": n_faces,
            }
        )
    return rows


def _strict_per_pair_metrics(name, by_id, pairs, pred_flat, faces, edges, subunits, landmarks):
    """STRICT drop-in for baselines.per_pair_metrics (signature-compatible)."""
    return strict_metric_rows(by_id, pairs, pred_flat)


_APPLIED = False


def apply() -> None:
    """Monkeypatch the repo's two per-pair scorers to the strict protocol."""
    global _APPLIED
    if _APPLIED:
        return
    from . import stats
    from . import baselines

    stats.metric_rows_for_method = strict_metric_rows
    baselines.per_pair_metrics = _strict_per_pair_metrics
    _APPLIED = True
    print("[strict_protocol_patch] STRICT protocol active: "
          "landmarks hard-fixed -> free RMSE; normal_flip_pct = baseline-relative NEW flip.",
          flush=True)
