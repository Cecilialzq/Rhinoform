"""Five-subunit STRICT scoring and exact streaming spatial diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .data import edge_index
from .safe_fusion import signed_fold_indicator

REGIONS = ("root", "dorsum", "tip", "alar_left", "alar_right")
FACE_ASSIGNMENTS = ("legacy_majority_vote", "unique_majority_sensitivity")
REGIONAL_METRICS = (
    "free_rmse", "landmark_rmse", "edge_strain_p95", "normal_flip_pct",
    "abs_flip_pct", "missed_flip_pct", "target_flip_pct",
)


@dataclass(frozen=True)
class RegionTopology:
    regions: tuple[str, ...]
    vertex_region: np.ndarray
    region_vertices: tuple[np.ndarray, ...]
    region_free_vertices: tuple[np.ndarray, ...]
    region_landmarks: tuple[np.ndarray, ...]
    faces: np.ndarray
    face_region_legacy: np.ndarray
    face_region_unique_majority: np.ndarray
    edges: np.ndarray
    edge_region: np.ndarray
    landmarks: np.ndarray
    n_face_ties: int

    def audit_dict(self) -> dict[str, object]:
        return {
            "regions": list(self.regions),
            "vertex_counts": {n: int(len(self.region_vertices[i])) for i, n in enumerate(self.regions)},
            "free_vertex_counts": {n: int(len(self.region_free_vertices[i])) for i, n in enumerate(self.regions)},
            "landmark_counts": {n: int(len(self.region_landmarks[i])) for i, n in enumerate(self.regions)},
            "legacy_face_counts": {n: int(np.sum(self.face_region_legacy == i)) for i, n in enumerate(self.regions)},
            "unique_majority_face_counts": {n: int(np.sum(self.face_region_unique_majority == i)) for i, n in enumerate(self.regions)},
            "intra_region_edge_counts": {n: int(np.sum(self.edge_region == i)) for i, n in enumerate(self.regions)},
            "legacy_tie_break_order": list(self.regions),
            "n_face_ties_1_1_1": int(self.n_face_ties),
            "n_sensitivity_boundary_faces": int(np.sum(self.face_region_unique_majority < 0)),
            "n_cross_region_edges": int(np.sum(self.edge_region < 0)),
        }


def build_region_topology(subunits, faces, landmarks, n_vertices: int) -> RegionTopology:
    missing = [name for name in REGIONS if name not in subunits]
    if missing:
        raise ValueError(f"Five-mask partition is missing subunits: {missing}")
    groups = []
    for name in REGIONS:
        values = np.asarray(subunits[name], dtype=np.int64).reshape(-1)
        if len(values) != len(np.unique(values)) or np.any(values < 0) or np.any(values >= n_vertices):
            raise ValueError(f"Invalid vertices in subunit {name}")
        groups.append(np.sort(values))
    groups = tuple(groups)
    joined = np.concatenate(groups)
    if len(joined) != n_vertices or not np.array_equal(np.sort(joined), np.arange(n_vertices)):
        raise ValueError("The five subunit masks must form a complete, disjoint vertex partition")
    vertex_region = np.full(n_vertices, -1, dtype=np.int64)
    for index, values in enumerate(groups):
        vertex_region[values] = index
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or np.any(faces < 0) or np.any(faces >= n_vertices):
        raise ValueError("faces must be valid triangles")
    labels = vertex_region[faces]
    counts = np.stack([np.sum(labels == i, axis=1) for i in range(len(REGIONS))], axis=1)
    legacy = np.argmax(counts, axis=1).astype(np.int64)
    maxima = np.max(counts, axis=1)
    unique = np.where(maxima >= 2, legacy, -1).astype(np.int64)
    edges = edge_index(faces)
    left, right = vertex_region[edges[:, 0]], vertex_region[edges[:, 1]]
    edge_region = np.where(left == right, left, -1).astype(np.int64)
    landmarks = np.asarray(landmarks, dtype=np.int64).reshape(-1)
    if len(landmarks) != len(np.unique(landmarks)) or np.any(landmarks < 0) or np.any(landmarks >= n_vertices):
        raise ValueError("Invalid landmarks")
    region_landmarks = tuple(landmarks[vertex_region[landmarks] == i] for i in range(len(REGIONS)))
    free = tuple(values[~np.isin(values, landmarks)] for values in groups)
    return RegionTopology(
        REGIONS, vertex_region, groups, free, region_landmarks, faces, legacy, unique,
        edges, edge_region, landmarks, int(np.sum(maxima == 1)),
    )


def empty_spatial_accumulator(topology: RegionTopology) -> dict[str, np.ndarray | int]:
    return {
        "n_pairs": 0,
        "vertex_error_sum": np.zeros(len(topology.vertex_region), dtype=np.float64),
        "vertex_error_max": np.zeros(len(topology.vertex_region), dtype=np.float64),
        "edge_strain_sum": np.zeros(len(topology.edges), dtype=np.float64),
        "edge_strain_max": np.zeros(len(topology.edges), dtype=np.float64),
        "face_new_fold_count": np.zeros(len(topology.faces), dtype=np.int64),
        "face_abs_fold_count": np.zeros(len(topology.faces), dtype=np.int64),
        "face_missed_fold_count": np.zeros(len(topology.faces), dtype=np.int64),
        "face_target_fold_count": np.zeros(len(topology.faces), dtype=np.int64),
    }


def _rmse(pred, target, indices) -> float:
    if not len(indices):
        return float("nan")
    error = pred[indices] - target[indices]
    return float(np.sqrt(np.mean(np.sum(error * error, axis=1))))


def _pct(mask) -> float:
    return float(np.mean(mask) * 100.0) if len(mask) else float("nan")


def regional_metric_row_sets(
    by_id, pairs, pred_flat, topology: RegionTopology, *, method: str,
    face_assignments: tuple[str, ...] = FACE_ASSIGNMENTS,
    pair_index_offset: int = 0, spatial_accumulator=None,
) -> dict[str, list[dict[str, object]]]:
    if not face_assignments or any(value not in FACE_ASSIGNMENTS for value in face_assignments):
        raise ValueError(f"Unknown face assignments: {face_assignments}")
    pred = np.asarray(pred_flat, dtype=np.float64)
    if pred.ndim == 2:
        pred = pred.reshape(len(pred), -1, 3)
    if pred.shape != (len(pairs), len(topology.vertex_region), 3):
        raise ValueError(f"Prediction shape mismatch: {pred.shape}")
    face_maps = {
        "legacy_majority_vote": topology.face_region_legacy,
        "unique_majority_sensitivity": topology.face_region_unique_majority,
    }
    outputs = {name: [] for name in face_assignments}
    for local, ((source_id, target_id), raw) in enumerate(zip(pairs, pred)):
        source = np.asarray(by_id[str(source_id)]["vertices"], dtype=np.float64)
        target = np.asarray(by_id[str(target_id)]["vertices"], dtype=np.float64)
        true_delta = target - source
        prediction = raw.copy()
        prediction[topology.landmarks] = true_delta[topology.landmarks]
        edited = source + prediction
        e0 = np.linalg.norm(source[topology.edges[:, 0]] - source[topology.edges[:, 1]], axis=1)
        e1 = np.linalg.norm(edited[topology.edges[:, 0]] - edited[topology.edges[:, 1]], axis=1)
        strain = np.abs(e1 - e0) / np.maximum(e0, 1e-12)
        pred_fold = signed_fold_indicator(source, edited, topology.faces)[0] < 0.0
        target_fold = signed_fold_indicator(source, target, topology.faces)[0] < 0.0
        new_fold, missed_fold = pred_fold & ~target_fold, target_fold & ~pred_fold
        if spatial_accumulator is not None:
            vertex_error = np.linalg.norm(prediction - true_delta, axis=1)
            spatial_accumulator["vertex_error_sum"] += vertex_error
            spatial_accumulator["vertex_error_max"] = np.maximum(spatial_accumulator["vertex_error_max"], vertex_error)
            spatial_accumulator["edge_strain_sum"] += strain
            spatial_accumulator["edge_strain_max"] = np.maximum(spatial_accumulator["edge_strain_max"], strain)
            spatial_accumulator["face_new_fold_count"] += new_fold
            spatial_accumulator["face_abs_fold_count"] += pred_fold
            spatial_accumulator["face_missed_fold_count"] += missed_fold
            spatial_accumulator["face_target_fold_count"] += target_fold
            spatial_accumulator["n_pairs"] += 1
        for region_index, region in enumerate(topology.regions):
            free = topology.region_free_vertices[region_index]
            controls = topology.region_landmarks[region_index]
            edge_mask = topology.edge_region == region_index
            regional_strain = strain[edge_mask]
            shared = {
                "pair_index": int(pair_index_offset + local), "source_id": str(source_id),
                "target_id": str(target_id), "method": str(method), "region": region,
                "free_rmse": _rmse(prediction, true_delta, free),
                "landmark_rmse": _rmse(prediction, true_delta, controls),
                "edge_strain_p95": float(np.percentile(regional_strain, 95)) if len(regional_strain) else float("nan"),
                "n_free_vertices": int(len(free)), "n_landmarks": int(len(controls)),
                "n_edges": int(np.sum(edge_mask)),
            }
            for assignment in face_assignments:
                face_mask = face_maps[assignment] == region_index
                outputs[assignment].append({
                    **shared, "face_assignment": assignment,
                    "normal_flip_pct": _pct(new_fold[face_mask]),
                    "abs_flip_pct": _pct(pred_fold[face_mask]),
                    "missed_flip_pct": _pct(missed_fold[face_mask]),
                    "target_flip_pct": _pct(target_fold[face_mask]),
                    "n_faces": int(np.sum(face_mask)),
                    "n_new_flip_faces": int(np.sum(new_fold[face_mask])),
                })
    return outputs


def regional_metric_rows(
    by_id, pairs, pred_flat, topology, *, method: str,
    face_assignment: str = "legacy_majority_vote", pair_index_offset: int = 0,
):
    return regional_metric_row_sets(
        by_id, pairs, pred_flat, topology, method=method,
        face_assignments=(face_assignment,), pair_index_offset=pair_index_offset,
    )[face_assignment]
