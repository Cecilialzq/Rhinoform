"""Geometry-aware fusion of a smooth deformation base and learned residual.

The module separates two concepts which the legacy code conflated:

* geometric distortion: triangle area/stretch, edge strain and self-contact;
* source-normal reversal: a useful diagnostic, but not an injectivity proof.

The projection is deterministic. It attenuates only vertices incident to unsafe
faces/edges, diffuses attenuation to avoid seams, and returns either a checked
mesh or an explicit failure/fallback status.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SafetyThresholds:
    min_area_ratio: float = 0.02
    min_sigma: float = 0.10
    max_sigma: float = 3.0
    max_edge_strain: float = 1.0
    max_normal_rotation_deg: float | None = None
    allow_baseline_violations: bool = True
    attenuation: float = 0.75
    smoothing_steps: int = 2
    max_iterations: int = 50
    uniform_steps: int = 101


@dataclass(frozen=True)
class TriangleDistortion:
    area_ratio: np.ndarray
    sigma_min: np.ndarray
    sigma_max: np.ndarray
    normal_cosine: np.ndarray


@dataclass(frozen=True)
class SafetyReport:
    certified: bool
    unsafe_face_count: int
    unsafe_edge_count: int
    min_area_ratio: float
    min_sigma: float
    max_sigma: float
    max_edge_strain: float
    legacy_normal_reversal_count: int
    unsafe_faces: np.ndarray
    unsafe_edges: np.ndarray


@dataclass(frozen=True)
class ProjectionResult:
    delta: np.ndarray
    weights: np.ndarray
    certified: bool
    absolute_certified: bool
    status: str
    retention: float
    iterations: int
    report: SafetyReport


@dataclass(frozen=True)
class RidgeFoldProjectionResult:
    delta: np.ndarray
    weights: np.ndarray
    certified: bool
    status: str
    retention: float
    iterations: int
    ridge_fold_count: int
    projected_fold_count: int
    new_vs_ridge_fold_count: int


def mesh_edges(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    raw = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    raw.sort(axis=1)
    return np.unique(raw, axis=0)


def triangle_distortion(source: np.ndarray, edited: np.ndarray, faces: np.ndarray) -> TriangleDistortion:
    source = np.asarray(source, dtype=np.float64)
    edited = np.asarray(edited, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    source_tri = source[faces]
    edited_tri = edited[faces]

    source_e1 = source_tri[:, 1] - source_tri[:, 0]
    source_e2 = source_tri[:, 2] - source_tri[:, 0]
    edited_e1 = edited_tri[:, 1] - edited_tri[:, 0]
    edited_e2 = edited_tri[:, 2] - edited_tri[:, 0]

    source_l1 = np.linalg.norm(source_e1, axis=1)
    source_axis = source_e1 / np.maximum(source_l1[:, None], 1e-12)
    source_x2 = np.sum(source_e2 * source_axis, axis=1)
    source_y2 = np.sqrt(np.maximum(np.sum(source_e2 * source_e2, axis=1) - source_x2**2, 0.0))

    inv00 = 1.0 / np.maximum(source_l1, 1e-12)
    inv11 = 1.0 / np.maximum(source_y2, 1e-12)
    inv01 = -source_x2 * inv00 * inv11
    jacobian_0 = edited_e1 * inv00[:, None]
    jacobian_1 = edited_e1 * inv01[:, None] + edited_e2 * inv11[:, None]
    gram00 = np.sum(jacobian_0 * jacobian_0, axis=1)
    gram01 = np.sum(jacobian_0 * jacobian_1, axis=1)
    gram11 = np.sum(jacobian_1 * jacobian_1, axis=1)
    trace = gram00 + gram11
    determinant = np.maximum(gram00 * gram11 - gram01**2, 0.0)
    discriminant = np.sqrt(np.maximum(trace**2 - 4.0 * determinant, 0.0))
    eigen_max = np.maximum(0.5 * (trace + discriminant), 0.0)
    eigen_min = np.maximum(0.5 * (trace - discriminant), 0.0)

    source_cross = np.cross(source_e1, source_e2)
    edited_cross = np.cross(edited_e1, edited_e2)
    source_area2 = np.linalg.norm(source_cross, axis=1)
    edited_area2 = np.linalg.norm(edited_cross, axis=1)
    normal_cosine = np.sum(source_cross * edited_cross, axis=1) / np.maximum(source_area2 * edited_area2, 1e-12)
    return TriangleDistortion(
        area_ratio=edited_area2 / np.maximum(source_area2, 1e-12),
        sigma_min=np.sqrt(eigen_min),
        sigma_max=np.sqrt(eigen_max),
        normal_cosine=np.clip(normal_cosine, -1.0, 1.0),
    )


def signed_fold_indicator(source: np.ndarray, edited: np.ndarray, faces: np.ndarray, eps: float = 1e-12):
    """Return per-face signed 2D Jacobian determinant in the SOURCE tangent frame.

    The unsigned ``area_ratio`` / Gram singular values used elsewhere are always
    non-negative and therefore blind to orientation reversal. Here we build an
    orthonormal frame ``(u, v)`` inside each source triangle's plane and express
    the edited edges in that frame, then take the signed determinant of the 2x2
    edge matrix.

    ``det > 0``  keeps the source orientation,
    ``det <= 0`` is an orientation reversal (true foldover),
    and a face is degenerate when ``det <= eps * src_area2``.

    Returns ``(det, src_area2)`` where ``src_area2`` is twice the source triangle
    area (the magnitude of the un-normalised source normal).
    """
    source = np.asarray(source, dtype=np.float64)
    edited = np.asarray(edited, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    s = source[faces]
    e = edited[faces]
    se1 = s[:, 1] - s[:, 0]
    se2 = s[:, 2] - s[:, 0]
    ee1 = e[:, 1] - e[:, 0]
    ee2 = e[:, 2] - e[:, 0]
    l1 = np.linalg.norm(se1, axis=1, keepdims=True)
    u = se1 / np.maximum(l1, eps)
    n = np.cross(se1, se2)
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    w = n / np.maximum(nn, eps)
    v = np.cross(w, u)
    a = np.sum(ee1 * u, axis=1)
    b = np.sum(ee1 * v, axis=1)
    c = np.sum(ee2 * u, axis=1)
    d = np.sum(ee2 * v, axis=1)
    det = a * d - b * c
    src_area2 = nn[:, 0]
    return det, src_area2


def new_and_missed_folds(pred_fold: np.ndarray, target_fold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Baseline-relative foldover bookkeeping on top of a validated detector.

    Given two per-face orientation-reversal masks (e.g. ``det < 0`` from
    :func:`signed_fold_indicator`) for a method's PREDICTION and for the true
    i2i TARGET, return ``(new_fold, missed_fold)`` where

    * ``new_fold``    -- faces the prediction reverses that the target does NOT
      (the model-induced *new* foldover; the ``n_new`` analogue), and
    * ``missed_fold`` -- faces the target reverses that the prediction fails to
      reproduce.

    This decouples real intrinsic folds (a person->person nose difference
    reverses ~2% of faces by itself) from artefacts the model introduces, so a
    method is not penalised for faithfully reproducing the ground truth.
    """
    pred_fold = np.asarray(pred_fold, dtype=bool)
    target_fold = np.asarray(target_fold, dtype=bool)
    if pred_fold.shape != target_fold.shape:
        raise ValueError("pred_fold and target_fold must have the same shape")
    new_fold = pred_fold & ~target_fold
    missed_fold = target_fold & ~pred_fold
    return new_fold, missed_fold


def edge_strain(source: np.ndarray, edited: np.ndarray, edges: np.ndarray) -> np.ndarray:
    source_length = np.linalg.norm(source[edges[:, 0]] - source[edges[:, 1]], axis=1)
    edited_length = np.linalg.norm(edited[edges[:, 0]] - edited[edges[:, 1]], axis=1)
    return np.abs(edited_length - source_length) / np.maximum(source_length, 1e-12)


def _absolute_unsafe(
    distortion: TriangleDistortion, strain: np.ndarray, thresholds: SafetyThresholds
) -> tuple[np.ndarray, np.ndarray]:
    unsafe_faces = (
        (distortion.area_ratio < thresholds.min_area_ratio)
        | (distortion.sigma_min < thresholds.min_sigma)
        | (distortion.sigma_max > thresholds.max_sigma)
    )
    if thresholds.max_normal_rotation_deg is not None:
        cosine_limit = np.cos(np.deg2rad(float(thresholds.max_normal_rotation_deg)))
        unsafe_faces |= distortion.normal_cosine < cosine_limit
    return unsafe_faces, strain > thresholds.max_edge_strain


def _relative_unsafe(
    current_distortion: TriangleDistortion,
    current_strain: np.ndarray,
    base_distortion: TriangleDistortion,
    base_strain: np.ndarray,
    thresholds: SafetyThresholds,
) -> tuple[np.ndarray, np.ndarray]:
    current_faces, current_edges = _absolute_unsafe(current_distortion, current_strain, thresholds)
    base_faces, base_edges = _absolute_unsafe(base_distortion, base_strain, thresholds)
    worsened_base_face = base_faces & (
        (current_distortion.area_ratio + 1e-10 < base_distortion.area_ratio)
        | (current_distortion.sigma_min + 1e-10 < base_distortion.sigma_min)
        | (current_distortion.sigma_max > base_distortion.sigma_max + 1e-10)
    )
    if thresholds.max_normal_rotation_deg is not None:
        worsened_base_face |= base_faces & (
            current_distortion.normal_cosine + 1e-10 < base_distortion.normal_cosine
        )
    unsafe_faces = (current_faces & ~base_faces) | worsened_base_face
    unsafe_edges = (current_edges & ~base_edges) | (base_edges & (current_strain > base_strain + 1e-10))
    return unsafe_faces, unsafe_edges


def evaluate_mesh_safety(
    source: np.ndarray,
    edited: np.ndarray,
    faces: np.ndarray,
    thresholds: SafetyThresholds,
    baseline_edited: np.ndarray | None = None,
    edges: np.ndarray | None = None,
) -> SafetyReport:
    faces = np.asarray(faces, dtype=np.int64)
    edges = mesh_edges(faces) if edges is None else np.asarray(edges, dtype=np.int64)
    distortion = triangle_distortion(source, edited, faces)
    strain = edge_strain(source, edited, edges)
    absolute_faces, absolute_edges = _absolute_unsafe(distortion, strain, thresholds)
    unsafe_faces, unsafe_edges = absolute_faces, absolute_edges
    if baseline_edited is not None and thresholds.allow_baseline_violations:
        base_distortion = triangle_distortion(source, baseline_edited, faces)
        base_strain = edge_strain(source, baseline_edited, edges)
        unsafe_faces, unsafe_edges = _relative_unsafe(
            distortion, strain, base_distortion, base_strain, thresholds
        )
    return SafetyReport(
        certified=not bool(np.any(unsafe_faces) or np.any(unsafe_edges)),
        unsafe_face_count=int(np.sum(unsafe_faces)),
        unsafe_edge_count=int(np.sum(unsafe_edges)),
        min_area_ratio=float(np.min(distortion.area_ratio)),
        min_sigma=float(np.min(distortion.sigma_min)),
        max_sigma=float(np.max(distortion.sigma_max)),
        max_edge_strain=float(np.max(strain)),
        legacy_normal_reversal_count=int(np.sum(distortion.normal_cosine < 0.0)),
        unsafe_faces=np.asarray(unsafe_faces, dtype=bool),
        unsafe_edges=np.asarray(unsafe_edges, dtype=bool),
    )


def neutralize_handle_residual(
    residual: np.ndarray, handles: np.ndarray, interpolation_basis: np.ndarray | None = None
) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64).copy()
    handles = np.asarray(handles, dtype=np.int64)
    if interpolation_basis is not None:
        basis = np.asarray(interpolation_basis, dtype=np.float64)
        residual += np.einsum("vk,kd->vd", basis, -residual[handles])
    residual[handles] = 0.0
    return residual


def _neighbor_average_min(weights: np.ndarray, edges: np.ndarray, steps: int) -> np.ndarray:
    result = weights.copy()
    edge0, edge1 = edges[:, 0], edges[:, 1]
    for _ in range(max(0, int(steps))):
        total = result.copy()
        count = np.ones(len(result), dtype=np.float64)
        np.add.at(total, edge0, result[edge1])
        np.add.at(total, edge1, result[edge0])
        np.add.at(count, edge0, 1.0)
        np.add.at(count, edge1, 1.0)
        result = np.minimum(result, total / count)
    return result


def _retention(delta: np.ndarray, base_delta: np.ndarray, residual: np.ndarray) -> float:
    denominator = float(np.linalg.norm(residual))
    if denominator <= 1e-12:
        return 1.0
    return float(np.clip(np.linalg.norm(delta - base_delta) / denominator, 0.0, 1.0))


def project_residual_no_new_ridge_folds(
    source: np.ndarray,
    ridge_delta: np.ndarray,
    residual: np.ndarray,
    faces: np.ndarray,
    handles: np.ndarray,
    *,
    attenuation: float = 0.5,
    smoothing_steps: int = 0,
    max_iterations: int = 32,
    uniform_steps: int = 1001,
) -> RidgeFoldProjectionResult:
    """Project a residual so its final fold set is a subset of Ridge's.

    The certificate is target-independent once the supplied Ridge delta already
    contains the known handle controls. Consequently, for any fixed target fold
    mask, target-relative new folds cannot exceed those of matched Ridge.
    """
    source = np.asarray(source, dtype=np.float64)
    ridge_delta = np.asarray(ridge_delta, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    handles = np.asarray(handles, dtype=np.int64)
    if source.shape != ridge_delta.shape or source.shape != residual.shape:
        raise ValueError("source, ridge_delta and residual must have identical [vertices, 3] shapes")
    if not 0.0 < float(attenuation) < 1.0:
        raise ValueError("attenuation must lie strictly between zero and one")
    if max_iterations < 0 or uniform_steps < 2 or smoothing_steps < 0:
        raise ValueError("Invalid fold-projection iteration configuration")

    residual = neutralize_handle_residual(residual, handles)
    ridge_edited = source + ridge_delta
    ridge_fold = signed_fold_indicator(source, ridge_edited, faces)[0] < 0.0
    edges = mesh_edges(faces)
    weights = np.ones(len(source), dtype=np.float64)
    weights[handles] = 0.0

    for iteration in range(max_iterations + 1):
        delta = ridge_delta + weights[:, None] * residual
        projected_fold = signed_fold_indicator(source, source + delta, faces)[0] < 0.0
        new_vs_ridge = projected_fold & ~ridge_fold
        if not np.any(new_vs_ridge):
            return RidgeFoldProjectionResult(
                delta=delta,
                weights=weights,
                certified=True,
                status="local_fold_subset_certified",
                retention=_retention(delta, ridge_delta, residual),
                iterations=iteration,
                ridge_fold_count=int(np.sum(ridge_fold)),
                projected_fold_count=int(np.sum(projected_fold)),
                new_vs_ridge_fold_count=0,
            )
        if iteration == max_iterations:
            break
        bad_vertices = np.unique(faces[new_vs_ridge].reshape(-1))
        weights[bad_vertices] *= float(attenuation)
        if smoothing_steps:
            weights = _neighbor_average_min(weights, edges, smoothing_steps)
        weights[handles] = 0.0

    for alpha in np.linspace(1.0, 0.0, uniform_steps):
        uniform_weights = np.full(len(source), float(alpha), dtype=np.float64)
        uniform_weights[handles] = 0.0
        delta = ridge_delta + uniform_weights[:, None] * residual
        projected_fold = signed_fold_indicator(source, source + delta, faces)[0] < 0.0
        new_vs_ridge = projected_fold & ~ridge_fold
        if not np.any(new_vs_ridge):
            return RidgeFoldProjectionResult(
                delta=delta,
                weights=uniform_weights,
                certified=True,
                status="local_failed_uniform_fold_subset_certified",
                retention=_retention(delta, ridge_delta, residual),
                iterations=max_iterations,
                ridge_fold_count=int(np.sum(ridge_fold)),
                projected_fold_count=int(np.sum(projected_fold)),
                new_vs_ridge_fold_count=0,
            )
    raise AssertionError("alpha=0 must reproduce Ridge and certify the fold-subset fallback")


def uniform_safe_projection(
    source: np.ndarray,
    base_delta: np.ndarray,
    residual: np.ndarray,
    faces: np.ndarray,
    handles: np.ndarray,
    thresholds: SafetyThresholds,
) -> ProjectionResult:
    source = np.asarray(source, dtype=np.float64)
    base_delta = np.asarray(base_delta, dtype=np.float64)
    residual = neutralize_handle_residual(residual, handles)
    baseline = source + base_delta
    edges = mesh_edges(faces)
    for alpha in np.linspace(1.0, 0.0, max(2, thresholds.uniform_steps)):
        delta = base_delta + float(alpha) * residual
        report = evaluate_mesh_safety(source, source + delta, faces, thresholds, baseline, edges)
        if report.certified:
            absolute = evaluate_mesh_safety(source, source + delta, faces, thresholds, edges=edges).certified
            return ProjectionResult(
                delta=delta,
                weights=np.full(len(source), float(alpha)),
                certified=True,
                absolute_certified=absolute,
                status="uniform_certified",
                retention=_retention(delta, base_delta, residual),
                iterations=0,
                report=report,
            )
    report = evaluate_mesh_safety(source, baseline, faces, thresholds, baseline, edges)
    return ProjectionResult(
        delta=base_delta.copy(),
        weights=np.zeros(len(source)),
        certified=report.certified,
        absolute_certified=evaluate_mesh_safety(source, baseline, faces, thresholds, edges=edges).certified,
        status="base_fallback",
        retention=0.0,
        iterations=0,
        report=report,
    )


def project_residual(
    source: np.ndarray,
    base_delta: np.ndarray,
    residual: np.ndarray,
    faces: np.ndarray,
    handles: np.ndarray,
    thresholds: SafetyThresholds,
) -> ProjectionResult:
    source = np.asarray(source, dtype=np.float64)
    base_delta = np.asarray(base_delta, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    handles = np.asarray(handles, dtype=np.int64)
    residual = neutralize_handle_residual(residual, handles)
    baseline = source + base_delta
    edges = mesh_edges(faces)
    weights = np.ones(len(source), dtype=np.float64)
    weights[handles] = 0.0

    for iteration in range(thresholds.max_iterations + 1):
        delta = base_delta + weights[:, None] * residual
        report = evaluate_mesh_safety(source, source + delta, faces, thresholds, baseline, edges)
        if report.certified:
            absolute = evaluate_mesh_safety(source, source + delta, faces, thresholds, edges=edges).certified
            return ProjectionResult(
                delta=delta,
                weights=weights,
                certified=True,
                absolute_certified=absolute,
                status="local_certified",
                retention=_retention(delta, base_delta, residual),
                iterations=iteration,
                report=report,
            )
        if iteration == thresholds.max_iterations:
            break
        bad_vertices = []
        if report.unsafe_face_count:
            bad_vertices.append(faces[report.unsafe_faces].reshape(-1))
        if report.unsafe_edge_count:
            bad_vertices.append(edges[report.unsafe_edges].reshape(-1))
        if not bad_vertices:
            break
        bad = np.unique(np.concatenate(bad_vertices))
        weights[bad] *= float(thresholds.attenuation)
        weights = _neighbor_average_min(weights, edges, thresholds.smoothing_steps)
        weights[handles] = 0.0

    fallback = uniform_safe_projection(source, base_delta, residual, faces, handles, thresholds)
    return ProjectionResult(
        delta=fallback.delta,
        weights=fallback.weights,
        certified=fallback.certified,
        absolute_certified=fallback.absolute_certified,
        status="local_failed_" + fallback.status,
        retention=fallback.retention,
        iterations=thresholds.max_iterations,
        report=fallback.report,
    )


def _point_in_triangle_2d(point: np.ndarray, triangle: np.ndarray, eps: float) -> bool:
    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    signs = np.asarray([orient(triangle[i], triangle[(i + 1) % 3], point) for i in range(3)])
    return bool(np.all(signs >= -eps) or np.all(signs <= eps))


def _segments_intersect_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    ):
        return True
    return False


def _segment_triangle_intersection(start: np.ndarray, end: np.ndarray, triangle: np.ndarray, eps: float) -> bool:
    direction = end - start
    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]
    pvec = np.cross(direction, edge2)
    det = float(np.dot(edge1, pvec))
    if abs(det) <= eps:
        return False
    inv_det = 1.0 / det
    tvec = start - triangle[0]
    u = float(np.dot(tvec, pvec) * inv_det)
    if u < -eps or u > 1.0 + eps:
        return False
    qvec = np.cross(tvec, edge1)
    v = float(np.dot(direction, qvec) * inv_det)
    if v < -eps or u + v > 1.0 + eps:
        return False
    t = float(np.dot(edge2, qvec) * inv_det)
    return -eps <= t <= 1.0 + eps


def _triangles_intersect(first: np.ndarray, second: np.ndarray, eps: float = 1e-9) -> bool:
    for triangle, other in ((first, second), (second, first)):
        for index in range(3):
            if _segment_triangle_intersection(triangle[index], triangle[(index + 1) % 3], other, eps):
                return True
    normal = np.cross(first[1] - first[0], first[2] - first[0])
    if np.linalg.norm(normal) <= eps:
        return False
    distances = np.abs((second - first[0]) @ normal) / np.linalg.norm(normal)
    if np.max(distances) > eps:
        return False
    axis = int(np.argmax(np.abs(normal)))
    keep = [value for value in range(3) if value != axis]
    first2 = first[:, keep]
    second2 = second[:, keep]
    for i in range(3):
        for j in range(3):
            if _segments_intersect_2d(first2[i], first2[(i + 1) % 3], second2[j], second2[(j + 1) % 3], eps):
                return True
    return _point_in_triangle_2d(first2[0], second2, eps) or _point_in_triangle_2d(second2[0], first2, eps)


def count_self_intersections(
    vertices: np.ndarray,
    faces: np.ndarray,
    stop_after: int | None = None,
    baseline_vertices: np.ndarray | None = None,
) -> int:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    triangles = vertices[faces]
    baseline_triangles = None
    if baseline_vertices is not None:
        baseline_triangles = np.asarray(baseline_vertices, dtype=np.float64)[faces]
    lower = triangles.min(axis=1)
    upper = triangles.max(axis=1)
    nonzero_size = np.linalg.norm(upper - lower, axis=1)
    positive_size = nonzero_size[nonzero_size > 1e-12]
    bbox_size = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    cell_size = max(
        float(np.percentile(positive_size, 75) * 2.0) if len(positive_size) else 0.0,
        bbox_size / 128.0,
        1e-9,
    )
    origin = vertices.min(axis=0) - 1e-9
    count = 0
    grid: dict[tuple[int, int, int], list[int]] = {}
    for current in range(len(faces)):
        first_cell = np.floor((lower[current] - origin) / cell_size).astype(np.int64)
        last_cell = np.floor((upper[current] - origin) / cell_size).astype(np.int64)
        cells = [
            (x, y, z)
            for x in range(int(first_cell[0]), int(last_cell[0]) + 1)
            for y in range(int(first_cell[1]), int(last_cell[1]) + 1)
            for z in range(int(first_cell[2]), int(last_cell[2]) + 1)
        ]
        candidates: set[int] = set()
        for cell in cells:
            candidates.update(grid.get(cell, ()))
        current_vertices = set(int(value) for value in faces[current])
        for other in candidates:
            if current_vertices.intersection(int(value) for value in faces[other]):
                continue
            if np.any(upper[other] < lower[current] - 1e-9) or np.any(
                upper[current] < lower[other] - 1e-9
            ):
                continue
            if _triangles_intersect(triangles[current], triangles[other]):
                if baseline_triangles is not None and _triangles_intersect(
                    baseline_triangles[current], baseline_triangles[other]
                ):
                    continue
                count += 1
                if stop_after is not None and count >= stop_after:
                    return count
        for cell in cells:
            grid.setdefault(cell, []).append(current)
    return count
