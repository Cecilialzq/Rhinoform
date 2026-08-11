from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from rhinoform.safe_fusion import (  # noqa: E402
    SafetyThresholds,
    count_self_intersections,
    evaluate_mesh_safety,
    neutralize_handle_residual,
    new_and_missed_folds,
    project_residual,
    project_residual_no_new_ridge_folds,
    signed_fold_indicator,
    triangle_distortion,
    uniform_safe_projection,
)


class SafeFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        self.faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)

    def test_rigid_transform_has_unit_triangle_distortion(self) -> None:
        theta = np.deg2rad(37.0)
        rotation = np.asarray(
            [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]
        )
        edited = self.source @ rotation.T + np.asarray([3.0, -2.0, 4.0])
        distortion = triangle_distortion(self.source, edited, self.faces)
        np.testing.assert_allclose(distortion.area_ratio, 1.0, atol=1e-10)
        np.testing.assert_allclose(distortion.sigma_min, 1.0, atol=1e-10)
        np.testing.assert_allclose(distortion.sigma_max, 1.0, atol=1e-10)

    def test_collapsed_triangle_fails_safety_certificate(self) -> None:
        edited = self.source.copy()
        edited[2] = [0.5, 0.0, 0.0]
        report = evaluate_mesh_safety(
            self.source,
            edited,
            self.faces,
            SafetyThresholds(min_area_ratio=0.05, min_sigma=0.05, max_sigma=4.0, max_edge_strain=3.0),
        )
        self.assertFalse(report.certified)
        self.assertGreater(report.unsafe_face_count, 0)

    def test_projection_preserves_handles_and_certifies_output(self) -> None:
        base = np.zeros_like(self.source)
        proposal = np.zeros_like(self.source)
        proposal[2] = [0.7, -1.2, 0.0]
        handles = np.asarray([0, 1, 3], dtype=np.int64)
        residual = neutralize_handle_residual(proposal - base, handles)
        result = project_residual(
            self.source,
            base,
            residual,
            self.faces,
            handles,
            SafetyThresholds(min_area_ratio=0.1, min_sigma=0.1, max_sigma=3.0, max_edge_strain=2.0),
        )
        self.assertTrue(result.certified)
        np.testing.assert_allclose(result.delta[handles], base[handles], atol=1e-12)
        self.assertGreaterEqual(result.retention, 0.0)
        self.assertLessEqual(result.retention, 1.0 + 1e-9)

    def test_local_projection_retains_more_than_uniform_scaling(self) -> None:
        source = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0], [2, 1, 0]], dtype=np.float64
        )
        faces = np.asarray([[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3]], dtype=np.int64)
        base = np.zeros_like(source)
        residual = np.zeros_like(source)
        residual[2] = [0.8, -1.1, 0.0]
        residual[4:] = [0.0, 0.35, 0.0]
        handles = np.asarray([0, 1, 3], dtype=np.int64)
        thresholds = SafetyThresholds(
            min_area_ratio=0.08,
            min_sigma=0.08,
            max_sigma=3.0,
            max_edge_strain=2.0,
            smoothing_steps=0,
        )
        local = project_residual(source, base, residual, faces, handles, thresholds)
        uniform = uniform_safe_projection(source, base, residual, faces, handles, thresholds)
        self.assertTrue(local.certified)
        self.assertTrue(uniform.certified)
        self.assertGreater(local.weights[5], uniform.weights[5])
        self.assertGreater(np.linalg.norm(local.delta[5]), np.linalg.norm(uniform.delta[5]))

    def test_ridge_fold_projection_removes_new_fold_and_keeps_disconnected_safe_residual(self) -> None:
        source = np.asarray(
            [
                [0, 0, 0], [1, 0, 0], [0, 1, 0],
                [3, 0, 0], [4, 0, 0], [3, 1, 0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        ridge = np.zeros_like(source)
        residual = np.zeros_like(source)
        residual[2] = [0.0, -2.0, 0.0]  # unsafe: flips face 0
        residual[3:] = [0.0, 0.25, 0.0]  # safe motion on disconnected face 1

        result = project_residual_no_new_ridge_folds(
            source,
            ridge,
            residual,
            faces,
            handles=np.asarray([], dtype=np.int64),
            attenuation=0.5,
            max_iterations=16,
        )

        ridge_fold = signed_fold_indicator(source, source + ridge, faces)[0] < 0.0
        projected_fold = signed_fold_indicator(source, source + result.delta, faces)[0] < 0.0
        self.assertTrue(result.certified)
        self.assertFalse(np.any(projected_fold & ~ridge_fold))
        np.testing.assert_allclose(result.delta[3:], residual[3:], atol=1e-12)

    def test_ridge_fold_projection_allows_preexisting_ridge_fold_but_no_additional_one(self) -> None:
        source = np.asarray(
            [
                [0, 0, 0], [1, 0, 0], [0, 1, 0],
                [3, 0, 0], [4, 0, 0], [3, 1, 0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        ridge = np.zeros_like(source)
        ridge[2] = [0.0, -2.0, 0.0]  # Ridge already folds face 0.
        residual = np.zeros_like(source)
        residual[5] = [0.0, -2.0, 0.0]  # Proposal would additionally fold face 1.

        result = project_residual_no_new_ridge_folds(
            source,
            ridge,
            residual,
            faces,
            handles=np.asarray([], dtype=np.int64),
        )

        ridge_fold = signed_fold_indicator(source, source + ridge, faces)[0] < 0.0
        projected_fold = signed_fold_indicator(source, source + result.delta, faces)[0] < 0.0
        self.assertEqual(np.flatnonzero(ridge_fold).tolist(), [0])
        self.assertEqual(np.flatnonzero(projected_fold).tolist(), [0])
        self.assertEqual(result.new_vs_ridge_fold_count, 0)

    def test_self_intersection_detects_crossing_nonadjacent_triangles(self) -> None:
        vertices = np.asarray(
            [
                [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0],
                [0.0, -0.5, -1.0], [0.0, -0.5, 1.0], [0.0, 0.8, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        self.assertEqual(count_self_intersections(vertices, faces), 1)
        vertices[3:, 0] += 4.0
        self.assertEqual(count_self_intersections(vertices, faces), 0)

    def test_relative_self_intersection_excludes_preexisting_contact(self) -> None:
        vertices = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.2, -1], [0.2, 0.2, 1], [1, 1, 0]],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        self.assertEqual(count_self_intersections(vertices, faces, baseline_vertices=vertices), 0)

    def test_relative_self_intersection_detects_new_contact(self) -> None:
        baseline = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 2, -1], [2, 2, 1], [3, 3, 0]],
            dtype=np.float64,
        )
        edited = baseline.copy()
        edited[3:] = np.asarray([[0.2, 0.2, -1], [0.2, 0.2, 1], [1, 1, 0]])
        faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        self.assertEqual(count_self_intersections(edited, faces, baseline_vertices=baseline), 1)


class SignedFoldIndicatorTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(20260615)
        # An irregular but well-formed patch of triangles in 3D.
        self.source = rng.standard_normal((12, 3))
        self.faces = np.asarray(
            [[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4], [4, 5, 6], [5, 7, 6],
             [6, 7, 8], [7, 9, 8], [8, 9, 10], [9, 11, 10]],
            dtype=np.int64,
        )

    def _bounded_rotation(self, seed: int, max_deg: float = 45.0) -> np.ndarray:
        # A rotation by <90 deg keeps every face normal in the same hemisphere,
        # which is the orientation-preserving regime that handle-based editing
        # operates in. (A near-180 deg global tumble legitimately reverses the
        # source-frame orientation and is out of scope for deformation edits.)
        rng = np.random.default_rng(seed)
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        angle = np.deg2rad(rng.uniform(5.0, max_deg))
        k = np.asarray([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)

    def test_rigid_transform_all_positive(self) -> None:
        for seed in range(8):
            rotation = self._bounded_rotation(seed)
            edited = self.source @ rotation.T + np.asarray([2.0, -3.0, 1.5])
            det, _ = signed_fold_indicator(self.source, edited, self.faces)
            self.assertTrue(np.all(det > 0))

    def test_uniform_scaling_all_positive(self) -> None:
        for scale in (0.3, 2.5):
            edited = self.source * scale
            det, area2 = signed_fold_indicator(self.source, edited, self.faces)
            self.assertTrue(np.all(det > 0))
            np.testing.assert_allclose(det, (scale ** 2) * area2, rtol=1e-9, atol=1e-9)

    def test_inplane_reflection_makes_det_negative(self) -> None:
        # On a planar patch an in-plane reflection (y -> -y) turns every CCW
        # triangle clockwise, reversing the source-frame orientation -> det < 0.
        rng = np.random.default_rng(7)
        xy = rng.uniform(-1.0, 1.0, size=(9, 2))
        source = np.concatenate([xy, np.zeros((9, 1))], axis=1)
        faces = np.asarray(
            [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8], [0, 4, 8], [1, 5, 7]],
            dtype=np.int64,
        )
        # Force consistent counter-clockwise winding in the source plane.
        e1 = source[faces[:, 1]] - source[faces[:, 0]]
        e2 = source[faces[:, 2]] - source[faces[:, 0]]
        ccw = (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) > 0
        faces[~ccw] = faces[~ccw][:, [0, 2, 1]]
        edited = source.copy()
        edited[:, 1] *= -1.0
        det, _ = signed_fold_indicator(source, edited, faces)
        det_src, _ = signed_fold_indicator(source, source.copy(), faces)
        self.assertTrue(np.all(det < 0))
        self.assertTrue(np.all(det_src > 0))

    def test_local_vertex_swap_flips_only_incident_faces(self) -> None:
        # Swapping two vertices of a single triangle reverses only that triangle.
        edited = self.source.copy()
        target_face = 0
        i, j = int(self.faces[target_face, 1]), int(self.faces[target_face, 2])
        edited[[i, j]] = edited[[j, i]]
        det, _ = signed_fold_indicator(self.source, edited, self.faces)
        incident = np.any(np.isin(self.faces, [i, j]), axis=1)
        # Faces touching the swapped vertices change; the target face must flip.
        self.assertLess(det[target_face], 0.0)
        self.assertTrue(np.all(det[~incident] > 0))

    def test_degenerate_face_detected_by_relative_threshold(self) -> None:
        eps = 1e-12
        edited = self.source.copy()
        # Collapse vertex 2 onto the edge (0,1) of face 0 -> zero area.
        edited[2] = 0.5 * (edited[0] + edited[1])
        det, area2 = signed_fold_indicator(self.source, edited, self.faces)
        degenerate = det <= eps * area2
        self.assertTrue(degenerate[0])

    def test_signed_indicator_catches_fold_that_area_ratio_misses(self) -> None:
        # Reflect a single triangle's apex through the shared edge: area is
        # preserved (area_ratio == 1) but orientation reverses.
        source = np.asarray([[0, 0, 0], [1, 0, 0], [0.5, 1.0, 0.0]], dtype=np.float64)
        faces = np.asarray([[0, 1, 2]], dtype=np.int64)
        edited = source.copy()
        edited[2] = [0.5, -1.0, 0.0]  # mirror apex across the x-axis edge
        distortion = triangle_distortion(source, edited, faces)
        det, _ = signed_fold_indicator(source, edited, faces)
        np.testing.assert_allclose(distortion.area_ratio, 1.0, atol=1e-9)
        self.assertLess(det[0], 0.0)


class NewFoldBookkeepingTests(unittest.TestCase):
    def test_new_flip_counts_only_model_introduced_faces(self) -> None:
        # target reverses face X={0}; the model reverses X={0} and Y={2}.
        # new-flip must be exactly {Y}; the reproduced intrinsic fold {0} is not new.
        target_fold = np.asarray([True, False, False])
        pred_fold = np.asarray([True, False, True])
        new_fold, missed_fold = new_and_missed_folds(pred_fold, target_fold)
        self.assertEqual(np.flatnonzero(new_fold).tolist(), [2])
        self.assertEqual(np.flatnonzero(missed_fold).tolist(), [])

    def test_missed_flip_counts_unreproduced_target_folds(self) -> None:
        # target reverses {0,1}; model reverses only {0}. missed must be {1}.
        target_fold = np.asarray([True, True, False])
        pred_fold = np.asarray([True, False, False])
        new_fold, missed_fold = new_and_missed_folds(pred_fold, target_fold)
        self.assertEqual(np.flatnonzero(new_fold).tolist(), [])
        self.assertEqual(np.flatnonzero(missed_fold).tolist(), [1])

    def test_new_flip_end_to_end_through_detector(self) -> None:
        # Three independent in-plane triangles. The true target reverses triangle
        # X=0 (intrinsic fold); the model reproduces that fold AND adds a new one
        # on triangle Y=2. End-to-end through signed_fold_indicator, new-flip={2}.
        source = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0],      # tri 0 (X)
             [2, 0, 0], [3, 0, 0], [2, 1, 0],      # tri 1
             [4, 0, 0], [5, 0, 0], [4, 1, 0]],     # tri 2 (Y)
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64)
        target_delta = np.zeros_like(source)
        target_delta[2] = [0.0, -2.0, 0.0]                  # flip tri 0 apex below the edge
        pred_delta = target_delta.copy()
        pred_delta[8] = [0.0, -2.0, 0.0]                    # also flip tri 2 (model-induced)
        tgt_fold = signed_fold_indicator(source, source + target_delta, faces)[0] < 0.0
        pred_fold = signed_fold_indicator(source, source + pred_delta, faces)[0] < 0.0
        self.assertEqual(np.flatnonzero(tgt_fold).tolist(), [0])
        self.assertEqual(np.flatnonzero(pred_fold).tolist(), [0, 2])
        new_fold, missed_fold = new_and_missed_folds(pred_fold, tgt_fold)
        self.assertEqual(np.flatnonzero(new_fold).tolist(), [2])
        self.assertEqual(np.flatnonzero(missed_fold).tolist(), [])


if __name__ == "__main__":
    unittest.main()
