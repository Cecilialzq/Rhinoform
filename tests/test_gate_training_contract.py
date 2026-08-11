from __future__ import annotations

import unittest

import numpy as np
import torch


class GateTrainingContractTests(unittest.TestCase):
    def test_infeasible_soft_gate_is_proposer_only(self) -> None:
        from rhinoform.train_rbsr_gate import gate_deployment_status

        self.assertEqual(
            gate_deployment_status({"feasible": False}, "primal_dual"),
            "RESIDUAL_PROPOSER_ONLY_REQUIRES_CERTIFIED_HARD_PROJECTION",
        )

    def test_projection_none_still_enforces_exact_landmark_controls(self) -> None:
        from rhinoform.train_rbsr_gate import project_handles

        prediction = torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]],
            dtype=torch.float32,
        )
        landmarks = torch.tensor([0, 2], dtype=torch.long)
        controls = torch.tensor(
            [[[10.0, 20.0, 30.0], [70.0, 80.0, 90.0]]],
            dtype=torch.float32,
        )

        projected = project_handles(prediction, controls, landmarks, projection_basis=None)

        self.assertTrue(torch.equal(projected[:, landmarks], controls))
        self.assertTrue(torch.equal(projected[:, 1], prediction[:, 1]))
        self.assertTrue(torch.equal(prediction[:, 0], torch.tensor([[1.0, 2.0, 3.0]])))

    def test_training_reconstruction_matches_free_vertex_vector_rmse(self) -> None:
        from rhinoform.train_rbsr_gate import strict_free_roi_rmse_loss

        target = torch.zeros((2, 4, 3), dtype=torch.float64)
        prediction = target.clone()
        # Landmark errors must be excluded entirely.
        prediction[0, 0] = torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64)
        prediction[1, 0] = torch.tensor([-100.0, -100.0, -100.0], dtype=torch.float64)
        # Free-vertex vector errors: pair 0 norms [3, 4, 0], pair 1 [0, 0, 6].
        prediction[0, 1, 0] = 3.0
        prediction[0, 2, 1] = 4.0
        prediction[1, 3, 2] = 6.0
        landmarks = torch.tensor([0], dtype=torch.long)

        actual = strict_free_roi_rmse_loss(prediction, target, landmarks, epsilon=0.0)
        expected = 0.5 * (np.sqrt((9.0 + 16.0 + 0.0) / 3.0) + np.sqrt(36.0 / 3.0))

        self.assertAlmostEqual(float(actual), float(expected), places=12)

    def test_proxy_rmse_excludes_landmarks_and_hard_fixes_them(self) -> None:
        from rhinoform.train_rbsr_gate import strict_free_pair_rmse

        target = np.zeros((1, 3, 3), dtype=np.float32)
        prediction = target.copy()
        prediction[0, 0] = 1000.0
        prediction[0, 1, 0] = 3.0
        prediction[0, 2, 1] = 4.0

        pair_rmse = strict_free_pair_rmse(prediction, target, np.asarray([0], dtype=np.int64))

        self.assertAlmostEqual(float(pair_rmse[0]), np.sqrt((9.0 + 16.0) / 2.0), places=6)


if __name__ == "__main__":
    unittest.main()
