from __future__ import annotations

import unittest

import numpy as np


class LAMMResumeContractTests(unittest.TestCase):
    def test_resume_signature_changes_with_any_training_contract_input(self) -> None:
        from experiments.lamm.run_lamm_facescape import lamm_resume_signature

        common = {
            "stage": "ae",
            "model_config": {"dim": 512, "manipulation": False},
            "train_array_sha256": "train",
            "validation_array_sha256": "validation",
            "seed": 20260609,
            "epochs": 1500,
            "batch_size": 32,
            "eval_every": 25,
            "implementation_sha256": "implementation",
            "official_commit": "commit",
        }
        signature = lamm_resume_signature(**common)

        self.assertEqual(signature, lamm_resume_signature(**common))
        self.assertNotEqual(signature, lamm_resume_signature(**{**common, "batch_size": 16}))
        self.assertNotEqual(
            signature,
            lamm_resume_signature(**{**common, "implementation_sha256": "changed"}),
        )

    def test_validation_objective_is_mean_per_pair_free_vector_rmse(self) -> None:
        from experiments.lamm.run_lamm_facescape import free_pair_vector_rmse

        error = np.zeros((2, 3, 3), dtype=np.float64)
        error[0, 0, 0] = 6.0
        error[1, 0, 0] = 1.0
        free = np.asarray([True, True, False])

        values = free_pair_vector_rmse(error, free)

        np.testing.assert_allclose(values, [np.sqrt(36.0 / 2.0), np.sqrt(1.0 / 2.0)])
        self.assertNotAlmostEqual(float(np.mean(values)), float(np.sqrt(37.0 / 4.0)), places=6)


if __name__ == "__main__":
    unittest.main()
