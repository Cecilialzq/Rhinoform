from __future__ import annotations

import unittest

import numpy as np


class RBSRCalibrationTests(unittest.TestCase):
    def test_exact_controls_are_reapplied_after_calibration(self) -> None:
        from rhinoform.rbsr_calibration import enforce_exact_controls

        prediction = np.arange(24, dtype=np.float32).reshape(2, 12)
        controls = np.asarray([[100, 101, 102, 200, 201, 202], [300, 301, 302, 400, 401, 402]], dtype=np.float32)
        fixed = enforce_exact_controls(prediction, controls, np.asarray([1, 3]))

        vertices = fixed.reshape(2, 4, 3)
        self.assertTrue(np.array_equal(vertices[:, 1], controls.reshape(2, 2, 3)[:, 0]))
        self.assertTrue(np.array_equal(vertices[:, 3], controls.reshape(2, 2, 3)[:, 1]))
        self.assertTrue(np.array_equal(vertices[:, [0, 2]], prediction.reshape(2, 4, 3)[:, [0, 2]]))

    def test_zero_gate_is_exact_ridge_identity(self) -> None:
        from rhinoform.rbsr_calibration import fuse_calibrated_residual

        ridge = np.asarray(
            [[0.1, -0.2, 0.3, 0.4, 0.5, -0.6]], dtype=np.float32
        )
        cvae = ridge + np.asarray(
            [[0.7, 0.8, -0.9, 1.0, -1.1, 1.2]], dtype=np.float32
        )
        learned_gate = np.asarray([[0.2, 0.8]], dtype=np.float32)

        prediction, calibrated_gate = fuse_calibrated_residual(
            ridge,
            cvae,
            learned_gate,
            force_zero_gate=True,
        )

        self.assertTrue(np.array_equal(prediction, ridge))
        self.assertTrue(np.array_equal(calibrated_gate, np.zeros_like(learned_gate)))

    def test_selector_requires_lower_rmse_and_no_new_flip_increase(self) -> None:
        from rhinoform.rbsr_calibration import select_strict_operating_point

        ridge = {
            "label": "ridge_anchor",
            "roi_rmse": 1.0,
            "normal_flip_pct": 0.30,
        }
        candidates = [
            {"label": "unsafe_best_rmse", "roi_rmse": 0.90, "normal_flip_pct": 0.31},
            {"label": "safe_but_not_better", "roi_rmse": 1.00, "normal_flip_pct": 0.20},
            {"label": "strict_success", "roi_rmse": 0.98, "normal_flip_pct": 0.30},
            {"label": "strict_success_slower", "roi_rmse": 0.99, "normal_flip_pct": 0.25},
        ]

        selected, assessed = select_strict_operating_point(candidates, ridge)

        self.assertEqual(selected["label"], "strict_success")
        by_label = {row["label"]: row for row in assessed}
        self.assertFalse(by_label["unsafe_best_rmse"]["strictly_beats_ridge"])
        self.assertFalse(by_label["safe_but_not_better"]["strictly_beats_ridge"])
        self.assertTrue(by_label["strict_success"]["strictly_beats_ridge"])

    def test_reference_selector_requires_rmse_flip_and_strain_dominance(self) -> None:
        from rhinoform.rbsr_calibration import select_reference_dominating_operating_point

        reference = {
            "roi_rmse": 1.0,
            "normal_flip_pct": 0.8,
            "edge_strain_p95": 0.3,
        }
        candidates = [
            {
                "label": "rmse_only",
                "roi_rmse": 0.90,
                "normal_flip_pct": 0.81,
                "edge_strain_p95": 0.20,
            },
            {
                "label": "all_three",
                "roi_rmse": 0.95,
                "normal_flip_pct": 0.70,
                "edge_strain_p95": 0.25,
            },
            {
                "label": "fragile_rmse_optimum",
                "roi_rmse": 0.90,
                "normal_flip_pct": 0.78,
                "edge_strain_p95": 0.29,
            },
            {
                "label": "all_three_slower",
                "roi_rmse": 0.97,
                "normal_flip_pct": 0.60,
                "edge_strain_p95": 0.20,
            },
        ]

        selected, assessed = select_reference_dominating_operating_point(
            candidates, reference, relative_margin=0.01
        )

        self.assertEqual(selected["label"], "all_three")
        by_label = {row["label"]: row for row in assessed}
        self.assertFalse(by_label["rmse_only"]["dominates_reference_core"])
        self.assertTrue(by_label["all_three"]["dominates_reference_core"])
        self.assertTrue(by_label["fragile_rmse_optimum"]["dominates_reference_core"])
        self.assertGreater(
            by_label["all_three"]["minimum_core_relative_improvement"],
            by_label["fragile_rmse_optimum"]["minimum_core_relative_improvement"],
        )

    def test_reference_dominating_freeze_unlocks_only_matching_hash_chain(self) -> None:
        from rhinoform.rbsr_calibration import validate_test_unlock

        frozen = {
            "status": "FROZEN_REFERENCE_DOMINATING_VALIDATION_CALIBRATION",
            "test_access": True,
            "base_model_package_sha256": "base",
            "rbsr_package_sha256": "gate",
            "selected": {
                "label": "beta_0",
                "logit_offset": 0.0,
                "force_zero_gate": False,
                "dominates_reference_core": True,
            },
        }
        selected = validate_test_unlock(frozen, base_hash="base", rbsr_hash="gate")
        self.assertEqual(selected["label"], "beta_0")
        with self.assertRaises(ValueError):
            validate_test_unlock(frozen, base_hash="base", rbsr_hash="other")

    def test_test_unlock_rejects_failed_or_mismatched_validation_freeze(self) -> None:
        from rhinoform.rbsr_calibration import validate_test_unlock

        failed = {
            "status": "NO_FEASIBLE_STRICT_CALIBRATION",
            "test_access": False,
            "base_model_package_sha256": "base",
            "rbsr_package_sha256": "gate",
            "selected": None,
        }
        with self.assertRaises(ValueError):
            validate_test_unlock(failed, base_hash="base", rbsr_hash="gate")

        frozen = {
            "status": "FROZEN_STRICT_VALIDATION_CALIBRATION",
            "test_access": True,
            "base_model_package_sha256": "base",
            "rbsr_package_sha256": "gate",
            "selected": {
                "label": "beta_1",
                "logit_offset": 1.0,
                "force_zero_gate": False,
                "strictly_beats_ridge": True,
            },
        }
        with self.assertRaises(ValueError):
            validate_test_unlock(frozen, base_hash="different", rbsr_hash="gate")
        selected = validate_test_unlock(frozen, base_hash="base", rbsr_hash="gate")
        self.assertEqual(selected["label"], "beta_1")

    def test_ridge_identity_audit_allows_only_serialisation_scale_roundoff(self) -> None:
        from rhinoform.rbsr_calibration import ridge_reference_matches

        computed = {"roi_rmse": 1.0, "normal_flip_pct": 0.3, "edge_strain_p95": 0.2}
        roundtripped = {
            "roi_rmse": 1.0 + 2e-9,
            "normal_flip_pct": 0.3,
            "edge_strain_p95": 0.2 - 2e-9,
        }
        changed = {**roundtripped, "normal_flip_pct": 0.300001}

        self.assertTrue(ridge_reference_matches(computed, roundtripped))
        self.assertFalse(ridge_reference_matches(computed, changed))

    def test_projection_selector_requires_hard_certificate_as_well_as_metrics(self) -> None:
        from rhinoform.rbsr_calibration import select_certified_projection

        ridge = {"label": "ridge_anchor", "roi_rmse": 1.0, "normal_flip_pct": 0.30}
        candidates = [
            {
                "label": "uncertified",
                "roi_rmse": 0.90,
                "normal_flip_pct": 0.20,
                "certificate_rate": 0.999,
            },
            {
                "label": "certified",
                "roi_rmse": 0.95,
                "normal_flip_pct": 0.25,
                "certificate_rate": 1.0,
            },
        ]

        selected, assessed = select_certified_projection(candidates, ridge)

        self.assertEqual(selected["label"], "certified")
        by_label = {row["label"]: row for row in assessed}
        self.assertFalse(by_label["uncertified"]["strictly_beats_ridge"])
        self.assertTrue(by_label["certified"]["strictly_beats_ridge"])

    def test_projection_unlock_rejects_hash_or_certificate_mismatch(self) -> None:
        from rhinoform.rbsr_calibration import validate_projection_test_unlock

        frozen = {
            "status": "FROZEN_CERTIFIED_RIDGE_FOLD_PROJECTION",
            "test_access": True,
            "zero_gate_ridge_identity": {"passed": True},
            "base_model_package_sha256": "base",
            "rbsr_package_sha256": "gate",
            "projection_signature": "projection",
            "selected": {
                "label": "attenuation_0p75",
                "strictly_beats_ridge": True,
                "certificate_rate": 1.0,
            },
        }
        with self.assertRaises(ValueError):
            validate_projection_test_unlock(
                frozen,
                base_hash="base",
                rbsr_hash="different",
                projection_signature="projection",
            )
        selected = validate_projection_test_unlock(
            frozen,
            base_hash="base",
            rbsr_hash="gate",
            projection_signature="projection",
        )
        self.assertEqual(selected["label"], "attenuation_0p75")

    def test_projection_unlock_requires_exact_zero_gate_ridge_fallback(self) -> None:
        from rhinoform.rbsr_calibration import validate_projection_test_unlock

        frozen = {
            "status": "FROZEN_CERTIFIED_RIDGE_FOLD_PROJECTION",
            "test_access": True,
            "base_model_package_sha256": "base",
            "rbsr_package_sha256": "gate",
            "projection_signature": "projection",
            "selected": {
                "label": "attenuation_0p75",
                "strictly_beats_ridge": True,
                "certificate_rate": 1.0,
            },
        }
        with self.assertRaises(ValueError):
            validate_projection_test_unlock(
                frozen,
                base_hash="base",
                rbsr_hash="gate",
                projection_signature="projection",
            )


if __name__ == "__main__":
    unittest.main()
