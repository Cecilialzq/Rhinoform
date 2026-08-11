from __future__ import annotations

import unittest


class ConfirmationSplitTests(unittest.TestCase):
    def test_repartition_is_deterministic_disjoint_and_excludes_observed_test(self) -> None:
        from rhinoform.confirmation import blind_confirmation_partition

        parent = {
            "train_pool_ids": [str(value) for value in range(8)],
            "val_ids": ["8", "9"],
            "test_ids": ["10", "11"],
        }
        first = blind_confirmation_partition(parent, seed=20260809, val_size=2, test_size=2)
        second = blind_confirmation_partition(parent, seed=20260809, val_size=2, test_size=2)

        self.assertEqual(first, second)
        train = set(first["train_pool_ids"])
        validation = set(first["val_ids"])
        test = set(first["test_ids"])
        self.assertFalse(train & validation or train & test or validation & test)
        self.assertEqual(train | validation | test, {str(value) for value in range(10)})
        self.assertFalse({"10", "11"} & (train | validation | test))

    def test_repartition_fails_closed_when_requested_sizes_exhaust_pool(self) -> None:
        from rhinoform.confirmation import blind_confirmation_partition

        parent = {"train_pool_ids": ["1", "2"], "val_ids": ["3"], "test_ids": ["4"]}
        with self.assertRaises(ValueError):
            blind_confirmation_partition(parent, seed=1, val_size=2, test_size=1)

    def test_all_models_freeze_requires_exact_declared_hash_chain(self) -> None:
        from rhinoform.confirmation import validate_all_models_freeze

        payload = {
            "status": "ALL_MATCHED_MODELS_FROZEN_BEFORE_TEST",
            "test_access": False,
            "confirmation_policy_sha256": "policy",
            "split_manifest_sha256": "split",
            "rbsr_base_sha256": "base",
            "implementation_hashes": {"scripts/evaluate.py": "code"},
        }
        validate_all_models_freeze(
            payload,
            expected={
                "confirmation_policy_sha256": "policy",
                "split_manifest_sha256": "split",
                "rbsr_base_sha256": "base",
            },
            expected_implementations={"scripts/evaluate.py": "code"},
        )

        for key, value in (
            ("status", "wrong"),
            ("test_access", True),
            ("rbsr_base_sha256", "changed"),
        ):
            broken = dict(payload)
            broken[key] = value
            with self.assertRaises(ValueError):
                validate_all_models_freeze(
                    broken,
                    expected={
                        "confirmation_policy_sha256": "policy",
                        "split_manifest_sha256": "split",
                        "rbsr_base_sha256": "base",
                    },
                    expected_implementations={"scripts/evaluate.py": "code"},
                )

    def test_all_models_freeze_rejects_missing_or_empty_expected_hash(self) -> None:
        from rhinoform.confirmation import validate_all_models_freeze

        payload = {
            "status": "ALL_MATCHED_MODELS_FROZEN_BEFORE_TEST",
            "test_access": False,
            "confirmation_policy_sha256": "policy",
        }
        with self.assertRaises(ValueError):
            validate_all_models_freeze(payload, expected={"missing": "hash"})
        with self.assertRaises(ValueError):
            validate_all_models_freeze(payload, expected={"confirmation_policy_sha256": ""})

    def test_all_models_freeze_rejects_changed_evaluation_implementation(self) -> None:
        from rhinoform.confirmation import validate_all_models_freeze

        payload = {
            "status": "ALL_MATCHED_MODELS_FROZEN_BEFORE_TEST",
            "test_access": False,
            "confirmation_policy_sha256": "policy",
            "implementation_hashes": {"scripts/evaluate.py": "frozen-code"},
        }
        with self.assertRaises(ValueError):
            validate_all_models_freeze(
                payload,
                expected={"confirmation_policy_sha256": "policy"},
                expected_implementations={"scripts/evaluate.py": "changed-code"},
            )

    def test_frozen_classical_configuration_is_complete_and_positive(self) -> None:
        from rhinoform.confirmation import validate_frozen_classical_configuration

        valid = {
            "laplacian": {"handle_weight": 100.0, "system_ridge": 1e-4, "arap_iter": None},
            "bilaplacian": {"handle_weight": 100.0, "system_ridge": 1e-6, "arap_iter": None},
            "arap": {"handle_weight": 100000.0, "system_ridge": 1e-8, "arap_iter": 3},
        }
        normalised = validate_frozen_classical_configuration(valid)
        self.assertEqual(normalised["arap"]["arap_iter"], 3)

        for broken in (
            {key: value for key, value in valid.items() if key != "arap"},
            {**valid, "arap": {"handle_weight": 0.0, "system_ridge": 1e-8, "arap_iter": 3}},
            {**valid, "arap": {"handle_weight": 1.0, "system_ridge": 1e-8, "arap_iter": None}},
        ):
            with self.assertRaises(ValueError):
                validate_frozen_classical_configuration(broken)


if __name__ == "__main__":
    unittest.main()
