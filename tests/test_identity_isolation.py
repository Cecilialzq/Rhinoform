from __future__ import annotations

import sys
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from rhinoform.data import training_mean_geometry  # noqa: E402
from rhinoform.train import assert_feature_template_package, build_static_vertex_features  # noqa: E402
from rhinoform.train_rbsr_gate import build_rbf_projection  # noqa: E402


def row(vertices: np.ndarray) -> dict:
    return {
        "vertices": np.asarray(vertices, dtype=np.float64),
        "faces": np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
        "landmarks": np.asarray([0, 1, 2], dtype=np.int64),
        "subunits": {
            "root": np.asarray([0]),
            "dorsum": np.asarray([1]),
            "tip": np.asarray([2]),
            "alar_left": np.asarray([3]),
            "alar_right": np.asarray([3]),
        },
    }


class IdentityIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        train_a = np.asarray([[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]], dtype=np.float64)
        train_b = train_a + np.asarray([0.5, -0.25, 0.2])
        held_out = train_a * 100.0 + 1000.0
        self.by_id = OrderedDict([
            ("test", row(held_out)),
            ("train_b", row(train_b)),
            ("train_a", row(train_a)),
        ])
        self.train_ids = ["train_a", "train_b"]

    def test_training_mean_geometry_ignores_held_out_first_row(self) -> None:
        expected = 0.5 * (self.by_id["train_a"]["vertices"] + self.by_id["train_b"]["vertices"])
        actual = training_mean_geometry(self.by_id, self.train_ids)
        np.testing.assert_allclose(actual, expected)

    def test_static_features_are_independent_of_held_out_first_row(self) -> None:
        features, static = build_static_vertex_features(self.by_id, self.train_ids, use_subunit_features=False)
        reordered = OrderedDict([
            ("train_a", self.by_id["train_a"]),
            ("test", self.by_id["test"]),
            ("train_b", self.by_id["train_b"]),
        ])
        reordered_features, reordered_static = build_static_vertex_features(
            reordered, self.train_ids, use_subunit_features=False
        )
        np.testing.assert_allclose(features, reordered_features)
        np.testing.assert_allclose(static["template_vertices"], reordered_static["template_vertices"])
        self.assertEqual(static["template_sha256"], reordered_static["template_sha256"])

    def test_rbf_basis_uses_the_same_training_mean_geometry(self) -> None:
        features, static = build_static_vertex_features(self.by_id, self.train_ids, use_subunit_features=False)
        self.assertEqual(features.shape[0], 4)
        expected = build_rbf_projection(static["template_vertices"], self.by_id["train_a"]["landmarks"])
        actual = build_rbf_projection(training_mean_geometry(self.by_id, self.train_ids), self.by_id["train_a"]["landmarks"])
        np.testing.assert_allclose(actual, expected)

    def test_legacy_package_without_template_hash_fails_closed(self) -> None:
        _, static = build_static_vertex_features(self.by_id, self.train_ids, use_subunit_features=False)
        with self.assertRaises(RuntimeError):
            assert_feature_template_package({}, static, "legacy")

    def test_mismatched_template_hash_fails_closed(self) -> None:
        _, static = build_static_vertex_features(self.by_id, self.train_ids, use_subunit_features=False)
        with self.assertRaises(RuntimeError):
            assert_feature_template_package({"feature_template_sha256": "wrong"}, static, "bad")


if __name__ == "__main__":
    unittest.main()
