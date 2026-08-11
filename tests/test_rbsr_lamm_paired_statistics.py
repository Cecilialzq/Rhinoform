from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluation.rbsr_lamm_paired_statistics import (
    aggregate_by_identity,
    holm,
    safe_wilcoxon_two_sided,
)


class RBSRLAMMPairedStatisticsTests(unittest.TestCase):
    def test_identity_aggregation_preserves_balanced_mean(self) -> None:
        rows = [
            {"source_id": "1", "target_id": "2"},
            {"source_id": "1", "target_id": "3"},
            {"source_id": "2", "target_id": "1"},
            {"source_id": "2", "target_id": "3"},
        ]
        differences = np.asarray([-2.0, 0.0, -1.0, -1.0])
        source = aggregate_by_identity(rows, differences, "source_id")
        self.assertTrue(np.array_equal(source, np.asarray([-1.0, -1.0])))
        self.assertEqual(float(np.mean(source)), float(np.mean(differences)))

    def test_degenerate_wilcoxon_is_non_significant(self) -> None:
        self.assertEqual(safe_wilcoxon_two_sided(np.zeros(100)), 1.0)
        self.assertEqual(safe_wilcoxon_two_sided(np.asarray([np.nan])), 1.0)

    def test_holm_is_monotone_and_never_smaller_than_raw(self) -> None:
        raw = [0.03, 0.001, 0.02]
        adjusted = holm(raw)
        ordered = sorted(zip(raw, adjusted))
        self.assertTrue(all(a[1] <= b[1] for a, b in zip(ordered, ordered[1:])))
        self.assertTrue(all(r <= a <= 1.0 for r, a in zip(raw, adjusted)))


if __name__ == "__main__":
    unittest.main()
