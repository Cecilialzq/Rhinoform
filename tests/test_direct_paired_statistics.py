from __future__ import annotations

import unittest

import numpy as np


class DirectPairedStatisticsTests(unittest.TestCase):
    def test_degenerate_all_zero_wilcoxon_is_finite_and_non_significant(self) -> None:
        from scripts.evaluation.direct_paired_statistics import safe_wilcoxon

        self.assertEqual(safe_wilcoxon(np.zeros(100, dtype=np.float64)), 1.0)
        self.assertEqual(safe_wilcoxon(np.asarray([np.nan, np.nan])), 1.0)

    def test_holm_adjustment_is_monotone_in_sorted_raw_p_values(self) -> None:
        from scripts.evaluation.direct_paired_statistics import holm

        raw = [0.04, 0.001, 0.02, 1.0]
        adjusted = holm(raw)
        ordered = sorted(zip(raw, adjusted))
        self.assertTrue(all(left[1] <= right[1] for left, right in zip(ordered, ordered[1:])))
        self.assertTrue(all(raw_value <= adjusted_value <= 1.0 for raw_value, adjusted_value in zip(raw, adjusted)))


if __name__ == "__main__":
    unittest.main()
