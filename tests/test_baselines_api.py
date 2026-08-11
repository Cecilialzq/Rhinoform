from __future__ import annotations

import unittest


class ClassicalBaselineApiTests(unittest.TestCase):
    def test_required_classical_baseline_api_is_present(self) -> None:
        from rhinoform import baselines

        for name in (
            "arap_predict_vectorised",
            "per_pair_metrics",
            "aggregate",
            "write_pair_csv",
        ):
            self.assertTrue(callable(getattr(baselines, name, None)), name)

if __name__ == "__main__":
    unittest.main()
