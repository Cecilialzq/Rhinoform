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

    def test_required_reproduction_helpers_are_present(self) -> None:
        from rhinoform.novelty_upgrade import srg_common

        for name in (
            "load_package",
            "load_available_rows",
            "build_vertex_features",
            "regenerate_predictions",
        ):
            self.assertTrue(callable(getattr(srg_common, name, None)), name)


if __name__ == "__main__":
    unittest.main()
