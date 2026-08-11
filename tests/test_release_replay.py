from __future__ import annotations

import unittest

from tools.replay_release import compare_payloads


class ReleaseReplayComparisonTests(unittest.TestCase):
    def test_accepts_platform_level_floating_roundoff(self) -> None:
        compare_payloads({"value": 0.1}, {"value": 0.1 + 1e-16})

    def test_rejects_scientifically_material_change(self) -> None:
        with self.assertRaises(AssertionError):
            compare_payloads({"value": 0.1}, {"value": 0.1001})

    def test_rejects_missing_fields(self) -> None:
        with self.assertRaises(AssertionError):
            compare_payloads({"status": "PASS", "rows": []}, {"status": "PASS"})


if __name__ == "__main__":
    unittest.main()
