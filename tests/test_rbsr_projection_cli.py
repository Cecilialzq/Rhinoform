import unittest

from scripts.evaluation.rbsr_ridge_fold_projection import parse_floats


class RbsrProjectionCliTests(unittest.TestCase):
    def test_accepts_zero_identity_audit_and_strict_interior_candidates(self):
        self.assertEqual(parse_floats("0,0.25,0.5,0.75"), [0.0, 0.25, 0.5, 0.75])

    def test_rejects_unit_attenuation_before_projection_work_starts(self):
        with self.assertRaisesRegex(ValueError, "strictly between zero and one"):
            parse_floats("0,0.25,1")

    def test_requires_zero_identity_audit(self):
        with self.assertRaisesRegex(ValueError, "zero-residual Ridge identity audit"):
            parse_floats("0.25,0.5,0.75")


if __name__ == "__main__":
    unittest.main()
