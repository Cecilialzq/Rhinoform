from __future__ import annotations

import unittest

from tools.audit_public_release import audit_public_release


class PublicReleaseAuditTests(unittest.TestCase):
    def test_current_checkout_passes_public_release_audit(self) -> None:
        report = audit_public_release()
        self.assertEqual(report["status"], "PASS_PUBLIC_RELEASE_AUDIT")
        self.assertEqual(report["privacy_hits"], 0)


if __name__ == "__main__":
    unittest.main()
