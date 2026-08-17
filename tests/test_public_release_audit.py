from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from tools.audit_public_release import audit_public_release, process_artifact_hits


class PublicReleaseAuditTests(unittest.TestCase):
    def test_internal_process_artifacts_are_rejected(self) -> None:
        root = Path("/repo")
        files = [
            root / "reviews/audit_artifacts/report.csv",
            root / "tmp/render/page-01.png",
            root / "package/__pycache__/module.cpython-312.pyc",
            root / "demo/dist/index.html",
            root / "notebooks/internal.ipynb",
            root / "benchmarks/runtime.json",
            root / "reproducibility/source_snapshots/run/source.py",
            root / "results/rbsr_final_rerun_holdout_v1/posthoc_subunit_analysis_v1/plot.png",
            root / "docs/final_tables/main_results.csv",
        ]
        self.assertEqual(
            process_artifact_hits(root, files),
            [
                "reviews/audit_artifacts/report.csv",
                "tmp/render/page-01.png",
                "package/__pycache__/module.cpython-312.pyc",
                "demo/dist/index.html",
                "notebooks/internal.ipynb",
                "benchmarks/runtime.json",
                "reproducibility/source_snapshots/run/source.py",
                "results/rbsr_final_rerun_holdout_v1/posthoc_subunit_analysis_v1/plot.png",
            ],
        )

    def test_current_checkout_passes_public_release_audit(self) -> None:
        report = audit_public_release()
        self.assertEqual(report["status"], "PASS_PUBLIC_RELEASE_AUDIT")
        self.assertEqual(report["privacy_hits"], 0)

    def test_git_grep_fallback_passes_without_ripgrep(self) -> None:
        with mock.patch("tools.audit_public_release.shutil.which", return_value=None):
            report = audit_public_release()
        self.assertEqual(report["status"], "PASS_PUBLIC_RELEASE_AUDIT")
        self.assertEqual(report["privacy_hits"], 0)


if __name__ == "__main__":
    unittest.main()
