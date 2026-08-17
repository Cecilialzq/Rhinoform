import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT
    / "notebooks"
    / "Rhinoform_RBSR_internal_final_rerun_holdout_colab.ipynb"
)


class FinalHoldoutNotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = [
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        cls.source = "\n".join(cls.code)

    def test_all_code_cells_compile(self) -> None:
        for index, source in enumerate(self.code):
            compile(source, f"{NOTEBOOK}:code-cell-{index}", "exec")

    def test_kernel_and_subprocess_import_bootstrap_is_persisted(self) -> None:
        self.assertIn("sys.path.insert(0, repo_str)", self.source)
        self.assertIn('LIVE_ENV["PYTHONPATH"] = os.environ["PYTHONPATH"]', self.source)
        self.assertIn("rhinoform.__file__", self.source)

    def test_validation_projection_reuses_raw_report_and_freezes_both_attenuations(self) -> None:
        self.assertIn("if not valid_sha256_sidecar(raw_report_path):", self.source)
        self.assertIn("'--attenuations', '0,0.75'", self.source)
        self.assertIn("zero_gate_ridge_identity", self.source)

    def test_pretest_freeze_covers_all_fourteen_implementations(self) -> None:
        freeze_cell = next(source for source in self.code if source.startswith("ALL_MODELS_FREEZE ="))
        expected = {
            "rhinoform/rbsr_calibration.py",
            "scripts/evaluation/rbsr_ridge_fold_projection.py",
            "scripts/analysis/select_rbsr_certified_projection.py",
        }
        for relative_path in expected:
            self.assertIn(relative_path, freeze_cell)
        self.assertIn("len(implementation_hashes)", freeze_cell)

    def test_completed_run_is_idempotent_and_holdout_is_enabled(self) -> None:
        self.assertIn("RUN_FINAL_HOLDOUT = True", self.source)
        self.assertIn("FINAL_RUN_ALREADY_COMPLETE", self.source)
        self.assertIn("skipping licensed-data rsync", self.source)
        self.assertIn("immutable final evidence already exists", self.source)

    def test_paired_statistics_json_sidecars_are_repaired_without_recomputation(self) -> None:
        final_cell = next(
            source
            for source in self.code
            if source.startswith("if RUN_FINAL_HOLDOUT and FINAL_RUN_ALREADY_COMPLETE:")
        )
        self.assertIn("expected_stats_json_rows", final_cell)
        self.assertIn("write_sha256_sidecar(json_path)", final_cell)
        self.assertIn("len(stats_payload['rows']) == expected_rows", final_cell)


if __name__ == "__main__":
    unittest.main()
