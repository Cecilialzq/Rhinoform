from __future__ import annotations

import json
from pathlib import Path
import unittest

from rhinoform.repro import sha256_file, write_sha256_sidecar
from scripts.evaluation.lamm_validation_reference import (
    comparable_model_config,
    relocate_comparable_model_config,
    require_frozen_result_artifact,
)


class LAMMValidationReferenceTests(unittest.TestCase):
    def test_constructor_derived_defaults_are_canonicalised(self) -> None:
        live = {
            "backbone": "transformer",
            "region_ids_file": "/new/mount/region_ids.pickle",
            "dim": 512,
        }
        recorded = {
            **live,
            "region_ids_file": "/old/mount/region_ids.pickle",
            "scale_dim_token": 0.5,
            "depth": None,
            "Dinput": 3,
            "Dlms": 8,
        }
        self.assertTrue(comparable_model_config(recorded, live))

    def test_structural_configuration_difference_is_rejected(self) -> None:
        live = {
            "backbone": "transformer",
            "region_ids_file": "/new/mount/region_ids.pickle",
            "dim": 512,
        }
        recorded = {
            **live,
            "region_ids_file": "/old/mount/region_ids.pickle",
            "dim": 256,
            "scale_dim_token": 0.5,
            "depth": None,
            "Dinput": 3,
            "Dlms": 8,
        }
        self.assertFalse(comparable_model_config(recorded, live))
        with self.assertRaisesRegex(ValueError, "structural configuration mismatch"):
            relocate_comparable_model_config(recorded, live)

    def test_only_region_mount_path_is_relocated_in_memory(self) -> None:
        live = {
            "backbone": "transformer",
            "region_ids_file": "/repo/results/region_ids.pickle",
            "dim": 512,
        }
        recorded = {
            **live,
            "region_ids_file": "/legacy/FYP final/region_ids.pickle",
            "scale_dim_token": 0.5,
            "depth": None,
            "Dinput": 3,
            "Dlms": 8,
        }
        relocated = relocate_comparable_model_config(recorded, live)
        self.assertEqual(relocated["region_ids_file"], live["region_ids_file"])
        self.assertEqual(
            {key: value for key, value in relocated.items() if key != "region_ids_file"},
            {key: value for key, value in recorded.items() if key != "region_ids_file"},
        )

class FrozenResultArtifactTests(unittest.TestCase):
    def test_manifest_match_passes_and_content_drift_fails_closed(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "lamm/seed/region_ids.pickle"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"frozen-region-map")
            write_sha256_sidecar(artifact)
            manifest = root / "RESULT_TREE_MANIFEST.json"
            manifest.write_text(json.dumps({
                "status": "CANONICAL_RESULT_TREE_FROZEN",
                "artifacts": [{
                    "path": "lamm/seed/region_ids.pickle",
                    "bytes": artifact.stat().st_size,
                    "sha256": sha256_file(artifact),
                }],
            }))
            write_sha256_sidecar(manifest)
            self.assertEqual(
                require_frozen_result_artifact(root, artifact),
                sha256_file(artifact),
            )
            artifact.write_bytes(b"different-region-map")
            write_sha256_sidecar(artifact)
            with self.assertRaisesRegex(RuntimeError, "result-tree manifest"):
                require_frozen_result_artifact(root, artifact)

    def test_external_legacy_mount_is_bound_by_canonical_relative_path(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "github_results"
            root.mkdir()
            external = workspace / "FYP final/region_ids.pickle"
            external.parent.mkdir()
            external.write_bytes(b"same-frozen-region-map")
            write_sha256_sidecar(external)
            manifest = root / "RESULT_TREE_MANIFEST.json"
            manifest.write_text(json.dumps({
                "status": "CANONICAL_RESULT_TREE_FROZEN",
                "artifacts": [{
                    "path": "lamm/seed20260609/region_ids.pickle",
                    "bytes": external.stat().st_size,
                    "sha256": sha256_file(external),
                }],
            }))
            write_sha256_sidecar(manifest)
            self.assertEqual(
                require_frozen_result_artifact(
                    root,
                    external,
                    manifest_relative_path="lamm/seed20260609/region_ids.pickle",
                ),
                sha256_file(external),
            )


if __name__ == "__main__":
    unittest.main()
