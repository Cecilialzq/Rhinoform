import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.training.run_rbsr_gate_with_package_metadata import (
    add_training_implementation_metadata,
)


class GatePackageMetadataLauncherTests(unittest.TestCase):
    def test_injects_exact_training_implementation_hash_without_mutating_input(self):
        with tempfile.TemporaryDirectory() as directory:
            implementation = Path(directory) / "train_rbsr_gate.py"
            implementation.write_bytes(b"numerical trainer fixture\n")
            payload = {"method": "rbsr_gate_prototype"}
            result = add_training_implementation_metadata(payload, implementation)
            self.assertNotIn("training_implementation_sha256", payload)
            self.assertEqual(
                result["training_implementation_sha256"],
                hashlib.sha256(implementation.read_bytes()).hexdigest(),
            )

    def test_rejects_conflicting_existing_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            implementation = Path(directory) / "train_rbsr_gate.py"
            implementation.write_bytes(b"current trainer\n")
            with self.assertRaisesRegex(ValueError, "different training implementation hash"):
                add_training_implementation_metadata(
                    {"training_implementation_sha256": "0" * 64},
                    implementation,
                )


if __name__ == "__main__":
    unittest.main()
