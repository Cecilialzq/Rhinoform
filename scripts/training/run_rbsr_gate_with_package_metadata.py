"""Run RB-SR gate training with the required final-package implementation hash.

The numerical trainer already binds checkpoints to ``train_rbsr_gate.py`` but
one packaging path omitted the same hash from the final package payload while
still declaring it as required.  This launcher changes no training operation;
it injects the live trainer hash immediately before the existing atomic save.
It therefore preserves exact resume compatibility with checkpoints created by
the underlying trainer.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rhinoform import train_rbsr_gate
from rhinoform.repro import sha256_file


def add_training_implementation_metadata(
    payload: dict,
    implementation_path: Path,
) -> dict:
    """Return a copy carrying the exact numerical trainer source hash."""
    result = dict(payload)
    live_hash = sha256_file(implementation_path)
    recorded = result.get("training_implementation_sha256")
    if recorded is not None and recorded != live_hash:
        raise ValueError("Gate package already carries a different training implementation hash")
    result["training_implementation_sha256"] = live_hash
    return result


def main() -> int:
    original_atomic_torch_save = train_rbsr_gate.atomic_torch_save
    implementation_path = Path(train_rbsr_gate.__file__).resolve()

    def atomic_torch_save_with_metadata(path, payload, required_keys=()):
        final_payload = payload
        if Path(path).name == "rbsr_gate_model.pt":
            final_payload = add_training_implementation_metadata(payload, implementation_path)
        return original_atomic_torch_save(path, final_payload, required_keys=required_keys)

    train_rbsr_gate.atomic_torch_save = atomic_torch_save_with_metadata
    try:
        return int(train_rbsr_gate.main())
    finally:
        train_rbsr_gate.atomic_torch_save = original_atomic_torch_save


if __name__ == "__main__":
    raise SystemExit(main())
