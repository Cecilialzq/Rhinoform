"""Run an existing evaluation script under the STRICT protocol patch.

This applies strict_protocol_patch BEFORE the target script imports the per-pair
scorers, then executes the target as ``__main__`` with the remaining argv. It lets
us re-score every existing evaluation entry point under one unified contract
without editing those scripts.

Usage:
    python scripts/pipeline/strict_runner.py scripts/evaluation/rbsr_gate.py --repo data ... --seed 2026
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = str(Path(__file__).resolve().parent)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from rhinoform import strict_protocol_patch as strict


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/pipeline/strict_runner.py <target_script.py> [args...]", file=sys.stderr)
        return 2
    strict.apply()
    target = sys.argv[1]
    # hand the target its own argv (script name + its flags)
    sys.argv = sys.argv[1:]
    runpy.run_path(target, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
