"""Recompute the two report-level statistical analyses from frozen pair tables."""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from rhinoform.repro import sha256_file, valid_sha256_sidecar


REPO_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_ROOT = REPO_ROOT / "results/rbsr_final_rerun_holdout_v1"
SUPPLEMENTARY_ROOT = (
    REPO_ROOT / "results/supplemental_rbsr_lamm_dominance_search_v1/release_evidence"
)
RTOL = 1e-12
ATOL = 1e-15
PAIR_INPUTS = (
    PRIMARY_ROOT
    / "direct_paired_statistics/pair_metrics/identity_bootstrap_pair_metrics_certified_rbsr.csv",
    PRIMARY_ROOT
    / "direct_paired_statistics/pair_metrics/identity_bootstrap_pair_metrics_ridge.csv",
    SUPPLEMENTARY_ROOT / "pair_metrics/identity_bootstrap_pair_metrics_lamm.csv",
    SUPPLEMENTARY_ROOT / "pair_metrics/pair_metrics_rbsr_optimized_test.csv",
)


def compare_payloads(expected: Any, actual: Any, path: str = "$") -> None:
    """Compare deterministic JSON while tolerating platform-level floating roundoff."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected is not actual:
            raise AssertionError(f"{path}: {actual!r} != {expected!r}")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(float(actual), float(expected), rel_tol=RTOL, abs_tol=ATOL):
            raise AssertionError(f"{path}: {actual!r} != {expected!r}")
        return
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(actual) != set(expected):
            raise AssertionError(
                f"{path}: key mismatch actual-only={sorted(set(actual) - set(expected))} "
                f"expected-only={sorted(set(expected) - set(actual))}"
            )
        for key in expected:
            compare_payloads(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: length {len(actual)} != {len(expected)}")
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            compare_payloads(expected_item, actual_item, f"{path}[{index}]")
        return
    if actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def _run(command: list[str]) -> None:
    completed = subprocess.run(
        command, cwd=REPO_ROOT, text=True, check=False, capture_output=True
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Release replay failed with exit code {completed.returncode}: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def _load_verified(path: Path) -> dict[str, Any]:
    if not valid_sha256_sidecar(path):
        raise RuntimeError(f"Frozen JSON or SHA-256 sidecar is invalid: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_final_table_sources() -> dict[str, str]:
    table_manifest = _load_verified(REPO_ROOT / "docs/final_tables/FINAL_PAPER_TABLES.json")
    verified: dict[str, str] = {}
    for relative, expected_hash in table_manifest["source_sha256"].items():
        path = REPO_ROOT / relative
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Paper-table source changed: {relative}: {actual_hash} != {expected_hash}"
            )
        verified[relative] = actual_hash
    return verified


def _verify_pair_inputs() -> list[str]:
    verified = []
    for path in PAIR_INPUTS:
        if not valid_sha256_sidecar(path):
            raise RuntimeError(f"Frozen pair table or SHA-256 sidecar is invalid: {path}")
        verified.append(path.relative_to(REPO_ROOT).as_posix())
    return verified


def replay_release() -> dict[str, Any]:
    """Run both frozen analyses into a temporary directory and compare every field."""
    started = time.monotonic()
    pair_inputs = _verify_pair_inputs()
    with tempfile.TemporaryDirectory(prefix="rhinoform-release-replay-") as temp:
        output_root = Path(temp)
        primary_output = output_root / "primary_rbsr_vs_ridge.csv"
        _run(
            [
                sys.executable,
                "-m",
                "scripts.evaluation.direct_paired_statistics",
                "--pair-dir",
                str(PRIMARY_ROOT / "direct_paired_statistics/pair_metrics"),
                "--baseline",
                "ridge",
                "--methods",
                "certified_rbsr",
                "--out",
                str(primary_output),
                "--seed",
                "20260609",
                "--n-boot",
                "10000",
            ]
        )
        primary_expected = _load_verified(
            PRIMARY_ROOT / "direct_paired_statistics/paired_primary_rbsr_vs_ridge.json"
        )
        primary_actual = json.loads(
            primary_output.with_suffix(".json").read_text(encoding="utf-8")
        )
        compare_payloads(primary_expected, primary_actual)

        supplemental_output = output_root / "rbsr_vs_lamm_core_paired_statistics"
        _run(
            [
                sys.executable,
                "-m",
                "scripts.evaluation.rbsr_lamm_paired_statistics",
                "--lamm-pairs",
                str(
                    SUPPLEMENTARY_ROOT
                    / "pair_metrics/identity_bootstrap_pair_metrics_lamm.csv"
                ),
                "--rbsr-pairs",
                str(SUPPLEMENTARY_ROOT / "pair_metrics/pair_metrics_rbsr_optimized_test.csv"),
                "--out",
                str(supplemental_output),
                "--seed",
                "20260609",
                "--n-boot",
                "20000",
            ]
        )
        supplemental_expected = _load_verified(
            SUPPLEMENTARY_ROOT / "analysis/rbsr_vs_lamm_core_paired_statistics.json"
        )
        supplemental_actual = json.loads(
            supplemental_output.with_suffix(".json").read_text(encoding="utf-8")
        )
        compare_payloads(supplemental_expected, supplemental_actual)

    sources = _verify_final_table_sources()
    return {
        "status": "PASS_RELEASE_STATISTICS_REPLAY",
        "scope": "frozen pair-level predictions; no model inference or retraining",
        "floating_comparison": {"rtol": RTOL, "atol": ATOL},
        "primary": {
            "comparison": "certified_rbsr_vs_ridge",
            "pairs": 9900,
            "metrics": 6,
            "status": "MATCH",
        },
        "supplementary": {
            "comparison": "pca128_calibrated_rbsr_vs_frozen_lamm",
            "pairs": 9900,
            "metrics": 3,
            "status": supplemental_actual["status"],
        },
        "paper_table_sources_verified": len(sources),
        "pair_inputs_verified": len(pair_inputs),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def main() -> int:
    print(json.dumps(replay_release(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
