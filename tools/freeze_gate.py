from __future__ import annotations

import argparse
import json
from pathlib import Path

from rhinoform.repro import atomic_write_json


REQUIRED_PATHS = [   # trimmed to the artefacts actually shipped in this release

    "seed_registry.json",
    "reproducibility_manifest.json",
    "data/DATASET_DONE.json",
    "splits/facescape_847/split_manifest.json",
    "execution_plan.json",
    "progress_ledger.json",
    "frozen_artifacts/learning_curve/table_lc_primary.csv",
    "frozen_artifacts/learning_curve/table_lc_secondary.csv",
    "frozen_artifacts/learning_curve/table_lc_anchor.csv",
    "frozen_artifacts/learning_curve/table_lc_primary_original50.csv",
    "frozen_artifacts/learning_curve/table_lc_secondary_original50.csv",
    "frozen_artifacts/learning_curve/lc_gap_primary.csv",
    "frozen_artifacts/learning_curve/lc_gap_secondary.csv",
    "frozen_artifacts/operating_points/pareto_points.csv",
    "frozen_artifacts/operating_points/pareto_front.csv",
    "frozen_artifacts/qualitative_cases/qualitative_cases.csv",
    "frozen_artifacts/qualitative_cases/qualitative_cases_manifest.json",
    "frozen_artifacts/roi_freeze_audit/roi_freeze_audit.json",
    "frozen_artifacts/sensitivity/seed_subset_sensitivity_manifest.json",
    "frozen_artifacts/sensitivity/sensitivity_scale30.npz",
    "frozen_artifacts/sensitivity/sensitivity_scale676.npz",
    "results/dependence/dependence_robustness.csv",
    "results/constants/implementation_constants.csv",
    "results/constants/implementation_constants.md",
    "docs/reproducibility_statement.md",
    "outputs/reproducibility/reproducibility_check_frozen_statistics.json",
    "outputs/reproducibility/reproducibility_check_lightweight_pipeline.json",
]

REQUIRED_FIGURE_GLOBS = [
]


def check_path(root: Path, rel: str) -> dict:
    path = root / rel
    return {"path": rel, "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="REQUIRED_TIER_FREEZE_AUDIT.json")
    parser.add_argument("--write-complete", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    checks = [check_path(root, p) for p in REQUIRED_PATHS + REQUIRED_FIGURE_GLOBS]

    plan_path = root / "execution_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        queue_ok = plan.get("run_count") == plan.get("done_count") and plan.get("blocked_count", 1) == 0 and plan.get("failed_count", 1) == 0
    else:
        plan = {}
        queue_ok = False
    ledger_path = root / "progress_ledger.json"
    ledger = {}
    ledger_parse_error = None
    if ledger_path.exists():
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except Exception as exc:
            ledger_parse_error = repr(exc)
    ledger_ok = ledger.get("state") == "done"
    missing = [c for c in checks if not c["exists"] or c["size_bytes"] == 0]
    status = "pass" if not missing and queue_ok and ledger_ok else "fail"
    report = {
        "status": status,
        "missing_or_empty": missing,
        "queue_ok": queue_ok,
        "ledger_ok": ledger_ok,
        "ledger_parse_error": ledger_parse_error,
        "plan_counts": {k: plan.get(k) for k in ("run_count", "done_count", "blocked_count", "failed_count")},
        "ledger": ledger,
        "checks": checks,
    }
    atomic_write_json(root / args.out, report)
    if status == "pass" and args.write_complete:
        (root / "REQUIRED_TIER_COMPLETE.md").write_text(
            "# Required Tier Complete\n\nAll Required-tier freeze audit checks passed.\n",
            encoding="utf-8",
        )
    print(json.dumps({"status": status, "missing": len(missing), "queue_ok": queue_ok, "ledger_ok": ledger_ok}, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
