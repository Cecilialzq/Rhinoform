"""Single path-independent entry point for Rhinoform release verification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rhinoform.reproduction import (
    ReproductionPaths,
    validate_processed_facescape,
    validate_release_assets,
)
from tools.audit_source_snapshots import audit_snapshot


def resolve_paths(args: argparse.Namespace) -> ReproductionPaths:
    return ReproductionPaths.resolve(
        config=args.config,
        data_root=args.data_root,
        artifact_root=args.artifact_root,
        output_root=args.output_root,
        lamm_root=args.lamm_root,
    )


def verify_snapshots(paths: ReproductionPaths) -> list[dict]:
    root = paths.repo_root / "reproducibility/source_snapshots"
    return [audit_snapshot(path) for path in sorted(root.iterdir()) if path.is_dir()]


def verify_evidence_roots(paths: ReproductionPaths) -> list[dict[str, object]]:
    specifications = [
        (
            "results/rbsr_final_rerun_holdout_v1/FINAL_RESULT_FREEZE_AUDIT.json",
            "PASS_CANONICAL_INTERNAL_FINAL_RERUN_HOLDOUT_FROZEN",
        ),
        (
            "results/supplemental_rbsr_lamm_dominance_search_v1/release_evidence/RELEASE_MANIFEST.json",
            "PASS",
        ),
        (
            "results/rbsr_final_rerun_holdout_v1/posthoc_subunit_analysis_v1/POSTHOC_SUBUNIT_ANALYSIS_EVIDENCE.json",
            "COMPLETE_POSTHOC_SUBUNIT_ANALYSIS",
        ),
    ]
    reports=[]
    for relative, expected_status in specifications:
        path=paths.locate(relative)
        record=json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != expected_status:
            raise RuntimeError(
                f"Unexpected evidence status in {path}: {record.get('status')} != {expected_status}"
            )
        reports.append({"path":relative,"status":record["status"]})
    return reports


def parser() -> argparse.ArgumentParser:
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "command", choices=("show-paths", "verify", "verify-assets", "preflight")
    )
    result.add_argument("--config", type=Path)
    result.add_argument("--data-root", type=Path)
    result.add_argument("--artifact-root", type=Path)
    result.add_argument("--output-root", type=Path)
    result.add_argument("--lamm-root", type=Path)
    result.add_argument(
        "--full-data-hash",
        action="store_true",
        help="Hash all 846 processed meshes instead of checking manifest and presence only.",
    )
    return result


def main() -> int:
    args=parser().parse_args()
    paths=resolve_paths(args)
    report: dict[str, object]={"paths":paths.public_dict()}
    if args.command in {"verify", "verify-assets", "preflight"}:
        report["source_snapshots"]=verify_snapshots(paths)
        report["evidence_roots"]=verify_evidence_roots(paths)
    if args.command in {"verify-assets", "preflight"}:
        report["release_assets"] = validate_release_assets(paths)
    if args.command == "preflight":
        report["processed_facescape"]=validate_processed_facescape(
            paths, full_hash=args.full_data_hash
        )
        paths.output_root.mkdir(parents=True, exist_ok=True)
    report["status"]="PASS"
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
