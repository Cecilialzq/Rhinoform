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
from rhinoform.repro import valid_sha256_sidecar
from rhinoform.source_runtime import (
    materialize_source_runtime,
    run_runtime_module,
    verify_runtime_import_origins,
)
from tools.audit_source_snapshots import audit_snapshot
from tools.replay_release import replay_release


EXPERIMENT_SNAPSHOTS = {
    "final-holdout": "final_holdout_v1",
    "supplemental-dominance": "supplemental_dominance_v1",
    "posthoc-subunits": "posthoc_subunits_v1",
}
IMPORT_PROBE_PATHS = {
    "final_holdout_v1": (
        "rhinoform/rbsr_calibration.py",
        "rhinoform/train_rbsr_gate.py",
        "scripts/evaluation/rbsr_gate.py",
        "scripts/evaluation/rbsr_ridge_fold_projection.py",
    ),
    "supplemental_dominance_v1": (
        "rhinoform/rbsr_calibration.py",
        "rhinoform/train_rbsr_gate.py",
        "scripts/evaluation/rbsr_gate.py",
    ),
    "posthoc_subunits_v1": (
        "rhinoform/regional_analysis.py",
        "scripts/evaluation/posthoc_subunit_analysis.py",
    ),
}


def _module_name(relative: str) -> str:
    path = Path(relative)
    if path.suffix != ".py":
        raise ValueError(f"Import probe is not Python source: {relative}")
    return ".".join(path.with_suffix("").parts)


def _snapshot_import_probes(
    repo_root: Path, snapshot_name: str, variant: str | None
) -> dict[str, str]:
    manifest_path = (
        repo_root / "reproducibility/source_snapshots" / snapshot_name / "SOURCE_MANIFEST.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "variants" in manifest:
        if variant not in manifest["variants"]:
            raise ValueError(
                f"{snapshot_name} requires --source-variant from "
                f"{sorted(manifest['variants'])}"
            )
        rows = manifest["variants"][variant]["files"]
    else:
        if variant is not None:
            raise ValueError(f"{snapshot_name} does not define source variants")
        rows = manifest["files"]
    hashes = {str(row["path"]): str(row["sha256"]) for row in rows}
    selected = IMPORT_PROBE_PATHS[snapshot_name]
    missing = sorted(set(selected) - set(hashes))
    if missing:
        raise RuntimeError(f"Source manifest lacks required import probes: {missing}")
    return {_module_name(path): hashes[path] for path in selected}


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
        if not valid_sha256_sidecar(path):
            raise RuntimeError(f"Evidence file or SHA-256 sidecar is invalid: {path}")
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
        "command",
        choices=(
            "show-paths",
            "verify",
            "replay",
            "verify-assets",
            "preflight",
            "materialize",
            "runtime-exec",
        ),
    )
    result.add_argument("--config", type=Path)
    result.add_argument("--data-root", type=Path)
    result.add_argument("--artifact-root", type=Path)
    result.add_argument("--output-root", type=Path)
    result.add_argument("--lamm-root", type=Path)
    result.add_argument(
        "--experiment",
        choices=tuple(EXPERIMENT_SNAPSHOTS),
        default="final-holdout",
        help="Historical experiment whose isolated source runtime is required.",
    )
    result.add_argument(
        "--runtime-root",
        type=Path,
        help="New empty runtime directory (default: OUTPUT/source_runtimes/EXPERIMENT).",
    )
    result.add_argument(
        "--source-variant",
        choices=("pre_erratum", "corrected"),
        help="Required only for a source snapshot that declares variants.",
    )
    result.add_argument(
        "--module",
        help="Python module inside a materialized runtime (runtime-exec only).",
    )
    result.add_argument(
        "--full-data-hash",
        action="store_true",
        help="Hash all 846 processed meshes instead of checking manifest and presence only.",
    )
    result.add_argument(
        "--module-args",
        dest="runtime_arguments",
        nargs=argparse.REMAINDER,
        default=(),
        help="Remaining values passed verbatim to the isolated module.",
    )
    return result


def main() -> int:
    args=parser().parse_args()
    paths=resolve_paths(args)
    if args.command == "runtime-exec":
        if args.runtime_root is None or args.module is None:
            raise SystemExit("runtime-exec requires --runtime-root and --module")
        arguments = list(args.runtime_arguments)
        if arguments[:1] == ["--"]:
            arguments = arguments[1:]
        completed = run_runtime_module(
            runtime_root=args.runtime_root,
            module=args.module,
            arguments=arguments,
        )
        return completed.returncode
    report: dict[str, object]={"paths":paths.public_dict()}
    if args.command in {"verify", "verify-assets", "preflight"}:
        report["source_snapshots"]=verify_snapshots(paths)
        report["evidence_roots"]=verify_evidence_roots(paths)
        release_manifest = paths.repo_root / "reproducibility/RELEASE_ASSET_MANIFEST.json"
        if not valid_sha256_sidecar(release_manifest):
            raise RuntimeError("Release-asset manifest or SHA-256 sidecar is invalid")
        release_record = json.loads(release_manifest.read_text(encoding="utf-8"))
        if (
            release_record.get("status") != "FROZEN_PRIVATE_INVENTORY"
            or release_record.get("public_distribution_authorized") is not False
            or len(release_record.get("artifacts", [])) != 8
        ):
            raise RuntimeError("Unexpected release-asset manifest schema or inventory")
        report["release_asset_inventory"] = {
            "status": release_record["status"],
            "artifacts": len(release_record["artifacts"]),
        }
    if args.command == "replay":
        report["release_statistics"] = replay_release()
    if args.command in {"verify-assets", "preflight"}:
        report["release_assets"] = validate_release_assets(paths)
    if args.command == "preflight":
        report["processed_facescape"]=validate_processed_facescape(
            paths, full_hash=args.full_data_hash
        )
        paths.output_root.mkdir(parents=True, exist_ok=True)
    if args.command == "materialize":
        snapshot_name = EXPERIMENT_SNAPSHOTS[args.experiment]
        runtime_root = (
            args.runtime_root
            or paths.output_root / "source_runtimes" / args.experiment
        ).resolve()
        report["source_runtime"] = materialize_source_runtime(
            repo_root=paths.repo_root,
            snapshot_name=snapshot_name,
            destination=runtime_root,
            variant=args.source_variant,
        )
        report["source_runtime_imports"] = verify_runtime_import_origins(
            runtime_root=runtime_root,
            modules=_snapshot_import_probes(
                paths.repo_root, snapshot_name, args.source_variant
            ),
        )
    report["status"]="PASS"
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
