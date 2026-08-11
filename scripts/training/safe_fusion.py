"""Resumable orchestration for cache, evaluation, and stress testing."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from rhinoform.repro import atomic_write_json, sha256_file, sha256_json


def run_stage(name: str, command: list[str], marker: Path, force: bool) -> None:
    if marker.exists() and not force:
        print(f"[{name}] already complete: {marker}", flush=True)
        return
    print(f"[{name}] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)
    if not marker.exists():
        raise RuntimeError(f"Stage {name} returned successfully but did not create {marker}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True, help="FaceScape data directory")
    parser.add_argument("--base-package", type=Path, required=True)
    parser.add_argument("--rbsr-package", type=Path, required=True)
    parser.add_argument("--geometric-tuning", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-pair-budget", type=int, default=800)
    parser.add_argument("--test-pair-budget", type=int, default=0)
    parser.add_argument("--stress-pair-budget", type=int, default=800)
    parser.add_argument("--self-intersection-cases", type=int, default=20)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--noise-seeds", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    code_dir = Path(__file__).resolve().parent
    args.out.mkdir(parents=True, exist_ok=True)
    frozen_sources = [
        "safe_fusion.py", "safe_fusion_cache.py", "safe_fusion.py",
        "safe_fusion_stress.py", "safe_fusion.py", "baselines.py", "geometry.py", "data.py",
        "train.py", "train_rbsr_gate.py", "stats.py", "repro.py",
    ]
    signature_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "force"
    }
    signature_payload["artifact_sha256"] = {
        "base_package": sha256_file(args.base_package),
        "rbsr_package": sha256_file(args.rbsr_package),
        "geometric_tuning": sha256_file(args.geometric_tuning),
        "data_manifest": sha256_file(args.repo / "manifest.json"),
        "split_manifest": sha256_file(args.split_manifest),
    }
    signature_payload["source_sha256"] = {
        name: sha256_file(code_dir / name) for name in frozen_sources
    }
    run_signature = sha256_json(signature_payload)
    config_path = args.out / "PIPELINE_CONFIG.json"
    if config_path.exists() and not args.force:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("run_signature") != run_signature:
            raise RuntimeError(
                f"Pipeline arguments differ from the partial/completed run in {args.out}. "
                "Use a new output directory; do not mix quick and formal artifacts."
            )
    atomic_write_json(config_path, {"run_signature": run_signature, "arguments": signature_payload})
    validation_cache = args.out / "cache_validation"
    test_cache = args.out / "cache_test"
    evaluation = args.out / "evaluation"
    stress = args.out / "stress"
    common = [
        "--repo", str(args.repo),
        "--base-package", str(args.base_package),
        "--rbsr-package", str(args.rbsr_package),
        "--geometric-tuning", str(args.geometric_tuning),
        "--split-manifest", str(args.split_manifest),
        "--device", args.device,
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
    ]
    force_flag = ["--force"] if args.force else []
    run_stage(
        "validation-cache",
        [sys.executable, str(code_dir / "safe_fusion_cache.py"), *common,
         "--split", "validation", "--out", str(validation_cache),
         "--pair-budget", str(args.validation_pair_budget), *force_flag],
        validation_cache / "CACHE_COMPLETE.json",
        args.force,
    )
    run_stage(
        "test-cache",
        [sys.executable, str(code_dir / "safe_fusion_cache.py"), *common,
         "--split", "test", "--out", str(test_cache),
         "--pair-budget", str(args.test_pair_budget), *force_flag],
        test_cache / "CACHE_COMPLETE.json",
        args.force,
    )
    run_stage(
        "evaluation",
        [
            sys.executable, str(code_dir / "safe_fusion.py"),
            "--repo", str(args.repo),
            "--validation-cache", str(validation_cache),
            "--test-cache", str(test_cache),
            "--out", str(evaluation),
            "--n-boot", str(args.n_boot),
            "--self-intersection-cases", str(args.self_intersection_cases),
            "--max-configs", str(args.max_configs),
            "--checkpoint-every", str(args.checkpoint_every),
            "--seed", str(args.seed),
        ],
        evaluation / "FINAL_REPORT.json",
        args.force,
    )
    run_stage(
        "stress",
        [
            sys.executable, str(code_dir / "safe_fusion_stress.py"),
            "--repo", str(args.repo),
            "--test-cache", str(test_cache),
            "--selected-config", str(evaluation / "selected_config.json"),
            "--out", str(stress),
            "--pair-budget", str(args.stress_pair_budget),
            "--noise-seeds", str(args.noise_seeds),
            "--checkpoint-every", str(args.checkpoint_every),
            "--seed", str(args.seed),
        ],
        stress / "STRESS_REPORT.json",
        args.force,
    )
    final = {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_report": str(evaluation / "FINAL_REPORT.json"),
        "stress_report": str(stress / "STRESS_REPORT.json"),
        "validation_cache": str(validation_cache / "CACHE_COMPLETE.json"),
        "test_cache": str(test_cache / "CACHE_COMPLETE.json"),
        "run_signature": run_signature,
    }
    atomic_write_json(args.out / "PIPELINE_COMPLETE.json", final)
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
