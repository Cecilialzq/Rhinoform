"""Reproduce the dissertation's batch-one LAMM forward-latency protocol."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from experiments.lamm.run_lamm_facescape import (
    LAMM_COMMIT,
    delta_controls,
    load_dataset,
    model_config,
    sha256,
    valid_sha256_sidecar,
    write_json,
    write_sha256_sidecar,
)


def _git(path: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *arguments], text=True
    ).strip()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--lamm-root", type=Path, required=True)
    result.add_argument("--checkpoint-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--warmup", type=int, default=100)
    result.add_argument("--repeats", type=int, default=1000)
    return result


@torch.no_grad()
def main() -> int:
    args = parser().parse_args()
    if args.warmup < 1 or args.repeats < 1:
        raise ValueError("--warmup and --repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("The recorded LAMM benchmark requires a CUDA GPU")

    repo_root = args.repo_root.resolve()
    data_root = args.data_root.resolve()
    lamm_root = args.lamm_root.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoint = checkpoint_dir / "manipulation_best.pt"
    region_file = checkpoint_dir / "region_ids.pickle"
    if _git(lamm_root, "rev-parse", "HEAD") != LAMM_COMMIT:
        raise RuntimeError(f"LAMM checkout must be pinned to {LAMM_COMMIT}")
    if _git(lamm_root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("LAMM checkout must be clean")
    if not valid_sha256_sidecar(checkpoint):
        raise RuntimeError("LAMM checkpoint or SHA-256 sidecar is invalid")
    if not region_file.is_file():
        raise FileNotFoundError(region_file)

    split_manifest = (
        repo_root
        / "results/rbsr_final_rerun_holdout_v1/protocol/"
        "final_rerun_holdout_split_manifest.json"
    )
    _, arrays, _, _, _, _, regions, controls, _, _ = load_dataset(
        repo_root,
        data_root=data_root,
        split_manifest=split_manifest,
        load_test=False,
    )
    if len(regions) != 5 or arrays["validation"].shape[0] < 2:
        raise RuntimeError("Unexpected LAMM region or validation-data contract")

    sys.path.insert(0, str(lamm_root))
    from models import LAMM  # type: ignore[import-not-found]

    device = torch.device("cuda")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    stored_configuration = state.get("model_config")
    configuration = model_config(region_file, controls, manipulation=True)
    comparable_stored = (
        {**stored_configuration, "region_ids_file": str(region_file)}
        if isinstance(stored_configuration, dict)
        else None
    )
    if comparable_stored != configuration or "model" not in state:
        raise RuntimeError("LAMM checkpoint/model configuration mismatch")
    model = LAMM(configuration).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    mean = arrays["train"].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = arrays["train"].std(axis=0, dtype=np.float64).astype(np.float32) + 1e-7
    validation = ((arrays["validation"] - mean) / std).astype(np.float32)
    source = torch.from_numpy(validation[0:1]).to(device)
    target = torch.from_numpy(validation[1:2]).to(device)
    controls_delta = delta_controls(source, target, model)

    for _ in range(args.warmup):
        model((source, controls_delta))[-1]
    torch.cuda.synchronize(device)

    samples: list[float] = []
    for _ in range(args.repeats):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        model((source, controls_delta))[-1]
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - started) * 1000.0)

    record = {
        "schema": "rhinoform_lamm_batch1_forward_benchmark_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(device),
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "scope": (
            "LAMM forward only; batch=1; inputs pre-staged on GPU; excludes "
            "normalisation, transfer, strict scoring and rendering"
        ),
        "lamm_commit": LAMM_COMMIT,
        "checkpoint_sha256": sha256(checkpoint),
        "split_manifest_sha256": sha256(split_manifest),
    }
    write_json(args.output, record)
    write_sha256_sidecar(args.output)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
