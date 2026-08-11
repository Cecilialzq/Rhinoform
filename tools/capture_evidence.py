#!/usr/bin/env python3
"""
Capture the "Evidence Missing / Evidence Limited" items that CAN be recorded from
the current environment, and honestly scan for historical timing that mostly was
NOT logged.

Two clearly separated blocks (EXECUTION_PLAN.md A6/A7):
  * current_runtime_env.json  -- today's supplemental runtime (GPU model, CUDA,
    cuDNN, RAM, CPU). Labelled SUPPLEMENTAL: it describes the machine running THIS
    script, and must NOT be retro-applied to historical frozen results.
  * historical_training_time.json -- every genuinely logged wall-clock found by
    scanning run/gate metadata. If little/nothing is logged, the verdict is
    "Exploratory Compute -- Not Systematically Logged". Nothing is invented.

Writes only under --out (results/additional_experiments/evidence_capture).
Reads the repo read-only. Degrades gracefully if torch / nvidia-smi are absent.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip() or None
    except Exception:
        return None


def capture_runtime_env() -> dict:
    env: dict = {
        "label": "SUPPLEMENTAL current runtime; NOT the hardware of historical frozen results",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    # RAM
    try:
        meminfo = Path("/proc/meminfo").read_text()
        m = re.search(r"MemTotal:\s+(\d+)\s+kB", meminfo)
        if m:
            env["ram_total_gb"] = round(int(m.group(1)) / (1024 * 1024), 2)
    except Exception:
        env["ram_total_gb"] = None
    # torch / CUDA
    try:
        import torch  # noqa: PLC0415
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_version"] = getattr(torch.version, "cuda", None)
        try:
            env["cudnn_version"] = torch.backends.cudnn.version()
        except Exception:
            env["cudnn_version"] = None
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            env["gpu_total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
            env["gpu_multiprocessors"] = props.multi_processor_count
        else:
            env["gpu_name"] = None
    except Exception as exc:  # torch not installed
        env["torch"] = None
        env["torch_import_error"] = repr(exc)
    # nvidia-smi (raw, authoritative for the GPU model)
    smi = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader"])
    env["nvidia_smi_query"] = smi
    env["nvidia_smi_full"] = _run(["nvidia-smi"])
    return env


TIMING_KEYS = ("elapsed_sec", "elapsed", "duration_sec", "training_seconds",
               "wall_clock_sec", "runtime_sec")


def scan_historical_timing(root: Path) -> dict:
    """Walk run/gate metadata for any genuinely logged wall-clock. No inference."""
    found = []
    scan_dirs = [root / "runs", root / "plan_a_clean_retrain"]
    for base in scan_dirs:
        if not base.exists():
            continue
        for p in base.rglob("*.json"):
            try:
                if p.stat().st_size > 5_000_000:  # skip huge files
                    continue
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            for key in TIMING_KEYS:
                if key in obj and isinstance(obj[key], (int, float)):
                    found.append({
                        "path": str(p.relative_to(root)),
                        "key": key,
                        "seconds": float(obj[key]),
                        "device": obj.get("device"),
                    })
    verdict = ("systematically_logged" if len(found) >= 10
               else "Exploratory Compute -- Not Systematically Logged")
    return {
        "label": "HISTORICAL frozen results; hardware/timing mostly NOT logged",
        "n_timing_records_found": len(found),
        "records": found,
        "verdict": verdict,
        "note": ("Only the genuinely recorded wall-clock values are listed. Missing "
                 "training times are NOT reconstructed or assumed. The current-runtime "
                 "GPU is NOT retro-applied to these historical runs."),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    env = capture_runtime_env()
    (out / "current_runtime_env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    hist = scan_historical_timing(root)
    (out / "historical_training_time.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")

    summary = [
        "# Evidence capture summary",
        "",
        "## Current runtime (SUPPLEMENTAL)",
        f"- GPU: {env.get('gpu_name')}",
        f"- CUDA: {env.get('cuda_version')}  cuDNN: {env.get('cudnn_version')}",
        f"- torch: {env.get('torch')}  RAM: {env.get('ram_total_gb')} GB  CPU cores: {env.get('cpu_count')}",
        f"- nvidia-smi: {env.get('nvidia_smi_query')}",
        "",
        "## Historical training time (frozen results)",
        f"- timing records found: {hist['n_timing_records_found']}",
        f"- verdict: **{hist['verdict']}**",
        "",
        "> The current GPU describes only this run. Historical frozen results were produced on"
        " unlogged hardware and are NOT claimed to be the same device.",
    ]
    (out / "evidence_capture_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"runtime": env.get("gpu_name"), "cuda": env.get("cuda_version"),
                      "historical_timing_records": hist["n_timing_records_found"],
                      "verdict": hist["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
