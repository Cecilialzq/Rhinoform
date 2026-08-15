"""Steady-state runtime benchmark of the frozen certified RB-SR deployment.

Protocol (deployment benchmark, CPU, batch size 1):
  - inputs: the 11 frozen presets plus scaled variants, cycled round-robin
    so no single easy control vector dominates;
  - >=100 warmup runs, >=1000 measured runs;
  - cold start (model load + first inference) reported separately;
  - reports per-stage and end-to-end median/p95/p99/mean, peak RSS,
    certificate iteration stats and status distribution.

Results feed the UI timeout threshold and the paper's runtime section.
Writes deploy/benchmark_results.json.
"""
from __future__ import annotations

import json
import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from rhinoform_runtime import RhinoformRuntime

DEPLOY_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = DEPLOY_DIR.parent / "public" / "bundle"

WARMUP = 100
MEASURE = 1000


def quantiles(values: list[float]) -> dict:
    arr = np.asarray(values)
    return {
        "median_ms": round(float(np.median(arr)), 2),
        "p95_ms": round(float(np.quantile(arr, 0.95)), 2),
        "p99_ms": round(float(np.quantile(arr, 0.99)), 2),
        "mean_ms": round(float(arr.mean()), 2),
        "max_ms": round(float(arr.max()), 2),
    }


def main() -> None:
    manifest = json.loads((BUNDLE_DIR / "manifest.json").read_text(encoding="utf-8"))
    presets = json.loads((BUNDLE_DIR / "certified_presets.json").read_text(encoding="utf-8"))
    mapping = np.asarray(manifest["sliders"]["matrix_6x27"], dtype=np.float64)

    t_cold = time.perf_counter()
    runtime = RhinoformRuntime(verbose=False)
    runtime.set_mean_source()
    cold_load_s = time.perf_counter() - t_cold

    # Request panel: every preset at 1.0x, 0.5x and 1.5x slider amplitude.
    panel = []
    for preset in presets["presets"]:
        sliders = np.asarray(preset["sliders"], dtype=np.float64)
        for scale in (1.0, 0.5, 1.5):
            panel.append((f"{preset['key']}@{scale}", (sliders * scale) @ mapping))

    t_first = time.perf_counter()
    runtime.certified_rbsr(panel[0][1])
    cold_first_infer_s = time.perf_counter() - t_first

    for i in range(WARMUP):
        runtime.certified_rbsr(panel[i % len(panel)][1])

    stage_times: dict[str, list[float]] = {"ridge_ms": [], "cvae_ms": [], "gate_ms": [], "projection_ms": []}
    end_to_end: list[float] = []
    ridge_only: list[float] = []
    iterations: list[int] = []
    retentions: list[float] = []
    statuses: dict[str, int] = {}
    certified_count = 0

    for i in range(MEASURE):
        name, ctrl = panel[i % len(panel)]
        t = time.perf_counter()
        result = runtime.certified_rbsr(ctrl)
        end_to_end.append((time.perf_counter() - t) * 1e3)
        for key in stage_times:
            stage_times[key].append(result.timings_ms[key])
        iterations.append(result.iterations)
        retentions.append(result.retention)
        statuses[result.status] = statuses.get(result.status, 0) + 1
        certified_count += int(result.certified)

        t = time.perf_counter()
        runtime.ridge_anchor(ctrl)
        ridge_only.append((time.perf_counter() - t) * 1e3)

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)

    results = {
        "schema": "rhinoform_runtime_benchmark_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "locked_identities": runtime.identity,
        "environment": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "float32 (network) / float64 (projection)",
            "torch_threads": torch.get_num_threads(),
            "batch_size": 1,
        },
        "protocol": {
            "warmup_runs": WARMUP,
            "measured_runs": MEASURE,
            "request_panel": [name for name, _ in panel],
            "panel_note": "11 frozen presets x {1.0, 0.5, 1.5} slider amplitude, round-robin",
        },
        "cold_start": {
            "model_load_and_hash_verify_s": round(cold_load_s, 2),
            "first_inference_s": round(cold_first_infer_s, 3),
        },
        "steady_state": {
            "ridge_anchor_compute": quantiles(ridge_only),
            "certified_rbsr_end_to_end": quantiles(end_to_end),
            "stages": {key: quantiles(vals) for key, vals in stage_times.items()},
        },
        "certificate": {
            "success_rate": certified_count / MEASURE,
            "status_counts": statuses,
            "iterations": {
                "mean": round(float(np.mean(iterations)), 2),
                "p95": int(np.quantile(iterations, 0.95)),
                "max": int(np.max(iterations)),
            },
            "retention": {
                "mean": round(float(np.mean(retentions)), 4),
                "min": round(float(np.min(retentions)), 4),
            },
        },
        "memory": {"peak_rss_mb": round(peak_rss_mb, 1)},
    }

    out = DEPLOY_DIR / "benchmark_results.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps({k: results[k] for k in ("cold_start", "steady_state", "certificate", "memory")}, indent=1))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
