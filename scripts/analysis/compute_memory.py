#!/usr/bin/env python3
"""
Measure peak GPU memory (and wall-clock) for a representative CVAE forward pass on
the CURRENT runtime, plus an optional, safety-guarded preprocessing wall-clock probe.

* Peak-memory: loads the frozen base package, rebuilds the CVAE and runs a forward
  over a batch of production-shaped inputs, recording
  torch.cuda.max_memory_allocated(). Inputs are synthetic (memory depends on the
  tensor shapes, not their values); this is labelled a SUPPLEMENTAL measurement of
  the current GPU, not a historical figure.
* Preprocessing probe (OFF by default): only runs if --enable-preprocessing-probe
  AND --raw-facescape are given, and it writes to a SCRATCH dir under the output
  (never data/, never --force on the frozen dataset). If raw data is not staged it
  records "preprocessing wall-clock not measurable" and does nothing destructive.

Writes only under --out. Reads the base package read-only.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def measure_peak_memory(base_pkg: Path, device_arg: str, batch: int) -> dict:
    import torch  # noqa: PLC0415

    result: dict = {"label": "SUPPLEMENTAL current-runtime measurement (synthetic-shaped inputs)"}
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    result["device"] = str(device)

    base = torch.load(base_pkg, map_location="cpu", weights_only=False)
    args = base["args"]
    obs_dim = int(np.asarray(base["obs_train_mean"]).shape[1])
    cond_dim = int(np.asarray(base["cond_mean"]).reshape(-1).shape[0])
    vertex_dim = int(base.get("vertex_feature_dim") or 0)
    n_vertices = int(base.get("n_vertices") or base.get("feature_template_vertices", np.zeros((3934, 3))).__len__())
    if not vertex_dim:
        # infer from a stored template if present, else the documented 15 (no subunit features)
        vertex_dim = 15
    result["shapes"] = {"vertex_dim": vertex_dim, "cond_dim": cond_dim,
                        "obs_dim": obs_dim, "n_vertices": n_vertices, "batch_pairs": batch}

    try:
        from rhinoform.train import NeuralFieldCVAE, predict_field  # noqa: PLC0415
        model = NeuralFieldCVAE(vertex_dim, cond_dim, obs_dim,
                                latent_dim=int(args["latent_dim"]), hidden=int(args["hidden"]))
        try:
            model.load_state_dict(base["cvae_state_dict"])
            result["loaded_frozen_weights"] = True
        except Exception as exc:
            result["loaded_frozen_weights"] = False
            result["weight_load_note"] = f"synthetic weights (dim mismatch: {exc!r})"
        model.to(device).eval()
        vertex_feat = np.random.default_rng(0).standard_normal((n_vertices, vertex_dim)).astype(np.float32)
        cond = np.random.default_rng(1).standard_normal((batch, cond_dim)).astype(np.float32)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = predict_field(model, vertex_feat, cond, is_cvae=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        result["forward_wall_sec"] = round(time.perf_counter() - t0, 4)
        if device.type == "cuda":
            result["peak_gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
            result["peak_gpu_mem_reserved_mb"] = round(torch.cuda.max_memory_reserved() / (1024 ** 2), 1)
        else:
            result["peak_gpu_mem_mb"] = None
            result["note"] = "no CUDA on this runtime; peak GPU memory not measurable here"
    except Exception as exc:  # noqa: BLE001
        result["error"] = repr(exc)
    return result


def preprocessing_probe(root: Path, out: Path, raw_facescape: str | None, enabled: bool) -> dict:
    if not enabled or not raw_facescape:
        return {"status": "skipped",
                "reason": "preprocessing wall-clock not measurable "
                          "(no --enable-preprocessing-probe / --raw-facescape). "
                          "build_data is a deterministic skip on the frozen dataset."}
    raw = Path(raw_facescape)
    if not raw.exists():
        return {"status": "skipped", "reason": f"raw FaceScape path not found: {raw}"}
    scratch = out / "preproc_probe_scratch"  # NEVER data/
    scratch.mkdir(parents=True, exist_ok=True)
    import subprocess  # noqa: PLC0415
    cmd = ["python", str(Path(__file__).with_name("build_data.py")),
           "--repo", str(raw), "--out", str(scratch), "--force"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    return {"status": "ran" if proc.returncode == 0 else "failed",
            "scratch_out": str(scratch), "wall_sec": round(elapsed, 2),
            "returncode": proc.returncode,
            "note": "wrote to SCRATCH only; frozen data/ untouched"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--base-model-package", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--enable-preprocessing-probe", action="store_true")
    ap.add_argument("--raw-facescape", default="")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"peak_memory": None, "preprocessing": None}
    try:
        report["peak_memory"] = measure_peak_memory(Path(args.base_model_package), args.device, args.batch)
    except Exception as exc:  # noqa: BLE001
        report["peak_memory"] = {"error": repr(exc)}
    report["preprocessing"] = preprocessing_probe(
        Path(args.repo).resolve().parent, out, args.raw_facescape or None, args.enable_preprocessing_probe)

    (out / "compute_memory.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
