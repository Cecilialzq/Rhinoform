from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rhinoform.baselines import arap_predict_vectorised
from rhinoform.data import load_rows
from rhinoform.geometry import solve_linear_handle_baseline, uniform_laplacian
from scripts.evaluation.noise_robustness import controls_and_sources
from rhinoform.repro import atomic_write_json, sha256_file
from rhinoform.stats import metric_rows_for_method, write_csv


METRICS = ["roi_rmse", "landmark_rmse", "dorsum_rmse", "tip_rmse", "edge_strain_p95", "normal_flip_pct"]


def aggregate(rows: list[dict]) -> dict[str, float]:
    return {name: float(np.mean([float(row[name]) for row in rows])) for name in METRICS}


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute ARAP noise rows with the validation-selected ARAP initializer.")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--geometric-tuning", type=Path, required=True)
    parser.add_argument("--noise-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--levels", default="0,0.05,0.10,0.20")
    parser.add_argument("--seeds", default="20260609,20260610,20260611")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    _, by_id = load_rows(args.repo)
    package = torch.load(args.model_package, map_location="cpu", weights_only=False)
    tuning = json.loads(args.geometric_tuning.read_text(encoding="utf-8"))
    prior_summary = json.loads(args.noise_summary.read_text(encoding="utf-8"))
    cfg = tuning["selected"]["arap"]
    test_pairs = [(str(a), str(b)) for a, b in package["test_pairs"]]
    template = next(iter(by_id.values()))
    faces = np.asarray(template["faces"], dtype=np.int32)
    landmarks = np.asarray(template["landmarks"], dtype=np.int64)
    controls, _ = controls_and_sources(by_id, test_pairs, landmarks)
    lap = uniform_laplacian(len(template["vertices"]), faces)
    source_meshes = [by_id[source_id]["vertices"] for source_id, _ in test_pairs]
    median_control_delta = float(prior_summary["median_control_delta_from_validation"])
    levels = [float(value) for value in args.levels.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    summary_rows: list[dict] = []
    zero_rows: list[dict] | None = None

    for level in levels:
        for seed in seeds:
            sigma = float(level * median_control_delta)
            if level == 0.0 and zero_rows is not None:
                rows = zero_rows
            else:
                rng = np.random.default_rng(seed)
                noisy_controls = controls + rng.normal(0.0, sigma, size=controls.shape)
                init = solve_linear_handle_baseline(
                    noisy_controls,
                    landmarks,
                    len(template["vertices"]),
                    lap,
                    float(cfg["handle_weight"]),
                    float(cfg.get("system_ridge") or 1e-8),
                )
                prediction = arap_predict_vectorised(
                    source_meshes,
                    noisy_controls,
                    landmarks,
                    faces,
                    lap,
                    init,
                    float(cfg["handle_weight"]),
                    float(cfg.get("system_ridge") or 1e-8),
                    int(cfg["arap_iter"]),
                )
                rows = metric_rows_for_method(by_id, test_pairs, prediction)
                if level == 0.0:
                    zero_rows = rows
            pair_path = args.out / f"noise_c{level:g}_seed{seed}_arap_corrected.csv"
            write_csv(pair_path, rows)
            summary_rows.append(
                {
                    "method": "arap_corrected",
                    "noise_level_c": level,
                    "noise_fraction": level,
                    "noise_seed": seed,
                    "sigma": sigma,
                    **aggregate(rows),
                }
            )
            print(f"level={level:g} seed={seed} roi={summary_rows[-1]['roi_rmse']:.6f}", flush=True)

    write_csv(args.out / "noise_arap_corrected_summary.csv", summary_rows)
    report = {
        "status": "replacement_rows_for_arap_only",
        "reason": "Original noise script initialized ARAP from the Laplacian-selected configuration instead of the ARAP-selected configuration.",
        "selected_arap": cfg,
        "median_control_delta_from_validation": median_control_delta,
        "rows": summary_rows,
        "inputs": {
            "model_package_sha256": sha256_file(args.model_package),
            "geometric_tuning_sha256": sha256_file(args.geometric_tuning),
            "noise_summary_sha256": sha256_file(args.noise_summary),
        },
    }
    atomic_write_json(args.out / "noise_arap_corrected_summary.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
