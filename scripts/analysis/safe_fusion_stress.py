"""Stress-test the selected safe-fusion rule without changing its thresholds."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rhinoform.data import load_rows, rmse_vertices
from scripts.evaluation.safe_fusion import atomic_save_npz, load_cache, prepare_residual, thresholds_from_config
from rhinoform.repro import atomic_write_json, sha256_json
from rhinoform.safe_fusion import edge_strain, mesh_edges, project_residual, triangle_distortion


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proposal", choices=["global_hybrid", "rbsr"], default="rbsr")
    parser.add_argument("--pair-budget", type=int, default=800)
    parser.add_argument("--scales", default="0.5,1.0,1.25,1.5,2.0")
    parser.add_argument("--noise-fractions", default="0.0,0.05,0.1,0.2")
    parser.add_argument("--noise-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    _, by_id = load_rows(args.repo)
    pairs, arrays, cache_metadata = load_cache(args.test_cache)
    template = by_id[pairs[0][0]]
    faces = np.asarray(template["faces"], dtype=np.int64)
    edges = mesh_edges(faces)
    handles = np.asarray(template["landmarks"], dtype=np.int64)
    rbf_basis = np.asarray(arrays["rbf_basis"], dtype=np.float64)
    selected_payload = json.loads(args.selected_config.read_text(encoding="utf-8"))
    selected = selected_payload["selected"]
    thresholds = thresholds_from_config(selected)

    rng = np.random.default_rng(args.seed)
    if 0 < args.pair_budget < len(pairs):
        indices = sorted(rng.choice(len(pairs), size=args.pair_budget, replace=False).tolist())
    else:
        indices = list(range(len(pairs)))
    scales = parse_floats(args.scales)
    noise_fractions = parse_floats(args.noise_fractions)
    checkpoint_signature = sha256_json({
        "cache": cache_metadata.get("run_signature"),
        "selected": selected,
        "indices": indices,
        "scales": scales,
        "noise_fractions": noise_fractions,
        "noise_seeds": args.noise_seeds,
        "seed": args.seed,
        "proposal": args.proposal,
        "checkpoint_every": args.checkpoint_every,
    })
    checkpoint_dir = args.out / "checkpoints" / checkpoint_signature[:16]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = max(1, int(args.checkpoint_every))
    for scale in scales:
        for noise_fraction in noise_fractions:
            for repeat in range(args.noise_seeds):
                setting_seed = args.seed + 10000 * repeat + int(scale * 1000) + int(noise_fraction * 100000)
                print(
                    f"Stress scale={scale:g} noise={noise_fraction:g} repeat={repeat + 1}/{args.noise_seeds}",
                    flush=True,
                )
                setting_tag = f"s{scale:g}_n{noise_fraction:g}_r{repeat}".replace(".", "p")
                for chunk_start in range(0, len(indices), chunk_size):
                    chunk_stop = min(len(indices), chunk_start + chunk_size)
                    shard = checkpoint_dir / f"{setting_tag}_{chunk_start:06d}_{chunk_stop:06d}.npz"
                    if shard.exists():
                        print(f"Stress checkpoint reused: {setting_tag} {chunk_stop}/{len(indices)}", flush=True)
                        continue
                    chunk_rows = []
                    for index in indices[chunk_start:chunk_stop]:
                        source_id, target_id = pairs[index]
                        source = np.asarray(by_id[source_id]["vertices"], dtype=np.float64)
                        true_delta = np.asarray(arrays["target_delta"][index], dtype=np.float64).reshape(-1, 3)
                        base_delta = np.asarray(arrays["arap"][index], dtype=np.float64).reshape(-1, 3)
                        proposal_delta = np.asarray(arrays[args.proposal][index], dtype=np.float64).reshape(-1, 3)
                        residual = prepare_residual(
                            proposal_delta, base_delta, handles, selected["neutralization"], rbf_basis
                        )
                        residual_rms = float(np.sqrt(np.mean(residual**2)))
                        pair_rng = np.random.default_rng(setting_seed + 1000003 * int(index))
                        noise = pair_rng.normal(size=residual.shape) * residual_rms * noise_fraction
                        stressed = prepare_residual(
                            base_delta + scale * residual + noise, base_delta, handles,
                            selected["neutralization"], rbf_basis,
                        )
                        full_delta = base_delta + stressed
                        result = project_residual(source, base_delta, stressed, faces, handles, thresholds)
                        for method, delta, status, retention, certified in (
                            ("unprojected", full_delta, "unprojected", 1.0, False),
                            ("safe_fusion", result.delta, result.status, result.retention, result.certified),
                        ):
                            edited = source + delta
                            distortion = triangle_distortion(source, edited, faces)
                            strain = edge_strain(source, edited, edges)
                            chunk_rows.append({
                                "pair_index": index, "source_id": source_id, "target_id": target_id,
                                "method": method, "residual_scale": scale,
                                "noise_fraction": noise_fraction, "repeat": repeat,
                                "setting_seed": setting_seed, "roi_rmse": rmse_vertices(delta, true_delta),
                                "edge_strain_p95": float(np.percentile(strain, 95)),
                                "edge_strain_max": float(np.max(strain)),
                                "area_ratio_min": float(np.min(distortion.area_ratio)),
                                "sigma_min": float(np.min(distortion.sigma_min)),
                                "sigma_max": float(np.max(distortion.sigma_max)),
                                "normal_flip_pct": float(np.mean(distortion.normal_cosine < 0.0) * 100.0),
                                "certified": int(certified), "retention": retention, "status": status,
                                "fallback": int(status.startswith("local_failed") or status == "base_fallback"),
                            })
                    atomic_save_npz(shard, rows_json=np.asarray(json.dumps(chunk_rows)))
                    print(f"Stress checkpoint saved: {setting_tag} {chunk_stop}/{len(indices)}", flush=True)

    rows: list[dict] = []
    for scale in scales:
        for noise_fraction in noise_fractions:
            for repeat in range(args.noise_seeds):
                setting_tag = f"s{scale:g}_n{noise_fraction:g}_r{repeat}".replace(".", "p")
                for chunk_start in range(0, len(indices), chunk_size):
                    chunk_stop = min(len(indices), chunk_start + chunk_size)
                    shard = np.load(
                        checkpoint_dir / f"{setting_tag}_{chunk_start:06d}_{chunk_stop:06d}.npz",
                        allow_pickle=False,
                    )
                    rows.extend(json.loads(str(shard["rows_json"])))

    summary = []
    for method in ("unprojected", "safe_fusion"):
        for scale in scales:
            for noise_fraction in noise_fractions:
                selected_rows = [
                    row for row in rows
                    if row["method"] == method
                    and row["residual_scale"] == scale
                    and row["noise_fraction"] == noise_fraction
                ]
                summary.append({
                    "method": method,
                    "residual_scale": scale,
                    "noise_fraction": noise_fraction,
                    "n": len(selected_rows),
                    "roi_rmse_mean": float(np.mean([row["roi_rmse"] for row in selected_rows])),
                    "edge_strain_p95_mean": float(np.mean([row["edge_strain_p95"] for row in selected_rows])),
                    "area_ratio_min_p05": float(np.percentile([row["area_ratio_min"] for row in selected_rows], 5)),
                    "sigma_min_p05": float(np.percentile([row["sigma_min"] for row in selected_rows], 5)),
                    "sigma_max_p95": float(np.percentile([row["sigma_max"] for row in selected_rows], 95)),
                    "normal_flip_pct_mean": float(np.mean([row["normal_flip_pct"] for row in selected_rows])),
                    "certified_rate": float(np.mean([row["certified"] for row in selected_rows])),
                    "retention_mean": float(np.mean([row["retention"] for row in selected_rows])),
                    "fallback_rate": float(np.mean([row["fallback"] for row in selected_rows])),
                })

    write_csv(args.out / "stress_pair_metrics.csv", rows)
    write_csv(args.out / "stress_summary.csv", summary)
    report = {
        "status": "complete",
        "selected_config": selected,
        "cache": cache_metadata,
        "pair_indices": indices,
        "scales": scales,
        "noise_fractions": noise_fractions,
        "noise_seeds": args.noise_seeds,
        "stress_definition": "Scale and isotropic Gaussian perturbation are applied to the handle-neutral learned residual, not to the frozen ARAP base.",
        "summary": summary,
    }
    atomic_write_json(args.out / "STRESS_REPORT.json", report)
    print(json.dumps({"status": "complete", "n_rows": len(rows), "n_settings": len(summary)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
