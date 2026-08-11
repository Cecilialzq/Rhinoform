"""Validation selection and one-shot test evaluation for safe residual fusion."""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rhinoform.data import load_rows, rmse_vertices
from rhinoform.safe_fusion import (
    SafetyThresholds,
    count_self_intersections,
    edge_strain,
    evaluate_mesh_safety,
    mesh_edges,
    neutralize_handle_residual,
    project_residual,
    triangle_distortion,
    uniform_safe_projection,
)
from rhinoform.stats import paired_identity_bootstrap
from rhinoform.repro import atomic_write_json, sha256_json


METHODS = ("ridge", "global_hybrid", "rbsr", "arap", "full_fusion", "uniform", "safe_fusion")
BOOTSTRAP_METRICS = ("roi_rmse", "landmark_rmse", "dorsum_rmse", "tip_rmse", "edge_strain_p95", "normal_flip_pct")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def atomic_save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_cache(path: Path) -> tuple[list[tuple[str, str]], dict[str, np.ndarray], dict]:
    marker = path / "CACHE_COMPLETE.json"
    if not marker.exists():
        raise FileNotFoundError(f"Incomplete cache: {marker}")
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    pairs = [(str(a), str(b)) for a, b in json.loads((path / "pairs.json").read_text(encoding="utf-8"))]
    arrays = {
        name: np.load(path / f"{name}.npy", mmap_mode="r")
        for name in ("target_delta", "controls", "ridge", "global_hybrid", "rbsr", "arap")
    }
    arrays["rbf_basis"] = np.load(path / "rbf_basis.npy", mmap_mode="r")
    if any(len(arrays[name]) != len(pairs) for name in ("target_delta", "controls", "ridge", "global_hybrid", "rbsr", "arap")):
        raise ValueError(f"Cache arrays and pair list disagree in {path}")
    return pairs, arrays, metadata


def candidate_configs() -> list[dict]:
    geometry = [
        ("permissive", 0.005, 0.03, 6.0, 2.0),
        ("balanced", 0.010, 0.05, 4.0, 1.0),
        ("strict", 0.020, 0.10, 3.0, 0.75),
    ]
    schedules = [
        ("local_fast", 0.70, 1),
        ("local_smooth", 0.75, 2),
    ]
    return [
        {
            "name": f"{name}_{schedule}_{mode}",
            "neutralization": mode,
            "min_area_ratio": area,
            "min_sigma": min_sigma,
            "max_sigma": max_sigma,
            "max_edge_strain": max_strain,
            "attenuation": attenuation,
            "smoothing_steps": smoothing,
            "max_iterations": 30,
            "uniform_steps": 101,
            "allow_baseline_violations": True,
            "max_normal_rotation_deg": None,
        }
        for name, area, min_sigma, max_sigma, max_strain in geometry
        for schedule, attenuation, smoothing in schedules
        for mode in ("hard_zero", "rbf")
    ]


def thresholds_from_config(config: dict) -> SafetyThresholds:
    fields = set(SafetyThresholds.__dataclass_fields__)
    return SafetyThresholds(**{key: value for key, value in config.items() if key in fields})


def prepare_residual(
    proposal_delta: np.ndarray,
    base_delta: np.ndarray,
    handles: np.ndarray,
    mode: str,
    rbf_basis: np.ndarray,
) -> np.ndarray:
    residual = np.asarray(proposal_delta, dtype=np.float64) - np.asarray(base_delta, dtype=np.float64)
    basis = rbf_basis if mode == "rbf" else None
    return neutralize_handle_residual(residual, handles, basis)


def prediction_metrics(
    pair_index: int,
    source_id: str,
    target_id: str,
    source: np.ndarray,
    true_delta: np.ndarray,
    pred_delta: np.ndarray,
    controls: np.ndarray,
    faces: np.ndarray,
    edges: np.ndarray,
    handles: np.ndarray,
    subunits: dict,
    thresholds: SafetyThresholds,
    baseline_delta: np.ndarray,
    method: str,
    status: str = "not_applicable",
    retention: float = float("nan"),
    iterations: int = 0,
) -> dict:
    edited = source + pred_delta
    distortion = triangle_distortion(source, edited, faces)
    strain = edge_strain(source, edited, edges)
    absolute = evaluate_mesh_safety(source, edited, faces, thresholds)
    relative = evaluate_mesh_safety(source, edited, faces, thresholds, source + baseline_delta)
    return {
        "pair_index": pair_index,
        "source_id": source_id,
        "target_id": target_id,
        "method": method,
        "roi_rmse": rmse_vertices(pred_delta, true_delta),
        "landmark_rmse": rmse_vertices(pred_delta, true_delta, handles),
        "control_rmse": rmse_vertices(pred_delta[handles], controls),
        "dorsum_rmse": rmse_vertices(pred_delta, true_delta, np.asarray(subunits["dorsum"], dtype=np.int64)),
        "tip_rmse": rmse_vertices(pred_delta, true_delta, np.asarray(subunits["tip"], dtype=np.int64)),
        "edge_strain_p95": float(np.percentile(strain, 95)),
        "edge_strain_max": float(np.max(strain)),
        "area_ratio_p01": float(np.percentile(distortion.area_ratio, 1)),
        "area_ratio_min": float(np.min(distortion.area_ratio)),
        "sigma_min_p01": float(np.percentile(distortion.sigma_min, 1)),
        "sigma_min": float(np.min(distortion.sigma_min)),
        "sigma_max_p99": float(np.percentile(distortion.sigma_max, 99)),
        "sigma_max": float(np.max(distortion.sigma_max)),
        "normal_flip_pct": float(np.mean(distortion.normal_cosine < 0.0) * 100.0),
        "absolute_certified": int(absolute.certified),
        "relative_certified": int(relative.certified),
        "unsafe_face_count": relative.unsafe_face_count,
        "unsafe_edge_count": relative.unsafe_edge_count,
        "status": status,
        "retention": retention,
        "iterations": iterations,
    }


def validate_config(
    config: dict,
    pairs: list[tuple[str, str]],
    arrays: dict[str, np.ndarray],
    by_id: dict,
    faces: np.ndarray,
    handles: np.ndarray,
    rbf_basis: np.ndarray,
    proposal: str,
) -> dict:
    thresholds = thresholds_from_config(config)
    roi, strain, retention = [], [], []
    absolute, relative, fallback = [], [], []
    statuses: Counter[str] = Counter()
    edges = mesh_edges(faces)
    for index, (source_id, target_id) in enumerate(pairs):
        source = np.asarray(by_id[source_id]["vertices"], dtype=np.float64)
        true_delta = np.asarray(arrays["target_delta"][index], dtype=np.float64).reshape(-1, 3)
        base_delta = np.asarray(arrays["arap"][index], dtype=np.float64).reshape(-1, 3)
        proposal_delta = np.asarray(arrays[proposal][index], dtype=np.float64).reshape(-1, 3)
        residual = prepare_residual(proposal_delta, base_delta, handles, config["neutralization"], rbf_basis)
        result = project_residual(source, base_delta, residual, faces, handles, thresholds)
        distortion_strain = edge_strain(source, source + result.delta, edges)
        roi.append(rmse_vertices(result.delta, true_delta))
        strain.append(float(np.percentile(distortion_strain, 95)))
        retention.append(result.retention)
        absolute.append(result.absolute_certified)
        relative.append(result.certified)
        fallback.append(result.status.startswith("local_failed") or result.status == "base_fallback")
        statuses[result.status] += 1
    mean_roi = float(np.mean(roi))
    mean_strain = float(np.mean(strain))
    mean_retention = float(np.mean(retention))
    fallback_rate = float(np.mean(fallback))
    selection_score = mean_roi + 0.25 * mean_strain + 0.10 * (1.0 - mean_retention) + 0.25 * fallback_rate
    return {
        "name": config["name"],
        "neutralization": config["neutralization"],
        "n_pairs": len(pairs),
        "roi_rmse": mean_roi,
        "edge_strain_p95": mean_strain,
        "retention": mean_retention,
        "relative_certified_rate": float(np.mean(relative)),
        "absolute_certified_rate": float(np.mean(absolute)),
        "fallback_rate": fallback_rate,
        "selection_score": selection_score,
        "statuses": json.dumps(statuses, sort_keys=True),
    }


def select_config(configs: list[dict], rows: list[dict]) -> tuple[dict, dict]:
    eligible = [row for row in rows if row["relative_certified_rate"] >= 0.99]
    pool = eligible or rows
    selected_row = min(pool, key=lambda row: (row["fallback_rate"], row["selection_score"], -row["retention"]))
    selected = next(config for config in configs if config["name"] == selected_row["name"])
    return selected, selected_row


def evaluate_test(
    selected: dict,
    pairs: list[tuple[str, str]],
    arrays: dict[str, np.ndarray],
    by_id: dict,
    faces: np.ndarray,
    handles: np.ndarray,
    subunits: dict,
    rbf_basis: np.ndarray,
    proposal: str,
    out: Path,
    checkpoint_every: int,
    checkpoint_signature: str,
) -> tuple[list[dict], dict[str, Path]]:
    thresholds = thresholds_from_config(selected)
    edges = mesh_edges(faces)
    n_vertices = len(next(iter(by_id.values()))["vertices"])
    checkpoint_dir = out / "checkpoints" / f"test_{checkpoint_signature[:16]}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = out / "test_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: prediction_dir / f"{name}.npy" for name in ("full_fusion", "uniform", "safe_fusion")}
    chunk_size = max(1, int(checkpoint_every))
    for start in range(0, len(pairs), chunk_size):
        stop = min(len(pairs), start + chunk_size)
        shard = checkpoint_dir / f"pairs_{start:06d}_{stop:06d}.npz"
        if shard.exists():
            print(f"Test checkpoint reused: {stop}/{len(pairs)}", flush=True)
            continue
        chunk_rows: list[dict] = []
        chunk_predictions = {
            name: np.empty((stop - start, n_vertices * 3), dtype=np.float32)
            for name in paths
        }
        for local, index in enumerate(range(start, stop)):
            source_id, target_id = pairs[index]
            source = np.asarray(by_id[source_id]["vertices"], dtype=np.float64)
            true_delta = np.asarray(arrays["target_delta"][index], dtype=np.float64).reshape(-1, 3)
            controls = np.asarray(arrays["controls"][index], dtype=np.float64).reshape(-1, 3)
            base_delta = np.asarray(arrays["arap"][index], dtype=np.float64).reshape(-1, 3)
            proposal_delta = np.asarray(arrays[proposal][index], dtype=np.float64).reshape(-1, 3)
            residual = prepare_residual(proposal_delta, base_delta, handles, selected["neutralization"], rbf_basis)
            full_delta = base_delta + residual
            uniform = uniform_safe_projection(source, base_delta, residual, faces, handles, thresholds)
            safe = project_residual(source, base_delta, residual, faces, handles, thresholds)
            generated = {
                "full_fusion": (full_delta, "unprojected", 1.0, 0),
                "uniform": (uniform.delta, uniform.status, uniform.retention, uniform.iterations),
                "safe_fusion": (safe.delta, safe.status, safe.retention, safe.iterations),
            }
            for method in ("ridge", "global_hybrid", "rbsr", "arap"):
                delta = np.asarray(arrays[method][index], dtype=np.float64).reshape(-1, 3)
                chunk_rows.append(prediction_metrics(
                    index, source_id, target_id, source, true_delta, delta, controls, faces, edges,
                    handles, subunits, thresholds, base_delta, method,
                ))
            for method, (delta, status, retained, iterations) in generated.items():
                chunk_predictions[method][local] = np.asarray(delta, dtype=np.float32).reshape(-1)
                chunk_rows.append(prediction_metrics(
                    index, source_id, target_id, source, true_delta, delta, controls, faces, edges,
                    handles, subunits, thresholds, base_delta, method, status, retained, iterations,
                ))
        atomic_save_npz(
            shard,
            rows_json=np.asarray(json.dumps(chunk_rows)),
            **chunk_predictions,
        )
        print(f"Test checkpoint saved: {stop}/{len(pairs)}", flush=True)

    rows: list[dict] = []
    writers = {
        name: np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(pairs), n_vertices * 3))
        for name, path in paths.items()
    }
    for start in range(0, len(pairs), chunk_size):
        stop = min(len(pairs), start + chunk_size)
        shard = np.load(checkpoint_dir / f"pairs_{start:06d}_{stop:06d}.npz", allow_pickle=False)
        rows.extend(json.loads(str(shard["rows_json"])))
        for name, writer in writers.items():
            writer[start:stop] = shard[name]
    for writer in writers.values():
        writer.flush()
    return rows, paths


def summarize(rows: list[dict]) -> list[dict]:
    numeric = [
        "roi_rmse", "landmark_rmse", "control_rmse", "dorsum_rmse", "tip_rmse",
        "edge_strain_p95", "edge_strain_max", "area_ratio_p01", "area_ratio_min",
        "sigma_min_p01", "sigma_min", "sigma_max_p99", "sigma_max", "normal_flip_pct",
        "absolute_certified", "relative_certified", "retention", "iterations",
    ]
    summary = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        row = {"method": method, "n_pairs": len(selected)}
        for metric in numeric:
            values = np.asarray([float(item[metric]) for item in selected], dtype=np.float64)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(finite)) if len(finite) else float("nan")
            row[f"{metric}_p95"] = float(np.percentile(finite, 95)) if len(finite) else float("nan")
        row["fallback_rate"] = float(np.mean([
            str(item["status"]).startswith("local_failed") or item["status"] == "base_fallback"
            for item in selected
        ]))
        row["statuses"] = json.dumps(Counter(str(item["status"]) for item in selected), sort_keys=True)
        summary.append(row)
    return summary


def bootstrap(rows: list[dict], n_boot: int, seed: int) -> list[dict]:
    grouped = {method: [row for row in rows if row["method"] == method] for method in METHODS}
    output = []
    for baseline in ("arap", "rbsr", "full_fusion", "uniform"):
        for metric in BOOTSTRAP_METRICS:
            output.append(paired_identity_bootstrap(
                grouped[baseline], grouped["safe_fusion"], "safe_fusion", baseline,
                metric, "source_id", n_boot, seed + len(output),
            ))
    return output


def self_intersection_audit(
    rows: list[dict], pairs: list[tuple[str, str]], arrays: dict[str, np.ndarray], by_id: dict,
    faces: np.ndarray, prediction_paths: dict[str, Path], cases: int, seed: int,
    checkpoint_dir: Path,
) -> list[dict]:
    if cases <= 0:
        return []
    safe_rows = [row for row in rows if row["method"] == "safe_fusion"]
    ordered = sorted(safe_rows, key=lambda row: (float(row["sigma_min"]), -float(row["edge_strain_max"])))
    worst_count = min((cases + 1) // 2, len(ordered))
    indices = [int(row["pair_index"]) for row in ordered[:worst_count]]
    remaining = [int(row["pair_index"]) for row in ordered if int(row["pair_index"]) not in indices]
    random_count = min(cases - len(indices), len(remaining))
    if random_count:
        rng = np.random.default_rng(seed)
        indices.extend(sorted(rng.choice(remaining, size=random_count, replace=False).tolist()))
    predictions = {name: np.load(path, mmap_mode="r") for name, path in prediction_paths.items()}
    audit = []
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for index in indices:
        checkpoint_path = checkpoint_dir / f"pair_{index:06d}.json"
        if checkpoint_path.exists():
            audit.extend(json.loads(checkpoint_path.read_text(encoding="utf-8")))
            print(f"Self-intersection checkpoint reused: pair {index}", flush=True)
            continue
        source_id, target_id = pairs[index]
        source = np.asarray(by_id[source_id]["vertices"], dtype=np.float64)
        method_delta = {
            "arap": np.asarray(arrays["arap"][index]).reshape(-1, 3),
            "rbsr": np.asarray(arrays["rbsr"][index]).reshape(-1, 3),
            **{name: np.asarray(values[index]).reshape(-1, 3) for name, values in predictions.items()},
        }
        pair_rows = []
        for method, delta in method_delta.items():
            pair_rows.append({
                "pair_index": index,
                "source_id": source_id,
                "target_id": target_id,
                "method": method,
                "new_self_intersection_detected": int(
                    count_self_intersections(
                        source + delta, faces, stop_after=1, baseline_vertices=source
                    ) > 0
                ),
            })
        atomic_write_json(checkpoint_path, pair_rows)
        audit.extend(pair_rows)
        print(f"Self-intersection audit: pair {index}", flush=True)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proposal", choices=["global_hybrid", "rbsr"], default="rbsr")
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--self-intersection-cases", type=int, default=20)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    _, by_id = load_rows(args.repo)
    validation_pairs, validation_arrays, validation_metadata = load_cache(args.validation_cache)
    test_pairs, test_arrays, test_metadata = load_cache(args.test_cache)
    if validation_metadata.get("split_manifest_sha256") != test_metadata.get("split_manifest_sha256"):
        raise RuntimeError("Validation and test caches use different split manifests")
    if validation_metadata.get("feature_template_sha256") != test_metadata.get("feature_template_sha256"):
        raise RuntimeError("Validation and test caches use different feature templates")
    if not np.allclose(validation_arrays["rbf_basis"], test_arrays["rbf_basis"], atol=1e-7, rtol=1e-7):
        raise RuntimeError("Validation and test caches use different RBF projection bases")
    template = by_id[validation_pairs[0][0]]
    faces = np.asarray(template["faces"], dtype=np.int64)
    handles = np.asarray(template["landmarks"], dtype=np.int64)
    subunits = template["subunits"]
    rbf_basis = np.asarray(validation_arrays["rbf_basis"], dtype=np.float64)

    configs = candidate_configs()
    if args.max_configs > 0:
        configs = configs[: args.max_configs]
    validation_signature = sha256_json({
        "cache": validation_metadata.get("run_signature"), "configs": configs, "proposal": args.proposal,
    })
    validation_checkpoint_dir = args.out / "checkpoints" / f"validation_{validation_signature[:16]}"
    validation_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    validation_rows = []
    for index, config in enumerate(configs):
        checkpoint_path = validation_checkpoint_dir / f"{config['name']}.json"
        if checkpoint_path.exists():
            validation_rows.append(json.loads(checkpoint_path.read_text(encoding="utf-8")))
            print(f"Validation checkpoint reused: {config['name']}", flush=True)
            continue
        print(f"Validation config {index + 1}/{len(configs)}: {config['name']}", flush=True)
        result = validate_config(
            config, validation_pairs, validation_arrays, by_id, faces, handles, rbf_basis, args.proposal
        )
        atomic_write_json(checkpoint_path, result)
        validation_rows.append(result)
    selected, selected_row = select_config(configs, validation_rows)
    write_csv(args.out / "validation_grid.csv", validation_rows)
    atomic_write_json(args.out / "selected_config.json", {
        "selection_rule": "relative_certified_rate >= 0.99; then minimum fallback rate, selection score, and maximum retention",
        "selection_score": "roi_rmse + 0.25*edge_strain_p95 + 0.10*(1-retention) + 0.25*fallback_rate",
        "selected": selected,
        "validation_result": selected_row,
        "validation_cache": validation_metadata,
    })
    print(f"Selected: {selected['name']}", flush=True)

    test_signature = sha256_json({
        "cache": test_metadata.get("run_signature"), "selected": selected, "proposal": args.proposal,
        "checkpoint_every": args.checkpoint_every,
    })
    rows, prediction_paths = evaluate_test(
        selected, test_pairs, test_arrays, by_id, faces, handles, subunits,
        rbf_basis, args.proposal, args.out, args.checkpoint_every, test_signature,
    )
    summary = summarize(rows)
    bootstrap_rows = bootstrap(rows, args.n_boot, args.seed)
    audit_rows = self_intersection_audit(
        rows, test_pairs, test_arrays, by_id, faces, prediction_paths, args.self_intersection_cases, args.seed,
        args.out / "checkpoints" / f"self_intersection_{test_signature[:16]}",
    )
    write_csv(args.out / "test_pair_metrics.csv", rows)
    write_csv(args.out / "test_summary.csv", summary)
    write_csv(args.out / "bootstrap_summary.csv", bootstrap_rows)
    write_csv(args.out / "self_intersection_audit.csv", audit_rows)
    report = {
        "status": "complete",
        "selected_config": selected,
        "selected_validation_result": selected_row,
        "proposal": args.proposal,
        "test_cache": test_metadata,
        "test_summary": summary,
        "bootstrap": bootstrap_rows,
        "self_intersection_audit": audit_rows,
        "self_intersection_note": "The bounded audit uses half worst-conditioned and half seeded-random cases. It detects intersections newly introduced relative to the source mesh; pre-existing FaceScape contacts are excluded.",
        "metric_note": "normal_flip_pct is source-normal reversal calibration, not a topological flip certificate.",
        "certificate_note": "relative certification proves the learned residual introduces no new threshold violation beyond the frozen ARAP base; absolute certification is reported separately.",
        "thresholds": asdict(thresholds_from_config(selected)),
    }
    atomic_write_json(args.out / "FINAL_REPORT.json", report)
    print(json.dumps({"status": "complete", "selected": selected["name"], "n_test": len(test_pairs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
