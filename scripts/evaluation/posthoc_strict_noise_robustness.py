"""Frozen-validation control-noise robustness for four deployment methods.

Noise perturbs the nine input controls only.  Before scoring, every method is
evaluated by ``strict_metric_rows``, which overwrites the nine output controls
with the unnoised true target controls, excludes them from free-ROI RMSE, and
uses target-relative new flip.  The old absolute flip / nonzero landmark-RMSE
noise tables are provenance context only and are never imported as results.
All 4,830 frozen validation pairs are evaluated at 0, 0.25, 0.5 and 1.0
millimetres with three fixed nonzero-noise seeds and no operating-point update.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rhinoform.baselines import arap_predict_vectorised
from rhinoform.confirmation import validate_frozen_classical_configuration
from rhinoform.data import load_rows, ridge_predict
from rhinoform.geometry import solve_linear_handle_baseline, uniform_laplacian
from rhinoform.rbsr_calibration import enforce_exact_controls
from rhinoform.repro import (
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
    sha256_json,
    valid_sha256_sidecar,
    write_sha256_sidecar,
)
from rhinoform.safe_fusion import project_residual_no_new_ridge_folds
from rhinoform.strict_protocol_patch import strict_metric_rows
from rhinoform.train import (
    NeuralFieldCVAE,
    assert_feature_template_package,
    build_static_vertex_features,
    predict_field,
)
from rhinoform.train_rbsr_gate import SpatialRiskGate, predict_gate_batches
from scripts.evaluation.direct_paired_statistics import holm, safe_wilcoxon
from scripts.evaluation.lamm_validation_reference import (
    relocate_comparable_model_config,
    require_frozen_result_artifact,
)
from scripts.evaluation.posthoc_subunit_analysis import (
    LAMM_COMMIT,
    Context,
    read_csv,
    require_valid,
)


ROBUSTNESS_METHODS = ("ridge", "certified_rbsr", "lamm", "arap")
NOISE_LEVELS = (0.0, 0.25, 0.5, 1.0)
NOISE_SEEDS = (20260609, 20260610, 20260611)
NOISE_METRICS = (
    "roi_rmse",
    "normal_flip_pct",
    "edge_strain_p95",
)


def deterministic_noise_draw(
    level_mm: float,
    noise_seed: int,
    n_pairs: int,
    n_landmarks: int,
) -> np.ndarray:
    """Return the protocol-frozen full-panel draw, independent of chunking."""
    rng = np.random.default_rng(noise_seed)
    return rng.normal(
        0.0,
        float(level_mm),
        size=(n_pairs, n_landmarks, 3),
    ).astype(np.float64, copy=False)


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def output_valid(path: Path, expected_rows: int) -> bool:
    if not valid_sha256_sidecar(path):
        return False
    try:
        return len(read_csv(path)) == expected_rows
    except (OSError, csv.Error):
        return False


def noisy_condition(
    noisy_controls: np.ndarray,
    source: np.ndarray,
    base: dict,
) -> np.ndarray:
    source_flat = source.reshape(len(source), -1)
    source_code = (
        source_flat - np.asarray(base["source_pca"]["mean"])
    ) @ np.asarray(base["source_pca"]["components"]).T
    condition = np.concatenate([noisy_controls.reshape(len(source), -1), source_code], axis=1)
    return ((condition - base["cond_mean"]) / base["cond_std"]).astype(np.float32)


def compare_to_canonical(
    observed: list[dict[str, object]],
    canonical: list[dict[str, str]],
    indices: list[int],
    atol: float = 1e-5,
) -> None:
    expected = [canonical[index] for index in indices]
    for row_index, (left, right) in enumerate(zip(expected, observed)):
        if (left["source_id"], left["target_id"]) != (
            str(right["source_id"]), str(right["target_id"])
        ):
            raise AssertionError(f"Zero-noise pair mismatch at {row_index}")
        for metric in NOISE_METRICS:
            if not np.isclose(float(left[metric]), float(right[metric]), rtol=0.0, atol=atol):
                raise AssertionError(
                    f"Zero-noise self-check failed method metric={metric}: "
                    f"{left[metric]} != {right[metric]}"
                )


def load_canonical(context: Context) -> dict[str, list[dict[str, str]]]:
    projection = context.results / (
        "rbsr/seed20260609/certified_projection_validation/pair_metrics"
    )
    paths = {
        "ridge": projection / "pair_metrics_ridge_validation.csv",
        "certified_rbsr": projection / "pair_metrics_attenuation_0p75_validation.csv",
    }
    output = {}
    for method, path in paths.items():
        require_valid(path, f"canonical clean metrics for {method}")
        rows = read_csv(path)
        if len(rows) != 4830:
            raise RuntimeError(f"Incomplete canonical clean metrics: {path}")
        output[method] = rows
    return output


def prepare_neural(context: Context, all_by_id: dict[str, dict], device: torch.device):
    base = context.base
    train_ids = [str(value) for value in base["train_ids"]]
    features, static = build_static_vertex_features(
        all_by_id,
        train_ids,
        use_subunit_features=bool(base["use_subunit_features"]),
    )
    assert_feature_template_package(base, static, "Frozen base package")
    obs_dim = int(np.asarray(base["obs_train_mean"]).shape[1])
    args = base["args"]
    cvae = NeuralFieldCVAE(
        features.shape[1],
        int(np.asarray(base["cond_mean"]).shape[1]),
        obs_dim,
        latent_dim=int(args["latent_dim"]),
        hidden=int(args["hidden"]),
    ).to(device)
    cvae.load_state_dict(base["cvae_state_dict"], strict=True)
    cvae.eval()

    gate_path = context.results / "rbsr/seed20260609/gate/rbsr_gate_model.pt"
    require_valid(gate_path, "frozen RB-SR gate")
    if sha256_file(gate_path) != context.freeze["rbsr_gate_sha256"]:
        raise RuntimeError("RB-SR gate differs from the pre-test freeze")
    gate_package = torch.load(gate_path, map_location="cpu", weights_only=False)
    gate_args = gate_package["args"]
    gate = SpatialRiskGate(
        int(gate_package["gate_vertex_dim"]),
        int(gate_package["gate_cond_dim"]),
        int(gate_args["hidden"]),
        float(gate_package.get("warm_start") or base.get("selected_alpha") or 0.25),
    ).to(device)
    gate.load_state_dict(gate_package["gate_state_dict"], strict=True)
    gate.eval()
    projection_path = (
        context.results
        / "rbsr/seed20260609/certified_projection_validation/RBSR_CERTIFIED_PROJECTION_FREEZE.json"
    )
    require_valid(projection_path, "frozen certified projection")
    if sha256_file(projection_path) != context.freeze["rbsr_projection_freeze_sha256"]:
        raise RuntimeError("Projection freeze differs from the pre-test freeze")
    projection = json.loads(projection_path.read_text(encoding="utf-8"))["selected"]
    return features, cvae, gate_package, gate, projection


def prepare_lamm(context: Context, device: torch.device):
    root = context.args.lamm_root
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    if head != LAMM_COMMIT or dirty:
        raise RuntimeError("Official LAMM checkout is not the pinned clean commit")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from models import LAMM
    from experiments.lamm.run_lamm_facescape import model_config

    lamm_out = context.lamm_artifact_root
    checkpoint = lamm_out / "manipulation_best.pt"
    normalisation = lamm_out / "train_only_normalisation.npz"
    region_file = lamm_out / "region_ids.pickle"
    for path in (checkpoint, normalisation, region_file):
        require_valid(path)
        require_frozen_result_artifact(
            context.results,
            path,
            manifest_relative_path=f"lamm/seed20260609/{path.name}",
        )
    if sha256_file(checkpoint) != context.freeze["lamm_manipulation_sha256"]:
        raise RuntimeError("LAMM checkpoint differs from the pre-test freeze")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    recorded_config = deepcopy(state["model_config"])
    raw = model_config(region_file, {
        index: [
            int(value) for value in context.topology.landmarks
            if int(value) in set(map(int, context.topology.region_vertices[index]))
        ]
        for index in range(len(context.topology.regions))
    }, manipulation=True)
    raw["control_vertices"] = {key: value for key, value in raw["control_vertices"].items() if value}
    config = relocate_comparable_model_config(recorded_config, raw)
    model = LAMM(deepcopy(config)).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    with np.load(normalisation) as values:
        mean = np.asarray(values["mean"], dtype=np.float32)
        std = np.asarray(values["std"], dtype=np.float32)
    return model, mean, std


def lamm_controls(model, noisy_controls: np.ndarray, landmarks: np.ndarray, std: np.ndarray, device):
    landmark_position = {int(vertex): index for index, vertex in enumerate(landmarks)}
    normalised = noisy_controls / std[landmarks][None, :, :]
    return [
        torch.as_tensor(
            normalised[:, [landmark_position[int(vertex)] for vertex in model.control_vertices[key]]],
            dtype=torch.float32,
            device=device,
        ).reshape(len(noisy_controls), -1)
        for key in model.control_region_keys
    ]


def project_certified(
    source: np.ndarray,
    ridge: np.ndarray,
    cvae: np.ndarray,
    gate_values: np.ndarray,
    input_controls: np.ndarray,
    landmarks: np.ndarray,
    faces: np.ndarray,
    projection: dict,
) -> np.ndarray:
    # This is a deployment-time constraint, so it must use the controls that
    # the method actually observes (noisy in robustness runs). The shared
    # STRICT scorer alone restores unnoised controls after inference.
    ridge_fixed = enforce_exact_controls(ridge, input_controls, landmarks).reshape(len(source), -1, 3)
    residual = gate_values[..., None] * (cvae.reshape(len(source), -1, 3) - ridge_fixed)
    residual[:, landmarks] = 0.0
    output = []
    for source_mesh, anchor, correction in zip(source, ridge_fixed, residual):
        result = project_residual_no_new_ridge_folds(
            source_mesh,
            anchor,
            correction,
            faces,
            landmarks,
            attenuation=float(projection["attenuation"]),
            smoothing_steps=int(projection["smoothing_steps"]),
            max_iterations=int(projection["max_iterations"]),
            uniform_steps=int(projection["uniform_steps"]),
        )
        if not result.certified:
            raise AssertionError("Noisy certified projection failed")
        output.append(result.delta)
    return np.asarray(output, dtype=np.float32).reshape(len(source), -1)


def aggregate_noise(output_root: Path, panel_size: int, seed: int, n_boot: int) -> None:
    paths = sorted((output_root / "noise/chunks").glob("*/*.csv"))
    rows = [row for path in paths for row in read_csv(path)]
    expected_draws = 1 + 3 * (len(NOISE_LEVELS) - 1)
    expected = len(ROBUSTNESS_METHODS) * panel_size * expected_draws
    if len(rows) != expected:
        raise RuntimeError(f"Incomplete noise rows: {len(rows)} != {expected}")
    rows.sort(key=lambda row: (
        ROBUSTNESS_METHODS.index(row["method"]), float(row["noise_mm"]),
        int(row["noise_seed"]), int(float(row["validation_pair_index"])),
    ))
    pair_path = output_root / "noise/noise_pair_metrics_all_methods.csv"
    atomic_write_csv(pair_path, rows)

    averaged_rows = []
    for method in ROBUSTNESS_METHODS:
        for level in NOISE_LEVELS:
            block = [
                row for row in rows
                if row["method"] == method and float(row["noise_mm"]) == level
            ]
            grouped: dict[tuple[str, str, int], list[dict[str, str]]] = {}
            for row in block:
                key = (
                    row["source_id"], row["target_id"],
                    int(float(row["validation_pair_index"])),
                )
                grouped.setdefault(key, []).append(row)
            expected_seeds = 1 if level == 0.0 else len(NOISE_SEEDS)
            if len(grouped) != panel_size or any(
                len(values) != expected_seeds for values in grouped.values()
            ):
                raise RuntimeError(
                    f"Incomplete pair/seed cells for {method} at {level:g} mm"
                )
            for (source_id, target_id, pair_index), values in sorted(
                grouped.items(), key=lambda item: item[0][2]
            ):
                averaged = {
                    "method": method,
                    "noise_mm": level,
                    "source_id": source_id,
                    "target_id": target_id,
                    "validation_pair_index": pair_index,
                    "n_noise_seeds_averaged": len(values),
                }
                for metric in NOISE_METRICS:
                    averaged[metric] = float(np.mean([
                        float(row[metric]) for row in values
                    ]))
                averaged_rows.append(averaged)
    averaged_path = output_root / "noise/noise_pair_metrics_seed_averaged.csv"
    atomic_write_csv(averaged_path, averaged_rows)

    summary = []
    for method in ROBUSTNESS_METHODS:
        for level in NOISE_LEVELS:
            block = [
                row for row in averaged_rows
                if row["method"] == method and float(row["noise_mm"]) == level
            ]
            record = {
                "method": method,
                "noise_mm": level,
                "n_pairs": len(block),
                "seed_aggregation": "within_pair_mean_before_summary",
            }
            for metric in NOISE_METRICS:
                values = np.asarray([float(row[metric]) for row in block])
                record[f"{metric}_mean"] = float(np.mean(values))
                record[f"{metric}_p95"] = float(np.percentile(values, 95))
                record[f"{metric}_p99"] = float(np.percentile(values, 99))
            summary.append(record)
    summary_path = output_root / "noise/noise_robustness_summary.csv"
    atomic_write_csv(summary_path, summary)

    inference_rows = []
    test_index = 0
    for method in ROBUSTNESS_METHODS:
        clean = [
            row for row in averaged_rows
            if row["method"] == method and float(row["noise_mm"]) == 0.0
        ]
        for level in NOISE_LEVELS[1:]:
            noisy = [
                row for row in averaged_rows
                if row["method"] == method and float(row["noise_mm"]) == level
            ]
            for metric in NOISE_METRICS:
                def pair_values(block):
                    return {
                        (row["source_id"], row["target_id"]): float(row[metric])
                        for row in block
                    }

                def identity_mean(pair_differences, identity_position):
                    grouped: dict[str, list[float]] = {}
                    for pair, difference in pair_differences.items():
                        grouped.setdefault(pair[identity_position], []).append(difference)
                    return {key: float(np.mean(value)) for key, value in grouped.items()}

                left, right = pair_values(clean), pair_values(noisy)
                if set(left) != set(right) or len(left) != panel_size:
                    raise RuntimeError("Clean/noisy validation-pair sets differ")
                pair_differences = {
                    pair: right[pair] - left[pair]
                    for pair in left
                }
                source_cluster = identity_mean(pair_differences, 0)
                target_cluster = identity_mean(pair_differences, 1)
                source_differences = np.asarray([
                    source_cluster[value]
                    for value in sorted(source_cluster, key=int)
                ])
                target_differences = np.asarray([
                    target_cluster[value]
                    for value in sorted(target_cluster, key=int)
                ])

                def bootstrap_ci(values, bootstrap_seed):
                    rng = np.random.default_rng(bootstrap_seed)
                    indices = rng.integers(
                        0, len(values), size=(n_boot, len(values))
                    )
                    means = np.mean(values[indices], axis=1)
                    return tuple(map(float, np.percentile(means, [2.5, 97.5])))

                source_ci = bootstrap_ci(source_differences, seed + 2 * test_index)
                target_ci = bootstrap_ci(target_differences, seed + 2 * test_index + 1)
                source_sd = float(np.std(source_differences, ddof=1))
                inference_rows.append({
                    "method": method,
                    "noise_mm": level,
                    "metric": metric,
                    "pair_mean_difference_noisy_minus_clean": float(np.mean(
                        list(pair_differences.values())
                    )),
                    "source_cluster_mean_difference": float(np.mean(source_differences)),
                    "target_cluster_mean_difference": float(np.mean(target_differences)),
                    "source_ci95_low": source_ci[0],
                    "source_ci95_high": source_ci[1],
                    "target_ci95_low": target_ci[0],
                    "target_ci95_high": target_ci[1],
                    "wilcoxon_cluster": "source_identity_primary",
                    "wilcoxon_p_two_sided": safe_wilcoxon(source_differences),
                    "effect_size_dz_source": (
                        float(np.mean(source_differences)) / source_sd
                        if source_sd > 0.0 else 0.0
                    ),
                    "source_identity_worsening_rate": float(np.mean(source_differences > 0.0)),
                    "degenerate_all_zero": bool(np.allclose(
                        source_differences, 0.0, atol=1e-15, rtol=0.0
                    )),
                })
                test_index += 1
    adjusted = holm([float(row["wilcoxon_p_two_sided"]) for row in inference_rows])
    for row, value in zip(inference_rows, adjusted):
        row["holm_family_size"] = len(inference_rows)
        row["holm_p_two_sided"] = value
        if (
            value < 0.05
            and row["source_ci95_low"] > 0.0
            and row["target_ci95_low"] > 0.0
        ):
            row["final_judgement"] = "worse_under_noise"
        elif (
            value < 0.05
            and row["source_ci95_high"] < 0.0
            and row["target_ci95_high"] < 0.0
        ):
            row["final_judgement"] = "improved_under_noise"
        else:
            row["final_judgement"] = "not_significant_or_cluster_sensitive"
    statistics_path = output_root / "noise/noise_vs_clean_paired_statistics.csv"
    atomic_write_csv(statistics_path, inference_rows)
    protocol_path = output_root / "noise/STRICT_NOISE_ROBUSTNESS_PROTOCOL_FREEZE.json"
    draw_manifest_path = output_root / "noise/SHARED_CONTROL_NOISE_DRAW_MANIFEST.json"
    require_valid(draw_manifest_path, "shared control-noise draw manifest")
    evidence = {
        "status": "COMPLETE_STRICT_POSTHOC_NOISE_ROBUSTNESS",
        "validation_pairs": panel_size,
        "levels_mm": list(NOISE_LEVELS),
        "noise_seeds": list(NOISE_SEEDS),
        "methods": list(ROBUSTNESS_METHODS),
        "metrics": list(NOISE_METRICS),
        "strict_contract": "true controls hard-fixed; free RMSE; target-relative new flip",
        "seed_aggregation": "three seeds averaged within each pair before inference",
        "primary_inference_cluster": "source identity",
        "cluster_sensitivity_gate": "target-identity bootstrap CI must agree in direction",
        "old_absolute_flip_results_reused": False,
        "holm_family_size": len(inference_rows),
        "operating_point_reselected": False,
        "protocol_freeze_sha256": sha256_file(protocol_path),
        "shared_noise_draw_manifest_sha256": sha256_file(draw_manifest_path),
        "outputs": {
            pair_path.name: sha256_file(pair_path),
            averaged_path.name: sha256_file(averaged_path),
            summary_path.name: sha256_file(summary_path),
            statistics_path.name: sha256_file(statistics_path),
        },
    }
    atomic_write_json(output_root / "noise/STRICT_NOISE_ROBUSTNESS_EVIDENCE.json", evidence)
    write_sha256_sidecar(output_root / "noise/STRICT_NOISE_ROBUSTNESS_EVIDENCE.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lamm-root", type=Path, required=True)
    parser.add_argument("--chunk-pairs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()
    if (args.seed, args.n_boot) != (20260810, 10000):
        raise ValueError(
            "This notebook freezes validation pairs, seed=20260810 and n_boot=10000; "
            "use a new output/protocol version for a different design"
        )
    context_args = argparse.Namespace(**vars(args))
    context_args.lamm_batch_size = args.batch_size
    context = Context(context_args)
    main_protocol_path = args.out / "POSTHOC_SUBUNIT_ANALYSIS_PROTOCOL_FREEZE.json"
    require_valid(main_protocol_path, "parent post-hoc protocol freeze")
    main_protocol = json.loads(main_protocol_path.read_text(encoding="utf-8"))
    if main_protocol.get("status") != "FROZEN_POSTHOC_AFTER_COMPLETED_TEST_BEFORE_REGIONAL_COMPUTATION":
        raise RuntimeError("Parent post-hoc protocol freeze has an unexpected status")
    panel = [(str(a), str(b)) for a, b in context.base["val_pairs"]]
    if len(panel) != 4830:
        raise RuntimeError(f"Expected all 4,830 frozen validation pairs, got {len(panel)}")
    canonical_indices = list(range(len(panel)))
    all_ids = set(map(str, context.base["train_ids"])) | set(map(str, context.base["val_ids"]))
    _, all_by_id = load_rows(args.repo, allowed_ids=all_ids)
    landmarks = context.topology.landmarks
    noise_draws = {}
    draw_records = []
    for level in NOISE_LEVELS:
        seeds = (NOISE_SEEDS[0],) if level == 0.0 else NOISE_SEEDS
        for noise_seed in seeds:
            draw = deterministic_noise_draw(
                level, noise_seed, len(panel), len(landmarks)
            )
            noise_draws[(level, noise_seed)] = draw
            draw_records.append({
                "noise_mm": level,
                "noise_seed": noise_seed,
                "shape": list(draw.shape),
                "dtype": str(draw.dtype),
                "array_sha256": array_sha256(draw),
            })
    draw_manifest = {
        "status": "FROZEN_SHARED_CONTROL_NOISE_DRAWS",
        "generator": "numpy.random.default_rng(seed).normal(0, sigma_mm)",
        "pair_order_sha256": sha256_json(panel),
        "coordinate_order": "pair, landmark in frozen order, xyz",
        "records": draw_records,
    }
    draw_manifest_path = args.out / "noise/SHARED_CONTROL_NOISE_DRAW_MANIFEST.json"
    if draw_manifest_path.exists():
        require_valid(draw_manifest_path)
        if json.loads(draw_manifest_path.read_text(encoding="utf-8")) != draw_manifest:
            raise RuntimeError("Existing noise-draw manifest differs")
    else:
        atomic_write_json(draw_manifest_path, draw_manifest)
        write_sha256_sidecar(draw_manifest_path)
    protocol = {
        "status": "FROZEN_STRICT_POSTHOC_NOISE_ROBUSTNESS",
        "claim_boundary": "secondary post-hoc frozen-validation robustness; not blind confirmation",
        "validation_pairs": len(panel),
        "validation_pairs_sha256": sha256_json(panel),
        "levels_mm": list(NOISE_LEVELS),
        "noise_seeds": list(NOISE_SEEDS),
        "zero_level_seed_policy": "one deterministic clean execution; nonzero levels use all three seeds",
        "sigma_rule": "isotropic independent Gaussian landmark-coordinate noise in millimetres",
        "methods": list(ROBUSTNESS_METHODS),
        "metrics": list(NOISE_METRICS),
        "shared_noise_rule": "each validation pair/level/seed uses identical noise for all four methods",
        "shared_noise_draw_manifest_sha256": sha256_file(draw_manifest_path),
        "seed_aggregation": "average three seeds within each pair before inferential analysis",
        "inference_unit": "source identity primary; target-identity bootstrap CI sensitivity gate",
        "scoring": "unnoised true controls hard-fixed; free-vertex ROI RMSE; target-relative new flip; edge-strain P95",
        "operating_point_reselected": False,
        "holm_family": "4 methods x 3 nonzero levels x 3 metrics = 36 tests",
        "parent_posthoc_protocol_sha256": sha256_file(main_protocol_path),
        "frozen_artifact_hashes": {
            "base": context.freeze["rbsr_base_sha256"],
            "gate": context.freeze["rbsr_gate_sha256"],
            "projection": context.freeze["rbsr_projection_freeze_sha256"],
            "lamm_manipulation": context.freeze["lamm_manipulation_sha256"],
        },
        "posthoc_inference_implementation_hashes": {
            relative: sha256_file(args.repo_root / relative)
            for relative in (
                "rhinoform/train.py",
                "rhinoform/train_rbsr_gate.py",
                "rhinoform/rbsr_calibration.py",
                "rhinoform/safe_fusion.py",
                "scripts/evaluation/posthoc_strict_noise_robustness.py",
            )
        },
        "zero_noise_requirement": (
            "Ridge and certified RB-SR reproduce repository-frozen validation metrics; "
            "LAMM and ARAP use their frozen implementation/configuration without reselection"
        ),
    }
    protocol_path = args.out / "noise/STRICT_NOISE_ROBUSTNESS_PROTOCOL_FREEZE.json"
    if protocol_path.exists():
        require_valid(protocol_path)
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("Existing strict noise protocol differs; use a new output version")
    else:
        atomic_write_json(protocol_path, protocol)
        write_sha256_sidecar(protocol_path)

    canonical = load_canonical(context)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Strict all-method noise robustness requires a GPU runtime")
    features, cvae_model, gate_package, gate_model, projection = prepare_neural(context, all_by_id, device)
    lamm_model, lamm_mean, lamm_std = prepare_lamm(context, device)
    vertex_features = torch.as_tensor(features, dtype=torch.float32, device=device)
    gate_center = torch.as_tensor(np.asarray(gate_package["center"]), dtype=torch.float32, device=device)
    projection_basis = None if gate_package.get("projection_basis") is None else torch.as_tensor(
        gate_package["projection_basis"], dtype=torch.float32, device=device
    )
    laplacian = uniform_laplacian(len(context.topology.vertex_region), context.topology.faces)
    classical = validate_frozen_classical_configuration(context.policy["frozen_classical_configuration"])

    for level in NOISE_LEVELS:
        seeds = (NOISE_SEEDS[0],) if level == 0.0 else NOISE_SEEDS
        for noise_seed in seeds:
            draw_root = args.out / "noise/chunks" / f"noise_mm{level:g}_seed{noise_seed}"
            for start in range(0, len(panel), args.chunk_pairs):
                stop = min(start + args.chunk_pairs, len(panel))
                stem = f"chunk_{start:05d}_{stop:05d}"
                paths = {method: draw_root / f"{stem}_{method}.csv" for method in ROBUSTNESS_METHODS}
                if all(output_valid(path, stop - start) for path in paths.values()):
                    print(f"STRICT NOISE resume c={level:g} seed={noise_seed} {stem}", flush=True)
                    continue
                pairs = panel[start:stop]
                source = np.stack([all_by_id[s]["vertices"] for s, _ in pairs]).astype(np.float32)
                target = np.stack([all_by_id[t]["vertices"] for _, t in pairs]).astype(np.float32)
                true_controls = (target - source)[:, landmarks].astype(np.float64)
                sigma = float(level)
                # The full-panel draw is materialised and hashed before any
                # inference, so chunk/resume order cannot change perturbations.
                all_noise = noise_draws[(level, noise_seed)]
                noisy_controls = true_controls + all_noise[start:stop]
                condition = noisy_condition(noisy_controls, source, context.base)
                ridge = ridge_predict(condition, context.base["ridge_cond"]).astype(np.float32)
                cvae = predict_field(cvae_model, features, condition, is_cvae=True, batch_size=args.batch_size).astype(np.float32)
                _, gate_values = predict_gate_batches(
                    gate_model, vertex_features, condition, source, ridge, cvae,
                    gate_center, float(gate_package["scale"]),
                    torch.as_tensor(landmarks, dtype=torch.long, device=device),
                    noisy_controls, projection_basis, args.batch_size,
                )
                certified = project_certified(
                    source, ridge, cvae, gate_values, noisy_controls, landmarks,
                    context.topology.faces, projection,
                )
                arap_cfg = classical["arap"]
                arap_init = solve_linear_handle_baseline(
                    noisy_controls, landmarks, len(context.topology.vertex_region), laplacian,
                    arap_cfg["handle_weight"], arap_cfg["system_ridge"],
                )
                arap_pred = arap_predict_vectorised(
                    [
                        np.asarray(all_by_id[source_id]["vertices"], dtype=np.float64)
                        for source_id, _ in pairs
                    ],
                    noisy_controls, landmarks, context.topology.faces, laplacian,
                    arap_init, arap_cfg["handle_weight"], arap_cfg["system_ridge"], arap_cfg["arap_iter"],
                )
                source_norm = torch.from_numpy((source - lamm_mean) / lamm_std).to(device)
                with torch.no_grad():
                    lamm_prediction = lamm_model((
                        source_norm,
                        lamm_controls(lamm_model, noisy_controls, landmarks, lamm_std, device),
                    ))[-1]
                lamm_abs = (lamm_prediction * torch.from_numpy(lamm_std).to(device) + torch.from_numpy(lamm_mean).to(device)).cpu().numpy()
                lamm_pred = (lamm_abs - source).reshape(len(pairs), -1)
                predictions = {
                    "ridge": ridge,
                    "certified_rbsr": certified,
                    "lamm": lamm_pred,
                    "arap": arap_pred,
                }
                for method, prediction in predictions.items():
                    metric_rows = strict_metric_rows(all_by_id, pairs, prediction)
                    if any(abs(float(row["landmark_rmse"])) > 1e-12 for row in metric_rows):
                        raise AssertionError("Final STRICT noise scorer did not hard-fix true controls")
                    if level == 0.0 and method in canonical:
                        compare_to_canonical(
                            metric_rows,
                            canonical[method],
                            canonical_indices[start:stop],
                        )
                    for offset, row in enumerate(metric_rows):
                        row.update({
                            "method": method,
                            "noise_mm": level,
                            "noise_seed": noise_seed,
                            "sigma_mm": sigma,
                            "validation_pair_index": start + offset,
                        })
                    atomic_write_csv(paths[method], metric_rows)
                print(f"STRICT NOISE persisted c={level:g} seed={noise_seed} {stop}/{len(panel)}", flush=True)
    aggregate_noise(args.out, len(panel), args.seed, args.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
