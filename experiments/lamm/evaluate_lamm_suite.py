"""Run the method-independent Rhinoform evaluation suite for a trained LAMM.

This is deliberately separate from training.  It loads the frozen best
manipulation checkpoint and applies the same test identities, pair order,
control-noise draws, strict scorer, subunit masks and dependence-ready CSV
schema used by the other Rhinoform baselines.

The control-noise protocol is fail-closed against the frozen baseline summary:
levels, seeds and the validation-derived scale must agree before any LAMM noise
result is written.  Target dense geometry is used only to construct the nine
allowed controls and by the scorer; it is never passed to LAMM inference.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.lamm.run_lamm_facescape import (
    LAMM_COMMIT,
    load_dataset,
    read_valid_csv,
    set_seed,
    sha256,
    valid_sha256_sidecar,
    write_csv,
    write_json,
    write_sha256_sidecar,
)
from rhinoform.strict_protocol_patch import signed_fold_indicator, strict_metric_rows


STRICT_METRICS = (
    "roi_rmse",
    "landmark_rmse",
    "dorsum_rmse",
    "tip_rmse",
    "edge_strain_p95",
    "normal_flip_pct",
    "abs_flip_pct",
    "missed_flip_pct",
    "target_flip_pct",
)
SUBUNITS = ("root", "dorsum", "tip", "alar_left", "alar_right")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def reusable_pair_rows(path: Path, pairs: list[tuple[str, str]]) -> list[dict[str, str]] | None:
    """Return a completed atomic pair CSV only when its order is exact."""
    rows = read_valid_csv(path)
    if rows is None or len(rows) != len(pairs):
        return None
    required = {"source_id", "target_id", *STRICT_METRICS}
    if not rows or not required.issubset(rows[0]):
        return None
    for row, expected in zip(rows, pairs):
        if (str(row["source_id"]), str(row["target_id"])) != expected:
            return None
    return rows


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    with np.load(tmp, allow_pickle=True) as probe:
        if set(probe.files) != set(arrays):
            raise ValueError(f"Incomplete NPZ payload: {tmp}")
    os.replace(tmp, path)
    write_sha256_sidecar(path)


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def aggregate(rows: list[dict]) -> dict[str, float]:
    return {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in STRICT_METRICS
    }


def frozen_noise_contract(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report.get("rows", [])
    return {
        "levels": [float(value) for value in report["levels"]],
        "noise_seeds": [int(value) for value in report["noise_seeds"]],
        "median_control_delta_from_validation": float(report["median_control_delta_from_validation"]),
        "row_levels": sorted({float(row["noise_level_c"]) for row in rows}),
        "row_seeds": sorted({int(row["noise_seed"]) for row in rows}),
        "sha256": sha256(path),
    }


def validate_noise_contract(
    contract: dict,
    levels: list[float],
    seeds: list[int],
    median_control_delta: float,
) -> None:
    if levels != contract["levels"] or levels != contract["row_levels"]:
        raise ValueError(
            f"Noise levels do not match frozen baselines: requested={levels}, frozen={contract['levels']}"
        )
    if seeds != contract["noise_seeds"] or seeds != contract["row_seeds"]:
        raise ValueError(
            f"Noise seeds do not match frozen baselines: requested={seeds}, frozen={contract['noise_seeds']}"
        )
    if not np.isclose(
        median_control_delta,
        contract["median_control_delta_from_validation"],
        rtol=0.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Validation-derived noise scale differs from the frozen baselines: "
            f"recomputed={median_control_delta}, frozen={contract['median_control_delta_from_validation']}"
        )


def load_model(lamm_root: Path, run_out: Path, device: torch.device):
    head = subprocess.check_output(
        ["git", "-C", str(lamm_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != LAMM_COMMIT:
        raise ValueError(f"LAMM checkout is {head}, expected {LAMM_COMMIT}")
    sys.path.insert(0, str(lamm_root))
    from models import LAMM

    checkpoint_path = run_out / "manipulation_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["model_config"])
    config["region_ids_file"] = str(run_out / "region_ids.pickle")
    model = LAMM(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint_path, config


def controls_and_sources(
    by_id: dict[str, dict],
    pairs: list[tuple[str, str]],
    landmarks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.stack([by_id[source_id]["vertices"] for source_id, _ in pairs]).astype(np.float32)
    target = np.stack([by_id[target_id]["vertices"] for _, target_id in pairs]).astype(np.float32)
    controls = (target - source)[:, landmarks].astype(np.float64)
    return controls, source


def controls_only(
    by_id: dict[str, dict],
    pairs: list[tuple[str, str]],
    landmarks: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            (by_id[target_id]["vertices"] - by_id[source_id]["vertices"])[landmarks]
            for source_id, target_id in pairs
        ]
    ).astype(np.float64)


def model_controls(
    noisy_controls: np.ndarray,
    source_abs: np.ndarray,
    landmarks: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    model,
    begin: int,
    end: int,
    device: torch.device,
) -> list[torch.Tensor]:
    landmark_position = {int(vertex): position for position, vertex in enumerate(landmarks)}
    output = []
    for key in model.control_region_keys:
        vertices = [int(value) for value in model.control_vertices[key]]
        positions = [landmark_position[value] for value in vertices]
        vertex_ids = np.asarray(vertices, dtype=np.int64)
        source_points = source_abs[begin:end, vertex_ids].astype(np.float32)
        target_points = (
            source_points.astype(np.float64) + noisy_controls[begin:end, positions]
        ).astype(np.float32)
        source_normalised = (source_points - mean[vertex_ids]) / std[vertex_ids]
        target_normalised = (target_points - mean[vertex_ids]) / std[vertex_ids]
        values = target_normalised - source_normalised
        output.append(torch.from_numpy(values.astype(np.float32).reshape(end - begin, -1)).to(device))
    return output


def face_regions(template: dict) -> np.ndarray:
    faces = np.asarray(template["faces"], dtype=np.int64)
    labels = np.full(len(template["vertices"]), -1, dtype=np.int64)
    for region_index, name in enumerate(SUBUNITS):
        labels[np.asarray(template["subunits"][name], dtype=np.int64)] = region_index
    output = np.full(len(faces), -1, dtype=np.int64)
    for face_index, face in enumerate(faces):
        valid = labels[face][labels[face] >= 0]
        if len(valid):
            output[face_index] = int(np.bincount(valid, minlength=len(SUBUNITS)).argmax())
    if np.any(output < 0):
        raise ValueError("Every ROI face must map to a semantic subunit")
    return output


def subunit_rows(
    by_id: dict[str, dict],
    pairs: list[tuple[str, str]],
    pred_delta: np.ndarray,
    pair_offset: int,
    mapped_faces: np.ndarray,
) -> list[dict]:
    template = next(iter(by_id.values()))
    landmarks = np.asarray(template["landmarks"], dtype=np.int64)
    faces = np.asarray(template["faces"], dtype=np.int64)
    free_vertices = {
        name: np.asarray(template["subunits"][name], dtype=np.int64)[
            ~np.isin(np.asarray(template["subunits"][name], dtype=np.int64), landmarks)
        ]
        for name in SUBUNITS
    }
    rows = []
    for local_index, (source_id, target_id) in enumerate(pairs):
        source = np.asarray(by_id[source_id]["vertices"], dtype=np.float64)
        target = np.asarray(by_id[target_id]["vertices"], dtype=np.float64)
        truth = target - source
        prediction = np.asarray(pred_delta[local_index], dtype=np.float64).reshape(-1, 3).copy()
        prediction[landmarks] = truth[landmarks]
        predicted_vertices = source + prediction
        pred_fold = signed_fold_indicator(source, predicted_vertices, faces)[0] < 0.0
        target_fold = signed_fold_indicator(source, target, faces)[0] < 0.0
        new_fold = pred_fold & ~target_fold
        row: dict[str, float | str] = {
            "pair_index": float(pair_offset + local_index),
            "source_id": source_id,
            "target_id": target_id,
            "method": "lamm",
        }
        for region_index, name in enumerate(SUBUNITS):
            vertices = free_vertices[name]
            error = prediction[vertices] - truth[vertices]
            row[f"{name}_rmse"] = float(np.sqrt(np.mean(np.sum(error * error, axis=1))))
            mask = mapped_faces == region_index
            row[f"{name}_new_flip_pct"] = float(np.mean(new_fold[mask]) * 100.0)
        rows.append(row)
    return rows


@torch.no_grad()
def predict_and_score(
    model,
    by_id: dict[str, dict],
    pairs: list[tuple[str, str]],
    source_abs: np.ndarray,
    noisy_controls: np.ndarray,
    landmarks: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
    collect_subunits: bool,
    collect_predictions: bool,
    progress_label: str,
    resume_root: Path,
    chunk_pairs: int = 320,
) -> tuple[list[dict], list[dict], np.ndarray | None]:
    model.eval()
    strict_rows: list[dict] = []
    region_rows: list[dict] = []
    predictions = (
        np.empty((len(pairs), source_abs.shape[1], 3), dtype=np.float32)
        if collect_predictions
        else None
    )
    mapped_faces = face_regions(next(iter(by_id.values()))) if collect_subunits else None
    mean_t = torch.from_numpy(mean.astype(np.float32)).to(device)
    std_t = torch.from_numpy(std.astype(np.float32)).to(device)
    chunk_pairs = max(batch_size, int(chunk_pairs))
    resume_root.mkdir(parents=True, exist_ok=True)
    for chunk_begin in range(0, len(pairs), chunk_pairs):
        chunk_end = min(chunk_begin + chunk_pairs, len(pairs))
        expected_pairs = pairs[chunk_begin:chunk_end]
        stem = f"pairs_{chunk_begin:05d}_{chunk_end:05d}"
        strict_path = resume_root / f"{stem}_strict.csv"
        region_path = resume_root / f"{stem}_subunits.csv"
        prediction_chunk_path = resume_root / f"{stem}_predictions.npz"

        chunk_strict = read_valid_csv(strict_path)
        strict_valid = chunk_strict is not None and len(chunk_strict) == len(expected_pairs)
        if strict_valid:
            strict_valid = all(
                (str(row["source_id"]), str(row["target_id"])) == expected
                for row, expected in zip(chunk_strict, expected_pairs)
            )
        chunk_regions = read_valid_csv(region_path) if collect_subunits else []
        regions_valid = (
            not collect_subunits
            or (chunk_regions is not None and len(chunk_regions) == len(expected_pairs))
        )
        if collect_subunits and regions_valid:
            regions_valid = all(
                (str(row["source_id"]), str(row["target_id"])) == expected
                for row, expected in zip(chunk_regions, expected_pairs)
            )
        chunk_predictions = None
        predictions_valid = not collect_predictions
        if collect_predictions and valid_sha256_sidecar(prediction_chunk_path):
            try:
                with np.load(prediction_chunk_path, allow_pickle=False) as package:
                    chunk_predictions = np.asarray(package["lamm_pred_delta"], dtype=np.float32)
                predictions_valid = chunk_predictions.shape == (
                    len(expected_pairs), source_abs.shape[1], 3
                )
            except (OSError, KeyError, ValueError):
                predictions_valid = False

        if strict_valid and regions_valid and predictions_valid:
            strict_rows.extend(chunk_strict)
            if collect_subunits:
                region_rows.extend(chunk_regions)
            if predictions is not None:
                predictions[chunk_begin:chunk_end] = chunk_predictions
            print(
                f"{progress_label} resume from Drive {chunk_end}/{len(pairs)}: {stem}",
                flush=True,
            )
            continue

        chunk_strict = []
        chunk_regions = []
        chunk_prediction_blocks = []
        for begin in range(chunk_begin, chunk_end, batch_size):
            end = min(begin + batch_size, chunk_end)
            source = torch.from_numpy(
                ((source_abs[begin:end] - mean) / std).astype(np.float32)
            ).to(device)
            controls = model_controls(
                noisy_controls, source_abs, landmarks, mean, std, model, begin, end, device
            )
            output = model((source, controls))[-1]
            prediction_abs = (output * std_t + mean_t).cpu().numpy()
            delta = (prediction_abs - source_abs[begin:end]).astype(np.float32)
            if predictions is not None:
                predictions[begin:end] = delta
                chunk_prediction_blocks.append(delta)
            batch_pairs = pairs[begin:end]
            batch_rows = strict_metric_rows(by_id, batch_pairs, delta)
            for local_index, row in enumerate(batch_rows):
                row["pair_index"] = float(begin + local_index)
            chunk_strict.extend(batch_rows)
            if collect_subunits:
                chunk_regions.extend(subunit_rows(by_id, batch_pairs, delta, begin, mapped_faces))
            print(
                f"{progress_label} live {end}/{len(pairs)} "
                f"({100.0 * end / len(pairs):.1f}%)",
                flush=True,
            )
        write_csv(strict_path, chunk_strict)
        if collect_subunits:
            write_csv(region_path, chunk_regions)
        if collect_predictions:
            atomic_savez(
                prediction_chunk_path,
                lamm_pred_delta=np.concatenate(chunk_prediction_blocks, axis=0),
            )
        strict_rows.extend(chunk_strict)
        region_rows.extend(chunk_regions)
        print(
            f"{progress_label} persisted to Drive {chunk_end}/{len(pairs)}: {stem}",
            flush=True,
        )
    return strict_rows, region_rows, predictions


def verify_pair_order(reference: Path, rows: list[dict]) -> None:
    frozen = read_csv(reference)
    if len(frozen) != len(rows):
        raise ValueError(f"Pair count mismatch: LAMM={len(rows)}, frozen={len(frozen)}")
    for index, (left, right) in enumerate(zip(frozen, rows)):
        if (left["source_id"], left["target_id"]) != (str(right["source_id"]), str(right["target_id"])):
            raise ValueError(f"Pair order mismatch at row {index}")



def verify_clean_reproduction(clean_reference: Path, rows: list[dict]) -> dict:
    frozen = read_csv(clean_reference)
    if len(frozen) != len(rows):
        raise ValueError(
            "Clean LAMM pair count differs from the training-run evaluation"
        )

    for index, (left, right) in enumerate(zip(frozen, rows)):
        if (
            left["source_id"],
            left["target_id"],
        ) != (
            str(right["source_id"]),
            str(right["target_id"]),
        ):
            raise ValueError(
                f"Clean LAMM pair order differs at row {index}"
            )

    # Continuous metrics remain under strict numerical tolerances.
    continuous_tolerances = {
        "roi_rmse": 1e-5,
        "landmark_rmse": 1e-6,
        "dorsum_rmse": 1e-5,
        "tip_rmse": 1e-5,
        "edge_strain_p95": 1e-4,
    }

    continuous_audit = {}
    failures = {}

    for metric, tolerance in continuous_tolerances.items():
        reference = np.asarray(
            [float(row[metric]) for row in frozen],
            dtype=np.float64,
        )
        reproduced = np.asarray(
            [float(row[metric]) for row in rows],
            dtype=np.float64,
        )
        maximum = float(
            np.max(np.abs(reference - reproduced))
        )
        continuous_audit[metric] = {
            "max_abs_difference": maximum,
            "tolerance": tolerance,
        }
        if maximum > tolerance:
            failures[metric] = continuous_audit[metric]

    reference_faces = np.asarray(
        [
            int(round(float(row["n_faces"])))
            for row in frozen
        ],
        dtype=np.int64,
    )
    reproduced_faces = np.asarray(
        [
            int(round(float(row["n_faces"])))
            for row in rows
        ],
        dtype=np.int64,
    )

    if not np.array_equal(reference_faces, reproduced_faces):
        failures["n_faces"] = "face counts differ"

    # Flip percentages are discrete. One face on this mesh is
    # 100 / 7682 = 0.013017443 percentage points, so a uniform
    # floating-point tolerance is mathematically inappropriate.
    flip_metrics = (
        "normal_flip_pct",
        "abs_flip_pct",
        "missed_flip_pct",
    )
    flip_audit = {}

    for metric in flip_metrics:
        reference_pct = np.asarray(
            [float(row[metric]) for row in frozen],
            dtype=np.float64,
        )
        reproduced_pct = np.asarray(
            [float(row[metric]) for row in rows],
            dtype=np.float64,
        )

        reference_count = np.rint(
            reference_pct * reference_faces / 100.0
        ).astype(np.int64)
        reproduced_count = np.rint(
            reproduced_pct * reproduced_faces / 100.0
        ).astype(np.int64)

        count_difference = np.abs(
            reference_count - reproduced_count
        )

        flip_audit[metric] = {
            "max_abs_percentage_point_difference": float(
                np.max(
                    np.abs(reference_pct - reproduced_pct)
                )
            ),
            "max_face_count_difference_per_pair": int(
                count_difference.max()
            ),
            "changed_pairs": int(
                np.count_nonzero(count_difference)
            ),
            "total_absolute_face_count_difference": int(
                count_difference.sum()
            ),
        }

        # Permit only one determinant-boundary face per pair.
        if count_difference.max() > 1:
            failures[metric] = flip_audit[metric]

    # Target geometry is unchanged and must reproduce exactly.
    target_reference = np.asarray(
        [
            float(row["target_flip_pct"])
            for row in frozen
        ],
        dtype=np.float64,
    )
    target_reproduced = np.asarray(
        [
            float(row["target_flip_pct"])
            for row in rows
        ],
        dtype=np.float64,
    )
    target_maximum = float(
        np.max(
            np.abs(target_reference - target_reproduced)
        )
    )

    if target_maximum != 0.0:
        failures["target_flip_pct"] = {
            "max_abs_difference": target_maximum,
            "tolerance": 0.0,
        }

    if failures:
        raise ValueError(
            "Zero-noise evaluation does not reproduce the "
            f"clean LAMM run: {failures}"
        )

    return {
        "status": "PASS",
        "rule": (
            "metric-aware; continuous tolerances plus at most "
            "one boundary face per pair"
        ),
        "continuous_audit": continuous_audit,
        "flip_audit": flip_audit,
        "target_flip_pct_max_abs_difference": target_maximum,
    }


def summarize_subunits(rows: list[dict]) -> dict:
    summary = {"method": "lamm", "n_pairs": len(rows)}
    for name in SUBUNITS:
        summary[f"{name}_rmse"] = float(np.mean([float(row[f"{name}_rmse"]) for row in rows]))
        summary[f"{name}_new_flip_pct"] = float(
            np.mean([float(row[f"{name}_new_flip_pct"]) for row in rows])
        )
    return summary


def failure_summary(rows: list[dict]) -> dict:
    output: dict[str, object] = {"n_pairs": len(rows)}
    for metric in STRICT_METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        output[metric] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(values.max()),
        }
    new_faces = np.asarray([float(row["n_new_flip_faces"]) for row in rows])
    output["pairs_with_any_new_flip"] = int(np.sum(new_faces > 0))
    output["worst_roi_pairs"] = sorted(
        (
            {
                "source_id": str(row["source_id"]),
                "target_id": str(row["target_id"]),
                "roi_rmse": float(row["roi_rmse"]),
                "new_flip_pct": float(row["normal_flip_pct"]),
            }
            for row in rows
        ),
        key=lambda row: row["roi_rmse"],
        reverse=True,
    )[:20]
    return output


@torch.no_grad()
def latency_batch_one(
    model,
    source_abs: np.ndarray,
    controls: np.ndarray,
    landmarks: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict:
    staged = min(repeats, len(source_abs))
    indices = np.arange(repeats) % staged
    source_tensor = torch.from_numpy(
        ((source_abs[:staged] - mean) / std).astype(np.float32)
    ).to(device)
    control_tensors = model_controls(
        controls, source_abs, landmarks, mean, std, model, 0, staged, device
    )

    def one(index: int):
        source = source_tensor[index:index + 1]
        control = [value[index:index + 1] for value in control_tensors]
        return model((source, control))[-1]

    print(f"LATENCY warmup start {warmup} forwards", flush=True)
    for warmup_index, index in enumerate(indices[:warmup], start=1):
        one(int(index))
        if warmup_index % 25 == 0 or warmup_index == warmup:
            print(f"LATENCY warmup live {warmup_index}/{warmup}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = []
    print(f"LATENCY measurement start {repeats} forwards", flush=True)
    for repeat_index, index in enumerate(indices, start=1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        one(int(index))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed.append((time.perf_counter() - started) * 1000.0)
        if repeat_index % 100 == 0 or repeat_index == repeats:
            print(f"LATENCY live {repeat_index}/{repeats}", flush=True)
    values = np.asarray(elapsed)
    return {
        "scope": "LAMM forward only; batch=1; inputs pre-staged on GPU; excludes normalisation, transfer, strict scoring and rendering",
        "warmup": warmup,
        "repeats": repeats,
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fyp-root", type=Path, required=True)
    parser.add_argument("--lamm-root", type=Path, required=True)
    parser.add_argument("--run-out", type=Path, required=True)
    parser.add_argument("--frozen-pair-dir", type=Path, required=True)
    parser.add_argument("--frozen-noise-summary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--levels", default="0,0.05,0.10,0.20")
    parser.add_argument("--noise-seeds", default="2026,2027,2028")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latency-warmup", type=int, default=100)
    parser.add_argument("--latency-repeats", type=int, default=1000)
    parser.add_argument("--save-clean-predictions", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("The full LAMM evaluation suite requires a Colab GPU")
    device = torch.device("cuda")
    provenance_path = args.run_out / "run_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("status") != "COMPLETE" or int(provenance["protocol"]["test_pairs"]) != 9900:
        raise ValueError("LAMM run is not a complete 9,900-pair run")
    if provenance.get("lamm_commit") != LAMM_COMMIT:
        raise ValueError("Training provenance does not use the pinned LAMM commit")

    (
        by_id,
        arrays,
        train_ids,
        val_ids,
        test_ids,
        pairs,
        _regions,
        _controls,
        protocol,
        hashes,
    ) = load_dataset(args.fyp_root)
    normalisation_path = args.run_out / "train_only_normalisation.npz"
    normalisation = np.load(normalisation_path)
    mean = np.asarray(normalisation["mean"], dtype=np.float32)
    std = np.asarray(normalisation["std"], dtype=np.float32)
    recomputed_mean = arrays["train"].mean(axis=0, dtype=np.float64).astype(np.float32)
    recomputed_std = arrays["train"].std(axis=0, dtype=np.float64).astype(np.float32) + 1e-7
    if not np.array_equal(mean, recomputed_mean) or not np.array_equal(std, recomputed_std):
        raise ValueError("Stored normalisation is not the exact train-only normalisation")

    model, checkpoint_path, config = load_model(args.lamm_root, args.run_out, device)
    checkpoint_signature = sha256(checkpoint_path)
    template = by_id[train_ids[0]]
    landmarks = np.asarray(template["landmarks"], dtype=np.int64)
    test_controls, test_sources = controls_and_sources(by_id, pairs, landmarks)
    validation_pairs = [(source, target) for source in val_ids for target in val_ids if source != target]
    validation_controls = controls_only(by_id, validation_pairs, landmarks)
    median_control_delta = float(np.median(np.linalg.norm(validation_controls, axis=2)))
    levels = parse_floats(args.levels)
    noise_seeds = parse_ints(args.noise_seeds)
    contract = frozen_noise_contract(args.frozen_noise_summary)
    validate_noise_contract(contract, levels, noise_seeds, median_control_delta)

    clean_rows: list[dict] | None = None
    clean_regions: list[dict] | None = None
    clean_predictions: np.ndarray | None = None
    subunit_pair_path = args.run_out / "lamm_subunit_pair_metrics.csv"
    prediction_path = args.run_out / "lamm_clean_predictions.npz"
    noise_summary_path = args.run_out / "lamm_noise_robustness_summary.csv"
    noise_progress_path = args.run_out / "lamm_noise_progress.json"
    noise_summary_rows = []
    noise_pair_files = []
    for level in levels:
        for noise_seed in noise_seeds:
            sigma = float(level * median_control_delta)
            pair_path = (
                args.run_out / "robustness_pair_metrics"
                / f"noise_c{level:g}_seed{noise_seed}_lamm.csv"
            )
            rows = reusable_pair_rows(pair_path, pairs)
            reused = rows is not None

            # The first zero-noise draw also owns the auxiliary clean artefacts.
            # Reuse is allowed only if all requested components survived and pass
            # their integrity/order checks; otherwise this draw is recomputed.
            if level == 0.0 and clean_rows is None and reused:
                regions = read_valid_csv(subunit_pair_path)
                regions_valid = regions is not None and len(regions) == len(pairs)
                predictions = None
                predictions_valid = not args.save_clean_predictions
                if args.save_clean_predictions and valid_sha256_sidecar(prediction_path):
                    try:
                        with np.load(prediction_path, allow_pickle=True) as package:
                            stored_pairs = [(str(a), str(b)) for a, b in package["test_pairs"].tolist()]
                            predictions = np.asarray(package["lamm_pred_delta"], dtype=np.float32)
                        predictions_valid = stored_pairs == pairs and len(predictions) == len(pairs)
                    except (OSError, KeyError, ValueError):
                        predictions_valid = False
                if regions_valid and predictions_valid:
                    clean_rows = rows
                    clean_regions = regions
                    clean_predictions = predictions
                else:
                    rows = None
                    reused = False

            if level == 0.0 and clean_rows is not None:
                rows = clean_rows
                reused = pair_path.is_file() and valid_sha256_sidecar(pair_path)
            elif rows is None:
                rng = np.random.default_rng(noise_seed)
                noisy_controls = test_controls + rng.normal(0.0, sigma, size=test_controls.shape)
                rows, regions, predictions = predict_and_score(
                    model,
                    by_id,
                    pairs,
                    test_sources,
                    noisy_controls,
                    landmarks,
                    mean,
                    std,
                    device,
                    args.batch_size,
                    collect_subunits=level == 0.0,
                    collect_predictions=bool(level == 0.0 and args.save_clean_predictions),
                    progress_label=f"NOISE level={level:g} seed={noise_seed}",
                    resume_root=(
                        args.run_out / "noise_evaluation_chunks" / checkpoint_signature[:16]
                        / f"level_{level:g}_seed_{noise_seed}"
                    ),
                )
                if level == 0.0:
                    clean_rows, clean_regions, clean_predictions = rows, regions, predictions
                    write_csv(subunit_pair_path, clean_regions)
                    write_json(args.run_out / "lamm_subunit_summary.json", summarize_subunits(clean_regions))
                    write_json(args.run_out / "lamm_failure_tail_summary.json", failure_summary(clean_rows))
                    if clean_predictions is not None:
                        atomic_savez(
                            prediction_path,
                            test_pairs=np.asarray(pairs),
                            lamm_pred_delta=clean_predictions,
                        )
                    print(
                        "CLEAN AUX persisted to Drive: subunits, failure tails"
                        + (", predictions" if clean_predictions is not None else ""),
                        flush=True,
                    )

            assert rows is not None
            if not reused:
                write_csv(pair_path, rows)
                print(
                    f"NOISE persisted to Drive level={level:g} seed={noise_seed}: {pair_path}",
                    flush=True,
                )
            else:
                print(
                    f"NOISE resume from Drive level={level:g} seed={noise_seed}: {pair_path.name}",
                    flush=True,
                )
            noise_pair_files.append({"path": str(pair_path), "sha256": sha256(pair_path)})
            noise_summary_rows.append(
                {
                    "method": "lamm",
                    "noise_level_c": level,
                    "noise_fraction": level,
                    "noise_seed": noise_seed,
                    "sigma": sigma,
                    **aggregate(rows),
                }
            )
            write_csv(noise_summary_path, noise_summary_rows)
            write_json(noise_progress_path, {
                "status": (
                    "COMPLETE"
                    if len(noise_summary_rows) == len(levels) * len(noise_seeds)
                    else "IN_PROGRESS"
                ),
                "completed_draws": len(noise_summary_rows),
                "total_draws": len(levels) * len(noise_seeds),
                "latest": {"level": level, "seed": noise_seed, "pair_file": str(pair_path)},
                "summary_csv": str(noise_summary_path),
                "summary_sha256": sha256(noise_summary_path),
            })
            print(
                f"NOISE summary {len(noise_summary_rows)}/{len(levels) * len(noise_seeds)} "
                f"level={level:g} seed={noise_seed} "
                f"roi={noise_summary_rows[-1]['roi_rmse']:.6f} "
                f"new_flip={noise_summary_rows[-1]['normal_flip_pct']:.6f} "
                f"abs_flip={noise_summary_rows[-1]['abs_flip_pct']:.6f}",
                flush=True,
            )

    assert clean_rows is not None and clean_regions is not None
    pair_reference = args.frozen_pair_dir / "identity_bootstrap_pair_metrics_ridge_sourcepca.csv"
    verify_pair_order(pair_reference, clean_rows)
    clean_reproduction = verify_clean_reproduction(
        args.run_out / "identity_bootstrap_pair_metrics_lamm.csv", clean_rows
    )
    write_csv(subunit_pair_path, clean_regions)
    subunit_summary = summarize_subunits(clean_regions)
    write_json(args.run_out / "lamm_subunit_summary.json", subunit_summary)
    write_json(args.run_out / "lamm_failure_tail_summary.json", failure_summary(clean_rows))
    write_csv(noise_summary_path, noise_summary_rows)
    if not args.save_clean_predictions:
        prediction_path = None

    latency = latency_batch_one(
        model,
        test_sources,
        test_controls,
        landmarks,
        mean,
        std,
        device,
        args.latency_warmup,
        args.latency_repeats,
    )
    write_json(args.run_out / "lamm_batch1_latency.json", latency)
    report = {
        "status": "PASS",
        "selection_boundary": "frozen validation-selected checkpoint; no test tuning",
        "protocol": {
            "train_identities": len(train_ids),
            "validation_identities": len(val_ids),
            "test_identities": len(test_ids),
            "test_pairs": len(pairs),
            "noise_levels": levels,
            "noise_seeds": noise_seeds,
            "noise_draw": "numpy.random.default_rng(seed).normal(0, level * validation_median_control_norm)",
            "validation_median_control_norm": median_control_delta,
            "strict_scoring": "nine true controls hard-fixed for scoring; free-ROI RMSE; target-relative new flip plus absolute flip transparency column",
        },
        "comparability": {
            "frozen_noise_contract": contract,
            "clean_reproduction": clean_reproduction,
            "pair_order_reference_sha256": sha256(pair_reference),
        },
        "inputs": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "normalisation_sha256": sha256(normalisation_path),
            "source_hashes": hashes,
            "lamm_commit": LAMM_COMMIT,
            "model_config": config,
        },
        "outputs": {
            "noise_pair_files": noise_pair_files,
            "noise_summary": "lamm_noise_robustness_summary.csv",
            "subunit_pair_metrics": "lamm_subunit_pair_metrics.csv",
            "subunit_summary": subunit_summary,
            "failure_tail_summary": "lamm_failure_tail_summary.json",
            "batch1_latency": latency,
            "clean_predictions": str(prediction_path) if prediction_path else None,
            "clean_predictions_sha256": sha256(prediction_path) if prediction_path else None,
        },
        "metric_warning": (
            "The frozen noise CSV's normal_flip_pct values follow the strict target-relative NEW-flip scorer. "
            "They are not absolute flips; abs_flip_pct is emitted separately here."
        ),
    }
    write_json(args.run_out / "lamm_full_evaluation_suite.json", report)
    print(json.dumps({"status": "PASS", "out": str(args.run_out)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
