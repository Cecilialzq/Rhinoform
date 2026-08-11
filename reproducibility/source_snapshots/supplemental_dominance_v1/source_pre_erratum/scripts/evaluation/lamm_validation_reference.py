"""Evaluate the already-frozen LAMM checkpoint on the matched validation split.

This command never trains LAMM and never loads test identities.  It exists so
RB-SR operating points can be selected against a genuinely matched validation
reference rather than against already-observed test numbers.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

# Allow direct execution from any working directory.  Colab still installs the
# package, but this removes an avoidable dependency on caller-supplied
# PYTHONPATH and mirrors `python -m` import semantics.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from experiments.lamm.run_lamm_facescape import (
    LAMM_COMMIT,
    delta_controls,
    load_dataset,
    model_config,
)
from rhinoform.repro import (
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
    sha256_json,
    valid_sha256_sidecar,
    validate_torch_artifact,
    write_sha256_sidecar,
)
from rhinoform.strict_protocol_patch import strict_metric_rows


METRICS = (
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

LAMM_CONSTRUCTOR_DERIVED_DEFAULTS: dict[str, object] = {
    "scale_dim_token": 0.5,
    "depth": None,
    "Dinput": 3,
    "Dlms": 8,
}


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def comparable_model_config(recorded: dict[str, object], live: dict[str, object]) -> bool:
    """Canonicalise official constructor defaults and ignore only mount paths.

    The pinned official LAMM constructor mutates its input configuration by
    materialising these four derived defaults.  Training therefore recorded a
    post-constructor dictionary, while validation reconstructs the equivalent
    pre-constructor dictionary.  Existing, non-default recorded values are not
    overwritten and will still fail the exact structural comparison.
    """
    def canonical(config: dict[str, object]) -> dict[str, object]:
        result = dict(config)
        for key, value in LAMM_CONSTRUCTOR_DERIVED_DEFAULTS.items():
            result.setdefault(key, value)
        result.pop("region_ids_file", None)
        return result

    return canonical(recorded) == canonical(live)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fyp-root", type=Path, required=True)
    parser.add_argument("--lamm-root", type=Path, required=True)
    parser.add_argument("--lamm-out", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-pairs", type=int, default=320)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1 or args.chunk_pairs < 1:
        raise ValueError("batch and chunk sizes must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    git_head = subprocess.check_output(
        ["git", "-C", str(args.lamm_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if git_head != LAMM_COMMIT:
        raise ValueError(f"LAMM checkout is {git_head}, expected {LAMM_COMMIT}")
    checkpoint = args.lamm_out / "manipulation_best.pt"
    normalisation = args.lamm_out / "train_only_normalisation.npz"
    region_file = args.lamm_out / "region_ids.pickle"
    for path in (args.split_manifest, checkpoint, normalisation, region_file):
        if not valid_sha256_sidecar(path):
            raise RuntimeError(f"Frozen artifact or sidecar is invalid: {path}")
    if not validate_torch_artifact(
        checkpoint,
        required_keys=("model", "epoch", "model_config", "resume_signature"),
        repair_sidecar=False,
    ):
        raise RuntimeError("Frozen LAMM manipulation checkpoint is invalid")

    (
        by_id,
        _arrays,
        _train_ids,
        val_ids,
        _test_ids,
        _test_pairs,
        _regions,
        controls,
        _protocol,
        source_hashes,
    ) = load_dataset(args.fyp_root, split_manifest=args.split_manifest, load_test=False)
    pairs = [(source, target) for source in val_ids for target in val_ids if source != target]
    if len(pairs) != 4830:
        raise AssertionError(f"Expected 4,830 validation pairs, got {len(pairs)}")

    sys.path.insert(0, str(args.lamm_root))
    from models import LAMM

    config = model_config(region_file, controls, manipulation=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not comparable_model_config(dict(state["model_config"]), config):
        raise ValueError("Frozen LAMM checkpoint/model configuration mismatch")
    model = LAMM(config).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    del state
    with np.load(normalisation, allow_pickle=False) as saved:
        mean = np.asarray(saved["mean"], dtype=np.float32)
        std = np.asarray(saved["std"], dtype=np.float32)
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)

    signature = sha256_json(
        {
            "schema": "frozen_lamm_validation_reference_v1",
            "checkpoint_sha256": sha256_file(checkpoint),
            "normalisation_sha256": sha256_file(normalisation),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "strict_scorer_sha256": sha256_file(
                Path(__file__).resolve().parents[2] / "rhinoform/strict_protocol_patch.py"
            ),
            "pairs": pairs,
        }
    )
    args.out.mkdir(parents=True, exist_ok=True)
    chunk_root = args.out / "validation_chunks" / signature[:16]
    chunk_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    with torch.no_grad():
        for chunk_start in range(0, len(pairs), args.chunk_pairs):
            chunk_stop = min(chunk_start + args.chunk_pairs, len(pairs))
            chunk_path = chunk_root / f"pairs_{chunk_start:05d}_{chunk_stop:05d}.csv"
            chunk_rows = read_csv(chunk_path) if valid_sha256_sidecar(chunk_path) else []
            if len(chunk_rows) == chunk_stop - chunk_start:
                expected = pairs[chunk_start:chunk_stop]
                if [
                    (str(row["source_id"]), str(row["target_id"])) for row in chunk_rows
                ] == expected:
                    rows.extend(chunk_rows)
                    print(f"LAMM VALIDATION resume {chunk_stop}/{len(pairs)}", flush=True)
                    continue
            chunk_rows = []
            for start in range(chunk_start, chunk_stop, args.batch_size):
                stop = min(start + args.batch_size, chunk_stop)
                batch_pairs = pairs[start:stop]
                source_abs = np.stack([by_id[source]["vertices"] for source, _ in batch_pairs]).astype(np.float32)
                target_abs = np.stack([by_id[target]["vertices"] for _, target in batch_pairs]).astype(np.float32)
                source = torch.from_numpy((source_abs - mean) / std).to(device)
                target = torch.from_numpy((target_abs - mean) / std).to(device)
                prediction = model((source, delta_controls(source, target, model)))[-1]
                pred_abs = (prediction * std_t + mean_t).cpu().numpy()
                pred_delta = (pred_abs - source_abs).reshape(len(batch_pairs), -1)
                batch_rows = strict_metric_rows(by_id, batch_pairs, pred_delta)
                for local_index, row in enumerate(batch_rows):
                    row["pair_index"] = float(start + local_index)
                chunk_rows.extend(batch_rows)
                print(f"LAMM VALIDATION live {stop}/{len(pairs)}", flush=True)
            atomic_write_csv(chunk_path, chunk_rows)
            rows.extend(chunk_rows)
            print(f"LAMM VALIDATION persisted {chunk_stop}/{len(pairs)}", flush=True)

    pair_path = args.out / "pair_metrics_lamm_validation.csv"
    atomic_write_csv(pair_path, rows)
    report = {
        "status": "FROZEN_LAMM_VALIDATION_REFERENCE_COMPLETE",
        "method": "lamm_cvpr2024_five_semantic_patches",
        "split": "validation",
        "test_access": False,
        "n_pairs": len(rows),
        "means": {
            metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in METRICS
        },
        "pair_metrics": pair_path.name,
        "pair_metrics_sha256": sha256_file(pair_path),
        "checkpoint_sha256": sha256_file(checkpoint),
        "normalisation_sha256": sha256_file(normalisation),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "source_hashes": source_hashes,
        "evaluation_signature": signature,
    }
    report_path = args.out / "lamm_validation_summary.json"
    atomic_write_json(report_path, report)
    write_sha256_sidecar(report_path)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
