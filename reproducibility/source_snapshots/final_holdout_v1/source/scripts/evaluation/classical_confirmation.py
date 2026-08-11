"""One-shot strict classical baselines on the frozen final-rerun holdout.

The development-selected Laplacian, bi-Laplacian and ARAP configurations are
read only from the frozen confirmation policy.  No hyperparameter is selected
after test access.  Test meshes are loaded only after an exact hash-chain
receipt has been atomically persisted.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from rhinoform.baselines import arap_predict_vectorised
from rhinoform.confirmation import (
    validate_all_models_freeze,
    validate_frozen_classical_configuration,
)
from rhinoform.data import load_rows
from rhinoform.geometry import solve_linear_handle_baseline, uniform_laplacian
from rhinoform.repro import (
    atomic_write_json,
    sha256_file,
    sha256_json,
    valid_sha256_sidecar,
    validate_torch_artifact,
    write_sha256_sidecar,
)
from rhinoform.stats import write_csv
from rhinoform.strict_protocol_patch import strict_metric_rows


METHODS = ("laplacian", "bilaplacian", "arap")
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summary(rows: list[dict[str, object]]) -> dict[str, float]:
    return {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in METRICS
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-model-package", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--confirmation-policy", type=Path, required=True)
    parser.add_argument("--all-models-freeze", type=Path, required=True)
    parser.add_argument("--test-access-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk-pairs", type=int, default=160)
    args = parser.parse_args()
    if args.chunk_pairs < 1:
        raise ValueError("--chunk-pairs must be positive")

    if not validate_torch_artifact(
        args.base_model_package,
        required_keys=("args", "ridge_cond", "test_pairs", "split_manifest_sha256"),
        repair_sidecar=False,
    ):
        raise RuntimeError("Base package or SHA-256 sidecar is invalid")
    for path, label in (
        (args.split_manifest, "split manifest"),
        (args.confirmation_policy, "confirmation policy"),
        (args.all_models_freeze, "all-model freeze"),
    ):
        if not valid_sha256_sidecar(path):
            raise RuntimeError(f"{label} or SHA-256 sidecar is invalid")

    policy = json.loads(args.confirmation_policy.read_text(encoding="utf-8"))
    if policy.get("status") != "FROZEN_INTERNAL_FINAL_RERUN_HOLDOUT_POLICY":
        raise ValueError("Confirmation policy is not frozen")
    if policy.get("test_access") is not False or int(policy.get("test_access_count", -1)) != 0:
        raise ValueError("Confirmation policy does not describe a locked final-rerun holdout")
    split_hash = sha256_file(args.split_manifest)
    policy_hash = sha256_file(args.confirmation_policy)
    base_hash = sha256_file(args.base_model_package)
    if policy.get("split_manifest_sha256") != split_hash:
        raise ValueError("Classical split/confirmation policy hash mismatch")

    base = torch.load(args.base_model_package, map_location="cpu", weights_only=False)
    if base.get("split_manifest_sha256") != split_hash:
        raise ValueError("Classical split/base package hash mismatch")
    configuration = validate_frozen_classical_configuration(
        dict(policy.get("frozen_classical_configuration", {}))
    )
    configuration_hash = sha256_json(configuration)
    implementation_root = Path(__file__).resolve().parents[2]
    all_models_freeze = json.loads(args.all_models_freeze.read_text(encoding="utf-8"))
    validate_all_models_freeze(
        all_models_freeze,
        expected={
            "confirmation_policy_sha256": policy_hash,
            "split_manifest_sha256": split_hash,
            "rbsr_base_sha256": base_hash,
            "classical_configuration_sha256": configuration_hash,
        },
        expected_implementations={
            "rhinoform/baselines.py": sha256_file(implementation_root / "rhinoform/baselines.py"),
            "rhinoform/confirmation.py": sha256_file(implementation_root / "rhinoform/confirmation.py"),
            "rhinoform/geometry.py": sha256_file(implementation_root / "rhinoform/geometry.py"),
            "rhinoform/safe_fusion.py": sha256_file(implementation_root / "rhinoform/safe_fusion.py"),
            "rhinoform/strict_protocol_patch.py": sha256_file(
                implementation_root / "rhinoform/strict_protocol_patch.py"
            ),
            "scripts/evaluation/classical_confirmation.py": sha256_file(Path(__file__)),
        },
    )
    all_models_freeze_hash = sha256_file(args.all_models_freeze)

    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    test_ids = [str(value) for value in split["test_ids"]]
    test_id_set = set(test_ids)
    pairs = [(str(source), str(target)) for source, target in base["test_pairs"]]
    if len(pairs) != len(test_ids) * (len(test_ids) - 1):
        raise ValueError("Base package does not contain the complete ordered holdout pair set")
    if any(source == target or source not in test_id_set or target not in test_id_set for source, target in pairs):
        raise ValueError("Base package test pairs do not match the frozen holdout identities")
    if len(set(pairs)) != len(pairs):
        raise ValueError("Base package test pairs contain duplicates")

    receipt_chain = {
        "confirmation_policy_sha256": policy_hash,
        "split_manifest_sha256": split_hash,
        "base_model_package_sha256": base_hash,
        "all_models_freeze_sha256": all_models_freeze_hash,
        "classical_configuration_sha256": configuration_hash,
    }
    if args.test_access_receipt.is_file():
        if not valid_sha256_sidecar(args.test_access_receipt):
            raise RuntimeError("Existing classical test receipt or sidecar is invalid")
        receipt = json.loads(args.test_access_receipt.read_text(encoding="utf-8"))
        if any(receipt.get(key) != value for key, value in receipt_chain.items()):
            raise ValueError("Classical receipt belongs to a different frozen hash chain")
        if receipt.get("status") not in {"IN_PROGRESS_RESUMABLE", "COMPLETE"}:
            raise ValueError("Unrecognised classical test receipt status")
        print(f"CLASSICAL confirmation exact-hash resume: {receipt['status']}", flush=True)
    else:
        atomic_write_json(
            args.test_access_receipt,
            {
                "status": "IN_PROGRESS_RESUMABLE",
                "test_access_count": 1,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                **receipt_chain,
            },
        )
        write_sha256_sidecar(args.test_access_receipt)
        print("CLASSICAL receipt persisted before loading holdout meshes", flush=True)

    # This is the first operation that reads a holdout mesh.
    _, by_id = load_rows(args.repo, allowed_ids=test_id_set)
    template = next(iter(by_id.values()))
    landmarks = np.asarray(template["landmarks"], dtype=np.int64)
    faces = np.asarray(template["faces"], dtype=np.int64)
    n_vertices = int(np.asarray(template["vertices"]).shape[0])
    laplacian = uniform_laplacian(n_vertices, faces)

    signature = sha256_json(
        {
            "schema": "classical_final_rerun_holdout_chunks_v1",
            **receipt_chain,
            "pairs": pairs,
            "chunk_pairs": args.chunk_pairs,
        }
    )
    args.out.mkdir(parents=True, exist_ok=True)
    chunk_root = args.out / "test_chunks" / signature[:16]
    chunk_root.mkdir(parents=True, exist_ok=True)
    rows_by_method: dict[str, list[dict[str, object]]] = {method: [] for method in METHODS}

    for start in range(0, len(pairs), args.chunk_pairs):
        stop = min(start + args.chunk_pairs, len(pairs))
        stem = f"chunk_{start:05d}_{stop:05d}"
        paths = {method: chunk_root / f"{stem}_{method}.csv" for method in METHODS}
        meta_path = chunk_root / f"{stem}.json"
        resumed = False
        if valid_sha256_sidecar(meta_path) and all(valid_sha256_sidecar(path) for path in paths.values()):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            resumed = (
                meta.get("signature") == signature
                and int(meta.get("start", -1)) == start
                and int(meta.get("stop", -1)) == stop
                and all(meta.get("pair_metrics_sha256", {}).get(method) == sha256_file(path) for method, path in paths.items())
            )
        if resumed:
            print(f"CLASSICAL STRICT test resume from Drive {stop}/{len(pairs)}", flush=True)
            for method, path in paths.items():
                rows_by_method[method].extend(read_csv(path))
            continue

        print(f"CLASSICAL STRICT test live {start + 1}-{stop}/{len(pairs)}", flush=True)
        pair_chunk = pairs[start:stop]
        controls = np.stack(
            [
                (np.asarray(by_id[target]["vertices"]) - np.asarray(by_id[source]["vertices"]))[landmarks]
                for source, target in pair_chunk
            ],
            axis=0,
        )
        sources = [np.asarray(by_id[source]["vertices"], dtype=np.float64) for source, _ in pair_chunk]
        lap_cfg = configuration["laplacian"]
        bilap_cfg = configuration["bilaplacian"]
        arap_cfg = configuration["arap"]
        predictions = {
            "laplacian": solve_linear_handle_baseline(
                controls,
                landmarks,
                n_vertices,
                laplacian,
                float(lap_cfg["handle_weight"]),
                float(lap_cfg["system_ridge"]),
            ),
            "bilaplacian": solve_linear_handle_baseline(
                controls,
                landmarks,
                n_vertices,
                laplacian @ laplacian,
                float(bilap_cfg["handle_weight"]),
                float(bilap_cfg["system_ridge"]),
            ),
        }
        arap_initial = solve_linear_handle_baseline(
            controls,
            landmarks,
            n_vertices,
            laplacian,
            float(arap_cfg["handle_weight"]),
            float(arap_cfg["system_ridge"]),
        )
        predictions["arap"] = arap_predict_vectorised(
            sources,
            controls,
            landmarks,
            faces,
            laplacian,
            arap_initial,
            float(arap_cfg["handle_weight"]),
            float(arap_cfg["system_ridge"]),
            int(arap_cfg["arap_iter"]),
        )
        chunk_hashes: dict[str, str] = {}
        for method, prediction in predictions.items():
            rows = strict_metric_rows(by_id, pair_chunk, prediction)
            for offset, row in enumerate(rows):
                row["pair_index"] = float(start + offset)
            write_csv(paths[method], rows)
            rows_by_method[method].extend(rows)
            chunk_hashes[method] = sha256_file(paths[method])
        atomic_write_json(
            meta_path,
            {
                "signature": signature,
                "start": start,
                "stop": stop,
                "pair_metrics_sha256": chunk_hashes,
            },
        )
        write_sha256_sidecar(meta_path)
        print(f"CLASSICAL STRICT test persisted to Drive {stop}/{len(pairs)}", flush=True)

    reports: dict[str, object] = {}
    for method in METHODS:
        rows = rows_by_method[method]
        if len(rows) != len(pairs):
            raise ValueError(f"Incomplete classical pair metrics: {method}")
        output_path = args.out / f"identity_bootstrap_pair_metrics_{method}.csv"
        write_csv(output_path, rows)
        reports[method] = {
            "summary": summary(rows),
            "pair_metrics": output_path.name,
            "pair_metrics_sha256": sha256_file(output_path),
        }
    report = {
        "status": "COMPLETE_INTERNAL_FINAL_RERUN_HOLDOUT",
        "claim_boundary": policy.get("claim_boundary"),
        "n_pairs": len(pairs),
        "configuration": configuration,
        "configuration_sha256": configuration_hash,
        "split_manifest_sha256": split_hash,
        "confirmation_policy_sha256": policy_hash,
        "base_model_package_sha256": base_hash,
        "all_models_freeze_sha256": all_models_freeze_hash,
        "chunk_signature": signature,
        "chunk_root": str(chunk_root.relative_to(args.out)),
        "methods": reports,
    }
    report_path = args.out / "classical_confirmation_test_summary.json"
    atomic_write_json(report_path, report)
    write_sha256_sidecar(report_path)
    receipt = json.loads(args.test_access_receipt.read_text(encoding="utf-8"))
    receipt.update(
        {
            "status": "COMPLETE",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "test_access_count": 1,
            "test_summary_sha256": sha256_file(report_path),
        }
    )
    atomic_write_json(args.test_access_receipt, receipt)
    write_sha256_sidecar(args.test_access_receipt)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
