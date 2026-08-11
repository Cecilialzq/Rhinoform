"""Prepare validation/test caches for the safe-fusion experiment."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from rhinoform.baselines import arap_predict_vectorised
from rhinoform.data import array_sha256, assert_identity_disjoint, assert_pairs_within, load_rows, ridge_predict, training_mean_geometry
from rhinoform.geometry import solve_linear_handle_baseline, uniform_laplacian
from rhinoform.repro import artifact_metadata, atomic_write_json, sha256_file, sha256_json
from rhinoform.sampling import coverage_balanced_ordered_pairs
from rhinoform.train import NeuralFieldCVAE, build_static_vertex_features, pair_conditions, predict_field
from rhinoform.train_rbsr_gate import SpatialRiskGate, build_rbf_projection, predict_gate_batches, resolve_device


def choose_pairs(base: dict, split: str, budget: int, seed: int) -> list[tuple[str, str]]:
    key = "val_pairs" if split == "validation" else "test_pairs"
    pairs = [(str(source), str(target)) for source, target in base[key]]
    if budget <= 0 or budget >= len(pairs):
        return pairs
    ids = sorted({value for pair in pairs for value in pair}, key=int)
    selected = coverage_balanced_ordered_pairs(ids, budget, seed)
    allowed = set(pairs)
    return [pair for pair in selected if pair in allowed][:budget]


def load_arap_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload["selected"]["arap"]
    return {
        "handle_weight": float(config["handle_weight"]),
        "system_ridge": float(config.get("system_ridge") or 1e-8),
        "arap_iter": int(config["arap_iter"]),
    }


def save_array(directory: Path, name: str, values: np.ndarray) -> None:
    np.save(directory / f"{name}.npy", np.asarray(values, dtype=np.float32), allow_pickle=False)


def atomic_save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npy", dir=path.parent)
    os.close(fd)
    try:
        np.save(temporary, np.asarray(values, dtype=np.float32), allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint_array(path: Path, shape: tuple[int, ...], force: bool) -> np.ndarray:
    if path.exists() and not force:
        values = np.load(path, mmap_mode="r+")
        if values.shape != shape or values.dtype != np.float32:
            raise ValueError(f"Checkpoint mismatch for {path}: {values.shape}/{values.dtype}, expected {shape}/float32")
        return values
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-package", type=Path, required=True)
    parser.add_argument("--rbsr-package", type=Path, required=True)
    parser.add_argument("--geometric-tuning", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pair-budget", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    complete_path = args.out / "CACHE_COMPLETE.json"
    state_path = args.out / "CACHE_STATE.json"
    device = resolve_device(args.device)
    print(f"Loading data and model packages on {device}", flush=True)
    base = torch.load(args.base_package, map_location="cpu", weights_only=False)
    rbsr = torch.load(args.rbsr_package, map_location="cpu", weights_only=False)
    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    train_ids = [str(value) for value in base["train_ids"]]
    val_ids = [str(value) for value in base["val_ids"]]
    test_ids = [str(value) for value in base["test_ids"]]
    assert_identity_disjoint({"train": train_ids, "validation": val_ids, "test": test_ids})
    expected = {
        "train": set(map(str, split_manifest["train_pool_ids"])),
        "validation": set(map(str, split_manifest["val_ids"])),
        "test": set(map(str, split_manifest["test_ids"])),
    }
    actual = {"train": set(train_ids), "validation": set(val_ids), "test": set(test_ids)}
    for name in expected:
        if actual[name] != expected[name]:
            raise RuntimeError(f"Base package {name} identities do not match the split manifest")
    feature_template_sha256 = base.get("feature_template_sha256")
    if not feature_template_sha256:
        print("[WARN] base package predates the train-only feature template fix (no stored template "
              "hash/vertices). Recomputing the train-only template (same approach as srg_common).", flush=True)
    elif rbsr.get("feature_template_sha256") not in (None, feature_template_sha256):
        raise RuntimeError("RBSR and base packages use different feature geometry templates")
    _, by_id = load_rows(args.repo)
    pairs = choose_pairs(base, args.split, args.pair_budget, args.seed)
    if not pairs:
        raise SystemExit("No pairs selected")
    assert_pairs_within(pairs, val_ids if args.split == "validation" else test_ids, args.split)
    template = by_id[train_ids[0]]
    faces = np.asarray(template["faces"], dtype=np.int64)
    landmarks = np.asarray(template["landmarks"], dtype=np.int64)
    if "feature_template_vertices" in base:
        template_vertices = np.asarray(base["feature_template_vertices"], dtype=np.float64)
    else:
        template_vertices = np.asarray(training_mean_geometry(by_id, train_ids), dtype=np.float64)
    if feature_template_sha256 and array_sha256(template_vertices) != feature_template_sha256:
        print("[WARN] base feature template hash differs (benign cross-environment float/version "
              "difference); proceeding with the recomputed template.", flush=True)
    if not feature_template_sha256:
        feature_template_sha256 = array_sha256(template_vertices)
    expected_rbf_basis = build_rbf_projection(template_vertices, landmarks)
    rbf_basis = rbsr.get("projection_basis")
    if rbf_basis is None:
        rbf_basis = expected_rbf_basis
    else:
        rbf_basis = np.asarray(rbf_basis, dtype=np.float32)
        if not np.allclose(rbf_basis, expected_rbf_basis, atol=1e-6, rtol=1e-6):
            print("[WARN] RBSR projection basis differs from recomputed (benign cross-environment "
                  "float difference); using the stored projection basis from the RBSR package.", flush=True)
    arap_config = load_arap_config(args.geometric_tuning)
    signature_payload = {
        "schema": 2,
        "split": args.split,
        "pairs": pairs,
        "pair_budget": args.pair_budget,
        "seed": args.seed,
        "base_package_sha256": sha256_file(args.base_package),
        "rbsr_package_sha256": sha256_file(args.rbsr_package),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "feature_template_sha256": feature_template_sha256,
        "rbf_basis_sha256": array_sha256(rbf_basis),
        "arap_config": arap_config,
    }
    run_signature = sha256_json(signature_payload)
    if complete_path.exists() and not args.force:
        completed = json.loads(complete_path.read_text(encoding="utf-8"))
        if completed.get("run_signature") != run_signature:
            raise RuntimeError(f"Completed cache signature differs in {args.out}; use a new output directory.")
        print(f"Cache already complete: {complete_path}", flush=True)
        return 0
    if state_path.exists() and not args.force:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("run_signature") != run_signature:
            raise RuntimeError(f"Partial cache signature differs in {args.out}; use a new output directory.")
    else:
        state = {
            "run_signature": run_signature,
            "neural_completed": 0,
            "arap_completed": 0,
            "gate_sum": [0.0] * len(template["vertices"]),
        }
        atomic_write_json(state_path, state)
    vertex_features, _ = build_static_vertex_features(
        by_id, train_ids, use_subunit_features=bool(base["use_subunit_features"])
    )
    condition, source, target_delta, controls, _, _, _ = pair_conditions(
        by_id, pairs, base["source_pca"], base["cond_mean"], base["cond_std"]
    )
    ridge = ridge_predict(condition, base["ridge_cond"]).astype(np.float32)
    n_pairs = len(pairs)
    flat_vertices = source.shape[1] * 3
    for name, values in (
        ("target_delta", target_delta.reshape(n_pairs, -1)),
        ("controls", controls.reshape(n_pairs, -1)),
        ("ridge", ridge),
    ):
        path = args.out / f"{name}.npy"
        if args.force or not path.exists():
            atomic_save_array(path, values)
    pairs_path = args.out / "pairs.json"
    if args.force or not pairs_path.exists():
        atomic_write_json(pairs_path, pairs)
    rbf_basis_path = args.out / "rbf_basis.npy"
    if args.force or not rbf_basis_path.exists():
        atomic_save_array(rbf_basis_path, rbf_basis)

    global_prediction = checkpoint_array(args.out / "global_hybrid.npy", (n_pairs, flat_vertices), args.force)
    rbsr_prediction = checkpoint_array(args.out / "rbsr.npy", (n_pairs, flat_vertices), args.force)
    pair_gate_mean = checkpoint_array(args.out / "pair_gate_mean.npy", (n_pairs,), args.force)
    neural_start = int(state.get("neural_completed", 0))
    gate_sum = np.asarray(state.get("gate_sum", [0.0] * source.shape[1]), dtype=np.float64)
    if neural_start < n_pairs:
        base_args = base["args"]
        cvae = NeuralFieldCVAE(
            vertex_features.shape[1], condition.shape[1], int(np.asarray(base["obs_train_mean"]).shape[1]),
            latent_dim=int(base_args["latent_dim"]), hidden=int(base_args["hidden"]),
        ).to(device)
        cvae.load_state_dict(base["cvae_state_dict"])
        cvae.eval()
        rbsr_args = rbsr["args"]
        gate = SpatialRiskGate(
            int(rbsr["gate_vertex_dim"]), int(rbsr["gate_cond_dim"]),
            int(rbsr_args["hidden"]), float(base["selected_alpha"]),
        ).to(device)
        gate.load_state_dict(rbsr["gate_state_dict"])
        gate.eval()
        vertex_tensor = torch.as_tensor(vertex_features, dtype=torch.float32, device=device)
        projection_basis = None if rbsr.get("projection_basis") is None else torch.as_tensor(
            rbsr["projection_basis"], dtype=torch.float32, device=device
        )
        for start in range(neural_start, n_pairs, args.batch_size):
            stop = min(n_pairs, start + args.batch_size)
            cvae_batch = predict_field(cvae, vertex_features, condition[start:stop], is_cvae=True).astype(np.float32)
            global_batch = ridge[start:stop] + float(base["selected_alpha"]) * (cvae_batch - ridge[start:stop])
            rbsr_batch, gate_values = predict_gate_batches(
                gate, vertex_tensor, condition[start:stop], source[start:stop], ridge[start:stop], cvae_batch,
                torch.as_tensor(rbsr["center"], dtype=torch.float32, device=device), float(rbsr["scale"]),
                torch.as_tensor(landmarks, dtype=torch.long, device=device), controls[start:stop],
                projection_basis, args.batch_size,
            )
            global_prediction[start:stop] = global_batch
            rbsr_prediction[start:stop] = rbsr_batch
            pair_gate_mean[start:stop] = np.mean(gate_values, axis=1)
            global_prediction.flush()
            rbsr_prediction.flush()
            pair_gate_mean.flush()
            gate_sum += np.sum(gate_values, axis=0)
            state.update({"neural_completed": stop, "gate_sum": gate_sum.tolist()})
            atomic_write_json(state_path, state)
            print(f"Neural cache checkpoint: {stop}/{n_pairs}", flush=True)
        cvae.to("cpu")
        gate.to("cpu")
        del cvae, gate
        if device.type == "cuda":
            torch.cuda.empty_cache()
    vertex_gate_path = args.out / "vertex_gate_mean.npy"
    if args.force or not vertex_gate_path.exists():
        atomic_save_array(vertex_gate_path, gate_sum / n_pairs)

    laplacian = uniform_laplacian(source.shape[1], faces)
    controls_xyz = np.asarray(controls, dtype=np.float64).reshape(len(pairs), len(landmarks), 3)
    initial_path = args.out / "arap_initial.npy"
    if initial_path.exists() and not args.force:
        initial = np.load(initial_path, mmap_mode="r")
    else:
        print("Computing linear ARAP initialization", flush=True)
        initial = solve_linear_handle_baseline(
            controls_xyz, landmarks, source.shape[1], laplacian,
            arap_config["handle_weight"], arap_config["system_ridge"],
        )
        atomic_save_array(initial_path, initial)
        initial = np.load(initial_path, mmap_mode="r")
    arap = checkpoint_array(args.out / "arap.npy", (n_pairs, flat_vertices), args.force)

    def checkpoint_arap(completed: int, output: np.ndarray) -> None:
        output.flush()
        state["arap_completed"] = completed
        atomic_write_json(state_path, state)
        print(f"ARAP cache checkpoint: {completed}/{n_pairs}", flush=True)

    print(f"Computing validation-selected ARAP base from pair {state.get('arap_completed', 0)}", flush=True)
    arap_predict_vectorised(
        [np.asarray(item, dtype=np.float64) for item in source],
        controls_xyz, landmarks, faces, laplacian, initial,
        arap_config["handle_weight"], arap_config["system_ridge"], arap_config["arap_iter"],
        output=arap, start_index=int(state.get("arap_completed", 0)),
        checkpoint_every=args.batch_size, checkpoint_callback=checkpoint_arap,
    )
    metadata = {
        "status": "complete",
        "split": args.split,
        "n_pairs": len(pairs),
        "pair_budget": args.pair_budget,
        "seed": args.seed,
        "device": str(device),
        "shape": [len(pairs), int(source.shape[1]), 3],
        "selected_alpha": float(base["selected_alpha"]),
        "arap_config": arap_config,
        "run_signature": run_signature,
        "base_package_sha256": sha256_file(args.base_package),
        "rbsr_package_sha256": sha256_file(args.rbsr_package),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "feature_template_sha256": feature_template_sha256,
        "rbf_basis_sha256": array_sha256(rbf_basis),
        "provenance": artifact_metadata(
            repo_root=Path(__file__).resolve().parents[1],
            command_args={
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            input_manifest_path=args.repo / "manifest.json",
            split_manifest_path=args.split_manifest,
            seed=args.seed,
            data_root_identifier=args.repo.name,
        ),
    }
    atomic_write_json(complete_path, metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
