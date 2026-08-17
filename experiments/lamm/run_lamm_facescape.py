"""Train and evaluate a real LAMM baseline on the frozen Rhinoform protocol.

The script imports the official CVPR 2024 LAMM model from a separately cloned,
pinned repository. It implements only the FaceScape/ROI adapter, deterministic
two-stage training orchestration, strict evaluation and provenance capture.

No test identity is used for normalisation, training, checkpoint selection or
hyper-parameter selection. At inference the target contributes only the nine
control displacements; dense target vertices are used only by the scorer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from rhinoform.confirmation import validate_all_models_freeze


LAMM_COMMIT = "87354c05dec341c6d8dd319665dd52553fb03084"
PATCH_ORDER = ("root", "dorsum", "tip", "alar_left", "alar_right")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(array.dtype).encode("utf-8"))
    h.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    h.update(memoryview(array).cast("B"))
    return h.hexdigest()


def lamm_resume_signature(
    *,
    stage: str,
    model_config: dict,
    train_array_sha256: str,
    validation_array_sha256: str,
    seed: int,
    epochs: int,
    batch_size: int,
    eval_every: int,
    implementation_sha256: str,
    official_commit: str,
    extra: dict | None = None,
) -> str:
    """Bind a resumable checkpoint to the complete training contract."""
    return json_sha256({
        "schema": "lamm_resume_contract_v1",
        "stage": stage,
        "model_config": model_config,
        "train_array_sha256": train_array_sha256,
        "validation_array_sha256": validation_array_sha256,
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "eval_every": int(eval_every),
        "implementation_sha256": implementation_sha256,
        "official_commit": official_commit,
        "extra": extra or {},
    })


def free_pair_vector_rmse(error, free_mask):
    """Per-pair strict vector RMSE over the free ROI vertices."""
    if isinstance(error, torch.Tensor):
        mask = torch.as_tensor(free_mask, dtype=torch.bool, device=error.device)
        squared_norm = torch.sum(error[:, mask].square(), dim=-1)
        return torch.sqrt(torch.mean(squared_norm, dim=1))
    array = np.asarray(error)
    mask = np.asarray(free_mask, dtype=bool)
    squared_norm = np.sum(array[:, mask] ** 2, axis=-1)
    return np.sqrt(np.mean(squared_norm, axis=1))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_sha256_sidecar(path: Path) -> None:
    write_json(path.with_suffix(path.suffix + ".sha256.json"), {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    })


def valid_sha256_sidecar(path: Path) -> bool:
    sidecar = path.with_suffix(path.suffix + ".sha256.json")
    if not path.is_file() or not sidecar.is_file():
        return False
    try:
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        return (
            int(record["bytes"]) == path.stat().st_size
            and str(record["sha256"]) == sha256(path)
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def atomic_torch_save(path: Path, payload: dict) -> None:
    """Write a checkpoint without ever replacing a valid file by a partial one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    # Fail before replacement if the serialised checkpoint cannot be read back.
    probe = torch.load(tmp, map_location="cpu", weights_only=False)
    if "model" not in probe or "epoch" not in probe:
        raise ValueError(f"Incomplete checkpoint payload: {tmp}")
    del probe
    os.replace(tmp, path)
    write_sha256_sidecar(path)


def atomic_savez_compressed(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        with np.load(tmp, allow_pickle=False) as probe:
            if set(probe.files) != set(arrays):
                raise ValueError(f"Incomplete NumPy artefact: {tmp}")
        os.replace(tmp, path)
        write_sha256_sidecar(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("wb") as handle:
            pickle.dump(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        with tmp.open("rb") as handle:
            pickle.load(handle)
        os.replace(tmp, path)
        write_sha256_sidecar(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass(frozen=True)
class Protocol:
    seed: int
    train_identities: int
    validation_identities: int
    test_identities: int
    test_pairs: int
    roi_vertices: int
    free_vertices: int
    landmarks: int
    patches: int
    lamm_commit: str = LAMM_COMMIT


def numeric_id(value: str) -> tuple[int, str]:
    return (int(value), value) if value.isdigit() else (10**9, value)


def load_dataset(
    fyp_root: Path,
    *,
    data_root: Path | None = None,
    split_manifest: Path | None = None,
    load_test: bool = True,
):
    data_root = data_root or (fyp_root / "data")
    manifest_path = data_root / "manifest.json"
    split_path = split_manifest or (fyp_root / "splits/facescape_847/split_manifest.json")
    roi_path = fyp_root / "roi/vertices.json"
    subunit_path = fyp_root / "roi/subunits.json"
    for path in (manifest_path, split_path, roi_path, subunit_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    roi = json.loads(roi_path.read_text(encoding="utf-8"))
    subunits = json.loads(subunit_path.read_text(encoding="utf-8"))

    train_ids = [str(x) for x in split["train_pool_ids"]]
    val_ids = [str(x) for x in split["val_ids"]]
    test_ids = [str(x) for x in split["test_ids"]]
    if len(train_ids) < 1 or (len(val_ids), len(test_ids)) != (70, 100):
        raise ValueError("Expected a non-empty train pool plus frozen 70/100 validation/test sizes")
    groups = [set(train_ids), set(val_ids), set(test_ids)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Identity overlap in split manifest")

    rows = {str(row["subject_id"]): row for row in manifest["rows"]}
    expected = set().union(*groups)
    if not expected.issubset(rows):
        raise ValueError(f"Split identities are not a subset of the data manifest: {len(rows)} vs {len(expected)}")

    ids_to_load = set(train_ids) | set(val_ids) | (set(test_ids) if load_test else set())

    by_id: dict[str, dict] = {}
    for index, subject_id in enumerate(sorted(ids_to_load, key=numeric_id), 1):
        row = rows[subject_id]
        path = Path(row["npz_path"])
        if not path.is_absolute():
            path = data_root / path
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing licensed mesh cache {path}. Stage FYP final/data/meshes in Drive."
            )
        if row.get("sha256") and sha256(path) != row["sha256"]:
            raise ValueError(f"Mesh hash mismatch for identity {subject_id}")
        item = np.load(path, allow_pickle=False)
        by_id[subject_id] = {
            "vertices": np.asarray(item["vertices"], dtype=np.float32),
            "faces": np.asarray(item["faces"], dtype=np.int64),
            "landmarks": np.asarray(item["local_landmarks"], dtype=np.int64),
            "subunits": {
                name: np.asarray(item[f"subunit_{name}"], dtype=np.int64)
                for name in ("root", "dorsum", "tip", "alar_left", "alar_right")
            },
        }
        if index % 100 == 0:
            print(f"loaded {index}/{len(ids_to_load)} identities", flush=True)

    template = by_id[train_ids[0]]
    n_vertices = int(template["vertices"].shape[0])
    landmarks = template["landmarks"]
    if n_vertices != 3934 or len(landmarks) != 9:
        raise ValueError(f"Unexpected ROI/control shape: {n_vertices}, {len(landmarks)}")
    roi_indices = [int(x) for x in roi["roi_indices"]]
    if roi.get("roi_vertex_count") != n_vertices or len(roi_indices) != n_vertices:
        raise ValueError("ROI JSON does not declare exactly 3,934 vertices")
    if len(set(roi_indices)) != n_vertices:
        raise ValueError("ROI JSON contains duplicate global vertex indices")
    global_landmarks = [int(roi["landmarks_27_35"][str(i)]) for i in range(27, 36)]
    global_to_local = {global_index: local_index for local_index, global_index in enumerate(roi_indices)}
    expected_local_landmarks = np.asarray(
        [global_to_local[index] for index in global_landmarks], dtype=np.int64)
    if not np.array_equal(landmarks, expected_local_landmarks):
        raise ValueError("Mesh-cache landmark order does not match ROI landmarks 27--35")
    if subunits.get("landmarks_27_35") != roi.get("landmarks_27_35"):
        raise ValueError("ROI and subunit landmark definitions disagree")
    for subject_id, item in by_id.items():
        if item["vertices"].shape != (n_vertices, 3):
            raise ValueError(f"Shape mismatch for identity {subject_id}")
        if not np.array_equal(item["faces"], template["faces"]):
            raise ValueError(f"Topology mismatch for identity {subject_id}")
        if not np.array_equal(item["landmarks"], landmarks):
            raise ValueError(f"Landmark mismatch for identity {subject_id}")

    region_ids = [np.asarray(subunits["groups"][name]["local_roi_indices"], dtype=np.int64)
                  for name in PATCH_ORDER]
    concatenated = np.concatenate(region_ids)
    if len(concatenated) != n_vertices or not np.array_equal(np.sort(concatenated), np.arange(n_vertices)):
        raise ValueError("The five semantic subunits must partition all ROI vertices exactly")
    control_vertices: dict[int, list[int]] = {}
    for patch_index, region in enumerate(region_ids):
        region_set = set(region.tolist())
        controls = [int(x) for x in landmarks if int(x) in region_set]
        if controls:
            control_vertices[patch_index] = controls
    flat_controls = [x for values in control_vertices.values() for x in values]
    if sorted(flat_controls) != sorted(map(int, landmarks)):
        raise ValueError("Each landmark must occur in exactly one semantic patch")

    arrays = {
        "train": np.stack([by_id[x]["vertices"] for x in train_ids]),
        "validation": np.stack([by_id[x]["vertices"] for x in val_ids]),
    }
    if load_test:
        arrays["test"] = np.stack([by_id[x]["vertices"] for x in test_ids])
    test_pairs = [(source, target) for source in test_ids for target in test_ids if source != target]
    if len(test_pairs) != 9900:
        raise AssertionError(len(test_pairs))
    protocol = Protocol(
        seed=0,
        train_identities=len(train_ids),
        validation_identities=len(val_ids),
        test_identities=len(test_ids),
        test_pairs=len(test_pairs),
        roi_vertices=n_vertices,
        free_vertices=n_vertices - len(landmarks),
        landmarks=len(landmarks),
        patches=len(region_ids),
    )
    source_hashes = {
        "manifest": sha256(manifest_path),
        "split": sha256(split_path),
        "roi": sha256(roi_path),
        "subunits": sha256(subunit_path),
    }
    return by_id, arrays, train_ids, val_ids, test_ids, test_pairs, region_ids, control_vertices, protocol, source_hashes


def make_loader(array: np.ndarray, batch_size: int, seed: int, shuffle: bool, drop_last: bool = False):
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(array.astype(np.float32, copy=False)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      num_workers=0, generator=generator)


def model_config(region_file: Path, control_vertices: dict[int, list[int]], manipulation: bool) -> dict:
    return {
        "backbone": "transformer",
        "region_ids_file": str(region_file),
        "Npatches": 5,
        "dim": 512,
        "bottleneck_dim": 256,
        "encoder_depth": 5,
        "decoder_depth": 3,
        "heads": 8,
        "scale_dim": 4,
        "dropout": 0.0,
        "manipulation": manipulation,
        "control_vertices": control_vertices,
    }


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    best: float,
    config: dict,
    resume_signature: str,
) -> None:
    atomic_torch_save(path, {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_validation": best,
        "model_config": config,
        "resume_signature": resume_signature,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    })


def load_checkpoint(path: Path, model, optimizer=None, *, expected_resume_signature: str) -> tuple[int, float]:
    if not valid_sha256_sidecar(path):
        if path.is_file():
            # Validate the checkpoint itself before creating/mending its sidecar.
            # This also covers an interruption after os.replace() but before the
            # corresponding sidecar was replaced.
            probe = torch.load(path, map_location="cpu", weights_only=False)
            if "model" not in probe or "epoch" not in probe or "resume_signature" not in probe:
                raise ValueError(f"Checkpoint is incomplete: {path}")
            del probe
            write_sha256_sidecar(path)
            print(f"validated checkpoint and repaired SHA sidecar: {path.name}", flush=True)
        else:
            raise ValueError(f"Checkpoint does not exist: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("resume_signature") != expected_resume_signature:
        raise ValueError(f"LAMM checkpoint training contract mismatch: {path}")
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
        rng = state.get("rng_state")
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch_cpu"])
            if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
    return int(state.get("epoch", 0)), float(state.get("best_validation", np.inf))


def cosine_learning_rate(step: int, total_steps: int, warmup_steps: int,
                         base_lr: float, min_lr: float, start_lr: float) -> float:
    """Step-wise warm-up plus cosine decay, matching official LAMM settings."""
    if warmup_steps > 0 and step < warmup_steps:
        return start_lr + (base_lr - start_lr) * step / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_learning_rate(optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def manipulation_alpha_max(epoch: int) -> float:
    """Official curriculum: 0.25 towards 1.0 during the first 100 epochs."""
    return 0.25 + (1.0 - 0.25) * min(1.0, epoch / 100.0)


@torch.no_grad()
def ae_validation(model, loader, device: torch.device) -> float:
    model.eval()
    total = count = 0
    for batch_index, (vertices,) in enumerate(loader, start=1):
        vertices = vertices.to(device)
        pred = model(vertices)[-1]
        total += torch.abs(pred - vertices).sum().item()
        count += vertices.numel()
        print(f"AE validation live batch={batch_index}/{len(loader)}", flush=True)
    return total / max(count, 1)


def train_autoencoder(model_cls, config: dict, train: np.ndarray, validation: np.ndarray,
                      out: Path, device: torch.device, seed: int, epochs: int,
                      batch_size: int, eval_every: int, checkpoint_every: int):
    model = model_cls(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    val_loader = make_loader(validation, max(batch_size, 32), seed, False)
    last_path, best_path = out / "ae_last.pt", out / "ae_best.pt"
    resume_signature = lamm_resume_signature(
        stage="autoencoder",
        model_config=config,
        train_array_sha256=array_sha256(train),
        validation_array_sha256=array_sha256(validation),
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        eval_every=eval_every,
        implementation_sha256=sha256(Path(__file__)),
        official_commit=LAMM_COMMIT,
    )
    start, best = (
        load_checkpoint(last_path, model, optimizer, expected_resume_signature=resume_signature)
        if last_path.exists() else (0, np.inf)
    )
    print(
        f"AE {'resume from Drive' if start else 'start'} epoch={start}/{epochs} "
        f"validation_interval={eval_every} drive_checkpoint_interval={checkpoint_every}",
        flush=True,
    )
    n_layers = config["encoder_depth"] + config["decoder_depth"] + 2
    layer_alpha = torch.cat((torch.linspace(1, 0, config["encoder_depth"] + 1),
                             torch.linspace(0, 1, config["decoder_depth"] + 1))).to(device)
    layer_alpha = layer_alpha.view(n_layers, 1, 1, 1)
    history_path = out / "ae_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() and start else []
    history = [row for row in history if int(row["epoch"]) <= start]
    steps_per_epoch = math.ceil(len(train) / batch_size)
    total_steps = epochs * steps_per_epoch
    warmup_steps = steps_per_epoch
    for epoch in range(start + 1, epochs + 1):
        train_loader = make_loader(train, batch_size, seed + epoch, True)
        model.train()
        running = 0.0
        for step_in_epoch, (vertices,) in enumerate(train_loader):
            absolute_step = (epoch - 1) * steps_per_epoch + step_in_epoch
            set_learning_rate(optimizer, cosine_learning_rate(
                absolute_step, total_steps, warmup_steps, 1e-3, 1e-6, 1e-8))
            vertices = vertices.to(device)
            outputs = model(vertices)
            targets = layer_alpha * vertices.unsqueeze(0)
            loss = (layer_alpha * torch.abs(outputs - targets)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
            completed_batches = step_in_epoch + 1
            if completed_batches % 10 == 0 or completed_batches == len(train_loader):
                print(
                    f"AE live epoch={epoch}/{epochs} "
                    f"batch={completed_batches}/{len(train_loader)}",
                    flush=True,
                )
        print(
            f"AE progress epoch={epoch}/{epochs} "
            f"train_loss={running / len(train_loader):.8f} "
            f"lr={optimizer.param_groups[0]['lr']:.3e} "
            f"next_drive_checkpoint_in={checkpoint_every - (epoch % checkpoint_every) if epoch % checkpoint_every else 0}",
            flush=True,
        )
        if epoch % eval_every == 0 or epoch == epochs:
            value = ae_validation(model, val_loader, device)
            history.append({"epoch": epoch, "train_loss": running / len(train_loader),
                            "validation_mae": value,
                            "learning_rate": optimizer.param_groups[0]["lr"]})
            print(f"AE epoch={epoch} validation_mae={value:.8f}", flush=True)
            if value < best:
                best = value
                save_checkpoint(best_path, model, optimizer, epoch, best, config, resume_signature)
            write_json(history_path, history)
            write_sha256_sidecar(history_path)
        if epoch % checkpoint_every == 0 or epoch == epochs:
            print(f"AE checkpoint write to Drive start epoch={epoch}", flush=True)
            save_checkpoint(last_path, model, optimizer, epoch, best, config, resume_signature)
            print(
                f"AE persisted to Drive epoch={epoch}: {last_path.name}, "
                f"best={best:.8f}, history={history_path.name}",
                flush=True,
            )
    load_checkpoint(best_path, model, expected_resume_signature=resume_signature)
    return model, best_path


def delta_controls(source: torch.Tensor, target: torch.Tensor, model) -> list[torch.Tensor]:
    delta = target - source
    return [delta[:, torch.as_tensor(model.control_vertices[key], device=delta.device)].reshape(len(delta), -1)
            for key in model.control_region_keys]


@torch.no_grad()
def manipulation_validation(model, values: torch.Tensor, landmarks: np.ndarray,
                            mean: torch.Tensor, std: torch.Tensor, device: torch.device,
                            batch_size: int) -> float:
    model.eval()
    n = len(values)
    free = torch.ones(values.shape[1], dtype=torch.bool, device=device)
    free[torch.as_tensor(landmarks, device=device)] = False
    total_pair_rmse = 0.0
    count = 0
    for source_i in range(n):
        target_indices = [i for i in range(n) if i != source_i]
        for begin in range(0, len(target_indices), batch_size):
            idx = target_indices[begin:begin + batch_size]
            source = values[source_i].unsqueeze(0).expand(len(idx), -1, -1).to(device)
            target = values[idx].to(device)
            pred = model((source, delta_controls(source, target, model)))[-1]
            pred_abs = pred * std + mean
            target_abs = target * std + mean
            error = pred_abs[:, free] - target_abs[:, free]
            pair_rmse = free_pair_vector_rmse(error, torch.ones(error.shape[1], dtype=torch.bool, device=device))
            total_pair_rmse += torch.sum(pair_rmse).item()
            count += int(error.shape[0])
        if (source_i + 1) % 5 == 0 or source_i + 1 == n:
            print(
                f"MANIP validation live source={source_i + 1}/{n} "
                f"({100.0 * (source_i + 1) / n:.1f}%)",
                flush=True,
            )
    return float(total_pair_rmse / max(count, 1))


def train_manipulation(model_cls, config: dict, ae_path: Path, train: np.ndarray,
                       validation: np.ndarray, landmarks: np.ndarray, mean: np.ndarray,
                       std: np.ndarray, out: Path, device: torch.device, seed: int,
                       epochs: int, batch_size: int, eval_every: int, checkpoint_every: int):
    model = model_cls(config).to(device)
    ae_state = torch.load(ae_path, map_location="cpu", weights_only=False)["model"]
    missing, unexpected = model.load_state_dict(ae_state, strict=False)
    if unexpected or any(not key.startswith("delta_control_net") for key in missing):
        raise ValueError(f"Unexpected AE transfer mismatch: missing={missing}, unexpected={unexpected}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    val_tensor = torch.from_numpy(validation.astype(np.float32, copy=False))
    mean_t = torch.from_numpy(mean.astype(np.float32)).to(device)
    std_t = torch.from_numpy(std.astype(np.float32)).to(device)
    last_path, best_path = out / "manipulation_last.pt", out / "manipulation_best.pt"
    resume_signature = lamm_resume_signature(
        stage="manipulation",
        model_config=config,
        train_array_sha256=array_sha256(train),
        validation_array_sha256=array_sha256(validation),
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        eval_every=eval_every,
        implementation_sha256=sha256(Path(__file__)),
        official_commit=LAMM_COMMIT,
        extra={
            "ae_checkpoint_sha256": sha256(ae_path),
            "landmarks": np.asarray(landmarks, dtype=np.int64).tolist(),
            "mean_sha256": array_sha256(mean),
            "std_sha256": array_sha256(std),
            "validation_objective": "mean_per_pair_vector_rmse_over_non_landmark_roi_vertices_v1",
        },
    )
    start, best = (
        load_checkpoint(last_path, model, optimizer, expected_resume_signature=resume_signature)
        if last_path.exists() else (0, np.inf)
    )
    print(
        f"MANIP {'resume from Drive' if start else 'start'} epoch={start}/{epochs} "
        f"validation_interval={eval_every} drive_checkpoint_interval={checkpoint_every}",
        flush=True,
    )
    enc_alpha = torch.linspace(1, 0, config["encoder_depth"] + 1, device=device).view(-1, 1, 1, 1)
    dec_alpha = torch.linspace(0, 1, config["decoder_depth"] + 1, device=device).view(-1, 1, 1, 1)
    history_path = out / "manipulation_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() and start else []
    history = [row for row in history if int(row["epoch"]) <= start]
    steps_per_epoch = len(train) // batch_size
    total_steps = epochs * steps_per_epoch
    warmup_steps = 10 * steps_per_epoch
    for epoch in range(start + 1, epochs + 1):
        train_loader = make_loader(train, batch_size, seed + 100000 + epoch, True, drop_last=True)
        model.train()
        running = 0.0
        alpha_max_epoch = manipulation_alpha_max(epoch)
        for step_in_epoch, (source,) in enumerate(train_loader):
            absolute_step = (epoch - 1) * steps_per_epoch + step_in_epoch
            set_learning_rate(optimizer, cosine_learning_rate(
                absolute_step, total_steps, warmup_steps, 1e-4, 1e-8, 1e-8))
            source = source.to(device)
            target = torch.roll(source, 1, 0)
            batch = len(source)
            alpha = 0.25 + (alpha_max_epoch - 0.25) * torch.rand(batch, 1, 1, device=device)
            inputs = torch.cat((source, source), dim=0)
            mixed_target = alpha * target + (1 - alpha) * source
            targets = torch.cat((source, mixed_target), dim=0)
            controls = delta_controls(source, target, model)
            controls = [torch.cat((torch.zeros_like(value), alpha.reshape(batch, 1) * value), dim=0)
                        for value in controls]
            expanded = torch.cat((enc_alpha * inputs.unsqueeze(0), dec_alpha * targets.unsqueeze(0)), dim=0)
            outputs = model((inputs, controls))
            loss = torch.abs(outputs - expanded)
            mask = torch.ones_like(loss)
            mask[config["encoder_depth"] + 1, batch:] = 0
            scalar = (mask * loss).sum() / mask.sum()
            optimizer.zero_grad(set_to_none=True)
            scalar.backward()
            optimizer.step()
            running += scalar.item()
            completed_batches = step_in_epoch + 1
            if completed_batches % 10 == 0 or completed_batches == len(train_loader):
                print(
                    f"MANIP live epoch={epoch}/{epochs} "
                    f"batch={completed_batches}/{len(train_loader)}",
                    flush=True,
                )
        print(
            f"MANIP progress epoch={epoch}/{epochs} "
            f"train_loss={running / len(train_loader):.8f} "
            f"lr={optimizer.param_groups[0]['lr']:.3e} "
            f"alpha_max={alpha_max_epoch:.4f} "
            f"next_drive_checkpoint_in={checkpoint_every - (epoch % checkpoint_every) if epoch % checkpoint_every else 0}",
            flush=True,
        )
        if epoch % eval_every == 0 or epoch == epochs:
            value = manipulation_validation(model, val_tensor, landmarks, mean_t, std_t, device,
                                            max(batch_size, 32))
            history.append({"epoch": epoch, "train_loss": running / len(train_loader),
                            "validation_free_roi_rmse": value,
                            "alpha_max": alpha_max_epoch,
                            "learning_rate": optimizer.param_groups[0]["lr"]})
            print(f"MANIP epoch={epoch} validation_free_roi_rmse={value:.8f}", flush=True)
            if value < best:
                best = value
                save_checkpoint(best_path, model, optimizer, epoch, best, config, resume_signature)
            write_json(history_path, history)
            write_sha256_sidecar(history_path)
        if epoch % checkpoint_every == 0 or epoch == epochs:
            print(f"MANIP checkpoint write to Drive start epoch={epoch}", flush=True)
            save_checkpoint(last_path, model, optimizer, epoch, best, config, resume_signature)
            print(
                f"MANIP persisted to Drive epoch={epoch}: {last_path.name}, "
                f"best={best:.8f}, history={history_path.name}",
                flush=True,
            )
    load_checkpoint(best_path, model, expected_resume_signature=resume_signature)
    return model, best_path


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    write_sha256_sidecar(path)


def read_valid_csv(path: Path) -> list[dict[str, str]] | None:
    if not valid_sha256_sidecar(path):
        return None
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return None


@torch.no_grad()
def evaluate_test(model, by_id: dict[str, dict], pairs: list[tuple[str, str]], mean: np.ndarray,
                  std: np.ndarray, out: Path, device: torch.device, batch_size: int,
                  checkpoint_signature: str, max_pairs: int = 0, chunk_pairs: int = 320):
    from rhinoform.strict_protocol_patch import strict_metric_rows

    selected = pairs[:max_pairs] if max_pairs else pairs
    mean_t = torch.from_numpy(mean.astype(np.float32)).to(device)
    std_t = torch.from_numpy(std.astype(np.float32)).to(device)
    model.eval()
    rows: list[dict] = []
    latency = []

    def infer(batch_pairs: list[tuple[str, str]]) -> tuple[list[dict], float]:
        source_abs = np.stack([by_id[s]["vertices"] for s, _ in batch_pairs]).astype(np.float32)
        target_abs = np.stack([by_id[t]["vertices"] for _, t in batch_pairs]).astype(np.float32)
        source = torch.from_numpy((source_abs - mean) / std).to(device)
        target = torch.from_numpy((target_abs - mean) / std).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        prediction = model((source, delta_controls(source, target, model)))[-1]
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0 / len(batch_pairs)
        pred_abs = (prediction * std_t + mean_t).cpu().numpy()
        pred_delta = (pred_abs - source_abs).reshape(len(batch_pairs), -1)
        batch_rows = strict_metric_rows(by_id, batch_pairs, pred_delta)
        return batch_rows, elapsed

    chunk_pairs = max(batch_size, int(chunk_pairs))
    chunk_root = out / "clean_test_chunks" / checkpoint_signature[:16]
    chunk_root.mkdir(parents=True, exist_ok=True)
    progress_path = out / "clean_test_progress.json"
    for chunk_begin in range(0, len(selected), chunk_pairs):
        chunk_end = min(chunk_begin + chunk_pairs, len(selected))
        chunk_path = chunk_root / f"pairs_{chunk_begin:05d}_{chunk_end:05d}.csv"
        chunk_rows = read_valid_csv(chunk_path)
        reusable = chunk_rows is not None and len(chunk_rows) == chunk_end - chunk_begin
        if reusable:
            for local_index, row in enumerate(chunk_rows):
                expected_pair = selected[chunk_begin + local_index]
                if (str(row["source_id"]), str(row["target_id"])) != expected_pair:
                    reusable = False
                    break
        if reusable:
            rows.extend(chunk_rows)
            print(
                f"CLEAN TEST resume from Drive {chunk_end}/{len(selected)}: {chunk_path.name}",
                flush=True,
            )
            continue

        chunk_rows = []
        for begin in range(chunk_begin, chunk_end, batch_size):
            batch_end = min(begin + batch_size, chunk_end)
            batch_pairs = selected[begin:batch_end]
            batch_rows, elapsed = infer(batch_pairs)
            latency.append(elapsed)
            for offset, row in enumerate(batch_rows):
                row["pair_index"] = float(begin + offset)
            chunk_rows.extend(batch_rows)
            print(
                f"CLEAN TEST live {batch_end}/{len(selected)} "
                f"({100.0 * batch_end / len(selected):.1f}%)",
                flush=True,
            )
        write_csv(chunk_path, chunk_rows)
        rows.extend(chunk_rows)
        write_json(progress_path, {
            "status": "IN_PROGRESS" if chunk_end < len(selected) else "COMPLETE",
            "checkpoint_sha256": checkpoint_signature,
            "completed_pairs": chunk_end,
            "total_pairs": len(selected),
            "latest_drive_chunk": str(chunk_path),
            "latest_drive_chunk_sha256": sha256(chunk_path),
        })
        write_sha256_sidecar(progress_path)
        print(
            f"CLEAN TEST persisted to Drive {chunk_end}/{len(selected)}: {chunk_path}",
            flush=True,
        )

    if not latency:
        # All chunks were resumed; take one fresh model-only timing sample.
        _, elapsed = infer(selected[:batch_size])
        latency.append(elapsed)

    csv_path = out / "identity_bootstrap_pair_metrics_lamm.csv"
    write_csv(csv_path, rows)
    metrics = ("roi_rmse", "landmark_rmse", "dorsum_rmse", "tip_rmse",
               "edge_strain_p95", "normal_flip_pct", "abs_flip_pct", "missed_flip_pct")
    summary = {
        "method": "lamm_cvpr2024_five_semantic_patches",
        "n_pairs": len(rows),
        "means": {metric: float(np.mean([float(row[metric]) for row in rows])) for metric in metrics},
        "latency_ms_per_pair_model_only_median": float(np.median(latency)),
        "pair_metrics_sha256": sha256(csv_path),
    }
    write_json(out / "lamm_test_summary.json", summary)
    write_sha256_sidecar(out / "lamm_test_summary.json")
    write_json(progress_path, {
        "status": "COMPLETE",
        "checkpoint_sha256": checkpoint_signature,
        "completed_pairs": len(rows),
        "total_pairs": len(selected),
        "final_pair_metrics": str(csv_path),
        "final_pair_metrics_sha256": sha256(csv_path),
    })
    write_sha256_sidecar(progress_path)
    print(f"CLEAN TEST final artefacts persisted to Drive: {csv_path}", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fyp-root", type=Path, required=True,
                        help="Google Drive FYP final containing data/meshes, splits and roi")
    parser.add_argument("--lamm-root", type=Path, required=True,
                        help="Official LAMM checkout pinned to the declared commit")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument(
        "--defer-test-evaluation",
        action="store_true",
        help="Train/select on train+validation only; do not load confirmation test meshes.",
    )
    parser.add_argument("--confirmation-policy", type=Path, default=None)
    parser.add_argument("--test-access-receipt", type=Path, default=None)
    parser.add_argument(
        "--all-models-freeze",
        type=Path,
        default=None,
        help="Exact pre-test freeze for every matched RB-SR/LAMM model artifact.",
    )
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--ae-epochs", type=int, default=1500)
    parser.add_argument("--manipulation-epochs", type=int, default=1500)
    parser.add_argument("--ae-batch-size", type=int, default=32)
    parser.add_argument("--manipulation-batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=1,
                        help="persist an atomic resume checkpoint to Drive every N completed epochs")
    parser.add_argument("--max-test-pairs", type=int, default=0,
                        help="nonzero is a smoke test and must not be reported")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.eval_every < 1 or args.checkpoint_every < 1:
        raise ValueError("--eval-every and --checkpoint-every must be positive")

    args.out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    if not torch.cuda.is_available() and not args.audit_only:
        raise RuntimeError("Select a GPU runtime in Colab")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve the exact official implementation before any possible holdout
    # receipt or mesh access.  A matching commit with tracked/untracked changes
    # is not the frozen official baseline.
    git_head = subprocess.check_output(
        ["git", "-C", str(args.lamm_root), "rev-parse", "HEAD"], text=True
    ).strip()
    git_dirty = subprocess.check_output(
        ["git", "-C", str(args.lamm_root), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    if git_head != LAMM_COMMIT:
        raise ValueError(f"LAMM checkout is {git_head}, expected {LAMM_COMMIT}")
    if git_dirty:
        raise ValueError("LAMM official checkout has tracked or untracked changes")

    split_path = args.split_manifest or (args.fyp_root / "splits/facescape_847/split_manifest.json")
    confirmation_policy_hash = None
    all_models_freeze_hash = None
    access_receipt_chain = None
    deferred_training_provenance_hash = None
    training_implementation_hash = None
    if args.confirmation_policy is not None:
        if args.audit_only and not args.defer_test_evaluation:
            raise ValueError("Final-rerun holdout audit must also defer holdout evaluation")
        if not valid_sha256_sidecar(args.confirmation_policy):
            raise RuntimeError("Confirmation policy or SHA-256 sidecar is invalid")
        policy = json.loads(args.confirmation_policy.read_text(encoding="utf-8"))
        if policy.get("status") != "FROZEN_INTERNAL_FINAL_RERUN_HOLDOUT_POLICY":
            raise ValueError("Confirmation policy is not frozen")
        if policy.get("test_access") is not False or int(policy.get("test_access_count", -1)) != 0:
            raise ValueError("Confirmation policy does not describe a locked final-rerun holdout")
        if sha256(split_path) != policy.get("split_manifest_sha256"):
            raise ValueError("LAMM split/confirmation policy hash mismatch")
        confirmation_policy_hash = sha256(args.confirmation_policy)
        if not args.defer_test_evaluation and not args.audit_only:
            manipulation_path = args.out / "manipulation_best.pt"
            ae_path = args.out / "ae_best.pt"
            if not valid_sha256_sidecar(manipulation_path) or not valid_sha256_sidecar(ae_path):
                raise RuntimeError("LAMM confirmation test requires frozen valid training checkpoints")
            training_provenance_path = args.out / "run_provenance.json"
            if not valid_sha256_sidecar(training_provenance_path):
                raise RuntimeError("LAMM deferred-training provenance or sidecar is invalid")
            training_provenance = json.loads(training_provenance_path.read_text(encoding="utf-8"))
            if training_provenance.get("status") != "TRAINED_VALIDATION_SELECTED_TEST_DEFERRED":
                raise ValueError("LAMM confirmation checkpoint was not frozen before test access")
            if training_provenance.get("source_hashes", {}).get("split") != sha256(split_path):
                raise ValueError("LAMM deferred-training provenance/split mismatch")
            if training_provenance.get("confirmation_policy_sha256") != confirmation_policy_hash:
                raise ValueError("LAMM deferred-training provenance/policy mismatch")
            if training_provenance.get("lamm_commit") != git_head:
                raise ValueError("LAMM deferred-training provenance/official commit mismatch")
            deferred_training_provenance_hash = sha256(training_provenance_path)
            training_implementation_hash = training_provenance.get("implementation_sha256")
            if not isinstance(training_implementation_hash, str) or not training_implementation_hash:
                raise ValueError("LAMM deferred-training provenance is missing its implementation hash")
            expected_checkpoints = training_provenance.get("checkpoints", {})
            if expected_checkpoints.get("ae") != sha256(ae_path) or expected_checkpoints.get("manipulation") != sha256(manipulation_path):
                raise ValueError("LAMM frozen checkpoint hash changed after deferred training")
            if args.all_models_freeze is None:
                raise ValueError("LAMM confirmation test requires --all-models-freeze")
            if not valid_sha256_sidecar(args.all_models_freeze):
                raise RuntimeError("All-model freeze or SHA-256 sidecar is invalid")
            all_models_freeze = json.loads(args.all_models_freeze.read_text(encoding="utf-8"))
            validate_all_models_freeze(
                all_models_freeze,
                expected={
                    "confirmation_policy_sha256": confirmation_policy_hash,
                    "split_manifest_sha256": sha256(split_path),
                    "lamm_ae_sha256": sha256(ae_path),
                    "lamm_manipulation_sha256": sha256(manipulation_path),
                    "lamm_official_commit": git_head,
                    "lamm_deferred_training_provenance_sha256": deferred_training_provenance_hash,
                    "lamm_training_implementation_sha256": training_implementation_hash,
                },
                expected_implementations={
                    "rhinoform/confirmation.py": sha256(
                        Path(__file__).resolve().parents[2] / "rhinoform/confirmation.py"
                    ),
                    "experiments/lamm/run_lamm_facescape.py": sha256(Path(__file__)),
                },
            )
            all_models_freeze_hash = sha256(args.all_models_freeze)
            if args.test_access_receipt is None:
                raise ValueError("LAMM confirmation test requires --test-access-receipt")
            access_receipt_chain = {
                "confirmation_policy_sha256": confirmation_policy_hash,
                "ae_checkpoint_sha256": sha256(ae_path),
                "manipulation_checkpoint_sha256": sha256(manipulation_path),
                "split_manifest_sha256": sha256(split_path),
                "all_models_freeze_sha256": all_models_freeze_hash,
                "lamm_official_commit": git_head,
                "lamm_deferred_training_provenance_sha256": deferred_training_provenance_hash,
                "lamm_training_implementation_sha256": training_implementation_hash,
            }
            if args.test_access_receipt.is_file():
                if not valid_sha256_sidecar(args.test_access_receipt):
                    raise RuntimeError("Existing LAMM test-access receipt or sidecar is invalid")
                receipt = json.loads(args.test_access_receipt.read_text(encoding="utf-8"))
                if any(receipt.get(key) != value for key, value in access_receipt_chain.items()):
                    raise ValueError("LAMM receipt belongs to a different frozen hash chain")
                if receipt.get("status") not in {"IN_PROGRESS_RESUMABLE", "COMPLETE"}:
                    raise ValueError("Unrecognised LAMM test-access receipt status")
                print(f"LAMM confirmation test exact-hash resume: {receipt['status']}", flush=True)
            else:
                write_json(
                    args.test_access_receipt,
                    {
                        "status": "IN_PROGRESS_RESUMABLE",
                        "test_access_count": 1,
                        "started_at_utc": datetime.now(timezone.utc).isoformat(),
                        **access_receipt_chain,
                    },
                )
                write_sha256_sidecar(args.test_access_receipt)
                print("LAMM confirmation receipt persisted before loading test meshes", flush=True)

    by_id, arrays, train_ids, val_ids, test_ids, pairs, regions, controls, protocol, hashes = load_dataset(
        args.fyp_root,
        split_manifest=split_path,
        load_test=not args.defer_test_evaluation,
    )
    protocol = Protocol(**{**asdict(protocol), "seed": args.seed})
    protocol_audit_path = args.out / "protocol_audit.json"
    write_json(protocol_audit_path, {
        "status": "PASS", "protocol": asdict(protocol), "source_hashes": hashes,
        "test_pair_head": pairs[:5], "test_pair_tail": pairs[-5:],
        "control_vertices_by_patch": controls, "patch_order": PATCH_ORDER,
        "inference_contract": "source dense mesh + nine target-source control displacements only",
    })
    write_sha256_sidecar(protocol_audit_path)
    if args.audit_only:
        print("Protocol audit PASS", flush=True)
        return 0

    sys.path.insert(0, str(args.lamm_root))
    from models import LAMM

    mean = arrays["train"].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = arrays["train"].std(axis=0, dtype=np.float64).astype(np.float32) + 1e-7
    atomic_savez_compressed(args.out / "train_only_normalisation.npz", mean=mean, std=std)
    normalised = {key: ((value - mean) / std).astype(np.float32) for key, value in arrays.items()}
    region_file = args.out / "region_ids.pickle"
    atomic_pickle(region_file, {i: region for i, region in enumerate(regions)})

    frozen_confirmation_test = (
        args.confirmation_policy is not None
        and not args.defer_test_evaluation
        and not args.audit_only
    )
    resumed_from_checkpoint = frozen_confirmation_test or (
        (args.out / "ae_last.pt").exists()
        or (args.out / "manipulation_last.pt").exists()
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    manipulation_cfg = model_config(region_file, controls, manipulation=True)
    if frozen_confirmation_test:
        # The one-shot confirmation is evaluation-only.  The checkpoint hashes,
        # split, policy and implementation were validated above before the test
        # receipt was written.  Never enter either training function here: a
        # missing/renamed *_last.pt must not silently trigger 1500 fresh epochs.
        ae_path = args.out / "ae_best.pt"
        manipulation_path = args.out / "manipulation_best.pt"
        state = torch.load(manipulation_path, map_location="cpu", weights_only=False)
        if state.get("model_config") != manipulation_cfg:
            raise ValueError("Frozen LAMM manipulation checkpoint/model configuration mismatch")
        if "model" not in state or "epoch" not in state or "resume_signature" not in state:
            raise ValueError("Frozen LAMM manipulation checkpoint is incomplete")
        model = LAMM(manipulation_cfg).to(device)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        del state
        print(
            "LAMM frozen confirmation: loaded manipulation_best.pt directly; training bypassed",
            flush=True,
        )
    else:
        ae_cfg = model_config(region_file, controls, manipulation=False)
        _, ae_path = train_autoencoder(
            LAMM, ae_cfg, normalised["train"], normalised["validation"],
            args.out, device, args.seed, args.ae_epochs,
            args.ae_batch_size, args.eval_every, args.checkpoint_every,
        )
        landmarks = by_id[train_ids[0]]["landmarks"]
        model, manipulation_path = train_manipulation(
            LAMM, manipulation_cfg, ae_path, normalised["train"], normalised["validation"],
            landmarks, mean, std, args.out, device, args.seed, args.manipulation_epochs,
            args.manipulation_batch_size, args.eval_every, args.checkpoint_every,
        )
    training_invocation_elapsed = time.perf_counter() - training_started
    if args.defer_test_evaluation:
        training_report = {
            "status": "TRAINED_VALIDATION_SELECTED_TEST_DEFERRED",
            "run_role": "internal_final_rerun_holdout_training",
            "protocol": asdict(protocol),
            "source_hashes": hashes,
            "confirmation_policy_sha256": confirmation_policy_hash,
            "test_access": False,
            "lamm_official_repository": "https://github.com/michaeltrs/LAMM",
            "lamm_commit": LAMM_COMMIT,
            "model": manipulation_cfg,
            "training": {
                "ae_epochs": args.ae_epochs,
                "manipulation_epochs": args.manipulation_epochs,
                "ae_batch_size": args.ae_batch_size,
                "manipulation_batch_size": args.manipulation_batch_size,
                "eval_every": args.eval_every,
                "checkpoint_every": args.checkpoint_every,
                "lr_schedule": "step-wise warm-up plus cosine decay",
                "alpha_schedule": "0.25 to 1.0 over first 100 manipulation epochs",
                "validation_objective": "mean_per_pair_vector_rmse_over_non_landmark_roi_vertices_v1",
                "resume_contract": "data_config_commit_and_implementation_bound_v1",
            },
            "checkpoints": {"ae": sha256(ae_path), "manipulation": sha256(manipulation_path)},
            "implementation_sha256": sha256(Path(__file__)),
            "training_current_invocation_elapsed_sec": float(training_invocation_elapsed),
        }
        training_report_path = args.out / "run_provenance.json"
        write_json(training_report_path, training_report)
        write_sha256_sidecar(training_report_path)
        print(json.dumps(training_report, indent=2), flush=True)
        return 0
    evaluation_started = time.perf_counter()
    summary = evaluate_test(model, by_id, pairs, mean, std, args.out, device,
                            max(args.manipulation_batch_size, 32), sha256(manipulation_path),
                            args.max_test_pairs)
    evaluation_elapsed = time.perf_counter() - evaluation_started
    is_full_run = (
        args.ae_epochs == 1500 and args.manipulation_epochs == 1500
        and args.ae_batch_size == 32 and args.manipulation_batch_size == 16
        and args.max_test_pairs == 0
    )
    provenance_path = args.out / "run_provenance.json"
    write_json(provenance_path, {
        "status": "COMPLETE" if is_full_run else "SMOKE_ONLY",
        "run_role": (
            "internal_final_rerun_holdout"
            if args.confirmation_policy is not None
            else ("primary_predeclared" if args.seed == 20260609 else "confirmatory_seed")
        ),
        "protocol": asdict(protocol), "source_hashes": hashes,
        "confirmation_policy_sha256": confirmation_policy_hash,
        "all_models_freeze_sha256": all_models_freeze_hash,
        "deferred_training_provenance_sha256": deferred_training_provenance_hash,
        "training_implementation_sha256": training_implementation_hash,
        "lamm_official_repository": "https://github.com/michaeltrs/LAMM",
        "lamm_commit": LAMM_COMMIT,
        "model": manipulation_cfg,
        "training": {"ae_epochs": args.ae_epochs,
                     "manipulation_epochs": args.manipulation_epochs,
                     "ae_batch_size": args.ae_batch_size,
                     "manipulation_batch_size": args.manipulation_batch_size,
                     "eval_every": args.eval_every,
                     "checkpoint_every": args.checkpoint_every,
                     "lr_schedule": "step-wise warm-up plus cosine decay",
                     "alpha_schedule": "0.25 to 1.0 over first 100 manipulation epochs"},
        "compute": {
            "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "resumed_from_checkpoint": bool(resumed_from_checkpoint),
            "frozen_confirmation_training_bypassed": bool(frozen_confirmation_test),
            "training_current_invocation_elapsed_sec": float(training_invocation_elapsed),
            "training_timing_is_complete_wall_clock": bool(not resumed_from_checkpoint),
            "test_evaluation_elapsed_sec": float(evaluation_elapsed),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        },
        "checkpoints": {"ae": sha256(ae_path), "manipulation": sha256(manipulation_path)},
        "environment": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "summary": summary,
    })
    write_sha256_sidecar(provenance_path)
    if args.test_access_receipt is not None and access_receipt_chain is not None:
        receipt = json.loads(args.test_access_receipt.read_text(encoding="utf-8"))
        receipt.update(
            {
                "status": "COMPLETE",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "test_access_count": 1,
                "test_summary_sha256": sha256(args.out / "lamm_test_summary.json"),
                "pair_metrics_sha256": sha256(args.out / "identity_bootstrap_pair_metrics_lamm.csv"),
                "run_provenance_sha256": sha256(provenance_path),
            }
        )
        write_json(args.test_access_receipt, receipt)
        write_sha256_sidecar(args.test_access_receipt)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
