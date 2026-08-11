from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rhinoform.cvae import fit_truncated_pca
from rhinoform.data import edge_index
from rhinoform.repro import atomic_write_json, seed_worker_factory, set_global_seed, sha256_file


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        device = torch.device(value)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        if device.type == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError("--device mps requested, but MPS is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_selected_rows(repo: Path, identity_ids: set[str], workers: int = 16) -> dict[str, dict]:
    manifest = json.loads((repo / "manifest.json").read_text(encoding="utf-8"))
    by_id: dict[str, dict] = {}
    selected = [row for row in manifest["rows"] if str(row["subject_id"]) in identity_ids]

    def load_one(row: dict) -> tuple[str, dict]:
        subject_id = str(row["subject_id"])
        npz_path = Path(row["npz_path"])
        if not npz_path.is_absolute():
            npz_path = repo / npz_path
        with np.load(npz_path, allow_pickle=False) as data:
            loaded = {
                "subject_id": subject_id,
                "vertices": np.asarray(data["vertices"], dtype=np.float32),
                "faces": np.asarray(data["faces"], dtype=np.int64),
                "landmarks": np.asarray(data["local_landmarks"], dtype=np.int64),
            }
        return subject_id, loaded

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for index, (subject_id, loaded) in enumerate(pool.map(load_one, selected), start=1):
            by_id[subject_id] = loaded
            if index % 100 == 0 or index == len(selected):
                print(f"loaded identities {index}/{len(selected)}", flush=True)
    missing = identity_ids - set(by_id)
    if missing:
        raise ValueError(f"Missing requested identities: {sorted(missing)}")
    return by_id


class ResidualBasisExpert(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(width, hidden), nn.SiLU(), nn.LayerNorm(hidden)])
            width = hidden
        self.body = nn.Sequential(*layers)
        self.output = nn.Linear(width, output_dim)
        # The initial model is exactly the frozen ridge anchor.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor, output_scale: torch.Tensor) -> torch.Tensor:
        return self.output(self.body(features)) * output_scale


def identity_codes(
    by_id: dict[str, dict],
    identity_ids: list[str],
    components: np.ndarray,
) -> dict[str, np.ndarray]:
    basis_t = np.asarray(components, dtype=np.float32).T
    return {
        subject_id: by_id[subject_id]["vertices"].reshape(-1) @ basis_t
        for subject_id in identity_ids
    }


def source_codes(by_id: dict[str, dict], identity_ids: list[str], source_pca: dict) -> dict[str, np.ndarray]:
    mean = np.asarray(source_pca["mean"], dtype=np.float32)
    components_t = np.asarray(source_pca["components"], dtype=np.float32).T
    return {
        subject_id: (by_id[subject_id]["vertices"].reshape(-1) - mean) @ components_t
        for subject_id in identity_ids
    }


def pair_design(
    by_id: dict[str, dict],
    pairs: list[tuple[str, str]],
    source_code_by_id: dict[str, np.ndarray],
    basis_code_by_id: dict[str, np.ndarray],
    package: dict,
    handle_mean: np.ndarray | None = None,
    handle_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    landmarks = next(iter(by_id.values()))["landmarks"]
    controls = np.stack(
        [
            (by_id[target]["vertices"] - by_id[source]["vertices"])[landmarks].reshape(-1)
            for source, target in pairs
        ],
        axis=0,
    ).astype(np.float32)
    raw_condition = np.concatenate(
        [controls, np.stack([source_code_by_id[source] for source, _ in pairs], axis=0)],
        axis=1,
    )
    condition = (
        raw_condition - np.asarray(package["cond_mean"], dtype=np.float32)
    ) / np.asarray(package["cond_std"], dtype=np.float32)

    ridge_w = np.asarray(package["ridge_cond"][0], dtype=np.float32)
    ridge_b = np.asarray(package["ridge_cond"][1], dtype=np.float32)
    landmark_columns = (landmarks[:, None] * 3 + np.arange(3)[None, :]).reshape(-1)
    ridge_landmarks = condition @ ridge_w[:, landmark_columns] + ridge_b[landmark_columns]
    handle_error = controls - ridge_landmarks
    if handle_mean is None:
        handle_mean = handle_error.mean(axis=0, keepdims=True)
        handle_std = np.maximum(handle_error.std(axis=0, keepdims=True), 1e-6)
    handle_normalized = (handle_error - handle_mean) / handle_std
    features = np.concatenate([condition, handle_normalized], axis=1).astype(np.float32)

    basis = np.asarray(package["residual_basis"], dtype=np.float32)
    ridge_basis_w = ridge_w @ basis.T
    ridge_basis_b = ridge_b @ basis.T
    ridge_coeff = condition @ ridge_basis_w + ridge_basis_b
    true_coeff = np.stack(
        [basis_code_by_id[target] - basis_code_by_id[source] for source, target in pairs],
        axis=0,
    )
    residual_coeff = (true_coeff - ridge_coeff).astype(np.float32)
    return features, residual_coeff, condition.astype(np.float32), handle_mean, handle_std


def predict_coefficients(
    model: ResidualBasisExpert,
    features: np.ndarray,
    output_scale: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    scale = torch.as_tensor(output_scale, dtype=torch.float32, device=device)
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=device)
            predictions.append(model(batch, scale).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def ridge_pair_sse(
    by_id: dict[str, dict],
    pairs: list[tuple[str, str]],
    condition: np.ndarray,
    package: dict,
    batch_size: int,
) -> np.ndarray:
    ridge_w = np.asarray(package["ridge_cond"][0], dtype=np.float32)
    ridge_b = np.asarray(package["ridge_cond"][1], dtype=np.float32)
    values = np.empty(len(pairs), dtype=np.float64)
    for start in range(0, len(pairs), batch_size):
        stop = min(len(pairs), start + batch_size)
        ridge = condition[start:stop] @ ridge_w + ridge_b
        truth = np.stack(
            [
                (by_id[target]["vertices"] - by_id[source]["vertices"]).reshape(-1)
                for source, target in pairs[start:stop]
            ],
            axis=0,
        )
        diff = ridge - truth
        values[start:stop] = np.sum(diff * diff, axis=1)
    return values


def roi_curve_from_coefficients(
    ridge_sse: np.ndarray,
    target_residual_coeff: np.ndarray,
    correction_coeff: np.ndarray,
    alphas: np.ndarray,
    vertex_count: int,
) -> dict[str, np.ndarray]:
    # PCA rows are orthonormal, so this is the exact full-mesh SSE after adding
    # a basis-space correction; no dense reconstruction is needed per epoch.
    cross = np.sum(target_residual_coeff * correction_coeff, axis=1)
    correction_sse = np.sum(correction_coeff * correction_coeff, axis=1)
    pair_sse = np.stack(
        [
            np.maximum(
                ridge_sse - 2.0 * float(alpha) * cross + float(alpha) ** 2 * correction_sse,
                0.0,
            )
            for alpha in alphas
        ],
        axis=0,
    )
    pair_rmse = np.sqrt(pair_sse / float(vertex_count))
    return {
        "mean_pair_rmse": pair_rmse.mean(axis=1),
        "global_rmse": np.sqrt(pair_sse.mean(axis=1) / float(vertex_count)),
        "pair_rmse": pair_rmse,
    }


def exact_geometry_metrics(
    by_id: dict[str, dict],
    pairs: list[tuple[str, str]],
    condition: np.ndarray,
    correction_coeff: np.ndarray,
    alpha: float,
    package: dict,
    batch_size: int,
) -> dict[str, float]:
    template = next(iter(by_id.values()))
    faces = template["faces"]
    edges = edge_index(faces)
    basis = np.asarray(package["residual_basis"], dtype=np.float32)
    ridge_w = np.asarray(package["ridge_cond"][0], dtype=np.float32)
    ridge_b = np.asarray(package["ridge_cond"][1], dtype=np.float32)
    roi_values: list[np.ndarray] = []
    strain_values: list[np.ndarray] = []
    flip_values: list[np.ndarray] = []
    for start in range(0, len(pairs), batch_size):
        stop = min(len(pairs), start + batch_size)
        sources = np.stack([by_id[source]["vertices"] for source, _ in pairs[start:stop]], axis=0)
        targets = np.stack([by_id[target]["vertices"] for _, target in pairs[start:stop]], axis=0)
        ridge = (condition[start:stop] @ ridge_w + ridge_b).reshape(stop - start, -1, 3)
        correction = (correction_coeff[start:stop] @ basis).reshape(stop - start, -1, 3)
        prediction = ridge + float(alpha) * correction
        truth = targets - sources
        roi_values.append(np.sqrt(np.mean(np.sum((prediction - truth) ** 2, axis=2), axis=1)))

        source_edge = sources[:, edges[:, 0]] - sources[:, edges[:, 1]]
        edited = sources + prediction
        edited_edge = edited[:, edges[:, 0]] - edited[:, edges[:, 1]]
        source_length = np.linalg.norm(source_edge, axis=2)
        edited_length = np.linalg.norm(edited_edge, axis=2)
        strain = np.abs(edited_length - source_length) / np.maximum(source_length, 1e-8)
        strain_values.append(np.percentile(strain, 95, axis=1))

        source_tri = sources[:, faces]
        edited_tri = edited[:, faces]
        source_normal = np.cross(
            source_tri[:, :, 1] - source_tri[:, :, 0],
            source_tri[:, :, 2] - source_tri[:, :, 0],
        )
        edited_normal = np.cross(
            edited_tri[:, :, 1] - edited_tri[:, :, 0],
            edited_tri[:, :, 2] - edited_tri[:, :, 0],
        )
        flip_values.append(100.0 * np.mean(np.sum(source_normal * edited_normal, axis=2) < 0.0, axis=1))
    return {
        "roi_rmse": float(np.mean(np.concatenate(roi_values))),
        "edge_strain_p95": float(np.mean(np.concatenate(strain_values))),
        "normal_flip_pct": float(np.mean(np.concatenate(flip_values))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a validation-only identity-basis ridge residual expert.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base-model-package", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--basis-dim", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pair-rmse-weight", type=float, default=0.25)
    parser.add_argument("--correction-weight", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--alpha-grid", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    set_global_seed(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    package = torch.load(args.base_model_package, map_location="cpu", weights_only=False)
    train_ids = [str(value) for value in package["train_ids"]]
    val_ids = [str(value) for value in package["val_ids"]]
    train_pairs = [(str(a), str(b)) for a, b in package["train_pairs"]]
    val_pairs = [(str(a), str(b)) for a, b in package["val_pairs"]]
    print(
        f"device={device} train_ids={len(train_ids)} val_ids={len(val_ids)} "
        f"train_pairs={len(train_pairs)} val_pairs={len(val_pairs)}",
        flush=True,
    )
    by_id = load_selected_rows(Path(args.repo), set(train_ids) | set(val_ids))

    source_components = np.asarray(package["source_pca"]["components"], dtype=np.float32)
    if args.basis_dim <= source_components.shape[0]:
        residual_basis = source_components[: args.basis_dim].copy()
        basis_source = "frozen_source_pca"
    else:
        print(f"fitting {args.basis_dim}-D identity shape PCA", flush=True)
        shape_matrix = np.stack([by_id[subject_id]["vertices"].reshape(-1) for subject_id in train_ids], axis=0)
        residual_basis = fit_truncated_pca(shape_matrix, args.basis_dim)["components"].astype(np.float32)
        basis_source = "train_identity_pca"
    package_for_run = dict(package)
    package_for_run["residual_basis"] = residual_basis

    all_ids = train_ids + val_ids
    source_code_by_id = source_codes(by_id, all_ids, package["source_pca"])
    basis_code_by_id = identity_codes(by_id, all_ids, residual_basis)
    train_features, train_target, train_condition, handle_mean, handle_std = pair_design(
        by_id, train_pairs, source_code_by_id, basis_code_by_id, package_for_run
    )
    val_features, val_target, val_condition, _, _ = pair_design(
        by_id,
        val_pairs,
        source_code_by_id,
        basis_code_by_id,
        package_for_run,
        handle_mean,
        handle_std,
    )
    output_scale = np.maximum(train_target.std(axis=0, keepdims=True), 1e-4).astype(np.float32)
    vertex_count = next(iter(by_id.values()))["vertices"].shape[0]
    print("computing frozen ridge validation errors once", flush=True)
    val_ridge_sse = ridge_pair_sse(
        by_id, val_pairs, val_condition, package_for_run, args.eval_batch_size
    )
    print(
        f"design train={train_features.shape} target={train_target.shape} "
        f"residual_scale_mean={float(output_scale.mean()):.6f}",
        flush=True,
    )

    model = ResidualBasisExpert(train_features.shape[1], train_target.shape[1], args.hidden, args.depth).to(device)
    scale_t = torch.as_tensor(output_scale, dtype=torch.float32, device=device)
    features_t = torch.as_tensor(train_features, dtype=torch.float32)
    target_t = torch.as_tensor(train_target, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    alphas = np.asarray([float(value) for value in args.alpha_grid.split(",") if value.strip()], dtype=np.float32)
    best_score = math.inf
    best_epoch = 0
    best_alpha = 0.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    patience = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(
            TensorDataset(features_t, target_t),
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            worker_init_fn=seed_worker_factory(args.seed),
            num_workers=0,
        )
        model.train()
        epoch_loss = 0.0
        for features_cpu, target_cpu in loader:
            features = features_cpu.to(device)
            target = target_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features, scale_t)
            normalized_error = (prediction - target) / scale_t
            coefficient_loss = F.smooth_l1_loss(normalized_error, torch.zeros_like(normalized_error))
            pair_loss = torch.mean(torch.sqrt(torch.sum((prediction - target) ** 2, dim=1) + 1e-8))
            correction_penalty = torch.mean((prediction / scale_t) ** 2)
            loss = (
                coefficient_loss
                + args.pair_rmse_weight * pair_loss
                + args.correction_weight * correction_penalty
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(features)

        if epoch % args.eval_every != 0 and epoch != args.epochs:
            continue
        val_coeff = predict_coefficients(model, val_features, output_scale, device, args.eval_batch_size)
        curve = roi_curve_from_coefficients(
            val_ridge_sse,
            val_target,
            val_coeff,
            alphas,
            vertex_count,
        )
        selected = int(np.argmin(curve["mean_pair_rmse"]))
        score = float(curve["mean_pair_rmse"][selected])
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / len(train_features),
            "val_roi_rmse": score,
            "alpha": float(alphas[selected]),
            "ridge_val_roi_rmse": float(curve["mean_pair_rmse"][0]),
        }
        history.append(record)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_alpha = float(alphas[selected])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += args.eval_every
        print(
            f"epoch={epoch} loss={record['train_loss']:.6f} val_roi={score:.6f} "
            f"alpha={record['alpha']:.2f} best={best_score:.6f}",
            flush=True,
        )
        if patience >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Training produced no validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    val_coeff = predict_coefficients(model, val_features, output_scale, device, args.eval_batch_size)
    selected_curve = roi_curve_from_coefficients(
        val_ridge_sse,
        val_target,
        val_coeff,
        alphas,
        vertex_count,
    )
    oracle_curve = roi_curve_from_coefficients(
        val_ridge_sse,
        val_target,
        val_target,
        np.asarray([0.0, 1.0], dtype=np.float32),
        vertex_count,
    )
    print("computing exact validation geometry for ridge and selected expert", flush=True)
    ridge_metrics = exact_geometry_metrics(
        by_id, val_pairs, val_condition, val_coeff, 0.0, package_for_run, args.eval_batch_size
    )
    selected_metrics = exact_geometry_metrics(
        by_id, val_pairs, val_condition, val_coeff, best_alpha, package_for_run, args.eval_batch_size
    )

    model_package = {
        "method": "identity_basis_ridge_residual_expert",
        "status": "validation_selected_no_test_access",
        "args": vars(args),
        "base_model_package": str(Path(args.base_model_package)),
        "base_model_package_sha256": sha256_file(Path(args.base_model_package)),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "basis_source": basis_source,
        "residual_basis": residual_basis,
        "handle_mean": handle_mean,
        "handle_std": handle_std,
        "output_scale": output_scale,
        "input_dim": train_features.shape[1],
        "output_dim": train_target.shape[1],
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "selected_alpha": best_alpha,
        "best_epoch": best_epoch,
        "best_validation_roi_rmse": best_score,
    }
    model_path = out_dir / "residual_basis_expert.pt"
    torch.save(model_package, model_path)
    report = {
        "method": model_package["method"],
        "status": model_package["status"],
        "device": str(device),
        "elapsed_sec": time.perf_counter() - started,
        "basis_source": basis_source,
        "basis_dim": int(residual_basis.shape[0]),
        "best_epoch": best_epoch,
        "selected_alpha": best_alpha,
        "validation": {
            "ridge": ridge_metrics,
            "selected_expert": selected_metrics,
            "relative_roi_improvement_vs_ridge_pct": 100.0
            * (ridge_metrics["roi_rmse"] - selected_metrics["roi_rmse"])
            / ridge_metrics["roi_rmse"],
            "basis_oracle_roi_rmse": float(oracle_curve["mean_pair_rmse"][1]),
            "basis_oracle_relative_improvement_vs_ridge_pct": 100.0
            * (oracle_curve["mean_pair_rmse"][0] - oracle_curve["mean_pair_rmse"][1])
            / oracle_curve["mean_pair_rmse"][0],
            "alpha_curve": [
                {"alpha": float(alpha), "roi_rmse": float(value)}
                for alpha, value in zip(alphas, selected_curve["mean_pair_rmse"])
            ],
        },
        "history": history,
        "model_package": str(model_path),
        "model_package_sha256": sha256_file(model_path),
    }
    atomic_write_json(out_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
