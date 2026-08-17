from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import edge_index, load_rows, ridge_predict
from .repro import (
    atomic_torch_save,
    atomic_write_json,
    capture_rng_state,
    restore_rng_state,
    set_global_seed,
    sha256_file,
    sha256_json,
    validate_torch_artifact,
)
from .train import NeuralFieldCVAE, build_static_vertex_features, pair_conditions, predict_field


STRICT_GATE_RECONSTRUCTION_OBJECTIVE = "mean_per_pair_vector_rmse_over_non_landmark_roi_vertices_v1"
STRICT_GATE_EXACT_HANDLE_CONTRACT = "target_controls_hard_overwrite_for_all_projection_modes_v1"
# Exact source hash used by the completed 2026-08-09 gate.  The next revision
# added only the package-level ``training_implementation_sha256`` metadata key;
# the numerical training path was unchanged.  Its checkpoint signature binds
# this hash to the full args/base/pair contract.
STRICT_GATE_PRE_METADATA_IMPLEMENTATION_SHA256 = "bfa0dbe88835b9982a570d32f8a95daa51a38888cf12439f5e5d22a322c695a2"


def validate_gate_package_training_contract(package: dict, package_path: Path) -> dict[str, object]:
    """Validate current packages or the one exact pre-metadata strict package."""
    if package.get("reconstruction_objective") != STRICT_GATE_RECONSTRUCTION_OBJECTIVE:
        raise ValueError("RB-SR package does not use the strict free-ROI RMSE objective")
    if package.get("exact_handle_contract") != STRICT_GATE_EXACT_HANDLE_CONTRACT:
        raise ValueError("RB-SR package does not enforce the exact handle contract")
    if package.get("args", {}).get("projection") != "none":
        raise ValueError("Certified RB-SR requires a projection=none residual proposer")

    live_hash = sha256_file(Path(__file__))
    recorded_hash = package.get("training_implementation_sha256")
    if recorded_hash is not None:
        if recorded_hash != live_hash:
            raise ValueError("RB-SR package was trained by a different gate implementation")
        return {
            "status": "EXACT_CURRENT_IMPLEMENTATION_HASH_MATCH",
            "training_implementation_sha256": recorded_hash,
        }

    checkpoint_path = Path(package_path).with_name("rbsr_gate_last.pt")
    if not validate_torch_artifact(
        checkpoint_path,
        required_keys=("tag", "epoch", "gate_state_dict", "optimizer_state", "resume_signature", "rng_state"),
        repair_sidecar=False,
    ):
        raise RuntimeError("Pre-metadata gate requires its valid hash-chained final checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = dict(checkpoint.get("args", {}))
    if int(checkpoint.get("epoch", -1)) != int(checkpoint_args.get("epochs", -2)):
        raise ValueError("Pre-metadata gate checkpoint did not complete the declared epochs")
    if sha256_json(checkpoint_args) != sha256_json(package.get("args", {})):
        raise ValueError("Pre-metadata gate package/checkpoint args mismatch")
    if sha256_json(checkpoint.get("best_record")) != sha256_json(package.get("best_validation")):
        raise ValueError("Pre-metadata gate package/checkpoint validation selection mismatch")
    signature_args = {
        key: value for key, value in checkpoint_args.items()
        if key not in {"resume_checkpoint", "device"}
    }
    expected_signature = sha256_json({
        "schema": "rbsr_gate_resume_contract_v2_strict_free_rmse_exact_handles",
        "args": signature_args,
        "base_model_package_sha256": package["base_model_package_sha256"],
        "train_pairs": package["train_pairs"],
        "val_pairs": package["val_pairs"],
        "implementation_sha256": STRICT_GATE_PRE_METADATA_IMPLEMENTATION_SHA256,
        "reconstruction_objective": STRICT_GATE_RECONSTRUCTION_OBJECTIVE,
        "exact_handle_contract": STRICT_GATE_EXACT_HANDLE_CONTRACT,
    })
    if checkpoint.get("resume_signature") != expected_signature:
        raise ValueError("Pre-metadata gate checkpoint does not match the frozen strict implementation contract")
    return {
        "status": "VERIFIED_PRE_METADATA_STRICT_IMPLEMENTATION_SIGNATURE",
        "training_implementation_sha256": STRICT_GATE_PRE_METADATA_IMPLEMENTATION_SHA256,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
    }


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SpatialRiskGate(nn.Module):
    def __init__(self, vertex_dim: int, cond_dim: int, hidden: int, initial_gate: float) -> None:
        super().__init__()
        input_dim = vertex_dim + cond_dim + 9
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        self.logit = nn.Linear(hidden, 1)
        nn.init.zeros_(self.logit.weight)
        initial = min(max(float(initial_gate), 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.logit.bias, math.log(initial / (1.0 - initial)))

    def forward(
        self,
        vertex_features: torch.Tensor,
        condition: torch.Tensor,
        source_normalized: torch.Tensor,
        ridge_delta: torch.Tensor,
        neural_residual: torch.Tensor,
    ) -> torch.Tensor:
        batch = condition.shape[0]
        vertices = vertex_features.shape[0]
        static = vertex_features.unsqueeze(0).expand(batch, vertices, -1)
        cond = condition.unsqueeze(1).expand(batch, vertices, -1)
        inputs = torch.cat([static, cond, source_normalized, ridge_delta, neural_residual], dim=-1)
        return torch.sigmoid(self.logit(self.body(inputs)))


def build_rbf_projection(template_vertices: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    vertices = np.asarray(template_vertices, dtype=np.float64)
    landmarks = np.asarray(landmarks, dtype=np.int64)
    handles = vertices[landmarks]
    pair_distance = np.linalg.norm(handles[:, None, :] - handles[None, :, :], axis=2)
    nonzero = pair_distance[pair_distance > 1e-12]
    sigma = float(np.median(nonzero)) if len(nonzero) else 1.0
    sigma = max(sigma, 1e-6)
    kernel_handles = np.exp(-(pair_distance**2) / (2.0 * sigma**2))
    all_distance = np.linalg.norm(vertices[:, None, :] - handles[None, :, :], axis=2)
    kernel_all = np.exp(-(all_distance**2) / (2.0 * sigma**2))
    basis = kernel_all @ np.linalg.inv(kernel_handles + 1e-8 * np.eye(len(landmarks)))
    basis[landmarks] = np.eye(len(landmarks))
    return basis.astype(np.float32)


def project_handles(
    prediction: torch.Tensor,
    target_controls: torch.Tensor,
    landmarks: torch.Tensor,
    projection_basis: torch.Tensor | None,
) -> torch.Tensor:
    if projection_basis is None:
        projected = prediction.clone()
    else:
        error = target_controls - prediction[:, landmarks]
        correction = torch.einsum("vk,bkd->bvd", projection_basis, error)
        projected = prediction + correction
    # The nine controls are hard constraints under every projection mode.
    # ``projection=none`` means no spatial interpolation, not unconstrained handles.
    projected = projected.clone()
    projected[:, landmarks] = target_controls
    return projected


def strict_free_roi_rmse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    landmarks: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Mean per-pair vector RMSE over non-landmark ROI vertices.

    This is the differentiable analogue of the strict scorer's headline
    ``roi_rmse``.  It deliberately excludes the hard-fixed controls and takes
    the Euclidean vector error before averaging vertices.
    """
    if prediction.ndim != 3 or prediction.shape[-1] != 3 or prediction.shape != target.shape:
        raise ValueError("prediction and target must both have shape [pairs, vertices, 3]")
    free = torch.ones(prediction.shape[1], dtype=torch.bool, device=prediction.device)
    free[landmarks] = False
    if not bool(torch.any(free)):
        raise ValueError("At least one free ROI vertex is required")
    squared_norm = torch.sum((prediction[:, free] - target[:, free]).square(), dim=-1)
    per_pair_mse = torch.mean(squared_norm, dim=1)
    if epsilon > 0.0:
        per_pair_rmse = torch.sqrt(per_pair_mse.clamp_min(float(epsilon)))
    else:
        per_pair_rmse = torch.sqrt(per_pair_mse)
    return torch.mean(per_pair_rmse)


def strict_free_pair_rmse(
    prediction: np.ndarray,
    target: np.ndarray,
    landmarks: np.ndarray,
) -> np.ndarray:
    """NumPy validation counterpart of :func:`strict_free_roi_rmse_loss`."""
    prediction_arr = np.asarray(prediction).reshape(np.asarray(target).shape)
    target_arr = np.asarray(target)
    if prediction_arr.ndim != 3 or prediction_arr.shape[-1] != 3:
        raise ValueError("prediction and target must both have shape [pairs, vertices, 3]")
    free = np.ones(prediction_arr.shape[1], dtype=bool)
    free[np.asarray(landmarks, dtype=np.int64)] = False
    if not np.any(free):
        raise ValueError("At least one free ROI vertex is required")
    squared_norm = np.sum((prediction_arr[:, free] - target_arr[:, free]) ** 2, axis=2)
    return np.sqrt(np.mean(squared_norm, axis=1))


def gate_deployment_status(best_validation: dict[str, object], mode: str) -> str:
    """Separate a trainable proposer from a validation-feasible soft gate."""
    if mode == "primal_dual" and not bool(best_validation.get("feasible", False)):
        return "RESIDUAL_PROPOSER_ONLY_REQUIRES_CERTIFIED_HARD_PROJECTION"
    return "VALIDATION_SELECTED_GATE_REQUIRES_STRICT_OPERATING_POINT_FREEZE"


def _orientation_deficit(
    source: torch.Tensor, edited: torch.Tensor, faces: torch.Tensor, orientation_margin: float, temperature: float
) -> torch.Tensor:
    """Per-face softplus orientation deficit of ``edited`` relative to ``source`` normals."""
    source_tri = source[:, faces]
    edited_tri = edited[:, faces]
    source_normal = torch.cross(source_tri[:, :, 1] - source_tri[:, :, 0], source_tri[:, :, 2] - source_tri[:, :, 0], dim=-1)
    edited_normal = torch.cross(edited_tri[:, :, 1] - edited_tri[:, :, 0], edited_tri[:, :, 2] - edited_tri[:, :, 0], dim=-1)
    cosine = torch.sum(source_normal * edited_normal, dim=-1) / (
        torch.linalg.norm(source_normal, dim=-1) * torch.linalg.norm(edited_normal, dim=-1)
    ).clamp_min(1e-12)
    return F.softplus((float(orientation_margin) - cosine) / temperature)


def _strain_excess(
    source: torch.Tensor, edited: torch.Tensor, edge0: torch.Tensor, edge1: torch.Tensor,
    strain_threshold: float, temperature: float,
) -> torch.Tensor:
    """Per-edge softplus strain excess of ``edited`` over the source edge lengths."""
    source_length = torch.linalg.norm(source[:, edge0] - source[:, edge1], dim=-1).clamp_min(1e-8)
    edited_edge = edited[:, edge0] - edited[:, edge1]
    strain = torch.abs(torch.linalg.norm(edited_edge, dim=-1) - source_length) / source_length
    return F.softplus((strain - float(strain_threshold)) / temperature)


def geometric_surrogates(
    source: torch.Tensor,
    prediction: torch.Tensor,
    faces: torch.Tensor,
    edge0: torch.Tensor,
    edge1: torch.Tensor,
    orientation_margin: float,
    strain_threshold: float,
    temperature: float,
    target: torch.Tensor | None = None,
    baseline: str = "source",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable orientation / strain surrogates.

    baseline="source" (legacy): penalise every fold/strain of the prediction
    relative to the SOURCE mesh. This conflates artefacts the model introduces
    with folds/strain that are INTRINSIC to the true i2i deformation (a real
    person->person nose difference reverses ~2% of faces by itself).

    baseline="target": penalise only the EXCESS over the true target's own
    intrinsic deficit, i.e. the model-induced *new* fold/strain (the n_new
    analogue from Gate 2B). Requires ``target`` (the ground-truth delta). With a
    monotone softplus, relu(deficit_pred - deficit_target) > 0 exactly when the
    prediction folds/strains a face MORE than the true target does there.
    """
    edited = source + prediction
    pred_orient = _orientation_deficit(source, edited, faces, orientation_margin, temperature)
    pred_strain = _strain_excess(source, edited, edge0, edge1, strain_threshold, temperature)
    if baseline == "target":
        if target is None:
            raise ValueError("baseline='target' requires the ground-truth target delta")
        target_edited = source + target
        tgt_orient = _orientation_deficit(source, target_edited, faces, orientation_margin, temperature)
        tgt_strain = _strain_excess(source, target_edited, edge0, edge1, strain_threshold, temperature)
        orientation = temperature * F.relu(pred_orient - tgt_orient)
        strain_excess = temperature * F.relu(pred_strain - tgt_strain)
    else:
        orientation = temperature * pred_orient
        strain_excess = temperature * pred_strain
    return orientation.mean(), strain_excess.mean()


def predict_gate_batches(
    gate_model: SpatialRiskGate,
    vertex_features: torch.Tensor,
    condition: np.ndarray,
    source: np.ndarray,
    ridge: np.ndarray,
    cvae: np.ndarray,
    center: torch.Tensor,
    scale: float,
    landmarks: torch.Tensor,
    controls: np.ndarray,
    projection_basis: torch.Tensor | None,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    device = next(gate_model.parameters()).device
    predictions: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    gate_model.eval()
    with torch.no_grad():
        for start in range(0, len(condition), batch_size):
            stop = min(len(condition), start + batch_size)
            cond = torch.as_tensor(condition[start:stop], dtype=torch.float32, device=device)
            src = torch.as_tensor(source[start:stop], dtype=torch.float32, device=device)
            anchor = torch.as_tensor(ridge[start:stop], dtype=torch.float32, device=device).reshape(stop - start, -1, 3)
            neural = torch.as_tensor(cvae[start:stop], dtype=torch.float32, device=device).reshape(stop - start, -1, 3)
            ctrl = torch.as_tensor(controls[start:stop], dtype=torch.float32, device=device).reshape(stop - start, -1, 3)
            source_normalized = (src - center) / float(scale)
            residual = neural - anchor
            gate = gate_model(vertex_features, cond, source_normalized, anchor, residual)
            pred = anchor + gate * residual
            pred = project_handles(pred, ctrl, landmarks, projection_basis)
            predictions.append(pred.cpu().numpy().reshape(stop - start, -1))
            gates.append(gate.cpu().numpy().reshape(stop - start, -1))
    return np.concatenate(predictions, axis=0), np.concatenate(gates, axis=0)


def proxy_summary(
    source: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    gate: np.ndarray,
    landmarks: np.ndarray,
    faces: np.ndarray,
    edges: np.ndarray,
    orientation_margin: float,
    strain_threshold: float,
    temperature: float,
    device: torch.device,
    batch_size: int,
    baseline: str = "source",
) -> dict[str, float]:
    prediction_arr = np.asarray(prediction, dtype=np.float32).reshape(target.shape).copy()
    target_arr = np.asarray(target, dtype=np.float32)
    landmarks_arr = np.asarray(landmarks, dtype=np.int64)
    # Validation uses the identical hard-control and free-ROI contract as the
    # strict pair-level scorer, including when the learned projection is none.
    prediction_arr[:, landmarks_arr] = target_arr[:, landmarks_arr]
    pair_rmse = strict_free_pair_rmse(prediction_arr, target_arr, landmarks_arr)
    orientation_total = 0.0
    strain_total = 0.0
    surrogate_pairs = 0
    faces_t = torch.as_tensor(faces, dtype=torch.long, device=device)
    edge0 = torch.as_tensor(edges[:, 0], dtype=torch.long, device=device)
    edge1 = torch.as_tensor(edges[:, 1], dtype=torch.long, device=device)
    with torch.no_grad():
        for start in range(0, len(source), batch_size):
            stop = min(len(source), start + batch_size)
            src = torch.as_tensor(source[start:stop], dtype=torch.float32, device=device)
            pred = torch.as_tensor(prediction_arr[start:stop], dtype=torch.float32, device=device)
            tgt = None
            if baseline == "target":
                tgt = torch.as_tensor(target_arr[start:stop], dtype=torch.float32, device=device).reshape(stop - start, -1, 3)
            orientation, strain = geometric_surrogates(
                src,
                pred,
                faces_t,
                edge0,
                edge1,
                orientation_margin,
                strain_threshold,
                temperature,
                target=tgt,
                baseline=baseline,
            )
            batch_pairs = stop - start
            orientation_total += float(orientation.cpu()) * batch_pairs
            strain_total += float(strain.cpu()) * batch_pairs
            surrogate_pairs += batch_pairs
    return {
        "roi_rmse": float(np.mean(pair_rmse)),
        "orientation_surrogate": orientation_total / max(surrogate_pairs, 1),
        "strain_surrogate": strain_total / max(surrogate_pairs, 1),
        "gate_mean": float(np.mean(gate)),
        "gate_p95": float(np.percentile(gate, 95)),
        "gate_active_fraction_0p5": float(np.mean(gate >= 0.5)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the RBSR gate-only constrained prototype.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base-model-package", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dual-lr", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=10.0)
    parser.add_argument("--tv-weight", type=float, default=0.01)
    parser.add_argument("--gate-mean-weight", type=float, default=0.001)
    parser.add_argument("--orientation-margin", type=float, default=0.0)
    parser.add_argument("--strain-threshold", type=float, default=0.1)
    parser.add_argument("--surrogate-temperature", type=float, default=0.02)
    parser.add_argument(
        "--surrogate-baseline",
        choices=["source", "target"],
        default="source",
        help=(
            "source: legacy penalty of all folds/strain vs the source mesh. "
            "target: baseline-relative penalty of only the model-induced NEW folds/strain "
            "beyond the true i2i target's intrinsic geometry (decontaminated budget)."
        ),
    )
    parser.add_argument("--orientation-budget-multiplier", type=float, default=1.0)
    parser.add_argument("--strain-budget-multiplier", type=float, default=1.0)
    parser.add_argument("--constraint-tolerance", type=float, default=0.02)
    parser.add_argument("--projection", choices=["rbf", "none"], default="rbf")
    parser.add_argument("--mode", choices=["primal_dual", "unconstrained", "weighted"], default="primal_dual")
    parser.add_argument("--orientation-weight", type=float, default=1.0)
    parser.add_argument("--strain-weight", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-pairs", type=int, default=0, help="Smoke-test limiter; 0 uses the frozen train pairs.")
    parser.add_argument("--max-val-pairs", type=int, default=0, help="Smoke-test limiter; 0 uses the frozen validation pairs.")
    parser.add_argument("--warm-start-package", default="", help="Optional prior RBSR gate package selected on validation only.")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume-checkpoint", default="")
    args = parser.parse_args()

    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive")
    if args.warm_start_package and args.resume_checkpoint:
        raise ValueError("Use either --warm-start-package or --resume-checkpoint, not both")

    set_global_seed(args.seed, deterministic=True)
    device = resolve_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = Path(args.base_model_package)
    if not validate_torch_artifact(
        base_path,
        required_keys=("args", "cvae_state_dict", "feature_template_sha256", "ridge_cond"),
    ):
        raise RuntimeError("Base package or SHA-256 sidecar is invalid")
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    if not bool(base.get("args", {}).get("defer_test_evaluation", False)):
        raise RuntimeError("Gate training requires a clean base trained with test evaluation deferred")
    if base.get("feature_template_schema") != "mean_neutral_roi_of_training_identities_v1":
        raise RuntimeError("Gate training requires the clean train-only feature-template schema")
    train_ids = [str(value) for value in base["train_ids"]]
    val_ids = [str(value) for value in base["val_ids"]]
    print(f"Loading train/validation data and frozen anchor/residual model on {device}", flush=True)
    _, by_id = load_rows(Path(args.repo), allowed_ids=set(train_ids) | set(val_ids))
    train_pairs = [(str(a), str(b)) for a, b in base["train_pairs"]]
    val_pairs = [(str(a), str(b)) for a, b in base["val_pairs"]]
    if args.max_train_pairs > 0:
        train_pairs = train_pairs[: args.max_train_pairs]
    if args.max_val_pairs > 0:
        val_pairs = val_pairs[: args.max_val_pairs]
    template = by_id[train_ids[0]]
    faces = np.asarray(template["faces"], dtype=np.int64)
    edges = edge_index(faces)
    landmarks_np = np.asarray(template["landmarks"], dtype=np.int64)
    vertex_features_np, static = build_static_vertex_features(
        by_id,
        train_ids,
        use_subunit_features=bool(base["use_subunit_features"]),
    )
    base_template_sha256 = base.get("feature_template_sha256")
    if not base_template_sha256:
        raise RuntimeError("Base package predates the train-only feature template fix; retrain the base model first")
    if static["template_sha256"] != base_template_sha256:
        raise RuntimeError("Recomputed training template does not match the base model package")
    center_np = np.asarray(static["center"], dtype=np.float32).reshape(1, 1, 3)
    scale = float(np.asarray(static["scale"]).reshape(-1)[0])

    cond_train, source_train, delta_train, controls_train, _, _, _ = pair_conditions(
        by_id, train_pairs, base["source_pca"], base["cond_mean"], base["cond_std"]
    )
    cond_val, source_val, delta_val, controls_val, _, _, _ = pair_conditions(
        by_id, val_pairs, base["source_pca"], base["cond_mean"], base["cond_std"]
    )
    ridge_train = ridge_predict(cond_train, base["ridge_cond"]).astype(np.float32)
    ridge_val = ridge_predict(cond_val, base["ridge_cond"]).astype(np.float32)

    obs_dim = int(np.asarray(base["obs_train_mean"]).shape[1])
    base_args = base["args"]
    cvae = NeuralFieldCVAE(
        vertex_features_np.shape[1],
        cond_train.shape[1],
        obs_dim,
        latent_dim=int(base_args["latent_dim"]),
        hidden=int(base_args["hidden"]),
    )
    cvae.load_state_dict(base["cvae_state_dict"])
    cvae.to(device).eval()
    for parameter in cvae.parameters():
        parameter.requires_grad_(False)
    print("Caching frozen CVAE train/validation predictions", flush=True)
    cvae_train = predict_field(
        cvae,
        vertex_features_np,
        cond_train,
        is_cvae=True,
        progress_label="RBSR GATE frozen CVAE train cache",
    ).astype(np.float32)
    cvae_val = predict_field(
        cvae,
        vertex_features_np,
        cond_val,
        is_cvae=True,
        progress_label="RBSR GATE frozen CVAE validation cache",
    ).astype(np.float32)
    cvae.to("cpu")
    del cvae
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    vertex_features = torch.as_tensor(vertex_features_np, dtype=torch.float32, device=device)
    center = torch.as_tensor(center_np, dtype=torch.float32, device=device)
    faces_t = torch.as_tensor(faces, dtype=torch.long, device=device)
    edge0 = torch.as_tensor(edges[:, 0], dtype=torch.long, device=device)
    edge1 = torch.as_tensor(edges[:, 1], dtype=torch.long, device=device)
    landmarks = torch.as_tensor(landmarks_np, dtype=torch.long, device=device)
    projection_basis = None
    if args.projection == "rbf":
        projection_basis = torch.as_tensor(
            build_rbf_projection(static["template_vertices"], landmarks_np), dtype=torch.float32, device=device
        )

    print("Deriving ridge validation risk budgets", flush=True)
    ridge_gate = np.zeros((len(ridge_val), len(vertex_features_np)), dtype=np.float32)
    ridge_proxy = proxy_summary(
        source_val,
        delta_val,
        ridge_val,
        ridge_gate,
        landmarks_np,
        faces,
        edges,
        args.orientation_margin,
        args.strain_threshold,
        args.surrogate_temperature,
        device,
        args.batch_size,
        baseline=args.surrogate_baseline,
    )
    orientation_budget = ridge_proxy["orientation_surrogate"] * args.orientation_budget_multiplier
    strain_budget = ridge_proxy["strain_surrogate"] * args.strain_budget_multiplier
    print(
        f"budgets orientation={orientation_budget:.8f} strain={strain_budget:.8f} "
        f"from ridge validation (surrogate_baseline={args.surrogate_baseline})",
        flush=True,
    )

    gate_model = SpatialRiskGate(
        vertex_features_np.shape[1], cond_train.shape[1], args.hidden, float(base["selected_alpha"])
    ).to(device)
    optimizer = torch.optim.AdamW(gate_model.parameters(), lr=args.lr, weight_decay=1e-4)
    lambda_orientation = torch.tensor(0.0, dtype=torch.float32, device=device)
    lambda_strain = torch.tensor(0.0, dtype=torch.float32, device=device)
    warm_start_record: dict[str, object] | None = None
    if args.warm_start_package:
        warm_path = Path(args.warm_start_package)
        if not validate_torch_artifact(
            warm_path,
            required_keys=("gate_state_dict", "base_model_package_sha256", "best_validation"),
        ):
            raise RuntimeError("Warm-start package or SHA-256 sidecar is invalid")
        warm = torch.load(warm_path, map_location="cpu", weights_only=False)
        if warm.get("base_model_package_sha256") != sha256_file(base_path):
            raise ValueError("Warm-start gate is not chained to the supplied clean base package")
        if int(warm["gate_vertex_dim"]) != int(vertex_features_np.shape[1]) or int(warm["gate_cond_dim"]) != int(cond_train.shape[1]):
            raise ValueError("Warm-start gate dimensions do not match the current frozen base model")
        gate_model.load_state_dict(warm["gate_state_dict"])
        warm_best = warm.get("best_validation", {})
        lambda_orientation.fill_(float(warm_best.get("lambda_orientation", 0.0)))
        lambda_strain.fill_(float(warm_best.get("lambda_strain", 0.0)))
        warm_start_record = {
            "path": str(warm_path),
            "sha256": sha256_file(warm_path),
            "validation_record": warm_best,
        }
        print(
            f"Warm-started gate and dual variables from {warm_path} "
            f"(lambda_orientation={float(lambda_orientation.cpu()):.4f}, "
            f"lambda_strain={float(lambda_strain.cpu()):.4f})",
            flush=True,
        )
    best_state: dict[str, torch.Tensor] | None = None
    best_record: dict[str, float | int | bool] | None = None
    history: list[dict[str, float | int | bool]] = []
    start_epoch = 1
    signature_args = {
        key: value for key, value in vars(args).items()
        if key not in {"resume_checkpoint", "device"}
    }
    resume_signature = sha256_json({
        "schema": "rbsr_gate_resume_contract_v2_strict_free_rmse_exact_handles",
        "args": signature_args,
        "base_model_package_sha256": sha256_file(Path(args.base_model_package)),
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "implementation_sha256": sha256_file(Path(__file__)),
        "reconstruction_objective": "mean_per_pair_vector_rmse_over_non_landmark_roi_vertices_v1",
        "exact_handle_contract": "target_controls_hard_overwrite_for_all_projection_modes_v1",
        "training_implementation_sha256": sha256_file(Path(__file__)),
    })
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        if not validate_torch_artifact(
            resume_path,
            required_keys=(
                "tag", "epoch", "gate_state_dict", "optimizer_state", "resume_signature", "rng_state",
            ),
        ):
            raise RuntimeError(f"Resume checkpoint or SHA-256 sidecar is invalid: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint.get("resume_signature") != resume_signature:
            raise ValueError("Gate resume checkpoint configuration/base/pairs do not match this run")
        gate_model.load_state_dict(checkpoint["gate_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        lambda_orientation.fill_(float(checkpoint["lambda_orientation"]))
        lambda_strain.fill_(float(checkpoint["lambda_strain"]))
        best_state = checkpoint.get("best_state")
        best_record = checkpoint.get("best_record")
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["epoch"]) + 1
        restore_rng_state(checkpoint.get("rng_state"))
        print(
            f"RBSR GATE resume from Drive epoch={start_epoch - 1}/{args.epochs} "
            f"checkpoint={resume_path.name}",
            flush=True,
        )
    started = time.perf_counter()

    for epoch in range(start_epoch, args.epochs + 1):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(
            TensorDataset(torch.arange(len(train_pairs), dtype=torch.long)),
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
        )
        gate_model.train()
        epoch_loss = 0.0
        total_batches = len(loader)
        for batch_index, (indices_cpu,) in enumerate(loader, start=1):
            indices = indices_cpu.numpy()
            condition = torch.as_tensor(cond_train[indices], dtype=torch.float32, device=device)
            source = torch.as_tensor(source_train[indices], dtype=torch.float32, device=device)
            target = torch.as_tensor(delta_train[indices], dtype=torch.float32, device=device)
            controls = torch.as_tensor(controls_train[indices], dtype=torch.float32, device=device).reshape(len(indices), -1, 3)
            ridge = torch.as_tensor(ridge_train[indices], dtype=torch.float32, device=device).reshape(len(indices), -1, 3)
            neural = torch.as_tensor(cvae_train[indices], dtype=torch.float32, device=device).reshape(len(indices), -1, 3)
            source_normalized = (source - center) / scale
            residual = neural - ridge
            gate = gate_model(vertex_features, condition, source_normalized, ridge, residual)
            prediction = project_handles(ridge + gate * residual, controls, landmarks, projection_basis)
            reconstruction = strict_free_roi_rmse_loss(prediction, target, landmarks)
            orientation, strain = geometric_surrogates(
                source,
                prediction,
                faces_t,
                edge0,
                edge1,
                args.orientation_margin,
                args.strain_threshold,
                args.surrogate_temperature,
                target=target,
                baseline=args.surrogate_baseline,
            )
            gate_tv = torch.mean(torch.abs(gate[:, edge0] - gate[:, edge1]))
            gate_mean = torch.mean(gate)
            orientation_violation = orientation - float(orientation_budget)
            strain_violation = strain - float(strain_budget)
            loss = reconstruction + args.tv_weight * gate_tv + args.gate_mean_weight * gate_mean
            if args.mode == "primal_dual":
                loss = (
                    loss
                    + lambda_orientation * orientation_violation
                    + 0.5 * args.rho * torch.relu(orientation_violation).pow(2)
                    + lambda_strain * strain_violation
                    + 0.5 * args.rho * torch.relu(strain_violation).pow(2)
                )
            elif args.mode == "weighted":
                loss = loss + args.orientation_weight * orientation + args.strain_weight * strain

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate_model.parameters(), 1.0)
            optimizer.step()
            if args.mode == "primal_dual":
                with torch.no_grad():
                    lambda_orientation.copy_(
                        torch.clamp(lambda_orientation + args.dual_lr * orientation_violation.detach(), min=0.0)
                    )
                    lambda_strain.copy_(
                        torch.clamp(lambda_strain + args.dual_lr * strain_violation.detach(), min=0.0)
                    )
            epoch_loss += float(loss.detach().cpu()) * len(indices)
            if batch_index % 10 == 0 or batch_index == total_batches:
                print(
                    f"RBSR GATE live epoch={epoch}/{args.epochs} "
                    f"batch={batch_index}/{total_batches}",
                    flush=True,
                )

        validation_message = "validation=not_due"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_prediction, val_gate = predict_gate_batches(
                gate_model,
                vertex_features,
                cond_val,
                source_val,
                ridge_val,
                cvae_val,
                center,
                scale,
                landmarks,
                controls_val,
                projection_basis,
                args.batch_size,
            )
            summary = proxy_summary(
                source_val,
                delta_val,
                val_prediction,
                val_gate,
                landmarks_np,
                faces,
                edges,
                args.orientation_margin,
                args.strain_threshold,
                args.surrogate_temperature,
                device,
                args.batch_size,
                baseline=args.surrogate_baseline,
            )
            feasible = (
                summary["orientation_surrogate"] <= orientation_budget * (1.0 + args.constraint_tolerance)
                and summary["strain_surrogate"] <= strain_budget * (1.0 + args.constraint_tolerance)
            )
            violation = max(0.0, summary["orientation_surrogate"] / max(orientation_budget, 1e-12) - 1.0) + max(
                0.0, summary["strain_surrogate"] / max(strain_budget, 1e-12) - 1.0
            )
            record: dict[str, float | int | bool] = {
                "epoch": epoch,
                "train_loss": epoch_loss / len(train_pairs),
                **summary,
                "feasible": feasible,
                "relative_violation": violation,
                "lambda_orientation": float(lambda_orientation.cpu()),
                "lambda_strain": float(lambda_strain.cpu()),
            }
            history.append(record)
            if best_record is None:
                better = True
            elif args.mode != "primal_dual":
                better = float(record["roi_rmse"]) < float(best_record["roi_rmse"])
            elif feasible and not bool(best_record["feasible"]):
                better = True
            elif feasible and bool(best_record["feasible"]):
                better = float(record["roi_rmse"]) < float(best_record["roi_rmse"])
            elif not feasible and not bool(best_record["feasible"]):
                better = float(record["relative_violation"]) < float(best_record["relative_violation"])
            else:
                better = False
            if better:
                best_record = record
                best_state = {key: value.detach().cpu().clone() for key, value in gate_model.state_dict().items()}
            print(
                f"epoch={epoch} val_rmse={summary['roi_rmse']:.6f} "
                f"orientation={summary['orientation_surrogate']:.8f}/{orientation_budget:.8f} "
                f"strain={summary['strain_surrogate']:.8f}/{strain_budget:.8f} "
                f"gate={summary['gate_mean']:.4f} feasible={feasible}",
                flush=True,
            )
            validation_message = f"val_rmse={summary['roi_rmse']:.6f} feasible={feasible}"
        print(
            f"RBSR GATE epoch={epoch}/{args.epochs} "
            f"train_loss={epoch_loss / len(train_pairs):.6f} {validation_message}",
            flush=True,
        )
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            checkpoint_path = out_dir / "rbsr_gate_last.pt"
            print(f"RBSR GATE checkpoint write to Drive start epoch={epoch}", flush=True)
            atomic_torch_save(
                checkpoint_path,
                {
                    "tag": "rbsr_gate",
                    "epoch": epoch,
                    "gate_state_dict": gate_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "lambda_orientation": float(lambda_orientation.detach().cpu()),
                    "lambda_strain": float(lambda_strain.detach().cpu()),
                    "best_state": best_state,
                    "best_record": best_record,
                    "history": history,
                    "args": vars(args),
                    "resume_signature": resume_signature,
                    "rng_state": capture_rng_state(),
                },
                required_keys=(
                    "tag", "epoch", "gate_state_dict", "optimizer_state", "resume_signature", "rng_state",
                ),
            )
            print(f"RBSR GATE checkpoint persisted to Drive epoch={epoch}", flush=True)

    if best_state is None or best_record is None:
        raise RuntimeError("Training produced no evaluated checkpoint")
    gate_model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    deployment_status = gate_deployment_status(best_record, args.mode)
    package = {
        "method": "rbsr_gate_prototype",
        "args": vars(args),
        "gate_state_dict": gate_model.state_dict(),
        "gate_vertex_dim": int(vertex_features_np.shape[1]),
        "gate_cond_dim": int(cond_train.shape[1]),
        "base_model_package": str(Path(args.base_model_package)),
        "base_model_package_sha256": sha256_file(Path(args.base_model_package)),
        "orientation_budget": orientation_budget,
        "strain_budget": strain_budget,
        "ridge_validation_proxy": ridge_proxy,
        "best_validation": best_record,
        "deployment_status": deployment_status,
        "reconstruction_objective": "mean_per_pair_vector_rmse_over_non_landmark_roi_vertices_v1",
        "exact_handle_contract": "target_controls_hard_overwrite_for_all_projection_modes_v1",
        "warm_start": warm_start_record,
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "projection_basis": None if projection_basis is None else projection_basis.detach().cpu().numpy(),
        "center": center_np,
        "scale": scale,
        "feature_template_sha256": static["template_sha256"],
        "feature_template_schema": static["template_schema"],
    }
    package_path = out_dir / "rbsr_gate_model.pt"
    atomic_torch_save(
        package_path,
        package,
        required_keys=(
            "method", "gate_state_dict", "base_model_package_sha256",
            "feature_template_sha256", "best_validation", "deployment_status",
            "reconstruction_objective", "exact_handle_contract", "training_implementation_sha256",
        ),
    )
    report = {
        "method": "rbsr_gate_prototype",
        "status": "trained_validation_selected_no_test_access",
        "deployment_status": deployment_status,
        "reconstruction_objective": "mean_per_pair_vector_rmse_over_non_landmark_roi_vertices_v1",
        "exact_handle_contract": "target_controls_hard_overwrite_for_all_projection_modes_v1",
        "elapsed_sec": elapsed,
        "device": str(device),
        "best_validation": best_record,
        "warm_start": warm_start_record,
        "ridge_validation_proxy": ridge_proxy,
        "orientation_budget": orientation_budget,
        "strain_budget": strain_budget,
        "history": history,
        "model_package": str(package_path),
        "model_package_sha256": sha256_file(package_path),
    }
    atomic_write_json(out_dir / "rbsr_gate_training.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
