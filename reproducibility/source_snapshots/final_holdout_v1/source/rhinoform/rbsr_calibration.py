from __future__ import annotations

import numpy as np


RIDGE_IDENTITY_METRICS = ("roi_rmse", "normal_flip_pct", "edge_strain_p95")


def ridge_reference_matches(
    computed: dict[str, object],
    frozen: dict[str, object],
    *,
    absolute_tolerance: float = 1e-8,
) -> bool:
    """Check Ridge identity allowing only float serialisation round-off."""
    return all(
        abs(float(computed[metric]) - float(frozen[metric])) <= absolute_tolerance
        for metric in RIDGE_IDENTITY_METRICS
    )


def select_strict_operating_point(
    candidates: list[dict[str, object]],
    ridge_reference: dict[str, object],
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    """Select validation RMSE only among candidates no less safe than Ridge."""
    ridge_rmse = float(ridge_reference["roi_rmse"])
    ridge_flip = float(ridge_reference["normal_flip_pct"])
    assessed: list[dict[str, object]] = []
    for candidate in candidates:
        row = dict(candidate)
        rmse = float(row["roi_rmse"])
        flip = float(row["normal_flip_pct"])
        row["rmse_improvement_vs_ridge"] = ridge_rmse - rmse
        row["new_flip_delta_vs_ridge"] = flip - ridge_flip
        row["strictly_beats_ridge"] = rmse < ridge_rmse and flip <= ridge_flip
        assessed.append(row)
    feasible = [row for row in assessed if bool(row["strictly_beats_ridge"])]
    if not feasible:
        return None, assessed
    selected = min(
        feasible,
        key=lambda row: (
            float(row["roi_rmse"]),
            float(row["normal_flip_pct"]),
            float(row.get("edge_strain_p95", float("inf"))),
            float(row.get("gate_mean", float("inf"))),
            str(row["label"]),
        ),
    )
    return selected, assessed


def select_certified_projection(
    candidates: list[dict[str, object]],
    ridge_reference: dict[str, object],
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    """Select only a complete hard-certified projection that beats Ridge."""
    selected, assessed = select_strict_operating_point(candidates, ridge_reference)
    for row in assessed:
        metric_success = bool(row["strictly_beats_ridge"])
        certified = float(row.get("certificate_rate", 0.0)) == 1.0
        row["hard_certificate_complete"] = certified
        row["strictly_beats_ridge"] = metric_success and certified
    feasible = [row for row in assessed if bool(row["strictly_beats_ridge"])]
    if not feasible:
        return None, assessed
    return min(
        feasible,
        key=lambda row: (
            float(row["roi_rmse"]),
            float(row["normal_flip_pct"]),
            float(row.get("edge_strain_p95", float("inf"))),
            str(row["label"]),
        ),
    ), assessed


def validate_test_unlock(
    freeze: dict[str, object],
    *,
    base_hash: str,
    rbsr_hash: str,
) -> dict[str, object]:
    """Validate the immutable validation evidence required before test access."""
    if freeze.get("status") != "FROZEN_STRICT_VALIDATION_CALIBRATION":
        raise ValueError("RB-SR test access requires a feasible frozen validation calibration")
    if freeze.get("test_access") is not True:
        raise ValueError("Validation freeze does not unlock one-shot test evaluation")
    if freeze.get("base_model_package_sha256") != base_hash:
        raise ValueError("Calibration freeze/base package hash mismatch")
    if freeze.get("rbsr_package_sha256") != rbsr_hash:
        raise ValueError("Calibration freeze/gate package hash mismatch")
    selected = freeze.get("selected")
    if not isinstance(selected, dict) or selected.get("strictly_beats_ridge") is not True:
        raise ValueError("Calibration freeze does not contain a strict Ridge-beating operating point")
    return dict(selected)


def validate_projection_test_unlock(
    freeze: dict[str, object],
    *,
    base_hash: str,
    rbsr_hash: str,
    projection_signature: str,
) -> dict[str, object]:
    """Validate a frozen, validation-selected hard-projection contract."""
    if freeze.get("status") != "FROZEN_CERTIFIED_RIDGE_FOLD_PROJECTION":
        raise ValueError("Test access requires a frozen certified Ridge-fold projection")
    if freeze.get("test_access") is not True:
        raise ValueError("Projection freeze does not unlock one-shot test evaluation")
    if freeze.get("base_model_package_sha256") != base_hash:
        raise ValueError("Projection freeze/base package hash mismatch")
    if freeze.get("rbsr_package_sha256") != rbsr_hash:
        raise ValueError("Projection freeze/gate package hash mismatch")
    if freeze.get("projection_signature") != projection_signature:
        raise ValueError("Projection freeze/signature mismatch")
    zero_identity = freeze.get("zero_gate_ridge_identity")
    if not isinstance(zero_identity, dict) or zero_identity.get("passed") is not True:
        raise ValueError("Projection freeze lacks an exact zero-gate Ridge fallback identity")
    selected = freeze.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("Projection freeze has no selected operating point")
    if selected.get("strictly_beats_ridge") is not True or float(selected.get("certificate_rate", 0.0)) != 1.0:
        raise ValueError("Projection freeze is not both Ridge-beating and completely certified")
    return dict(selected)


def fuse_calibrated_residual(
    ridge_prediction: np.ndarray,
    neural_prediction: np.ndarray,
    learned_gate: np.ndarray,
    *,
    logit_offset: float = 0.0,
    force_zero_gate: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse a frozen RB-SR residual after validation-only gate calibration.

    ``force_zero_gate`` is a first-class fallback rather than a limiting logit
    value so that its prediction is bitwise identical to the Ridge anchor.
    """
    ridge = np.asarray(ridge_prediction, dtype=np.float32)
    neural = np.asarray(neural_prediction, dtype=np.float32)
    gate = np.asarray(learned_gate, dtype=np.float32)
    if ridge.shape != neural.shape:
        raise ValueError("Ridge and neural predictions must have identical shapes")
    if ridge.ndim != 2 or gate.ndim != 2 or ridge.shape[0] != gate.shape[0]:
        raise ValueError("Expected flat predictions [pairs, 3*vertices] and gate [pairs, vertices]")
    if ridge.shape[1] != gate.shape[1] * 3:
        raise ValueError("Gate vertex count does not match the flattened prediction")
    if force_zero_gate:
        return ridge.copy(), np.zeros_like(gate)

    if not np.isfinite(logit_offset) or float(logit_offset) < 0.0:
        raise ValueError("logit_offset must be a finite non-negative value")
    eps = np.finfo(np.float32).eps
    clipped = np.clip(gate, eps, 1.0 - eps)
    logits = np.log(clipped) - np.log1p(-clipped)
    calibrated = 1.0 / (1.0 + np.exp(-(logits - float(logit_offset))))
    residual = (neural - ridge).reshape(ridge.shape[0], gate.shape[1], 3)
    prediction = ridge.reshape(ridge.shape[0], gate.shape[1], 3) + calibrated[..., None] * residual
    return prediction.reshape(ridge.shape).astype(np.float32), calibrated.astype(np.float32)
