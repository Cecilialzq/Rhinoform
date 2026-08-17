from __future__ import annotations

import hashlib
import math


def validate_frozen_classical_configuration(
    payload: dict[str, object],
) -> dict[str, dict[str, float | int | None]]:
    """Validate and normalise the three pre-test classical configurations."""
    required = {"laplacian", "bilaplacian", "arap"}
    if set(payload) != required:
        raise ValueError("Frozen classical configuration must contain exactly Laplacian, bi-Laplacian and ARAP")
    normalised: dict[str, dict[str, float | int | None]] = {}
    for method in sorted(required):
        raw = payload[method]
        if not isinstance(raw, dict):
            raise ValueError(f"Frozen classical configuration is not a mapping: {method}")
        handle_weight = float(raw.get("handle_weight", 0.0))
        system_ridge = float(raw.get("system_ridge", 0.0))
        if not math.isfinite(handle_weight) or handle_weight <= 0.0:
            raise ValueError(f"Invalid classical handle weight: {method}")
        if not math.isfinite(system_ridge) or system_ridge <= 0.0:
            raise ValueError(f"Invalid classical system ridge: {method}")
        arap_iter: int | None = None
        if method == "arap":
            raw_iter = raw.get("arap_iter")
            if isinstance(raw_iter, bool) or raw_iter is None or int(raw_iter) <= 0:
                raise ValueError("Frozen ARAP iteration count must be a positive integer")
            arap_iter = int(raw_iter)
        normalised[method] = {
            "handle_weight": handle_weight,
            "system_ridge": system_ridge,
            "arap_iter": arap_iter,
        }
    return normalised


def validate_all_models_freeze(
    payload: dict[str, object],
    *,
    expected: dict[str, str],
    expected_implementations: dict[str, str] | None = None,
) -> None:
    """Fail closed unless a pre-test freeze matches the live hash chain.

    Callers construct ``expected`` from artifacts they have just validated and
    hashed.  This deliberately checks only exact equality; there is no repair,
    tolerance, or fallback after blind-test access.
    """
    if payload.get("status") != "ALL_MATCHED_MODELS_FROZEN_BEFORE_TEST":
        raise ValueError("All-model freeze does not have the frozen pre-test status")
    if payload.get("test_access") is not False:
        raise ValueError("All-model freeze was not created before test access")
    if not expected:
        raise ValueError("At least one live artifact hash must be checked")
    for key, live_hash in expected.items():
        if not isinstance(live_hash, str) or not live_hash:
            raise ValueError(f"Live hash is empty or invalid: {key}")
        frozen_hash = payload.get(key)
        if not isinstance(frozen_hash, str) or not frozen_hash:
            raise ValueError(f"All-model freeze is missing a valid hash: {key}")
        if frozen_hash != live_hash:
            raise ValueError(f"All-model freeze/live artifact mismatch: {key}")
    frozen_implementations = payload.get("implementation_hashes")
    if expected_implementations is not None:
        if not isinstance(frozen_implementations, dict):
            raise ValueError("All-model freeze is missing implementation hashes")
        for path, live_hash in expected_implementations.items():
            if not isinstance(live_hash, str) or not live_hash:
                raise ValueError(f"Live implementation hash is empty or invalid: {path}")
            if frozen_implementations.get(path) != live_hash:
                raise ValueError(f"Frozen evaluation implementation changed: {path}")


def _identity_sort_key(value: str) -> tuple[int, object]:
    try:
        return 0, int(value)
    except ValueError:
        return 1, value


def blind_confirmation_partition(
    parent: dict[str, object],
    *,
    seed: int,
    val_size: int,
    test_size: int,
) -> dict[str, object]:
    """Repartition only previously non-test identities without reading meshes.

    SHA-256 ranking makes the split independent of platform RNG versions. The
    parent's already-observed test identities are excluded from every group.
    """
    train_parent = [str(value) for value in parent["train_pool_ids"]]
    val_parent = [str(value) for value in parent["val_ids"]]
    observed_test = [str(value) for value in parent["test_ids"]]
    eligible = train_parent + val_parent
    if len(eligible) != len(set(eligible)):
        raise ValueError("Parent train/validation identities overlap or contain duplicates")
    if set(eligible) & set(observed_test):
        raise ValueError("Parent non-test identities overlap the already-observed test")
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= len(eligible):
        raise ValueError("Confirmation validation/test sizes must leave a non-empty train pool")

    namespace = "rhinoform_internal_blind_confirmation_v1"
    ranked = sorted(
        eligible,
        key=lambda subject_id: hashlib.sha256(
            f"{namespace}|{int(seed)}|{subject_id}".encode("utf-8")
        ).hexdigest(),
    )
    test_ids = ranked[:test_size]
    val_ids = ranked[test_size : test_size + val_size]
    train_ids = ranked[test_size + val_size :]
    ordered = lambda values: sorted(values, key=_identity_sort_key)
    return {
        "schema": namespace,
        "selection_algorithm": "ascending SHA256(schema|seed|subject_id); test then validation then train",
        "split_seed": int(seed),
        "train_pool_ids": ordered(train_ids),
        "val_ids": ordered(val_ids),
        "test_ids": ordered(test_ids),
        "excluded_previously_observed_test_ids": ordered(observed_test),
        "sizes": {"train_pool": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
    }
