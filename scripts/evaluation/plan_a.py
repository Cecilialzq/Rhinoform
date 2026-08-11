"""Plan A final evaluation: ridge vs RBSR (baseline-relative flip budget).

Compares the clean (leakage-fixed) linear ridge anchor against the RBSR gate
trained with two orientation/strain surrogates:

* ``rbsr_source``  -- legacy penalty of ALL folds vs the source mesh.
* ``rbsr_target``  -- baseline-relative penalty of only the model-induced *new*
  folds beyond the true i2i target (the decontaminated budget from Plan A).

All three methods pass through the *same* RBF control-point projector so that
accuracy/regularity differences cannot come from control-point error. Every
method is scored on held-out TEST identities with:

* ``roi_rmse``        -- L2 accuracy vs the ground-truth displacement.
* ``abs_flip_pct``    -- legacy ``normal_flip_pct`` (faces with det<0 vs source).
* ``new_flip_pct``    -- model-induced foldover: faces the method reverses that
  the TRUE target does NOT reverse (the corrected regularity metric).
* ``missed_flip_pct`` -- true target folds the method failed to reproduce.

Intrinsic target flip is reported for context. Identity-level bootstrap CIs are
computed by resampling source identities. Deterministic given ``--seed``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from rhinoform.data import load_rows, ridge_predict
from rhinoform.safe_fusion import new_and_missed_folds, signed_fold_indicator
from rhinoform.train import NeuralFieldCVAE, build_static_vertex_features, pair_conditions, predict_field
from rhinoform.train_rbsr_gate import (
    SpatialRiskGate,
    build_rbf_projection,
    predict_gate_batches,
    project_handles,
    resolve_device,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fold_mask(source_v: np.ndarray, edited_v: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-face orientation-reversal mask (det < 0): the legacy normal-flip set."""
    det, _ = signed_fold_indicator(source_v, edited_v, faces)
    return det < 0.0


def ordered_pairs(ids: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for a in ids:
        for b in ids:
            if a != b:
                pairs.append((a, b))
    return pairs


def identity_bootstrap_ci(
    per_pair: np.ndarray, source_ids: list[str], n_boot: int, seed: int, alpha: float = 0.05
) -> tuple[float, float, float]:
    per_pair = np.asarray(per_pair, dtype=np.float64)
    mean = float(per_pair.mean()) if len(per_pair) else float("nan")
    unique = sorted(set(source_ids))
    index_by_id: dict[str, list[int]] = {uid: [] for uid in unique}
    for i, sid in enumerate(source_ids):
        index_by_id[sid].append(i)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    n = len(unique)
    for b in range(n_boot):
        chosen = rng.integers(0, n, size=n)
        idx: list[int] = []
        for c in chosen:
            idx.extend(index_by_id[unique[c]])
        boot[b] = per_pair[idx].mean()
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return mean, lo, hi


def per_pair_flip_stats(
    source_abs: np.ndarray, pred_delta: np.ndarray, target_delta: np.ndarray, faces: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-pair flip percentages for a method's predicted displacement."""
    n = source_abs.shape[0]
    abs_pp = np.empty(n)
    new_pp = np.empty(n)
    missed_pp = np.empty(n)
    intrinsic_pp = np.empty(n)
    nfaces = faces.shape[0]
    for i in range(n):
        src = source_abs[i]
        pred_fold = fold_mask(src, src + pred_delta[i], faces)
        tgt_fold = fold_mask(src, src + target_delta[i], faces)
        new_fold, missed_fold = new_and_missed_folds(pred_fold, tgt_fold)
        abs_pp[i] = 100.0 * pred_fold.mean()
        intrinsic_pp[i] = 100.0 * tgt_fold.mean()
        new_pp[i] = 100.0 * new_fold.mean()
        missed_pp[i] = 100.0 * missed_fold.mean()
    return {"abs": abs_pp, "new": new_pp, "missed": missed_pp, "intrinsic": intrinsic_pp}


def per_pair_rmse(pred_delta: np.ndarray, target_delta: np.ndarray) -> np.ndarray:
    squared = (pred_delta - target_delta) ** 2
    return np.sqrt(np.mean(np.sum(squared, axis=2), axis=1))


def summarise(per_pair: np.ndarray, source_ids: list[str], n_boot: int, seed: int) -> dict:
    mean, lo, hi = identity_bootstrap_ci(per_pair, source_ids, n_boot, seed)
    return {
        "mean": mean,
        "ci95": [lo, hi],
        "p95": float(np.percentile(per_pair, 95)),
        "max": float(per_pair.max()),
    }


def load_gate(path: Path, vertex_dim: int, cond_dim: int, selected_alpha: float, device) -> tuple[SpatialRiskGate, np.ndarray | None]:
    pkg = torch.load(path, map_location="cpu", weights_only=False)
    if int(pkg["gate_vertex_dim"]) != vertex_dim or int(pkg["gate_cond_dim"]) != cond_dim:
        raise ValueError(f"Gate dims in {path} do not match the base model")
    gate = SpatialRiskGate(vertex_dim, cond_dim, int(pkg["args"]["hidden"]), selected_alpha).to(device)
    gate.load_state_dict(pkg["gate_state_dict"])
    gate.eval()
    basis = pkg.get("projection_basis")
    basis_np = None if basis is None else np.asarray(basis, dtype=np.float32)
    return gate, basis_np


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan A clean ridge-vs-RBSR evaluation on test identities.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base-model-package", required=True)
    parser.add_argument("--gate-target-package", required=True)
    parser.add_argument("--gate-source-package", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=1000, help="Deterministic subsample of test pairs; 0 uses all.")
    parser.add_argument(
        "--project",
        choices=["gate_only", "all"],
        default="gate_only",
        help=(
            "gate_only: ridge/cvae/hybrid scored in native form (comparable to frozen baselines), "
            "RBSR keeps its built-in control-point projector. "
            "all: every method passes through the same projector (A1-style control-error isolation)."
        ),
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    rng = np.random.default_rng(args.seed)
    print(f"[eval] device={device} seed={args.seed}", flush=True)

    _, by_id = load_rows(Path(args.repo))
    base = torch.load(args.base_model_package, map_location="cpu", weights_only=False)
    if not base.get("feature_template_sha256"):
        raise RuntimeError("Base package predates the train-only feature template fix; retrain first")
    train_ids = [str(v) for v in base["train_ids"]]
    test_ids = [str(v) for v in base["test_ids"]]
    template = by_id[train_ids[0]]
    faces = np.asarray(template["faces"], dtype=np.int64)
    landmarks_np = np.asarray(template["landmarks"], dtype=np.int64)

    vertex_features_np, static = build_static_vertex_features(
        by_id, train_ids, use_subunit_features=bool(base["use_subunit_features"])
    )
    if static["template_sha256"] != base["feature_template_sha256"]:
        raise RuntimeError("Recomputed template does not match the base package")
    center_np = np.asarray(static["center"], dtype=np.float32).reshape(1, 1, 3)
    scale = float(np.asarray(static["scale"]).reshape(-1)[0])

    test_pairs = ordered_pairs(test_ids)
    if args.max_pairs and len(test_pairs) > args.max_pairs:
        sel = rng.choice(len(test_pairs), size=args.max_pairs, replace=False)
        sel.sort()
        test_pairs = [test_pairs[i] for i in sel]
    source_ids = [a for a, _ in test_pairs]
    print(f"[eval] evaluating {len(test_pairs)} test pairs over {len(test_ids)} identities", flush=True)

    cond, source, delta, controls, *_ = pair_conditions(
        by_id, test_pairs, base["source_pca"], base["cond_mean"], base["cond_std"]
    )
    source = np.asarray(source, dtype=np.float32)
    delta = np.asarray(delta, dtype=np.float32)

    ridge_flat = ridge_predict(cond, base["ridge_cond"]).astype(np.float32)

    print("[eval] caching frozen CVAE predictions", flush=True)
    cvae = NeuralFieldCVAE(
        vertex_features_np.shape[1], cond.shape[1], int(np.asarray(base["obs_train_mean"]).shape[1]),
        latent_dim=int(base["args"]["latent_dim"]), hidden=int(base["args"]["hidden"]),
    )
    cvae.load_state_dict(base["cvae_state_dict"])
    cvae.to(device).eval()
    for p in cvae.parameters():
        p.requires_grad_(False)
    cvae_flat = predict_field(cvae, vertex_features_np, cond, is_cvae=True).astype(np.float32)
    cvae.to("cpu")
    del cvae
    if device.type == "cuda":
        torch.cuda.empty_cache()

    vertex_features = torch.as_tensor(vertex_features_np, dtype=torch.float32, device=device)
    center = torch.as_tensor(center_np, dtype=torch.float32, device=device)
    landmarks = torch.as_tensor(landmarks_np, dtype=torch.long, device=device)
    basis_template = torch.as_tensor(
        build_rbf_projection(static["template_vertices"], landmarks_np), dtype=torch.float32, device=device
    )

    method_deltas: dict[str, np.ndarray] = {}

    def to_deltas(flat: np.ndarray, force_project: bool) -> np.ndarray:
        """Reshape a flat displacement field, optionally through the RBF projector."""
        out: list[np.ndarray] = []
        for s in range(0, len(test_pairs), args.batch_size):
            e = min(len(test_pairs), s + args.batch_size)
            anchor = torch.as_tensor(flat[s:e], dtype=torch.float32, device=device).reshape(e - s, -1, 3)
            if force_project:
                ctrl = torch.as_tensor(controls[s:e], dtype=torch.float32, device=device).reshape(e - s, -1, 3)
                anchor = project_handles(anchor, ctrl, landmarks, basis_template)
            out.append(anchor.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    project_linear = args.project == "all"
    selected_alpha = float(base["selected_alpha"]) if base.get("selected_alpha") is not None else None
    method_deltas["ridge"] = to_deltas(ridge_flat, project_linear)
    method_deltas["cvae"] = to_deltas(cvae_flat, project_linear)
    if selected_alpha is not None:
        hybrid_flat = (1.0 - selected_alpha) * ridge_flat + selected_alpha * cvae_flat
        method_deltas["hybrid"] = to_deltas(hybrid_flat.astype(np.float32), project_linear)

    gate_specs = [("rbsr_target", args.gate_target_package)]
    if args.gate_source_package:
        gate_specs.append(("rbsr_source", args.gate_source_package))
    for name, path in gate_specs:
        gate, basis_np = load_gate(Path(path), vertex_features_np.shape[1], cond.shape[1], float(base["selected_alpha"]), device)
        basis_t = basis_template if basis_np is None else torch.as_tensor(basis_np, dtype=torch.float32, device=device)
        pred_flat, _ = predict_gate_batches(
            gate, vertex_features, cond, source, ridge_flat, cvae_flat,
            center, scale, landmarks, controls, basis_t, args.batch_size,
        )
        method_deltas[name] = pred_flat.reshape(len(test_pairs), -1, 3).astype(np.float32)
        del gate
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Intrinsic target flip (method-independent).
    intrinsic = np.empty(len(test_pairs))
    for i in range(len(test_pairs)):
        intrinsic[i] = 100.0 * fold_mask(source[i], source[i] + delta[i], faces).mean()

    results: dict[str, dict] = {}
    for name, pred in method_deltas.items():
        print(f"[eval] scoring {name}", flush=True)
        rmse_pp = per_pair_rmse(pred, delta)
        flips = per_pair_flip_stats(source, pred, delta, faces)
        results[name] = {
            "roi_rmse": summarise(rmse_pp, source_ids, args.n_boot, args.seed),
            "abs_flip_pct": summarise(flips["abs"], source_ids, args.n_boot, args.seed),
            "new_flip_pct": summarise(flips["new"], source_ids, args.n_boot, args.seed),
            "missed_flip_pct": summarise(flips["missed"], source_ids, args.n_boot, args.seed),
        }

    # Paired deltas vs ridge (positive => RBSR better / lower).
    deltas_vs_ridge: dict[str, dict] = {}
    ridge_rmse = per_pair_rmse(method_deltas["ridge"], delta)
    ridge_new = per_pair_flip_stats(source, method_deltas["ridge"], delta, faces)["new"]
    for name in method_deltas:
        if name == "ridge":
            continue
        rmse_gain = ridge_rmse - per_pair_rmse(method_deltas[name], delta)  # >0 => RBSR more accurate
        new_gain = ridge_new - per_pair_flip_stats(source, method_deltas[name], delta, faces)["new"]  # >0 => fewer new flips
        deltas_vs_ridge[name] = {
            "rmse_gain_ridge_minus_method": summarise(rmse_gain, source_ids, args.n_boot, args.seed),
            "new_flip_gain_ridge_minus_method": summarise(new_gain, source_ids, args.n_boot, args.seed),
        }

    out = {
        "provenance": {
            "seed": args.seed,
            "project": args.project,
            "selected_alpha": selected_alpha,
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": str(device),
            "base_model_package": str(args.base_model_package),
            "base_model_package_sha256": sha256_file(Path(args.base_model_package)),
            "gate_target_package": str(args.gate_target_package),
            "gate_target_package_sha256": sha256_file(Path(args.gate_target_package)),
            "gate_source_package": str(args.gate_source_package) or None,
            "gate_source_package_sha256": sha256_file(Path(args.gate_source_package)) if args.gate_source_package else None,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "metric_note": (
            "abs_flip_pct = legacy normal-flip (det<0 vs source). "
            "new_flip_pct = model-induced foldover beyond the true target (corrected regularity metric). "
            "missed_flip_pct = true target folds not reproduced. rmse on ROI displacement. "
            "All methods share the same RBF control-point projector. CIs are identity-level bootstrap."
        ),
        "n_pairs": len(test_pairs),
        "n_test_ids": len(test_ids),
        "intrinsic_target_flip_pct": {
            "mean": float(intrinsic.mean()),
            "p95": float(np.percentile(intrinsic, 95)),
            "max": float(intrinsic.max()),
        },
        "methods": results,
        "deltas_vs_ridge": deltas_vs_ridge,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[eval] wrote {out_path}", flush=True)
    print(json.dumps({k: {m: v[m]["mean"] for m in ("roi_rmse", "abs_flip_pct", "new_flip_pct")}
                      for k, v in results.items()}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
