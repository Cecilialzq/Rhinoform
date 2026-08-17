from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .cvae import fit_truncated_pca, flatten_vertices, pca_transform
from .data import (
    array_sha256,
    assert_identity_disjoint,
    assert_pairs_within,
    edge_index,
    evaluate_method,
    fixed_ordered_pairs,
    load_rows,
    ordered_pairs,
    pair_arrays,
    ridge_fit,
    ridge_predict,
    split_ids,
    training_mean_geometry,
)
from .repro import (
    artifact_metadata,
    atomic_savez_compressed,
    atomic_torch_save,
    atomic_write_csv,
    atomic_write_json,
    capture_rng_state,
    restore_rng_state,
    seed_worker_factory,
    set_global_seed,
    sha256_file,
    sha256_json,
    validate_torch_artifact,
)
from .sampling import read_ids, read_pairs
from .stats import write_csv as write_metric_csv
from .strict_protocol_patch import strict_metric_rows as metric_rows_for_method


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    return device


def face_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float64)
    tri = faces.astype(np.int64)
    fn = np.cross(vertices[tri[:, 1]] - vertices[tri[:, 0]], vertices[tri[:, 2]] - vertices[tri[:, 0]])
    for j in range(3):
        np.add.at(normals, tri[:, j], fn)
    denom = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(denom, 1e-12)


def subunit_one_hot(row: dict) -> np.ndarray:
    n = row["vertices"].shape[0]
    out = np.zeros((n, 5), dtype=np.float64)
    names = ["root", "dorsum", "tip", "alar_left", "alar_right"]
    for i, name in enumerate(names):
        idx = row["subunits"][name]
        out[idx] = 0.0
        out[idx, i] = 1.0
    return out


def build_static_vertex_features(by_id: dict[str, dict], train_ids: list[str], use_subunit_features: bool = True) -> tuple[np.ndarray, dict[str, object]]:
    base = training_mean_geometry(by_id, train_ids)
    template = by_id[str(train_ids[0])]
    faces = template["faces"]
    landmarks = template["landmarks"]
    all_train = np.concatenate([by_id[sid]["vertices"] for sid in train_ids], axis=0)
    center = all_train.mean(axis=0, keepdims=True)
    scale = float(np.linalg.norm(all_train.max(axis=0) - all_train.min(axis=0)))
    scale = max(scale, 1e-6)
    coords = (base - center) / scale
    normals = face_vertex_normals(base, faces)
    lm = base[landmarks]
    dist = np.linalg.norm(base[:, None, :] - lm[None, :, :], axis=2)
    dist = dist / np.maximum(np.percentile(dist, 95, axis=0, keepdims=True), 1e-6)
    parts = [coords, normals]
    if use_subunit_features:
        parts.append(subunit_one_hot(template))
    parts.append(dist)
    feat = np.concatenate(parts, axis=1)
    return feat.astype(np.float32), {
        "center": center.astype(np.float32),
        "scale": np.asarray([scale], dtype=np.float32),
        "template_vertices": base.astype(np.float32),
        "template_sha256": array_sha256(base),
        "template_schema": "mean_neutral_roi_of_training_identities_v1",
    }


def assert_feature_template_package(package: dict, static: dict[str, object], label: str) -> None:
    expected = package.get("feature_template_sha256")
    if not expected:
        raise RuntimeError(
            f"{label} predates the train-only feature-template fix and stores no "
            "feature_template_sha256; refusing to evaluate a legacy package as clean evidence"
        )
    if static["template_sha256"] != expected:
        raise RuntimeError(
            f"{label} feature-template hash mismatch: package={expected}, "
            f"recomputed={static['template_sha256']}"
        )


def pair_conditions(
    by_id: dict[str, dict],
    pairs: list[tuple[str, str]],
    source_pca: dict[str, np.ndarray],
    cond_mean: np.ndarray | None = None,
    cond_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    landmarks = next(iter(by_id.values()))["landmarks"]
    ctrl = []
    source_code_in = []
    source_vertices = []
    target_delta = []
    for src_id, tgt_id in pairs:
        src = by_id[src_id]["vertices"]
        tgt = by_id[tgt_id]["vertices"]
        source_vertices.append(src)
        delta = tgt - src
        target_delta.append(delta)
        ctrl.append(delta[landmarks].reshape(-1))
        source_code_in.append(src.reshape(-1))
    source_code = pca_transform(np.stack(source_code_in, axis=0), source_pca)
    cond = np.concatenate([np.stack(ctrl, axis=0), source_code], axis=1)
    if cond_mean is None:
        cond_mean = cond.mean(axis=0, keepdims=True)
        cond_std = np.maximum(cond.std(axis=0, keepdims=True), 1e-6)
    cond = (cond - cond_mean) / cond_std
    return (
        cond.astype(np.float32),
        np.stack(source_vertices, axis=0).astype(np.float32),
        np.stack(target_delta, axis=0).astype(np.float32),
        np.stack(ctrl, axis=0).astype(np.float32),
        source_code.astype(np.float32),
        cond_mean,
        cond_std,
    )


class NeuralFieldDeformer(nn.Module):
    def __init__(self, vertex_dim: int, cond_dim: int, hidden: int = 128) -> None:
        super().__init__()
        in_dim = vertex_dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 3),
        )

    def forward(self, vertex_feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b = cond.shape[0]
        v = vertex_feat.shape[0]
        vf = vertex_feat.unsqueeze(0).expand(b, v, -1)
        cf = cond.unsqueeze(1).expand(b, v, -1)
        return self.net(torch.cat([vf, cf], dim=-1))


class NeuralFieldCVAE(nn.Module):
    def __init__(self, vertex_dim: int, cond_dim: int, obs_dim: int, latent_dim: int = 8, hidden: int = 128) -> None:
        super().__init__()
        self.prior = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        self.pmu = nn.Linear(hidden, latent_dim)
        self.plogvar = nn.Linear(hidden, latent_dim)
        self.enc = nn.Sequential(nn.Linear(cond_dim + obs_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        self.qmu = nn.Linear(hidden, latent_dim)
        self.qlogvar = nn.Linear(hidden, latent_dim)
        self.field = NeuralFieldDeformer(vertex_dim, cond_dim + latent_dim, hidden=hidden)

    def forward(self, vertex_feat: torch.Tensor, cond: torch.Tensor, obs: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hp = self.prior(cond)
        pmu = self.pmu(hp)
        plogvar = self.plogvar(hp).clamp(-8.0, 8.0)
        if self.training and obs is not None:
            hq = self.enc(torch.cat([cond, obs], dim=1))
            qmu = self.qmu(hq)
            qlogvar = self.qlogvar(hq).clamp(-8.0, 8.0)
            z = qmu + torch.randn_like(qmu) * torch.exp(0.5 * qlogvar)
        else:
            qmu = pmu
            qlogvar = plogvar
            z = pmu
        pred = self.field(vertex_feat, torch.cat([cond, z], dim=1))
        return {"pred": pred, "pmu": pmu, "plogvar": plogvar, "qmu": qmu, "qlogvar": qlogvar}


def kl_normal(qmu: torch.Tensor, qlogvar: torch.Tensor, pmu: torch.Tensor, plogvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(
        plogvar
        - qlogvar
        + (torch.exp(qlogvar) + (qmu - pmu).pow(2)) / torch.exp(plogvar).clamp_min(1e-8)
        - 1.0
    )


def loss_terms(pred: torch.Tensor, target: torch.Tensor, landmarks: torch.Tensor, edge0: torch.Tensor, edge1: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    dense = F.l1_loss(pred, target)
    ctrl = F.l1_loss(pred[:, landmarks], target[:, landmarks])
    edge = F.l1_loss(pred[:, edge0] - pred[:, edge1], target[:, edge0] - target[:, edge1])
    total = dense + 5.0 * ctrl + 0.1 * edge
    return total, {"dense": float(dense.detach()), "ctrl": float(ctrl.detach()), "edge": float(edge.detach())}


def loss_terms_weighted(
    pred: torch.Tensor,
    target: torch.Tensor,
    source: torch.Tensor | None,
    landmarks: torch.Tensor,
    edge0: torch.Tensor,
    edge1: torch.Tensor,
    dense_weight: float,
    ctrl_weight: float,
    edge_weight: float,
    lap_weight: float,
    strain_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    dense = F.l1_loss(pred, target)
    ctrl = F.l1_loss(pred[:, landmarks], target[:, landmarks])
    pred_edge = pred[:, edge0] - pred[:, edge1]
    target_edge = target[:, edge0] - target[:, edge1]
    edge = F.l1_loss(pred_edge, target_edge)
    if float(lap_weight) > 0.0:
        # Edge-neighborhood displacement smoothness. This approximates Laplacian regularity
        # without building a dense vertex Laplacian.
        lap = torch.mean(torch.linalg.norm(pred_edge, dim=-1))
    else:
        lap = edge * 0.0
    if source is not None and float(strain_weight) > 0.0:
        src_edge_vec = source[:, edge0] - source[:, edge1]
        edited_edge_vec = src_edge_vec + pred_edge
        src_len = torch.linalg.norm(src_edge_vec, dim=-1).clamp_min(1e-8)
        edited_len = torch.linalg.norm(edited_edge_vec, dim=-1)
        strain = torch.mean(torch.abs(edited_len - src_len) / src_len)
    else:
        strain = edge * 0.0
    total = (
        float(dense_weight) * dense
        + float(ctrl_weight) * ctrl
        + float(edge_weight) * edge
        + float(lap_weight) * lap
        + float(strain_weight) * strain
    )
    return total, {
        "dense": float(dense.detach()),
        "ctrl": float(ctrl.detach()),
        "edge": float(edge.detach()),
        "lap": float(lap.detach()),
        "strain": float(strain.detach()),
    }


def val_rmse(model: nn.Module, vertex_feat: torch.Tensor, cond: torch.Tensor, target: np.ndarray, is_cvae: bool) -> float:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(cond), 32):
            c = cond[start : start + 32]
            if is_cvae:
                p = model(vertex_feat, c, None)["pred"]
            else:
                p = model(vertex_feat, c)
            preds.append(p.cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    diff = pred - target
    return float(np.mean(np.sqrt(np.mean(np.sum(diff * diff, axis=2), axis=1))))


def train_field(
    model: nn.Module,
    vertex_feat: np.ndarray,
    cond_train: np.ndarray,
    source_train: np.ndarray,
    delta_train: np.ndarray,
    obs_train: np.ndarray,
    cond_val: np.ndarray,
    delta_val: np.ndarray,
    landmarks_np: np.ndarray,
    faces: np.ndarray,
    args: argparse.Namespace,
    is_cvae: bool,
    checkpoint_tag: str,
) -> tuple[nn.Module, dict[str, object]]:
    device = torch.device(args.device_resolved)
    model.to(device)
    vf = torch.as_tensor(vertex_feat, dtype=torch.float32, device=device)
    ct = torch.as_tensor(cond_train, dtype=torch.float32, device=device)
    st = torch.as_tensor(source_train, dtype=torch.float32, device=device)
    dt = torch.as_tensor(delta_train, dtype=torch.float32, device=device)
    ot = torch.as_tensor(obs_train, dtype=torch.float32, device=device)
    cv = torch.as_tensor(cond_val, dtype=torch.float32, device=device)
    landmarks = torch.as_tensor(landmarks_np, dtype=torch.long, device=device)
    edges = edge_index(faces)
    if args.train_edge_sample and int(args.train_edge_sample) > 0 and int(args.train_edge_sample) < len(edges):
        edge_rng = np.random.default_rng(args.seed + 991)
        edges = edges[edge_rng.choice(len(edges), size=int(args.train_edge_sample), replace=False)]
    edge0 = torch.as_tensor(edges[:, 0], dtype=torch.long, device=device)
    edge1 = torch.as_tensor(edges[:, 1], dtype=torch.long, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = float("inf")
    best_epoch = 0
    best_state = None
    patience = 0
    start_epoch = 1
    checkpoint_records: list[dict[str, str]] = []
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    signature_args = {
        key: value for key, value in vars(args).items()
        if key not in {"resume_checkpoint", "device_resolved"}
    }
    resume_signature = sha256_json({
        "tag": checkpoint_tag,
        "args": signature_args,
        "vertex_feature_shape": list(vertex_feat.shape),
        "condition_shape": list(cond_train.shape),
        "target_shape": list(delta_train.shape),
    })
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        if not validate_torch_artifact(
            resume_path,
            required_keys=("tag", "epoch", "model_state", "optimizer_state", "resume_signature", "rng_state"),
        ):
            raise RuntimeError(f"Resume checkpoint or SHA-256 sidecar is invalid: {resume_path}")
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        if ckpt.get("tag") != checkpoint_tag:
            raise ValueError(f"Checkpoint tag {ckpt.get('tag')!r} does not match {checkpoint_tag!r}")
        if ckpt.get("resume_signature") != resume_signature:
            raise ValueError("Base resume checkpoint configuration/data shapes do not match this run")
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        best = float(ckpt.get("best", best))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        best_state = ckpt.get("best_state", best_state)
        patience = int(ckpt.get("patience", patience))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        restore_rng_state(ckpt.get("rng_state"))
        print(
            f"{checkpoint_tag} resume from Drive epoch={start_epoch - 1}/{args.epochs} "
            f"checkpoint={resume_path.name}",
            flush=True,
        )
    for epoch in range(start_epoch, args.epochs + 1):
        generator = torch.Generator()
        generator.manual_seed(int(args.seed) + int(epoch))
        loader = DataLoader(
            TensorDataset(torch.arange(len(cond_train), dtype=torch.long)),
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            worker_init_fn=seed_worker_factory(int(args.seed)),
            num_workers=int(args.num_workers),
        )
        beta = min(args.beta, args.beta * epoch / max(1, args.kl_warmup))
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        total_batches = len(loader)
        for batch_index, (idx_cpu,) in enumerate(loader, start=1):
            idx = idx_cpu.to(device=device, dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            if is_cvae:
                out = model(vf, ct[idx], ot[idx])
                pred = out["pred"]
                loss, _ = loss_terms_weighted(
                    pred,
                    dt[idx],
                    st[idx],
                    landmarks,
                    edge0,
                    edge1,
                    args.dense_weight,
                    args.ctrl_weight,
                    args.edge_weight,
                    args.lap_weight,
                    args.strain_weight,
                )
                loss = loss + beta * kl_normal(out["qmu"], out["qlogvar"], out["pmu"], out["plogvar"])
            else:
                pred = model(vf, ct[idx])
                loss, _ = loss_terms_weighted(
                    pred,
                    dt[idx],
                    st[idx],
                    landmarks,
                    edge0,
                    edge1,
                    args.dense_weight,
                    args.ctrl_weight,
                    args.edge_weight,
                    args.lap_weight,
                    args.strain_weight,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += float(loss.detach().cpu())
            epoch_batches += 1
            if batch_index % 10 == 0 or batch_index == total_batches:
                print(
                    f"{checkpoint_tag} live epoch={epoch}/{args.epochs} "
                    f"batch={batch_index}/{total_batches}",
                    flush=True,
                )
        validation_message = "validation=not_due"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            score = val_rmse(model, vf, cv, delta_val, is_cvae)
            tag = "field_cvae" if is_cvae else "field"
            if score < best:
                best = score
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += args.eval_every
            validation_message = f"val_roi_rmse={score:.4f} best={best:.4f}"
        print(
            f"{checkpoint_tag} epoch={epoch}/{args.epochs} "
            f"train_loss={epoch_loss / max(epoch_batches, 1):.6f} {validation_message}",
            flush=True,
        )
        if checkpoint_dir is not None and args.checkpoint_every > 0 and (
            epoch % args.checkpoint_every == 0 or epoch == args.epochs
        ):
            print(f"{checkpoint_tag} checkpoint write to Drive start epoch={epoch}", flush=True)
            rec = save_checkpoint(
                checkpoint_dir / f"{checkpoint_tag}_last.pt",
                {
                    "tag": checkpoint_tag,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "best": best,
                    "best_epoch": best_epoch,
                    "best_state": best_state,
                    "patience": patience,
                    "args": vars(args),
                    "resume_signature": resume_signature,
                    "rng_state": capture_rng_state(),
                },
            )
            checkpoint_records = [rec]
            print(f"{checkpoint_tag} checkpoint persisted to Drive epoch={epoch}", flush=True)
        if patience >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    if checkpoint_dir is not None:
        best_payload = {
            "tag": checkpoint_tag,
            "epoch": best_epoch,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "best": best,
            "best_epoch": best_epoch,
            "best_state": best_state,
            "patience": patience,
            "args": vars(args),
            "resume_signature": resume_signature,
            "rng_state": capture_rng_state(),
        }
        print(f"{checkpoint_tag} best checkpoint write to Drive start epoch={best_epoch}", flush=True)
        rec = save_checkpoint(
            checkpoint_dir / f"{checkpoint_tag}_best.pt",
            best_payload,
        )
        checkpoint_records.append(rec)
        print(f"{checkpoint_tag} best checkpoint persisted to Drive epoch={best_epoch}", flush=True)
    return model, {"best_val_roi_rmse": best, "best_epoch": float(best_epoch), "checkpoints": checkpoint_records}


def predict_field(
    model: nn.Module,
    vertex_feat: np.ndarray,
    cond: np.ndarray,
    is_cvae: bool,
    progress_label: str | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = next(model.parameters()).device
    vf = torch.as_tensor(vertex_feat, dtype=torch.float32, device=device)
    c = torch.as_tensor(cond, dtype=torch.float32, device=device)
    preds = []
    total_batches = (len(c) + batch_size - 1) // batch_size
    if progress_label:
        print(
            f"{progress_label} start pairs={len(c)} batches={total_batches}",
            flush=True,
        )
    model.eval()
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, len(c), batch_size), start=1):
            if is_cvae:
                p = model(vf, c[start : start + batch_size], None)["pred"]
            else:
                p = model(vf, c[start : start + batch_size])
            preds.append(p.cpu().numpy().reshape(p.shape[0], -1))
            if progress_label and (batch_index % 10 == 0 or batch_index == total_batches):
                completed = min(start + batch_size, len(c))
                print(
                    f"{progress_label} live pairs={completed}/{len(c)} "
                    f"batches={batch_index}/{total_batches}",
                    flush=True,
                )
    if progress_label:
        print(f"{progress_label} complete pairs={len(c)}", flush=True)
    return np.concatenate(preds, axis=0)


def mean_pair_rmse(pred_flat: np.ndarray, target_delta: np.ndarray) -> float:
    pred = pred_flat.reshape(target_delta.shape)
    diff = pred - target_delta
    return float(np.mean(np.sqrt(np.mean(np.sum(diff * diff, axis=2), axis=1))))


def select_alpha_on_validation(
    base_val: np.ndarray,
    neural_val: np.ndarray,
    delta_val: np.ndarray,
    alpha_grid: list[float],
) -> dict[str, float]:
    rows = []
    for alpha in alpha_grid:
        pred = (1.0 - alpha) * base_val + alpha * neural_val
        rows.append({"alpha": float(alpha), "val_roi_rmse": mean_pair_rmse(pred, delta_val)})
    best = min(rows, key=lambda r: r["val_roi_rmse"])
    return {"best_alpha": float(best["alpha"]), "best_val_roi_rmse": float(best["val_roi_rmse"]), "grid": rows}


def write_per_pair_metrics(out_dir: Path, by_id: dict[str, dict], pairs: list[tuple[str, str]], method_preds: dict[str, np.ndarray]) -> list[dict[str, str]]:
    pair_dir = out_dir / "pair_metrics"
    records: list[dict[str, str]] = []
    for name, pred in method_preds.items():
        rows = metric_rows_for_method(by_id, pairs, pred)
        path = pair_dir / f"identity_bootstrap_pair_metrics_{name}.csv"
        write_metric_csv(path, rows)
        records.append({"method": name, "path": str(path), "n_pairs": str(len(rows)), "sha256": sha256_file(path)})
    return records


def mean_metrics(rows: list[dict[str, float | str]]) -> dict[str, float]:
    keys = [
        "roi_rmse", "landmark_rmse", "dorsum_rmse", "tip_rmse",
        "edge_strain_p95", "normal_flip_pct", "abs_flip_pct",
        "missed_flip_pct", "target_flip_pct",
    ]
    return {k: float(np.mean([float(r[k]) for r in rows])) for k in keys}


def strict_validation_metrics_with_progress(
    by_id: dict[str, dict],
    val_pairs: list[tuple[str, str]],
    prediction: np.ndarray,
    label: str,
    chunk_pairs: int = 320,
) -> dict[str, float]:
    rows: list[dict[str, float | str]] = []
    print(
        f"BASE VALIDATION strict score start candidate={label} pairs={len(val_pairs)}",
        flush=True,
    )
    for start in range(0, len(val_pairs), chunk_pairs):
        stop = min(start + chunk_pairs, len(val_pairs))
        rows.extend(metric_rows_for_method(by_id, val_pairs[start:stop], prediction[start:stop]))
        print(
            f"BASE VALIDATION strict score live candidate={label} pairs={stop}/{len(val_pairs)}",
            flush=True,
        )
    metrics = mean_metrics(rows)
    print(
        f"BASE VALIDATION strict score complete candidate={label} "
        f"roi_rmse={metrics['roi_rmse']:.6f} "
        f"new_flip={metrics['normal_flip_pct']:.6f} "
        f"abs_flip={metrics['abs_flip_pct']:.6f} "
        f"strain={metrics['edge_strain_p95']:.6f}",
        flush=True,
    )
    return metrics


def write_operating_point_candidates(
    out_dir: Path,
    by_id: dict[str, dict],
    val_pairs: list[tuple[str, str]],
    ridge_val: np.ndarray,
    cvae_val: np.ndarray,
    alpha_grid: list[float],
    model_signature: str,
) -> Path:
    candidates: list[tuple[str, str, float | str]] = [
        ("ridge_anchor", "ridge_anchor", ""),
        ("cvae_only", "cvae_only", ""),
        *[(f"hybrid_alpha_{alpha:g}", "hybrid", float(alpha)) for alpha in alpha_grid],
    ]
    progress_path = out_dir / "validation_operating_point_progress.json"
    progress_signature = sha256_json({
        "schema": "base_validation_operating_points_v1",
        "model_signature": model_signature,
        "val_pairs": val_pairs,
        "candidate_labels": [label for label, _, _ in candidates],
        "ridge_shape": list(ridge_val.shape),
        "cvae_shape": list(cvae_val.shape),
    })
    progress: dict[str, object] = {
        "status": "IN_PROGRESS",
        "signature": progress_signature,
        "completed_rows": [],
    }
    if progress_path.is_file():
        try:
            saved = json.loads(progress_path.read_text(encoding="utf-8"))
            if saved.get("signature") == progress_signature:
                progress = saved
        except (OSError, json.JSONDecodeError):
            pass
    completed = {
        str(row["label"]): row for row in progress.get("completed_rows", [])
    }
    for label, method, alpha in candidates:
        if label in completed:
            print(f"BASE VALIDATION resume from Drive candidate={label}", flush=True)
            continue
        if label == "ridge_anchor":
            prediction = ridge_val
        elif label == "cvae_only":
            prediction = cvae_val
        else:
            alpha_value = float(alpha)
            prediction = (1.0 - alpha_value) * ridge_val + alpha_value * cvae_val
        metrics = strict_validation_metrics_with_progress(
            by_id, val_pairs, prediction, label, chunk_pairs=320
        )
        completed[label] = {"label": label, "method": method, "alpha": alpha, **metrics}
        progress["completed_rows"] = [
            completed[declared_label]
            for declared_label, _, _ in candidates
            if declared_label in completed
        ]
        atomic_write_json(progress_path, progress)
        print(f"BASE VALIDATION persisted to Drive candidate={label}", flush=True)
    rows = [completed[label] for label, _, _ in candidates]
    path = out_dir / "validation_operating_point_candidates.csv"
    atomic_write_csv(path, rows)
    progress["status"] = "COMPLETE"
    progress["final_csv_sha256"] = sha256_file(path)
    atomic_write_json(progress_path, progress)
    print(f"BASE VALIDATION all operating points persisted to Drive: {path}", flush=True)
    return path


def ids_from_args(by_id: dict[str, dict], split_name: str, ids_json: str) -> list[str]:
    if ids_json:
        return read_ids(Path(ids_json))
    return split_ids(by_id, split_name)


def pairs_from_args(
    ids: list[str],
    pairs_json: str,
    pair_budget: int,
    seed: int,
    coverage_balanced: bool,
) -> list[tuple[str, str]]:
    if pairs_json:
        return read_pairs(Path(pairs_json))
    return fixed_ordered_pairs(ids, pair_budget if pair_budget > 0 else None, seed, coverage_balanced)


def save_checkpoint(path: Path, payload: dict) -> dict[str, str]:
    atomic_torch_save(
        path,
        payload,
        required_keys=(
            "tag", "epoch", "model_state", "optimizer_state", "resume_signature", "rng_state",
        ),
    )
    return {"path": str(path), "sha256": sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="../data")
    parser.add_argument("--out", default="../results")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--source-pca-dim", type=int, default=16)
    parser.add_argument("--delta-pca-dim", type=int, default=16)
    parser.add_argument("--ridge-lambda", type=float, default=100.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--ctrl-weight", type=float, default=5.0)
    parser.add_argument("--edge-weight", type=float, default=0.1)
    parser.add_argument("--lap-weight", type=float, default=0.0)
    parser.add_argument("--strain-weight", type=float, default=0.0)
    parser.add_argument("--train-edge-sample", type=int, default=0)
    parser.add_argument("--model-kind", choices=["both", "field", "cvae"], default="both")
    parser.add_argument("--alpha-grid", default="0,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.75,1.0")
    parser.add_argument("--kl-warmup", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-subunit-features", choices=["true", "false"], default="true")
    parser.add_argument("--no-use-subunit-features", dest="use_subunit_features", action="store_const", const="false")
    parser.add_argument("--train-split", default="clean-prior train")
    parser.add_argument("--val-split", default="clean-prior validation")
    parser.add_argument("--test-split", default="main test")
    parser.add_argument("--train-ids-json", default="")
    parser.add_argument("--val-ids-json", default="")
    parser.add_argument("--test-ids-json", default="")
    parser.add_argument("--train-pairs-json", default="")
    parser.add_argument("--val-pairs-json", default="")
    parser.add_argument("--test-pairs-json", default="")
    parser.add_argument("--train-pair-budget", type=int, default=0)
    parser.add_argument("--secondary-pair-budget", type=int, default=0)
    parser.add_argument("--coverage-balanced-pairs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pair-seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--metadata-out", default="")
    parser.add_argument("--split-manifest", default="")
    parser.add_argument("--save-dense-predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--defer-test-evaluation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Train/select using train+validation identities only and package the model without "
            "loading test meshes. Use scripts/evaluation/rbsr_gate.py for the one-shot test after "
            "the gate operating point has been frozen."
        ),
    )
    args = parser.parse_args()

    set_global_seed(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    args.device_resolved = str(device)
    print(f"torch device: {device}", flush=True)
    repo = Path(args.repo)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = None
    if args.split_manifest:
        split_manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    if args.defer_test_evaluation:
        if split_manifest is None:
            raise ValueError("--defer-test-evaluation requires --split-manifest")
        train_ids = (
            read_ids(Path(args.train_ids_json))
            if args.train_ids_json
            else [str(value) for value in split_manifest["train_pool_ids"]]
        )
        val_ids = (
            read_ids(Path(args.val_ids_json))
            if args.val_ids_json
            else [str(value) for value in split_manifest["val_ids"]]
        )
        test_ids = (
            read_ids(Path(args.test_ids_json))
            if args.test_ids_json
            else [str(value) for value in split_manifest["test_ids"]]
        )
        _, by_id = load_rows(repo, allowed_ids=set(train_ids) | set(val_ids))
    else:
        _, by_id = load_rows(repo)
        train_ids = ids_from_args(by_id, args.train_split, args.train_ids_json)
        val_ids = ids_from_args(by_id, args.val_split, args.val_ids_json)
        test_ids = ids_from_args(by_id, args.test_split, args.test_ids_json)
    pair_seed = args.pair_seed if args.pair_seed else args.seed
    pair_budget = args.secondary_pair_budget if args.secondary_pair_budget > 0 else args.train_pair_budget
    train_pairs = pairs_from_args(train_ids, args.train_pairs_json, pair_budget, pair_seed, args.coverage_balanced_pairs)
    if args.train_pairs_json and not args.train_ids_json:
        inferred = sorted({sid for pair in train_pairs for sid in pair}, key=lambda s: int(s) if s.isdigit() else 10**9)
        train_ids = inferred
    val_pairs = pairs_from_args(val_ids, args.val_pairs_json, 0, pair_seed, False)
    test_pairs = pairs_from_args(test_ids, args.test_pairs_json, 0, pair_seed, False)
    assert_identity_disjoint({"train": train_ids, "validation": val_ids, "test": test_ids})
    assert_pairs_within(train_pairs, train_ids, "train")
    assert_pairs_within(val_pairs, val_ids, "validation")
    assert_pairs_within(test_pairs, test_ids, "test")
    if split_manifest is not None:
        train_pool = set(map(str, split_manifest["train_pool_ids"]))
        if not set(train_ids).issubset(train_pool):
            raise ValueError("Training identities are not a subset of the split-manifest train pool")
        if set(val_ids) != set(map(str, split_manifest["val_ids"])):
            raise ValueError("Validation identities do not match the split manifest")
        if set(test_ids) != set(map(str, split_manifest["test_ids"])):
            raise ValueError("Test identities do not match the split manifest")
    print(f"pairs train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}", flush=True)

    use_subunit_features = str(args.use_subunit_features).lower() == "true"
    vertex_feat, static_features = build_static_vertex_features(
        by_id, train_ids, use_subunit_features=use_subunit_features
    )
    # Topology only (faces/landmarks are shared across FaceScape registered
    # meshes per the identity-isolation audit -> not leakage).
    topology = by_id[str(train_ids[0])]
    source_pca = fit_truncated_pca(flatten_vertices(by_id, train_ids), args.source_pca_dim)
    cond_train, source_train, delta_train, _, _, mean, std = pair_conditions(by_id, train_pairs, source_pca)
    cond_val, _, delta_val, _, _, _, _ = pair_conditions(by_id, val_pairs, source_pca, mean, std)
    _, y_train_flat = pair_arrays(by_id, train_pairs)
    x_ctrl_train, _ = pair_arrays(by_id, train_pairs)
    ridge_ctrl = ridge_fit(x_ctrl_train, y_train_flat, args.ridge_lambda)
    source_flat_train = flatten_vertices(by_id, train_ids)
    # Reuse pair condition vector as the stronger linear baseline.
    ridge_cond = ridge_fit(cond_train, y_train_flat, args.ridge_lambda)
    cond_test: np.ndarray | None = None
    ridge_ctrl_pred: np.ndarray | None = None
    ridge_cond_pred: np.ndarray | None = None
    if not args.defer_test_evaluation:
        cond_test, _, _, _, _, _, _ = pair_conditions(by_id, test_pairs, source_pca, mean, std)
        x_ctrl_test, _ = pair_arrays(by_id, test_pairs)
        ridge_ctrl_pred = ridge_predict(x_ctrl_test, ridge_ctrl)
        ridge_cond_pred = ridge_predict(cond_test, ridge_cond)

    # Low-rank delta observation only feeds the CVAE posterior during training.
    delta_pca = fit_truncated_pca(y_train_flat, args.delta_pca_dim)
    obs_train = pca_transform(y_train_flat, delta_pca).astype(np.float32)
    obs_mean = obs_train.mean(axis=0, keepdims=True)
    obs_std = np.maximum(obs_train.std(axis=0, keepdims=True), 1e-6)
    obs_train = (obs_train - obs_mean) / obs_std

    started = time.perf_counter()
    field_info: dict[str, object] | None = None
    cvae_info: dict[str, object] | None = None
    field_pred: np.ndarray | None = None
    cvae_pred: np.ndarray | None = None
    alpha_selection: dict[str, float] | None = None
    if args.model_kind in {"both", "field"}:
        field = NeuralFieldDeformer(vertex_feat.shape[1], cond_train.shape[1], hidden=args.hidden)
        field, field_info = train_field(
            field,
            vertex_feat,
            cond_train,
            source_train,
            delta_train,
            obs_train,
            cond_val,
            delta_val,
            topology["landmarks"],
            topology["faces"],
            args,
            is_cvae=False,
            checkpoint_tag="field",
        )
        if cond_test is not None:
            field_pred = predict_field(field, vertex_feat, cond_test, is_cvae=False)
    if args.model_kind in {"both", "cvae"}:
        cvae = NeuralFieldCVAE(vertex_feat.shape[1], cond_train.shape[1], obs_train.shape[1], latent_dim=args.latent_dim, hidden=args.hidden)
        cvae, cvae_info = train_field(
            cvae,
            vertex_feat,
            cond_train,
            source_train,
            delta_train,
            obs_train,
            cond_val,
            delta_val,
            topology["landmarks"],
            topology["faces"],
            args,
            is_cvae=True,
            checkpoint_tag="cvae",
        )
        if cond_test is not None:
            cvae_pred = predict_field(cvae, vertex_feat, cond_test, is_cvae=True)
        cvae_val_pred = predict_field(
            cvae,
            vertex_feat,
            cond_val,
            is_cvae=True,
            progress_label="CVAE validation prediction",
        )
    elapsed = time.perf_counter() - started

    results: list[dict[str, float | str]] = []
    if not args.defer_test_evaluation:
        assert ridge_ctrl_pred is not None and ridge_cond_pred is not None
        results = [
            evaluate_method("source_copy", by_id, test_pairs, None),
            evaluate_method(f"ridge_controls_lambda_{args.ridge_lambda:g}", by_id, test_pairs, ridge_ctrl_pred),
            evaluate_method(
                f"ridge_controls_plus_sourcepca_lambda_{args.ridge_lambda:g}",
                by_id,
                test_pairs,
                ridge_cond_pred,
            ),
        ]
        if field_pred is not None:
            results.append(evaluate_method("shared_vertex_neural_field", by_id, test_pairs, field_pred))
    if cvae_info is not None:
        alpha_grid = [float(v) for v in args.alpha_grid.split(",") if v.strip()]
        ridge_cond_val = ridge_predict(cond_val, ridge_cond)
        alpha_selection = select_alpha_on_validation(ridge_cond_val, cvae_val_pred, delta_val, alpha_grid)
        cvae_checkpoint_signature = str(cvae_info["checkpoints"][-1]["sha256"])
        operating_candidates_path = write_operating_point_candidates(
            out_dir,
            by_id,
            val_pairs,
            ridge_cond_val,
            cvae_val_pred,
            alpha_grid,
            model_signature=cvae_checkpoint_signature,
        )
        selected_alpha = alpha_selection["best_alpha"]
        if cvae_pred is not None:
            assert ridge_cond_pred is not None
            results.append(evaluate_method("shared_vertex_neural_field_cvae", by_id, test_pairs, cvae_pred))
            for alpha in [0.25, 0.5, 0.75, selected_alpha]:
                hybrid = (1.0 - alpha) * ridge_cond_pred + alpha * cvae_pred
                label = f"hybrid_ridge_sourcepca_plus_cvae_alpha_{alpha:g}"
                if abs(alpha - selected_alpha) < 1e-12:
                    label = f"hybrid_validation_selected_alpha_{alpha:g}"
                results.append(evaluate_method(label, by_id, test_pairs, hybrid))
    suffix = f"{args.model_kind}_ew{args.edge_weight:g}_lw{args.lap_weight:g}".replace(".", "p")
    if results:
        atomic_write_csv(out_dir / f"neural_field_results_{suffix}.csv", results)
    method_preds: dict[str, np.ndarray] = {}
    if ridge_cond_pred is not None:
        method_preds["ridge_sourcepca"] = np.asarray(ridge_cond_pred, dtype=np.float32)
    if cvae_pred is not None:
        method_preds["cvae_only"] = np.asarray(cvae_pred, dtype=np.float32)
        if alpha_selection is not None:
            a = float(alpha_selection["best_alpha"])
            method_preds[f"hybrid_alpha_{a:g}"] = np.asarray((1.0 - a) * ridge_cond_pred + a * cvae_pred, dtype=np.float32)
    if field_pred is not None:
        method_preds["field"] = np.asarray(field_pred, dtype=np.float32)
    pair_metric_records = (
        write_per_pair_metrics(out_dir, by_id, test_pairs, method_preds)
        if method_preds
        else []
    )
    dense_npz_path = None
    if args.save_dense_predictions and ridge_cond_pred is not None:
        arrays: dict[str, np.ndarray] = {
            "test_pairs": np.asarray(test_pairs, dtype=object),
            "ridge_cond_pred": np.asarray(ridge_cond_pred, dtype=np.float32),
        }
        if cvae_pred is not None:
            arrays["cvae_pred"] = np.asarray(cvae_pred, dtype=np.float32)
            if alpha_selection is not None:
                a = float(alpha_selection["best_alpha"])
                arrays["hybrid_selected_pred"] = np.asarray((1.0 - a) * ridge_cond_pred + a * cvae_pred, dtype=np.float32)
                arrays["selected_alpha"] = np.asarray([a], dtype=np.float32)
        if field_pred is not None:
            arrays["field_pred"] = np.asarray(field_pred, dtype=np.float32)
        dense_npz_path = out_dir / f"neural_field_predictions_{suffix}.npz"
        atomic_savez_compressed(dense_npz_path, **arrays)
    package = {
        "args": vars(args),
        "input_manifest_sha256": sha256_file(repo / "manifest.json"),
        "split_manifest_sha256": (
            sha256_file(Path(args.split_manifest)) if args.split_manifest else None
        ),
        "train_pair_manifest_sha256": (
            sha256_file(Path(args.train_pairs_json)) if args.train_pairs_json else None
        ),
        "training_implementation_sha256": sha256_file(Path(__file__)),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "test_pairs": test_pairs,
        "source_pca": source_pca,
        "cond_mean": mean,
        "cond_std": std,
        "delta_pca": delta_pca,
        "obs_train_mean": obs_mean,
        "obs_train_std": obs_std,
        "ridge_cond": ridge_cond,
        "ridge_ctrl": ridge_ctrl,
        "selected_alpha": float(alpha_selection["best_alpha"]) if alpha_selection else None,
        "use_subunit_features": use_subunit_features,
        "vertex_feature_dim": int(vertex_feat.shape[1]),
        "feature_template_vertices": static_features["template_vertices"],
        "feature_template_sha256": static_features["template_sha256"],
        "feature_template_schema": static_features["template_schema"],
        "pair_metric_records": pair_metric_records,
        "dense_prediction_npz": str(dense_npz_path) if dense_npz_path else None,
        "validation_operating_point_candidates": str(operating_candidates_path) if cvae_info is not None else None,
    }
    if cvae_info is not None:
        package["cvae_state_dict"] = cvae.state_dict()
    if field_info is not None:
        package["field_state_dict"] = field.state_dict()
    package_path = out_dir / f"neural_field_model_package_{suffix}.pt"
    atomic_torch_save(
        package_path,
        package,
        required_keys=(
            "args", "train_ids", "val_ids", "test_ids", "feature_template_sha256",
            "feature_template_schema", "ridge_cond", "cvae_state_dict",
        ) if cvae_info is not None else ("args", "train_ids", "val_ids", "test_ids"),
    )
    summary = {
        "status": (
            "trained_validation_selected_test_deferred"
            if args.defer_test_evaluation
            else "trained_and_test_evaluated"
        ),
        "training": {"elapsed_sec": elapsed, "field": field_info, "cvae": cvae_info, "args": vars(args)},
        "alpha_selection": alpha_selection,
        "results": results,
        "pair_metric_records": pair_metric_records,
        "dense_prediction_npz": str(dense_npz_path) if dense_npz_path else None,
        "model_package": str(package_path),
        "model_package_sha256": sha256_file(package_path),
        "validation_operating_point_candidates": str(operating_candidates_path) if cvae_info is not None else None,
    }
    atomic_write_json(out_dir / f"neural_field_summary_{suffix}.json", summary)
    meta = artifact_metadata(
        repo_root=Path(__file__).resolve().parents[1],
        command_args=args,
        input_manifest_path=repo / "manifest.json",
        split_manifest_path=Path(args.split_manifest) if args.split_manifest else None,
        seed=args.seed,
        data_root_identifier=repo.name,
    )
    metadata_path = Path(args.metadata_out) if args.metadata_out else out_dir / f"neural_field_metadata_{suffix}.json"
    atomic_write_json(metadata_path, meta)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
