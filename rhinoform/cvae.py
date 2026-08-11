from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import svds
from torch import nn
import torch.nn.functional as F

from .data import (
    edge_index,
    evaluate_method,
    load_rows,
    ordered_pairs,
    pair_arrays,
    ridge_fit,
    ridge_predict,
    split_ids,
)
from .repro import set_global_seed


def flatten_vertices(by_id: dict[str, dict], ids: list[str]) -> np.ndarray:
    return np.stack([by_id[sid]["vertices"].reshape(-1) for sid in ids], axis=0)


def fit_truncated_pca(mat: np.ndarray, k: int) -> dict[str, np.ndarray]:
    mean = mat.mean(axis=0)
    centered = mat - mean
    kk = min(int(k), centered.shape[0] - 1, centered.shape[1] - 1)
    if kk <= 0:
        raise ValueError("Not enough samples for PCA")
    _, singular, vt = svds(centered, k=kk)
    order = np.argsort(singular)[::-1]
    return {"mean": mean, "components": vt[order], "singular": singular[order]}


def pca_transform(mat: np.ndarray, pca: dict[str, np.ndarray]) -> np.ndarray:
    return (mat - pca["mean"]) @ pca["components"].T


def build_condition(
    by_id: dict[str, dict],
    pairs: list[tuple[str, str]],
    source_pca: dict[str, np.ndarray],
) -> np.ndarray:
    landmarks = next(iter(by_id.values()))["landmarks"]
    ctrl = []
    source_flat = []
    for src_id, tgt_id in pairs:
        src = by_id[src_id]["vertices"]
        tgt = by_id[tgt_id]["vertices"]
        ctrl.append((tgt - src)[landmarks].reshape(-1))
        source_flat.append(src.reshape(-1))
    source_code = pca_transform(np.stack(source_flat, axis=0), source_pca)
    return np.concatenate([np.stack(ctrl, axis=0), source_code], axis=1)


def fit_delta_pca(y_train: np.ndarray, k: int) -> dict[str, np.ndarray]:
    return fit_truncated_pca(y_train, k)


class DeterministicDeformer(nn.Module):
    def __init__(self, cond_dim: int, out_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 2),
            nn.SiLU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, out_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)


class ConditionalDeformationVAE(nn.Module):
    def __init__(self, cond_dim: int, obs_dim: int, out_dim: int, latent_dim: int = 8, hidden: int = 256) -> None:
        super().__init__()
        self.prior = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        self.prior_mu = nn.Linear(hidden, latent_dim)
        self.prior_logvar = nn.Linear(hidden, latent_dim)
        self.encoder = nn.Sequential(nn.Linear(cond_dim + obs_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        self.enc_mu = nn.Linear(hidden, latent_dim)
        self.enc_logvar = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(cond_dim + latent_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 2),
            nn.SiLU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, out_dim),
        )

    def stats(self, cond: torch.Tensor, obs: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hp = self.prior(cond)
        pmu = self.prior_mu(hp)
        plogvar = self.prior_logvar(hp).clamp(-8.0, 8.0)
        if obs is None:
            return pmu, plogvar, pmu, plogvar
        he = self.encoder(torch.cat([cond, obs], dim=1))
        qmu = self.enc_mu(he)
        qlogvar = self.enc_logvar(he).clamp(-8.0, 8.0)
        return pmu, plogvar, qmu, qlogvar

    def forward(self, cond: torch.Tensor, obs: torch.Tensor | None = None, deterministic: bool = False) -> dict[str, torch.Tensor]:
        pmu, plogvar, qmu, qlogvar = self.stats(cond, obs)
        if self.training and obs is not None and not deterministic:
            z = qmu + torch.randn_like(qmu) * torch.exp(0.5 * qlogvar)
        else:
            z = pmu
        pred = self.decoder(torch.cat([cond, z], dim=1))
        return {"pred": pred, "pmu": pmu, "plogvar": plogvar, "qmu": qmu, "qlogvar": qlogvar}


def kl_normal(qmu: torch.Tensor, qlogvar: torch.Tensor, pmu: torch.Tensor, plogvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(
        plogvar
        - qlogvar
        + (torch.exp(qlogvar) + (qmu - pmu).pow(2)) / torch.exp(plogvar).clamp_min(1e-8)
        - 1.0
    )


def make_batches(n: int, batch_size: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return [order[i : i + batch_size] for i in range(0, n, batch_size)]


def torch_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    landmark_flat: torch.Tensor,
    edge0: torch.Tensor,
    edge1: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    dense = F.l1_loss(pred, target)
    ctrl = F.l1_loss(pred[:, landmark_flat], target[:, landmark_flat])
    p = pred.reshape(pred.shape[0], -1, 3)
    t = target.reshape(target.shape[0], -1, 3)
    edge = F.l1_loss(p[:, edge0] - p[:, edge1], t[:, edge0] - t[:, edge1])
    total = dense + 5.0 * ctrl + 0.1 * edge
    return total, {"dense_l1": float(dense.detach()), "ctrl_l1": float(ctrl.detach()), "edge_l1": float(edge.detach())}


def validation_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    diff = pred.reshape(pred.shape[0], -1, 3) - target.reshape(target.shape[0], -1, 3)
    return float(np.mean(np.sqrt(np.mean(np.sum(diff * diff, axis=2), axis=1))))


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    faces: np.ndarray,
    landmarks: np.ndarray,
    args: argparse.Namespace,
) -> tuple[DeterministicDeformer, dict[str, float]]:
    device = torch.device("cpu")
    model = DeterministicDeformer(x_train.shape[1], y_train.shape[1], hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    xt = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    xv = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    edge = edge_index(faces)
    edge0 = torch.as_tensor(edge[:, 0], dtype=torch.long, device=device)
    edge1 = torch.as_tensor(edge[:, 1], dtype=torch.long, device=device)
    landmark_flat = torch.as_tensor(np.concatenate([landmarks * 3 + d for d in range(3)]), dtype=torch.long, device=device)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for idx in make_batches(len(x_train), args.batch_size, args.seed + epoch):
            opt.zero_grad(set_to_none=True)
            pred = model(xt[idx])
            loss, _ = torch_loss(pred, yt[idx], landmark_flat, edge0, edge1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                pred_val = model(xv).cpu().numpy()
            val = validation_rmse(pred_val, y_val)
            if val < best_val:
                best_val = val
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += args.eval_every
            print(f"mlp epoch={epoch} val_roi_rmse={val:.4f} best={best_val:.4f}", flush=True)
            if patience >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_roi_rmse": best_val, "best_epoch": float(best_epoch)}


def train_cvae(
    x_train: np.ndarray,
    y_train: np.ndarray,
    obs_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    obs_val: np.ndarray,
    faces: np.ndarray,
    landmarks: np.ndarray,
    args: argparse.Namespace,
) -> tuple[ConditionalDeformationVAE, dict[str, float]]:
    device = torch.device("cpu")
    model = ConditionalDeformationVAE(x_train.shape[1], obs_train.shape[1], y_train.shape[1], args.latent_dim, args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    xt = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    ot = torch.as_tensor(obs_train, dtype=torch.float32, device=device)
    xv = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    yv_obs = torch.as_tensor(obs_val, dtype=torch.float32, device=device)
    edge = edge_index(faces)
    edge0 = torch.as_tensor(edge[:, 0], dtype=torch.long, device=device)
    edge1 = torch.as_tensor(edge[:, 1], dtype=torch.long, device=device)
    landmark_flat = torch.as_tensor(np.concatenate([landmarks * 3 + d for d in range(3)]), dtype=torch.long, device=device)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    patience = 0
    for epoch in range(1, args.epochs + 1):
        beta = min(args.beta, args.beta * epoch / max(1, args.kl_warmup))
        model.train()
        for idx in make_batches(len(x_train), args.batch_size, args.seed + epoch):
            opt.zero_grad(set_to_none=True)
            out = model(xt[idx], ot[idx])
            recon_loss, _ = torch_loss(out["pred"], yt[idx], landmark_flat, edge0, edge1)
            kl = kl_normal(out["qmu"], out["qlogvar"], out["pmu"], out["plogvar"])
            loss = recon_loss + beta * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                pred_val = model(xv, yv_obs, deterministic=True)["pred"].cpu().numpy()
            val = validation_rmse(pred_val, y_val)
            if val < best_val:
                best_val = val
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += args.eval_every
            print(f"cvae epoch={epoch} beta={beta:.2e} val_roi_rmse={val:.4f} best={best_val:.4f}", flush=True)
            if patience >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_roi_rmse": best_val, "best_epoch": float(best_epoch)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="../data")
    parser.add_argument("--out", default="../results")
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--source-pca-dim", type=int, default=16)
    parser.add_argument("--delta-pca-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--kl-warmup", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    set_global_seed(args.seed, deterministic=True)
    repo = Path(args.repo)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, by_id = load_rows(repo)
    train_ids = split_ids(by_id, "clean-prior train")
    val_ids = split_ids(by_id, "clean-prior validation")
    test_ids = split_ids(by_id, "main test")
    train_pairs = ordered_pairs(train_ids, None, args.seed)
    val_pairs = ordered_pairs(val_ids, None, args.seed)
    test_pairs = ordered_pairs(test_ids, None, args.seed)
    print(f"pairs train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}", flush=True)

    x_ctrl_train, y_train = pair_arrays(by_id, train_pairs)
    x_ctrl_val, y_val = pair_arrays(by_id, val_pairs)
    x_ctrl_test, _ = pair_arrays(by_id, test_pairs)
    source_pca = fit_truncated_pca(flatten_vertices(by_id, train_ids), args.source_pca_dim)
    x_train = build_condition(by_id, train_pairs, source_pca)
    x_val = build_condition(by_id, val_pairs, source_pca)
    x_test = build_condition(by_id, test_pairs, source_pca)
    cond_mean = x_train.mean(axis=0, keepdims=True)
    cond_std = np.maximum(x_train.std(axis=0, keepdims=True), 1e-6)
    x_train = (x_train - cond_mean) / cond_std
    x_val = (x_val - cond_mean) / cond_std
    x_test = (x_test - cond_mean) / cond_std

    delta_pca = fit_delta_pca(y_train, args.delta_pca_dim)
    obs_train = pca_transform(y_train, delta_pca)
    obs_val = pca_transform(y_val, delta_pca)
    obs_mean = obs_train.mean(axis=0, keepdims=True)
    obs_std = np.maximum(obs_train.std(axis=0, keepdims=True), 1e-6)
    obs_train = (obs_train - obs_mean) / obs_std
    obs_val = (obs_val - obs_mean) / obs_std

    template = next(iter(by_id.values()))
    faces = template["faces"]
    landmarks = template["landmarks"]

    ridge_ctrl = ridge_fit(x_ctrl_train, y_train, 100.0)
    ridge_cond = ridge_fit(x_train, y_train, 100.0)
    ridge_ctrl_pred = ridge_predict(x_ctrl_test, ridge_ctrl)
    ridge_cond_pred = ridge_predict(x_test, ridge_cond)

    started = time.perf_counter()
    mlp, mlp_info = train_mlp(x_train, y_train, x_val, y_val, faces, landmarks, args)
    cvae, cvae_info = train_cvae(x_train, y_train, obs_train, x_val, y_val, obs_val, faces, landmarks, args)
    train_elapsed = time.perf_counter() - started

    with torch.no_grad():
        xt = torch.as_tensor(x_test, dtype=torch.float32)
        mlp_pred = mlp(xt).cpu().numpy()
        cvae_pred = cvae(xt, None, deterministic=True)["pred"].cpu().numpy()

    results = [
        evaluate_method("source_copy", by_id, test_pairs, None),
        evaluate_method("ridge_controls_lambda_100", by_id, test_pairs, ridge_ctrl_pred),
        evaluate_method("ridge_controls_plus_sourcepca_lambda_100", by_id, test_pairs, ridge_cond_pred),
        evaluate_method("deterministic_mlp_controls_plus_sourcepca", by_id, test_pairs, mlp_pred),
        evaluate_method("conditional_deformation_vae_v0", by_id, test_pairs, cvae_pred),
    ]
    path = out_dir / "conditional_vae_results.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "training": {
            "elapsed_sec": train_elapsed,
            "mlp": mlp_info,
            "cvae": cvae_info,
            "args": vars(args),
        },
        "results": results,
    }
    (out_dir / "conditional_vae_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(
        {
            "mlp": mlp.state_dict(),
            "cvae": cvae.state_dict(),
            "source_pca": source_pca,
            "delta_pca": delta_pca,
            "cond_mean": cond_mean,
            "cond_std": cond_std,
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "args": vars(args),
        },
        out_dir / "conditional_vae_v0.pt",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
