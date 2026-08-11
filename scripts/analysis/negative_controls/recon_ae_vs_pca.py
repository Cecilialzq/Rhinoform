"""Test 1 -- unconditional reconstruction positive control: AE vs PCA.

Question answered: *is the neural model intrinsically competent on this ROI?*

We autoencode the per-identity nose-ROI shape (no control points, no
conditioning) and compare a neural autoencoder against linear PCA at matched
latent dimension, on identity-disjoint held-out test identities. This mirrors
the CoMA-style claim ("a mesh autoencoder reconstructs better than PCA") on our
own data and ROI.

Two architectures (``--arch``):

* ``cvae`` (MAIN evidence) -- the *same CVAE-family decoder* used by the main
  conditional model (``NeuralFieldDeformer`` per-vertex field + the same static
  vertex features), but with the sparse-control conditioning REMOVED and the
  encoder fed the identity ROI shape. This directly rebuts "your main CVAE is
  too weak, so the conditional failure is unreliable": it is the same decoder
  family, only unconditional. Deterministic by default (no KL); pass
  ``--variational`` for the VAE ablation (appendix only -- with KL a loss to PCA
  is confounded by the reconstruction/regularisation trade-off).
* ``mlp`` (SUPPLEMENTARY) -- a generic MLP autoencoder. A model-agnostic upper
  sanity check that nonlinear modelling is feasible on the ROI; it does NOT
  substitute for the CVAE-family ablation because it changes the model object.
* ``coma`` (STRONG AUDIT / REVIEWER SHIELD) -- a CoMA-style Chebyshev
  graph-convolutional mesh autoencoder built on the ROI mesh graph (from-scratch
  ChebConv, no torch_geometric). Answers the strongest variant of the rebuttal:
  "does the conclusion hold even with a classical face-mesh AE?". It does NOT
  replace the CVAE-family ablation (again a different model object); it is an
  additional, optional shield.

This is a *diagnostic / positive control*, NOT a preset conclusion. If the AE
matches or beats PCA at matched latent dim, the neural family is demonstrably
competent, so a later "no conditional gain" finding cannot be dismissed as a
weak network. The clean narrative: the neural model is not intrinsically
ineffective (it helps unconditionally); the advantage collapses under
sparse-control conditioning.

Identity-disjoint: fit on ``train pool``, early-stop on ``clean-prior
validation``, evaluate reconstruction RMSE on ``main test``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rhinoform.diag_common import (
    atomic_write_json,
    identity_bootstrap_ci,
    identity_matrix,
    manifest_splits,
    provenance,
    recon_rmse_per_identity,
)
from rhinoform.data import edge_index, load_rows
from rhinoform.train import NeuralFieldDeformer, build_static_vertex_features


class ShapeAutoencoder(nn.Module):
    """Generic MLP autoencoder (supplementary positive control)."""

    def __init__(self, dim: int, latent: int, hidden: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.LayerNorm(hidden), nn.Linear(hidden, latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.SiLU(), nn.LayerNorm(hidden), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor, vertex_feat: torch.Tensor | None = None):
        return self.decoder(self.encoder(x)), None, None


class NeuralFieldAutoencoder(nn.Module):
    """Unconditional ablation of the main CVAE family.

    Reuses the *exact* ``NeuralFieldDeformer`` decoder (the per-vertex field that
    the main conditional model uses) and the same static vertex features. The
    only change vs the main CVAE is that conditioning is removed: the encoder
    maps a single identity's ROI shape to the latent, and the decoder
    reconstructs the (centred) shape as ``field(vertex_feat, z)``.
    """

    def __init__(self, shape_dim: int, vertex_dim: int, latent: int, hidden: int, variational: bool) -> None:
        super().__init__()
        self.variational = variational
        self.enc = nn.Sequential(nn.Linear(shape_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        self.qmu = nn.Linear(hidden, latent)
        self.qlogvar = nn.Linear(hidden, latent)
        self.field = NeuralFieldDeformer(vertex_dim, latent, hidden=hidden)

    def forward(self, x: torch.Tensor, vertex_feat: torch.Tensor):
        h = self.enc(x)
        mu = self.qmu(h)
        if self.variational and self.training:
            logvar = self.qlogvar(h).clamp(-8.0, 8.0)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            logvar = torch.zeros_like(mu)
            z = mu
        pred = self.field(vertex_feat, z)  # (B, V, 3)
        return pred.reshape(x.shape[0], -1), mu, logvar


def build_scaled_laplacian(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    """Chebyshev-scaled normalised graph Laplacian L_scaled = 2L/lmax - I.

    Adjacency comes from the template mesh edges; ``L = I - D^-1/2 A D^-1/2``.
    Returned dense (V, V) float32 for cheap batched propagation.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    edges = edge_index(faces)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    vals = np.ones(rows.shape[0], dtype=np.float64)
    adj = sp.coo_matrix((vals, (rows, cols)), shape=(n_vertices, n_vertices)).tocsr()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    dmat = sp.diags(dinv)
    lap = sp.identity(n_vertices) - dmat @ adj @ dmat
    try:
        lmax = float(eigsh(lap, k=1, which="LM", return_eigenvectors=False)[0])
    except Exception:
        lmax = 2.0
    lmax = max(lmax, 1e-6)
    scaled = (2.0 / lmax) * lap - sp.identity(n_vertices)
    return np.asarray(scaled.todense(), dtype=np.float32)


class ChebConv(nn.Module):
    """Chebyshev spectral graph convolution (CoMA building block), from scratch.

    out = sum_{k=0}^{K-1} T_k(L_scaled) x W_k, with the Chebyshev recurrence
    T_0=x, T_1=L x, T_k = 2 L T_{k-1} - T_{k-2}. Implemented as a concat of the
    K propagated signals followed by a single linear map.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int) -> None:
        super().__init__()
        self.k = k
        self.lin = nn.Linear(in_ch * k, out_ch)

    @staticmethod
    def _propagate(lap: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        b, v, c = x.shape
        xr = x.permute(1, 0, 2).reshape(v, b * c)  # (V, B*C)
        out = lap @ xr  # (V, B*C)
        return out.reshape(v, b, c).permute(1, 0, 2)

    def forward(self, x: torch.Tensor, lap: torch.Tensor) -> torch.Tensor:
        terms = [x]
        if self.k > 1:
            terms.append(self._propagate(lap, x))
        for _ in range(2, self.k):
            terms.append(2.0 * self._propagate(lap, terms[-1]) - terms[-2])
        return self.lin(torch.cat(terms, dim=-1))


class CoMAAutoencoder(nn.Module):
    """CoMA-style graph-convolutional mesh autoencoder (strong audit / reviewer shield).

    Classical face-mesh AE archetype: stacked Chebyshev graph convolutions on the
    ROI mesh graph + a dense bottleneck to the latent, then symmetric decoder back
    to (V, 3). No mesh-pooling hierarchy (would need precomputed transforms); the
    graph convs already give the spectral, neighbourhood-aware inductive bias that
    distinguishes this from the MLP-AE. Deterministic (mu/logvar = None).
    """

    def __init__(self, n_vertices: int, latent: int, hidden: int, k: int, feat: int) -> None:
        super().__init__()
        self.n_vertices = n_vertices
        self.feat = feat
        self.act = nn.SiLU()
        self.enc1 = ChebConv(3, feat, k)
        self.enc2 = ChebConv(feat, feat, k)
        self.enc_lin = nn.Linear(n_vertices * feat, latent)
        self.dec_lin = nn.Linear(latent, n_vertices * feat)
        self.dec1 = ChebConv(feat, feat, k)
        self.dec2 = ChebConv(feat, 3, k)

    def forward(self, x: torch.Tensor, lap: torch.Tensor):
        b = x.shape[0]
        h = x.reshape(b, self.n_vertices, 3)
        h = self.act(self.enc1(h, lap))
        h = self.act(self.enc2(h, lap))
        z = self.enc_lin(h.reshape(b, -1))
        h = self.dec_lin(z).reshape(b, self.n_vertices, self.feat)
        h = self.act(self.dec1(h, lap))
        out = self.dec2(h, lap)  # (B, V, 3)
        return out.reshape(b, -1), None, None


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pca_reconstruct(train_c: np.ndarray, eval_c: np.ndarray, k: int) -> np.ndarray:
    _, _, vt = np.linalg.svd(train_c, full_matrices=False)
    basis = vt[:k]
    return (eval_c @ basis.T) @ basis


def kl_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(mu.pow(2) + torch.exp(logvar) - logvar - 1.0)


def train_autoencoder(
    model: nn.Module,
    train_s: np.ndarray,
    val_s: np.ndarray,
    vertex_feat: torch.Tensor | None,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    beta: float,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, int]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xtr = torch.as_tensor(train_s, dtype=torch.float32, device=device)
    xva = torch.as_tensor(val_s, dtype=torch.float32, device=device)
    n = xtr.shape[0]
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    rng = np.random.default_rng(seed)
    bs = min(batch_size, n)
    for epoch in range(epochs):
        model.train()
        for idx in (rng.permutation(n)[s : s + bs] for s in range(0, n, bs)):
            batch = xtr[idx]
            opt.zero_grad()
            recon, mu, logvar = model(batch, vertex_feat)
            loss = nn.functional.mse_loss(recon, batch)
            if beta > 0.0 and mu is not None:
                loss = loss + beta * kl_to_standard_normal(mu, logvar)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            recon, _, _ = model(xva, vertex_feat)
            vloss = float(nn.functional.mse_loss(recon, xva).item())
        if vloss < best_val - 1e-9:
            best_val, best_epoch = vloss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def main() -> int:
    parser = argparse.ArgumentParser(description="Test 1: unconditional AE vs PCA reconstruction on the nose ROI.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arch", choices=["cvae", "mlp", "coma"], default="cvae",
                        help="cvae = main CVAE-family unconditional ablation; mlp = supplementary generic AE; "
                             "coma = CoMA-style graph-conv AE (strong audit / reviewer shield).")
    parser.add_argument("--variational", action="store_true",
                        help="cvae only: add KL (VAE ablation, appendix). Default deterministic AE.")
    parser.add_argument("--beta", type=float, default=1e-4, help="KL weight when --variational.")
    parser.add_argument("--use-subunit-features", action="store_true",
                        help="cvae only: match a main model trained with subunit vertex features.")
    parser.add_argument("--cheb-k", type=int, default=6, help="coma only: Chebyshev polynomial order.")
    parser.add_argument("--coma-feat", type=int, default=16, help="coma only: graph-conv channel width.")
    parser.add_argument("--dims", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260615, 20260616, 20260617])
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    variational = bool(args.variational) and args.arch == "cvae"
    beta = args.beta if variational else 0.0
    print(f"[test1] arch={args.arch} variational={variational} device={device} dims={args.dims} seeds={args.seeds}", flush=True)

    _, by_id = load_rows(Path(args.repo))
    train_ids, val_ids, test_ids = manifest_splits(by_id)
    print(f"[test1] identities train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}", flush=True)

    train_abs = identity_matrix(by_id, train_ids)
    val_abs = identity_matrix(by_id, val_ids)
    test_abs = identity_matrix(by_id, test_ids)
    mean = train_abs.mean(axis=0, keepdims=True)
    train_c, val_c, test_c = train_abs - mean, val_abs - mean, test_abs - mean
    scale = max(float(np.sqrt(np.mean(train_c**2))), 1e-9)
    train_s = (train_c / scale).astype(np.float32)
    val_s = (val_c / scale).astype(np.float32)
    test_s = (test_c / scale).astype(np.float32)
    shape_dim = train_s.shape[1]

    # The per-arch side input passed through train_autoencoder's `vertex_feat` slot:
    #   cvae -> static vertex features; coma -> scaled graph Laplacian; mlp -> None.
    vertex_feat = None
    vertex_dim = 0
    n_vertices = shape_dim // 3
    if args.arch == "cvae":
        vertex_features_np, _ = build_static_vertex_features(
            by_id, train_ids, use_subunit_features=args.use_subunit_features
        )
        vertex_dim = int(vertex_features_np.shape[1])
        vertex_feat = torch.as_tensor(vertex_features_np, dtype=torch.float32, device=device)
        print(f"[test1] CVAE-family decoder: vertex_features dim={vertex_dim}", flush=True)
    elif args.arch == "coma":
        faces = np.asarray(by_id[train_ids[0]]["faces"], dtype=np.int64)
        lap_np = build_scaled_laplacian(faces, n_vertices)
        vertex_feat = torch.as_tensor(lap_np, dtype=torch.float32, device=device)
        print(f"[test1] CoMA graph: V={n_vertices} edges->Laplacian {lap_np.shape} K={args.cheb_k} feat={args.coma_feat}", flush=True)

    test_ids_str = [str(i) for i in test_ids]
    results: dict[str, dict] = {}
    for k in args.dims:
        pca_pred = pca_reconstruct(train_c, test_c, k) + mean
        pca_rmse_pp = recon_rmse_per_identity(pca_pred, test_abs)
        pca_mean, pca_lo, pca_hi = identity_bootstrap_ci(pca_rmse_pp, test_ids_str, args.n_boot, 20260615)

        ae_seed_means: list[float] = []
        ae_stop_epochs: list[int] = []
        ae_rmse_pp_seeds: list[np.ndarray] = []
        for seed in args.seeds:
            # Seed BEFORE constructing the model so weight initialisation is
            # controlled by this seed (not by construction order). train_autoencoder
            # re-seeds again for the data shuffle / VAE sampling noise.
            torch.manual_seed(seed)
            np.random.seed(seed)
            if args.arch == "cvae":
                model = NeuralFieldAutoencoder(shape_dim, vertex_dim, k, args.hidden, variational).to(device)
            elif args.arch == "coma":
                model = CoMAAutoencoder(n_vertices, k, args.hidden, args.cheb_k, args.coma_feat).to(device)
            else:
                model = ShapeAutoencoder(shape_dim, k, args.hidden).to(device)
            model, stop = train_autoencoder(
                model, train_s, val_s, vertex_feat, args.epochs, args.patience,
                args.batch_size, args.lr, beta, seed, device,
            )
            with torch.no_grad():
                recon_s, _, _ = model(torch.as_tensor(test_s, dtype=torch.float32, device=device), vertex_feat)
            ae_pred = recon_s.cpu().numpy().astype(np.float64) * scale + mean
            ae_rmse_pp = recon_rmse_per_identity(ae_pred, test_abs)
            ae_rmse_pp_seeds.append(ae_rmse_pp)
            ae_seed_means.append(float(ae_rmse_pp.mean()))
            ae_stop_epochs.append(int(stop))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        ae_mean = float(np.mean(ae_seed_means))
        ae_std = float(np.std(ae_seed_means))
        # Paired (per-identity) AE error averaged over seeds, then paired bootstrap
        # of the AE-minus-PCA delta over the SAME held-out identities.
        ae_rmse_pp_mean = np.mean(np.stack(ae_rmse_pp_seeds, axis=0), axis=0)
        delta_pp = ae_rmse_pp_mean - pca_rmse_pp  # <0 => AE more accurate on that identity
        delta_mean, delta_lo, delta_hi = identity_bootstrap_ci(delta_pp, test_ids_str, args.n_boot, 20260615)
        results[str(k)] = {
            "pca_recon_rmse": {"mean": pca_mean, "ci95": [pca_lo, pca_hi]},
            "ae_recon_rmse": {"mean": ae_mean, "std_over_seeds": ae_std, "per_seed": ae_seed_means},
            "ae_stop_epochs": ae_stop_epochs,
            "ae_beats_pca": bool(delta_hi < 0.0),  # AE clearly better only if paired CI < 0
            "ae_minus_pca": delta_mean,
            "ae_minus_pca_ci95": [delta_lo, delta_hi],
            "relative_gain_pct": float((pca_mean - ae_mean) / pca_mean * 100.0) if pca_mean > 0 else float("nan"),
        }
        print(f"[test1] k={k:3d}  PCA={pca_mean:.4f}  AE={ae_mean:.4f}±{ae_std:.4f}  "
              f"Δ(AE-PCA)={delta_mean:.4f} CI[{delta_lo:.4f},{delta_hi:.4f}]  "
              f"AE_clearly_beats={results[str(k)]['ae_beats_pca']}  gain={results[str(k)]['relative_gain_pct']:.1f}%", flush=True)

    # Three-tier verdict on paired-bootstrap evidence (stricter than any-win).
    if args.arch == "cvae":
        tag = "CVAE-family unconditional " + ("VAE" if variational else "AE")
    elif args.arch == "coma":
        tag = "CoMA-style graph-conv AE (strong audit / reviewer shield)"
    else:
        tag = "MLP-AE (supplementary)"
    clear_wins = [k for k, v in results.items() if v["ae_minus_pca_ci95"][1] < 0.0]      # paired CI upper < 0
    soft_wins = [k for k, v in results.items() if v["ae_minus_pca"] < 0.0]               # mean < 0
    best_hi = min((v["ae_minus_pca_ci95"][1] for v in results.values()), default=float("inf"))

    if len(clear_wins) >= 2:
        verdict = (f"STRONG_POSITIVE_CONTROL_OK ({tag}): AE clearly beats PCA (paired CI<0) at "
                   f"{len(clear_wins)} latent dim(s) {clear_wins} -> the neural family is competent; "
                   "the 'too-weak-network' rebuttal is firmly closed.")
    elif len(clear_wins) == 1:
        verdict = (f"WEAK_POSITIVE_CONTROL ({tag}): AE clearly beats PCA (paired CI<0) at only one latent "
                   f"dim {clear_wins}; narrow evidence -- state cautiously.")
    elif len(soft_wins) >= 1:
        verdict = (f"WEAK_POSITIVE_CONTROL ({tag}): AE mean below PCA at {soft_wins} but no paired CI clearly "
                   "<0; evidence not significant -- state cautiously.")
    else:
        verdict = (f"POSITIVE_CONTROL_FAILED ({tag}): AE does not beat PCA (paired) at any tested dim; "
                   "report honestly -- this architecture cannot serve as the positive control.")
    out = {
        "test": "test1_unconditional_recon_ae_vs_pca",
        "arch": args.arch,
        "variational": variational,
        "role": ("MAIN positive control (same decoder family as the conditional model)"
                 if args.arch == "cvae" and not variational else
                 "VAE ablation (appendix)" if variational else
                 "strong audit / reviewer shield (classical face-mesh AE)" if args.arch == "coma" else
                 "supplementary generic AE"),
        "purpose": (
            f"Positive control / diagnostic ({tag}): does the neural model reconstruct the nose ROI better "
            "than linear PCA at matched latent dim, on identity-disjoint held-out identities? NOT a preset conclusion."
        ),
        "provenance": provenance(20260615, {
            "repo": str(args.repo), "arch": args.arch, "variational": variational, "beta": beta,
            "use_subunit_features": bool(args.use_subunit_features), "vertex_feature_dim": vertex_dim,
            "cheb_k": args.cheb_k if args.arch == "coma" else None,
            "coma_feat": args.coma_feat if args.arch == "coma" else None,
            "hidden": args.hidden, "epochs": args.epochs, "patience": args.patience,
            "batch_size": args.batch_size, "lr": args.lr, "device": str(device),
        }),
        "splits": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "metric": (
            "per-identity reconstruction RMSE (mm). AE error averaged over seeds; AE-minus-PCA delta "
            "uses a paired identity-level bootstrap on the same held-out identities. "
            "ae_beats_pca = paired CI upper bound < 0 (clearly better)."
        ),
        "results_by_latent_dim": results,
        "verdict_basis": {
            "latent_dims_with_paired_ci_below_0": clear_wins,
            "latent_dims_with_mean_below_0": soft_wins,
            "best_paired_ci_upper": best_hi,
        },
        "verdict": verdict,
    }
    out_path = Path(args.out)
    atomic_write_json(out_path, out)
    print(f"[test1] wrote {out_path}", flush=True)
    print(f"[test1] VERDICT: {out['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
