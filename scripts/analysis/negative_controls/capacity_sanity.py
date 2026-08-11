"""Test 3 -- minimal capacity sanity check (appendix-grade).

Not a full latent/depth/width sweep. Just enough to refute a one-line
"your network underfits" objection: train a conditional MLP (legal input
``cond`` -> dense ROI displacement) at a small / medium / large width and report
the *training* loss alongside *held-out* ROI RMSE and model-induced new-flip.

Expected pattern if the negative result is real: training loss keeps dropping
with capacity (the model CAN fit), while held-out ROI RMSE does NOT improve
(there is no transferable nonlinear signal to extract). Run this only if Test 2
shows the residual is at least partly learnable; if Test 2 says "not learnable",
this is a confirmatory appendix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rhinoform.diag_common import atomic_write_json, build_conditions, deterministic_pairs, identity_bootstrap_ci, provenance, sha256_file
from rhinoform.data import load_rows
from rhinoform.safe_fusion import new_and_missed_folds, signed_fold_indicator


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CondMLP(nn.Module):
    def __init__(self, cond_dim: int, out_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fold_mask(src: np.ndarray, edited: np.ndarray, faces: np.ndarray) -> np.ndarray:
    det, _ = signed_fold_indicator(src, edited, faces)
    return det < 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Test 3: minimal capacity sanity for the conditional model.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base-model-package", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--max-train-pairs", type=int, default=4000)
    parser.add_argument("--max-test-pairs", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"[test3] device={device} widths={args.widths}", flush=True)

    _, by_id = load_rows(Path(args.repo))
    base = torch.load(args.base_model_package, map_location="cpu", weights_only=False)
    train_ids = [str(v) for v in base["train_ids"]]
    if base.get("test_pairs") is not None and len(base["test_pairs"]) > 0:
        test_pairs = [(str(a), str(b)) for a, b in base["test_pairs"]]
        if args.max_test_pairs and len(test_pairs) > args.max_test_pairs:
            rng = np.random.default_rng(args.seed)
            sel = np.sort(rng.choice(len(test_pairs), size=args.max_test_pairs, replace=False))
            test_pairs = [test_pairs[i] for i in sel]
    else:
        test_pairs = deterministic_pairs([str(v) for v in base["test_ids"]], args.max_test_pairs, args.seed)
    train_pairs = deterministic_pairs(train_ids, args.max_train_pairs, args.seed)
    faces = np.asarray(by_id[train_ids[0]]["faces"], dtype=np.int64)

    cond_tr, _, delta_tr, _, _ = build_conditions(by_id, train_pairs, base)
    cond_te, source_te, delta_te, _, _ = build_conditions(by_id, test_pairs, base)
    out_dim = delta_tr.reshape(len(train_pairs), -1).shape[1]
    y_tr = delta_tr.reshape(len(train_pairs), -1).astype(np.float32)
    y_te = delta_te.reshape(len(test_pairs), -1).astype(np.float32)
    scale = float(np.sqrt(np.mean(y_tr**2)))
    scale = max(scale, 1e-9)

    # Intrinsic target folds (method-independent) for new-flip bookkeeping.
    source_te = np.asarray(source_te, dtype=np.float64)
    delta_te_3 = y_te.reshape(len(test_pairs), -1, 3).astype(np.float64)
    tgt_folds = [fold_mask(source_te[i], source_te[i] + delta_te_3[i], faces) for i in range(len(test_pairs))]
    source_ids = [a for a, _ in test_pairs]

    xtr = torch.as_tensor(cond_tr, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(y_tr / scale, dtype=torch.float32, device=device)
    xte = torch.as_tensor(cond_te, dtype=torch.float32, device=device)
    n = xtr.shape[0]

    results: dict[str, dict] = {}
    for width in args.widths:
        torch.manual_seed(args.seed)
        model = CondMLP(cond_tr.shape[1], out_dim, width).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        n_params = sum(p.numel() for p in model.parameters())
        rng = np.random.default_rng(args.seed)
        bs = min(args.batch_size, n)
        last_train = float("nan")
        for epoch in range(args.epochs):
            model.train()
            perm = rng.permutation(n)
            ep_loss = 0.0
            nb = 0
            for s in range(0, n, bs):
                idx = perm[s : s + bs]
                opt.zero_grad()
                loss = nn.functional.mse_loss(model(xtr[idx]), ytr[idx])
                loss.backward()
                opt.step()
                ep_loss += float(loss.item())
                nb += 1
            last_train = ep_loss / max(1, nb)
        model.eval()
        with torch.no_grad():
            pred_te = model(xte).cpu().numpy().astype(np.float64) * scale
        pred3 = pred_te.reshape(len(test_pairs), -1, 3)
        rmse_pp = np.sqrt(np.mean(np.sum((pred3 - delta_te_3) ** 2, axis=2), axis=1))
        new_pp = np.empty(len(test_pairs))
        for i in range(len(test_pairs)):
            pred_fold = fold_mask(source_te[i], source_te[i] + pred3[i], faces)
            new_fold, _ = new_and_missed_folds(pred_fold, tgt_folds[i])
            new_pp[i] = 100.0 * new_fold.mean()
        rmse_mean, rmse_lo, rmse_hi = identity_bootstrap_ci(rmse_pp, source_ids, args.n_boot, args.seed)
        new_mean, new_lo, new_hi = identity_bootstrap_ci(new_pp, source_ids, args.n_boot, args.seed)
        results[str(width)] = {
            "n_params": int(n_params),
            "final_train_mse_scaled": float(last_train),
            "test_roi_rmse": {"mean": rmse_mean, "ci95": [rmse_lo, rmse_hi]},
            "test_new_flip_pct": {"mean": new_mean, "ci95": [new_lo, new_hi]},
        }
        print(f"[test3] width={width:5d} params={n_params/1e6:.1f}M  train_mse={last_train:.5f}  "
              f"test_rmse={rmse_mean:.4f}  new_flip={new_mean:.3f}%", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    widths_sorted = sorted(args.widths)
    train_losses = [results[str(w)]["final_train_mse_scaled"] for w in widths_sorted]
    test_rmses = [results[str(w)]["test_roi_rmse"]["mean"] for w in widths_sorted]
    train_drops = train_losses[-1] < train_losses[0]
    test_flat = abs(test_rmses[-1] - test_rmses[0]) / max(test_rmses[0], 1e-9) < 0.05
    out = {
        "test": "capacity_sanity",
        "purpose": "Refute 'underfitting': train loss should drop with capacity while held-out ROI RMSE stays flat.",
        "provenance": provenance(args.seed, {
            "repo": str(args.repo),
            "base_model_package": str(args.base_model_package),
            "base_model_package_sha256": sha256_file(Path(args.base_model_package)),
            "feature_template_sha256": base.get("feature_template_sha256"),
            "n_train_pairs": len(train_pairs), "n_test_pairs": len(test_pairs),
            "epochs": args.epochs, "device": str(device),
        }),
        "results_by_width": results,
        "verdict": (
            "CAPACITY_NOT_THE_LIMIT: training loss decreases with capacity while held-out ROI RMSE stays flat "
            "(<5% change) -> the model can fit but there is no transferable nonlinear signal; not underfitting."
            if (train_drops and test_flat) else
            "INCONCLUSIVE: pattern is not the clean 'train down / test flat'; inspect the per-width table before claiming."
        ),
    }
    out_path = Path(args.out)
    atomic_write_json(out_path, out)
    print(f"[test3] wrote {out_path}", flush=True)
    print(f"[test3] VERDICT: {out['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
