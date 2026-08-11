"""Re-select the hybrid mixing coefficient alpha on the VALIDATION set under the
STRICT protocol (landmarks hard-fixed -> free RMSE).

The frozen alpha=0.3 was chosen on validation under the OLD metric (ROI RMSE
including landmark vertices). For a self-consistent strict result set, alpha must
be re-selected on validation under the strict free-RMSE. This is inference only
(no retraining): ridge + CVAE validation predictions are regenerated from the
frozen model package with the verified srg_common reproduction utilities.

Outputs alpha_selection_strict.json with the per-alpha validation strict free
RMSE and the selected alpha (argmin).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE / "novelty_upgrade"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rhinoform import strict_protocol_patch as strict
import rhinoform.novelty_upgrade.srg_common as srg_common
from rhinoform.data import load_rows
from rhinoform.repro import atomic_write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--base-package", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha-grid", default="0,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.75,1.0")
    ap.add_argument("--max-pairs", type=int, default=0, help="smoke test: cap to first N validation pairs (0=all)")
    ap.add_argument("--force", action="store_true", help="recompute even if selection exists")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sel_path = out / "alpha_selection_strict.json"
    if sel_path.exists() and not args.force:
        print(f"[alpha] already selected (resume): {sel_path}", flush=True)
        return 0
    grid = [float(x) for x in args.alpha_grid.split(",") if x.strip()]

    strict.apply()
    from rhinoform.strict_protocol_patch import strict_metric_rows

    pkg = srg_common.load_package(Path(args.base_package))
    _, by_id = load_rows(Path(args.repo))
    train_ids = [str(v) for v in pkg["train_ids"]]
    available = [t for t in train_ids if t in by_id]
    vertex_feat = srg_common.build_vertex_features(
        by_id, train_ids, pkg["source_pca"], bool(pkg["use_subunit_features"]), available
    )
    val_pairs = [(str(a), str(b)) for a, b in pkg["val_pairs"]]
    if args.max_pairs and args.max_pairs > 0:
        val_pairs = val_pairs[: args.max_pairs]
        print(f"[alpha] SMOKE: capped to {len(val_pairs)} validation pairs", flush=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[alpha] device={device}; regenerating ridge/CVAE on {len(val_pairs)} validation pairs ...", flush=True)
    ridge, cvae, _ = srg_common.regenerate_predictions(by_id, pkg, val_pairs, vertex_feat, device=device)
    ridge = np.asarray(ridge, np.float64).reshape(len(val_pairs), -1)
    cvae = np.asarray(cvae, np.float64).reshape(len(val_pairs), -1)

    curve = []
    for a in grid:
        hybrid = (1.0 - a) * ridge + a * cvae
        rows = strict_metric_rows(by_id, val_pairs, hybrid)
        mean_free = float(np.mean([r["roi_rmse"] for r in rows]))
        mean_newflip = float(np.mean([r["normal_flip_pct"] for r in rows]))
        curve.append({"alpha": a, "val_strict_free_rmse": mean_free, "val_new_flip_pct": mean_newflip})
        print(f"  alpha={a:<5} val_free_rmse={mean_free:.5f}  new_flip%={mean_newflip:.3f}", flush=True)

    best = min(curve, key=lambda r: r["val_strict_free_rmse"])
    atomic_write_json(out / "alpha_selection_strict.json", {
        "protocol": "STRICT free-RMSE on validation",
        "alpha_grid": grid,
        "curve": curve,
        "selected_alpha": best["alpha"],
        "selected_val_strict_free_rmse": best["val_strict_free_rmse"],
        "n_val_pairs": len(val_pairs),
    })
    print(f"[alpha] selected alpha = {best['alpha']} (val free RMSE {best['val_strict_free_rmse']:.5f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
