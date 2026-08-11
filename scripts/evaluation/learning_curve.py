"""Full STRICT learning curve by RE-INFERRING from every scale's model package.

Stage B in the notebook only re-scored runs that had a saved prediction npz
(only scale30 and scale676). The intermediate scales (60/120/240/480) have a
trained model package (.pt) but no saved predictions. This script regenerates
ridge+CVAE predictions on each run's own test pairs from the model package
(GPU when available; no retraining), scores them under the STRICT protocol, and
emits the complete learning-curve table with identity-bootstrap CIs.

Resumable: per-run/method CSVs are reused if present.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
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
from rhinoform.stats import write_csv


def ident_ci(rows, metric, n_boot=2000, seed=2026):
    by: dict[str, list[float]] = {}
    for r in rows:
        by.setdefault(str(r["source_id"]), []).append(float(r[metric]))
    mu = np.array([np.mean(v) for v in by.values()])
    rng = np.random.default_rng(seed)
    bs = [float(np.mean(rng.choice(mu, len(mu), replace=True))) for _ in range(n_boot)]
    return float(mu.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.1, help="hybrid alpha (strict-selected)")
    ap.add_argument("--alpha-json", default="")
    args = ap.parse_args()

    alpha = args.alpha
    if args.alpha_json and Path(args.alpha_json).exists():
        alpha = float(json.loads(Path(args.alpha_json).read_text())["selected_alpha"])

    strict.apply()
    from rhinoform.strict_protocol_patch import strict_metric_rows

    out = Path(args.out)
    (out / "per_run").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[lc] device={device}  hybrid_alpha={alpha}", flush=True)

    _, by_id = load_rows(Path(args.repo))

    pkgs = sorted(Path(args.runs_dir).glob("required_*/neural_field_model_package_*.pt"))
    print(f"[lc] found {len(pkgs)} model packages", flush=True)
    table = []
    for pkg_path in pkgs:
        rid = pkg_path.parent.name
        m = re.search(r"chain(\w+)_scale(\d+)", rid) or re.search(r"(anchor)_scale(\d+)", rid)
        chain = m.group(1) if m else ""
        scale = int(m.group(2)) if m else 0
        try:
            pkg = srg_common.load_package(pkg_path)
            train_ids = [str(v) for v in pkg["train_ids"]]
            available = [t for t in train_ids if t in by_id]
            vfeat = srg_common.build_vertex_features(
                by_id, train_ids, pkg["source_pca"], bool(pkg["use_subunit_features"]), available
            )
            test_pairs = [(str(a), str(b)) for a, b in pkg["test_pairs"]]
            preds = None
            for nm in ["ridge", "cvae", "hybrid"]:
                csvp = out / "per_run" / f"{rid}__{nm}.csv"
                if csvp.exists():
                    rows = list(csv.DictReader(open(csvp)))
                else:
                    if preds is None:
                        rg, cv, _ = srg_common.regenerate_predictions(by_id, pkg, test_pairs, vfeat, device=device)
                        rg = np.asarray(rg, np.float64).reshape(len(test_pairs), -1)
                        cv = np.asarray(cv, np.float64).reshape(len(test_pairs), -1)
                        preds = {"ridge": rg, "cvae": cv, "hybrid": (1.0 - alpha) * rg + alpha * cv}
                    rows = strict_metric_rows(by_id, test_pairs, preds[nm])
                    write_csv(csvp, rows)
                mu, lo, hi = ident_ci(rows, "roi_rmse")
                table.append({
                    "run": rid, "chain": chain, "scale": scale, "method": nm,
                    "roi_rmse_mean": mu, "roi_rmse_ci_low": lo, "roi_rmse_ci_high": hi,
                    "new_flip_pct": float(np.mean([float(r["normal_flip_pct"]) for r in rows])),
                })
            print(f"[lc] done {rid} (scale {scale})", flush=True)
        except Exception as e:
            print(f"[lc] skip {rid}: {e}", flush=True)

    table.sort(key=lambda r: (str(r["chain"]), r["scale"], r["method"]))
    write_csv(out / "learning_curve_strict_table_full.csv", table)
    print(f"[lc] wrote full learning curve: {len(table)} rows -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
