"""STRICT re-evaluation of the MAIN benchmark table (no retraining).

Reuses the frozen scale676 neural prediction npz (ridge / CVAE / hybrid) on the
847-manifest 100-identity / 9900-pair test set, and RE-INFERS the classical
baselines (Laplacian, bi-Laplacian, ARAP) deterministically on the SAME pairs.
Everything is scored under the unified STRICT protocol (landmarks hard-fixed ->
free RMSE; baseline-relative NEW normal flips) and identity-clustered bootstrap.

Outputs (under --out):
  pair_metrics/identity_bootstrap_pair_metrics_<method>.csv   strict per-pair
  main_table_strict_means.csv                                 per-method means
  main_table_strict_bootstrap.csv                             vs-ridge bootstrap
  MAIN_TABLE_STRICT_DONE.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import csv as _csv
import os as _os
import tempfile as _tempfile

from rhinoform import strict_protocol_patch as strict
from rhinoform.data import load_rows
from rhinoform.geometry import solve_linear_handle_baseline, uniform_laplacian
from rhinoform.baselines import arap_predict_vectorised
from rhinoform.stats import LOWER_IS_BETTER, paired_identity_bootstrap
from rhinoform.repro import atomic_write_json


def write_csv(path: Path, rows: list[dict]) -> None:
    """Atomic CSV write: write to a temp file in the same dir, then os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fd, tmp = _tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with _os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        _os.replace(tmp, path)
    finally:
        if _os.path.exists(tmp):
            _os.remove(tmp)


HYBRID_ALPHA = 0.3
RIDGE_SYS = 1e-8
ARAP_ITERS = 3
DEFAULT_HANDLE_WEIGHT = 1000.0
METRIC_KEYS = ["roi_rmse", "landmark_rmse", "dorsum_rmse", "tip_rmse",
               "edge_strain_p95", "normal_flip_pct", "abs_flip_pct", "missed_flip_pct"]


def _find_key(files, *needles):
    for f in files:
        low = f.lower()
        if all(n in low for n in needles):
            return f
    return None


def load_neural(pred_npz: Path):
    d = np.load(pred_npz, allow_pickle=True)
    files = list(d.files)
    pairs_key = _find_key(files, "test", "pair") or _find_key(files, "pair")
    ridge_key = _find_key(files, "ridge", "cond") or _find_key(files, "ridge")
    cvae_key = _find_key(files, "cvae")
    if not (pairs_key and ridge_key and cvae_key):
        raise KeyError(f"could not locate ridge/cvae/pairs keys in {files}")
    pairs = [(str(a), str(b)) for a, b in np.asarray(d[pairs_key]).tolist()]
    ridge = np.asarray(d[ridge_key], dtype=np.float64)
    cvae = np.asarray(d[cvae_key], dtype=np.float64)
    return pairs, ridge, cvae


def handle_weight_from_tuning(path: Path | None) -> float:
    if path is None or not path.exists():
        return DEFAULT_HANDLE_WEIGHT
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_HANDLE_WEIGHT
    for k in ("handle_weight", "selected_handle_weight", "best_handle_weight"):
        if isinstance(cfg, dict) and k in cfg:
            return float(cfg[k])
        if isinstance(cfg, dict):
            for v in cfg.values():
                if isinstance(v, dict) and k in v:
                    return float(v[k])
    return DEFAULT_HANDLE_WEIGHT


def classical_predictions(by_id, pairs, handle_weight):
    template = next(iter(by_id.values()))
    lms = np.asarray(template["landmarks"], dtype=np.int64)
    faces = template["faces"]
    n = template["vertices"].shape[0]
    controls = np.stack([(by_id[t]["vertices"] - by_id[s]["vertices"])[lms] for s, t in pairs], axis=0)
    sources = [by_id[s]["vertices"] for s, _ in pairs]
    lap = uniform_laplacian(n, faces)
    lap_pred = solve_linear_handle_baseline(controls, lms, n, lap, handle_weight, RIDGE_SYS)
    bilap_pred = solve_linear_handle_baseline(controls, lms, n, lap @ lap, handle_weight, RIDGE_SYS)
    lap_init = lap_pred.reshape(len(pairs), -1, 3)
    arap_pred = arap_predict_vectorised(sources, controls, lms, faces, lap, lap_init,
                                        handle_weight, RIDGE_SYS, ARAP_ITERS)
    return {
        "laplacian_handles": np.asarray(lap_pred, dtype=np.float32),
        "bilaplacian_handles": np.asarray(bilap_pred, dtype=np.float32),
        "arap_handles_iter3": np.asarray(arap_pred, dtype=np.float32),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pred-npz", required=True, help="frozen scale676 neural prediction npz")
    ap.add_argument("--geometric-tuning", default="", help="geometric_validation_tuning.json (optional)")
    ap.add_argument("--alpha-json", default="", help="alpha_selection_strict.json (validation-selected alpha)")
    ap.add_argument("--hybrid-alpha", type=float, default=HYBRID_ALPHA, help="fallback if --alpha-json absent")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--max-pairs", type=int, default=0, help="smoke test: cap to first N test pairs (0=all)")
    ap.add_argument("--force", action="store_true", help="recompute even if DONE marker exists")
    args = ap.parse_args()

    out = Path(args.out)
    done_marker = out / "MAIN_TABLE_STRICT_DONE.json"
    if done_marker.exists() and not args.force:
        print(f"[main-table] already complete (resume): {done_marker}", flush=True)
        return 0

    strict.apply()
    from rhinoform.stats import metric_rows_for_method  # patched -> strict

    (out / "pair_metrics").mkdir(parents=True, exist_ok=True)

    alpha = float(args.hybrid_alpha)
    if args.alpha_json and Path(args.alpha_json).exists():
        alpha = float(json.loads(Path(args.alpha_json).read_text())["selected_alpha"])
    print(f"[main-table] strict-selected hybrid alpha = {alpha}", flush=True)

    pairs, ridge, cvae = load_neural(Path(args.pred_npz))
    if args.max_pairs and args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
        ridge = ridge[: args.max_pairs]
        cvae = cvae[: args.max_pairs]
        print(f"[main-table] SMOKE: capped to {len(pairs)} pairs", flush=True)
    hybrid = (1.0 - alpha) * ridge + alpha * cvae
    print(f"[main-table] pairs={len(pairs)}", flush=True)

    _, by_id = load_rows(Path(args.repo))
    hw = handle_weight_from_tuning(Path(args.geometric_tuning) if args.geometric_tuning else None)
    print(f"[main-table] classical handle_weight={hw}", flush=True)
    classical = classical_predictions(by_id, pairs, hw)

    method_preds = {
        "ridge_sourcepca": ridge,
        "cvae_only": cvae,
        f"hybrid_alpha_{alpha}": hybrid,
        **classical,
    }

    rows_by_method = {}
    means = []
    for name, pred in method_preds.items():
        rows = metric_rows_for_method(by_id, pairs, pred)
        rows_by_method[name] = rows
        write_csv(out / "pair_metrics" / f"identity_bootstrap_pair_metrics_{name}.csv", rows)
        means.append({"method": name, **{k: float(np.mean([r[k] for r in rows])) for k in METRIC_KEYS}})
    write_csv(out / "main_table_strict_means.csv", means)

    # vs-ridge identity bootstrap
    baseline = "ridge_sourcepca"
    boot_rows = []
    for name, rows in rows_by_method.items():
        if name == baseline:
            continue
        for metric in LOWER_IS_BETTER:
            res = paired_identity_bootstrap(rows_by_method[baseline], rows, name, baseline,
                                            metric, "source_id", args.n_boot, args.seed)
            boot_rows.append({"method": name, "baseline": baseline, "metric": metric, **res})
    write_csv(out / "main_table_strict_bootstrap.csv", boot_rows)

    atomic_write_json(out / "MAIN_TABLE_STRICT_DONE.json", {
        "protocol": "STRICT: landmarks hard-fixed -> free RMSE; normal_flip_pct = baseline-relative NEW flip",
        "n_pairs": len(pairs), "hybrid_alpha": alpha, "handle_weight": hw,
        "seed": args.seed, "methods": list(method_preds),
    })
    print("[main-table] done ->", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
