"""Verify that locally-regenerated ridge/CVAE/hybrid test predictions reproduce
the frozen Required-tier aggregate metrics. This is the gate that licenses the
SRG experiment: if reproduction is not tight, we stop.

Run: python3 rhinoform/novelty_upgrade/reproduce_check.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rhinoform.novelty_upgrade.srg_common import (ROOT, load_package, load_available_rows, build_vertex_features,
                        regenerate_predictions, per_pair_metric_rows, aggregate, topology)

FROZEN = {  # from runs/required_primary_chain0_scale676/neural_field_summary_*.json
    "ridge": {"roi_rmse": 1.0554193795461495, "edge_strain_p95": 0.21366145663027275, "normal_flip_pct": 0.7737134426798575},
    "cvae": {"roi_rmse": 1.2810834602734789, "edge_strain_p95": 0.3786172175878375, "normal_flip_pct": 2.9145385119089884},
    "hybrid": {"roi_rmse": 1.0505078653196749, "edge_strain_p95": 0.20669665462749245, "normal_flip_pct": 0.9500419450953165},
}


def main() -> int:
    pkg = load_package()
    by_id = load_available_rows(ROOT / "data")
    faces, edges, subunits, landmarks = topology(by_id)
    test_pairs = [tuple(p) for p in pkg["test_pairs"]]
    avail = sorted(by_id.keys(), key=lambda x: int(x))
    vf = build_vertex_features(by_id, pkg["train_ids"], pkg["source_pca"],
                               bool(pkg["use_subunit_features"]), avail)
    ridge, cvae, _ = regenerate_predictions(by_id, pkg, test_pairs, vf)
    a = float(pkg["selected_alpha"])
    hybrid = (1.0 - a) * ridge + a * cvae

    out = {"selected_alpha": a, "n_test_pairs": len(test_pairs), "methods": {}}
    for name, pred in (("ridge", ridge), ("cvae", cvae), ("hybrid", hybrid)):
        agg = aggregate(per_pair_metric_rows(by_id, test_pairs, pred, faces, edges, subunits, landmarks))
        fr = FROZEN[name]
        out["methods"][name] = {
            "repro": agg,
            "frozen": fr,
            "abs_diff_roi_rmse": abs(agg["roi_rmse"] - fr["roi_rmse"]),
            "abs_diff_flip": abs(agg["normal_flip_pct"] - fr["normal_flip_pct"]),
            "abs_diff_edge": abs(agg["edge_strain_p95"] - fr["edge_strain_p95"]),
        }
    # ridge must match near-exactly (no scale dependence); cvae/hybrid close
    out["ridge_exact_ok"] = out["methods"]["ridge"]["abs_diff_roi_rmse"] < 1e-6
    out["cvae_close_ok"] = out["methods"]["cvae"]["abs_diff_roi_rmse"] < 5e-3
    out["hybrid_close_ok"] = out["methods"]["hybrid"]["abs_diff_roi_rmse"] < 5e-3
    outdir = ROOT / "outputs/novelty_upgrade_2026_06_11/reproduce_check"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "reproduce_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
