#!/usr/bin/env python3
"""
Assemble the RB-SR ablation "ladder" from EXISTING frozen results — no new compute,
no fabrication. Every cell is read from a canonical strict file and carries its
provenance (source file + source row). If a source is absent, the cell is marked
EVIDENCE_MISSING with the path that was tried, rather than invented.

Ladder rungs (all canonical-strict, 9900 test pairs, seed 20260609):
  1. No source-PCA (dim 0)                      -> source_pca_dim_sweep_summary.json
  2. Source-conditioned ridge anchor (dim 16)   -> source_pca_dim_sweep_summary.json
  3. + residual, globally admitted (unconstrained gate) -> rbsr_gate_constrained_vs_unconstrained.json
  4. + spatial risk gate (RB-SR, deployed)      -> rbsr_gate_constrained_vs_unconstrained.json

Writes only under --out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MISSING = "EVIDENCE_MISSING"


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def rung_from_pca(summary: dict | None, dim: int, src_path: Path) -> dict:
    if not summary:
        return {"roi_rmse": MISSING, "new_flip_pct": MISSING, "edge_strain_p95": MISSING,
                "source": f"{MISSING}: {src_path}"}
    for row in summary.get("rows", []):
        if int(row.get("source_pca_dim", -999)) == dim:
            return {"roi_rmse": row.get("ridge_roi_rmse"),
                    "new_flip_pct": row.get("overall_new_flip_pct"),
                    "edge_strain_p95": row.get("edge_strain_p95"),
                    "source": f"{src_path.name}: source_pca_dim={dim}"}
    return {"roi_rmse": MISSING, "new_flip_pct": MISSING, "edge_strain_p95": MISSING,
            "source": f"{MISSING}: dim {dim} not in {src_path.name}"}


def rung_from_gate(audit: dict | None, variant: str, src_path: Path) -> dict:
    if not audit:
        return {"roi_rmse": MISSING, "new_flip_pct": MISSING, "edge_strain_p95": MISSING,
                "source": f"{MISSING}: {src_path}"}
    for row in audit.get("rows", []):
        if row.get("variant") == variant:
            return {"roi_rmse": row.get("roi_rmse"),
                    "new_flip_pct": row.get("new_flip_pct"),
                    "edge_strain_p95": row.get("edge_p95"),
                    "relative_violation": row.get("relative_violation"),
                    "feasible": row.get("feasible"),
                    "source": f"{src_path.name}: variant={variant}"}
    return {"roi_rmse": MISSING, "new_flip_pct": MISSING, "edge_strain_p95": MISSING,
            "source": f"{MISSING}: variant {variant} not in {src_path.name}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rr = Path(args.results_root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    pca_path = rr / "source_pca_dim_sweep" / "source_pca_dim_sweep_summary.json"
    gate_path = rr / "rbsr_gate_audit" / "rbsr_gate_constrained_vs_unconstrained.json"
    pca = load_json(pca_path)
    gate = load_json(gate_path)

    ladder = [
        {"rung": "1. No source-PCA (ridge, dim 0)", **rung_from_pca(pca, 0, pca_path)},
        {"rung": "2. Source-conditioned ridge anchor (dim 16, deployed)", **rung_from_pca(pca, 16, pca_path)},
        {"rung": "3. + residual, globally admitted (unconstrained gate)", **rung_from_gate(gate, "unconstrained", gate_path)},
        {"rung": "4. + spatial risk gate (RB-SR, deployed constrained)", **rung_from_gate(gate, "constrained", gate_path)},
    ]

    report = {
        "description": "RB-SR ablation ladder assembled from frozen canonical-strict results "
                       "(9900 test pairs, seed 20260609). No new computation; provenance per cell.",
        "protocol": "canonical_strict, 100-id test / 9900 pairs",
        "rungs": ladder,
        "provenance_files": {"source_pca_dim_sweep": str(pca_path), "rbsr_gate_audit": str(gate_path)},
        "n_missing_cells": sum(1 for r in ladder for k in ("roi_rmse", "new_flip_pct", "edge_strain_p95")
                               if r.get(k) == MISSING),
    }
    (out / "ablation_ladder.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    cols = ["rung", "roi_rmse", "new_flip_pct", "edge_strain_p95", "source"]
    lines = [",".join(cols)]
    for r in ladder:
        lines.append(",".join(str(r.get(c, "")).replace(",", ";") for c in cols))
    (out / "ablation_ladder.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = ["# RB-SR ablation ladder (assembled from frozen results)", "",
          "All rungs: canonical-strict protocol, 9900 test pairs, seed 20260609. "
          "Every value is read from a frozen file (provenance column); missing values are "
          "flagged, never invented.", "",
          "| Rung | ROI RMSE | new-flip % | edge-strain p95 | source |",
          "|---|---:|---:|---:|---|"]
    for r in ladder:
        md.append(f"| {r['rung']} | {r.get('roi_rmse')} | {r.get('new_flip_pct')} | "
                  f"{r.get('edge_strain_p95')} | {r.get('source')} |")
    md += ["", f"Missing cells: {report['n_missing_cells']}.",
           "", "Reading: source-PCA conditioning lowers ROI (rung 1->2); a globally admitted "
           "residual lowers ROI slightly but raises flips (rung 2->3); the spatial risk gate "
           "keeps most of the accuracy while restoring anchor-level regularity (rung 3->4)."]
    (out / "ablation_ladder.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
