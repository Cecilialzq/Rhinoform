"""Generate the publication-ready RB-SR capacity/trade-off figure."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty summary CSV: {path}")
    return sorted(rows, key=lambda row: int(row["source_pca_dim"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--lamm-roi", type=float, required=True)
    parser.add_argument("--lamm-flip", type=float, required=True)
    parser.add_argument("--lamm-strain", type=float, required=True)
    parser.add_argument("--deployment-dimension", type=int, default=64)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.summary)
    dimensions = [int(row["source_pca_dim"]) for row in rows]
    roi = [float(row["roi_rmse"]) for row in rows]
    flip = [float(row["normal_flip_pct"]) for row in rows]
    strain = [float(row["edge_strain_p95"]) for row in rows]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    blue, vermillion, green, gray = "#0072B2", "#D55E00", "#009E73", "#777777"
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.35))
    panels = (
        (axes[0], roi, args.lamm_roi, "Free-ROI RMSE (mm)", blue),
        (axes[1], flip, args.lamm_flip, "New flips (%)", vermillion),
        (axes[2], strain, args.lamm_strain, "Edge strain p95", green),
    )
    for ax, values, baseline, ylabel, color in panels:
        ax.plot(dimensions, values, color=color, marker="o", label="RB-SR")
        ax.axhline(baseline, color=gray, linestyle="--", linewidth=1.2, label="LAMM")
        for dimension, value in zip(dimensions, values):
            if dimension == args.deployment_dimension:
                ax.scatter([dimension], [value], s=58, facecolors="none", edgecolors="#000000", linewidths=1.2, zorder=5)
                ax.annotate("PCA-64", (dimension, value), xytext=(4, 5), textcoords="offset points", fontsize=7.5)
        ax.set_xlabel("Source PCA dimension")
        ax.set_ylabel(ylabel)
        ax.set_xticks(dimensions)
    axes[0].legend(loc="best")
    fig.tight_layout(w_pad=1.3)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_prefix.with_suffix(".pdf"))
    fig.savefig(args.out_prefix.with_suffix(".png"), dpi=300)
    print(args.out_prefix.with_suffix(".pdf"))
    print(args.out_prefix.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
