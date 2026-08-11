from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from plot_style_eval import apply_eval_style, projection_xy, style_axis_equal, vertex_edge_strain  # noqa: E402


OUT_DIR = ROOT / "results" / "heatmap_redraw"
SVG_DIR = OUT_DIR / "svg"

HEAT_CMAP = LinearSegmentedColormap.from_list(
    "rhino_demo_heat",
    ["#27377F", "#2166AC", "#1E9EAD", "#56C56F", "#F0D64E", "#F06A3D"],
)


def save_figure(fig: plt.Figure, name: str) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    paths = [OUT_DIR / f"{name}.pdf", SVG_DIR / f"{name}.svg", OUT_DIR / f"{name}.png"]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], dpi=450, bbox_inches="tight")
    plt.close(fig)
    return paths


def robust_norm(values: np.ndarray, low: float = 0.0, high: float = 99.0) -> colors.Normalize:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return colors.Normalize(0.0, 1.0)
    lo = float(np.nanpercentile(finite, low))
    hi = float(np.nanpercentile(finite, high))
    if hi <= lo:
        hi = lo + 1e-6
    return colors.Normalize(lo, hi, clip=True)


def draw_vertex_heatmap(
    ax: plt.Axes,
    vertices: np.ndarray,
    values: np.ndarray,
    norm: colors.Normalize,
    *,
    title: str,
    landmarks: np.ndarray,
) -> None:
    xy, depth = projection_xy(vertices, "front")
    order = np.argsort(depth)
    ax.scatter(
        xy[order, 0],
        xy[order, 1],
        c=np.asarray(values)[order],
        s=4.8,
        cmap=HEAT_CMAP,
        norm=norm,
        alpha=0.96,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        xy[landmarks, 0],
        xy[landmarks, 1],
        s=13,
        c="#FF6EA8",
        edgecolors="white",
        linewidths=0.45,
        zorder=5,
    )
    ax.set_title(title, fontsize=8.2, pad=2.5)
    style_axis_equal(ax, xy, pad=0.055)


def main() -> None:
    apply_eval_style()
    demo_path = ROOT / "demo" / "public" / "demo_case.json"
    demo = json.loads(demo_path.read_text(encoding="utf-8"))
    source = np.asarray(demo["geometry"]["source"], dtype=np.float64)
    output = np.asarray(demo["methods"]["rbsr"]["vertices"], dtype=np.float64)
    faces = np.asarray(demo["geometry"]["faces"], dtype=np.int64)
    landmarks = np.asarray(demo["geometry"]["landmarks"], dtype=np.int64)

    source_depth = source[:, 2]
    output_depth = output[:, 2]
    displacement = np.linalg.norm(output - source, axis=1)
    edge_strain = vertex_edge_strain(source, output, faces)

    depth_norm = robust_norm(np.concatenate([source_depth, output_depth]), 0.0, 99.0)
    disp_norm = colors.Normalize(0.0, max(1e-6, float(np.nanpercentile(displacement, 99.0))), clip=True)
    strain_norm = colors.Normalize(0.0, max(1e-6, float(np.nanpercentile(edge_strain, 99.0))), clip=True)

    panels = [
        ("A. Source ROI", source, source_depth, depth_norm, "Depth"),
        ("B. Edited ROI", output, output_depth, depth_norm, "Depth"),
        ("C. Dense displacement magnitude", output, displacement, disp_norm, "Displacement"),
        ("D. Edge-strain risk proxy", output, edge_strain, strain_norm, "Edge strain"),
    ]

    fig = plt.figure(figsize=(7.4, 5.45), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 0.045, 1.0, 0.045], wspace=0.035, hspace=0.12)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 2])]
    caxes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 3])]

    for ax, cax, (title, verts, values, norm, cbar_label) in zip(axes, caxes, panels):
        draw_vertex_heatmap(ax, verts, values, norm, title=title, landmarks=landmarks)
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=HEAT_CMAP), cax=cax)
        cbar.set_label(cbar_label, fontsize=6.8)
        cbar.ax.tick_params(labelsize=6.2)
        cbar.outline.set_linewidth(0.5)

    metrics = demo["methods"]["rbsr"].get("metrics", {})
    controls = np.asarray(demo["geometry"]["controls"], dtype=np.float64).reshape(9, 3)
    note = (
        f"Source ID {demo['case']['source_id']} -> target ID {demo['case'].get('target_id')}; "
        f"mean handle displacement={np.mean(np.linalg.norm(controls, axis=1)):.2f}; "
        f"max handle displacement={np.max(np.linalg.norm(controls, axis=1)):.2f}; "
        f"ROI RMSE={float(metrics.get('roi_rmse', np.nan)):.3f}; "
        f"normal flips={float(metrics.get('normal_flip_pct', np.nan)):.3f}%; "
        f"edge-strain p95={float(metrics.get('edge_strain_p95', np.nan)):.3f}."
    )

    paths = save_figure(fig, "fig_demo_roi_deformation_diagnostic_heatmap")
    payload = {
        "figure": "fig_demo_roi_deformation_diagnostic_heatmap",
        "outputs": [str(p) for p in paths],
        "data_sources": [str(demo_path)],
        "source_identity": str(demo["case"]["source_id"]),
        "target_identity": str(demo["case"].get("target_id")),
        "rendering": "frontal vertex-cloud heatmaps for ROI source/edit depth, displacement magnitude, and edge-strain risk",
        "mesh_required": "no; point-cloud heatmap is the recommended rendering for this diagnostic figure",
        "values_changed": "no",
        "metrics": {
            "roi_rmse": metrics.get("roi_rmse"),
            "normal_flip_pct": metrics.get("normal_flip_pct"),
            "edge_strain_p95": metrics.get("edge_strain_p95"),
            "mean_handle_displacement": float(np.mean(np.linalg.norm(controls, axis=1))),
            "max_handle_displacement": float(np.max(np.linalg.norm(controls, axis=1))),
        },
        "caption_metric_note": note,
    }
    (OUT_DIR / "fig_demo_roi_deformation_diagnostic_heatmap.source.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
