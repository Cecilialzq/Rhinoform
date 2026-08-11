from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.patheffects as pe

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from plot_style_eval import SUBUNIT_LABEL, SUBUNITS, apply_eval_style  # noqa: E402


OUT_DIR = ROOT / "results" / "heatmap_redraw"
SVG_DIR = OUT_DIR / "svg"

SUBUNIT_PALETTE = {
    "root": "#4B2E83",
    "dorsum": "#00A7A7",
    "tip": "#F4D35E",
    "alar_left": "#2F80ED",
    "alar_right": "#F25C54",
}


def save_figure(fig: plt.Figure, name: str) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    paths = [OUT_DIR / f"{name}.pdf", SVG_DIR / f"{name}.svg", OUT_DIR / f"{name}.png"]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], dpi=500, bbox_inches="tight")
    plt.close(fig)
    return paths


def load_test_template() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[str]]:
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["rows"]
    row = next((item for item in rows if "test" in str(item.get("split", "")).lower()), rows[0])
    npz_path = Path(row["npz_path"])
    if not npz_path.is_absolute():
        npz_path = ROOT / "data" / npz_path
    data = np.load(npz_path, allow_pickle=False)
    subunits = {name: np.asarray(data[f"subunit_{name}"], dtype=np.int64) for name in SUBUNITS}
    return (
        np.asarray(data["vertices"], dtype=np.float64),
        np.asarray(data["faces"], dtype=np.int64),
        subunits,
        [str(row["subject_id"])],
    )


def vertex_subunit_labels(n_vertices: int, subunits: dict[str, np.ndarray]) -> np.ndarray:
    labels = np.full(n_vertices, "", dtype=object)
    for name in SUBUNITS:
        labels[np.asarray(subunits[name], dtype=np.int64)] = name
    if np.any(labels == ""):
        raise ValueError("Some ROI vertices do not have a subunit label.")
    return labels


def rotate_vertices(vertices: np.ndarray, yaw_deg: float = -26.0, pitch_deg: float = 7.0) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    ry = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ]
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ]
    )
    return centered @ (rx @ ry).T


def frontal_vertices(vertices: np.ndarray) -> np.ndarray:
    return vertices - vertices.mean(axis=0, keepdims=True)


def majority_face_labels(faces: np.ndarray, vertex_labels: np.ndarray) -> np.ndarray:
    labels = []
    for face in faces:
        labels.append(Counter(vertex_labels[np.asarray(face, dtype=np.int64)]).most_common(1)[0][0])
    return np.asarray(labels, dtype=object)


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    denom = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(denom, 1e-12)


def shaded_face_colors(rotated: np.ndarray, faces: np.ndarray, face_labels: np.ndarray) -> np.ndarray:
    z = rotated[faces].mean(axis=1)[:, 2:3]
    depth = (z - z.min()) / max(float(z.max() - z.min()), 1e-9)
    normals = face_normals(rotated, faces)
    light = np.array([-0.25, -0.35, 0.90], dtype=float)
    light = light / np.linalg.norm(light)
    lambert = np.clip((normals @ light)[:, None], 0.0, 1.0)
    shade = 0.76 + 0.18 * depth + 0.10 * lambert
    shade = np.clip(shade, 0.72, 1.03)
    base = np.asarray([mcolors.to_rgb(SUBUNIT_PALETTE[str(label)]) for label in face_labels])
    ambient = np.array([0.08, 0.11, 0.14])
    rgb = base * shade + ambient * (1.0 - shade) * 0.25
    return np.clip(rgb, 0.0, 1.0)


def boundary_segments(xy: np.ndarray, faces: np.ndarray, vertex_labels: np.ndarray) -> list[np.ndarray]:
    edges: set[tuple[int, int]] = set()
    for a, b, c in np.asarray(faces, dtype=np.int64):
        for u, v in ((a, b), (b, c), (c, a)):
            u, v = sorted((int(u), int(v)))
            if vertex_labels[u] != vertex_labels[v]:
                edges.add((u, v))
    return [xy[[u, v]] for u, v in sorted(edges)]


def draw_subunit_map() -> dict:
    apply_eval_style()
    vertices, faces, subunits, test_ids = load_test_template()
    labels = vertex_subunit_labels(len(vertices), subunits)
    rotated = frontal_vertices(vertices)
    xy = rotated[:, [0, 1]]
    depth = rotated[:, 2]
    order = np.argsort(depth)
    codes = np.zeros(len(vertices), dtype=np.float64)
    for idx, name in enumerate(SUBUNITS):
        codes[np.asarray(subunits[name], dtype=np.int64)] = float(idx)
    cmap = ListedColormap([SUBUNIT_PALETTE[name] for name in SUBUNITS], name="nasal_subunits")
    norm = BoundaryNorm(np.arange(-0.5, len(SUBUNITS) + 0.5, 1.0), cmap.N)

    fig, ax = plt.subplots(figsize=(3.15, 4.05))
    ax.set_facecolor("white")

    ax.scatter(
        xy[order, 0] + 0.32,
        xy[order, 1] - 0.42,
        s=8.2,
        c="#1F2D36",
        alpha=0.055,
        linewidths=0,
        rasterized=True,
        zorder=0,
    )
    sc = ax.scatter(
        xy[order, 0],
        xy[order, 1],
        c=codes[order],
        s=7.4,
        cmap=cmap,
        norm=norm,
        alpha=0.96,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    label_offsets = {
        "root": (0.0, 0.10),
        "dorsum": (0.0, -0.05),
        "tip": (0.0, 0.55),
        "alar_left": (-0.45, 0.40),
        "alar_right": (0.45, 0.40),
    }
    for name in SUBUNITS:
        idx = np.asarray(subunits[name], dtype=np.int64)
        cx, cy = xy[idx].mean(axis=0) + np.asarray(label_offsets[name])
        text = ax.text(
            cx,
            cy,
            SUBUNIT_LABEL[name],
            ha="center",
            va="center",
            fontsize=6.2,
            fontweight="bold",
            color="#17252D",
            zorder=4,
        )
        text.set_path_effects([pe.withStroke(linewidth=1.8, foreground=(1.0, 1.0, 1.0, 0.86))])

    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    dx = xmax - xmin
    dy = ymax - ymin
    ax.set_xlim(xmin - 0.12 * dx, xmax + 0.12 * dx)
    ax.set_ylim(ymin - 0.10 * dy, ymax + 0.08 * dy)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02, ticks=np.arange(len(SUBUNITS)))
    cbar.ax.set_yticklabels([SUBUNIT_LABEL[name] for name in SUBUNITS])
    cbar.ax.tick_params(labelsize=6.0, length=0)
    cbar.outline.set_linewidth(0.5)

    paths = save_figure(fig, "fig_nasal_subunit_3d_map")
    source = {
        "figure": "fig_nasal_subunit_3d_map",
        "outputs": [str(path) for path in paths],
        "data_sources": [
            str(ROOT / "data" / "manifest.json"),
            "data/meshes/*_neutral.npz: vertices, faces, subunit_root, subunit_dorsum, subunit_tip, subunit_alar_left, subunit_alar_right",
        ],
        "mesh_template": "first ROI mesh in data/manifest.json whose split contains 'test'",
        "n_template_identities": len(test_ids),
        "subunit_vertex_counts": {name: int(len(subunits[name])) for name in SUBUNITS},
        "rendering": "frontal vertex-cloud projection with categorical subunit heatmap colors and depth-ordered points",
        "contains_benchmark_methods": "no",
        "contains_bar_chart": "no",
        "values_changed": "no; only existing categorical subunit labels and mesh geometry were rendered",
    }
    (OUT_DIR / "fig_nasal_subunit_3d_map.source.json").write_text(
        json.dumps(source, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return source


def main() -> None:
    record = draw_subunit_map()
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
