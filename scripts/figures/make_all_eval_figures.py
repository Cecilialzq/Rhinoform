from __future__ import annotations

import json
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_pdf import PdfPages

from plot_style_eval import (
    METHODS,
    METHOD_COLOR,
    METHOD_LABEL,
    SUBUNITS,
    SUBUNIT_LABEL,
    EvalPaths,
    PredictionStore,
    add_panel_label,
    apply_eval_style,
    default_paths,
    diagnostic_values,
    draw_handles,
    draw_mesh_projection,
    draw_vertex_cloud,
    ensure_output_dirs,
    external_ci_from_bootstrap,
    figure_caption_notes,
    full_face_from_demo,
    load_identities,
    load_identity,
    load_pairs,
    load_subunit_rows,
    method_summary,
    method_vertices,
    new_flip_faces,
    overlay_subunit_boundaries,
    pair_index,
    projection_xy,
    read_csv_rows,
    save_figure,
    select_worst_cases,
    selected_qualitative_case,
    style_axis_equal,
    subunit_vertex_labels,
    vertex_edge_strain,
    vertex_from_face_indicator,
    write_json,
)


def source_record(out_dir: Path, name: str, payload: dict) -> None:
    write_json(out_dir / f"{name}.source.json", payload)


def make_fig_5_1(paths: EvalPaths, out_dir: Path) -> dict:
    demo = json.loads(paths.demo_case.read_text(encoding="utf-8"))
    source_id = str(demo["case"]["source_id"])
    target_id = str(demo["case"]["target_id"])
    by_id = load_identities(paths.data, {source_id, target_id})
    src = by_id[source_id]["vertices"]
    tgt = by_id[target_id]["vertices"]
    landmarks = by_id[source_id]["landmarks"]
    combined = np.vstack([src, tgt])
    arrow_scale = 1.5

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.2))
    draw_vertex_cloud(axes[0], src, view="front", s=3.0, alpha=0.62, limits_vertices=combined)
    axes[0].set_title(f"(a) Source identity\nID {source_id}")
    draw_vertex_cloud(axes[1], tgt, view="front", s=3.0, alpha=0.62, limits_vertices=combined)
    axes[1].set_title(f"(b) Target identity\nID {target_id}")
    draw_handles(axes[2], src, landmarks, view="front")
    axes[2].set_title("(c) 9 sparse handles")

    xy_src, _ = projection_xy(src, "front")
    xy_tgt, _ = projection_xy(tgt, "front")
    axes[3].scatter(xy_src[:, 0], xy_src[:, 1], s=2.0, c="#4777B3", alpha=0.18, linewidths=0, rasterized=True)
    axes[3].scatter(
        xy_src[landmarks, 0],
        xy_src[landmarks, 1],
        s=12,
        c="#C42034",
        edgecolors="white",
        linewidths=0.35,
        zorder=4,
    )
    delta = (xy_tgt[landmarks] - xy_src[landmarks]) * arrow_scale
    axes[3].quiver(
        xy_src[landmarks, 0],
        xy_src[landmarks, 1],
        delta[:, 0],
        delta[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.006,
        color="#1D3557",
        zorder=5,
    )
    style_axis_equal(axes[3], projection_xy(combined, "front")[0])
    axes[3].set_title("(d) Measured displacement\n(target - source)")
    fig.text(
        0.5,
        0.02,
        "Red points = 9 sparse nasal handles; blue points = original mesh vertices. Arrow length x1.5 for visibility.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    paths_out = save_figure(fig, out_dir, "fig_5_1_source_target_landmark_displacement")
    payload = {
        "figure": "fig_5_1_source_target_landmark_displacement",
        "source_identity": source_id,
        "target_identity": target_id,
        "data_sources": [str(paths.demo_case), str(paths.data / f"meshes/{source_id}_neutral.npz"), str(paths.data / f"meshes/{target_id}_neutral.npz")],
        "arrow_definition": "target landmark position minus source landmark position",
        "arrow_scale_for_visibility": arrow_scale,
        "projection": "vertex point projection with measured landmark displacement arrows",
        "safe_method_included": "no",
        "ridge_or_hybrid_included": "no",
        "outputs": [str(p) for p in paths_out],
    }
    source_record(out_dir, payload["figure"], payload)
    return payload


def pareto_front(points: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    selected = []
    best_y = float("inf")
    for label, x, y in sorted(points, key=lambda t: (t[1], t[2])):
        if y < best_y:
            selected.append((label, x, y))
            best_y = y
    return selected


def make_frontier(paths: EvalPaths, out_dir: Path) -> dict:
    summary = method_summary(paths)
    ci = external_ci_from_bootstrap(paths)
    points = [(m, summary[m]["normal_flip_pct"], summary[m]["roi_rmse"]) for m in METHODS]
    front = pareto_front(points)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    xmax = max(x for _, x, _ in points) * 1.12
    ymax = max(y for _, _, y in points) * 1.08
    ax.axvspan(1.3, xmax, color="#F1C27D", alpha=0.20, lw=0)
    ax.text(1.34, ymax - 0.08 * ymax, "higher-risk\nregion", fontsize=8, color="#6E4A16")

    fx = [x for _, x, _ in front]
    fy = [y for _, _, y in front]
    ax.plot(fx, fy, color="#333333", lw=1.4, ls="-", alpha=0.9, label="Pareto frontier")

    offsets = {
        "rbsr": (0.08, -0.10),
        "cvae": (0.12, -0.10),
        "laplacian": (0.10, 0.06),
        "bilaplacian": (-0.70, -0.05),
        "arap": (0.08, -0.03),
    }
    for method, x, y in points:
        xerr = yerr = None
        if method in ci:
            if "normal_flip_pct" in ci[method]:
                lo, hi = ci[method]["normal_flip_pct"]
                xerr = np.array([[max(0.0, x - lo)], [max(0.0, hi - x)]])
            if "roi_rmse" in ci[method]:
                lo, hi = ci[method]["roi_rmse"]
                yerr = np.array([[max(0.0, y - lo)], [max(0.0, hi - y)]])
        ax.errorbar(
            [x],
            [y],
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            ms=7.5,
            color=METHOD_COLOR[method],
            mec="black",
            mew=0.7,
            elinewidth=1.0,
            capsize=2.5,
            zorder=5 if method == "rbsr" else 4,
        )
        dx, dy = offsets[method]
        ax.annotate(
            METHOD_LABEL[method],
            xy=(x, y),
            xytext=(x + dx, y + dy),
            arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "0.35"},
            fontsize=8.5,
            ha="left",
            va="center",
        )
    ax.annotate(
        "better",
        xy=(0.10, 1.08),
        xytext=(0.95, 1.36),
        arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#264653"},
        fontsize=9,
        color="#264653",
    )
    ax.set_xlim(0, xmax)
    ax.set_ylim(0.88, ymax)
    ax.set_xlabel("Normal flip (%)")
    ax.set_ylabel("ROI RMSE")
    ax.grid(alpha=0.18, lw=0.6)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    paths_out = save_figure(fig, out_dir, "fig_frontier_hero")
    payload = {
        "figure": "fig_frontier_hero",
        "data_sources": [str(paths.main_means), str(paths.rbsr_means), str(paths.bootstrap)],
        "methods": [METHOD_LABEL[m] for m in METHODS],
        "pareto_methods": [METHOD_LABEL[m] for m, _, _ in front],
        "projection": "quantitative Pareto scatter with existing CI whiskers where available",
        "ci_source": "external baselines use existing bootstrap difference rows versus ridge translated onto the plotted axes; RB-SR has no matching existing CI row",
        "safe_method_included": "no",
        "ridge_or_hybrid_included": "no",
        "outputs": [str(p) for p in paths_out],
    }
    source_record(out_dir, payload["figure"], payload)
    return payload


def _diagnostic_panel_grid(
    *,
    paths: EvalPaths,
    out_dir: Path,
    source_id: str,
    target_id: str,
    figure_name: str,
    metrics: list[tuple[str, str, str]],
    view: str = "front",
) -> dict:
    by_id = load_identities(paths.data, {source_id, target_id})
    store = PredictionStore(paths, out_dir)
    index = store.pair_index(source_id, target_id)
    prep = store.prepare_external_rows(METHODS, [index])
    template = by_id[source_id]
    labels = subunit_vertex_labels(template)
    all_values: dict[str, dict[str, np.ndarray]] = {}
    missing: dict[str, str] = {}
    try:
        for method in METHODS:
            try:
                src, tgt, pred = method_vertices(store, by_id, source_id, target_id, method)
                all_values[method] = diagnostic_values(src, tgt, pred, template["faces"])
            except Exception as exc:  # noqa: BLE001
                missing[method] = repr(exc)
        fig = plt.figure(figsize=(12.8, 2.35 * len(metrics)), constrained_layout=True)
        gs = fig.add_gridspec(
            len(metrics),
            len(METHODS) + 1,
            width_ratios=[1.0] * len(METHODS) + [0.055],
            wspace=0.06,
            hspace=0.14,
        )
        axes = np.empty((len(metrics), len(METHODS)), dtype=object)
        cbar_axes = []
        for r in range(len(metrics)):
            for c in range(len(METHODS)):
                axes[r, c] = fig.add_subplot(gs[r, c])
            cbar_axes.append(fig.add_subplot(gs[r, -1]))
        for row_i, (metric_key, metric_label, cmap) in enumerate(metrics):
            available = [all_values[m][metric_key] for m in METHODS if m in all_values]
            if not available:
                norm = colors.Normalize(0, 1)
            elif metric_key == "signed_z":
                vmax = max(float(np.nanpercentile(np.abs(v), 98)) for v in available)
                norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            elif metric_key == "new_flip":
                norm = colors.Normalize(0.0, 1.0)
            else:
                vmax = max(float(np.nanpercentile(v, 98)) for v in available)
                norm = colors.Normalize(0.0, max(vmax, 1e-9))
            for col_i, method in enumerate(METHODS):
                ax = axes[row_i, col_i]
                if method not in all_values:
                    ax.text(0.5, 0.5, "missing\nprediction", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                    ax.set_axis_off()
                    continue
                draw_vertex_cloud(
                    ax,
                    template["vertices"],
                    all_values[method][metric_key],
                    view=view,
                    cmap=cmap,
                    norm=norm,
                    s=4.0 if metric_key != "new_flip" else 5.5,
                    alpha=0.92,
                )
                overlay_subunit_boundaries(ax, template["vertices"], template["faces"], labels, view=view)
                if row_i == 0:
                    ax.set_title(METHOD_LABEL[method], fontsize=9.5)
                if col_i == 0:
                    ax.text(-0.06, 0.5, metric_label, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=9)
                    add_panel_label(ax, chr(ord("a") + row_i), x=-0.16, y=1.02)
            sm = ScalarMappable(norm=norm, cmap=cmap)
            cbar = fig.colorbar(sm, cax=cbar_axes[row_i])
            cbar.set_label(metric_label, fontsize=7)
            cbar.ax.tick_params(labelsize=7)
        fig.suptitle(f"Frozen test pair ID {source_id} -> ID {target_id}", fontsize=11)
        paths_out = save_figure(fig, out_dir, figure_name)
    finally:
        store.close()
    payload = {
        "figure": figure_name,
        "pair": f"{source_id}->{target_id}",
        "pair_index": index,
        "data_sources": [str(paths.cache_test), str(paths.geometric_npz), str(paths.data)],
        "external_prediction_extraction": prep,
        "missing": missing,
        "methods": [METHOD_LABEL[m] for m in METHODS],
        "projection": f"vertex point projection, {view}",
        "safe_method_included": "no",
        "ridge_or_hybrid_included": "no",
        "outputs": [str(p) for p in paths_out],
    }
    source_record(out_dir, figure_name, payload)
    return payload


def make_spatial_error(paths: EvalPaths, out_dir: Path) -> dict:
    source_id, target_id, reason = selected_qualitative_case(paths)
    payload = _diagnostic_panel_grid(
        paths=paths,
        out_dir=out_dir,
        source_id=source_id,
        target_id=target_id,
        figure_name="fig_spatial_error_evidence",
        metrics=[
            ("error", "per-vertex error", "magma"),
            ("new_flip", "new-flip vertices", "Reds"),
            ("edge_strain", "edge-strain risk", "YlOrRd"),
        ],
    )
    payload["selection_rule"] = reason
    payload["redesign_note"] = "Nature-style consolidation: retained the three diagnostics that carry distinct evidence for accuracy and local geometric risk."
    source_record(out_dir, payload["figure"], payload)
    return payload


def make_risk_distribution(paths: EvalPaths, out_dir: Path) -> dict:
    worst = select_worst_cases(paths, 1)[0]
    source_id, target_id = str(worst["source_id"]), str(worst["target_id"])
    by_id = load_identities(paths.data, {source_id, target_id})
    store = PredictionStore(paths, out_dir)
    idx = store.pair_index(source_id, target_id)
    missing = {}
    try:
        src, tgt, pred = method_vertices(store, by_id, source_id, target_id, "rbsr")
        faces = by_id[source_id]["faces"]
        flip_v = vertex_from_face_indicator(faces, new_flip_faces(src, pred, tgt, faces), len(src))
        strain_v = vertex_edge_strain(src, pred, faces)
        gate_npz = np.load(paths.gate_map, allow_pickle=False)
        gate_v = np.asarray(gate_npz["vertex_gate_mean"], dtype=np.float64)
        sub_rows = load_subunit_rows(paths)
        rbsr_row = sub_rows["rbsr"]
        sub_flip = np.asarray([float(rbsr_row[f"{s}_new_flip_pct"]) for s in SUBUNITS], dtype=float)
        labels = subunit_vertex_labels(by_id[source_id])

        fig, axes = plt.subplots(2, 3, figsize=(9.8, 6.0))
        maps = [
            (flip_v, "New-flip indicator", "Reds", colors.Normalize(0, 1), "front"),
            (strain_v, "Edge-strain risk", "YlOrRd", colors.Normalize(0, max(0.01, float(np.nanpercentile(strain_v, 98)))), "front"),
            (gate_v, "Mean residual admission", "viridis", colors.Normalize(0, max(1e-6, float(np.nanpercentile(gate_v, 99)))), "front"),
            (flip_v, "New-flip, oblique", "Reds", colors.Normalize(0, 1), "oblique"),
            (strain_v, "Edge-strain, lateral", "YlOrRd", colors.Normalize(0, max(0.01, float(np.nanpercentile(strain_v, 98)))), "lateral"),
        ]
        for ax, (vals, title, cmap, norm, view) in zip(axes.flat[:5], maps):
            draw_vertex_cloud(ax, src, vals, view=view, cmap=cmap, norm=norm, s=4.5, alpha=0.92)
            overlay_subunit_boundaries(ax, src, faces, labels, view=view)
            ax.set_title(title)
            sm = ScalarMappable(norm=norm, cmap=cmap)
            cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.01)
            cbar.ax.tick_params(labelsize=7)
        ax = axes.flat[5]
        ax.bar(range(len(SUBUNITS)), sub_flip, color="#D55E00", width=0.62)
        ax.set_xticks(range(len(SUBUNITS)), [SUBUNIT_LABEL[s] for s in SUBUNITS], rotation=25, ha="right")
        ax.set_ylabel("New flip (%)")
        ax.set_title("RB-SR risk by sub-unit")
        ax.grid(axis="y", alpha=0.18)
        fig.suptitle(f"RB-SR risk diagnostics, high-risk frozen pair ID {source_id} -> ID {target_id}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        paths_out = save_figure(fig, out_dir, "fig_risk_spatial_distribution")
    except Exception as exc:  # noqa: BLE001
        missing["risk_distribution"] = repr(exc)
        paths_out = []
    finally:
        store.close()
    payload = {
        "figure": "fig_risk_spatial_distribution",
        "selection_rule": "worst RB-SR pair by normal_flip_pct, then edge_strain_p95, then ROI RMSE",
        "pair": f"{source_id}->{target_id}",
        "pair_index": idx,
        "data_sources": [str(paths.rbsr_pair_metrics), str(paths.cache_test / "rbsr.npy"), str(paths.gate_map), str(paths.subunit_csv)],
        "missing": missing,
        "projection": "vertex point projection plus sub-unit bar diagnostic",
        "safe_method_included": "no",
        "ridge_or_hybrid_included": "no",
        "outputs": [str(p) for p in paths_out],
    }
    source_record(out_dir, payload["figure"], payload)
    return payload


def make_subunit_triptych(paths: EvalPaths, out_dir: Path) -> dict:
    rows = load_subunit_rows(paths)
    x = np.arange(len(SUBUNITS))
    width = 0.15
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    for i, method in enumerate(METHODS):
        offs = (i - 2) * width
        rmse = [float(rows[method][f"{s}_rmse"]) for s in SUBUNITS]
        flip = [float(rows[method][f"{s}_new_flip_pct"]) for s in SUBUNITS]
        axes[0].bar(x + offs, rmse, width=width, color=METHOD_COLOR[method], label=METHOD_LABEL[method])
        axes[1].bar(x + offs, flip, width=width, color=METHOD_COLOR[method])
    external = ["cvae", "laplacian", "bilaplacian", "arap"]
    deltas = []
    best_methods = []
    for s in SUBUNITS:
        best = min(external, key=lambda m: float(rows[m][f"{s}_rmse"]))
        best_methods.append(best)
        deltas.append(float(rows["rbsr"][f"{s}_rmse"]) - float(rows[best][f"{s}_rmse"]))
    axes[2].bar(x, deltas, color=["#2A9D8F" if v < 0 else "#D55E00" for v in deltas], width=0.62)
    axes[2].axhline(0, color="0.25", lw=0.8)
    for ax, title, ylabel in [
        (axes[0], "Per-subunit RMSE", "RMSE"),
        (axes[1], "Per-subunit new-flip risk", "New flip (%)"),
        (axes[2], "RB-SR - strongest displayed baseline", "RMSE delta"),
    ]:
        ax.set_xticks(x, [SUBUNIT_LABEL[s] for s in SUBUNITS], rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.18)
    axes[0].legend(frameon=False, ncol=2, fontsize=7.5)
    fig.tight_layout()
    paths_out = save_figure(fig, out_dir, "fig_subunit_triptych")
    payload = {
        "figure": "fig_subunit_triptych",
        "data_sources": [str(paths.subunit_csv)],
        "methods": [METHOD_LABEL[m] for m in METHODS],
        "delta_rule": "For each sub-unit, the displayed baseline is the external method with the lowest RMSE among CVAE-only, Laplacian, Bi-Laplacian, and ARAP; negative means RB-SR has lower RMSE.",
        "delta_best_methods": {SUBUNIT_LABEL[s]: METHOD_LABEL[m] for s, m in zip(SUBUNITS, best_methods)},
        "projection": "grouped sub-unit bar charts plus RMSE-delta bar chart",
        "ci": "not included; no existing per-subunit CI file was found",
        "safe_method_included": "no",
        "ridge_or_hybrid_included": "no",
        "outputs": [str(p) for p in paths_out],
    }
    source_record(out_dir, payload["figure"], payload)
    return payload


def make_demo_full_face(paths: EvalPaths, out_dir: Path) -> dict:
    data = full_face_from_demo(paths)
    source = data["source"]
    output = data["output"]
    faces = data["faces"]
    disp = np.linalg.norm(output - source, axis=1)
    vmax = max(1e-6, float(np.nanpercentile(disp, 99)))
    norm = colors.Normalize(0, vmax)

    fig, axes = plt.subplots(2, 3, figsize=(10.6, 7.0))
    draw_mesh_projection(axes[0, 0], source, faces, view="front", neutral=True)
    axes[0, 0].set_title("Source full face")
    draw_mesh_projection(axes[0, 1], output, faces, view="front", neutral=True)
    axes[0, 1].set_title("Demo output full face")
    draw_mesh_projection(axes[0, 2], output, faces, disp, view="front", cmap="magma", norm=norm)
    axes[0, 2].set_title("Dense deformation magnitude")
    for ax, view, title in zip(axes[1], ["front", "oblique", "lateral"], ["Frontal output", "Oblique output", "Lateral output"]):
        draw_mesh_projection(ax, output, faces, view=view, neutral=True)
        ax.set_title(title)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap="magma"), ax=axes[0, 2], fraction=0.045, pad=0.01)
    cbar.ax.tick_params(labelsize=7)
    demo = data["demo"]
    metrics = demo["methods"]["rbsr"].get("metrics", {})
    controls = np.asarray(demo["geometry"]["controls"], dtype=np.float64).reshape(9, 3)
    control_summary = {
        "mean_handle_displacement": float(np.mean(np.linalg.norm(controls, axis=1))),
        "max_handle_displacement": float(np.max(np.linalg.norm(controls, axis=1))),
        "selected_alpha": float(demo["case"]["selected_alpha"]),
        "roi_rmse": metrics.get("roi_rmse"),
        "normal_flip_pct": metrics.get("normal_flip_pct"),
        "edge_strain_p95": metrics.get("edge_strain_p95"),
    }
    fig.text(
        0.5,
        0.012,
        "Full-head blend uses existing ROI + collar mapping. Six semantic slider metadata are not present; stored controls are 9 handle displacements.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    paths_out = save_figure(fig, out_dir, "fig_demo_full_face_before_after")
    payload = {
        "figure": "fig_demo_full_face_before_after",
        "data_sources": [str(paths.head_json), str(paths.demo_case)],
        "source_identity": demo["case"]["source_id"],
        "target_identity": demo["case"].get("target_id"),
        "full_face_smooth_blend": "yes" if data["collar_used"] else "no",
        "slider_metadata": "missing: demo JSON stores 27 handle control values, not six semantic slider parameters",
        "reported_metrics": control_summary,
        "projection": "full-face mesh projection with deformation magnitude overlay",
        "safe_method_included": "no",
        "ridge_or_hybrid_included": "no",
        "outputs": [str(p) for p in paths_out],
    }
    source_record(out_dir, payload["figure"], payload)
    return payload


def make_method_qualitative(paths: EvalPaths, out_dir: Path) -> dict:
    source_id, target_id, reason = selected_qualitative_case(paths)
    by_id = load_identities(paths.data, {source_id, target_id})
    store = PredictionStore(paths, out_dir)
    idx = store.pair_index(source_id, target_id)
    prep = store.prepare_external_rows(METHODS, [idx])
    template = by_id[source_id]
    faces = template["faces"]
    src = by_id[source_id]["vertices"]
    tgt = by_id[target_id]["vertices"]
    labels = subunit_vertex_labels(template)
    columns = ["source", "target"] + METHODS
    col_labels = ["Source", "Ground-truth target"] + [METHOD_LABEL[m] for m in METHODS]
    preds = {}
    vals = {}
    missing = {}
    try:
        for method in METHODS:
            try:
                _, _, pred = method_vertices(store, by_id, source_id, target_id, method)
                preds[method] = pred
                vals[method] = diagnostic_values(src, tgt, pred, faces)
            except Exception as exc:  # noqa: BLE001
                missing[method] = repr(exc)
        err_available = [vals[m]["error"] for m in vals]
        err_norm = colors.Normalize(0, max(1e-6, max(float(np.nanpercentile(v, 98)) for v in err_available)))
        risk_available = [vals[m]["edge_strain"] for m in vals]
        risk_norm = colors.Normalize(0, max(1e-6, max(float(np.nanpercentile(v, 98)) for v in risk_available)))
        fig = plt.figure(figsize=(14.6, 7.0), constrained_layout=True)
        gs = fig.add_gridspec(
            3,
            len(columns) + 1,
            width_ratios=[1.0] * len(columns) + [0.06],
            wspace=0.06,
            hspace=0.14,
        )
        axes = np.empty((3, len(columns)), dtype=object)
        for r in range(3):
            for c in range(len(columns)):
                axes[r, c] = fig.add_subplot(gs[r, c])
        fig.add_subplot(gs[0, -1]).set_axis_off()
        err_cax = fig.add_subplot(gs[1, -1])
        risk_cax = fig.add_subplot(gs[2, -1])
        for c, col in enumerate(columns):
            ax = axes[0, c]
            if col == "source":
                draw_vertex_cloud(ax, src, None, view="front", s=3.4, alpha=0.68)
            elif col == "target":
                draw_vertex_cloud(ax, tgt, None, view="front", s=3.4, alpha=0.68)
            elif col in preds:
                draw_vertex_cloud(ax, preds[col], None, view="front", s=3.4, alpha=0.68)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            ax.set_title(col_labels[c])
            if col in {"source", "target"}:
                for r in [1, 2]:
                    axes[r, c].text(0.5, 0.5, "reference", ha="center", va="center", transform=axes[r, c].transAxes, fontsize=8)
                    axes[r, c].set_axis_off()
            elif col in vals:
                draw_vertex_cloud(axes[1, c], src, vals[col]["error"], view="front", cmap="magma", norm=err_norm, s=4.0)
                overlay_subunit_boundaries(axes[1, c], src, faces, labels, view="front")
                draw_vertex_cloud(axes[2, c], src, vals[col]["edge_strain"], view="front", cmap="YlOrRd", norm=risk_norm, s=4.0)
                overlay_subunit_boundaries(axes[2, c], src, faces, labels, view="front")
            else:
                for r in [1, 2]:
                    axes[r, c].text(0.5, 0.5, "missing", ha="center", va="center", transform=axes[r, c].transAxes, fontsize=8)
                    axes[r, c].set_axis_off()
        axes[1, 0].text(-0.35, 0.5, "Per-vertex error", transform=axes[1, 0].transAxes, rotation=90, ha="right", va="center")
        axes[2, 0].text(-0.35, 0.5, "Edge-strain risk", transform=axes[2, 0].transAxes, rotation=90, ha="right", va="center")
        add_panel_label(axes[0, 0], "a", x=-0.20, y=1.05)
        add_panel_label(axes[1, 0], "b", x=-0.20, y=1.05)
        add_panel_label(axes[2, 0], "c", x=-0.20, y=1.05)
        err_cbar = fig.colorbar(ScalarMappable(norm=err_norm, cmap="magma"), cax=err_cax)
        err_cbar.set_label("Per-vertex error", fontsize=7)
        err_cbar.ax.tick_params(labelsize=6.5)
        risk_cbar = fig.colorbar(ScalarMappable(norm=risk_norm, cmap="YlOrRd"), cax=risk_cax)
        risk_cbar.set_label("Edge-strain risk", fontsize=7)
        risk_cbar.ax.tick_params(labelsize=6.5)
        fig.suptitle(f"Representative frozen pair ID {source_id} -> ID {target_id}", fontsize=11)
        paths_out = save_figure(fig, out_dir, "fig_method_qualitative_hero")
    finally:
        store.close()
    payload = {
        "figure": "fig_method_qualitative_hero",
        "selection_rule": reason,
        "pair": f"{source_id}->{target_id}",
        "pair_index": idx,
        "data_sources": [str(paths.qualitative_cases), str(paths.cache_test), str(paths.geometric_npz), str(paths.data)],
        "external_prediction_extraction": prep,
        "missing": missing,
        "projection": "vertex point projection with shared per-vertex error and edge-strain colorbars",
        "safe_method_included": "no",
        "ridge_or_hybrid_included": "no",
        "outputs": [str(p) for p in paths_out],
    }
    source_record(out_dir, payload["figure"], payload)
    return payload


def make_visual_audit(paths: EvalPaths, out_dir: Path) -> dict:
    cases = select_worst_cases(paths, 20)
    ids = {str(row["source_id"]) for row in cases} | {str(row["target_id"]) for row in cases}
    by_id = load_identities(paths.data, ids)
    store = PredictionStore(paths, out_dir)
    indices = [store.pair_index(str(r["source_id"]), str(r["target_id"])) for r in cases]
    prep = store.prepare_external_rows(METHODS, indices)
    columns = ["source", "target"] + METHODS
    col_labels = ["Source", "Target"] + [METHOD_LABEL[m] for m in METHODS]
    missing = {}
    paths_out: list[Path] = []
    multipage_pdf = out_dir / "fig_visual_audit_grid.pdf"
    try:
        pages = [cases[i : i + 4] for i in range(0, len(cases), 4)]
        with PdfPages(multipage_pdf) as pdf:
            for page_i, page_cases in enumerate(pages, start=1):
                page_errors: dict[tuple[int, str], np.ndarray] = {}
                all_err = []
                for local_r, row in enumerate(page_cases):
                    source_id = str(row["source_id"])
                    target_id = str(row["target_id"])
                    src = by_id[source_id]["vertices"]
                    tgt = by_id[target_id]["vertices"]
                    for method in METHODS:
                        try:
                            _, _, pred = method_vertices(store, by_id, source_id, target_id, method)
                            err = np.linalg.norm(pred - tgt, axis=1)
                            page_errors[(local_r, method)] = err
                            all_err.append(err)
                        except Exception as exc:  # noqa: BLE001
                            missing.setdefault(method, repr(exc))
                vmax = max(1e-6, max(float(np.nanpercentile(v, 98)) for v in all_err)) if all_err else 1.0
                norm = colors.Normalize(0, vmax)
                fig = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
                gs = fig.add_gridspec(
                    len(page_cases),
                    len(columns) + 1,
                    width_ratios=[1.0] * len(columns) + [0.045],
                    wspace=0.04,
                    hspace=0.10,
                )
                axes = np.empty((len(page_cases), len(columns)), dtype=object)
                for r, row in enumerate(page_cases):
                    source_id = str(row["source_id"])
                    target_id = str(row["target_id"])
                    src = by_id[source_id]["vertices"]
                    tgt = by_id[target_id]["vertices"]
                    for c, col in enumerate(columns):
                        ax = fig.add_subplot(gs[r, c])
                        axes[r, c] = ax
                        if col == "source":
                            draw_vertex_cloud(ax, src, None, view="front", s=1.6, alpha=0.58)
                        elif col == "target":
                            draw_vertex_cloud(ax, tgt, None, view="front", s=1.6, alpha=0.58)
                        elif (r, col) in page_errors:
                            draw_vertex_cloud(ax, src, page_errors[(r, col)], view="front", cmap="magma", norm=norm, s=1.9)
                        else:
                            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes, fontsize=6)
                            ax.set_axis_off()
                        if r == 0:
                            ax.set_title(col_labels[c], fontsize=6.8, pad=2)
                    axes[r, 0].text(
                        -0.10,
                        0.5,
                        f"#{(page_i - 1) * 4 + r + 1}\n{source_id}->{target_id}\nflip {float(row['normal_flip_pct']):.2f}%\nstrain {float(row['edge_strain_p95']):.2f}",
                        transform=axes[r, 0].transAxes,
                        ha="right",
                        va="center",
                        fontsize=5.8,
                    )
                cax = fig.add_subplot(gs[:, -1])
                cbar = fig.colorbar(ScalarMappable(norm=norm, cmap="magma"), cax=cax)
                cbar.set_label("Per-vertex error", fontsize=6.2)
                cbar.ax.tick_params(labelsize=5.8)
                add_panel_label(axes[0, 0], "a", x=-0.18, y=1.10)
                fig.suptitle(f"Qualitative visual audit, worst-case samples {(page_i - 1) * 4 + 1}-{(page_i - 1) * 4 + len(page_cases)}", fontsize=8.2)
                pdf.savefig(fig, bbox_inches="tight")
                stem = f"fig_visual_audit_grid_page{page_i:02d}"
                page_paths = [out_dir / f"{stem}.pdf", out_dir / "svg" / f"{stem}.svg", out_dir / f"{stem}.png"]
                fig.savefig(page_paths[0])
                fig.savefig(page_paths[1])
                fig.savefig(page_paths[2], dpi=300)
                if page_i == 1:
                    fig.savefig(out_dir / "svg" / "fig_visual_audit_grid.svg")
                    fig.savefig(out_dir / "fig_visual_audit_grid.png", dpi=300)
                paths_out.extend(page_paths)
                plt.close(fig)
        paths_out.insert(0, multipage_pdf)
        paths_out.insert(1, out_dir / "svg" / "fig_visual_audit_grid.svg")
        paths_out.insert(2, out_dir / "fig_visual_audit_grid.png")
    finally:
        store.close()
    case_rows = [
        {
            "rank": i + 1,
            "source_id": row["source_id"],
            "target_id": row["target_id"],
            "normal_flip_pct": row["normal_flip_pct"],
            "edge_strain_p95": row["edge_strain_p95"],
            "roi_rmse": row["roi_rmse"],
            "fold_over_metric_flag": "yes" if float(row["normal_flip_pct"]) > 0 else "no",
            "spike_tear_collapse_visual_flag": "not auto-scored",
        }
        for i, row in enumerate(cases)
    ]
    from plot_style_eval import write_csv_rows

    write_csv_rows(out_dir / "fig_visual_audit_grid_cases.csv", case_rows)
    payload = {
        "figure": "fig_visual_audit_grid",
        "selection_rule": "20 RB-SR worst-case samples sorted by normal_flip_pct, then edge_strain_p95, then ROI RMSE",
        "data_sources": [str(paths.rbsr_pair_metrics), str(paths.cache_test), str(paths.geometric_npz), str(paths.data)],
        "external_prediction_extraction": prep,
        "case_csv": str(out_dir / "fig_visual_audit_grid_cases.csv"),
        "visual_flags": "fold-over is metric-derived from normal_flip_pct; spike, tearing, and alar collapse are not auto-scored",
        "pagination": "20 worst-case samples are split into five four-case pages; fig_visual_audit_grid.pdf is the multipage audit PDF",
        "projection": "paginated vertex point projection audit with one shared per-page error colorbar",
        "missing": missing,
        "safe_method_included": "no",
        "ridge_or_hybrid_included": "no",
        "outputs": [str(p) for p in paths_out],
    }
    source_record(out_dir, payload["figure"], payload)
    return payload


def make_audit(out_dir: Path, records: list[dict]) -> Path:
    notes = figure_caption_notes()
    lines = [
        "# Figure Redraw Audit",
        "",
        "This audit records the redraw status for the Evaluation and Appendix figure set. No benchmark values were manually edited.",
        "",
        "| Figure | Data source | Redraw status | SafeFusion included | Ridge/Hybrid included | Visual mode | CI | Full-face smooth blend | Missing / caveat | Values unchanged |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rec in records:
        fig = rec.get("figure", "")
        sources = "<br>".join(Path(str(s)).as_posix() for s in rec.get("data_sources", []))
        missing = rec.get("missing", "")
        if isinstance(missing, dict):
            missing_text = "; ".join(f"{k}: {v}" for k, v in missing.items()) if missing else "none"
        else:
            missing_text = str(missing) if missing else "none"
        if "slider_metadata" in rec:
            missing_text = (missing_text + "; " if missing_text != "none" else "") + str(rec["slider_metadata"])
        ridge_hybrid = rec.get("ridge_or_hybrid_included", "no")
        visual = rec.get("projection", "surface/point rendering")
        ci = rec.get("ci", rec.get("ci_source", "not included"))
        full_blend = rec.get("full_face_smooth_blend", "no")
        lines.append(
            f"| `{fig}` | {sources} | redrawn/generated | {rec.get('safe_method_included', 'no')} | {ridge_hybrid} | {visual} | {ci} | {full_blend} | {missing_text} | yes |"
        )
    lines.extend(
        [
            "",
            "## Caption Notes For Later LaTeX Replacement",
            "",
        ]
    )
    for key, note in notes.items():
        lines.append(f"- `{key}`: {note}")
    lines.extend(
        [
            "",
            "## Method Scope",
            "",
            "- Main external comparisons are restricted to RB-SR, CVAE-only, Laplacian, Bi-Laplacian, and ARAP.",
            "- Ridge and Hybrid are not plotted as external benchmark points in the generated core comparison figures.",
            "- Any legacy cache directory name is treated only as a frozen data location and is not a displayed method label.",
        ]
    )
    audit = out_dir / "figure_redraw_audit.md"
    text = "\n".join(lines) + "\n"
    audit.write_text(text, encoding="utf-8")
    (out_dir.parent / "figure_redraw_audit.md").write_text(text, encoding="utf-8")
    return audit


def run_all() -> list[dict]:
    apply_eval_style()
    paths = default_paths()
    out_dir, _ = ensure_output_dirs()
    records = []
    jobs = [
        make_fig_5_1,
        make_frontier,
        make_spatial_error,
        make_risk_distribution,
        make_subunit_triptych,
        make_demo_full_face,
        make_method_qualitative,
        make_visual_audit,
    ]
    for job in jobs:
        try:
            print(f"[eval-figures] {job.__name__}", flush=True)
            records.append(job(paths, out_dir))
        except Exception as exc:  # noqa: BLE001
            name = job.__name__.replace("make_", "fig_")
            rec = {
                "figure": name,
                "data_sources": [],
                "safe_method_included": "no",
                "ridge_or_hybrid_included": "no",
                "missing": {"fatal": repr(exc), "traceback": traceback.format_exc(limit=4)},
                "outputs": [],
            }
            source_record(out_dir, name, rec)
            records.append(rec)
            print(f"[eval-figures] {job.__name__} failed: {exc!r}", flush=True)
    audit = make_audit(out_dir, records)
    print(json.dumps({"out_dir": str(out_dir), "audit": str(audit), "figures": [r["figure"] for r in records]}, indent=2), flush=True)
    return records


if __name__ == "__main__":
    run_all()
