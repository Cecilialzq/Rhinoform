from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "strict_sparse_dense_diagnostics_final_base"
AUG = ROOT / "outputs" / "strict_sparse_dense_diagnostics_final_augmented"
OUT = ROOT / "figures" / "nature_diagnostics_final"
OUT.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "ridge": "#484878",
    "rbsr": "#D24B40",
    "oracle": "#2E9E44",
    "projection": "#0F4D92",
    "cvae": "#7884B4",
    "hybrid": "#B4C0E4",
    "classical": "#A8A8A8",
    "aug": "#E28E2C",
    "bad": "#B64342",
    "neutral": "#606060",
    "light": "#D8D8D8",
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_fig(fig: plt.Figure, stem: str) -> None:
    for ext, kwargs in {
        "svg": {},
        "pdf": {},
        "png": {"dpi": 300},
        "tiff": {"dpi": 600},
    }.items():
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def panel_label(ax, label: str, x: float = -0.16, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontweight="bold", fontsize=8.5,
            ha="left", va="bottom")


def method_color(name: str) -> str:
    if "Projection" in name or "projection" in name:
        return PALETTE["projection"]
    if "oracle" in name.lower():
        return PALETTE["oracle"]
    if "RBSR" in name:
        return PALETTE["rbsr"]
    if "CVAE" in name:
        return PALETTE["cvae"]
    if "Hybrid" in name:
        return PALETTE["hybrid"]
    if "Ridge" in name:
        return PALETTE["ridge"]
    return PALETTE["classical"]


def compact_method(name: str) -> str:
    mapping = {
        "Rigid/source_copy_strict": "Source copy",
        "Laplacian_strict": "Laplacian",
        "ARAP_iter3_strict": "ARAP",
        "Ridge_strict": "Ridge",
        "Best_CVAE_z4_w512_L3": "CVAE",
        "Hybrid_strict_alpha0.3": "Hybrid",
        "RBSR_predicted_K128_with_root": "RBSR K128",
        "RBSR_oracle_admission_K128_with_root": "RBSR oracle\nadmission",
        "RBSR_projection_upper_K128_with_root": "Projection\nupper bound",
        "RBSR_validation_selected_K128_ridge_none_with_root": "RBSR selected",
        "RBSR_validation_selected_K64_extra_trees_subunit_soft_with_root": "Aug. selected",
        "Ridge_augmented_f3": "Ridge aug.",
    }
    return mapping.get(name, name.replace("_", " "))


def add_reference_line(ax, y: float, label: str, color: str = "#606060") -> None:
    ax.axhline(y, color=color, lw=0.8, ls="--", zorder=0)
    ax.text(0.98, y, label, transform=ax.get_yaxis_transform(), ha="right",
            va="bottom", fontsize=6, color=color)


def fig_main_evidence() -> None:
    final_base = read_csv(BASE / "08_final_strict_analysis" / "final_strict_table.csv")
    basis = read_csv(BASE / "08_final_strict_analysis" / "basis_gap_root_included.csv")
    paired = read_json(BASE / "08_final_strict_analysis" / "paired_tests.json")

    methods = [
        "Laplacian_strict", "ARAP_iter3_strict", "Ridge_strict",
        "Best_CVAE_z4_w512_L3", "Hybrid_strict_alpha0.3",
        "RBSR_validation_selected_K128_ridge_none_with_root",
        "RBSR_oracle_admission_K128_with_root",
        "RBSR_projection_upper_K128_with_root",
    ]
    plot_df = final_base[final_base["method"].isin(methods)].copy()
    plot_df["label"] = plot_df["method"].map(compact_method)
    plot_df["order"] = plot_df["method"].map({m: i for i, m in enumerate(methods)})
    plot_df = plot_df.sort_values("order")

    fig = plt.figure(figsize=(7.15, 4.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.38, 1.3, 1.05], wspace=0.48)

    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["strict_free_rmse"], color=[method_color(m) for m in plot_df["method"]],
            edgecolor="black", lw=0.35)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Strict free-vertex RMSE")
    ax.set_title("Identity-disjoint test")
    ax.set_xlim(0, max(plot_df["strict_free_rmse"]) * 1.08)
    ax.axvline(float(final_base.loc[final_base["method"] == "Ridge_strict", "strict_free_rmse"].iloc[0]),
               color=PALETTE["ridge"], lw=0.9, ls="--")
    ax.text(float(final_base.loc[final_base["method"] == "Ridge_strict", "strict_free_rmse"].iloc[0]),
            -0.55, "Ridge", fontsize=6, color=PALETTE["ridge"], ha="right", va="bottom")
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    sub = basis[basis["kind"].isin(["predicted_coefficients", "projection_upper_bound"])].copy()
    for kind, label, color in [
        ("predicted_coefficients", "Predicted RBSR", PALETTE["rbsr"]),
        ("projection_upper_bound", "Projection upper bound", PALETTE["projection"]),
    ]:
        g = sub[sub["kind"] == kind].sort_values("basis")
        ax.plot(g["basis"], g["strict_free_rmse"], marker="o", ms=3.2, lw=1.3,
                color=color, label=label)
    ridge = float(final_base.loc[final_base["method"] == "Ridge_strict", "strict_free_rmse"].iloc[0])
    ax.axhline(ridge, color=PALETTE["ridge"], lw=1.0, ls="--", label="Ridge")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
    ax.get_xaxis().set_major_formatter(lambda x, pos: f"{int(x)}")
    ax.set_xlabel("Residual basis K")
    ax.set_ylabel("Strict free-vertex RMSE")
    ax.set_title("Capacity versus released gain")
    ax.legend(loc="lower left", fontsize=6, handlelength=1.8)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    tests = [
        ("Predicted", "ridge_minus_rbsr_predicted"),
        ("Soft gate", "ridge_minus_rbsr_legal_soft_admission"),
        ("Oracle adm.", "ridge_minus_rbsr_oracle_admission"),
        ("Selected", "ridge_minus_rbsr_validation_selected"),
    ]
    rows = []
    for label, key in tests:
        if key in paired:
            rows.append((label, paired[key]))
    if not rows:
        rows = [("Predicted", paired["ridge_minus_rbsr_predicted"]),
                ("Selected", paired["ridge_minus_rbsr_validation_selected"])]
    ypos = np.arange(len(rows))
    means = np.array([r[1]["mean_a_minus_b"] for r in rows])
    lows = np.array([r[1]["ci95_low"] for r in rows])
    highs = np.array([r[1]["ci95_high"] for r in rows])
    ax.errorbar(means, ypos, xerr=[means - lows, highs - means], fmt="o",
                color=PALETTE["rbsr"], ecolor=PALETTE["neutral"], elinewidth=1.0, capsize=2.5)
    ax.axvline(0, color="black", lw=0.8)
    pad = max(0.002, float(np.max(highs) - np.min(lows)) * 0.25)
    ax.set_xlim(min(0, float(np.min(lows)) - pad), float(np.max(highs)) + pad)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Paired gain over Ridge\n(Ridge - method)")
    ax.set_title("Bootstrap 95% CI")
    panel_label(ax, "c")
    save_fig(fig, "fig1_main_rbsr_evidence")


def fig_rbsr_mechanism() -> None:
    basis = read_csv(BASE / "08_final_strict_analysis" / "basis_gap_root_included.csv")
    coeff = read_csv(BASE / "08_final_strict_analysis" / "coefficient_prediction_diagnostics.csv")
    feature = read_csv(AUG / "08_final_strict_analysis" / "feature_ablation.csv")
    lc = read_csv(BASE / "06_rbsr_learning_curve" / "learning_curve_metrics.csv")
    capacity = read_json(BASE / "08_final_strict_analysis" / "residual_basis_capacity_audit.json")["regions"]

    fig = plt.figure(figsize=(7.15, 6.2))
    gs = fig.add_gridspec(2, 2, wspace=0.38, hspace=0.45)

    ax = fig.add_subplot(gs[0, 0])
    energy_rows = []
    for r in capacity:
        for k in [2, 4, 8, 16, 32, 64, 128]:
            energy_rows.append({
                "region": r["region"],
                "K": k,
                "energy": r[f"train_residual_energy_explained_k{k}"],
            })
    e = pd.DataFrame(energy_rows)
    regions = ["root", "dorsum", "tip", "alar_left", "alar_right", "alar"]
    mat = e.pivot(index="region", columns="K", values="energy").loc[regions]
    im = ax.imshow(mat.values, aspect="auto", vmin=0.25, vmax=1.0,
                   cmap=LinearSegmentedColormap.from_list("energy", ["#F7F7F7", "#B4C0E4", "#0F4D92"]))
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels([x.replace("_", " ") for x in mat.index])
    ax.set_xlabel("Basis K")
    ax.set_title("Training residual energy explained")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Fraction")
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    coeff = coeff[coeff["predictor"] == "ridge"].copy()
    coeff = coeff[coeff["region"].isin(regions)]
    ax.bar(np.arange(len(coeff)), coeff["coeff_r2_mean"], color=PALETTE["ridge"],
           edgecolor="black", lw=0.35)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(coeff)))
    ax.set_xticklabels([x.replace("_", " ") for x in coeff["region"]], rotation=30, ha="right")
    ax.set_ylabel("Mean coefficient R²")
    ax.set_title("Legal inputs poorly predict coefficients")
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    rows = feature[feature["strict_free_rmse"].notna()].copy()
    rows = rows.dropna(subset=["feature_mode"])
    rows = rows[rows["feature_mode"].isin(["f0", "f1", "f2", "f3"])]
    colors = [PALETTE["rbsr"], PALETTE["aug"], PALETTE["aug"], PALETTE["bad"]]
    ax.bar(rows["feature_mode"], rows["strict_free_rmse"], color=colors[:len(rows)],
           edgecolor="black", lw=0.35)
    add_reference_line(ax, 1.076862096786499, "Ridge")
    ax.set_ylabel("Strict free-vertex RMSE")
    ax.set_xlabel("RBSR input feature mode")
    ax.set_title("Augmented features degrade prediction")
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    for method, label, color in [
        ("ridge_strict", "Ridge", PALETTE["ridge"]),
        ("rbsr_region_oracle", "RBSR base", PALETTE["rbsr"]),
    ]:
        g = lc[lc["method"] == method].sort_values("x_value")
        ax.plot(g["x_value"], g["strict_free_rmse"], marker="o", ms=3.2, lw=1.3,
                color=color, label=label)
    ax.set_xlabel("Training pairs")
    ax.set_ylabel("Strict free-vertex RMSE")
    ax.set_title("Learning curve")
    ax.legend(fontsize=6)
    panel_label(ax, "d")
    save_fig(fig, "fig2_rbsr_mechanism_diagnostics")


def fig_region_heatmap_and_admission() -> None:
    final_base = read_csv(BASE / "08_final_strict_analysis" / "final_strict_table.csv")
    final_aug = read_csv(AUG / "08_final_strict_analysis" / "final_strict_table.csv")
    methods = [
        "Ridge_strict",
        "RBSR_predicted_K128_with_root",
        "RBSR_oracle_admission_K128_with_root",
        "RBSR_projection_upper_K128_with_root",
    ]
    regions = ["rmse_root", "rmse_dorsum", "rmse_tip", "rmse_alar_left", "rmse_alar_right"]

    fig = plt.figure(figsize=(7.15, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.22, 1.0], wspace=0.58)
    ax = fig.add_subplot(gs[0, 0])
    hdf = final_base[final_base["method"].isin(methods)].copy()
    hdf = hdf.set_index("method").loc[methods, regions]
    labels_y = [compact_method(x) for x in hdf.index]
    labels_x = ["root", "dorsum", "tip", "alar L", "alar R"]
    im = ax.imshow(hdf.values, aspect="auto", cmap=LinearSegmentedColormap.from_list("rmse", ["#F7F7F7", "#E4CCD8", "#B64342"]))
    ax.set_xticks(np.arange(len(labels_x)))
    ax.set_xticklabels(labels_x, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels_y)))
    ax.set_yticklabels(labels_y)
    for i in range(hdf.shape[0]):
        for j in range(hdf.shape[1]):
            ax.text(j, i, f"{hdf.values[i, j]:.2f}", ha="center", va="center", fontsize=5.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.015)
    cb.set_label("Region RMSE")
    ax.set_title("Region-wise strict error")
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    comp = pd.DataFrame([
        {"method": "Ridge", "rmse": final_base.loc[final_base["method"] == "Ridge_strict", "strict_free_rmse"].iloc[0]},
        {"method": "Base RBSR", "rmse": final_base.loc[final_base["method"] == "RBSR_validation_selected_K128_ridge_none_with_root", "strict_free_rmse"].iloc[0]},
        {"method": "Aug. selected", "rmse": final_aug.loc[final_aug["method"] == "RBSR_validation_selected_K64_extra_trees_subunit_soft_with_root", "strict_free_rmse"].iloc[0]},
        {"method": "Aug. predicted", "rmse": final_aug.loc[final_aug["method"] == "RBSR_predicted_K128_with_root", "strict_free_rmse"].iloc[0]},
        {"method": "Oracle adm.", "rmse": final_aug.loc[final_aug["method"] == "RBSR_oracle_admission_K128_with_root", "strict_free_rmse"].iloc[0]},
    ])
    ax.barh(np.arange(len(comp)), comp["rmse"], color=[PALETTE["ridge"], PALETTE["rbsr"], PALETTE["aug"], PALETTE["bad"], PALETTE["oracle"]],
            edgecolor="black", lw=0.35)
    ax.set_yticks(np.arange(len(comp)))
    ax.set_yticklabels(comp["method"])
    ax.invert_yaxis()
    ax.set_xlabel("Strict free-vertex RMSE")
    ax.set_title("Admission prevents some augmented harm")
    panel_label(ax, "b")
    save_fig(fig, "fig3_region_error_and_admission")


def fig_coefficient_and_selection() -> None:
    base_coeff_scan = read_csv(BASE / "08_final_strict_analysis" / "coefficient_predictor_scan.csv")
    aug_coeff_scan = read_csv(AUG / "08_final_strict_analysis" / "coefficient_predictor_scan.csv")
    base_sel = read_csv(BASE / "08_final_strict_analysis" / "validation_model_selection.csv")
    aug_sel = read_csv(AUG / "08_final_strict_analysis" / "validation_model_selection.csv")
    base_tests = read_json(BASE / "08_final_strict_analysis" / "paired_tests.json")
    aug_tests = read_json(AUG / "08_final_strict_analysis" / "paired_tests.json")

    fig = plt.figure(figsize=(7.15, 5.0))
    gs = fig.add_gridspec(2, 2, wspace=0.43, hspace=0.55)

    ax = fig.add_subplot(gs[0, 0])
    rows = []
    for label, df in [("base", base_coeff_scan), ("augmented", aug_coeff_scan)]:
        m = df[df["strict_free_rmse"].notna()].copy()
        for _, r in m.iterrows():
            rows.append({
                "setting": label,
                "predictor": r["coefficient_predictor"],
                "rmse": float(r["strict_free_rmse"]),
            })
    d = pd.DataFrame(rows)
    predictors = ["ridge", "kernel_rbf", "extra_trees", "mlp_supervised"]
    x = np.arange(len(predictors))
    w = 0.36
    for off, setting, color in [(-w/2, "base", PALETTE["rbsr"]), (w/2, "augmented", PALETTE["aug"])]:
        vals = [d[(d["setting"] == setting) & (d["predictor"] == p)]["rmse"].iloc[0] for p in predictors]
        ax.bar(x + off, vals, width=w, color=color, edgecolor="black", lw=0.35, label=setting)
    ax.axhline(1.076862096786499, color=PALETTE["ridge"], lw=0.9, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(["ridge", "RBF", "trees", "MLP"], rotation=25, ha="right")
    ax.set_ylabel("Strict free-vertex RMSE")
    ax.set_title("Coefficient predictor scan")
    ax.legend(fontsize=6)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    for df, label, color in [(base_sel, "base", PALETTE["rbsr"]), (aug_sel, "augmented", PALETTE["aug"])]:
        val = df[(df["split"] == "validation") & (df["predictor"] == ("ridge" if label == "base" else "extra_trees"))]
        val = val[val["gate_mode"].isin(["none", "subunit_soft"])].copy()
        for gate, ls in [("none", "-"), ("subunit_soft", "--")]:
            g = val[val["gate_mode"] == gate].sort_values("basis")
            if len(g):
                ax.plot(g["basis"], g["strict_free_rmse"], marker="o", ms=3, lw=1.1,
                        ls=ls, color=color, label=f"{label} / {gate}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
    ax.get_xaxis().set_major_formatter(lambda x, pos: f"{int(x)}")
    ax.set_xlabel("Basis K")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("Validation-selected complexity")
    ax.legend(fontsize=5.8)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, :])
    rows = []
    keys = [
        ("base predicted", base_tests["ridge_minus_rbsr_predicted"], PALETTE["rbsr"]),
        ("base selected", base_tests["ridge_minus_rbsr_validation_selected"], PALETTE["rbsr"]),
        ("aug predicted", aug_tests["ridge_minus_rbsr_predicted"], PALETTE["bad"]),
        ("aug soft gate", aug_tests["ridge_minus_rbsr_legal_soft_admission"], PALETTE["aug"]),
        ("aug selected", aug_tests["ridge_minus_rbsr_validation_selected"], PALETTE["aug"]),
    ]
    for label, obj, color in keys:
        rows.append({"label": label, "mean": obj["mean_a_minus_b"],
                     "low": obj["ci95_low"], "high": obj["ci95_high"], "color": color})
    y = np.arange(len(rows))
    means = np.array([r["mean"] for r in rows])
    lows = np.array([r["low"] for r in rows])
    highs = np.array([r["high"] for r in rows])
    colors = [r["color"] for r in rows]
    for i, r in enumerate(rows):
        ax.errorbar(r["mean"], i, xerr=[[r["mean"] - r["low"]], [r["high"] - r["mean"]]],
                    fmt="o", color=r["color"], ecolor=PALETTE["neutral"], capsize=2.5,
                    elinewidth=1.0)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Paired gain over Ridge (positive is better)")
    ax.set_title("Effect direction after bootstrap")
    panel_label(ax, "c", x=-0.08)
    save_fig(fig, "fig5_coefficient_predictor_and_selection")


def fig_cvae_diagnostics() -> None:
    cap = read_csv(BASE / "04_cvae_capacity_scan" / "summary.csv")
    loss = read_csv(BASE / "05_cvae_loss_matrix" / "summary.csv")
    oracle = read_csv(BASE / "07_cvae_stronger_input_oracle" / "summary.csv")

    fig = plt.figure(figsize=(7.15, 5.2))
    gs = fig.add_gridspec(2, 2, wspace=0.42, hspace=0.5)

    for ax_i, layer in enumerate([2, 3]):
        ax = fig.add_subplot(gs[0, ax_i])
        sub = cap[cap["layers"] == layer]
        mat = sub.pivot(index="width", columns="latent_dim", values="strict_free_rmse").sort_index()
        im = ax.imshow(mat.values, aspect="auto", cmap=LinearSegmentedColormap.from_list("cvae", ["#F7F7F7", "#B4C0E4", "#484878"]),
                       vmin=1.2, vmax=2.5)
        ax.set_xticks(np.arange(len(mat.columns)))
        ax.set_xticklabels(mat.columns)
        ax.set_yticks(np.arange(len(mat.index)))
        ax.set_yticklabels(mat.index)
        ax.set_xlabel("Latent dim")
        ax.set_ylabel("Width")
        ax.set_title(f"CVAE capacity, L={layer}")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if mat.values[i, j] > 1.8 else "black")
        panel_label(ax, "a" if ax_i == 0 else "b")
    cb = fig.colorbar(im, ax=fig.axes[:2], fraction=0.025, pad=0.02)
    cb.set_label("Strict RMSE")

    ax = fig.add_subplot(gs[1, 0])
    order = ["roi_only", "roi_landmark", "roi_edge", "roi_normal", "roi_laplacian"]
    loss = loss.set_index("loss_name").loc[order].reset_index()
    ax.bar(np.arange(len(loss)), loss["strict_free_rmse"], color=PALETTE["cvae"],
           edgecolor="black", lw=0.35)
    add_reference_line(ax, 1.076862096786499, "Ridge")
    ax.set_xticks(np.arange(len(loss)))
    ax.set_xticklabels(["ROI", "+LM", "+edge", "+normal", "+lap"], rotation=25, ha="right")
    ax.set_ylabel("Strict free-vertex RMSE")
    ax.set_title("Geometry losses did not rescue RMSE")
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    oracle_order = ["base", "source_pca64", "source_flat_pca128", "local_coordinate_summary", "subunit_error_oracle"]
    oracle = oracle.set_index("condition_variant").loc[oracle_order].reset_index()
    labels = ["base", "PCA64", "flat PCA128", "local coord.", "subunit oracle"]
    ax.barh(np.arange(len(oracle)), oracle["strict_free_rmse"], color=[PALETTE["cvae"]] * 4 + [PALETTE["oracle"]],
            edgecolor="black", lw=0.35)
    ax.axvline(1.076862096786499, color=PALETTE["ridge"], lw=0.9, ls="--")
    ax.text(1.076862096786499, -0.35, "Ridge", fontsize=6, ha="right", va="bottom", color=PALETTE["ridge"])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Strict free-vertex RMSE")
    ax.set_title("Stronger inputs improve CVAE but remain weak")
    panel_label(ax, "d")
    save_fig(fig, "fig4_cvae_diagnostic_matrix")


def main() -> None:
    fig_main_evidence()
    fig_rbsr_mechanism()
    fig_region_heatmap_and_admission()
    fig_cvae_diagnostics()
    fig_coefficient_and_selection()
    manifest = {
        "output_dir": str(OUT),
        "figures": sorted([p.name for p in OUT.glob("*.svg")]),
        "source_outputs": {
            "base": str(BASE),
            "augmented": str(AUG),
        },
        "notes": [
            "Vertex-level Ridge residual heatmap requires rerunning residual_heatmap because final runs did not store per-vertex predictions.",
            "fig3 panel a is a region-level strict-error heatmap derived from final_strict_table.csv, not a vertex spatial heatmap.",
        ],
    }
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
