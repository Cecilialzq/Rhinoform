from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = [
    ("roi_rmse", "ROI RMSE ↓"),
    ("landmark_rmse", "LM RMSE ↓"),
    ("dorsum_rmse", "Dorsum RMSE ↓"),
    ("tip_rmse", "Tip RMSE ↓"),
    ("edge_strain_p95", "Edge p95 ↓"),
    ("normal_flip_pct", "Flip % ↓"),
]

METHOD_LABELS = {
    "ridge_sourcepca": "Ridge + source PCA",
    "cvae_only": "CVAE only",
    "hybrid_alpha_0.3": "Hybrid α=0.3",
    "hybrid_alpha_0.4": "Hybrid α=0.4",
    "laplacian_handles": "Laplacian handles",
    "bilaplacian_handles": "Bi-Laplacian handles",
    "arap_handles_iter1": "ARAP handles (1 iter)",
    "arap_handles_iter3": "ARAP handles (3 iter)",
}

HANDLE_METHOD_PREFIXES = ("laplacian_handles", "bilaplacian_handles", "arap_handles")

JUDGEMENT_LABELS = {
    "improved": "significant improvement",
    "worse": "significant degradation",
    "mixed_or_not_significant": "mixed / not significant",
    "not_significant_after_holm": "not significant after Holm",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: str | float) -> float:
    return float(value)


def fmt_num(value: float, metric: str) -> str:
    if metric == "normal_flip_pct":
        return f"{value:.2f}"
    if metric == "edge_strain_p95":
        return f"{value:.4f}"
    return f"{value:.3f}"


def fmt_p(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def method_label(name: str) -> str:
    return METHOD_LABELS.get(name, name.replace("_", " "))


def is_handle_method(method: str) -> bool:
    return method.startswith(HANDLE_METHOD_PREFIXES)


def load_table_inputs(stats_dir: Path) -> tuple[str, list[dict[str, str]], list[dict[str, str]], dict]:
    summary_path = stats_dir / "identity_bootstrap_summary.csv"
    final_path = stats_dir / "final_statistical_table.csv"
    report_path = stats_dir / "final_statistical_table.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    if not final_path.exists():
        raise FileNotFoundError(f"Missing {final_path}")
    summary = read_csv(summary_path)
    final = read_csv(final_path)
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    baseline = final[0]["baseline"] if final else "ridge_sourcepca"
    return baseline, summary, final, report


def build_records(
    baseline: str,
    summary: list[dict[str, str]],
    final: list[dict[str, str]],
    method_order: list[str] | None,
) -> tuple[list[str], dict[tuple[str, str], dict[str, float | str]]]:
    records: dict[tuple[str, str], dict[str, float | str]] = {}
    methods = {baseline}

    for row in summary:
        method = row["method"]
        metric = row["metric"]
        row_baseline = row.get("baseline", baseline)
        methods.add(method)
        methods.add(row_baseline)
        records[(row_baseline, metric)] = {
            "mean": as_float(row["baseline_cluster_mean"]),
            "baseline": row_baseline,
            "method": row_baseline,
            "metric": metric,
            "is_baseline": row_baseline == baseline,
        }
        records[(method, metric)] = {
            "mean": as_float(row["method_cluster_mean"]),
            "baseline": baseline,
            "method": method,
            "metric": metric,
            "is_baseline": False,
        }

    for row in final:
        method = row["method"]
        metric = row["metric"]
        methods.add(method)
        rec = records.setdefault(
            (method, metric),
            {
                "baseline": baseline,
                "method": method,
                "metric": metric,
                "is_baseline": False,
            },
        )
        rec.update(
            {
                "mean_diff": as_float(row["mean_diff"]),
                "source_lo": as_float(row["source_ci95_low"]),
                "source_hi": as_float(row["source_ci95_high"]),
                "target_lo": as_float(row["target_ci95_low"]),
                "target_hi": as_float(row["target_ci95_high"]),
                "holm_p": as_float(row["holm_p_two_sided"]),
                "win_rate": as_float(row["source_identity_win_rate"]),
                "judgement": row["final_judgement"],
            }
        )
        base_rec = records.get((baseline, metric))
        if "mean" not in rec and base_rec and "mean" in base_rec:
            rec["mean"] = as_float(base_rec["mean"]) + as_float(row["mean_diff"])

    for metric, _ in METRICS:
        rec = records.get((baseline, metric))
        if rec:
            rec["is_baseline"] = True

    if method_order:
        return [m for m in method_order if m in methods], records

    preferred = [
        baseline,
        "hybrid_alpha_0.4",
        "cvae_only",
        "laplacian_handles",
        "bilaplacian_handles",
        "arap_handles_iter1",
        "arap_handles_iter3",
    ]
    ordered = [m for m in preferred if m in methods]
    ordered.extend(sorted(m for m in methods if m not in set(ordered)))
    return ordered, records


def best_methods(methods: list[str], records: dict[tuple[str, str], dict[str, float | str]]) -> dict[str, set[str]]:
    best: dict[str, set[str]] = {}
    for metric, _ in METRICS:
        vals = []
        for method in methods:
            if metric == "landmark_rmse" and is_handle_method(method):
                continue
            rec = records.get((method, metric))
            if rec and "mean" in rec:
                vals.append((as_float(rec["mean"]), method))
        if not vals:
            best[metric] = set()
            continue
        best_value = min(v for v, _ in vals)
        best[metric] = {m for v, m in vals if abs(v - best_value) < 1e-12}
    return best


def metric_cell(method: str, metric: str, records: dict[tuple[str, str], dict[str, float | str]], best: set[str]) -> str:
    rec = records.get((method, metric))
    if not rec or "mean" not in rec:
        return "—"
    mean = fmt_num(as_float(rec["mean"]), metric)
    if metric == "landmark_rmse" and is_handle_method(method):
        mean = f"{mean}†"
    if method in best:
        mean = f"**{mean}**"
    if rec.get("is_baseline"):
        return mean
    if "mean_diff" not in rec:
        return mean
    diff = fmt_num(as_float(rec["mean_diff"]), metric)
    lo = fmt_num(as_float(rec["source_lo"]), metric)
    hi = fmt_num(as_float(rec["source_hi"]), metric)
    p = fmt_p(as_float(rec["holm_p"]))
    return f"{mean}<br>Δ={diff}, CI [{lo}, {hi}], pH={p}"


def compact_metric_cell(method: str, metric: str, records: dict[tuple[str, str], dict[str, float | str]], best: set[str]) -> str:
    rec = records.get((method, metric))
    if not rec or "mean" not in rec:
        return "—"
    mean = fmt_num(as_float(rec["mean"]), metric)
    if metric == "landmark_rmse" and is_handle_method(method):
        mean = f"{mean}†"
    return f"**{mean}**" if method in best else mean


def judgement_cell(method: str, records: dict[tuple[str, str], dict[str, float | str]]) -> str:
    if all(records.get((method, metric), {}).get("is_baseline") for metric, _ in METRICS if records.get((method, metric))):
        return "reference"
    judgements = [records.get((method, metric), {}).get("judgement", "") for metric, _ in METRICS]
    improved = sum(j == "improved" for j in judgements)
    worse = sum(j == "worse" for j in judgements)
    nonsig = sum(j in {"mixed_or_not_significant", "not_significant_after_holm"} for j in judgements)
    return f"{improved} improved / {worse} worse / {nonsig} n.s."


def render_markdown(
    methods: list[str],
    records: dict[tuple[str, str], dict[str, float | str]],
    report: dict,
) -> str:
    best = best_methods(methods, records)
    header = ["Method"] + [label for _, label in METRICS] + ["Holm summary"]
    lines = [
        "# Main Quantitative Table",
        "",
        "All values are source-identity means. Lower is better for every metric.",
        f"Holm family size: {report.get('holm_family_size', 'unknown')}. Family definition: {report.get('holm_family_definition', 'not recorded')}",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for method in methods:
        row = [method_label(method)]
        for metric, _ in METRICS:
            row.append(compact_metric_cell(method, metric, records, best.get(metric, set())))
        row.append(judgement_cell(method, records))
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "Notes:",
            "- Bold marks the best absolute mean in each metric column.",
            "- The Holm summary counts metrics judged significant after Holm correction using source-identity paired differences.",
            "- Target-identity CI is retained as a sensitivity check in the detailed statistics table below.",
            "- † Landmark RMSE for handle-based geometric methods is a handle-constraint residual, not an unconstrained prediction error; it is excluded from best-value bolding and should not be used as evidence that handle methods predict landmarks better.",
            "",
            "## Detailed Statistical Checks",
            "",
            "| Method | Metric | Mean Δ | Source 95% CI | Target 95% CI | Holm p | Win rate | Final judgement |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for method in methods:
        if method == report.get("baseline"):
            continue
        for metric, label in METRICS:
            rec = records.get((method, metric))
            if not rec or "mean_diff" not in rec:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        method_label(method),
                        label,
                        fmt_num(as_float(rec["mean_diff"]), metric),
                        f"[{fmt_num(as_float(rec['source_lo']), metric)}, {fmt_num(as_float(rec['source_hi']), metric)}]",
                        f"[{fmt_num(as_float(rec['target_lo']), metric)}, {fmt_num(as_float(rec['target_hi']), metric)}]",
                        fmt_p(as_float(rec["holm_p"])),
                        f"{as_float(rec['win_rate']):.2f}",
                        JUDGEMENT_LABELS.get(str(rec["judgement"]), str(rec["judgement"])),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "Detailed-statistics notes:",
            "- Δ is method minus baseline; negative Δ means lower error than the selected baseline.",
            "- pH is the Holm-adjusted two-sided Wilcoxon p-value over source-identity paired differences.",
            "- Landmark RMSE rows for handle-based methods are included for constraint diagnostics only.",
        ]
    )
    return "\n".join(lines) + "\n"


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
        .replace("α", "$\\alpha$")
        .replace("Δ", "$\\Delta$")
        .replace("↓", "$\\downarrow$")
        .replace("—", "--")
    )


def markdown_cell_to_latex(cell: str) -> str:
    cell = cell.replace("<br>", "; ")
    if cell.startswith("**") and cell.endswith("**") and cell.count("**") == 2:
        return "\\textbf{" + cell[2:-2] + "}"
    return cell.replace("**", "")


def render_latex(methods: list[str], records: dict[tuple[str, str], dict[str, float | str]]) -> str:
    best = best_methods(methods, records)
    cols = "l" + "c" * len(METRICS) + "c"
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Sparse-control nasal deformation results. Lower is better for all metrics. Values are source-identity means; bold marks the best value in each metric column. $\\dagger$ marks landmark residuals for handle-based geometric methods, which are constraint residuals rather than unconstrained prediction errors and are excluded from best-value bolding.}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\hline",
        "Method & " + " & ".join(latex_escape(label) for _, label in METRICS) + " & Holm summary \\\\",
        "\\hline",
    ]
    for method in methods:
        row = [latex_escape(method_label(method))]
        for metric, _ in METRICS:
            cell = markdown_cell_to_latex(compact_metric_cell(method, metric, records, best.get(metric, set())))
            row.append(cell if cell.startswith("\\textbf{") else latex_escape(cell))
        row.append(latex_escape(judgement_cell(method, records)))
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-dir", default="../results")
    parser.add_argument("--out", default="", help="Output markdown path. Defaults to <stats-dir>/paper_main_table.md.")
    parser.add_argument("--latex-out", default="", help="Output LaTeX path. Defaults to <stats-dir>/paper_main_table.tex.")
    parser.add_argument("--method-order", default="", help="Comma-separated method order. Missing methods are appended alphabetically.")
    args = parser.parse_args()

    stats_dir = Path(args.stats_dir)
    method_order = [m.strip() for m in args.method_order.split(",") if m.strip()] if args.method_order.strip() else None
    baseline, summary, final, report = load_table_inputs(stats_dir)
    methods, records = build_records(baseline, summary, final, method_order)

    md = render_markdown(methods, records, report)
    tex = render_latex(methods, records)
    md_path = Path(args.out) if args.out else stats_dir / "paper_main_table.md"
    tex_path = Path(args.latex_out) if args.latex_out else stats_dir / "paper_main_table.tex"
    md_path.write_text(md, encoding="utf-8")
    tex_path.write_text(tex, encoding="utf-8")
    print(json.dumps({"markdown": str(md_path), "latex": str(tex_path), "methods": methods}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
