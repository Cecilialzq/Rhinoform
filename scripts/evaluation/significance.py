from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


LOWER_IS_BETTER = {
    "roi_rmse": True,
    "landmark_rmse": True,
    "dorsum_rmse": True,
    "tip_rmse": True,
    "edge_strain_p95": True,
    "normal_flip_pct": True,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def identity_diffs(
    baseline_rows: list[dict[str, str]],
    method_rows: list[dict[str, str]],
    metric: str,
    cluster_key: str,
) -> np.ndarray:
    def means(rows: list[dict[str, str]]) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for row in rows:
            grouped.setdefault(row[cluster_key], []).append(float(row[metric]))
        return {k: float(np.mean(v)) for k, v in grouped.items()}

    base = means(baseline_rows)
    meth = means(method_rows)
    ids = sorted(set(base) & set(meth), key=lambda x: int(x) if x.isdigit() else x)
    return np.asarray([meth[i] - base[i] for i in ids], dtype=np.float64)


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p_values[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def bootstrap_ci(diffs: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boot[i] = float(np.mean(diffs[rng.integers(0, n, size=n)]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="../results")
    parser.add_argument("--baseline", default="ridge_sourcepca")
    parser.add_argument("--methods", default="", help="Comma-separated method names. Defaults to every pair-metrics CSV except the baseline.")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2041)
    args = parser.parse_args()

    out_dir = Path(args.out)
    baseline_name = args.baseline
    if args.methods.strip():
        method_names = [m.strip() for m in args.methods.split(",") if m.strip()]
    else:
        method_names = []
        for path in sorted(out_dir.glob("identity_bootstrap_pair_metrics_*.csv")):
            name = path.name.removeprefix("identity_bootstrap_pair_metrics_").removesuffix(".csv")
            if name != baseline_name:
                method_names.append(name)
    if not method_names:
        raise FileNotFoundError(f"No method pair-metrics CSVs found in {out_dir}")
    baseline_rows = read_csv(out_dir / f"identity_bootstrap_pair_metrics_{baseline_name}.csv")
    method_rows_by_name = {
        name: read_csv(out_dir / f"identity_bootstrap_pair_metrics_{name}.csv")
        for name in method_names
    }

    rows: list[dict[str, float | str]] = []
    raw_p_values: list[float] = []
    for method_name in method_names:
        method_rows = method_rows_by_name[method_name]
        for metric in LOWER_IS_BETTER:
            source_diffs = identity_diffs(baseline_rows, method_rows, metric, "source_id")
            target_diffs = identity_diffs(baseline_rows, method_rows, metric, "target_id")
            source_lo, source_hi = bootstrap_ci(source_diffs, args.n_boot, args.seed + len(rows) * 2)
            target_lo, target_hi = bootstrap_ci(target_diffs, args.n_boot, args.seed + len(rows) * 2 + 1)
            mean_diff = float(np.mean(source_diffs))
            std = float(np.std(source_diffs, ddof=1))
            effect = float(mean_diff / std) if std > 0 else 0.0
            lower_better = LOWER_IS_BETTER[metric]
            win_rate = float(np.mean(source_diffs < 0.0 if lower_better else source_diffs > 0.0))
            try:
                p_two_sided = float(wilcoxon(source_diffs, zero_method="wilcox", alternative="two-sided").pvalue)
            except ValueError:
                p_two_sided = 1.0
            raw_p_values.append(p_two_sided)
            if lower_better:
                source_sig_better = source_hi < 0.0
                target_sig_better = target_hi < 0.0
                source_sig_worse = source_lo > 0.0
                target_sig_worse = target_lo > 0.0
            else:
                source_sig_better = source_lo > 0.0
                target_sig_better = target_lo > 0.0
                source_sig_worse = source_hi < 0.0
                target_sig_worse = target_hi < 0.0
            if source_sig_better and target_sig_better:
                bootstrap_direction = "improved"
            elif source_sig_worse and target_sig_worse:
                bootstrap_direction = "worse"
            else:
                bootstrap_direction = "mixed_or_not_significant"
            rows.append(
                {
                    "method": method_name,
                    "baseline": baseline_name,
                    "metric": metric,
                    "mean_diff": mean_diff,
                    "source_ci95_low": source_lo,
                    "source_ci95_high": source_hi,
                    "target_ci95_low": target_lo,
                    "target_ci95_high": target_hi,
                    "wilcoxon_p_two_sided": p_two_sided,
                    "effect_size_dz_source": effect,
                    "source_identity_win_rate": win_rate,
                    "bootstrap_direction": bootstrap_direction,
                }
            )

    adjusted = holm_adjust(raw_p_values)
    holm_family_size = len(raw_p_values)
    for row, p_holm in zip(rows, adjusted):
        row["holm_family_size"] = float(holm_family_size)
        row["holm_p_two_sided"] = float(p_holm)
        row["final_judgement"] = (
            row["bootstrap_direction"]
            if float(p_holm) < 0.05
            else "not_significant_after_holm"
        )

    csv_path = out_dir / "final_statistical_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "baseline": baseline_name,
        "methods": method_names,
        "n_boot": args.n_boot,
        "holm_family_size": holm_family_size,
        "holm_family_definition": "All method x metric comparisons included in this significance.py run.",
        "note": "Diffs are method minus baseline. For all included metrics lower is better, so negative means the method is better. Wilcoxon and Holm correction use source-identity paired diffs; target CI is a sensitivity check.",
        "rows": rows,
    }
    (out_dir / "final_statistical_table.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
