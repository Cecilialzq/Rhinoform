"""Audit a predeclared RB-SR PCA-capacity sweep against one matched LAMM run.

All input CSVs must contain the same ordered identity pairs and the same metric
schema.  Lower is better for every declared metric.  The script is descriptive
post-hoc evidence: it never selects or trains a model and it never rewrites an
input artifact.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from rhinoform.repro import (
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
    write_sha256_sidecar,
)


METRICS = (
    "roi_rmse",
    "dorsum_rmse",
    "tip_rmse",
    "edge_strain_p95",
    "normal_flip_pct",
    "abs_flip_pct",
    "missed_flip_pct",
)
CORE_PARETO_METRICS = ("roi_rmse", "normal_flip_pct", "edge_strain_p95")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty pair-metric CSV: {path}")
    missing = {"source_id", "target_id", *METRICS} - set(rows[0])
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return rows


def validate_pairs(reference: list[dict[str, str]], candidate: list[dict[str, str]], label: str) -> None:
    expected = [(row["source_id"], row["target_id"]) for row in reference]
    observed = [(row["source_id"], row["target_id"]) for row in candidate]
    if observed != expected:
        raise ValueError(f"Pair order mismatch for {label}")


def identity_differences(
    baseline: list[dict[str, str]],
    method: list[dict[str, str]],
    metric: str,
    cluster: str,
) -> np.ndarray:
    def aggregate(rows: list[dict[str, str]]) -> dict[str, float]:
        values: dict[str, list[float]] = {}
        for row in rows:
            values.setdefault(row[cluster], []).append(float(row[metric]))
        return {key: float(np.mean(item)) for key, item in values.items()}

    left, right = aggregate(baseline), aggregate(method)
    if set(left) != set(right):
        raise ValueError(f"Identity cluster mismatch for {metric}/{cluster}")
    ids = sorted(left, key=lambda value: (int(value), value) if value.isdigit() else (10**9, value))
    return np.asarray([right[value] - left[value] for value in ids], dtype=np.float64)


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        sample = values[rng.integers(0, len(values), len(values))]
        samples[index] = float(np.mean(sample))
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def safe_wilcoxon(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0 or np.allclose(finite, 0.0, rtol=0.0, atol=1e-15):
        return 1.0
    result = float(wilcoxon(finite, zero_method="wilcox", alternative="two-sided").pvalue)
    return result if np.isfinite(result) else 1.0


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * float(values[index]))
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def pareto_labels(rows: list[dict[str, object]]) -> set[str]:
    frontier: set[str] = set()
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            no_worse = all(float(other[key]) <= float(candidate[key]) for key in CORE_PARETO_METRICS)
            strictly_better = any(float(other[key]) < float(candidate[key]) for key in CORE_PARETO_METRICS)
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.add(str(candidate["label"]))
    return frontier


def parse_candidate(value: str) -> tuple[str, int, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Candidate must be LABEL=DIMENSION=CSV")
    label, dimension, path = parts
    try:
        parsed_dimension = int(dimension)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid PCA dimension: {dimension}") from error
    if parsed_dimension <= 0:
        raise argparse.ArgumentTypeError("PCA dimension must be positive")
    return label, parsed_dimension, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lamm-csv", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        required=True,
        help="Repeat as LABEL=DIMENSION=PAIR_METRICS_CSV",
    )
    parser.add_argument("--declared-deployment-dimension", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()

    labels = [label for label, _, _ in args.candidate]
    dimensions = [dimension for _, dimension, _ in args.candidate]
    if len(set(labels)) != len(labels) or len(set(dimensions)) != len(dimensions):
        raise ValueError("Candidate labels and dimensions must each be unique")
    if args.declared_deployment_dimension not in dimensions:
        raise ValueError("Declared deployment dimension is absent from the candidate grid")
    if args.n_boot <= 0:
        raise ValueError("n_boot must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lamm = read_csv(args.lamm_csv)
    candidate_rows: dict[str, list[dict[str, str]]] = {}
    input_hashes = {"lamm": sha256_file(args.lamm_csv)}
    for label, _, path in args.candidate:
        rows = read_csv(path)
        validate_pairs(lamm, rows, label)
        candidate_rows[label] = rows
        input_hashes[label] = sha256_file(path)

    lamm_means = {metric: float(np.mean([float(row[metric]) for row in lamm])) for metric in METRICS}
    summary_rows: list[dict[str, object]] = []
    statistics_rows: list[dict[str, object]] = []
    for candidate_index, (label, dimension, _) in enumerate(args.candidate):
        rows = candidate_rows[label]
        means = {metric: float(np.mean([float(row[metric]) for row in rows])) for metric in METRICS}
        summary: dict[str, object] = {"label": label, "source_pca_dim": dimension, **means}
        for metric in METRICS:
            summary[f"delta_vs_lamm_{metric}"] = means[metric] - lamm_means[metric]
            summary[f"ratio_vs_lamm_{metric}"] = means[metric] / lamm_means[metric]
        summary["mean_dominates_lamm_all_declared_metrics"] = all(
            means[metric] < lamm_means[metric] for metric in METRICS
        )
        summary["mean_beats_lamm_roi_and_new_flip"] = bool(
            means["roi_rmse"] < lamm_means["roi_rmse"]
            and means["normal_flip_pct"] < lamm_means["normal_flip_pct"]
        )
        summary_rows.append(summary)

        for metric_index, metric in enumerate(METRICS):
            source = identity_differences(lamm, rows, metric, "source_id")
            target = identity_differences(lamm, rows, metric, "target_id")
            seed_offset = candidate_index * len(METRICS) + metric_index
            source_ci = bootstrap_ci(source, args.seed + 2 * seed_offset, args.n_boot)
            target_ci = bootstrap_ci(target, args.seed + 2 * seed_offset + 1, args.n_boot)
            sd = float(np.std(source, ddof=1))
            mean_difference = float(np.mean(source))
            statistics_rows.append({
                "label": label,
                "source_pca_dim": dimension,
                "baseline": "lamm",
                "metric": metric,
                "mean_difference_rbsr_minus_lamm": mean_difference,
                "source_ci95_low": source_ci[0],
                "source_ci95_high": source_ci[1],
                "target_ci95_low": target_ci[0],
                "target_ci95_high": target_ci[1],
                "wilcoxon_p_two_sided": safe_wilcoxon(source),
                "effect_size_dz_source": mean_difference / sd if sd > 0 else 0.0,
                "source_identity_win_rate": float(np.mean(source < 0)),
                "degenerate_all_zero": bool(np.allclose(source, 0.0, rtol=0.0, atol=1e-15)),
            })

    frontier = pareto_labels(summary_rows)
    adjusted = holm([float(row["wilcoxon_p_two_sided"]) for row in statistics_rows])
    by_label: dict[str, list[dict[str, object]]] = {label: [] for label in labels}
    for row, p_value in zip(statistics_rows, adjusted):
        row["holm_family_size"] = len(statistics_rows)
        row["holm_p_two_sided"] = p_value
        improved = bool(
            p_value < 0.05
            and float(row["source_ci95_high"]) < 0
            and float(row["target_ci95_high"]) < 0
        )
        worse = bool(
            p_value < 0.05
            and float(row["source_ci95_low"]) > 0
            and float(row["target_ci95_low"]) > 0
        )
        row["judgement"] = "improved" if improved else "worse" if worse else "inconclusive_or_cluster_sensitive"
        by_label[str(row["label"])].append(row)

    for summary in summary_rows:
        label = str(summary["label"])
        summary["core_pareto_frontier"] = label in frontier
        summary["statistically_supported_all_declared_metrics"] = all(
            row["judgement"] == "improved" for row in by_label[label]
        )
        summary["statistically_supported_roi_and_new_flip"] = all(
            row["judgement"] == "improved"
            for row in by_label[label]
            if row["metric"] in {"roi_rmse", "normal_flip_pct"}
        )

    deployment = next(
        row for row in summary_rows
        if int(row["source_pca_dim"]) == args.declared_deployment_dimension
    )
    better_accuracy = [
        row for row in summary_rows
        if float(row["roi_rmse"]) < float(deployment["roi_rmse"])
    ]
    deployment_tradeoff_supported = bool(
        deployment["core_pareto_frontier"]
        and all(
            float(deployment["normal_flip_pct"]) <= float(row["normal_flip_pct"])
            for row in better_accuracy
        )
    )

    atomic_write_csv(args.out_dir / "capacity_tradeoff_summary.csv", summary_rows)
    atomic_write_csv(args.out_dir / "capacity_tradeoff_statistics.csv", statistics_rows)
    report = {
        "status": "POSTHOC_MATCHED_CAPACITY_AUDIT_COMPLETE",
        "claim_boundary": (
            "Supplementary post-hoc matched-control evidence on an already observed internal test; "
            "not a new blind confirmatory headline."
        ),
        "lower_is_better_metrics": list(METRICS),
        "core_pareto_metrics": list(CORE_PARETO_METRICS),
        "holm_family": "all declared PCA dimensions x seven non-degenerate deployment metrics",
        "n_pairs": len(lamm),
        "lamm_means": lamm_means,
        "declared_grid": dimensions,
        "declared_deployment_dimension": args.declared_deployment_dimension,
        "deployment_tradeoff_supported": deployment_tradeoff_supported,
        "deployment_record": deployment,
        "any_dimension_mean_beats_lamm_roi_and_new_flip": any(
            bool(row["mean_beats_lamm_roi_and_new_flip"]) for row in summary_rows
        ),
        "any_dimension_statistically_beats_lamm_roi_and_new_flip": any(
            bool(row["statistically_supported_roi_and_new_flip"]) for row in summary_rows
        ),
        "any_dimension_mean_dominates_lamm_all_declared_metrics": any(
            bool(row["mean_dominates_lamm_all_declared_metrics"]) for row in summary_rows
        ),
        "any_dimension_statistically_dominates_lamm_all_declared_metrics": any(
            bool(row["statistically_supported_all_declared_metrics"]) for row in summary_rows
        ),
        "input_sha256": input_hashes,
        "summary_rows": summary_rows,
    }
    report_path = args.out_dir / "capacity_tradeoff_report.json"
    atomic_write_json(report_path, report)
    for path in (
        args.out_dir / "capacity_tradeoff_summary.csv",
        args.out_dir / "capacity_tradeoff_statistics.csv",
        report_path,
    ):
        write_sha256_sidecar(path)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
