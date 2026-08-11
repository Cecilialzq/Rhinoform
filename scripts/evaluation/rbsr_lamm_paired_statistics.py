"""Matched RB-SR versus LAMM inference for the three core metrics.

The 9,900 ordered identity pairs are crossed rather than independent.  This
analysis therefore aggregates paired differences by source identity and by
target identity, reports both cluster views, and uses the larger of their
two-sided Wilcoxon p-values before Holm correction across the three predeclared
metrics.  A metric is called improved only when both clustered bootstrap
confidence intervals are wholly below zero as well.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy.stats import wilcoxon

from rhinoform.repro import (
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
    valid_sha256_sidecar,
    write_sha256_sidecar,
)


CORE_METRICS = ("roi_rmse", "normal_flip_pct", "edge_strain_p95")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ordered_pairs(rows: list[dict[str, str]]) -> list[tuple[str, str]]:
    return [(str(row["source_id"]), str(row["target_id"])) for row in rows]


def paired_differences(
    lamm: list[dict[str, str]],
    rbsr: list[dict[str, str]],
    metric: str,
) -> np.ndarray:
    return np.asarray(
        [float(candidate[metric]) - float(reference[metric]) for reference, candidate in zip(lamm, rbsr)],
        dtype=np.float64,
    )


def aggregate_by_identity(
    rows: list[dict[str, str]],
    differences: np.ndarray,
    field: str,
) -> np.ndarray:
    grouped: dict[str, list[float]] = {}
    for row, difference in zip(rows, differences):
        grouped.setdefault(str(row[field]), []).append(float(difference))
    keys = sorted(grouped, key=lambda value: (int(value), value) if value.isdigit() else (10**9, value))
    return np.asarray([np.mean(grouped[key]) for key in keys], dtype=np.float64)


def safe_wilcoxon_two_sided(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0 or np.allclose(finite, 0.0, rtol=0.0, atol=1e-15):
        return 1.0
    result = float(wilcoxon(finite, zero_method="wilcox", alternative="two-sided").pvalue)
    return result if np.isfinite(result) else 1.0


def bootstrap_mean_ci(values: np.ndarray, *, seed: int, n_boot: int) -> tuple[float, float]:
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        means[index] = np.mean(values[rng.integers(0, len(values), len(values))])
    return tuple(map(float, np.percentile(means, [2.5, 97.5])))


def holm(raw_p_values: list[float]) -> list[float]:
    order = np.argsort(raw_p_values)
    adjusted = np.empty(len(raw_p_values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(raw_p_values) - rank) * float(raw_p_values[index]))
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def effect_size_dz(values: np.ndarray) -> float:
    sd = float(np.std(values, ddof=1))
    return float(np.mean(values) / sd) if sd > 0.0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lamm-pairs", type=Path, required=True)
    parser.add_argument("--rbsr-pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--n-boot", type=int, default=20000)
    args = parser.parse_args()
    for path in (args.lamm_pairs, args.rbsr_pairs):
        if not valid_sha256_sidecar(path):
            raise RuntimeError(f"Pair metrics or SHA-256 sidecar is invalid: {path}")
    lamm = read_rows(args.lamm_pairs)
    rbsr = read_rows(args.rbsr_pairs)
    if len(lamm) != len(rbsr) or len(lamm) != 9900:
        raise ValueError(f"Expected two complete 9,900-pair tables, got {len(lamm)} and {len(rbsr)}")
    if ordered_pairs(lamm) != ordered_pairs(rbsr):
        raise ValueError("RB-SR and LAMM ordered pairs do not match exactly")

    rows: list[dict[str, object]] = []
    for metric_index, metric in enumerate(CORE_METRICS):
        pair = paired_differences(lamm, rbsr, metric)
        source = aggregate_by_identity(lamm, pair, "source_id")
        target = aggregate_by_identity(lamm, pair, "target_id")
        if len(source) != len(target) or len(source) != 100:
            raise ValueError(f"Expected 100 source and target identity clusters for {metric}")
        source_ci = bootstrap_mean_ci(
            source, seed=args.seed + 2 * metric_index, n_boot=args.n_boot
        )
        target_ci = bootstrap_mean_ci(
            target, seed=args.seed + 2 * metric_index + 1, n_boot=args.n_boot
        )
        source_p = safe_wilcoxon_two_sided(source)
        target_p = safe_wilcoxon_two_sided(target)
        rows.append(
            {
                "metric": metric,
                "difference_definition": "rbsr_minus_lamm_lower_is_better",
                "mean_paired_difference": float(np.mean(pair)),
                "median_paired_difference": float(np.median(pair)),
                "pair_win_rate": float(np.mean(pair < 0.0)),
                "source_identity_clusters": len(source),
                "source_mean_difference": float(np.mean(source)),
                "source_ci95_low": source_ci[0],
                "source_ci95_high": source_ci[1],
                "source_wilcoxon_p_two_sided": source_p,
                "source_effect_size_dz": effect_size_dz(source),
                "source_identity_win_rate": float(np.mean(source < 0.0)),
                "target_identity_clusters": len(target),
                "target_mean_difference": float(np.mean(target)),
                "target_ci95_low": target_ci[0],
                "target_ci95_high": target_ci[1],
                "target_wilcoxon_p_two_sided": target_p,
                "target_effect_size_dz": effect_size_dz(target),
                "target_identity_win_rate": float(np.mean(target < 0.0)),
                "robust_raw_p_two_sided": max(source_p, target_p),
                "degenerate_all_zero": bool(np.allclose(pair, 0.0, rtol=0.0, atol=1e-15)),
            }
        )

    adjusted = holm([float(row["robust_raw_p_two_sided"]) for row in rows])
    for row, adjusted_p in zip(rows, adjusted):
        row["holm_family_size"] = len(CORE_METRICS)
        row["holm_robust_p_two_sided"] = adjusted_p
        improved = (
            float(row["mean_paired_difference"]) < 0.0
            and float(row["source_ci95_high"]) < 0.0
            and float(row["target_ci95_high"]) < 0.0
            and adjusted_p < 0.05
        )
        row["statistically_improved"] = improved
        row["final_judgement"] = (
            "rbsr_significantly_improved" if improved else "not_significant_or_cluster_sensitive"
        )

    payload = {
        "status": (
            "ALL_THREE_CORE_METRICS_SIGNIFICANTLY_IMPROVED"
            if all(bool(row["statistically_improved"]) for row in rows)
            else "NOT_ALL_CORE_METRICS_SIGNIFICANTLY_IMPROVED"
        ),
        "claim_boundary": "supplementary post-hoc matched inference; not blind confirmatory evidence",
        "pairing": "exact same 9,900 ordered source-target pairs",
        "dependence_control": (
            "separate source-identity and target-identity aggregation; robust p is their maximum"
        ),
        "multiplicity": "Holm correction across the three predeclared core metrics",
        "alpha": 0.05,
        "seed": args.seed,
        "n_boot": args.n_boot,
        "lamm_pair_metrics_sha256": sha256_file(args.lamm_pairs),
        "rbsr_pair_metrics_sha256": sha256_file(args.rbsr_pairs),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(args.out.with_suffix(".csv"), rows)
    write_sha256_sidecar(args.out.with_suffix(".csv"))
    atomic_write_json(args.out.with_suffix(".json"), payload)
    write_sha256_sidecar(args.out.with_suffix(".json"))
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
