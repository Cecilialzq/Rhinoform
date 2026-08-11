from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rhinoform.repro import atomic_write_json
from rhinoform.stats import write_csv


METRICS = (
    "roi_rmse",
    "landmark_rmse",
    "dorsum_rmse",
    "tip_rmse",
    "edge_strain_p95",
    "normal_flip_pct",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aligned_differences(
    method_rows: list[dict[str, str]], baseline_rows: list[dict[str, str]]
) -> dict[str, float]:
    method = {
        (row["source_id"], row["target_id"]): row
        for row in method_rows
    }
    baseline = {
        (row["source_id"], row["target_id"]): row
        for row in baseline_rows
    }
    keys = sorted(set(method) & set(baseline))
    if len(keys) != len(method_rows) or len(keys) != len(baseline_rows):
        raise ValueError("Pair sets are not identical")
    return {
        metric: float(
            np.mean(
                [float(method[key][metric]) - float(baseline[key][metric]) for key in keys]
            )
        )
        for metric in METRICS
    }


def source_level_differences(
    method_rows: list[dict[str, str]], baseline_rows: list[dict[str, str]]
) -> dict[str, dict[str, float]]:
    method = {
        (row["source_id"], row["target_id"]): row
        for row in method_rows
    }
    baseline = {
        (row["source_id"], row["target_id"]): row
        for row in baseline_rows
    }
    keys = sorted(set(method) & set(baseline))
    grouped: dict[str, list[tuple[str, str]]] = {}
    for key in keys:
        grouped.setdefault(key[0], []).append(key)
    return {
        source_id: {
            metric: float(
                np.mean(
                    [
                        float(method[key][metric]) - float(baseline[key][metric])
                        for key in source_keys
                    ]
                )
            )
            for metric in METRICS
        }
        for source_id, source_keys in grouped.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize frozen RBSR results across chains.")
    parser.add_argument(
        "--chain",
        action="append",
        required=True,
        help="Repeat label=rbsr_csv,ridge_csv,hybrid_csv.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260611)
    args = parser.parse_args()

    chain_rows: list[dict[str, float | str]] = []
    details: dict[str, dict] = {}
    clustered: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for value in args.chain:
        label, paths = value.split("=", 1)
        path_values = [Path(path) for path in paths.split(",")]
        if len(path_values) != 3:
            raise ValueError(f"Expected three CSVs for {label}")
        rbsr_rows, ridge_rows, hybrid_rows = [read_rows(path) for path in path_values]
        versus_ridge = aligned_differences(rbsr_rows, ridge_rows)
        versus_hybrid = aligned_differences(rbsr_rows, hybrid_rows)
        clustered[label] = {
            "ridge": source_level_differences(rbsr_rows, ridge_rows),
            "global_hybrid": source_level_differences(rbsr_rows, hybrid_rows),
        }
        details[label] = {
            "n_pairs": len(rbsr_rows),
            "rbsr_minus_ridge": versus_ridge,
            "rbsr_minus_global_hybrid": versus_hybrid,
        }
        flat: dict[str, float | str] = {"chain": label, "n_pairs": float(len(rbsr_rows))}
        for metric in METRICS:
            flat[f"{metric}_vs_ridge"] = versus_ridge[metric]
            flat[f"{metric}_vs_global_hybrid"] = versus_hybrid[metric]
        chain_rows.append(flat)

    aggregate: dict[str, dict[str, float | int]] = {}
    rng = np.random.default_rng(args.seed)
    labels = sorted(clustered)
    for baseline in ("ridge", "global_hybrid"):
        aggregate[baseline] = {}
        for metric in METRICS:
            values = np.asarray(
                [float(row[f"{metric}_vs_{baseline}"]) for row in chain_rows], dtype=np.float64
            )
            boot = np.empty(args.n_boot, dtype=np.float64)
            for index in range(args.n_boot):
                sampled_labels = rng.choice(labels, size=len(labels), replace=True)
                sampled_chain_means = []
                for sampled_label in sampled_labels:
                    source_values = clustered[str(sampled_label)][baseline]
                    source_ids = list(source_values)
                    sampled_sources = rng.choice(source_ids, size=len(source_ids), replace=True)
                    sampled_chain_means.append(
                        float(
                            np.mean(
                                [source_values[str(source_id)][metric] for source_id in sampled_sources]
                            )
                        )
                    )
                boot[index] = float(np.mean(sampled_chain_means))
            ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
            aggregate[baseline][metric] = {
                "mean_difference": float(np.mean(values)),
                "std_across_chains": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "hierarchical_bootstrap_ci_low": float(ci_low),
                "hierarchical_bootstrap_ci_high": float(ci_high),
                "improved_chains": int(np.sum(values < 0.0)),
                "n_chains": int(len(values)),
            }

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "rbsr_multichain_differences.csv", chain_rows)
    report = {
        "interpretation": "Negative differences favor RBSR for all reported metrics.",
        "hierarchical_bootstrap": {
            "n_boot": args.n_boot,
            "seed": args.seed,
            "procedure": "Resample chains, then source identities within each resampled chain.",
        },
        "chains": details,
        "aggregate": aggregate,
    }
    atomic_write_json(args.out / "rbsr_multichain_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
