from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rhinoform.repro import atomic_write_json


METRICS = (
    "dame_area_proxy",
    "fmpd_roi_proxy",
    "msdm2_style_proxy",
    "normal_flip_face_count",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def identity_means(rows: list[dict[str, str]], metric: str, cluster: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row[cluster]), []).append(float(row[metric]))
    return {identity: float(np.mean(values)) for identity, values in grouped.items()}


def paired_bootstrap(
    baseline: list[dict[str, str]],
    method: list[dict[str, str]],
    metric: str,
    cluster: str,
    n_boot: int,
    seed: int,
) -> dict[str, float | str]:
    base = identity_means(baseline, metric, cluster)
    test = identity_means(method, metric, cluster)
    identities = sorted(set(base) & set(test), key=lambda value: int(value))
    differences = np.asarray([test[value] - base[value] for value in identities], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(n_boot, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    interpretation = "improved" if high < 0.0 else "worse" if low > 0.0 else "not_significant"
    return {
        "cluster_key": cluster,
        "metric": metric,
        "n_cluster_identities": len(identities),
        "baseline_cluster_mean": float(np.mean([base[value] for value in identities])),
        "method_cluster_mean": float(np.mean([test[value] for value in identities])),
        "mean_diff_method_minus_baseline": float(differences.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "probability_lower": float(np.mean(bootstrap < 0.0)),
        "interpretation": interpretation,
    }


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Identity-cluster bootstrap for perceptual mesh proxy results.")
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260612)
    args = parser.parse_args()

    methods = {
        name: read_rows(args.evaluation_dir / f"pair_metrics_{name}.csv")
        for name in ("ridge", "cvae", "global_hybrid", "rbsr")
    }
    comparisons = (
        ("rbsr_vs_global_hybrid", "global_hybrid", "rbsr"),
        ("rbsr_vs_ridge", "ridge", "rbsr"),
        ("global_hybrid_vs_ridge", "ridge", "global_hybrid"),
        ("cvae_vs_ridge", "ridge", "cvae"),
    )
    output_rows: list[dict[str, float | str]] = []
    counter = 0
    for comparison, baseline_name, method_name in comparisons:
        for metric in METRICS:
            for cluster in ("source_id", "target_id"):
                result = paired_bootstrap(
                    methods[baseline_name],
                    methods[method_name],
                    metric,
                    cluster,
                    args.n_boot,
                    args.seed + counter,
                )
                result.update(
                    {
                        "comparison": comparison,
                        "baseline": baseline_name,
                        "method": method_name,
                    }
                )
                output_rows.append(result)
                counter += 1

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "perceptual_mesh_proxy_identity_bootstrap.csv"
    write_csv(csv_path, output_rows)
    report = {
        "n_boot": args.n_boot,
        "difference_sign": "method minus baseline; all metrics are lower-is-better",
        "primary_cluster_unit": "source identity",
        "sensitivity_cluster_unit": "target identity",
        "results": output_rows,
    }
    atomic_write_json(args.out / "perceptual_mesh_proxy_identity_bootstrap.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
