"""Direct identity-clustered paired inference with safe degenerate handling."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from rhinoform.repro import atomic_write_csv, atomic_write_json


METRICS = ("roi_rmse", "landmark_rmse", "dorsum_rmse", "tip_rmse",
           "edge_strain_p95", "normal_flip_pct")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pairs(reference: list[dict], candidate: list[dict], label: str) -> None:
    ref = [(r["source_id"], r["target_id"]) for r in reference]
    got = [(r["source_id"], r["target_id"]) for r in candidate]
    if got != ref:
        raise ValueError(f"Pair order mismatch for {label}")


def identity_differences(base: list[dict], method: list[dict], metric: str, cluster: str) -> np.ndarray:
    def aggregate(rows):
        values: dict[str, list[float]] = {}
        for row in rows:
            values.setdefault(row[cluster], []).append(float(row[metric]))
        return {key: float(np.mean(value)) for key, value in values.items()}
    left, right = aggregate(base), aggregate(method)
    ids = sorted(set(left) & set(right), key=lambda x: (int(x), x) if x.isdigit() else (10**9, x))
    return np.asarray([right[x] - left[x] for x in ids], dtype=np.float64)


def bootstrap(values: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        samples[index] = np.mean(values[rng.integers(0, len(values), len(values))])
    return tuple(map(float, np.percentile(samples, [2.5, 97.5])))


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
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--baseline", default="ridge_sourcepca")
    parser.add_argument("--methods", required=True, help="comma-separated non-baseline methods")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()

    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    base_path = args.pair_dir / f"identity_bootstrap_pair_metrics_{args.baseline}.csv"
    base = read(base_path)
    signature = {
        "baseline": args.baseline,
        "methods": methods,
        "metrics": list(METRICS),
        "seed": args.seed,
        "n_boot": args.n_boot,
        "input_sha256": {
            args.baseline: sha256_file(base_path),
            **{
                method: sha256_file(
                    args.pair_dir / f"identity_bootstrap_pair_metrics_{method}.csv"
                )
                for method in methods
            },
        },
    }
    progress_path = args.out.with_suffix(".progress.json")
    progress = {"signature": signature, "completed_rows": []}
    if progress_path.is_file():
        try:
            saved = json.loads(progress_path.read_text(encoding="utf-8"))
            if saved.get("signature") == signature:
                progress = saved
        except (OSError, json.JSONDecodeError):
            pass
    completed = {
        (str(row["method"]), str(row["metric"])): row
        for row in progress.get("completed_rows", [])
    }
    for method_index, method in enumerate(methods):
        candidate = read(args.pair_dir / f"identity_bootstrap_pair_metrics_{method}.csv")
        validate_pairs(base, candidate, method)
        for metric_index, metric in enumerate(METRICS):
            test_index = method_index * len(METRICS) + metric_index + 1
            key = (method, metric)
            if key in completed:
                print(
                    f"PAIRED STATS resume from Drive {test_index}/{len(methods) * len(METRICS)} "
                    f"method={method} metric={metric}",
                    flush=True,
                )
                continue
            print(
                f"PAIRED STATS live {test_index}/{len(methods) * len(METRICS)} "
                f"method={method} metric={metric}",
                flush=True,
            )
            source = identity_differences(base, candidate, metric, "source_id")
            target = identity_differences(base, candidate, metric, "target_id")
            source_ci = bootstrap(source, args.seed + 2 * (method_index * len(METRICS) + metric_index), args.n_boot)
            target_ci = bootstrap(target, args.seed + 2 * (method_index * len(METRICS) + metric_index) + 1, args.n_boot)
            mean = float(np.mean(source))
            sd = float(np.std(source, ddof=1))
            effect = mean / sd if sd > 0 else 0.0
            p_value = safe_wilcoxon(source)
            completed[key] = {
                "method": method, "baseline": args.baseline, "metric": metric,
                "mean_difference_method_minus_baseline": mean,
                "source_ci95_low": source_ci[0], "source_ci95_high": source_ci[1],
                "target_ci95_low": target_ci[0], "target_ci95_high": target_ci[1],
                "wilcoxon_p_two_sided": p_value,
                "effect_size_dz_source": effect,
                "source_identity_win_rate": float(np.mean(source < 0)),
                "degenerate_all_zero": bool(np.allclose(source, 0.0, rtol=0.0, atol=1e-15)),
            }
            progress["completed_rows"] = [
                completed[(declared_method, declared_metric)]
                for declared_method in methods
                for declared_metric in METRICS
                if (declared_method, declared_metric) in completed
            ]
            atomic_write_json(progress_path, progress)
            print(f"PAIRED STATS persisted to Drive: {method}/{metric}", flush=True)
    rows = [completed[(method, metric)] for method in methods for metric in METRICS]
    raw = [float(row["wilcoxon_p_two_sided"]) for row in rows]
    for row, adjusted in zip(rows, holm(raw)):
        row["holm_family_size"] = len(rows)
        row["holm_p_two_sided"] = adjusted
        row["final_judgement"] = (
            "improved" if adjusted < 0.05 and row["source_ci95_high"] < 0 and row["target_ci95_high"] < 0
            else "worse" if adjusted < 0.05 and row["source_ci95_low"] > 0 and row["target_ci95_low"] > 0
            else "not_significant_or_cluster_sensitive"
        )
    atomic_write_csv(args.out, rows)
    atomic_write_json(args.out.with_suffix(".json"), {
        "family_definition": "all declared non-baseline methods x six frozen metrics",
        "seed": args.seed, "n_boot": args.n_boot, "rows": rows,
        "degenerate_rule": "all-zero/NaN Wilcoxon is p=1, effect=0; never assigned significance",
    })
    print(f"PAIRED STATS final family persisted to Drive: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
