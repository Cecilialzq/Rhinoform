from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rhinoform.data import load_rows
from scripts.evaluation.dependence import loio_sign_stability, two_way_pigeonhole_ci
from rhinoform.stats import LOWER_IS_BETTER, metric_rows_for_method, paired_identity_bootstrap, write_csv


def mix(ridge: np.ndarray, cvae: np.ndarray, gate: np.ndarray) -> np.ndarray:
    ridge_vertices = np.asarray(ridge, dtype=np.float32).reshape(len(ridge), -1, 3)
    cvae_vertices = np.asarray(cvae, dtype=np.float32).reshape(len(cvae), -1, 3)
    gate_vertices = np.asarray(gate, dtype=np.float32).reshape(1, -1, 1)
    return (ridge_vertices + gate_vertices * (cvae_vertices - ridge_vertices)).reshape(len(ridge), -1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Identity-aware uncertainty analysis for a validation-derived fixed spatial gate."
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gates", required=True)
    parser.add_argument("--gate-key", default="smoothed_hard_gate")
    parser.add_argument("--global-alpha", type=float, default=0.1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--n-two-way-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260611)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, by_id = load_rows(Path(args.repo))
    predictions = np.load(args.predictions, allow_pickle=True)
    gates = np.load(args.gates, allow_pickle=False)
    if args.gate_key not in gates:
        raise KeyError(f"Gate key {args.gate_key!r} not found; available: {gates.files}")

    pairs = [(str(source), str(target)) for source, target in predictions["test_pairs"].tolist()]
    ridge = np.asarray(predictions["ridge_cond_pred"], dtype=np.float32)
    cvae = np.asarray(predictions["cvae_pred"], dtype=np.float32)
    hybrid = mix(
        ridge,
        cvae,
        np.full(ridge.shape[1] // 3, args.global_alpha, dtype=np.float32),
    )
    spatial = mix(ridge, cvae, np.asarray(gates[args.gate_key], dtype=np.float32))

    print("Computing per-pair metrics", flush=True)
    rows_by_method = {
        "ridge_anchor": metric_rows_for_method(by_id, pairs, ridge),
        "global_hybrid": metric_rows_for_method(by_id, pairs, hybrid),
        "fixed_smoothed_hard_gate": metric_rows_for_method(by_id, pairs, spatial),
    }
    for name, rows in rows_by_method.items():
        write_csv(out_dir / f"pair_metrics_{name}.csv", rows)

    metrics = list(LOWER_IS_BETTER)
    comparisons = [
        ("global_hybrid", "fixed_smoothed_hard_gate"),
        ("ridge_anchor", "fixed_smoothed_hard_gate"),
    ]
    cluster_rows: list[dict[str, float | str]] = []
    loio_rows: list[dict[str, float | str]] = []
    two_way_rows: list[dict[str, float | str]] = []
    for comparison_index, (baseline_name, method_name) in enumerate(comparisons):
        for metric_index, metric in enumerate(metrics):
            for cluster_index, cluster_key in enumerate(("source_id", "target_id")):
                cluster_rows.append(
                    paired_identity_bootstrap(
                        rows_by_method[baseline_name],
                        rows_by_method[method_name],
                        method_name,
                        baseline_name,
                        metric,
                        cluster_key,
                        args.n_boot,
                        args.seed + comparison_index * 100 + metric_index * 10 + cluster_index,
                    )
                )
            loio = loio_sign_stability(rows_by_method[baseline_name], rows_by_method[method_name], metric)
            loio_rows.append(
                {
                    "method": method_name,
                    "baseline": baseline_name,
                    "metric": metric,
                    **loio,
                }
            )
            if metric in {"roi_rmse", "edge_strain_p95", "normal_flip_pct"}:
                two_way = two_way_pigeonhole_ci(
                    rows_by_method[baseline_name],
                    rows_by_method[method_name],
                    metric,
                    args.n_two_way_boot,
                    args.seed + 1000 + comparison_index * 100 + metric_index,
                    progress_label=f"{method_name} vs {baseline_name} {metric}",
                )
                two_way_rows.append(
                    {
                        "method": method_name,
                        "baseline": baseline_name,
                        "metric": metric,
                        **two_way,
                    }
                )

    write_csv(out_dir / "cluster_bootstrap.csv", cluster_rows)
    write_csv(out_dir / "loio_stability.csv", loio_rows)
    write_csv(out_dir / "two_way_bootstrap.csv", two_way_rows)
    report = {
        "status": "exploratory_stage_0",
        "selection_boundary": "Gate and smoothing were fixed from validation before test evaluation.",
        "method": "fixed_smoothed_hard_gate",
        "gate_key": args.gate_key,
        "n_pairs": len(pairs),
        "n_boot": args.n_boot,
        "n_two_way_boot": args.n_two_way_boot,
        "cluster_bootstrap": cluster_rows,
        "loio": loio_rows,
        "two_way_bootstrap": two_way_rows,
    }
    (out_dir / "spatial_gate_statistics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_dir), "comparisons": len(comparisons)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
