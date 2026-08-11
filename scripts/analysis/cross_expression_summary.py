from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rhinoform.repro import atomic_write_json
from rhinoform.stats import paired_identity_bootstrap, write_csv


KEY_METRICS = ("roi_rmse", "landmark_rmse", "edge_strain_p95", "normal_flip_pct")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize cross-expression effects with identity-cluster bootstrap.")
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260612)
    args = parser.parse_args()

    report_path = args.evaluation_dir / "cross_expression_evaluation.json"
    evaluation = json.loads(report_path.read_text(encoding="utf-8"))
    expressions = [value for value in evaluation["expressions"] if value != "neutral"]
    rows_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    for expression in evaluation["expressions"]:
        for method in ("ridge", "cvae", "global_hybrid", "rbsr"):
            rows_by_key[(expression, method)] = read_rows(
                args.evaluation_dir / f"pair_metrics_{expression}_{method}.csv"
            )

    bootstrap_rows: list[dict] = []
    counter = 0
    for expression in expressions:
        comparisons = [
            ("rbsr_vs_global", rows_by_key[(expression, "global_hybrid")], rows_by_key[(expression, "rbsr")], "rbsr", "global_hybrid"),
            ("rbsr_expression_degradation", rows_by_key[("neutral", "rbsr")], rows_by_key[(expression, "rbsr")], f"rbsr_{expression}", "rbsr_neutral"),
            ("global_expression_degradation", rows_by_key[("neutral", "global_hybrid")], rows_by_key[(expression, "global_hybrid")], f"global_{expression}", "global_neutral"),
        ]
        for comparison, baseline_rows, method_rows, method_name, baseline_name in comparisons:
            for metric in KEY_METRICS:
                for cluster_key in ("source_id", "target_id"):
                    result = paired_identity_bootstrap(
                        baseline_rows,
                        method_rows,
                        method_name,
                        baseline_name,
                        metric,
                        cluster_key,
                        args.n_boot,
                        args.seed + counter,
                    )
                    result["expression"] = expression
                    result["comparison"] = comparison
                    bootstrap_rows.append(result)
                    counter += 1

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "cross_expression_identity_bootstrap.csv", bootstrap_rows)
    concise = []
    for expression in expressions:
        concise.append(
            {
                "expression": expression,
                "absolute": evaluation["summaries"][expression],
                "gate": evaluation["gate_summaries"][expression],
                "source_cluster_key_results": [
                    row
                    for row in bootstrap_rows
                    if row["expression"] == expression and row["cluster_key"] == "source_id"
                ],
            }
        )
    report = {
        "protocol": evaluation["protocol"],
        "evaluated_identity_count": evaluation["evaluated_identity_count"],
        "n_pairs_per_expression": evaluation["n_pairs_per_expression"],
        "excluded_identity_ids": evaluation["excluded_identity_ids"],
        "n_boot": args.n_boot,
        "primary_cluster_unit": "source identity",
        "sensitivity_cluster_unit": "target identity",
        "difference_sign": "method or expression minus baseline; negative is lower/better for every metric",
        "neutral_absolute": evaluation["summaries"]["neutral"],
        "expressions": concise,
    }
    atomic_write_json(args.out / "cross_expression_summary.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
