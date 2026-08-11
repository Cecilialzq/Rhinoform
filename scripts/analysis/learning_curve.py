from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rhinoform.repro import SEED_REGISTRY, artifact_metadata, atomic_write_json
from rhinoform.stats import LOWER_IS_BETTER


METRICS = list(LOWER_IS_BETTER)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def parse_run_id(run_id: str) -> tuple[str, int | str, int]:
    # required_primary_chain2_scale480
    parts = run_id.split("_")
    curve = parts[1]
    chain = "anchor" if parts[2] == "anchor" else int(parts[2].removeprefix("chain"))
    scale = int(parts[3].removeprefix("scale"))
    return curve, chain, scale


def canonical_method(name: str) -> str:
    if name == "ridge_sourcepca":
        return "ridge"
    if name == "cvae_only":
        return "cvae"
    if name.startswith("hybrid_alpha_"):
        return "hybrid"
    return name


def source_means(rows: list[dict[str, str]], metric: str, restrict_ids: set[str] | None = None) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        sid = str(row["source_id"])
        tid = str(row["target_id"])
        if restrict_ids is not None and (sid not in restrict_ids or tid not in restrict_ids):
            continue
        grouped[sid].append(float(row[metric]))
    return {sid: float(np.mean(vals)) for sid, vals in grouped.items() if vals}


def hierarchical_ci(chain_source_values: list[dict[str, float]], n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if not chain_source_values:
        return float("nan"), float("nan")
    boot = np.empty(n_boot, dtype=np.float64)
    n_chains = len(chain_source_values)
    for i in range(n_boot):
        vals = []
        chain_idx = rng.integers(0, n_chains, size=n_chains)
        for ci in chain_idx:
            one = np.asarray(list(chain_source_values[int(ci)].values()), dtype=np.float64)
            if one.size:
                vals.append(float(np.mean(one[rng.integers(0, one.size, size=one.size)])))
        boot[i] = float(np.mean(vals)) if vals else np.nan
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def collect_run_rows(plan: dict) -> dict[tuple[str, int | str, int, str], list[dict[str, str]]]:
    out: dict[tuple[str, int | str, int, str], list[dict[str, str]]] = {}
    for run in plan.get("runs", []):
        curve, chain, scale = parse_run_id(run["run_id"])
        pair_dir = Path(run["out_dir"]) / "pair_metrics"
        for path in sorted(pair_dir.glob("identity_bootstrap_pair_metrics_*.csv")):
            method = canonical_method(path.name.removeprefix("identity_bootstrap_pair_metrics_").removesuffix(".csv"))
            out[(curve, chain, scale, method)] = read_csv(path)
    return out


def load_original50(split_manifest: Path | None) -> set[str] | None:
    if split_manifest is None or not split_manifest.exists():
        return None
    obj = json.loads(split_manifest.read_text(encoding="utf-8"))
    ids = obj.get("fixed_original", {}).get("test")
    return set(str(x) for x in ids) if ids else None


def aggregate_curve(
    rows_by_run: dict[tuple[str, int | str, int, str], list[dict[str, str]]],
    curve: str,
    restrict_ids: set[str] | None,
    n_boot: int,
    seed: int,
    include_anchor: bool = False,
) -> list[dict]:
    keys = sorted(
        (k for k in rows_by_run if k[0] == curve),
        key=lambda k: (str(k[0]), str(k[1]), int(k[2]), str(k[3])),
    )
    by_scale_method: dict[tuple[int, str], list[tuple[int | str, list[dict[str, str]]]]] = defaultdict(list)
    for _, chain, scale, method in keys:
        if chain == "anchor" and not include_anchor:
            continue
        if chain != "anchor" and include_anchor:
            continue
        by_scale_method[(scale, method)].append((chain, rows_by_run[(curve, chain, scale, method)]))
    out = []
    grouped_items = sorted(by_scale_method.items())
    total_tasks = len(grouped_items) * len(METRICS)
    task_index = 0
    subset = "original50" if restrict_ids is not None else "test100"
    chain_group = "anchor" if include_anchor else "random_chains"
    for (scale, method), chain_rows in grouped_items:
        for metric in METRICS:
            task_index += 1
            print(
                f"[learning_curve {curve}/{subset}/{chain_group}] bootstrap task "
                f"{task_index}/{total_tasks} ({100.0 * task_index / total_tasks:.1f}%) "
                f"scale={scale} method={method} metric={metric}",
                flush=True,
            )
            chain_values = []
            chain_source_values = []
            for chain, rows in chain_rows:
                sm = source_means(rows, metric, restrict_ids)
                if not sm:
                    continue
                chain_source_values.append(sm)
                chain_values.append(float(np.mean(list(sm.values()))))
            if not chain_values:
                continue
            lo, hi = hierarchical_ci(chain_source_values, n_boot, seed + scale + len(out))
            out.append(
                {
                    "curve": curve,
                    "scale": scale,
                    "method": method,
                    "metric": metric,
                    "subset": subset,
                    "chain_group": chain_group,
                    "n_chains": len(chain_values),
                    "mean": float(np.mean(chain_values)),
                    "std": float(np.std(chain_values, ddof=1)) if len(chain_values) > 1 else 0.0,
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )
    return out


def gap_rows(table: list[dict]) -> list[dict]:
    out = []
    by_key = {(r["curve"], r["scale"], r["metric"], r["subset"], r["method"]): r for r in table}
    for r in table:
        if r["method"] != "hybrid" or r["metric"] != "roi_rmse":
            continue
        for other in ("cvae", "ridge"):
            o = by_key.get((r["curve"], r["scale"], r["metric"], r["subset"], other))
            if o:
                out.append(
                    {
                        "curve": r["curve"],
                        "scale": r["scale"],
                        "gap": f"{other}_minus_hybrid",
                        "subset": r["subset"],
                        "metric": r["metric"],
                        "mean": float(o["mean"]) - float(r["mean"]),
                    }
                )
    return out


def load_geometric_reference_rows(path: Path, scales: list[int], curve: str, subset: str = "test100") -> list[dict]:
    if not path.exists():
        return []
    source = read_csv(path)
    rows = []
    method_map = {
        "arap": "arap",
        "laplacian": "laplacian",
        "bi_laplacian": "bilaplacian",
        "bi-laplacian": "bilaplacian",
        "bilaplacian": "bilaplacian",
    }
    for src in source:
        method_raw = str(src.get("method", "")).strip().lower().replace(" ", "_")
        method = method_map.get(method_raw, method_raw)
        if method not in {"arap", "laplacian", "bilaplacian"}:
            continue
        for metric in METRICS:
            if metric not in src or src[metric] == "":
                continue
            value = float(src[metric])
            for scale in scales:
                rows.append(
                    {
                        "curve": curve,
                        "scale": scale,
                        "method": method,
                        "metric": metric,
                        "subset": subset,
                        "chain_group": "validation_tuned_geometric_reference",
                        "n_chains": 0,
                        "mean": value,
                        "std": 0.0,
                        "ci95_low": value,
                        "ci95_high": value,
                    }
                )
    return rows


def load_geometric_reference_rows_from_pair_metrics(
    pair_dir: Path,
    scales: list[int],
    curve: str,
    restrict_ids: set[str],
) -> list[dict]:
    method_files = {
        "arap": pair_dir / "identity_bootstrap_pair_metrics_arap_validation_tuned.csv",
        "laplacian": pair_dir / "identity_bootstrap_pair_metrics_laplacian_validation_tuned.csv",
        "bilaplacian": pair_dir / "identity_bootstrap_pair_metrics_bilaplacian_validation_tuned.csv",
    }
    missing = [str(path) for path in method_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing cache: " + ", ".join(missing))
    rows = []
    for method, path in method_files.items():
        metric_rows = read_csv(path)
        for metric in METRICS:
            sm = source_means(metric_rows, metric, restrict_ids)
            if not sm:
                raise FileNotFoundError(f"missing cache: original50 rows for {method} {metric} in {path}")
            value = float(np.mean(list(sm.values())))
            for scale in scales:
                rows.append(
                    {
                        "curve": curve,
                        "scale": scale,
                        "method": method,
                        "metric": metric,
                        "subset": "original50",
                        "chain_group": "validation_tuned_geometric_reference",
                        "n_chains": 0,
                        "mean": value,
                        "std": 0.0,
                        "ci95_low": value,
                        "ci95_high": value,
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="execution_plan.json")
    parser.add_argument("--out", default="frozen_artifacts/learning_curve")
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--split-manifest", default="")
    parser.add_argument("--geometric-summary", default="results/geometric_tuning/geometric_validation_tuned_test_summary.csv")
    parser.add_argument("--geometric-pair-dir", default="results/geometric_tuning")
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    rows_by_run = collect_run_rows(plan)
    original50 = load_original50(Path(args.split_manifest) if args.split_manifest else None)
    out_dir = Path(args.out)
    primary = aggregate_curve(rows_by_run, "primary", None, args.n_boot, args.seed)
    secondary = aggregate_curve(rows_by_run, "secondary", None, args.n_boot, args.seed + 100000)
    anchor = aggregate_curve(rows_by_run, "primary", None, args.n_boot, args.seed + 400000, include_anchor=True)
    primary_scales = sorted({int(r["scale"]) for r in primary})
    secondary_scales = sorted({int(r["scale"]) for r in secondary})
    geometric_path = Path(args.geometric_summary)
    primary.extend(load_geometric_reference_rows(geometric_path, primary_scales, "primary"))
    secondary.extend(load_geometric_reference_rows(geometric_path, secondary_scales, "secondary"))
    write_csv(out_dir / "table_lc_primary.csv", primary)
    write_csv(out_dir / "table_lc_secondary.csv", secondary)
    write_csv(out_dir / "table_lc_anchor.csv", anchor)
    write_csv(out_dir / "lc_gap_primary.csv", gap_rows(primary))
    write_csv(out_dir / "lc_gap_secondary.csv", gap_rows(secondary))
    if original50:
        primary_original50 = aggregate_curve(rows_by_run, "primary", original50, args.n_boot, args.seed + 200000)
        secondary_original50 = aggregate_curve(rows_by_run, "secondary", original50, args.n_boot, args.seed + 300000)
        primary_original50.extend(load_geometric_reference_rows_from_pair_metrics(Path(args.geometric_pair_dir), primary_scales, "primary", original50))
        secondary_original50.extend(load_geometric_reference_rows_from_pair_metrics(Path(args.geometric_pair_dir), secondary_scales, "secondary", original50))
        write_csv(out_dir / "table_lc_primary_original50.csv", primary_original50)
        write_csv(out_dir / "table_lc_secondary_original50.csv", secondary_original50)
    report = {
        "plan": args.plan,
        "n_input_run_method_tables": len(rows_by_run),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "outputs": {
            "primary": str(out_dir / "table_lc_primary.csv"),
            "secondary": str(out_dir / "table_lc_secondary.csv"),
            "anchor": str(out_dir / "table_lc_anchor.csv"),
            "geometric_summary": str(geometric_path),
        },
        "metadata": artifact_metadata(
            repo_root=Path(__file__).resolve().parents[1],
            command_args=args,
            split_manifest_path=Path(args.split_manifest) if args.split_manifest else None,
            seed=args.seed,
            data_root_identifier="run_pair_metric_csvs",
        ),
    }
    atomic_write_json(out_dir / "learning_curve_manifest.json", report)
    print(json.dumps({"out": str(out_dir), "input_tables": len(rows_by_run), "primary_rows": len(primary), "secondary_rows": len(secondary)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
