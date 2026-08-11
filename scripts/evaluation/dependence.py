from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from rhinoform.repro import SEED_REGISTRY, artifact_metadata, atomic_write_json
from rhinoform.stats import LOWER_IS_BETTER


METRICS = list(LOWER_IS_BETTER)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def row_map(rows: list[dict[str, str]], metric: str) -> dict[tuple[str, str], float]:
    return {(str(r["source_id"]), str(r["target_id"])): float(r[metric]) for r in rows}


def two_way_pigeonhole_ci(
    baseline_rows: list[dict[str, str]],
    method_rows: list[dict[str, str]],
    metric: str,
    n_boot: int,
    seed: int,
    progress_label: str = "",
) -> dict[str, float]:
    base = row_map(baseline_rows, metric)
    meth = row_map(method_rows, metric)
    pairs = sorted(set(base) & set(meth))
    ids = sorted({x for pair in pairs for x in pair}, key=lambda x: int(x) if x.isdigit() else 10**9)
    diff = {p: meth[p] - base[p] for p in pairs}
    observed = float(np.mean([diff[p] for p in pairs]))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    n = len(ids)
    started = time.monotonic()
    report_every = max(1, n_boot // 20)
    for i in range(n_boot):
        src_sample = rng.choice(ids, size=n, replace=True)
        tgt_sample = rng.choice(ids, size=n, replace=True)
        vals = []
        for s in src_sample:
            for t in tgt_sample:
                if s == t:
                    continue
                val = diff.get((s, t))
                if val is not None:
                    vals.append(val)
        boot[i] = float(np.mean(vals)) if vals else np.nan
        completed = i + 1
        if completed == n_boot or completed % report_every == 0:
            elapsed = time.monotonic() - started
            rate = completed / elapsed if elapsed > 0 else 0.0
            eta = (n_boot - completed) / rate if rate > 0 else float("nan")
            print(
                f"[dependence {progress_label}] bootstrap {completed}/{n_boot} "
                f"({100.0 * completed / n_boot:.1f}%) | elapsed={elapsed:.1f}s | eta={eta:.1f}s",
                flush=True,
            )
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "mean_diff": observed,
        "two_way_ci95_low": float(lo),
        "two_way_ci95_high": float(hi),
        "n_test_identities": float(n),
        "n_pairs": float(len(pairs)),
        "n_boot_effective": float(len(boot)),
    }


def loio_sign_stability(
    baseline_rows: list[dict[str, str]],
    method_rows: list[dict[str, str]],
    metric: str,
) -> dict[str, float]:
    base = row_map(baseline_rows, metric)
    meth = row_map(method_rows, metric)
    pairs = sorted(set(base) & set(meth))
    ids = sorted({x for pair in pairs for x in pair}, key=lambda x: int(x) if x.isdigit() else 10**9)
    full = float(np.mean([meth[p] - base[p] for p in pairs]))
    full_sign = np.sign(full)
    signs = []
    means = []
    for held in ids:
        vals = [meth[p] - base[p] for p in pairs if held not in p]
        mean = float(np.mean(vals))
        means.append(mean)
        signs.append(np.sign(mean) == full_sign if full_sign != 0 else mean == 0)
    return {
        "loio_preserve_sign_fraction": float(np.mean(signs)),
        "loio_min_mean_diff": float(np.min(means)),
        "loio_max_mean_diff": float(np.max(means)),
        "loio_n_folds": float(len(ids)),
    }


def direction(row: dict[str, float], metric: str) -> str:
    lower = LOWER_IS_BETTER[metric]
    lo = row["two_way_ci95_low"]
    hi = row["two_way_ci95_high"]
    if lower:
        if hi < 0:
            return "improved"
        if lo > 0:
            return "worse"
        return "mixed_or_not_significant"
    if lo > 0:
        return "improved"
    if hi < 0:
        return "worse"
    return "mixed_or_not_significant"


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--method-csv", required=True)
    parser.add_argument("--baseline-name", default="ridge_sourcepca")
    parser.add_argument("--method-name", default="hybrid")
    parser.add_argument("--out", default="results/dependence")
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--split-manifest", default="")
    args = parser.parse_args()

    baseline_rows = read_csv(Path(args.baseline_csv))
    method_rows = read_csv(Path(args.method_csv))
    out_dir = Path(args.out)
    progress_path = out_dir / "dependence_progress.json"
    progress_signature = {
        "baseline_sha256": sha256_file(Path(args.baseline_csv)),
        "method_sha256": sha256_file(Path(args.method_csv)),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "metrics": METRICS,
    }
    progress = {"signature": progress_signature, "completed_rows": []}
    if progress_path.exists():
        try:
            saved = json.loads(progress_path.read_text(encoding="utf-8"))
            if saved.get("signature") == progress_signature:
                progress = saved
        except (OSError, json.JSONDecodeError):
            pass
    completed_by_metric = {str(row["metric"]): row for row in progress.get("completed_rows", [])}
    rows: list[dict[str, float | str]] = []
    for metric_i, metric in enumerate(METRICS):
        if metric in completed_by_metric:
            print(f"[dependence {metric_i + 1}/{len(METRICS)}] resume: {metric} already complete", flush=True)
            rows.append(completed_by_metric[metric])
            continue
        print(f"[dependence {metric_i + 1}/{len(METRICS)}] start metric={metric}", flush=True)
        tw = two_way_pigeonhole_ci(
            baseline_rows,
            method_rows,
            metric,
            args.n_boot,
            args.seed + metric_i,
            progress_label=f"{metric_i + 1}/{len(METRICS)} {metric}",
        )
        loio = loio_sign_stability(baseline_rows, method_rows, metric)
        row: dict[str, float | str] = {
            "method": args.method_name,
            "baseline": args.baseline_name,
            "metric": metric,
            **tw,
            **loio,
        }
        row["two_way_direction"] = direction(row, metric)
        row["significance_rule"] = "call significant only if source-clustered Holm survives and two-way CI direction agrees"
        rows.append(row)
        progress["completed_rows"] = rows
        atomic_write_json(progress_path, progress)
        print(f"[dependence {metric_i + 1}/{len(METRICS)}] complete metric={metric}", flush=True)

    write_csv(out_dir / "dependence_robustness.csv", rows)
    report = {
        "baseline_csv": args.baseline_csv,
        "method_csv": args.method_csv,
        "baseline": args.baseline_name,
        "method": args.method_name,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "bootstrap": "two-way pigeonhole: resample source and target identities independently, preserve multiplicity, exclude self-pairs",
        "loio": "drop each fixed test identity from both source and target roles; no retraining",
        "rows": rows,
        "metadata": artifact_metadata(
            repo_root=Path(__file__).resolve().parents[1],
            command_args=args,
            input_manifest_path=None,
            split_manifest_path=Path(args.split_manifest) if args.split_manifest else None,
            seed=args.seed,
            data_root_identifier="frozen_pair_csvs",
        ),
    }
    atomic_write_json(out_dir / "dependence_robustness.json", report)
    print(json.dumps({"out": str(out_dir), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
