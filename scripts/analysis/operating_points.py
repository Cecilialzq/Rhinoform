from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rhinoform.repro import artifact_metadata, atomic_write_json


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


def parse_tau(spec: str, rows: list[dict]) -> list[tuple[str, float]]:
    out = []
    for raw in [x.strip() for x in spec.split(",") if x.strip()]:
        if raw == "ridge-anchor":
            ridge = next((r for r in rows if "ridge" in r.get("label", "").lower() or "ridge" in r.get("method", "").lower()), None)
            if ridge is None:
                continue
            out.append((raw, float(ridge["normal_flip_pct"])))
        else:
            out.append((raw, float(raw)))
    return out


def pareto_front(rows: list[dict]) -> list[dict]:
    pts = sorted(rows, key=lambda r: (float(r["normal_flip_pct"]), float(r["roi_rmse"])))
    front = []
    best_roi = float("inf")
    for row in pts:
        roi = float(row["roi_rmse"])
        if roi < best_roi:
            front.append(row)
            best_roi = roi
    return front


def choose(rows: list[dict], tau_name: str, tau: float) -> dict | None:
    feasible = [r for r in rows if float(r["normal_flip_pct"]) <= tau]
    if not feasible:
        return None
    best = min(feasible, key=lambda r: float(r["roi_rmse"]))
    return {
        **best,
        "operating_point": f"flip_le_{tau_name}",
        "selection_tau": tau,
        "selection_rule": "minimise ROI_RMSE subject to flip <= tau",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="CSV with label/method, roi_rmse and normal_flip_pct columns.")
    parser.add_argument("--out", default="frozen_artifacts/operating_points")
    parser.add_argument("--taus", default="ridge-anchor,1.3,1.5")
    parser.add_argument("--split-manifest", default="")
    args = parser.parse_args()

    rows = read_csv(Path(args.candidates))
    normalised = []
    for i, row in enumerate(rows):
        if "roi_rmse" not in row or "normal_flip_pct" not in row:
            continue
        label = row.get("label") or row.get("method") or f"candidate_{i}"
        normalised.append(
            {
                **row,
                "label": label,
                "roi_rmse": float(row["roi_rmse"]),
                "normal_flip_pct": float(row["normal_flip_pct"]),
            }
        )
    if not normalised:
        raise SystemExit("No candidate rows with roi_rmse and normal_flip_pct found.")
    out_dir = Path(args.out)
    front = pareto_front(normalised)
    selected = []
    accuracy = min(normalised, key=lambda r: float(r["roi_rmse"]))
    selected.append({**accuracy, "operating_point": "accuracy_optimal", "selection_tau": "", "selection_rule": "minimum ROI_RMSE"})
    for tau_name, tau in parse_tau(args.taus, normalised):
        picked = choose(normalised, tau_name, tau)
        if picked is not None:
            selected.append(picked)
    # Deduplicate by label/rule while keeping named operating points.
    write_csv(out_dir / "pareto_front.csv", front)
    write_csv(out_dir / "pareto_points.csv", selected)
    report = {
        "candidates": args.candidates,
        "taus": args.taus,
        "n_candidates": len(normalised),
        "n_pareto": len(front),
        "n_selected": len(selected),
        "metadata": artifact_metadata(
            repo_root=Path(__file__).resolve().parents[1],
            command_args=args,
            split_manifest_path=Path(args.split_manifest) if args.split_manifest else None,
            seed=None,
            data_root_identifier="cached_operating_point_candidates",
        ),
    }
    atomic_write_json(out_dir / "operating_points_manifest.json", report)
    print(json.dumps({"out": str(out_dir), "selected": len(selected)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

