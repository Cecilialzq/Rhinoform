from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from rhinoform.baselines import arap_predict_vectorised, per_pair_metrics, write_pair_csv
from rhinoform.data import edge_index, load_rows, ordered_pairs, split_ids
from rhinoform.geometry import solve_linear_handle_baseline, uniform_laplacian
from rhinoform.repro import artifact_metadata, atomic_write_json


def controls_for_pairs(by_id: dict, pairs: list[tuple[str, str]], landmarks: np.ndarray) -> np.ndarray:
    return np.stack([(by_id[t]["vertices"] - by_id[s]["vertices"])[landmarks] for s, t in pairs], axis=0)


def mean_roi(rows: list[dict]) -> float:
    return float(np.mean([float(r["roi_rmse"]) for r in rows]))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def evaluate_pred(name: str, by_id: dict, pairs, pred, faces, edges, subunits, landmarks) -> tuple[float, list[dict]]:
    rows = per_pair_metrics(name, by_id, pairs, pred, faces, edges, subunits, landmarks)
    return mean_roi(rows), rows


def row_key(r: dict) -> tuple:
    hw = round(float(r["handle_weight"]), 6)
    ridge = round(float(r["system_ridge"]), 14) if str(r.get("system_ridge", "")) != "" else 0.0
    it = str(r.get("arap_iter", ""))
    return (r["method"], hw, ridge, it)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="data")
    parser.add_argument("--out", default="results/geometric_tuning")
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--val-split", default="clean-prior validation")
    parser.add_argument("--test-split", default="main test")
    parser.add_argument("--handle-weights", default="100,1000,10000,100000")
    parser.add_argument("--system-ridges", default="1e-10,1e-8,1e-6,1e-4")
    parser.add_argument("--arap-iters", default="3,5,10,20")
    parser.add_argument("--split-manifest", default="")
    parser.add_argument("--force", action="store_true", help="Ignore cached grid_progress.json and recompute everything.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_summary = out_dir / "geometric_validation_tuned_test_summary.csv"
    final_npz = out_dir / "geometric_validation_tuned_predictions.npz"
    final_json = out_dir / "geometric_validation_tuning.json"
    if final_summary.exists() and final_npz.exists() and final_json.exists() and not args.force:
        print("[geo] final outputs already exist; nothing to do (use --force to redo).", flush=True)
        return 0

    repo = Path(args.repo)
    print("[geo] loading ROI dataset ...", flush=True)
    _, by_id = load_rows(repo)
    val_ids = split_ids(by_id, args.val_split)
    test_ids = split_ids(by_id, args.test_split)
    if not val_ids or not test_ids:
        raise SystemExit("Validation/test splits are empty; build the required dataset first.")
    val_pairs = ordered_pairs(val_ids, None, args.seed)
    test_pairs = ordered_pairs(test_ids, None, args.seed)
    template = next(iter(by_id.values()))
    landmarks = template["landmarks"]
    faces = template["faces"]
    edges = edge_index(faces)
    subunits = template["subunits"]
    n = template["vertices"].shape[0]
    print(f"[geo] val_pairs={len(val_pairs)} test_pairs={len(test_pairs)}; building Laplacian ...", flush=True)
    lap = uniform_laplacian(n, faces)
    val_controls = controls_for_pairs(by_id, val_pairs, landmarks)
    test_controls = controls_for_pairs(by_id, test_pairs, landmarks)
    hws = [float(x) for x in args.handle_weights.split(",") if x]
    ridges = [float(x) for x in args.system_ridges.split(",") if x]
    iters = [int(x) for x in args.arap_iters.split(",") if x]

    progress_path = out_dir / "grid_progress.json"
    grid_rows: list[dict] = []
    if progress_path.exists() and not args.force:
        try:
            grid_rows = json.loads(progress_path.read_text(encoding="utf-8"))
            print(f"[geo] resumed {len(grid_rows)} cached grid configs from {progress_path}", flush=True)
        except Exception:
            grid_rows = []
    done_keys = {row_key(r) for r in grid_rows}

    total = len(hws) * len(ridges) * 2 + len(hws) * len(iters)
    idx = 0

    def commit(one: dict) -> None:
        grid_rows.append(one)
        done_keys.add(row_key(one))
        atomic_write_json(progress_path, grid_rows)

    for method in ["laplacian", "bilaplacian"]:
        op = lap if method == "laplacian" else lap @ lap
        for hw in hws:
            for ridge in ridges:
                idx += 1
                key = (method, round(hw, 6), round(ridge, 14), "")
                if key in done_keys:
                    print(f"[geo {idx}/{total}] skip {method} hw={hw:g} ridge={ridge:g} (cached)", flush=True)
                    continue
                t0 = time.time()
                pred = solve_linear_handle_baseline(val_controls, landmarks, n, op, hw, ridge)
                score, _ = evaluate_pred(f"{method}_hw{hw:g}_ridge{ridge:g}", by_id, val_pairs, pred, faces, edges, subunits, landmarks)
                commit({"method": method, "handle_weight": hw, "system_ridge": ridge, "arap_iter": "", "val_roi_rmse": score})
                print(f"[geo {idx}/{total}] {method} hw={hw:g} ridge={ridge:g} val_roi={score:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    sources = [by_id[s]["vertices"] for s, _ in val_pairs]
    for hw in hws:
        for n_iter in iters:
            idx += 1
            key = ("arap", round(hw, 6), round(1e-8, 14), str(n_iter))
            if key in done_keys:
                print(f"[geo {idx}/{total}] skip arap hw={hw:g} iter={n_iter} (cached)", flush=True)
                continue
            t0 = time.time()
            init = solve_linear_handle_baseline(val_controls, landmarks, n, lap, hw, 1e-8)
            pred = arap_predict_vectorised(sources, val_controls, landmarks, faces, lap, init, hw, 1e-8, n_iter)
            score, _ = evaluate_pred(f"arap_hw{hw:g}_iter{n_iter}", by_id, val_pairs, pred, faces, edges, subunits, landmarks)
            commit({"method": "arap", "handle_weight": hw, "system_ridge": 1e-8, "arap_iter": n_iter, "val_roi_rmse": score})
            print(f"[geo {idx}/{total}] arap hw={hw:g} iter={n_iter} val_roi={score:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    selected = {}
    for method in ["laplacian", "bilaplacian", "arap"]:
        rows_m = [r for r in grid_rows if r["method"] == method]
        selected[method] = min(rows_m, key=lambda r: float(r["val_roi_rmse"]))
    print("[geo] grid done; selected:", json.dumps(selected), flush=True)

    write_csv(out_dir / "geometric_validation_grid.csv", grid_rows)
    print("[geo] re-running selected configs on the fixed test set ...", flush=True)
    test_summary = []
    arrays = {"test_pairs": np.asarray(test_pairs, dtype=object)}
    for method, cfg in selected.items():
        t0 = time.time()
        hw = float(cfg["handle_weight"])
        ridge = float(cfg.get("system_ridge") or 1e-8)
        if method == "laplacian":
            pred = solve_linear_handle_baseline(test_controls, landmarks, n, lap, hw, ridge)
        elif method == "bilaplacian":
            pred = solve_linear_handle_baseline(test_controls, landmarks, n, lap @ lap, hw, ridge)
        else:
            init = solve_linear_handle_baseline(test_controls, landmarks, n, lap, hw, 1e-8)
            sources_test = [by_id[s]["vertices"] for s, _ in test_pairs]
            pred = arap_predict_vectorised(sources_test, test_controls, landmarks, faces, lap, init, hw, 1e-8, int(cfg["arap_iter"]))
        _, rows = evaluate_pred(method, by_id, test_pairs, pred, faces, edges, subunits, landmarks)
        write_pair_csv(out_dir / f"identity_bootstrap_pair_metrics_{method}_validation_tuned.csv", rows)
        test_summary.append({"method": method, **cfg, **{m: float(np.mean([float(r[m]) for r in rows])) for m in ["roi_rmse", "landmark_rmse", "dorsum_rmse", "tip_rmse", "edge_strain_p95", "normal_flip_pct"]}})
        arrays[f"{method}_validation_tuned_pred"] = np.asarray(pred, dtype=np.float32)
        print(f"[geo] test {method}: roi_rmse={test_summary[-1]['roi_rmse']:.4f}  ({time.time()-t0:.1f}s)", flush=True)
    write_csv(out_dir / "geometric_validation_tuned_test_summary.csv", test_summary)
    np.savez_compressed(out_dir / "geometric_validation_tuned_predictions.npz", **arrays)
    report = {
        "selection_objective": "validation ROI RMSE",
        "selected": selected,
        "test_summary": test_summary,
        "metadata": artifact_metadata(
            repo_root=Path(__file__).resolve().parents[1],
            command_args=args,
            input_manifest_path=repo / "manifest.json",
            split_manifest_path=Path(args.split_manifest) if args.split_manifest else None,
            seed=args.seed,
            data_root_identifier=repo.name,
        ),
    }
    atomic_write_json(final_json, report)
    print(json.dumps({"out": str(out_dir), "selected": selected}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
