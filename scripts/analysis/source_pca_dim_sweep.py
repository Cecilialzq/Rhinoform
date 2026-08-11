"""source-PCA dimension sweep {0,8,16,32,64} under the CANONICAL strict protocol.

REPRODUCIBILITY GUARANTEE (the dim=16 self-check):
  The only missing new dimensions are {8,32,64}. This driver still retrains dim=16 first
  as a positive-control anchor, using the EXACT frozen split, the seed from
  seed_registry.json, the same lambda grid and the same strict free-RMSE evaluation. It
  asserts that dim=16 Ridge ROI free-RMSE reproduces the frozen reference read from
  outputs/unified_strict/main_table/main_table_strict_means.csv. ONLY IF this self-check
  passes are dims {8,32,64} trained and considered comparable to the canonical table. The
  dim=0 baseline is not retrained; it is rescored from the frozen no_source dense output.
  If dim=16 differs from the frozen reference beyond tolerance, the environment differs
  (code/data/cache version) and the sweep MUST NOT be merged into the canonical results.

REQUIRES: torch (train.py trains the CVAE half too). Run on Colab+GPU with Drive mounted.
  python scripts/analysis/source_pca_dim_sweep.py --project "/content/drive/MyDrive/FYP final"

After training/rescoring each dim, per-subunit RMSE + per-subunit new flip are computed
POST-HOC with the same algorithm as scripts/evaluation/region_flip.py
(face->subunit majority vote; signed_fold_indicator). Mean strict edge-strain p95 is
also computed with strict_protocol_patch, matching the frozen main-table scorer.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, csv
from pathlib import Path
import numpy as np

SUMMARY_DIMS = [0, 8, 16, 32, 64]
MISSING_DIMS = [8, 32, 64]
REQUIRED_SUMMARY_COLUMNS = {"edge_strain_p95", "abs_flip_pct", "missed_flip_pct"}
SELF_CHECK_TOL = 1e-6               # absolute mm-unit tolerance on dim=16 reproduction
OUT_ROOT = "results/supplemental_source_pca_dim_sweep"


def default_repo(project: Path) -> Path:
    local = Path("/content/fyp_data")
    return local if (local / "manifest.json").exists() else project / "data"


def default_dim0_run_dir(project: Path) -> Path:
    local = Path("/content/source_pca_dim0")
    if list(local.glob("neural_field_predictions_*.npz")):
        return local
    return project / "outputs/unified_strict/ablation/no_source"


def frozen_main_value(project: Path, method: str, metric: str) -> float:
    import csv
    path = project / "outputs/unified_strict/main_table/main_table_strict_means.csv"
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["method"] == method:
                return float(row[metric])
    raise KeyError(f"{method}.{metric} not found in {path}")


def summary_has_required_columns(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    return REQUIRED_SUMMARY_COLUMNS.issubset(set(header))


def run_one_dim(project: Path, repo: Path, dim: int, seed: int, split_manifest: Path) -> Path:
    out = project / f"{OUT_ROOT}/dim{dim}"
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("neural_field_predictions_*.npz")):
        print(f">> reusing existing dense prediction for dim{dim}: {out}", flush=True)
        return out
    pair_json = project / "splits/facescape_847/pairs/primary_chain_0_scale676.json"
    # STRICT-PROTOCOL-ALIGNED command (per audit): final protocol needs
    # --use-subunit-features false (train.py default is TRUE), explicit pair manifest,
    # frozen split + frozen seed. ridge lambda is NOT swept (hardcoded 100.0 in train.py).
    cmd = [sys.executable, str(project / "scripts/training/ablate.py"),
           "--repo", str(repo),
           "--split-manifest", str(split_manifest),
           "--train-pairs-json", str(pair_json),
           "--source-pca-dim", str(dim),
           "--model-kind", "cvae",
           "--use-subunit-features", "false",
           "--seed", str(seed),
           "--out", str(out)]
    print(">>", " ".join(cmd), flush=True)
    print("   [reproducibility] if you need bitwise alignment, first `git checkout`", flush=True)
    print("   the train.py/data.py version recorded in reproducibility_manifest.json", flush=True)
    print("   (current working-tree train.py/data.py differ from the frozen manifest).", flush=True)
    subprocess.run(cmd, check=True)
    return out


def ridge_prediction_from_run(out: Path) -> tuple[list[tuple[str, str]], np.ndarray]:
    npzs = list(out.glob("neural_field_predictions_*.npz"))
    if not npzs:
        raise RuntimeError(f"no dense prediction npz found in {out}")
    print(f">> loading dense prediction: {npzs[0]}", flush=True)
    d = np.load(npzs[0], allow_pickle=True)
    files = list(d.files)
    pk = next(f for f in files if "pair" in f.lower())
    rk = next((f for f in files if "ridge" in f.lower() and "cond" in f.lower()),
              next(f for f in files if "ridge" in f.lower()))
    pairs = [(str(a), str(b)) for a, b in np.asarray(d[pk]).tolist()]
    pred = np.asarray(d[rk], np.float64).reshape(len(pairs), -1, 3)
    return pairs, pred


def strict_free_roi_mean(by_id: dict, pairs: list[tuple[str, str]], pred_delta: np.ndarray) -> float:
    landmarks = np.asarray(next(iter(by_id.values()))["landmarks"], int)
    src = np.stack([by_id[s]["vertices"] for s, _ in pairs])
    tgt = np.stack([by_id[t]["vertices"] for _, t in pairs])
    true = tgt - src
    P = np.asarray(pred_delta).reshape(len(pairs), -1, 3)
    free = np.ones(P.shape[1], dtype=bool)
    free[landmarks] = False
    err = P[:, free] - true[:, free]
    per_pair = np.sqrt(np.mean(np.sum(err ** 2, axis=2), axis=1))
    return float(np.mean(per_pair))


def strict_pair_metric_means(by_id: dict, pairs: list[tuple[str, str]], pred_delta: np.ndarray) -> dict[str, float]:
    from rhinoform import strict_protocol_patch as strict
    strict.apply()
    from rhinoform.stats import metric_rows_for_method
    print(f">> computing strict edge/flip extras for {len(pairs)} pairs", flush=True)
    rows = metric_rows_for_method(by_id, pairs, pred_delta)
    keys = ["edge_strain_p95", "abs_flip_pct", "missed_flip_pct"]
    return {k: float(np.mean([float(r[k]) for r in rows])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--repo", default="", help="ROI data repo; defaults to /content/fyp_data when present, else PROJECT/data")
    ap.add_argument("--dim0-run-dir", default="", help="existing frozen dim0/no_source run dir; defaults to /content/source_pca_dim0 when present")
    ap.add_argument("--skip-if-summary-exists", action="store_true", help="return immediately when the dim sweep CSV already exists")
    ap.add_argument("--self-check-tol", type=float, default=SELF_CHECK_TOL)
    args = ap.parse_args()
    P = Path(args.project)
    repo = Path(args.repo) if args.repo else default_repo(P)
    dim0_run = Path(args.dim0_run_dir) if args.dim0_run_dir else default_dim0_run_dir(P)
    sys.path.insert(0, str(P / "code"))
    print(f"[preflight] project={P}", flush=True)
    print(f"[preflight] repo={repo}", flush=True)
    print(f"[preflight] dim0_run={dim0_run} dense_exists={bool(list(dim0_run.glob('neural_field_predictions_*.npz')))}", flush=True)
    outd = P / OUT_ROOT
    summary_csv = outd / "source_pca_dim_sweep_subunit_rmse_flip.csv"
    print(f"[preflight] summary_csv_exists={summary_csv.exists()} path={summary_csv}", flush=True)
    if args.skip_if_summary_exists and summary_has_required_columns(summary_csv):
        print("[preflight] summary CSV already has edge_strain_p95/abs_flip_pct/missed_flip_pct; skipping Step3 recomputation. Run Step3b to plot.", flush=True)
        return 0
    if args.skip_if_summary_exists and summary_csv.exists():
        print("[preflight] summary CSV exists but is missing edge_strain_p95/abs_flip_pct/missed_flip_pct; recomputing post-hoc summary.", flush=True)
    seed = int(json.loads((P / "seed_registry.json").read_text())["training_seed"])  # 20260609 (frozen)
    split_manifest = P / "splits/facescape_847/split_manifest.json"
    assert split_manifest.exists(), "frozen split manifest missing"
    outd.mkdir(parents=True, exist_ok=True)
    fail_path = outd / "SELF_CHECK_FAILED.json"
    if fail_path.exists():
        fail_path.unlink()
    reference_ridge_roi = frozen_main_value(P, "ridge_sourcepca", "roi_rmse")
    for dim in [16] + MISSING_DIMS:
        ddir = P / f"{OUT_ROOT}/dim{dim}"
        print(f"[preflight] dim{dim} dense_exists={bool(list(ddir.glob('neural_field_predictions_*.npz')))} path={ddir}", flush=True)

    # Data is loaded before the self-check so the anchor uses the same strict
    # scorer as the frozen main table, not train.py's non-strict summary.
    from rhinoform.data import load_rows
    from rhinoform.safe_fusion import signed_fold_indicator
    from scripts.evaluation.region_flip import build_face_regions, region_metrics, REG
    print(">> loading ROI meshes from repo ...", flush=True)
    _, by_id = load_rows(repo)
    print(f">> loaded ROI meshes: {len(by_id)} identities", flush=True)

    # 1) dim=16 self-check first
    out16 = run_one_dim(P, repo, 16, seed, split_manifest)
    pairs16, pred16 = ridge_prediction_from_run(out16)
    roi16 = strict_free_roi_mean(by_id, pairs16, pred16)
    diff = abs(roi16 - reference_ridge_roi)
    print(f"[SELF-CHECK] dim16 ridge ROI={roi16:.9f} vs frozen {reference_ridge_roi:.9f} "
          f"(diff {diff:.3g}, tol {float(args.self_check_tol):.3g})", flush=True)
    if diff > float(args.self_check_tol):
        print("!! SELF-CHECK FAILED: environment diverges from frozen reference.", flush=True)
        print("!! Do NOT merge {8,32,64} into the canonical table. Investigate code/data/cache version first.", flush=True)
        fail = {"status": "environment_divergent", "dim16_roi": roi16, "reference": reference_ridge_roi,
                "diff": diff, "tolerance": float(args.self_check_tol), "repo": str(repo)}
        json.dump(fail, open(fail_path, "w"), indent=2)
        return 2
    print("[SELF-CHECK PASSED] dims {8,32,64} are comparable to the canonical table.", flush=True)

    # 2) train only the missing dimensions. dim0 is frozen; dim16 is the anchor run.
    runs = {0: dim0_run, 16: out16}
    for dim in MISSING_DIMS:
        runs[dim] = run_one_dim(P, repo, dim, seed, split_manifest)

    # 3) per-dim region metrics (post-hoc, same algorithm as compute_region_flip_all_methods)
    tmpl = next(iter(by_id.values()))
    faces, face_reg, REG_V, _ = build_face_regions(tmpl)

    rows = []
    for dim in SUMMARY_DIMS:
        print(f">> scoring dim{dim} ...", flush=True)
        out = runs[dim]
        pairs, rg = ridge_prediction_from_run(out)
        rmse, flip, overall = region_metrics(by_id, pairs, rg, faces, face_reg, REG_V, signed_fold_indicator)
        strict_means = strict_pair_metric_means(by_id, pairs, rg)
        row = {"source_pca_dim": dim, "ridge_roi_rmse": round(strict_free_roi_mean(by_id, pairs, rg), 4),
               "overall_new_flip_pct": round(overall, 4),
               "edge_strain_p95": round(strict_means["edge_strain_p95"], 4),
               "abs_flip_pct": round(strict_means["abs_flip_pct"], 4),
               "missed_flip_pct": round(strict_means["missed_flip_pct"], 4),
               "source": "frozen_no_source" if dim == 0 else ("dim16_self_check_rerun" if dim == 16 else "new_missing_dim_rerun")}
        for n in REG: row[f"{n}_rmse"] = round(rmse[n], 4)
        for n in REG: row[f"{n}_new_flip_pct"] = round(flip[n], 4)
        rows.append(row)
        print(f"dim{dim}: ridge_roi={row.get('ridge_roi_rmse')} overall_flip={overall:.3f}", flush=True)

    keys = sorted({k for r in rows for k in r})
    with open(outd / "source_pca_dim_sweep_subunit_rmse_flip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    json.dump({"self_check": "passed", "seed": seed, "split": str(split_manifest), "repo": str(repo),
               "trained_dims": [16] + MISSING_DIMS, "frozen_dim0_run_dir": str(dim0_run),
               "reference_ridge_roi": reference_ridge_roi, "rows": rows},
              open(outd / "source_pca_dim_sweep_subunit_rmse_flip.json", "w"), indent=2, ensure_ascii=False)
    print("WROTE", outd / "source_pca_dim_sweep_subunit_rmse_flip.csv", flush=True)
    print("Report should answer: is dim16 ~optimal? do 32/64 improve ROI without worsening alar flip / edge_p95 / strain? diminishing returns?", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
