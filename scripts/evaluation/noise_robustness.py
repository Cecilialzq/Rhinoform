from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from rhinoform import train
from rhinoform.cvae import pca_transform
from rhinoform.data import edge_index, load_rows, ordered_pairs, ridge_predict, split_ids
from rhinoform.geometry import solve_linear_handle_baseline, uniform_laplacian
from rhinoform.baselines import arap_predict_vectorised
from rhinoform.repro import SEED_REGISTRY, artifact_metadata, atomic_write_json
from rhinoform.stats import metric_rows_for_method


METRICS = ["roi_rmse", "landmark_rmse", "dorsum_rmse", "tip_rmse", "edge_strain_p95", "normal_flip_pct"]


def controls_and_sources(by_id: dict, pairs: list[tuple[str, str]], landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    controls = []
    source_flat = []
    for s, t in pairs:
        src = by_id[s]["vertices"]
        tgt = by_id[t]["vertices"]
        controls.append((tgt - src)[landmarks])
        source_flat.append(src.reshape(-1))
    return np.stack(controls, axis=0), np.stack(source_flat, axis=0)


def noisy_condition(
    controls: np.ndarray,
    source_flat: np.ndarray,
    source_pca: dict[str, np.ndarray],
    cond_mean: np.ndarray,
    cond_std: np.ndarray,
) -> np.ndarray:
    code = pca_transform(source_flat, source_pca)
    cond = np.concatenate([controls.reshape(len(controls), -1), code], axis=1)
    return ((cond - cond_mean) / cond_std).astype(np.float32)


def predict_cvae(package: dict, vertex_feat: np.ndarray, cond: np.ndarray, device: str) -> np.ndarray:
    obs_dim = int(package["delta_pca"]["components"].shape[0])
    model = train.NeuralFieldCVAE(
        int(package["vertex_feature_dim"]),
        cond.shape[1],
        obs_dim,
        latent_dim=int(package["args"].get("latent_dim", 8)),
        hidden=int(package["args"].get("hidden", 128)),
    )
    model.load_state_dict(package["cvae_state_dict"])
    model.to(torch.device(device))
    return train.predict_field(model, vertex_feat, cond, is_cvae=True)


def aggregate(rows: list[dict[str, float | str]]) -> dict[str, float]:
    return {m: float(np.mean([float(r[m]) for r in rows])) for m in METRICS}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    return path


def load_geometric_selection(path: Path) -> dict:
    report = json.loads(require(path).read_text(encoding="utf-8"))
    selected = report.get("selected", {})
    missing = [name for name in ("arap", "laplacian", "bilaplacian") if name not in selected]
    if missing:
        raise FileNotFoundError(f"missing cache: selected geometric configs {missing} in {path}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="data")
    parser.add_argument("--model-package", required=True)
    parser.add_argument("--out", default="results/noise_robustness")
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--levels", default="0,0.05,0.10,0.20")
    parser.add_argument("--noise-seeds", default="20260609,20260610,20260611")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-arap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-bilaplacian", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--geometric-tuning", default="results/geometric_tuning/geometric_validation_tuning.json")
    parser.add_argument("--split-manifest", default="")
    args = parser.parse_args()

    package_path = Path(args.model_package)
    if not package_path.exists():
        raise FileNotFoundError(f"missing cache: {package_path}")
    repo = Path(args.repo)
    _, by_id = load_rows(repo)
    package = torch.load(package_path, map_location="cpu", weights_only=False)
    val_pairs = [(str(a), str(b)) for a, b in package.get("val_pairs", ordered_pairs(split_ids(by_id, "clean-prior validation"), None, args.seed))]
    test_pairs = [(str(a), str(b)) for a, b in package.get("test_pairs", ordered_pairs(split_ids(by_id, "main test"), None, args.seed))]
    train_ids = [str(x) for x in package["train_ids"]]
    template = next(iter(by_id.values()))
    landmarks = template["landmarks"]
    faces = template["faces"]
    n = template["vertices"].shape[0]
    lap = uniform_laplacian(n, faces)
    geometric = load_geometric_selection(Path(args.geometric_tuning))

    val_ctrl, _ = controls_and_sources(by_id, val_pairs, landmarks)
    median_control_delta = float(np.median(np.linalg.norm(val_ctrl, axis=2)))
    test_ctrl, test_source_flat = controls_and_sources(by_id, test_pairs, landmarks)
    vertex_feat, static = train.build_static_vertex_features(
        by_id, train_ids, use_subunit_features=bool(package.get("use_subunit_features", False))
    )
    train.assert_feature_template_package(package, static, "Model package")

    levels = [float(x) for x in args.levels.split(",") if x]
    seeds = [int(x) for x in args.noise_seeds.split(",") if x]
    summary_rows = []
    out_dir = Path(args.out)
    for level in levels:
        active_seeds = seeds
        for noise_seed in active_seeds:
            rng = np.random.default_rng(noise_seed)
            sigma = float(level * median_control_delta)
            noisy_ctrl = test_ctrl + rng.normal(0.0, sigma, size=test_ctrl.shape)
            cond = noisy_condition(noisy_ctrl, test_source_flat, package["source_pca"], package["cond_mean"], package["cond_std"])
            ridge = ridge_predict(cond, package["ridge_cond"])
            cvae = predict_cvae(package, vertex_feat, cond, args.device)
            alpha = float(package.get("selected_alpha") if package.get("selected_alpha") is not None else 0.0)
            hybrid = (1.0 - alpha) * ridge + alpha * cvae
            lap_cfg = geometric["laplacian"]
            bilap_cfg = geometric["bilaplacian"]
            arap_cfg = geometric["arap"]
            laplacian = solve_linear_handle_baseline(
                noisy_ctrl,
                landmarks,
                n,
                lap,
                float(lap_cfg["handle_weight"]),
                float(lap_cfg["system_ridge"]),
            )
            method_preds = {
                "ridge_sourcepca": ridge,
                "cvae_only": cvae,
                "hybrid": hybrid,
                "laplacian": laplacian,
            }
            if args.include_bilaplacian:
                method_preds["bilaplacian_appendix"] = solve_linear_handle_baseline(
                    noisy_ctrl,
                    landmarks,
                    n,
                    lap @ lap,
                    float(bilap_cfg["handle_weight"]),
                    float(bilap_cfg["system_ridge"]),
                )
            if args.include_arap:
                sources = [by_id[s]["vertices"] for s, _ in test_pairs]
                # ARAP must be initialised from its OWN handle-weight solve to match the
                # validation-tuned learning-curve reference (tune_baselines.py). Using the
                # Laplacian-config solve (`laplacian`) as the init gave a different ARAP
                # result (1.722 vs 1.737 f.u. at zero noise); this restores consistency.
                arap_hw = float(arap_cfg["handle_weight"])
                arap_init = solve_linear_handle_baseline(noisy_ctrl, landmarks, n, lap, arap_hw, 1e-8)
                method_preds["arap"] = arap_predict_vectorised(
                    sources,
                    noisy_ctrl,
                    landmarks,
                    faces,
                    lap,
                    arap_init,
                    arap_hw,
                    float(arap_cfg.get("system_ridge") or 1e-8),
                    int(arap_cfg["arap_iter"]),
                )
            for method, pred in method_preds.items():
                rows = metric_rows_for_method(by_id, test_pairs, pred)
                write_csv(out_dir / "pair_metrics" / f"noise_c{level:g}_seed{noise_seed}_{method}.csv", rows)
                agg = aggregate(rows)
                summary_rows.append(
                    {
                        "method": method,
                        "noise_level_c": level,
                        "noise_fraction": level,
                        "noise_seed": noise_seed,
                        "sigma": sigma,
                        **agg,
                    }
                )
    write_csv(out_dir / "noise_robustness_summary.csv", summary_rows)
    report = {
        "median_control_delta_from_validation": median_control_delta,
        "levels": levels,
        "noise_seeds": seeds,
        "geometric_tuning": str(args.geometric_tuning),
        "geometric_selected": geometric,
        "model_package": str(package_path),
        "rows": summary_rows,
        "metadata": artifact_metadata(
            repo_root=Path(__file__).resolve().parents[1],
            command_args=args,
            input_manifest_path=repo / "manifest.json",
            split_manifest_path=Path(args.split_manifest) if args.split_manifest else None,
            seed=args.seed,
            data_root_identifier=repo.name,
        ),
    }
    atomic_write_json(out_dir / "noise_robustness_summary.json", report)
    print(json.dumps({"out": str(out_dir), "rows": len(summary_rows), "median_control_delta": median_control_delta}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
