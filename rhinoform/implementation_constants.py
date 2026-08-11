from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from . import train
from .repro import artifact_metadata, atomic_write_json


def count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["category", "item", "value", "source", "freeze_status"])
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/constants")
    parser.add_argument("--repo", default="data")
    args = parser.parse_args()

    vertex_dim_with_subunit = 3 + 3 + 5 + 9
    vertex_dim_no_subunit = 3 + 3 + 9
    cond_dim = 9 * 3 + 16
    obs_dim = 16
    hidden = 128
    latent = 8
    cvae_no_subunit = train.NeuralFieldCVAE(vertex_dim_no_subunit, cond_dim, obs_dim, latent_dim=latent, hidden=hidden)
    cvae_with_subunit = train.NeuralFieldCVAE(vertex_dim_with_subunit, cond_dim, obs_dim, latent_dim=latent, hidden=hidden)

    rows = [
        {"category": "scope", "item": "dataset", "value": "FaceScape only; neutral expression; registered 26317-vertex topology", "source": "GOAL_experimental_layer.md §0", "freeze_status": "required"},
        {"category": "roi", "item": "whole_roi_vertices", "value": "3934", "source": "roi/vertices.json", "freeze_status": "frozen"},
        {"category": "roi", "item": "subunit_role", "value": "diagnostic metrics/visualisation only; not model input for final runs", "source": "GOAL_experimental_layer.md §F.bis", "freeze_status": "required"},
        {"category": "model", "item": "use_subunit_features_final", "value": "false", "source": "rhinoform/train.py --use-subunit-features", "freeze_status": "required"},
        {"category": "model", "item": "vertex_features_without_subunits", "value": "normalised coordinates 3 + vertex normals 3 + landmark distances 9 = 15", "source": "rhinoform/train.py build_static_vertex_features", "freeze_status": "required"},
        {"category": "model", "item": "vertex_features_legacy_with_subunits", "value": "normalised coordinates 3 + vertex normals 3 + subunit one-hot 5 + landmark distances 9 = 20", "source": "rhinoform/train.py build_static_vertex_features", "freeze_status": "legacy_only"},
        {"category": "model", "item": "condition_dim", "value": "43 = 9 landmark deltas x 3 + 16 source-PCA components", "source": "rhinoform/train.py pair_conditions", "freeze_status": "required"},
        {"category": "model", "item": "source_pca_dim", "value": "16; fitted on training identities/pairs of each scale only", "source": "GOAL_experimental_layer.md §A.2; rhinoform/train.py", "freeze_status": "required"},
        {"category": "model", "item": "delta_pca_dim", "value": "16 observation components for neural-field CVAE posterior", "source": "rhinoform/train.py default", "freeze_status": "current_default"},
        {"category": "model", "item": "latent_dim", "value": str(latent), "source": "rhinoform/train.py default", "freeze_status": "current_default"},
        {"category": "model", "item": "hidden_width", "value": str(hidden), "source": "rhinoform/train.py default", "freeze_status": "current_default"},
        {"category": "model", "item": "activation_norm", "value": "SiLU + LayerNorm in prior/encoder/field MLP blocks", "source": "rhinoform/train.py NeuralFieldCVAE", "freeze_status": "current_default"},
        {"category": "model", "item": "param_count_no_subunit_features", "value": str(count_params(cvae_no_subunit)), "source": "instantiated from rhinoform/train.py", "freeze_status": "required"},
        {"category": "model", "item": "param_count_legacy_with_subunit_features", "value": str(count_params(cvae_with_subunit)), "source": "instantiated from rhinoform/train.py", "freeze_status": "legacy_only"},
        {"category": "training", "item": "epochs", "value": "180", "source": "rhinoform/train.py default", "freeze_status": "current_default"},
        {"category": "training", "item": "batch_size", "value": "16", "source": "rhinoform/train.py default", "freeze_status": "current_default"},
        {"category": "training", "item": "optimizer", "value": "AdamW, lr=1e-3, weight_decay=1e-4, grad_clip=1.0", "source": "rhinoform/train.py train_field", "freeze_status": "current_default"},
        {"category": "training", "item": "early_stopping", "value": "eval_every=10, patience=50, select best validation ROI RMSE", "source": "rhinoform/train.py train_field", "freeze_status": "current_default"},
        {"category": "training", "item": "loss_weights", "value": "dense=1.0, ctrl=5.0, edge=0.1, lap=0.0, strain=0.0 unless regularity sweep", "source": "rhinoform/train.py defaults", "freeze_status": "current_default"},
        {"category": "training", "item": "kl_beta_warmup", "value": "beta=1e-4, kl_warmup=80", "source": "rhinoform/train.py defaults", "freeze_status": "current_default"},
        {"category": "linear", "item": "ridge_lambda", "value": "100.0", "source": "rhinoform/train.py ridge_fit calls", "freeze_status": "current_default"},
        {"category": "linear", "item": "conditioning_standardisation", "value": "z-score cond = (cond - cond_mean) / max(cond_std, 1e-6); mean/std fitted on the training pairs of each scale only and reused unchanged for validation/test", "source": "rhinoform/train.py pair_conditions (lines 96-99)", "freeze_status": "required"},
        {"category": "geometric", "item": "default_handle_weight", "value": "1000.0", "source": "rhinoform/baselines.py default", "freeze_status": "legacy_default"},
        {"category": "geometric", "item": "default_system_ridge", "value": "1e-8", "source": "rhinoform/baselines.py default", "freeze_status": "legacy_default"},
        {"category": "geometric", "item": "default_arap_iterations", "value": "3", "source": "rhinoform/baselines.py default", "freeze_status": "legacy_default"},
        {"category": "geometric", "item": "boundary_anchoring", "value": "handles = the 9 nasal landmarks only (soft constraint via handle_weight); no extra fixed boundary ring", "source": "rhinoform/geometry.py solve_linear_handle_baseline / arap_predict_one (selector_matrix(landmarks))", "freeze_status": "required"},
        {"category": "geometric", "item": "final_tuning_grid", "value": "ARAP handle-weight {1e2,1e3,1e4,1e5} x iters {3,5,10,20}; Laplacian/Bi-Laplacian handle-weight {1e2,1e3,1e4,1e5} x ridge {1e-10,1e-8,1e-6,1e-4}", "source": "GOAL_experimental_layer.md §9 U3", "freeze_status": "required"},
        {"category": "seeds", "item": "seed_registry", "value": "global/split/training/bootstrap/noise/figure_case=20260609; chain seeds 20260609..20260613", "source": "seed_registry.json", "freeze_status": "required"},
        {"category": "units", "item": "coordinate_transform_in_build_data", "value": "none; ROI crop is the raw OBJ coordinates of models_reg/1_neutral.obj indexed by roi/vertices.json (no scaling applied)", "source": "scripts/data_processing/build_data.py docstring and implementation", "freeze_status": "verified_local"},
        {"category": "units", "item": "official_unit_declaration", "value": "The official FaceScape documentation for the registered TU models (doc_tu_model.md, doc_mview_model.md) specifies NO physical/metric unit for the registered coordinates; the dataset is non-metric (uncalibrated capture). Therefore NO millimetre factor is defined or applied.", "source": "https://github.com/zhuhao-nju/facescape doc_tu_model.md / doc_mview_model.md (official); corroborated by MICA 'Towards Metrical Reconstruction of Human Faces' ECCV 2022", "freeze_status": "resolved_official_no_metric_unit"},
        {"category": "units", "item": "reported_metric_unit", "value": "dimensionless FaceScape coordinate units (no official metric scale). RMSE and noise sigma are reported in these units; no mm conversion is claimed because no official scale exists and the unit is not self-estimated from the meshes.", "source": "GOAL_experimental_layer.md Goal F + official-source check", "freeze_status": "required"},
        {"category": "model", "item": "source_pca_explained_variance", "value": "data-dependent: recorded per run from the scale-specific train-only source-PCA fit (not a static constant; emitted in each run's artifacts, never estimated here)", "source": "rhinoform/train.py fit_truncated_pca", "freeze_status": "recorded_per_run"},
    ]

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "implementation_constants.csv", rows)
    md = [
        "# Implementation Constants",
        "",
        "Generated from code defaults and the binding experimental specification.",
        "",
        "| Category | Item | Value | Source | Freeze status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        md.append("| " + " | ".join(str(r[k]).replace("|", "\\|") for k in ["category", "item", "value", "source", "freeze_status"]) + " |")
    (out_dir / "implementation_constants.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    report = {
        "rows": rows,
        "metadata": artifact_metadata(
            repo_root=Path(__file__).resolve().parents[1],
            command_args=args,
            input_manifest_path=Path(args.repo) / "manifest.json",
            seed=20260609,
            data_root_identifier=args.repo,
        ),
    }
    atomic_write_json(out_dir / "implementation_constants.json", report)
    print(json.dumps({"out_dir": str(out_dir), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

