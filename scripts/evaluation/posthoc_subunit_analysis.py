"""Post-hoc five-subunit, spatial and tail analysis of the frozen final rerun."""
from __future__ import annotations

import argparse, csv, json, os, pickle, subprocess, sys
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rhinoform.baselines import arap_predict_vectorised
from rhinoform.confirmation import validate_frozen_classical_configuration
from rhinoform.data import load_rows, ridge_predict
from rhinoform.geometry import solve_linear_handle_baseline, uniform_laplacian
from rhinoform.rbsr_calibration import enforce_exact_controls
from rhinoform.regional_analysis import (
    REGIONAL_METRICS, REGIONS, build_region_topology, empty_spatial_accumulator,
    regional_metric_row_sets,
)
from rhinoform.repro import (
    atomic_savez_compressed, atomic_write_csv, atomic_write_json, sha256_file,
    valid_sha256_sidecar, validate_torch_artifact, write_sha256_sidecar,
)
from rhinoform.safe_fusion import project_residual_no_new_ridge_folds
from rhinoform.strict_protocol_patch import strict_metric_rows
from rhinoform.train import (
    NeuralFieldCVAE,
    assert_feature_template_package,
    build_static_vertex_features,
    pair_conditions,
    predict_field,
)
from rhinoform.train_rbsr_gate import SpatialRiskGate, predict_gate_batches
from scripts.evaluation.direct_paired_statistics import holm, safe_wilcoxon
from scripts.evaluation.lamm_validation_reference import (
    relocate_comparable_model_config,
    require_frozen_result_artifact,
)

METHODS = ("ridge", "certified_rbsr", "cvae", "hybrid", "lamm", "laplacian", "bilaplacian", "arap")
INFERENTIAL_REGIONAL_METRICS = (
    "free_rmse", "edge_strain_p95", "normal_flip_pct", "abs_flip_pct",
    "missed_flip_pct",
)
GLOBAL_CHECK_METRICS = (
    "roi_rmse", "landmark_rmse", "dorsum_rmse", "tip_rmse", "edge_strain_p95",
    "normal_flip_pct", "abs_flip_pct", "missed_flip_pct", "target_flip_pct",
)
LAMM_COMMIT = "87354c05dec341c6d8dd319665dd52553fb03084"


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_valid(path: Path, label: str | None = None):
    if not valid_sha256_sidecar(path):
        raise RuntimeError(f"Invalid {label or 'artifact'} or SHA sidecar: {path}")


def output_csv_valid(path: Path, rows: int) -> bool:
    try:
        return valid_sha256_sidecar(path) and len(read_csv(path)) == rows
    except (OSError, csv.Error):
        return False


def regional_complete(out: Path, stage: str, stem: str, method: str, n_pairs: int) -> bool:
    root = out / "chunks" / stage
    rows = n_pairs * len(REGIONS)
    return (
        output_csv_valid(root / f"{stem}_{method}.csv", rows)
        and output_csv_valid(root / f"{stem}_{method}_unique_majority_sensitivity.csv", rows)
        and valid_sha256_sidecar(root / f"{stem}_{method}_spatial.npz")
    )


def compare_global(observed, canonical_path: Path, *, start: int = 0, atol: float = 2e-6):
    require_valid(canonical_path, "canonical global pair metrics")
    all_expected = read_csv(canonical_path)
    if len(all_expected) != 9900:
        raise AssertionError(f"Canonical row-count mismatch: {canonical_path}")
    expected = all_expected[start:start + len(observed)]
    if len(expected) != len(observed):
        raise AssertionError(f"Canonical slice mismatch: {canonical_path}")
    for index, (left, right) in enumerate(zip(expected, observed)):
        if (left["source_id"], left["target_id"]) != (str(right["source_id"]), str(right["target_id"])):
            raise AssertionError(f"Pair mismatch {canonical_path}:{index}")
        for metric in GLOBAL_CHECK_METRICS:
            if metric in left and not np.isclose(float(left[metric]), float(right[metric]), atol=atol, rtol=0.0):
                raise AssertionError(f"Global self-check failed {canonical_path.name}:{index}:{metric}")


class Context:
    def __init__(self, args):
        self.args, self.results, self.output = args, args.results, args.out
        self.policy_path = self.results / "protocol/FINAL_RERUN_HOLDOUT_POLICY_FREEZE.json"
        self.split_path = self.results / "protocol/final_rerun_holdout_split_manifest.json"
        self.freeze_path = self.results / "ALL_MODELS_FROZEN_BEFORE_TEST.json"
        self.evidence_path = self.results / "FINAL_CONFIRMATION_EVIDENCE.json"
        self.base_path = self.results / "rbsr/seed20260609/base/neural_field_model_package_cvae_ew0p1_lw0.pt"
        for path in (self.policy_path, self.split_path, self.freeze_path, self.evidence_path):
            require_valid(path)
        if not validate_torch_artifact(
            self.base_path, required_keys=("ridge_cond", "source_pca", "cond_mean", "cond_std", "test_pairs"),
            repair_sidecar=False,
        ):
            raise RuntimeError(f"Invalid frozen base package: {self.base_path}")
        self.policy = json.loads(self.policy_path.read_text())
        self.split = json.loads(self.split_path.read_text())
        self.freeze = json.loads(self.freeze_path.read_text())
        configured_lamm_root = os.environ.get("RHINOFORM_LAMM_ARTIFACT_ROOT")
        self.lamm_artifact_root = (
            Path(configured_lamm_root).resolve()
            if configured_lamm_root
            else (self.results / "lamm/seed20260609").resolve()
        )
        if self.freeze["rbsr_base_sha256"] != sha256_file(self.base_path):
            raise RuntimeError("Base package no longer matches freeze")
        for relative in (
            "rhinoform/baselines.py", "rhinoform/geometry.py", "rhinoform/safe_fusion.py",
            "rhinoform/strict_protocol_patch.py", "experiments/lamm/run_lamm_facescape.py",
        ):
            if sha256_file(args.repo_root / relative) != self.freeze["implementation_hashes"][relative]:
                raise RuntimeError(f"Frozen numerical implementation changed: {relative}")
        for name in ("RBSR_TEST_ACCESS_RECEIPT.json", "LAMM_TEST_ACCESS_RECEIPT.json", "CLASSICAL_TEST_ACCESS_RECEIPT.json"):
            path = self.results / name
            require_valid(path)
            receipt = json.loads(path.read_text())
            if receipt.get("status") != "COMPLETE" or receipt.get("test_access_count") != 1:
                raise RuntimeError(f"Original receipt incomplete: {path}")
        self.base = torch.load(self.base_path, map_location="cpu", weights_only=False)
        self.pairs = [(str(a), str(b)) for a, b in self.base["test_pairs"]]
        test_ids = {str(value) for value in self.split["test_ids"]}
        if len(self.pairs) != 9900 or {x for pair in self.pairs for x in pair} != test_ids:
            raise RuntimeError("Frozen base package pair set mismatch")
        _, self.by_id = load_rows(args.repo, allowed_ids=test_ids)
        template = next(iter(self.by_id.values()))
        self.topology = build_region_topology(
            template["subunits"], template["faces"], template["landmarks"], len(template["vertices"])
        )
        declared = json.loads((args.repo_root / "roi/subunits.json").read_text())
        for index, name in enumerate(REGIONS):
            expected = np.sort(np.asarray(declared["groups"][name]["local_roi_indices"], dtype=np.int64))
            if not np.array_equal(expected, self.topology.region_vertices[index]):
                raise RuntimeError(f"Mesh mask differs from frozen subunits: {name}")


def freeze_protocol(context: Context):
    paths = (
        context.args.repo_root / "rhinoform/baselines.py",
        context.args.repo_root / "rhinoform/confirmation.py",
        context.args.repo_root / "rhinoform/data.py",
        context.args.repo_root / "rhinoform/geometry.py",
        context.args.repo_root / "rhinoform/rbsr_calibration.py",
        context.args.repo_root / "rhinoform/regional_analysis.py",
        context.args.repo_root / "rhinoform/repro.py",
        context.args.repo_root / "rhinoform/safe_fusion.py",
        context.args.repo_root / "rhinoform/strict_protocol_patch.py",
        context.args.repo_root / "rhinoform/train.py",
        context.args.repo_root / "rhinoform/train_rbsr_gate.py",
        context.args.repo_root / "experiments/lamm/run_lamm_facescape.py",
        context.args.repo_root / "requirements-lamm-inference.txt",
        context.args.repo_root / "scripts/evaluation/direct_paired_statistics.py",
        context.args.repo_root / "scripts/evaluation/lamm_validation_reference.py",
        context.args.repo_root / "scripts/evaluation/posthoc_subunit_analysis.py",
        context.args.repo_root / "scripts/evaluation/posthoc_strict_noise_robustness.py",
        context.args.repo_root / "scripts/evaluation/posthoc_qualitative_cases.py",
        context.args.repo_root / "scripts/evaluation/posthoc_personalization_ablation.py",
    )
    payload = {
        "status": "FROZEN_POSTHOC_AFTER_COMPLETED_TEST_BEFORE_REGIONAL_COMPUTATION",
        "claim_boundary": "secondary post-hoc regional analysis; not a new untouched confirmation",
        "original_final_evidence_sha256": sha256_file(context.evidence_path),
        "dense_prediction_policy": (
            "recomputed from repository-frozen checkpoints; no archival dense chunks read"
        ),
        "original_all_models_freeze_sha256": sha256_file(context.freeze_path),
        "original_final_implementation_hashes": context.freeze["implementation_hashes"],
        "split_manifest_sha256": sha256_file(context.split_path),
        "subunit_definition_sha256": sha256_file(context.args.repo_root / "roi/subunits.json"),
        "regions": list(REGIONS), "methods": list(METHODS), "metrics": list(REGIONAL_METRICS),
        "vertex_rule": "the five frozen masks partition all 3,934 ROI vertices exactly",
        "face_rule_reportable": "historical three-vertex vote; argmax tie break in frozen region order",
        "face_rule_sensitivity": "unique 2-of-3 majority; 1-1-1 faces boundary/excluded",
        "edge_rule": "same-region endpoints only; cross-region edges are boundary",
        "formulas": {
            "free_rmse": "sqrt(mean free regional squared Euclidean displacement error)",
            "landmark_rmse": "regional control RMSE after exact target-control overwrite",
            "edge_strain_p95": "P95 intra-region |l_pred-l_source|/max(l_source,1e-12)",
            "normal_flip_pct": "100*mean[(det_pred<0) and not(det_target<0)] on regional faces",
            "abs_flip_pct": "100*mean[det_pred<0]", "missed_flip_pct": "100*mean[target and not prediction fold]",
            "target_flip_pct": "100*mean[det_target<0]",
        },
        "multiple_comparison_families": {
            "all_methods_vs_ridge": "7x5x5=175 Holm tests",
            "certified_rbsr_vs_lamm": "1x5x5=25 Holm tests",
            "inferential_metrics": list(INFERENTIAL_REGIONAL_METRICS),
            "descriptive_invariants_excluded": {
                "landmark_rmse": "identically zero after hard-fixing controls",
                "target_flip_pct": "target-only quantity identical across methods",
            },
        },
        "regional_inference": {
            "pairing": "candidate minus baseline on identical directed identity pairs",
            "primary_test": "two-sided Wilcoxon on source-identity mean paired differences",
            "uncertainty": "10,000 source-identity and target-identity cluster bootstrap replicates",
            "decision_rule": "Holm p<0.05 and both source/target cluster CIs agree in direction",
            "effect_size": "source-identity paired Cohen dz and identity win rate",
        },
        "spatial_diagnostics": {
            "vertex": "mean/worst error", "face": "new/absolute/missed/target flip frequency",
            "edge": "mean/worst relative strain", "tails": "p50/p90/p95/p99/max/top-decile mass",
        },
        "noise_robustness": {
            "pair_panel": "all 4,830 frozen validation pairs", "levels_mm": [0, .25, .5, 1.0],
            "seeds": [20260609, 20260610, 20260611],
            "methods": ["ridge", "certified_rbsr", "lamm", "arap"],
            "metrics": ["roi_rmse", "normal_flip_pct", "edge_strain_p95"],
            "scoring": "unnoised true controls hard-fixed; free RMSE; target-relative new flip",
            "old_results": "absolute-flip/nonzero-landmark legacy results are not reused",
            "operating_point_reselected": False,
            "holm_family": "4x3x3=36 tests",
        },
        "qualitative_cases": {
            "n_cases": 6,
            "selection": (
                "deterministic median, P10/P90 control magnitude, high certificate iterations, "
                "largest RB-SR gain over Ridge, and worst real failure versus min(Ridge,LAMM)"
            ),
            "methods": ["ridge", "certified_rbsr", "lamm"],
            "selection_timing": "manifest frozen before dense re-inference",
            "visualisation": "shared error scale and target-relative new-flip overlay",
            "claim_boundary": "descriptive post-hoc; no inferential claim",
        },
        "personalization_ablation": {
            "pair_panel": "all 4,830 frozen validation pairs",
            "variants": [
                "controls_only_ridge", "source_pca_ridge",
                "mean_source_code", "shuffled_source_code",
            ],
            "primary_family": "3 source-PCA-vs-alternative free-ROI RMSE Holm tests",
            "secondary_family": "6 geometry-metric Holm tests",
            "shuffle": "fixed identity-level cyclic derangement; identical mapping across outgoing pairs",
            "claim_boundary": "mechanistic personalization evidence on validation",
        },
        "implementation_hashes": {p.relative_to(context.args.repo_root).as_posix(): sha256_file(p) for p in paths},
    }
    path = context.output / "POSTHOC_SUBUNIT_ANALYSIS_PROTOCOL_FREEZE.json"
    if path.exists():
        require_valid(path)
        if json.loads(path.read_text()) != payload:
            raise RuntimeError("Existing post-hoc protocol differs; use a new output version")
    else:
        atomic_write_json(path, payload); write_sha256_sidecar(path)
    audit = context.output / "region_topology_audit.json"
    atomic_write_json(audit, context.topology.audit_dict()); write_sha256_sidecar(audit)
    print(json.dumps(context.topology.audit_dict(), indent=2), flush=True)


def write_regional(out, stage, stem, method, by_id, pairs, prediction, topology, offset):
    root = out / "chunks" / stage
    if regional_complete(out, stage, stem, method, len(pairs)):
        print(f"POSTHOC {stage} resume {method} {stem}", flush=True); return
    accumulator = empty_spatial_accumulator(topology)
    sets = regional_metric_row_sets(
        by_id, pairs, prediction, topology, method=method, pair_index_offset=offset,
        spatial_accumulator=accumulator,
    )
    atomic_write_csv(root / f"{stem}_{method}.csv", sets["legacy_majority_vote"])
    atomic_write_csv(root / f"{stem}_{method}_unique_majority_sensitivity.csv", sets["unique_majority_sensitivity"])
    atomic_savez_compressed(root / f"{stem}_{method}_spatial.npz", **{k: np.asarray(v) for k, v in accumulator.items()})
    print(f"POSTHOC {stage} persisted {method} {stem}", flush=True)


def run_neural(context: Context):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Frozen neural post-hoc recomputation requires a GPU runtime")
    train_ids = [str(value) for value in context.base["train_ids"]]
    required_ids = set(train_ids) | {value for pair in context.pairs for value in pair}
    _, by_id = load_rows(context.args.repo, allowed_ids=required_ids)
    features, static = build_static_vertex_features(
        by_id, train_ids,
        use_subunit_features=bool(context.base["use_subunit_features"]),
    )
    assert_feature_template_package(context.base, static, "Frozen base package")
    args = context.base["args"]
    proposer = NeuralFieldCVAE(
        features.shape[1],
        int(np.asarray(context.base["cond_mean"]).shape[1]),
        int(np.asarray(context.base["obs_train_mean"]).shape[1]),
        latent_dim=int(args["latent_dim"]),
        hidden=int(args["hidden"]),
    ).to(device)
    proposer.load_state_dict(context.base["cvae_state_dict"], strict=True)
    proposer.eval()

    gate_path = context.results / "rbsr/seed20260609/gate/rbsr_gate_model.pt"
    require_valid(gate_path, "frozen RB-SR gate")
    if sha256_file(gate_path) != context.freeze["rbsr_gate_sha256"]:
        raise RuntimeError("RB-SR gate differs from the pre-test freeze")
    gate_package = torch.load(gate_path, map_location="cpu", weights_only=False)
    if gate_package.get("feature_template_sha256") != context.base.get("feature_template_sha256"):
        raise RuntimeError("RB-SR gate/base feature-template hashes differ")
    gate_args = gate_package["args"]
    gate = SpatialRiskGate(
        int(gate_package["gate_vertex_dim"]),
        int(gate_package["gate_cond_dim"]),
        int(gate_args["hidden"]),
        float(gate_package.get("initial_gate", context.base.get("selected_alpha") or 0.25)),
    ).to(device)
    gate.load_state_dict(gate_package["gate_state_dict"], strict=True)
    gate.eval()
    vertex_features = torch.as_tensor(features, dtype=torch.float32, device=device)
    center = torch.as_tensor(np.asarray(gate_package["center"]), dtype=torch.float32, device=device)
    landmarks_np = context.topology.landmarks
    landmarks = torch.as_tensor(landmarks_np, dtype=torch.long, device=device)
    projection_basis = None if gate_package.get("projection_basis") is None else torch.as_tensor(
        gate_package["projection_basis"], dtype=torch.float32, device=device
    )
    projection_path = context.results / (
        "rbsr/seed20260609/certified_projection_validation/"
        "RBSR_CERTIFIED_PROJECTION_FREEZE.json"
    )
    require_valid(projection_path, "frozen certified projection")
    if sha256_file(projection_path) != context.freeze["rbsr_projection_freeze_sha256"]:
        raise RuntimeError("Projection freeze differs from the pre-test freeze")
    projection = json.loads(projection_path.read_text(encoding="utf-8"))["selected"]
    global_alpha = float(context.base.get("selected_alpha") or 0.25)
    canonical = context.results / "rbsr/seed20260609/one_shot_test"
    canonical_paths = {
        "ridge": canonical / "pair_metrics_ridge_sourcepca_clean_test.csv",
        "certified_rbsr": canonical / "pair_metrics_rbsr_test.csv",
        "cvae": canonical / "pair_metrics_cvae_clean_test.csv",
        "hybrid": canonical / "pair_metrics_hybrid_validation_selected_clean_test.csv",
    }

    for start in range(0, len(context.pairs), context.args.chunk_pairs):
        stop = min(start + context.args.chunk_pairs, len(context.pairs))
        stem = f"chunk_{start:05d}_{stop:05d}"
        methods = ("ridge", "certified_rbsr", "cvae", "hybrid")
        if all(regional_complete(context.output, "neural", stem, method, stop-start) for method in methods):
            print(f"POSTHOC neural resume all {stem}", flush=True)
            continue
        pairs = context.pairs[start:stop]
        condition, source, _, controls, *_ = pair_conditions(
            by_id, pairs, context.base["source_pca"],
            context.base["cond_mean"], context.base["cond_std"],
        )
        ridge = ridge_predict(condition, context.base["ridge_cond"]).astype(np.float32)
        cvae = predict_field(
            proposer, features, condition, is_cvae=True,
            batch_size=context.args.neural_batch_size,
        ).astype(np.float32)
        hybrid = ridge + global_alpha * (cvae - ridge)
        _, gate_values = predict_gate_batches(
            gate, vertex_features, condition, source, ridge, cvae, center,
            float(gate_package["scale"]), landmarks, controls,
            projection_basis, context.args.neural_batch_size,
        )
        ridge_fixed = enforce_exact_controls(
            ridge, controls, landmarks_np
        ).reshape(len(pairs), -1, 3)
        residual = gate_values[..., None] * (cvae.reshape(len(pairs), -1, 3) - ridge_fixed)
        residual[:, landmarks_np] = 0.0
        projected = []
        for source_mesh, anchor, correction in zip(source, ridge_fixed, residual):
            result = project_residual_no_new_ridge_folds(
                source_mesh, anchor, correction, context.topology.faces,
                landmarks_np,
                attenuation=float(projection["attenuation"]),
                smoothing_steps=int(projection["smoothing_steps"]),
                max_iterations=int(projection["max_iterations"]),
                uniform_steps=int(projection["uniform_steps"]),
            )
            if not result.certified or result.new_vs_ridge_fold_count != 0:
                raise AssertionError("Certified Ridge-fold projection failed")
            projected.append(result.delta)
        predictions = {
            "ridge": ridge,
            "certified_rbsr": np.asarray(projected, dtype=np.float32).reshape(len(pairs), -1),
            "cvae": cvae,
            "hybrid": hybrid,
        }
        for method, prediction in predictions.items():
            observed = strict_metric_rows(by_id, pairs, prediction)
            compare_global(observed, canonical_paths[method], start=start)
            write_regional(
                context.output, "neural", stem, method, by_id, pairs,
                prediction, context.topology, start,
            )
    proposer.to("cpu"); gate.to("cpu"); torch.cuda.empty_cache()


def run_classical(context: Context):
    config = validate_frozen_classical_configuration(context.policy["frozen_classical_configuration"])
    lap = uniform_laplacian(len(context.topology.vertex_region), context.topology.faces)
    canonical_paths = {
        method: context.results / f"classical/identity_bootstrap_pair_metrics_{method}.csv"
        for method in ("laplacian", "bilaplacian", "arap")
    }
    for start in range(0, len(context.pairs), context.args.chunk_pairs):
        stop = min(start + context.args.chunk_pairs, len(context.pairs))
        stem = f"chunk_{start:05d}_{stop:05d}"
        if all(regional_complete(context.output, "classical", stem, m, stop-start) for m in ("laplacian", "bilaplacian", "arap")):
            print(f"POSTHOC classical resume all {stem}"); continue
        pairs = context.pairs[start:stop]
        controls = np.stack([(context.by_id[t]["vertices"]-context.by_id[s]["vertices"])[context.topology.landmarks] for s,t in pairs])
        sources = [np.asarray(context.by_id[s]["vertices"], dtype=np.float64) for s,_ in pairs]
        lc, bc, ac = config["laplacian"], config["bilaplacian"], config["arap"]
        predictions = {
            "laplacian": solve_linear_handle_baseline(controls, context.topology.landmarks, len(context.topology.vertex_region), lap, lc["handle_weight"], lc["system_ridge"]),
            "bilaplacian": solve_linear_handle_baseline(controls, context.topology.landmarks, len(context.topology.vertex_region), lap@lap, bc["handle_weight"], bc["system_ridge"]),
        }
        init = solve_linear_handle_baseline(controls, context.topology.landmarks, len(context.topology.vertex_region), lap, ac["handle_weight"], ac["system_ridge"])
        predictions["arap"] = arap_predict_vectorised(sources, controls, context.topology.landmarks, context.topology.faces, lap, init, ac["handle_weight"], ac["system_ridge"], ac["arap_iter"])
        for method, prediction in predictions.items():
            compare_global(
                strict_metric_rows(context.by_id, pairs, prediction),
                canonical_paths[method], start=start,
            )
            write_regional(context.output, "classical", stem, method, context.by_id, pairs, prediction, context.topology, start)


def run_lamm(context: Context):
    root = context.args.lamm_root
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True).strip()
    if head != LAMM_COMMIT or dirty: raise RuntimeError("LAMM checkout is not pinned and clean")
    if str(root) not in sys.path: sys.path.insert(0, str(root))
    from models import LAMM
    from experiments.lamm.run_lamm_facescape import delta_controls, model_config
    out = context.lamm_artifact_root
    checkpoint, norm, region_file = out/"manipulation_best.pt", out/"train_only_normalisation.npz", out/"region_ids.pickle"
    for path in (checkpoint, norm, region_file):
        require_valid(path)
        require_frozen_result_artifact(
            context.results,
            path,
            manifest_relative_path=f"lamm/seed20260609/{path.name}",
        )
    if sha256_file(checkpoint) != context.freeze["lamm_manipulation_sha256"]:
        raise RuntimeError("LAMM checkpoint differs from the pre-test freeze")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    recorded_config = deepcopy(state["model_config"])
    with region_file.open("rb") as handle: region_ids = pickle.load(handle)
    controls = {}
    for i, indices in region_ids.items():
        values = [int(x) for x in context.topology.landmarks if int(x) in set(map(int, np.asarray(indices).tolist()))]
        if values: controls[int(i)] = values
    raw = model_config(region_file, controls, manipulation=True)
    config = relocate_comparable_model_config(recorded_config, raw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("LAMM regional inference requires GPU")
    model = LAMM(deepcopy(config)).to(device); model.load_state_dict(state["model"], strict=True); model.eval()
    with np.load(norm) as values: mean, std = np.asarray(values["mean"], np.float32), np.asarray(values["std"], np.float32)
    mean_t, std_t = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    canonical = out / "identity_bootstrap_pair_metrics_lamm.csv"
    require_valid(canonical, "canonical LAMM pair metrics")
    for start in range(0, len(context.pairs), context.args.chunk_pairs):
        stop = min(start + context.args.chunk_pairs, len(context.pairs))
        stem=f"chunk_{start:05d}_{stop:05d}"
        if regional_complete(context.output, "lamm", stem, "lamm", stop-start): print(f"POSTHOC lamm resume {stem}"); continue
        pairs=context.pairs[start:stop]; predictions=[]
        with torch.no_grad():
            for begin in range(0,len(pairs),context.args.lamm_batch_size):
                batch=pairs[begin:begin+context.args.lamm_batch_size]
                source_abs=np.stack([context.by_id[s]["vertices"] for s,_ in batch]).astype(np.float32)
                target_abs=np.stack([context.by_id[t]["vertices"] for _,t in batch]).astype(np.float32)
                source=torch.from_numpy((source_abs-mean)/std).to(device); target=torch.from_numpy((target_abs-mean)/std).to(device)
                prediction=model((source,delta_controls(source,target,model)))[-1]
                predictions.append(((prediction*std_t+mean_t).cpu().numpy()-source_abs).reshape(len(batch),-1))
        prediction=np.concatenate(predictions).astype(np.float32)
        compare_global(
            strict_metric_rows(context.by_id,pairs,prediction),
            canonical,start=start,atol=1e-5,
        )
        write_regional(context.output,"lamm",stem,"lamm",context.by_id,pairs,prediction,context.topology,start)
    model.to("cpu"); torch.cuda.empty_cache()


def _bootstrap(values, seed, n_boot):
    rng=np.random.default_rng(seed); means=np.empty(n_boot); block=1000
    for start in range(0,n_boot,block):
        stop=min(start+block,n_boot); index=rng.integers(0,len(values),(stop-start,len(values))); means[start:stop]=np.mean(values[index],axis=1)
    return tuple(map(float,np.percentile(means,[2.5,97.5])))


def _identity_diff(base,candidate,metric,cluster):
    def agg(rows):
        out={}
        for row in rows: out.setdefault(row[cluster],[]).append(float(row[metric]))
        return {k:float(np.mean(v)) for k,v in out.items()}
    left,right=agg(base),agg(candidate); ids=sorted(set(left)&set(right),key=int)
    return np.asarray([right[x]-left[x] for x in ids])


def paired_family(rows_by_method, comparisons, seed, n_boot, family):
    output=[]; test=0
    for method,baseline in comparisons:
        for region in REGIONS:
            candidate=[r for r in rows_by_method[method] if r["region"]==region]; base=[r for r in rows_by_method[baseline] if r["region"]==region]
            if [(r["source_id"],r["target_id"]) for r in candidate] != [(r["source_id"],r["target_id"]) for r in base]: raise RuntimeError("Pair order mismatch")
            for metric in INFERENTIAL_REGIONAL_METRICS:
                source=_identity_diff(base,candidate,metric,"source_id"); target=_identity_diff(base,candidate,metric,"target_id")
                sci=_bootstrap(source,seed+2*test,n_boot); tci=_bootstrap(target,seed+2*test+1,n_boot); mean=float(np.mean(source)); sd=float(np.std(source,ddof=1))
                output.append({"family":family,"method":method,"baseline":baseline,"region":region,"metric":metric,"mean_difference_method_minus_baseline":mean,"source_ci95_low":sci[0],"source_ci95_high":sci[1],"target_ci95_low":tci[0],"target_ci95_high":tci[1],"wilcoxon_p_two_sided":safe_wilcoxon(source),"effect_size_dz_source":mean/sd if sd>0 else 0.0,"source_identity_win_rate":float(np.mean(source<0)),"degenerate_all_zero":bool(np.allclose(source,0,atol=1e-15,rtol=0))}); test+=1
    adjusted=holm([float(r["wilcoxon_p_two_sided"]) for r in output])
    for row,p in zip(output,adjusted):
        row["holm_family_size"]=len(output); row["holm_p_two_sided"]=p
        row["final_judgement"]="improved" if p<.05 and row["source_ci95_high"]<0 and row["target_ci95_high"]<0 else "worse" if p<.05 and row["source_ci95_low"]>0 and row["target_ci95_low"]>0 else "not_significant_or_cluster_sensitive"
    return output


def distribution_row(method,domain,name,values):
    values=np.asarray(values,dtype=np.float64); values=values[np.isfinite(values)]; ordered=np.sort(values); k=max(1,int(np.ceil(.1*len(values)))); denom=float(np.sum(np.abs(values)))
    return {"method":method,"domain":domain,"diagnostic":name,"n_spatial_elements":len(values),"mean":float(np.mean(values)),"p50":float(np.percentile(values,50)),"p90":float(np.percentile(values,90)),"p95":float(np.percentile(values,95)),"p99":float(np.percentile(values,99)),"max":float(np.max(values)),"top10pct_mass_share":float(np.sum(np.abs(ordered[-k:]))/denom) if denom else 0.0}


def aggregate_spatial(context):
    root=context.output/"spatial"; root.mkdir(parents=True,exist_ok=True); summary=[]; outputs=[]
    display=np.asarray(context.base.get("feature_template_vertices",next(iter(context.by_id.values()))["vertices"]),dtype=np.float32)
    free=np.ones(len(context.topology.vertex_region),bool); free[context.topology.landmarks]=False
    for method in METHODS:
        chunks=sorted((context.output/"chunks").glob(f"*/*_{method}_spatial.npz")); total=empty_spatial_accumulator(context.topology)
        if not chunks: raise RuntimeError(f"No spatial chunks for {method}")
        for path in chunks:
            require_valid(path)
            with np.load(path) as values:
                total["n_pairs"]+=int(values["n_pairs"].item())
                for key in ("vertex_error_sum","edge_strain_sum","face_new_fold_count","face_abs_fold_count","face_missed_fold_count","face_target_fold_count"): total[key]+=values[key]
                for key in ("vertex_error_max","edge_strain_max"): total[key]=np.maximum(total[key],values[key])
        n=int(total["n_pairs"])
        if n!=9900: raise RuntimeError(f"Spatial count mismatch {method}:{n}")
        maps={"vertex_error_mean":total["vertex_error_sum"]/n,"vertex_error_max":total["vertex_error_max"],"edge_strain_mean":total["edge_strain_sum"]/n,"edge_strain_max":total["edge_strain_max"],"face_new_flip_frequency_pct":100*total["face_new_fold_count"]/n,"face_abs_flip_frequency_pct":100*total["face_abs_fold_count"]/n,"face_missed_flip_frequency_pct":100*total["face_missed_fold_count"]/n,"face_target_flip_frequency_pct":100*total["face_target_fold_count"]/n}
        path=root/f"{method}_spatial_diagnostics.npz"
        atomic_savez_compressed(path,method=np.asarray([method]),n_pairs=np.asarray([n]),display_vertices=display,faces=context.topology.faces,edges=context.topology.edges,vertex_region=context.topology.vertex_region,face_region_legacy=context.topology.face_region_legacy,edge_region=context.topology.edge_region,**{k:np.asarray(v,np.float32) for k,v in maps.items()}); outputs.append(path)
        for name,values in maps.items():
            domain="vertex" if name.startswith("vertex") else "edge" if name.startswith("edge") else "face"; summary.append(distribution_row(method,domain,name,values[free] if domain=="vertex" else values))
    atomic_write_csv(context.output/"spatial_tail_diagnostics_all_methods.csv",summary); return outputs


def aggregate(context):
    primary_paths=sorted(p for p in (context.output/"chunks").glob("*/*.csv") if not p.name.endswith("_unique_majority_sensitivity.csv")); sensitivity_paths=sorted((context.output/"chunks").glob("*/*_unique_majority_sensitivity.csv"))
    primary=[r for p in primary_paths for r in read_csv(p)]; sensitivity=[r for p in sensitivity_paths for r in read_csv(p)]; order={m:i for i,m in enumerate(METHODS)}; reg={m:i for i,m in enumerate(REGIONS)}; key=lambda r:(order[r["method"]],int(float(r["pair_index"])),reg[r["region"]]); primary.sort(key=key); sensitivity.sort(key=key)
    expected=8*9900*5
    if len(primary)!=expected or len(sensitivity)!=expected: raise RuntimeError(f"Incomplete regional rows: {len(primary)}/{len(sensitivity)}")
    atomic_write_csv(context.output/"subunit_pair_metrics_all_methods.csv",primary); atomic_write_csv(context.output/"subunit_pair_metrics_unique_majority_sensitivity.csv",sensitivity)
    by={m:[r for r in primary if r["method"]==m] for m in METHODS}; means=[]; tails=[]
    for method in METHODS:
        for region in REGIONS:
            block=[r for r in by[method] if r["region"]==region]; mean={"method":method,"region":region,"n_pairs":len(block)}
            for metric in REGIONAL_METRICS:
                values=np.asarray([float(r[metric]) for r in block]); mean[metric]=float(np.mean(values)); tails.append({"method":method,"region":region,"metric":metric,"n_pairs":len(values),"mean":float(np.mean(values)),"p50":float(np.percentile(values,50)),"p90":float(np.percentile(values,90)),"p95":float(np.percentile(values,95)),"p99":float(np.percentile(values,99)),"max":float(np.max(values))})
            means.append(mean)
    atomic_write_csv(context.output/"subunit_means_all_methods.csv",means); atomic_write_csv(context.output/"subunit_pair_tail_diagnostics.csv",tails); spatial=aggregate_spatial(context)
    all_stats=paired_family(by,[(m,"ridge") for m in METHODS if m!="ridge"],context.args.seed,context.args.n_boot,"all_methods_vs_ridge_175"); direct=paired_family(by,[("certified_rbsr","lamm")],context.args.seed+100000,context.args.n_boot,"certified_rbsr_vs_lamm_25")
    atomic_write_csv(context.output/"paired_all_methods_vs_ridge.csv",all_stats); atomic_write_csv(context.output/"paired_certified_rbsr_vs_lamm.csv",direct)
    noise=context.output/"noise/STRICT_NOISE_ROBUSTNESS_EVIDENCE.json"; require_valid(noise)
    qualitative=context.output/"qualitative/QUALITATIVE_CASES_EVIDENCE.json"; require_valid(qualitative)
    personalization=context.output/"personalization/PERSONALIZATION_ABLATION_EVIDENCE.json"; require_valid(personalization)
    report={"status":"COMPLETE_POSTHOC_SUBUNIT_ANALYSIS","claim_boundary":"secondary post-hoc; original final evidence unchanged","n_pairs":9900,"methods":list(METHODS),"reported_metrics":list(REGIONAL_METRICS),"inferential_metrics":list(INFERENTIAL_REGIONAL_METRICS),"descriptive_invariants":["landmark_rmse","target_flip_pct"],"regions":list(REGIONS),"all_vs_ridge_family_size":175,"rbsr_vs_lamm_family_size":25,"noise_family_size":36,"personalization_primary_family_size":3,"personalization_secondary_family_size":6,"n_boot":context.args.n_boot,"strict_noise_evidence_sha256":sha256_file(noise),"qualitative_evidence_sha256":sha256_file(qualitative),"personalization_evidence_sha256":sha256_file(personalization),"outputs":{},"spatial_outputs":{p.name:sha256_file(p) for p in spatial}}
    for name in ("POSTHOC_SUBUNIT_ANALYSIS_PROTOCOL_FREEZE.json","region_topology_audit.json","subunit_pair_metrics_all_methods.csv","subunit_pair_metrics_unique_majority_sensitivity.csv","subunit_means_all_methods.csv","subunit_pair_tail_diagnostics.csv","spatial_tail_diagnostics_all_methods.csv","paired_all_methods_vs_ridge.csv","paired_certified_rbsr_vs_lamm.csv"):
        report["outputs"][name]=sha256_file(context.output/name)
    evidence=context.output/"POSTHOC_SUBUNIT_ANALYSIS_EVIDENCE.json"; atomic_write_json(evidence,report); write_sha256_sidecar(evidence); print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--stage",choices=("audit","neural","classical","lamm","aggregate","all"),required=True); parser.add_argument("--repo-root",type=Path,required=True); parser.add_argument("--repo",type=Path,required=True); parser.add_argument("--results",type=Path,required=True); parser.add_argument("--out",type=Path,required=True); parser.add_argument("--lamm-root",type=Path,default=Path("/content/LAMM_official")); parser.add_argument("--neural-batch-size",type=int,default=32); parser.add_argument("--lamm-batch-size",type=int,default=32); parser.add_argument("--chunk-pairs",type=int,default=160); parser.add_argument("--seed",type=int,default=20260609); parser.add_argument("--n-boot",type=int,default=10000); args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True); context=Context(args); freeze_protocol(context)
    stages=("neural","classical","lamm","aggregate") if args.stage=="all" else (args.stage,)
    for stage in stages:
        if stage=="neural": run_neural(context)
        elif stage=="classical": run_classical(context)
        elif stage=="lamm": run_lamm(context)
        elif stage=="aggregate": aggregate(context)
    return 0


if __name__=="__main__": raise SystemExit(main())
