"""Deterministic frozen-test qualitative comparison, including a real failure.

This is descriptive post-hoc evidence.  Six roles are selected mechanically
from already-frozen test metrics, persisted before dense re-inference, and then
rendered with Ridge, Certified RB-SR and official LAMM predictions.  No role is
hand-picked and no operating point is changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rhinoform.data import edge_index, load_rows, ridge_predict
from rhinoform.rbsr_calibration import enforce_exact_controls
from rhinoform.repro import (
    atomic_savez_compressed,
    atomic_write_json,
    sha256_file,
    sha256_json,
    valid_sha256_sidecar,
    write_sha256_sidecar,
)
from rhinoform.safe_fusion import (
    project_residual_no_new_ridge_folds,
    signed_fold_indicator,
)
from rhinoform.strict_protocol_patch import strict_metric_rows
from rhinoform.train import predict_field
from rhinoform.train_rbsr_gate import predict_gate_batches
from scripts.evaluation.posthoc_strict_noise_robustness import (
    lamm_controls,
    noisy_condition,
    prepare_lamm,
    prepare_neural,
)
from scripts.evaluation.posthoc_subunit_analysis import (
    GLOBAL_CHECK_METRICS,
    Context,
    read_csv,
    require_valid,
)


ROLE_ORDER = (
    "median_case",
    "small_control",
    "large_control",
    "high_certificate_iterations",
    "largest_rbsr_gain_over_ridge",
    "worst_rbsr_failure_vs_best_comparator",
)


def canonical_paths(results: Path) -> dict[str, Path]:
    one_shot = results / "rbsr/seed20260609/one_shot_test"
    return {
        "certified_rbsr": one_shot / "pair_metrics_rbsr_test.csv",
        "ridge": one_shot / "pair_metrics_ridge_sourcepca_clean_test.csv",
        "lamm": results / "lamm/seed20260609/identity_bootstrap_pair_metrics_lamm.csv",
        "projection": one_shot / "projection_diagnostics_test.csv",
    }


def load_aligned_canonical(results: Path):
    paths = canonical_paths(results)
    for path in paths.values():
        require_valid(path, "qualitative selection input")
    tables = {key: read_csv(path) for key, path in paths.items()}
    if any(len(rows) != 9900 for rows in tables.values()):
        raise RuntimeError("Qualitative selection requires all 9,900 frozen test pairs")
    expected = [
        (row["source_id"], row["target_id"])
        for row in tables["certified_rbsr"]
    ]
    for name, rows in tables.items():
        observed = [(row["source_id"], row["target_id"]) for row in rows]
        if observed != expected:
            raise RuntimeError(f"Frozen qualitative input pair order differs: {name}")
    return paths, tables


def nearest_rank(values: np.ndarray, target: float) -> list[int]:
    return sorted(range(len(values)), key=lambda index: (abs(values[index] - target), index))


def select_cases(context: Context, tables: dict[str, list[dict[str, str]]]):
    controls = np.asarray([
        context.by_id[target_id]["vertices"][context.topology.landmarks]
        - context.by_id[source_id]["vertices"][context.topology.landmarks]
        for source_id, target_id in context.pairs
    ], dtype=np.float64)
    # RMS Euclidean displacement of the nine controls, in FaceScape millimetres.
    control_rms_mm = np.sqrt(np.mean(np.sum(controls * controls, axis=2), axis=1))
    rbsr = np.asarray([float(row["roi_rmse"]) for row in tables["certified_rbsr"]])
    ridge = np.asarray([float(row["roi_rmse"]) for row in tables["ridge"]])
    lamm = np.asarray([float(row["roi_rmse"]) for row in tables["lamm"]])
    iterations = np.asarray([int(float(row["iterations"])) for row in tables["projection"]])
    retention = np.asarray([float(row["retention"]) for row in tables["projection"]])

    rankings = {
        "median_case": nearest_rank(rbsr, float(np.median(rbsr))),
        "small_control": nearest_rank(control_rms_mm, float(np.percentile(control_rms_mm, 10))),
        "large_control": nearest_rank(control_rms_mm, float(np.percentile(control_rms_mm, 90))),
        "high_certificate_iterations": sorted(
            range(len(context.pairs)),
            key=lambda index: (-iterations[index], retention[index], index),
        ),
        "largest_rbsr_gain_over_ridge": sorted(
            range(len(context.pairs)),
            key=lambda index: (-(ridge[index] - rbsr[index]), index),
        ),
        "worst_rbsr_failure_vs_best_comparator": sorted(
            range(len(context.pairs)),
            key=lambda index: (-(rbsr[index] - min(ridge[index], lamm[index])), index),
        ),
    }
    selected = []
    used = set()
    for role in ROLE_ORDER:
        index = next(candidate for candidate in rankings[role] if candidate not in used)
        used.add(index)
        source_id, target_id = context.pairs[index]
        selected.append({
            "role": role,
            "pair_index": index,
            "source_id": source_id,
            "target_id": target_id,
            "control_rms_mm": float(control_rms_mm[index]),
            "rbsr_roi_rmse": float(rbsr[index]),
            "ridge_roi_rmse": float(ridge[index]),
            "lamm_roi_rmse": float(lamm[index]),
            "rbsr_gain_over_ridge": float(ridge[index] - rbsr[index]),
            "rbsr_failure_margin_vs_best_comparator": float(
                rbsr[index] - min(ridge[index], lamm[index])
            ),
            "retention": float(retention[index]),
            "iterations": int(iterations[index]),
        })
    failure = selected[-1]
    if failure["rbsr_failure_margin_vs_best_comparator"] <= 0.0:
        raise RuntimeError("Frozen test set contains no genuine RB-SR failure case")
    return selected


def compare_selected_metrics(
    observed: list[dict[str, object]],
    canonical: list[dict[str, str]],
    selected_indices: list[int],
    atol: float,
) -> None:
    for row_number, (row, index) in enumerate(zip(observed, selected_indices)):
        expected = canonical[index]
        if (str(row["source_id"]), str(row["target_id"])) != (
            expected["source_id"], expected["target_id"]
        ):
            raise AssertionError(f"Selected-pair mismatch at row {row_number}")
        for metric in GLOBAL_CHECK_METRICS:
            if metric in expected and not np.isclose(
                float(row[metric]), float(expected[metric]), atol=atol, rtol=0.0
            ):
                raise AssertionError(
                    f"Qualitative recomputation mismatch {metric}: "
                    f"{row[metric]} != {expected[metric]}"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lamm-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    context_args = argparse.Namespace(**vars(args))
    context_args.chunk_pairs = 160
    context_args.neural_batch_size = args.batch_size
    context_args.lamm_batch_size = args.batch_size
    context_args.seed = 20260609
    context_args.n_boot = 10000
    context = Context(context_args)

    parent_protocol = args.out / "POSTHOC_SUBUNIT_ANALYSIS_PROTOCOL_FREEZE.json"
    require_valid(parent_protocol, "parent post-hoc protocol")
    paths, tables = load_aligned_canonical(args.results)
    selected = select_cases(context, tables)
    manifest = {
        "status": "FROZEN_DETERMINISTIC_QUALITATIVE_SELECTION_BEFORE_DENSE_REINFERENCE",
        "claim_boundary": "descriptive post-hoc frozen-test examples; no inferential claim",
        "roles_in_order": list(ROLE_ORDER),
        "uniqueness_rule": "first unused pair in each deterministic ranking",
        "tie_break": "ascending frozen pair_index",
        "small_large_rule": "nearest to frozen test control-RMS P10 and P90",
        "control_magnitude_formula": "sqrt(mean over nine controls of squared xyz norm), mm",
        "median_rule": "nearest to median Certified RB-SR free-ROI RMSE",
        "failure_rule": "max RB-SR RMSE minus min(Ridge RMSE, LAMM RMSE); must be positive",
        "high_iteration_rule": "max iterations, then lower retention, then pair index",
        "operating_point_reselected": False,
        "parent_protocol_sha256": sha256_file(parent_protocol),
        "selection_input_hashes": {key: sha256_file(path) for key, path in paths.items()},
        "selected": selected,
    }
    manifest_path = args.out / "qualitative/QUALITATIVE_CASE_SELECTION_FREEZE.json"
    if manifest_path.exists():
        require_valid(manifest_path)
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("Existing qualitative selection freeze differs")
    else:
        atomic_write_json(manifest_path, manifest)
        write_sha256_sidecar(manifest_path)

    selected_indices = [int(row["pair_index"]) for row in selected]
    selected_pairs = [context.pairs[index] for index in selected_indices]
    all_ids = set(map(str, context.base["train_ids"])) | {
        value for pair in selected_pairs for value in pair
    }
    _, all_by_id = load_rows(args.repo, allowed_ids=all_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Qualitative all-method dense re-inference requires a GPU")

    features, proposer, gate_package, gate, projection = prepare_neural(
        context, all_by_id, device
    )
    lamm_model, lamm_mean, lamm_std = prepare_lamm(context, device)
    landmarks = context.topology.landmarks
    faces = context.topology.faces
    source = np.stack([all_by_id[s]["vertices"] for s, _ in selected_pairs]).astype(np.float32)
    target = np.stack([all_by_id[t]["vertices"] for _, t in selected_pairs]).astype(np.float32)
    true_controls = (target - source)[:, landmarks].astype(np.float64)
    condition = noisy_condition(true_controls, source, context.base)
    ridge = ridge_predict(condition, context.base["ridge_cond"]).astype(np.float32)
    cvae = predict_field(
        proposer, features, condition, is_cvae=True, batch_size=args.batch_size
    ).astype(np.float32)
    vertex_features = torch.as_tensor(features, dtype=torch.float32, device=device)
    gate_center = torch.as_tensor(
        np.asarray(gate_package["center"]), dtype=torch.float32, device=device
    )
    projection_basis = None if gate_package.get("projection_basis") is None else torch.as_tensor(
        gate_package["projection_basis"], dtype=torch.float32, device=device
    )
    _, gate_values = predict_gate_batches(
        gate, vertex_features, condition, source, ridge, cvae,
        gate_center, float(gate_package["scale"]),
        torch.as_tensor(landmarks, dtype=torch.long, device=device),
        true_controls, projection_basis, args.batch_size,
    )
    ridge_fixed = enforce_exact_controls(ridge, true_controls, landmarks).reshape(len(source), -1, 3)
    residual = gate_values[..., None] * (cvae.reshape(len(source), -1, 3) - ridge_fixed)
    residual[:, landmarks] = 0.0
    projected = []
    diagnostics = []
    for source_mesh, anchor, correction in zip(source, ridge_fixed, residual):
        result = project_residual_no_new_ridge_folds(
            source_mesh, anchor, correction, faces, landmarks,
            attenuation=float(projection["attenuation"]),
            smoothing_steps=int(projection["smoothing_steps"]),
            max_iterations=int(projection["max_iterations"]),
            uniform_steps=int(projection["uniform_steps"]),
        )
        if not result.certified or result.new_vs_ridge_fold_count != 0:
            raise AssertionError("Qualitative RB-SR projection did not certify")
        projected.append(result.delta)
        diagnostics.append(result)
    certified = np.asarray(projected, dtype=np.float32).reshape(len(source), -1)

    source_norm = torch.from_numpy((source - lamm_mean) / lamm_std).to(device)
    with torch.no_grad():
        lamm_prediction = lamm_model((
            source_norm,
            lamm_controls(lamm_model, true_controls, landmarks, lamm_std, device),
        ))[-1]
    lamm_abs = (
        lamm_prediction * torch.from_numpy(lamm_std).to(device)
        + torch.from_numpy(lamm_mean).to(device)
    ).cpu().numpy()
    lamm = (lamm_abs - source).reshape(len(source), -1)
    predictions = {"ridge": ridge, "certified_rbsr": certified, "lamm": lamm}
    for method, prediction in predictions.items():
        observed = strict_metric_rows(all_by_id, selected_pairs, prediction)
        compare_selected_metrics(
            observed, tables[method], selected_indices,
            atol=1e-5 if method == "lamm" else 2e-6,
        )

    output_hashes = {}
    edges = edge_index(faces)
    for case_number, (selection, pair) in enumerate(zip(selected, selected_pairs), start=1):
        source_mesh = np.asarray(all_by_id[pair[0]]["vertices"], dtype=np.float64)
        target_mesh = np.asarray(all_by_id[pair[1]]["vertices"], dtype=np.float64)
        target_fold = signed_fold_indicator(source_mesh, target_mesh, faces)[0] < 0.0
        arrays = {
            "source_vertices": source_mesh.astype(np.float32),
            "target_vertices": target_mesh.astype(np.float32),
            "faces": np.asarray(faces, dtype=np.int64),
            "landmarks": np.asarray(landmarks, dtype=np.int64),
            "edges": np.asarray(edges, dtype=np.int64),
        }
        for method, prediction in predictions.items():
            delta = np.asarray(prediction[case_number - 1]).reshape(-1, 3).copy()
            delta[landmarks] = target_mesh[landmarks] - source_mesh[landmarks]
            edited = source_mesh + delta
            pred_fold = signed_fold_indicator(source_mesh, edited, faces)[0] < 0.0
            arrays[f"{method}_vertices"] = edited.astype(np.float32)
            arrays[f"{method}_vertex_error"] = np.linalg.norm(
                edited - target_mesh, axis=1
            ).astype(np.float32)
            arrays[f"{method}_new_flip_faces"] = (pred_fold & ~target_fold).astype(np.uint8)
            source_lengths = np.linalg.norm(
                source_mesh[edges[:, 0]] - source_mesh[edges[:, 1]], axis=1
            )
            edited_lengths = np.linalg.norm(
                edited[edges[:, 0]] - edited[edges[:, 1]], axis=1
            )
            arrays[f"{method}_edge_strain"] = (
                np.abs(edited_lengths - source_lengths)
                / np.maximum(source_lengths, 1e-12)
            ).astype(np.float32)
        diagnostic = diagnostics[case_number - 1]
        expected_diagnostic = tables["projection"][int(selection["pair_index"])]
        if (
            diagnostic.iterations != int(float(expected_diagnostic["iterations"]))
            or not np.isclose(
                diagnostic.retention, float(expected_diagnostic["retention"]),
                atol=2e-6, rtol=0.0,
            )
        ):
            raise AssertionError("Qualitative certificate diagnostic mismatch")
        arrays["certificate_retention"] = np.asarray([diagnostic.retention], np.float64)
        arrays["certificate_iterations"] = np.asarray([diagnostic.iterations], np.int64)
        arrays["role"] = np.asarray([selection["role"]])
        arrays["source_id"] = np.asarray([selection["source_id"]])
        arrays["target_id"] = np.asarray([selection["target_id"]])
        output_path = args.out / "qualitative/cases" / (
            f"case_{case_number:02d}_{selection['role']}.npz"
        )
        atomic_savez_compressed(output_path, **arrays)
        output_hashes[output_path.name] = sha256_file(output_path)

    evidence = {
        "status": "COMPLETE_DETERMINISTIC_QUALITATIVE_COMPARISON",
        "claim_boundary": "descriptive post-hoc; selection uses frozen test outcomes",
        "n_cases": len(selected),
        "methods": ["ridge", "certified_rbsr", "lamm"],
        "selection_freeze_sha256": sha256_file(manifest_path),
        "selection_sha256": sha256_json(selected),
        "dense_predictions_recomputed": True,
        "canonical_metric_self_check": "PASS",
        "hard_fixed_true_controls": True,
        "flip_definition": "prediction fold AND NOT target fold",
        "visualisation_contract": (
            "shared per-case vertex-error scale across methods; explicit new-flip overlay; "
            "edge-strain overlay; landmark-edit panel; failure case mandatory"
        ),
        "interpretation_boundary": (
            "geometrically natural/regular only; must not be called clinically preferred"
        ),
        "outputs": output_hashes,
    }
    evidence_path = args.out / "qualitative/QUALITATIVE_CASES_EVIDENCE.json"
    atomic_write_json(evidence_path, evidence)
    write_sha256_sidecar(evidence_path)
    print(json.dumps(evidence, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
