"""Frozen-validation source-personalization ablation for the Ridge anchor."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rhinoform.data import load_rows, ridge_predict
from rhinoform.repro import (
    atomic_savez_compressed,
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
    sha256_json,
    valid_sha256_sidecar,
    write_sha256_sidecar,
)
from rhinoform.rbsr_calibration import enforce_exact_controls
from rhinoform.strict_protocol_patch import strict_metric_rows
from rhinoform.train import pair_conditions
from scripts.evaluation.direct_paired_statistics import holm, safe_wilcoxon
from scripts.evaluation.posthoc_subunit_analysis import Context, read_csv, require_valid


VARIANTS = (
    "controls_only_ridge",
    "source_pca_ridge",
    "mean_source_code",
    "shuffled_source_code",
)
METRICS = ("roi_rmse", "normal_flip_pct", "edge_strain_p95")
PRIMARY_METRICS = ("roi_rmse",)
SECONDARY_METRICS = ("normal_flip_pct", "edge_strain_p95")
ALTERNATIVES = (
    "controls_only_ridge",
    "mean_source_code",
    "shuffled_source_code",
)


def output_valid(path: Path, expected_rows: int) -> bool:
    try:
        return valid_sha256_sidecar(path) and len(read_csv(path)) == expected_rows
    except (OSError, csv.Error):
        return False


def source_code(vertices: np.ndarray, source_pca: dict) -> np.ndarray:
    flat = np.asarray(vertices, dtype=np.float64).reshape(len(vertices), -1)
    return (
        (flat - np.asarray(source_pca["mean"], dtype=np.float64))
        @ np.asarray(source_pca["components"], dtype=np.float64).T
    ).astype(np.float32)


def standardise_condition(controls: np.ndarray, codes: np.ndarray, base: dict) -> np.ndarray:
    raw = np.concatenate([controls, codes], axis=1)
    return (
        (raw - np.asarray(base["cond_mean"])) / np.asarray(base["cond_std"])
    ).astype(np.float32)


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    block_size = 1000
    for start in range(0, n_boot, block_size):
        stop = min(start + block_size, n_boot)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = np.mean(values[indices], axis=1)
    return tuple(map(float, np.percentile(means, [2.5, 97.5])))


def clustered_difference(
    personalized: list[dict[str, str]],
    alternative: list[dict[str, str]],
    metric: str,
    cluster_position: int,
) -> np.ndarray:
    left = {
        (row["source_id"], row["target_id"]): float(row[metric])
        for row in personalized
    }
    right = {
        (row["source_id"], row["target_id"]): float(row[metric])
        for row in alternative
    }
    if set(left) != set(right):
        raise RuntimeError("Personalization ablation pair sets differ")
    grouped: dict[str, list[float]] = {}
    for pair in left:
        grouped.setdefault(pair[cluster_position], []).append(left[pair] - right[pair])
    return np.asarray([
        np.mean(grouped[identity]) for identity in sorted(grouped, key=int)
    ], dtype=np.float64)


def family_statistics(
    by_variant: dict[str, list[dict[str, str]]],
    metrics: tuple[str, ...],
    family: str,
    seed: int,
    n_boot: int,
) -> list[dict[str, object]]:
    rows = []
    test_index = 0
    for alternative in ALTERNATIVES:
        for metric in metrics:
            source = clustered_difference(
                by_variant["source_pca_ridge"], by_variant[alternative], metric, 0
            )
            target = clustered_difference(
                by_variant["source_pca_ridge"], by_variant[alternative], metric, 1
            )
            source_ci = bootstrap_ci(source, seed + 2 * test_index, n_boot)
            target_ci = bootstrap_ci(target, seed + 2 * test_index + 1, n_boot)
            source_sd = float(np.std(source, ddof=1))
            mean = float(np.mean(source))
            rows.append({
                "family": family,
                "personalized": "source_pca_ridge",
                "alternative": alternative,
                "metric": metric,
                "mean_difference_personalized_minus_alternative": mean,
                "source_ci95_low": source_ci[0],
                "source_ci95_high": source_ci[1],
                "target_ci95_low": target_ci[0],
                "target_ci95_high": target_ci[1],
                "wilcoxon_cluster": "source_identity_primary",
                "wilcoxon_p_two_sided": safe_wilcoxon(source),
                "effect_size_dz_source": mean / source_sd if source_sd > 0 else 0.0,
                "source_identity_win_rate": float(np.mean(source < 0.0)),
                "degenerate_all_zero": bool(np.allclose(source, 0.0, atol=1e-15, rtol=0.0)),
            })
            test_index += 1
    adjusted = holm([float(row["wilcoxon_p_two_sided"]) for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_family_size"] = len(rows)
        row["holm_p_two_sided"] = value
        row["final_judgement"] = (
            "personalized_better"
            if value < 0.05
            and row["source_ci95_high"] < 0.0
            and row["target_ci95_high"] < 0.0
            else "personalized_worse"
            if value < 0.05
            and row["source_ci95_low"] > 0.0
            and row["target_ci95_low"] > 0.0
            else "not_significant_or_cluster_sensitive"
        )
    return rows


def write_fixed_control_demo(
    output: Path,
    pairs: list[tuple[str, str]],
    by_id: dict[str, dict],
    validation_ids: list[str],
    code_by_identity: dict[str, np.ndarray],
    base: dict,
) -> Path:
    landmarks = np.asarray(next(iter(by_id.values()))["landmarks"], dtype=np.int64)
    controls = np.asarray([
        (by_id[target_id]["vertices"] - by_id[source_id]["vertices"])[landmarks].reshape(-1)
        for source_id, target_id in pairs
    ], dtype=np.float32)
    magnitudes = np.sqrt(np.mean(controls.reshape(len(controls), -1, 3) ** 2, axis=(1, 2)))
    median = float(np.median(magnitudes))
    edit_index = min(range(len(pairs)), key=lambda index: (abs(magnitudes[index] - median), index))
    fixed_control = controls[edit_index]

    ordered_sources = sorted(validation_ids, key=lambda identity: (
        float(code_by_identity[identity][0]), int(identity)
    ))
    positions = [0, round((len(ordered_sources) - 1) / 3), round(2 * (len(ordered_sources) - 1) / 3), len(ordered_sources) - 1]
    selected_sources = [ordered_sources[position] for position in positions]
    if len(set(selected_sources)) != 4:
        raise RuntimeError("Fixed-control demo source quantiles are not unique")
    repeated_controls = np.repeat(fixed_control[None, :], len(selected_sources), axis=0)
    codes = np.stack([code_by_identity[identity] for identity in selected_sources])
    controls_only = ridge_predict(repeated_controls, base["ridge_ctrl"])
    personalized = ridge_predict(
        standardise_condition(repeated_controls, codes, base), base["ridge_cond"]
    )
    sources = np.stack([by_id[identity]["vertices"] for identity in selected_sources]).astype(np.float32)
    controls_only = enforce_exact_controls(
        controls_only, repeated_controls.reshape(len(selected_sources), -1, 3), landmarks
    ).reshape(len(selected_sources), -1, 3)
    personalized = enforce_exact_controls(
        personalized, repeated_controls.reshape(len(selected_sources), -1, 3), landmarks
    ).reshape(len(selected_sources), -1, 3)
    path = output / "personalization/fixed_control_personalization_demo.npz"
    atomic_savez_compressed(
        path,
        source_ids=np.asarray(selected_sources),
        source_pc1=np.asarray([code_by_identity[value][0] for value in selected_sources]),
        source_vertices=sources,
        landmarks=landmarks,
        fixed_control_delta=fixed_control.reshape(-1, 3),
        controls_only_vertices=(sources + controls_only).astype(np.float32),
        source_pca_vertices=(sources + personalized).astype(np.float32),
        response_difference=np.linalg.norm(personalized - controls_only, axis=2).astype(np.float32),
        edit_pair_index=np.asarray([edit_index], dtype=np.int64),
        edit_source_id=np.asarray([pairs[edit_index][0]]),
        edit_target_id=np.asarray([pairs[edit_index][1]]),
        no_ground_truth_accuracy_claim=np.asarray([True]),
    )
    return path


def aggregate(output: Path, seed: int, n_boot: int) -> None:
    chunk_paths = sorted((output / "personalization/chunks").glob("*.csv"))
    all_rows = [row for path in chunk_paths for row in read_csv(path)]
    expected = len(VARIANTS) * 4830
    if len(all_rows) != expected:
        raise RuntimeError(f"Incomplete personalization rows: {len(all_rows)} != {expected}")
    all_rows.sort(key=lambda row: (
        VARIANTS.index(row["variant"]), int(float(row["validation_pair_index"]))
    ))
    pair_path = output / "personalization/pair_metrics_all_variants.csv"
    atomic_write_csv(pair_path, all_rows)
    by_variant = {
        variant: [row for row in all_rows if row["variant"] == variant]
        for variant in VARIANTS
    }
    summary = []
    for variant in VARIANTS:
        record = {"variant": variant, "n_validation_pairs": len(by_variant[variant])}
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in by_variant[variant]])
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_p95"] = float(np.percentile(values, 95))
        summary.append(record)
    summary_path = output / "personalization/personalization_ablation_summary.csv"
    atomic_write_csv(summary_path, summary)

    primary = family_statistics(
        by_variant, PRIMARY_METRICS, "primary_rmse_family_size_3", seed, n_boot
    )
    secondary = family_statistics(
        by_variant, SECONDARY_METRICS,
        "secondary_geometry_family_size_6", seed + 10000, n_boot,
    )
    primary_path = output / "personalization/paired_primary_personalization_rmse.csv"
    secondary_path = output / "personalization/paired_secondary_personalization_geometry.csv"
    atomic_write_csv(primary_path, primary)
    atomic_write_csv(secondary_path, secondary)
    supported = all(row["final_judgement"] == "personalized_better" for row in primary)
    protocol_path = output / "personalization/PERSONALIZATION_ABLATION_PROTOCOL_FREEZE.json"
    demo_path = output / "personalization/fixed_control_personalization_demo.npz"
    require_valid(demo_path, "fixed-control personalization demonstration")
    evidence = {
        "status": "COMPLETE_FROZEN_VALIDATION_PERSONALIZATION_ABLATION",
        "claim_boundary": "mechanistic validation evidence; not blind test confirmation",
        "personalization_supported_on_primary_rmse": supported,
        "primary_rmse_family_size": 3,
        "secondary_geometry_family_size": 6,
        "operating_point_reselected": False,
        "fixed_control_demo_sha256": sha256_file(demo_path),
        "fixed_control_demo_claim": "mechanistic source conditioning only; no_ground_truth_accuracy_claim",
        "protocol_freeze_sha256": sha256_file(protocol_path),
        "outputs": {
            path.name: sha256_file(path)
            for path in (pair_path, summary_path, primary_path, secondary_path)
        },
    }
    evidence_path = output / "personalization/PERSONALIZATION_ABLATION_EVIDENCE.json"
    atomic_write_json(evidence_path, evidence)
    write_sha256_sidecar(evidence_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk-pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()
    if (args.seed, args.n_boot) != (20260810, 10000):
        raise ValueError("Frozen personalization protocol requires seed=20260810, n_boot=10000")
    args.out.mkdir(parents=True, exist_ok=True)
    context_args = argparse.Namespace(**vars(args))
    context_args.lamm_root = Path("/unused")
    context_args.chunk_pairs = args.chunk_pairs
    context_args.neural_batch_size = 32
    context_args.lamm_batch_size = 32
    context = Context(context_args)
    parent_protocol = args.out / "POSTHOC_SUBUNIT_ANALYSIS_PROTOCOL_FREEZE.json"
    require_valid(parent_protocol, "parent post-hoc protocol")

    if "ridge_ctrl" not in context.base:
        raise RuntimeError("Frozen base package lacks the independently fitted controls-only Ridge")
    pairs = [(str(a), str(b)) for a, b in context.base["val_pairs"]]
    if len(pairs) != 4830:
        raise RuntimeError(f"Expected 4830 frozen validation pairs, got {len(pairs)}")
    validation_ids = sorted(map(str, context.base["val_ids"]), key=int)
    if len(validation_ids) < 2:
        raise RuntimeError("A cyclic source-code derangement needs at least two identities")
    shuffled_identity = {
        identity: validation_ids[(index + 1) % len(validation_ids)]
        for index, identity in enumerate(validation_ids)
    }
    if any(source == target for source, target in shuffled_identity.items()):
        raise AssertionError("fixed_cyclic_derangement unexpectedly contains a self-map")

    protocol = {
        "status": "FROZEN_VALIDATION_PERSONALIZATION_ABLATION",
        "claim_boundary": "mechanistic validation evidence; no untouched-test claim",
        "validation_pairs": len(pairs),
        "validation_pair_order_sha256": sha256_json(pairs),
        "variants": list(VARIANTS),
        "controls_only_definition": "original training-run ridge_ctrl weights; controls only",
        "source_pca_definition": "original frozen full Ridge with each pair's true source PCA code",
        "mean_source_definition": "same frozen full Ridge with raw source PCA code fixed to zero",
        "shuffle_rule": "fixed_cyclic_derangement over sorted validation identities; same map for every outgoing pair",
        "shuffle_mapping": shuffled_identity,
        "same_controls_rule": "all four variants receive identical nine target-minus-source controls pairwise",
        "fixed_control_demo": (
            "same control vector across four deterministic source identities selected at source-PC1 quantiles; "
            "mechanistic visualisation only"
        ),
        "metrics": list(METRICS),
        "scoring": "hard-fixed true controls; free-ROI RMSE; target-relative new flip; edge-strain P95",
        "primary_family": "three source-PCA-vs-alternative free-ROI RMSE comparisons; Holm",
        "secondary_family": "three alternatives x two geometry metrics = six; separate Holm family",
        "inference": "source-identity Wilcoxon plus source/target identity bootstrap CIs; both CIs must agree",
        "operating_point_reselected": False,
        "base_package_sha256": sha256_file(context.base_path),
        "parent_posthoc_protocol_sha256": sha256_file(parent_protocol),
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    protocol_path = args.out / "personalization/PERSONALIZATION_ABLATION_PROTOCOL_FREEZE.json"
    if protocol_path.exists():
        require_valid(protocol_path)
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("Existing personalization protocol differs; use a new output version")
    else:
        atomic_write_json(protocol_path, protocol)
        write_sha256_sidecar(protocol_path)

    _, by_id = load_rows(args.repo, allowed_ids=set(validation_ids))
    canonical_path = args.results / (
        "rbsr/seed20260609/validation_raw/"
        "pair_metrics_ridge_sourcepca_clean_validation.csv"
    )
    require_valid(canonical_path, "canonical source-PCA validation metrics")
    canonical = read_csv(canonical_path)
    if len(canonical) != len(pairs):
        raise RuntimeError("Canonical validation Ridge row count differs")

    code_by_identity = {
        identity: source_code(
            np.asarray(by_id[identity]["vertices"])[None, ...], context.base["source_pca"]
        )[0]
        for identity in validation_ids
    }
    write_fixed_control_demo(
        args.out, pairs, by_id, validation_ids, code_by_identity, context.base
    )
    code_dim = len(next(iter(code_by_identity.values())))
    for start in range(0, len(pairs), args.chunk_pairs):
        stop = min(start + args.chunk_pairs, len(pairs))
        stem = f"chunk_{start:05d}_{stop:05d}"
        paths = {
            variant: args.out / "personalization/chunks" / f"{stem}_{variant}.csv"
            for variant in VARIANTS
        }
        if all(output_valid(path, stop - start) for path in paths.values()):
            print(f"PERSONALIZATION resume {stem}", flush=True)
            continue
        chunk_pairs = pairs[start:stop]
        actual_condition, _, _, controls, _, _, _ = pair_conditions(
            by_id, chunk_pairs, context.base["source_pca"],
            context.base["cond_mean"], context.base["cond_std"],
        )
        mean_codes = np.zeros((len(chunk_pairs), code_dim), dtype=np.float32)
        shuffled_codes = np.stack([
            code_by_identity[shuffled_identity[source_id]]
            for source_id, _ in chunk_pairs
        ]).astype(np.float32)
        predictions = {
            "controls_only_ridge": ridge_predict(controls, context.base["ridge_ctrl"]),
            "source_pca_ridge": ridge_predict(actual_condition, context.base["ridge_cond"]),
            "mean_source_code": ridge_predict(
                standardise_condition(controls, mean_codes, context.base),
                context.base["ridge_cond"],
            ),
            "shuffled_source_code": ridge_predict(
                standardise_condition(controls, shuffled_codes, context.base),
                context.base["ridge_cond"],
            ),
        }
        for variant, prediction in predictions.items():
            metric_rows = strict_metric_rows(by_id, chunk_pairs, prediction)
            if variant == "source_pca_ridge":
                for offset, row in enumerate(metric_rows):
                    expected = canonical[start + offset]
                    if (str(row["source_id"]), str(row["target_id"])) != (
                        expected["source_id"], expected["target_id"]
                    ):
                        raise AssertionError("Canonical source-PCA validation pair mismatch")
                    for metric in METRICS:
                        if not np.isclose(
                            float(row[metric]), float(expected[metric]), atol=2e-6, rtol=0.0
                        ):
                            raise AssertionError(f"Canonical source-PCA mismatch: {metric}")
            for offset, row in enumerate(metric_rows):
                row.update({
                    "variant": variant,
                    "validation_pair_index": start + offset,
                })
            atomic_write_csv(paths[variant], metric_rows)
        print(f"PERSONALIZATION persisted {stop}/{len(pairs)}", flush=True)
    aggregate(args.out, args.seed, args.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
