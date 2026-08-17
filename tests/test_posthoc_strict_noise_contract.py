from pathlib import Path
import csv

from rhinoform.repro import (
    atomic_write_csv,
    atomic_write_json,
    valid_sha256_sidecar,
    write_sha256_sidecar,
)
from scripts.evaluation.posthoc_strict_noise_robustness import aggregate_noise
ROOT=Path(__file__).resolve().parents[1]


def test_strict_noise_source_contract():
    text=(ROOT/"scripts/evaluation/posthoc_strict_noise_robustness.py").read_text()
    for method in ("ridge","certified_rbsr","lamm","arap"):
        assert f'"{method}"' in text
    for value in ("4830","0.25","0.5","1.0","strict_metric_rows","true_controls","target-relative new flip","millimetres"):
        assert value in text
    assert "coverage_balanced_ordered_pairs" not in text
    assert "noise_pair_budget" not in text
    assert '"cvae": cvae' not in text
    assert '"hybrid": hybrid' not in text
    assert "metric_rows_for_method" not in text


def test_rbsr_never_receives_clean_controls_during_noisy_inference():
    """Only the common STRICT scorer may restore the unnoised controls."""
    text=(ROOT/"scripts/evaluation/posthoc_strict_noise_robustness.py").read_text()
    assert "gate_center, float(gate_package[\"scale\"]),\n                    torch.as_tensor(landmarks, dtype=torch.long, device=device),\n                    noisy_controls" in text
    assert "source, ridge, cvae, gate_values, noisy_controls, landmarks" in text


def test_noise_resume_uses_frozen_clean_baseline_and_restores_real_batch_size():
    import hashlib

    from scripts.evaluation.posthoc_strict_noise_resume_erratum import (
        CANONICAL_NEURAL_BATCH_SIZE,
        canonicalise_runtime_argv,
        reconcile_clean_rows,
    )

    argv = [
        "posthoc_strict_noise_resume_erratum.py",
        "--repo", "/data",
        "--batch-size", "16",
        "--chunk-pairs", "200",
    ]
    corrected = canonicalise_runtime_argv(argv)
    assert CANONICAL_NEURAL_BATCH_SIZE == 32
    assert corrected[corrected.index("--batch-size") + 1] == "32"
    assert corrected[corrected.index("--chunk-pairs") + 1] == "200"
    frozen = ROOT / "scripts/evaluation/posthoc_strict_noise_robustness.py"
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == (
        "7b38010dd80d7420f907a61b2a2c3311f9803c6d68c8fe745edc4549344a2541"
    )

    canonical = [{
        "source_id": "626", "target_id": "444", "roi_rmse": "1.0",
        "normal_flip_pct": "0.29940119760479045", "edge_strain_p95": "0.2",
    }]
    observed = [{
        "source_id": "626", "target_id": "444", "roi_rmse": 1.0 + 1e-7,
        "normal_flip_pct": 0.31241864097891175,
        "edge_strain_p95": 0.2 + 0.00014974518418231497,
        "validation_pair_index": 3623,
    }]
    audit = {
        "certified_rbsr": {
            "rows_checked_live_or_cached": 0,
            "max_abs_roi_rmse_difference": 0.0,
            "max_abs_edge_strain_p95_difference": 0.0,
            "max_abs_normal_flip_pct_difference": 0.0,
            "normal_flip_mismatch_count": 0,
            "normal_flip_mismatch_examples": [],
        }
    }
    reconcile_clean_rows("certified_rbsr", observed, canonical, [0], audit)
    assert observed[0]["normal_flip_pct"] == canonical[0]["normal_flip_pct"]
    assert audit["certified_rbsr"]["normal_flip_mismatch_count"] == 1
    assert audit["certified_rbsr"]["max_abs_edge_strain_p95_difference"] > 1e-5


def test_noise_aggregation_averages_seeds_within_pair_and_keeps_36_test_family(tmp_path):
    noise_root = tmp_path / "noise"
    protocol = noise_root / "STRICT_NOISE_ROBUSTNESS_PROTOCOL_FREEZE.json"
    draw_manifest = noise_root / "SHARED_CONTROL_NOISE_DRAW_MANIFEST.json"
    for path, payload in (
        (protocol, {"status": "test_protocol"}),
        (draw_manifest, {"status": "test_draws"}),
    ):
        atomic_write_json(path, payload)
        write_sha256_sidecar(path)

    pairs = (("1", "2"), ("2", "1"))
    methods = ("ridge", "certified_rbsr", "lamm", "arap")
    levels = (0.0, 0.25, 0.5, 1.0)
    seeds = (20260609, 20260610, 20260611)
    rows = []
    for method_index, method in enumerate(methods):
        for level in levels:
            active_seeds = seeds[:1] if level == 0.0 else seeds
            for seed_index, seed in enumerate(active_seeds):
                for pair_index, (source_id, target_id) in enumerate(pairs):
                    value = method_index + pair_index + level + 0.1 * seed_index
                    rows.append({
                        "method": method,
                        "noise_mm": level,
                        "noise_seed": seed,
                        "validation_pair_index": pair_index,
                        "source_id": source_id,
                        "target_id": target_id,
                        "roi_rmse": value,
                        "normal_flip_pct": value + 1.0,
                        "edge_strain_p95": value + 2.0,
                    })
    atomic_write_csv(noise_root / "chunks/synthetic/rows.csv", rows)
    aggregate_noise(tmp_path, panel_size=2, seed=11, n_boot=100)

    with (noise_root / "noise_pair_metrics_seed_averaged.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        averaged = list(csv.DictReader(handle))
    assert len(averaged) == 4 * 4 * 2
    nonzero = [row for row in averaged if float(row["noise_mm"]) > 0]
    assert {int(row["n_noise_seeds_averaged"]) for row in nonzero} == {3}

    with (noise_root / "noise_vs_clean_paired_statistics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        statistics = list(csv.DictReader(handle))
    assert len(statistics) == 36
    assert {int(row["holm_family_size"]) for row in statistics} == {36}
    assert all("target_ci95_low" in row and "target_ci95_high" in row for row in statistics)
    assert valid_sha256_sidecar(noise_root / "STRICT_NOISE_ROBUSTNESS_EVIDENCE.json")
