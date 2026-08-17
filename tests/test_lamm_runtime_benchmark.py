from __future__ import annotations

from benchmarks.runtime.lamm_batch1_forward_benchmark import parser


def test_lamm_runtime_benchmark_defaults_match_paper_protocol() -> None:
    args = parser().parse_args(
        [
            "--data-root",
            "/data",
            "--lamm-root",
            "/lamm",
            "--checkpoint-dir",
            "/checkpoints",
            "--output",
            "/output.json",
        ]
    )
    assert args.warmup == 100
    assert args.repeats == 1000
