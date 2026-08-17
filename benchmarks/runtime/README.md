# Runtime and deployment benchmarks

This directory separates numerical compute from browser presentation. Results
are committed as evidence records rather than inferred from architecture or old
screenshots.

## Current public-browser compute benchmark

Run from the repository root:

```bash
npm --prefix demo install
npm --prefix demo run benchmark:compute
```

`browser_certified_benchmark.mjs` executes the exact weights and projection
parameters in `demo/public/bundle/`. The protocol is batch size 1, 100 warmup
requests, and 1,000 measured requests drawn round-robin from the 11 frozen
presets at 0.5x, 1.0x, and 1.5x amplitude. It reports Ridge, raw RB-SR proposal,
certificate projection, display preparation, and certified end-to-end compute
separately. React/Three.js rendering is excluded.

The recorded Apple M3 result is
`results/browser_node_certified.json`:

| Compute stage | median (ms) | p95 (ms) | p99 (ms) |
| --- | ---: | ---: | ---: |
| Ridge anchor | 0.670 | 0.855 | 2.144 |
| Raw RB-SR proposal | 76.837 | 88.788 | 98.574 |
| Certificate projection | 1.693 | 2.589 | 3.051 |
| Display preparation | 2.803 | 4.298 | 6.348 |
| Certified RB-SR end to end | 81.487 | 94.661 | 108.881 |

All 1,000 measured requests were certified with zero new folds relative to the
Ridge anchor. Projection iterations were 14.53 on average, 23 at p95, and 25 at
maximum. The maximum sampled Node RSS was 156.0 MB. That RSS value is a sampled
process measure, not a browser-tab or GPU-memory peak.

## Browser presentation benchmark

On macOS with Google Chrome installed at its default path:

```bash
npm --prefix demo run benchmark:render
```

`browser_render_benchmark.mjs` runs the real React/Three.js interface in
headless Chrome. It records 1,000 steady presentation-frame intervals and 1,000
Ridge slider input-to-second-animation-frame observations after 100 warmups.
This is a browser scheduling/presentation measurement. It intentionally does
not relabel the 450-ms certified debounce as compute time, and it does not claim
that the second animation frame is a hardware GPU timer.

Chrome-process-tree RSS is sampled by process role. Chrome on macOS exposes no
stable per-tab peak GPU-memory counter through this protocol, so no GPU-memory
claim is made.

The recorded Apple M3 / Chrome 151 result is
`results/browser_chrome_render.json`: presentation frame intervals were
16.700/18.600/18.601 ms at median/p95/p99; the conservative Ridge
input-to-second-animation-frame proxy was 52.800/66.110/68.504 ms. Maximum
observed RSS across the whole headless Chrome process tree was 1,044.9 MB; this
is not a per-tab allocation, and the per-role maxima are not simultaneous
addends.

## LAMM timing boundary

The recovered `results/lamm_a100_batch1_forward.json` is the applicable
batch-size-one record: NVIDIA A100-SXM4-80GB, 100 warmups and 1,000 measured
forwards, with inputs pre-staged on the GPU. Its model-only latency is
37.085/39.093/41.931 ms at median/p95/p99. It excludes normalisation,
host--device transfer, strict scoring and rendering. Because the public-browser
benchmark above was measured on Apple M3 hardware, these rows must not be used
as a cross-device method ranking.

Reproduce the exact batch-one timing loop with authorised inputs:

```bash
python -m benchmarks.runtime.lamm_batch1_forward_benchmark \
  --data-root /absolute/path/to/processed_facescape \
  --lamm-root /absolute/path/to/LAMM \
  --checkpoint-dir /absolute/overlay/results/rbsr_final_rerun_holdout_v1/lamm/seed20260609 \
  --output reproduction_output/lamm_batch1_forward.json
```

The script verifies the official LAMM commit, clean checkout, checkpoint
sidecar and frozen split before running 100 warmups and 1,000 synchronized
forwards. It is not a same-hardware comparison with the M3 browser pipeline.

## Interpretation

The evidence supports an immediate Ridge interaction path and an asynchronous
certified enhancement. It does not support describing certified RB-SR as a
per-frame 60-Hz path. The user-facing deployment should therefore keep Ridge as
the real-time fallback and position certified RB-SR as an enhanced preview.
