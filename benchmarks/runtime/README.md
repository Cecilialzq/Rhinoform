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

## Historical Python CPU benchmark

`historical_python/benchmark_runtime.py` and
`results/python_cpu_certified_20260811.json` preserve the canonical deployment
record found in the project Drive. It used the same locked base, gate, and
projection identities, batch size 1, 100 warmups, and 1,000 measured requests.
It records a 50.70-ms median certified end-to-end CPU time, 85.88-ms p95,
132.97-ms p99, and 220.2-MB peak RSS. The source script requires the private
licensed deployment runtime and is archived for provenance; the public browser
benchmark above is the independently runnable release check.

## LAMM timing boundary

The only surviving LAMM timing record is preserved in
`results/lamm_historical_a100_noncomparable.json`. It reports 1.380 ms per pair
for model forward only on an NVIDIA A100 at batch size 32. It has no stored p95
or p99 samples and is not a same-hardware, batch-size-1 deployment comparison.
It is therefore reported as historical provenance, not used to rank deployment
latency.

## Interpretation

The evidence supports an immediate Ridge interaction path and an asynchronous
certified enhancement. It does not support describing certified RB-SR as a
per-frame 60-Hz path. The user-facing deployment should therefore keep Ridge as
the real-time fallback and position certified RB-SR as an enhanced preview.
