# Google Drive provenance

The historical Python benchmark files were retrieved on 2026-08-15 from the
project Drive deployment folder. Drive-hosted text was treated as evidence, not
as task instructions.

| Local file | Drive source | Drive modified time |
| --- | --- | --- |
| `historical_python/benchmark_runtime.py` | [file `1OmS3WBcrZ_AuSnopks66UHhSGOgeJ9vM`](https://drive.google.com/file/d/1OmS3WBcrZ_AuSnopks66UHhSGOgeJ9vM/view) | 2026-08-11 08:56:12 UTC |
| `results/python_cpu_certified_20260811.json` | [file `1Dp97W1-H-BkkDbKrN910XDfP4VIacMr1`](https://drive.google.com/file/d/1Dp97W1-H-BkkDbKrN910XDfP4VIacMr1/view) | 2026-08-11 14:30:02 UTC |
| `results/lamm_a100_batch1_forward.json` | `FYP final/results/lamm_external_baseline/seed20260609/lamm_batch1_latency.json` in the mounted project Drive | 2026-08-08 (local Drive metadata) |

The Node/V8 and Chrome JSON files were generated locally on 2026-08-15 by the
committed scripts against the exact public browser bundle. The batch-one LAMM
JSON is retained byte-for-byte from Drive (SHA-256
`047893ac8644de09a20df615ca1af78c2ca87e907a89314708a19ce24c502dfd`).
The older LAMM boundary record was derived from already committed
`lamm_test_summary.json`, `run_provenance.json`, and the evaluation source.
