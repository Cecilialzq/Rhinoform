# LAMM baseline contract

This adapter uses the official CVPR 2024 LAMM architecture at commit
`87354c05dec341c6d8dd319665dd52553fb03084`.

The pinned upstream tree does not expose a licence file. For that reason no
upstream source is vendored here: the notebook clones the official repository at
runtime and this repository contains only the independently written dataset,
training-orchestration and evaluation adapter. Clarify upstream reuse terms
before redistributing their source or weights.

Fairness controls:

- exact frozen 676/70/100 identities and 9,900 ordered test pairs;
- train-only per-vertex normalisation;
- five patches are the existing disjoint Rhinoform subunits and cover all 3,934
  vertices exactly;
- the same nine landmarks are assigned to their semantic patch;
- official two-stage scheme: reconstruction pretraining then manipulation;
- seed 20260609 and deterministic PyTorch settings;
- checkpoint choice uses validation identities only;
- test inference receives the dense source and nine control displacements, never
  dense target geometry;
- the shared strict scorer hard-fixes controls and computes free-ROI RMSE,
  dorsum/tip RMSE, target-relative new flips and edge-strain p95.

Default hyperparameters follow the official repository's manipulation model and
two-stage schedule: transformer backbone, 512-D tokens, 256-D bottleneck, 5
encoder and 3 decoder layers, eight heads, 1,500 epochs per stage, AdamW, AE LR
1e-3 and manipulation LR 1e-4. Training uses the official step-wise warm-up and
cosine decay, the 0.25-to-1.0 alpha ramp over the first 100 manipulation epochs,
AE batch size 32 and manipulation batch size 16. The same transformer
architecture is used in reconstruction pretraining so its weights transfer to
the manipulation model.

`--max-test-pairs` is only for a smoke test. Outputs carrying `SMOKE_ONLY` must
never enter the report.

## Progress and interruption recovery

The Colab command uses unbuffered Python and prints every completed training
epoch, validation progress, clean-test batches, noise draws, latency batches,
statistics tests and perceptual-proxy segments. With the notebook's
`--checkpoint-every 1`, each completed epoch is atomically persisted to Drive
with model, optimiser and random-number-generator state plus a SHA-256
sidecar. Validation and best-checkpoint selection remain at the declared
25-epoch interval; checkpoint frequency does not alter model selection.

Clean and noise evaluation resume from validated 320-pair chunks. Paired statistics resume per completed
test, dependence analysis per metric, and perceptual proxies from their latest
atomic pair CSV (written about every 500 pairs). A missing or mismatched
integrity record is never silently accepted as complete. Re-run the interrupted
cell: the log states either `resume from Drive` or the unit being recomputed.

## Common evaluation suite

After the primary run, `evaluate_lamm_suite.py` applies the method-independent
experiments used for the frozen baselines:

- exact 9,900-pair/order and zero-noise reproduction checks;
- four-level control-noise robustness using the frozen artefact's exact NumPy
  draws and validation-derived scale;
- strict five-subunit RMSE/new-flip localisation;
- pair-tail and worst-case summaries;
- batch-1 GPU model-only latency; and
- an optional clean dense-prediction archive for the existing auxiliary
  perceptual-proxy implementation.

The noise command must be given the frozen baseline summary and fails closed if
the levels, seeds or validation median differ. The actual frozen artefact uses
levels `0,0.05,0.10,0.20` and seeds `2026,2027,2028`. Its
`normal_flip_pct` column is the strict target-relative **new-flip** percentage;
the suite also emits `abs_flip_pct` so the two quantities cannot be conflated.

The Colab notebook additionally runs the existing 36-test Holm family,
two-way identity/LOIO dependence checks and two predeclared confirmatory
training seeds (`20260610`, `20260611`). Ridge-PCA, Hybrid-alpha and RB-SR-gate
ablations are method-specific and are therefore not applied to LAMM.
