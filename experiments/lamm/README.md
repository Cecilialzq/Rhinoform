# LAMM baseline contract

This adapter uses the official CVPR 2024 LAMM architecture at commit
`87354c05dec341c6d8dd319665dd52553fb03084`.

The pinned upstream tree does not expose a licence file. For that reason no
upstream source is vendored here: the notebook clones the official repository at
runtime and this repository contains only the independently written dataset,
training-orchestration and evaluation adapter. Clarify upstream reuse terms
before redistributing their source or weights.

Fairness controls:

- exact frozen 576/70/100 final-rerun identities and 9,900 ordered test pairs;
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

## Retained evaluation surface

The final confirmation notebook evaluates LAMM once on the same 9,900 ordered
pairs and with the same strict scorer used for every method. The five-subunit,
noise and qualitative analyses are implemented by the separate frozen post-hoc
notebook and shared method-independent evaluators. Historical suite runners and
multi-seed development utilities are intentionally not part of the public
release surface.

In every retained table, `normal_flip_pct` is the strict target-relative
**new-flip** percentage. `abs_flip_pct` is emitted separately and must not be
substituted for it.
