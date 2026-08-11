# Final internal rerun holdout results

This directory is the canonical, curated snapshot of the completed Rhinoform
final rerun. It contains the frozen RB-SR, matched Ridge/CVAE/Hybrid, LAMM and
classical-baseline results, all 9,900-pair metric tables, validation-selection
evidence, direct paired statistics, protocol files, test-access receipts,
checkpoints and provenance.

## Permitted claim boundary

These results are clean identity-disjoint evidence from the internal final
rerun holdout. They are not an external holdout and must not be described as a
historically untouched test set. The one-shot test-access receipts record one
access for each evaluation family. The frozen split excludes the 100 identities
from the previously observed test, then uses 576 training, 70 validation and
100 new internal-test identities (9,900 ordered test pairs).

## Primary result

| Method | ROI RMSE | Target-relative new flip |
|---|---:|---:|
| Matched Ridge | 1.073914 | 0.331663% |
| Certified RB-SR | **1.059931** | **0.226894%** |
| LAMM | **0.971544** | 0.878723% |

Certified RB-SR improves both predeclared primary metrics relative to matched
Ridge, with a projection certificate rate of 1.0. Relative to LAMM, RB-SR is a
safety--accuracy trade-off: LAMM has lower RMSE, while RB-SR has substantially
lower new-flip incidence.

## Integrity files

- `FINAL_CONFIRMATION_EVIDENCE.json`: headline result and claim boundary.
- `FINAL_RESULT_FREEZE_AUDIT.json`: final copy, protocol and consistency audit.
- `RESULT_TREE_MANIFEST.json`: SHA-256 and byte size for every canonical file.
- `ALL_MODELS_FROZEN_BEFORE_TEST.json`: pre-test model/implementation freeze.
- `*_TEST_ACCESS_RECEIPT.json`: one-shot access records.

The LAMM configuration-canonicalisation erratum is retained under
`lamm/seed20260609/`. It was recorded before the first LAMM inference and did
not change weights, checkpoints, split, operating point or metric semantics.

## Deliberate release exclusions

Regenerable chunk directories, dense gate maps, progress ledgers and `*_last.pt`
resume checkpoints remain in the read-only `FYP final` archive. Their final
aggregated pair metrics, selected best checkpoints and provenance are included
here. This follows `docs/ARTIFACT_SELECTION.md`.

The two LAMM best checkpoints exceed GitHub's ordinary 100 MB file limit. Use
Git LFS or GitHub Release assets; do not attempt a normal Git blob upload.
