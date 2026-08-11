# Final artifact selection

The public submission repository is intentionally limited to two result families. Older scientific iterations remain recoverable in the sibling private Google Drive archive `Rhinoform_Private_Archive_20260810`; they are not part of the paper evidence chain.

| Artifact family | Disposition | Reason |
|---|---|---|
| `results/rbsr_final_rerun_holdout_v1/` | keep intact | primary frozen protocol, models, pair metrics, baselines, statistics and freeze audit |
| `results/supplemental_rbsr_lamm_dominance_search_v1/` | keep curated | supplementary PCA-128 matched evidence, selected packages, validation selection, pair metrics and release manifest |
| two retained notebooks | keep | exact primary and supplementary workflows |
| `rhinoform/`, relevant `scripts/`, `experiments/lamm/`, `tests/` | keep | implementation and verification |
| `roi/`, `data/`, `splits/` | keep as infrastructure | protocol/data construction support; not separate paper result families |
| `docs/final_tables/` | keep | single paper-facing table source generated from frozen evidence |
| legacy notebooks and notebook builders | private archive | superseded or unexecuted workflows would confuse the release boundary |
| old result trees and old report snapshots | private archive | not used by the final report and not mergeable with the final-rerun protocol |
| old pending contracts and audit plans | private archive | superseded by completed evidence |
| dominance dense chunk caches and last-epoch checkpoints | private archive | regenerable; canonical pair CSVs, selected packages and hashes remain |
| Python caches and test caches | remove | regenerable build artifacts |

## Why the old ablations are not retained as paper evidence

An ablation is only interpretable when its split, source identities, features, scorer and target pair ordering match the claimed final model. Old-protocol ablations cannot be numerically merged into the final-rerun table. The retained final validation chain already supplies the necessary component evidence:

- Ridge anchor;
- CVAE proposer;
- fixed Hybrid;
- raw learned RB-SR gate; and
- frozen ridge-fold certified RB-SR.

This supports the mechanism claim without retraining. A new ablation is required only if the report wants an additional claim not covered by this chain, such as multi-seed stability, a new loss-term attribution, runtime/memory scaling, or subunit-specific robustness.

## Non-destructive curation

Scientific artifacts were moved, not deleted. Only regenerable caches were removed. The archive includes a pre-curation path/size inventory and a partial cloud-placeholder hash inventory; the canonical retained result manifests remain the integrity authority.
