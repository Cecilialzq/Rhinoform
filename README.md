# Rhinoform

This repository contains the implementation and the two experiment families used by the final report. FaceScape meshes are licensed separately and are not included.

## Final evidence boundary

The paper uses exactly two evidence families:

1. **Primary internal final-rerun confirmation** — the frozen PCA-64 certified RB-SR model and all seven matched baselines on the same 9,900 ordered test pairs.
2. **Supplementary post-hoc matched comparison** — a separately selected PCA-128 calibrated RB-SR operating point compared with the already frozen LAMM predictions on the same 9,900 ordered test pairs.

These are complementary, not one merged model. The PCA-64 certified model is the primary deployable accuracy–safety result. The PCA-128 result is capacity/trade-off evidence and must remain labelled supplementary post-hoc.

The final paper tables are generated directly from the frozen JSON and pair-level evidence in [`docs/final_tables`](docs/final_tables/README.md). The complete claim map is in [`docs/FINAL_EXPERIMENT_MAP.md`](docs/FINAL_EXPERIMENT_MAP.md).

## Main result

The primary predeclared comparison passed on 9,900 matched test pairs:

| Method | ROI RMSE ↓ | Target-relative new flip % ↓ | Edge-strain p95 ↓ |
|---|---:|---:|---:|
| Certified RB-SR | 1.059931 | 0.226894 | 0.203530 |
| Ridge | 1.073914 | 0.331663 | 0.203410 |
| LAMM | 0.971544 | 0.878723 | 0.319585 |

Direct paired RB-SR-versus-Ridge tests support the RMSE and new-flip improvements after Holm correction. Edge strain is statistically indistinguishable from Ridge, not improved.

The supplementary PCA-128 comparison obtains ROI RMSE 0.915294, new flip 0.793430%, and edge-strain p95 0.299634, all lower than frozen LAMM (0.971544, 0.878723%, 0.319585) with matched-pair significance. This is not the primary confirmatory headline.

## Repository layout

```text
rhinoform/                 reusable implementation
scripts/                   training, evaluation and analysis entry points
experiments/lamm/          LAMM baseline adapter
notebooks/                 three retained Colab experiment notebooks
splits/                    dataset construction inputs; not a paper result family
roi/                       ROI definitions
results/rbsr_final_rerun_holdout_v1/
                           primary frozen evidence
results/supplemental_rbsr_lamm_dominance_search_v1/
                           supplementary post-hoc matched evidence
docs/final_tables/         paper-ready tables derived from frozen evidence
```

## Retained notebooks

- `notebooks/Rhinoform_RBSR_internal_final_rerun_holdout_colab.ipynb`
- `notebooks/Rhinoform_RBSR_LAMM_dominance_search_colab.ipynb`
- `notebooks/Rhinoform_final_rerun_posthoc_subunits_colab.ipynb`

All three notebooks stage immutable data under Colab `/content`, persist
resumable checkpoints and outputs to Drive, use atomic writes and SHA-256
sidecars, and stream live progress. The result directories, rather than
notebook display output, are the evidence of record.

## Reproducibility and release

The primary result tree is bound by `RESULT_TREE_MANIFEST.json` and `FINAL_RESULT_FREEZE_AUDIT.json`. The supplementary evidence has its own `RELEASE_MANIFEST.json`. The large LAMM checkpoints exceed ordinary GitHub blob limits and must be published through Git LFS or release assets.

Evaluators do not edit source paths. Copy `configs/reproduction.example.toml` to
the ignored `reproduction.local.toml`, point it at a separately licensed and
processed FaceScape directory and an extracted release-asset overlay, then run:

```bash
python -m tools.reproduce verify --config reproduction.local.toml
python -m tools.reproduce verify-assets --config reproduction.local.toml
python -m tools.reproduce preflight --config reproduction.local.toml --full-data-hash
```

The three exact implementation lineages and their manifests are in
`reproducibility/source_snapshots`. See `reproducibility/README.md` for the
path-independent contract and tag verification command.

## Scope

This is non-clinical geometric research. It does not support medical, surgical, airway, diagnostic or aesthetic claims.
