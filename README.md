# Rhinoform

[![CI](https://github.com/Cecilialzq/Rhinoform/actions/workflows/ci.yml/badge.svg)](https://github.com/Cecilialzq/Rhinoform/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

Rhinoform is a safety-aware method for sparse-control deformation of registered
3D nasal meshes. Its proposed **residual-based safety refinement (RB-SR)**
combines a Ridge anchor, a learned residual proposer and gate, and a hard
geometric certificate. The project studies preoperative design geometry; it is
not a clinical planning, diagnostic, outcome-prediction or medical device tool.

This public repository contains the implementation, frozen pair-level evidence,
exact split and ROI contracts, source snapshots, tests, and portable verification
commands. FaceScape meshes are licensed separately and are not redistributed.

## Results and claim boundary

The primary experiment is a clean identity-disjoint internal final rerun on
9,900 matched ordered test pairs. The deployed PCA-64 certified model improves
the predeclared accuracy and target-relative new-flip objectives over its matched
Ridge anchor.

| Method | ROI RMSE ↓ | Target-relative new flip ↓ | Edge-strain p95 ↓ |
|---|---:|---:|---:|
| Certified RB-SR (primary, PCA-64) | **1.059931** | **0.226894%** | 0.203530 |
| Ridge | 1.073914 | 0.331663% | **0.203410** |
| LAMM | **0.971544** | 0.878723% | 0.319585 |

Direct RB-SR-versus-Ridge tests support the ROI RMSE and new-flip improvements
after Holm correction. Edge strain is statistically indistinguishable from
Ridge and is not claimed as improved.

A separate **supplementary post-hoc** PCA-128 operating point is evaluated on
the same 9,900 pairs and with the same strict scorer. It obtains ROI RMSE
0.915294, new flip 0.793430%, and edge-strain p95 0.299634, all below the frozen
LAMM values; all three matched comparisons pass the stored cluster-aware,
Holm-corrected procedure. This supplementary model is capacity/trade-off
evidence, not the primary confirmatory model.

The authoritative wording and evidence files are mapped in
[`docs/FINAL_EXPERIMENT_MAP.md`](docs/FINAL_EXPERIMENT_MAP.md) and
[`docs/EVIDENCE_STATUS.md`](docs/EVIDENCE_STATUS.md). Do not merge the PCA-64
and PCA-128 rows into a synthetic result.

## Reproduce in under one minute

The fastest independent check needs no FaceScape data, checkpoint, GPU, Colab,
or personal path. It verifies the frozen evidence and recomputes both report-level
statistical analyses from the tracked 9,900-pair tables.

```bash
git clone https://github.com/Cecilialzq/Rhinoform.git
cd Rhinoform
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-replay.txt
python -m pip install -e . --no-deps

python -m tools.reproduce verify
python -m tools.reproduce replay
```

Expected terminal statuses are `PASS` and
`PASS_RELEASE_STATISTICS_REPLAY`. The replay uses `rtol=1e-12` and
`atol=1e-15` only for cross-platform floating-point roundoff; categorical
judgements, pair ordering, row counts, hashes and schema fields must match
exactly.

## Runtime and deployment benchmark

The release includes batch-size-one, 100-warmup/1,000-run compute and
browser-presentation benchmarks for the exact certified browser bundle. See
[`benchmarks/runtime/README.md`](benchmarks/runtime/README.md) for the scripts,
committed JSON results, environment, memory scope, and the explicit separation
between compute-only and rendering measurements.

The measured product contract is Ridge as the interactive anchor/fallback and
certified RB-SR as an asynchronous enhanced preview. A recovered A100
batch-size-one LAMM forward distribution is also preserved, but it is not
ranked against the Apple-M3 paths; no browser GPU peak-memory ranking is claimed.

## Installation

The frozen environment uses Python 3.10+ and the versions in
[`requirements.txt`](requirements.txt). A CUDA-capable PyTorch installation is
needed only for model inference or training.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements-ci.txt
python -m pytest -q
```

Always invoke repository entry points as modules (`python -m ...`). This makes
imports independent of the checkout location and avoids the `No module named
'rhinoform'` failure caused by executing nested scripts directly.

## Data and release assets

1. Obtain FaceScape through its official licence route.
2. Rebuild the processed 846-mesh cache using [`data/README.md`](data/README.md).
3. Train the models from the frozen configuration using your licensed local
   FaceScape copy and a separately obtained LAMM checkout. The author's frozen
   checkpoint archive is currently withheld pending written third-party
   permission; it is not a public GitHub Release asset.
4. Copy [`configs/reproduction.example.toml`](configs/reproduction.example.toml)
   to the Git-ignored `reproduction.local.toml` and edit paths only.

```bash
python -m tools.reproduce verify-assets --config reproduction.local.toml
python -m tools.reproduce preflight \
  --config reproduction.local.toml \
  --full-data-hash
python -m tools.reproduce materialize \
  --experiment final-holdout \
  --config reproduction.local.toml
```

Preflight validates the canonical manifest, all 846 processed mesh hashes, the
split, every released model artifact, and all SHA-256 sidecars before expensive
work starts. CLI arguments and the environment variables
`RHINOFORM_DATA_ROOT`, `RHINOFORM_ARTIFACT_ROOT`, `RHINOFORM_OUTPUT_ROOT`, and
`RHINOFORM_LAMM_ROOT` can replace the TOML file; no source edit is required.

`materialize` constructs a non-overwriting historical runtime from the
annotated `final-holdout-v1` tag, overlays the exact 14-file source snapshot and
executes an isolated import-origin check for the four RB-SR files that later
changed. Historical stages must then be launched with `runtime-exec`; this
prevents the current root implementation from entering a v1 reproduction. See
[`reproducibility/README.md`](reproducibility/README.md) for the exact contract
and example command.

The exact private frozen-checkpoint inventory is
[`reproducibility/RELEASE_ASSET_MANIFEST.json`](reproducibility/RELEASE_ASSET_MANIFEST.json).
The manifest preserves the repository-relative path, byte size and SHA-256 of
each retained checkpoint without publishing a private local archive record.
It does not grant or imply redistribution permission. `verify-assets` is
available only to an authorised holder of those exact bytes; public evidence
verification and statistical replay need neither checkpoints nor FaceScape.
The original executed Colab notebooks are retained as protocol/provenance
records; portable verification starts from the commands above, not from their
author-specific Drive mount cells.

## Reproducibility levels

| Level | Inputs | Typical purpose | Command |
|---|---|---|---|
| Evidence verification | Git checkout | Check frozen statuses and source snapshots | `python -m tools.reproduce verify` |
| Statistical replay | Git checkout | Recompute the paper-level paired inference | `python -m tools.reproduce replay` |
| Private artifact preflight | Checkout + authorised frozen checkpoints + processed FaceScape | Prove the author's retained end-to-end inputs match | `python -m tools.reproduce preflight --config reproduction.local.toml --full-data-hash` |
| Historical source runtime | Checkout with annotated tags | Build and verify a tag-bound, non-hybrid final-holdout v1 execution tree | `python -m tools.reproduce materialize --experiment final-holdout --config reproduction.local.toml` |
| Full GPU rerun | Checkout + licensed FaceScape + separately obtained pinned LAMM + CUDA | Re-run training/evaluation from scratch | See [`reproducibility/README.md`](reproducibility/README.md) |

GPU training can differ at the last floating-point bits across CUDA, driver and
hardware versions. The release therefore guarantees exact inputs, code lineage,
pair ordering and artifact hashes, and uses frozen metric tolerances where
bitwise GPU equality is not technically portable. It does not claim universal
bit-for-bit retraining on arbitrary hardware.

## Repository layout

```text
rhinoform/                 reusable RB-SR implementation
scripts/                   data, training, evaluation and analysis modules
experiments/lamm/          independently written LAMM evaluation adapter
roi/                       frozen ROI, subunit and control definitions
splits/                    deterministic dataset/split construction contracts
results/                   primary, supplementary and post-hoc evidence
docs/final_tables/         report tables derived from frozen evidence
reproducibility/           source snapshots and release-asset manifests
notebooks/                 executed experiment provenance records
tests/                     protocol, geometry, statistics and release tests
```

The three annotated evidence tags are `final-holdout-v1`,
`supplemental-dominance-v1`, and `posthoc-subunits-v1`. Verify their byte-level
source snapshots with:

```bash
python -m tools.audit_source_snapshots --require-tags
```

## Citation

GitHub exposes the repository citation from [`CITATION.cff`](CITATION.cff).
Until a report DOI is available, cite the software release and include the exact
Git commit and evidence tag used.

## License and third-party material

Code authored for Rhinoform is released under the
[`BSD 3-Clause License`](LICENSE). That license does not cover FaceScape,
upstream LAMM, or checkpoint/model payloads whose redistribution rights are not
established. See [`THIRD_PARTY.md`](THIRD_PARTY.md) before downloading data,
running LAMM, or sharing trained artifacts.
