# Reproducibility

The public release keeps an evidence path for every experiment reported in the
dissertation. The machine-readable index is
[`PAPER_EXPERIMENT_MANIFEST.json`](PAPER_EXPERIMENT_MANIFEST.json); each entry
names the paper figure or table and its notebook, implementation, frozen
protocol, and numerical or figure output.

## Data-free verification and replay

These checks require neither licensed meshes nor checkpoints:

```bash
python -m tools.reproduce verify
python -m tools.reproduce replay
```

`verify` checks:

- all paths in the seven paper experiment families;
- the three executed Colab notebooks under [`../notebooks/`](../notebooks/);
- the exact historical source snapshots and their annotated Git tags;
- protocol, result, table, and evidence sidecars;
- every frozen publication figure listed by `FIGURE_EVIDENCE.json`; and
- the runtime records used by Table 5.12 against `benchmarks/runtime/SHA256SUMS`.

`replay` recomputes the primary RB-SR-versus-Ridge and supplementary
RB-SR-versus-LAMM analyses in a temporary directory and compares every output
field. Floating-point values use `rtol=1e-12` and `atol=1e-15`; hashes, row
counts, ordering, schemas, and categorical results must match exactly.

The Post hoc notebook is intentionally public. It is the executed record for
the regional, spatial, qualitative, source-conditioning, and control-noise
experiments reported in Figures 4.3(c), 5.2-5.6, 5.8, Tables 5.7-5.9, and
Appendix Figures A.1-A.2. Its generated numerical outputs and figures live under
[`../results/rbsr_final_rerun_holdout_v1/posthoc_subunit_analysis_v1/`](../results/rbsr_final_rerun_holdout_v1/posthoc_subunit_analysis_v1/).

## Historical source runtimes

The final holdout, supplementary capacity study, and Post hoc study were run
from frozen source states. Reconstruct an isolated runtime without importing
the current checkout:

```bash
python -m tools.reproduce materialize --experiment final-holdout
python -m tools.reproduce materialize --experiment supplemental-dominance
python -m tools.reproduce materialize --experiment posthoc-subunits
```

The materializer combines the experiment's annotated Git tag with its archived
source overlay in [`source_snapshots/`](source_snapshots/) and verifies import
origins and file hashes.

## Full-data preflight

FaceScape is licensed separately. Build the processed dataset using
[`../data/README.md`](../data/README.md), then configure its location without
editing source code:

```bash
cp configs/reproduction.example.toml reproduction.local.toml
# Edit data_root, artifact_root and lamm_root as applicable.
python -m tools.reproduce preflight \
  --config reproduction.local.toml \
  --full-data-hash
```

CLI arguments take precedence over `RHINOFORM_*` environment variables, which
take precedence over the TOML values. The preflight verifies the canonical
manifest and all 846 mesh hashes, then checks the checkpoint overlay against
[`RELEASE_ASSET_MANIFEST.json`](RELEASE_ASSET_MANIFEST.json).

FaceScape meshes, LAMM weights, and checkpoints derived from restricted inputs
are not redistributed. Their exact relative paths, byte sizes, and SHA-256
hashes are retained so an authorised holder can reconstruct the same overlay
without editing repository source.
