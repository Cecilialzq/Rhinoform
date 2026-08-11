# Reproducibility

Rhinoform never requires an evaluator to edit source paths. Machine-dependent
locations are supplied by one TOML file, command-line arguments, or environment
variables. Precedence is CLI > environment > TOML > repository defaults.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp configs/reproduction.example.toml reproduction.local.toml
# Edit only reproduction.local.toml.
python -m tools.reproduce verify --config reproduction.local.toml
python -m tools.reproduce verify-assets --config reproduction.local.toml
python -m tools.reproduce preflight --config reproduction.local.toml --full-data-hash
```

Equivalent environment variables are `RHINOFORM_DATA_ROOT`,
`RHINOFORM_ARTIFACT_ROOT`, `RHINOFORM_OUTPUT_ROOT`, and `RHINOFORM_LAMM_ROOT`.

FaceScape is licensed separately. The public `data/manifest.json` binds all 846
processed meshes by relative path and SHA-256. The preflight refuses a missing,
renamed, incomplete, or differently processed dataset.

Large checkpoints are not ordinary Git blobs. Download the release asset bundle
and either extract it over the checkout or set `artifact_root` to the extracted
overlay. The overlay must preserve repository-relative paths. The authoritative
file list, sizes and hashes are in `RELEASE_ASSET_MANIFEST.json`; `verify-assets`
checks every byte before an experiment starts.

For convenience the same checks are available as `make verify`,
`make verify-assets`, and `make preflight`. Override the default local config
with `make preflight REPRO_CONFIG=/absolute/path/to/config.toml`.

## Source lineages

- `source_snapshots/final_holdout_v1`: the exact 14-file pre-test implementation.
- `source_snapshots/supplemental_dominance_v1`: both the pre-erratum and corrected
  reportable supplemental implementations.
- `source_snapshots/posthoc_subunits_v1`: the frozen five-subunit implementation.

Run `python -m tools.audit_source_snapshots --require-tags` to verify every byte
and confirm that all three annotated Git tags contain the exact manifests.
