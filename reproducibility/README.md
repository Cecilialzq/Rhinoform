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

For the data-free reviewer path, install only `requirements-replay.txt` and the
package with `--no-deps`, then run `python -m tools.reproduce verify` followed
by `python -m tools.reproduce replay`. Replay recomputes the primary
RB-SR-versus-Ridge six-metric family and the supplementary RB-SR-versus-LAMM
three-metric family in a temporary directory and compares every JSON field to
the frozen record.

Equivalent environment variables are `RHINOFORM_DATA_ROOT`,
`RHINOFORM_ARTIFACT_ROOT`, `RHINOFORM_OUTPUT_ROOT`, and `RHINOFORM_LAMM_ROOT`.

FaceScape is licensed separately. The public `data/manifest.json` binds all 846
processed meshes by relative path and SHA-256. The preflight refuses a missing,
renamed, incomplete, or differently processed dataset.

Large checkpoints are not ordinary Git blobs and are currently withheld from
public distribution pending written FaceScape/LAMM permission. An authorised
holder may point `artifact_root` to a private overlay that preserves the
repository-relative paths in `RELEASE_ASSET_MANIFEST.json`; `verify-assets`
checks every byte before an experiment starts. The public repository does not
publish a machine-specific private archive record. Reviewers can run `verify`
and `replay` without checkpoints or FaceScape, or retrain from scratch with
their own licensed data and separately obtained upstream dependencies.

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

## What is and is not guaranteed

The release fails closed on changed manifests, pair ordering, source snapshots,
checkpoint bytes, split metadata, or processed mesh hashes. Statistical replay
is deterministic up to the declared `rtol=1e-12`, `atol=1e-15` platform
roundoff tolerance.

CUDA training is not promised to be bitwise identical across different GPU,
driver and library builds. A full rerun is considered reproduced when it uses
the frozen split/configuration and source lineage, passes the same preflight and
artifact contracts, and agrees within the frozen numerical evaluation
tolerances. The executed notebooks are provenance records; machine paths are
supplied to public commands only through TOML, CLI flags or environment
variables.
