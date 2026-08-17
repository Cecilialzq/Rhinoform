# Reproducibility

Two public checks require neither licensed meshes nor checkpoints:

```bash
python -m tools.reproduce verify
python -m tools.reproduce replay
```

`verify` checks the primary split/training protocol, the four retained
9,900-pair inputs, their stored statistical outputs, and the compact paper-table
manifest. `replay`
recomputes the primary RB-SR-versus-Ridge and supplementary
RB-SR-versus-LAMM analyses in a temporary directory and compares every output
field. Floating-point values use `rtol=1e-12` and `atol=1e-15`; hashes, row
counts, ordering, schemas, and categorical results must match exactly.

## Full-data preflight

FaceScape is licensed separately. Build the processed dataset using
[`../data/README.md`](../data/README.md), then configure its location without
editing source code:

```bash
cp configs/reproduction.example.toml reproduction.local.toml
# Edit data_root only.
python -m tools.reproduce preflight \
  --config reproduction.local.toml \
  --full-data-hash
```

CLI `--data-root` takes precedence over `RHINOFORM_DATA_ROOT`, which takes
precedence over the TOML value. The preflight verifies the canonical manifest
and all 846 mesh hashes before training or evaluation.

The repository keeps the primary split/training protocol and the result files
needed for public statistical replay. Model training and inference require the
user's own licensed data, checkpoints, and upstream LAMM checkout. Historical
source states remain available through Git tags rather than duplicated source
trees.
