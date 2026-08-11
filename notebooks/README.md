# Notebooks

The notebooks in this directory are the executed Colab records for the three
frozen experiment families. Their saved outputs document progress, patches,
selection gates and final completion. They intentionally preserve the original
Colab/Drive execution layout and are not the machine-portable public entry
point.

For a location-independent reproduction, start at the repository root with:

```bash
python -m tools.reproduce verify
python -m tools.reproduce replay
```

For artifact/data preflight, edit only `reproduction.local.toml` as described in
the root README. The source snapshots under `reproducibility/source_snapshots/`
are authoritative when a later shared implementation differs from the exact
implementation used by a frozen experiment.
