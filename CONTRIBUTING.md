# Contributing

Bug reports and reproducibility reports are welcome. Please include the exact
Git commit, Python/PyTorch/CUDA versions, the command run, and the complete
terminal error. Never attach FaceScape meshes or other licensed data.

Before opening a pull request:

```bash
python -m pytest -q
python -m tools.reproduce verify
python -m tools.reproduce replay
python -m tools.audit_public_release
```

Do not edit frozen evidence files, source snapshots, access receipts, hashes, or
annotated evidence tags. New experimental evidence must be written to a new
directory with its own protocol, provenance and manifest.
