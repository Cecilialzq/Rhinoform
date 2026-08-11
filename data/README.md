# Data

This repository does not include FaceScape meshes. FaceScape is a licensed dataset and must be obtained through the official access route.

Expected local layout:

```text
data/
  manifest.json
  meshes/
    1_neutral.npz
    2_neutral.npz
    ...
```

The included `data/manifest.json` records the immutable 846-mesh preprocessing
contract. Its parent split reference is dataset-construction provenance, not
the report-facing final split. Raw meshes and processed mesh arrays are
excluded from GitHub.

The ROI crop is generated per subject from full-head FaceScape registered
meshes. The preview meshes in `roi/` are examples only; the crop is not limited
to that preview identity.

## Rebuild from a licensed FaceScape copy

Obtain the FaceScape train-set archives through the official licence route.
Do not copy them into this Git repository. The following commands use example
locations; replace only command-line paths, never source code:

```bash
python -m scripts.data_processing.fetch_facescape \
  --tu-model-dir /absolute/path/to/facescape_archives \
  --scratch /absolute/path/to/facescape_scratch \
  --out /absolute/path/to/facescape_scratch/stage.json \
  --raw-zip-manifest /absolute/path/to/facescape_scratch/raw_zip_manifest.json

python -m scripts.data_processing.build_data \
  --obj-dir /absolute/path/to/facescape_scratch/tu_model_unzipped \
  --roi roi/vertices.json \
  --subunits roi/subunits.json \
  --split-manifest splits/facescape_847/split_manifest.json \
  --out /absolute/path/to/processed_facescape
```

The output must contain 846 `meshes/<id>_neutral.npz` files. Copy the canonical
`data/manifest.json` into that output only after verifying that the generated
file names and SHA-256 values match it. Then set `data_root` in the ignored
`reproduction.local.toml` and run:

```bash
python -m tools.reproduce preflight \
  --config reproduction.local.toml \
  --full-data-hash
```

The experiment-specific final holdout identities are not inferred from local
folder names. Training and evaluation consume the tracked frozen protocol at
`results/rbsr_final_rerun_holdout_v1/protocol/final_rerun_holdout_split_manifest.json`.
See `splits/README.md` for the distinction between the 847-identity source QC,
the 846 usable meshes and the final 576/70/100 experiment split.
