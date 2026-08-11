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

The included `data/manifest.json` records the frozen split and processed dataset metadata used for the final results. Raw meshes and processed mesh arrays are excluded from GitHub.

The ROI crop is generated per subject from full-head FaceScape registered
meshes. The preview meshes in `roi/` are examples only; the crop is not limited
to that preview identity.
