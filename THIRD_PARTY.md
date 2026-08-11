# Third-party material

Rhinoform's own licensing status is under review; no open-source licence is
currently granted. Any future project licence will not grant rights to datasets,
upstream repositories, or third-party assets.

## FaceScape

FaceScape meshes are not distributed in this repository or its Git history.
Users must obtain the dataset independently through the official FaceScape
access route and comply with its terms. The tracked manifest contains only the
relative processed-file contract and cryptographic hashes required to verify a
licensed local copy.

## LAMM

The LAMM comparison targets upstream commit
`87354c05dec341c6d8dd319665dd52553fb03084`. No upstream LAMM source is vendored.
`experiments/lamm/` contains an independently written FaceScape adapter and
evaluation orchestration. At the time of the frozen run, the pinned upstream
tree did not expose a licence file; users must determine the applicable terms
before using or redistributing upstream source or weights.

## Release artifacts

GitHub Release assets are integrity-bound by
`reproducibility/RELEASE_ASSET_MANIFEST.json`. They contain model/checkpoint
payloads, not FaceScape meshes. Users remain responsible for any conditions
attached to the underlying data and upstream methods.
