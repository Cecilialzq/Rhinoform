# Third-party material

Rhinoform-authored code is licensed under BSD-3-Clause. This does not grant
rights to datasets, upstream repositories, or third-party/derived assets.

## FaceScape

FaceScape meshes are not distributed in this repository or its Git history.
Users must obtain the dataset independently through the
[official FaceScape access route](https://nju-3dv.github.io/projects/FaceScape/)
and comply with its
[dataset agreement](https://nju-3dv.github.io/projects/FaceScape/static/license/LicenseAgreement_FaceScape.pdf).
The tracked manifest contains only the relative processed-file contract and
cryptographic hashes required to verify a licensed local copy.

The FaceScape agreement permits limited, non-transferable, non-commercial
research use, prohibits redistribution of the dataset, and requires prior
written permission for uses not permitted or specified. It does not expressly
authorise public redistribution of trained checkpoints. Consequently all
FaceScape-trained checkpoint payloads are withheld from the public release
unless and until CITE LAB grants written permission.

## LAMM

The LAMM comparison targets
[upstream commit `87354c05dec341c6d8dd319665dd52553fb03084`](https://github.com/michaeltrs/LAMM/tree/87354c05dec341c6d8dd319665dd52553fb03084).
No upstream LAMM source is vendored.
`experiments/lamm/` contains an independently written FaceScape adapter and
evaluation orchestration. At the time of the frozen run, the pinned upstream
tree did not expose a licence file; users must determine the applicable terms
before using or redistributing upstream source or weights.

GitHub's [default copyright rules](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
therefore apply to that upstream tree: public
visibility permits viewing and forking through GitHub's service terms but is not
an open-source grant to reproduce, distribute, or create derivative works.
Rhinoform does not vendor LAMM source. The frozen LAMM checkpoints are also
withheld pending written authorisation.

## Frozen artifacts

The author's private checkpoint archive is integrity-bound by
`reproducibility/RELEASE_ASSET_MANIFEST.json` and
`reproducibility/RELEASE_ARCHIVE.json`. These records are supplied for
provenance; the archive is not authorised for public distribution. The public
repository instead supplies frozen pair-level research evidence, hashes,
configuration, source snapshots and from-scratch reproduction code.
