# Rhinoform

[![CI](https://github.com/Cecilialzq/Rhinoform/actions/workflows/ci.yml/badge.svg)](https://github.com/Cecilialzq/Rhinoform/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

Rhinoform is a sparse-control deformation method for registered 3D nasal
meshes. Residual-based safety refinement (RB-SR) combines a Ridge anchor, a
learned residual proposal and gate, and a geometric certificate. This is a
research prototype for preoperative design geometry, not a medical device,
diagnostic system, outcome predictor, or clinical planning tool.

## Results

The primary identity-disjoint internal test contains 9,900 matched ordered
pairs.

| Method | ROI RMSE ↓ | Target-relative new flip ↓ | Edge-strain p95 ↓ |
|---|---:|---:|---:|
| Certified RB-SR (PCA-64) | **1.059931** | **0.226894%** | 0.203530 |
| Ridge | 1.073914 | 0.331663% | **0.203410** |
| LAMM | **0.971544** | 0.878723% | 0.319585 |

The RB-SR-versus-Ridge ROI RMSE and new-flip comparisons pass the stored
cluster-aware, Holm-corrected procedure; edge strain is not claimed as
improved. A separate supplementary PCA-128 comparison against LAMM is retained
as a capacity study. Compact report tables are in
[`docs/final_tables/`](docs/final_tables/).

## Verify and replay

The data-free path verifies SHA-256 sidecars and recomputes both stored
pair-level statistical analyses:

```bash
git clone https://github.com/Cecilialzq/Rhinoform.git
cd Rhinoform
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-replay.txt
python -m pip install -e . --no-deps
python -m tools.reproduce verify
python -m tools.reproduce replay
```

Expected statuses are `PASS` and `PASS_RELEASE_STATISTICS_REPLAY`.

For development and tests:

```bash
python -m pip install -r requirements-ci.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

## Licensed data and full runs

FaceScape meshes are not distributed. Obtain them through the official licence
route, build the 846-mesh processed dataset as described in
[`data/README.md`](data/README.md), and copy
[`configs/reproduction.example.toml`](configs/reproduction.example.toml) to the
ignored `reproduction.local.toml`.

```bash
python -m tools.reproduce preflight \
  --config reproduction.local.toml \
  --full-data-hash
```

The preflight binds the local dataset to `data/manifest.json`. Training and
evaluation entry points are under `scripts/` and `experiments/lamm/`; run them
as Python modules so imports do not depend on the checkout path.

## Browser demo

The self-contained demo uses the frozen public bundle and includes golden
conformance checks:

```bash
cd demo
npm install
npm run verify
npm run dev
```

See [`demo/README.md`](demo/README.md) for the bundle contents and production
build command.

## Repository layout

```text
rhinoform/          reusable RB-SR implementation
scripts/            data, training, evaluation, and analysis entry points
experiments/lamm/   LAMM evaluation adapter
roi/                ROI and control definitions
splits/             deterministic split contracts
results/            minimal inputs required for statistical replay
docs/final_tables/  compact result tables
demo/               browser implementation and frozen runtime bundle
tests/              implementation and reproduction tests
```

## License

Project code is available under the [BSD 3-Clause License](LICENSE). FaceScape,
LAMM, and third-party model or data rights are separate; see
[`THIRD_PARTY.md`](THIRD_PARTY.md). Citation metadata is in
[`CITATION.cff`](CITATION.cff).
