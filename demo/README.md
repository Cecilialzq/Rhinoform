# RHINOFORM — Interactive Nasal Preview with Certified Enhancement (research prototype)

A public demo of the Rhinoform system: a population-informed sparse-control
editor for interactive nasal morphology preview. The deployed architecture is
the paper's final frozen pipeline:

- **Ridge anchor (synchronous).** The exact frozen ridge operator (PCA-64 source
  basis, λ = 300) runs in the browser and answers every slider event
  immediately. The displayed field is the pure learned deformation at native
  amplitude (no display-frame handle enforcement, no control amplification),
  multiplied by an ROI boundary taper (smoothstep of the geodesic distance to
  the ROI boundary; band 5 units near the landmarks, widening smoothly to 12
  units in rim regions far from every landmark) and then rim-pinned Laplacian
  smoothed (12 rounds, λ = 0.5, boundary ring re-clamped to zero every round),
  so the rim never follows an edit and the taper never concentrates strain
  into a wrinkle ring at the ROI boundary. The frozen slider map applies a 2×
  row gain to tip rotation, tip projection and dorsum height (compensating the
  ridge regulariser's amplitude shrinkage on those directions); the OOD χ is
  computed on the resulting control vector, so the admission policy is
  unchanged.
- **Certified RB-SR (asynchronous enhancement).** The frozen CVAE residual
  proposer and learned spatial gate, followed by the validation-frozen
  fold-subset projection (attenuation 0.75, ≤ 64 rounds, 1001-step uniform
  fallback). A result is shown only if its fold set is a subset of the Ridge
  anchor's. The certified display adds the correction as the EDIT-RELATIVE
  projected residual (resid(c) − resid(0), removing the CVAE's
  edit-independent bias), landmark-filled and Laplacian-smoothed, in the same
  tapered and rim-pinned-smoothed display frame — it substantially raises the
  realised fraction of the requested handle motion (0.30–0.59 for the linear
  anchor vs 0.49–0.76 certified across the golden edits, reported live in the
  research panel) at anchor-level smoothness; at zero edit both displays are
  identical.
- **Automatic fallback.** Any certificate failure, runtime error, timeout or
  identity mismatch keeps the preview on the deterministic Ridge anchor, and
  the interface says so.

> **Research prototype — not a medical device.** Outputs do not predict
> surgical or anatomical results. Coordinates are uncalibrated FaceScape model
> units. For morphology communication only.

## Deployment identity (fail-closed)

Everything numerical derives from the frozen final-rerun artifacts
(`results/rbsr_final_rerun_holdout_v1`, seed 20260609). The deployed identities
and bundle-file hashes are recorded in `public/bundle/manifest.json` and
re-checked by `verify.mjs`:

| artifact | sha256 |
| --- | --- |
| base model package | `932db0a8…` |
| gate model package | `9185d1ba…` |
| projection freeze signature | `4a54c053…` |

If any identity check fails, Certified RB-SR is disabled and only the Ridge
anchor is served.

## Public browser runtime

| component | public implementation |
| --- | --- |
| Ridge anchor | exact in-browser operator, evaluated on every slider event |
| Certified RB-SR | in-browser port of the frozen pipeline (`src/engine/certifiedLocal.js`), golden-verified against the Python reference; 11 presets are also precomputed |
| source mesh | population-mean neutral ROI in a training-mean full-head context, or a user-uploaded registered OBJ/PLY bound entirely in the browser |

The browser runtime is self-contained. It may probe an optional loopback-only
research service at `127.0.0.1:8321`, but the public release does not require or
include that private service and falls back to the in-browser implementation.

First load fetches ≈18 MB of frozen artifacts with a live progress bar and a
retry control; `vercel.json` serves `/bundle/*` with
`Cache-Control: max-age=86400, stale-while-revalidate=604800`, so repeat
visits start instantly from the browser cache.

**Custom source upload.** The studio's *Source model* panel accepts an OBJ or PLY mesh
registered to the frozen correspondence — the nasal ROI (3,934 vertices) or a
full head (26,317 vertices, ROI extracted via the frozen index map) — in
calibrated model units. The client fetches `upload_support.bin` lazily,
recomputes the source-dependent statistics (source-PCA code, conditional Ridge
baseline, gate normalisation) and runs certified inference on the uploaded
identity. The mesh never leaves the browser. Presets stay bound to the mean
example; a one-click control restores it.

## Two separate guards (never blended)

1. **Input applicability (OOD policy).** χ = ‖c/σ‖₂ against the frozen
   validation control distribution. Requests near the boundary are scaled back
   (requested vs effective controls are both recorded); requests beyond it are
   withheld. Thresholds live in `public/bundle/manifest.json`.
2. **Residual certificate.** The certified result's fold set must be a subset
   of the Ridge anchor's fold set. Retention and projection rounds are shown
   whenever attenuation was required.

The UI never uses the words "safe" or "confidence" for either mechanism.

## Runtime state machine

`Idle` → `Editing · Anchor preview` → `Computing certified enhancement` →
`Certified enhancement available` / `Enhancement constrained` /
`Anchor fallback`, plus `Request outside evaluated range` for withheld
requests. Exactly one state is active at any time.

## Repository layout

```
public/bundle/
  manifest.json            identities, frozen params, OOD policy, slider map
  ridge_browser.json       exact browser Ridge operator + mean-ROI source
  certified_browser.json   frozen CVAE/gate weights + projection for the browser
  certified_presets.json   precomputed certified results (11 in-range presets; the two
                           boundary demonstrations in src/engine/demoEdits.js exercise
                           scale-back and withholding live and are UI-level by design)
  head_mean.json           training-mean full-head display context
  upload_support.bin       source-binding tensors for user-uploaded meshes
  golden_cases.json        Python-reference Ridge goldens for verify.mjs
  golden_certified.json    Python-reference certified goldens for verify.mjs
  golden_upload.json       Python-reference goldens for the uploaded-source path
src/engine/
  bundle.js  geometry.js  admission.js  certified.js  certifiedLocal.js  uploadSource.js
local_private/             real-identity meshes — never deployed or committed
verify.mjs                 conformance suite (runs in the Vercel build)
```

## Commands

```bash
npm install
npm run verify        # 51 conformance checks (Ridge + certified + upload goldens, hashes, fail-closed policy, display frame, boundary demos, session export)
npm run dev           # dev site on 127.0.0.1:5173
npm run build         # production build (verify runs first on Vercel)
npm run benchmark:compute  # exact bundle compute: 100 warmups + 1,000 batch-one requests
npm run benchmark:render   # Chrome presentation/input-to-paint: 100 warmups + 1,000 requests
```

The two benchmark commands intentionally keep numerical compute and browser
presentation separate. Canonical results and memory-scope caveats are in
`../benchmarks/runtime/README.md`.

The private export pipeline used to regenerate `public/bundle/` is intentionally
not part of the web deployment. The public conformance suite binds every
published bundle file to its frozen reference.

## Compliance

The project author publishes only aggregate, project-authored deployment
artifacts derived during licensed FaceScape research: the training-identity
mean neutral ROI, a training-mean full-head display context, and transformed
browser model parameters. No original or single-identity FaceScape scan is
included. Access to the FaceScape dataset itself remains governed by its own
agreement and must be requested from its provider. User-uploaded meshes are
parsed, validated and bound entirely client-side and are never transmitted to
any server; the upload path requires meshes already registered to the frozen
correspondence, so no acquisition or registration capability is claimed.
