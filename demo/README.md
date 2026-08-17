# Rhinoform browser demo

This browser implementation provides an interactive Ridge anchor and an
asynchronous Certified RB-SR enhancement. Certificate failure, runtime failure,
timeout, or bundle identity mismatch leaves the interface on the deterministic
anchor.

This is a research prototype, not a medical device. It does not predict
surgical outcomes, diagnose conditions, or provide clinical recommendations.

## Run

```bash
npm install
npm run verify
npm run dev
npm run build
```

`verify.mjs` checks bundle hashes, Python-reference golden cases, uploaded-mesh
goldens, and fallback behaviour. `npm run build` creates the ignored `dist/`
directory.

## Frozen bundle

`public/bundle/` contains the files required by the browser runtime:

- `manifest.json`: bundle identities and runtime parameters
- `ridge_browser.json`: Ridge operator and mean ROI source
- `certified_browser.json`: residual/gate weights and projection parameters
- `certified_presets.json`: example certified outputs
- `head_mean.json`: mean full-head display context
- `upload_support.bin`: tensors for registered OBJ/PLY uploads
- `golden_cases.json`, `golden_certified.json`, `golden_upload.json`: conformance inputs

Registered OBJ/PLY uploads are processed entirely in the browser. The demo does
not include FaceScape scans, a mesh registration service, or a private model
export pipeline.
