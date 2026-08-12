// Runtime conformance tests for the deployed Rhinoform demo.
//
// 1. Golden Ridge conformance: the browser Ridge adapter must reproduce the
//    authoritative Python Ridge anchor (edit-relative frame) on the frozen
//    golden control vectors within tolerance.
// 2. Bundle integrity: artifact hashes in the manifest must match the files;
//    certified presets must carry the locked model identities and a valid
//    fold-subset certificate.
// 3. Admission policy: the recalibrated thresholds must behave as declared
//    (scale-back lands exactly on the evaluated boundary; outside requests
//    are withheld).
// 4. Display frame (v4): the anchor field is locally smooth at every handle,
//    the ROI boundary rim is tapered to zero, the certified display is
//    exactly zero at zero controls (edit-relative residual), and RB-SR
//    delivers strictly more of the requested handle motion than the pure
//    Ridge anchor (the correction the demo exists to show).
//
// Run: node verify.mjs   (exit code 0 = all checks pass)

import fs from "fs";
import crypto from "crypto";
import {
  slidersToControls,
  humpRow,
  ridgeAnchorDelta,
  landmarkOneRings,
  controlChi,
  applyDelta,
  normalFlipPct,
  deformHead,
} from "./src/engine/geometry.js";
import { admitControls } from "./src/engine/admission.js";

const J = (p) => JSON.parse(fs.readFileSync(new URL(p, import.meta.url)));
const manifest = J("./public/bundle/manifest.json");
const ridge = J("./public/bundle/ridge_browser.json");
const golden = J("./public/bundle/golden_cases.json");
const presets = J("./public/bundle/certified_presets.json");

let failures = 0;
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`);
  if (!ok) failures += 1;
};

const ctx = {
  vertexDim: ridge.n_vertices * 3,
  wCtrl: Float32Array.from(ridge.w_ctrl),
  landmarks: ridge.landmarks,
  taper: Float64Array.from(ridge.display_taper.weights),
  facesFlat: Int32Array.from(ridge.faces.flat()),
};
const source = new Float32Array(ridge.source.flat());

// ---------------------------------------------------------- 1. golden ridge
let maxErr = 0;
for (const gc of golden.cases) {
  const ctrl = Float32Array.from(gc.controls);
  const delta = ridgeAnchorDelta(ctrl, ctx);
  // RMSE against the recorded value
  let acc = 0;
  for (let v = 0; v < ridge.n_vertices; v++) {
    acc += delta[v * 3] ** 2 + delta[v * 3 + 1] ** 2 + delta[v * 3 + 2] ** 2;
  }
  const rmse = Math.sqrt(acc / ridge.n_vertices);
  const rmseErr = Math.abs(rmse - gc.anchor_rmse);
  // first-vertex spot check
  let vertErr = 0;
  gc.anchor_first_vertices.forEach((vv, i) => {
    vertErr = Math.max(
      vertErr,
      Math.abs(delta[i * 3] - vv[0]),
      Math.abs(delta[i * 3 + 1] - vv[1]),
      Math.abs(delta[i * 3 + 2] - vv[2])
    );
  });
  maxErr = Math.max(maxErr, rmseErr, vertErr);
  check(
    `golden ridge conformance: ${gc.name}`,
    rmseErr <= gc.tolerance && vertErr <= gc.tolerance,
    `rmseErr=${rmseErr.toExponential(2)} vertErr=${vertErr.toExponential(2)}`
  );
}
console.log(`      max browser-vs-python deviation: ${maxErr.toExponential(2)}`);

// ------------------------------------------------------ 2. bundle integrity
const sha = (p) => crypto.createHash("sha256").update(fs.readFileSync(new URL(p, import.meta.url))).digest("hex");
check(
  "manifest hash: ridge_browser.json",
  sha("./public/bundle/ridge_browser.json") === manifest.artifacts["ridge_browser.json"]
);
check(
  "manifest hash: certified_presets.json",
  sha("./public/bundle/certified_presets.json") === manifest.artifacts["certified_presets.json"]
);
check(
  "preset identity lock",
  JSON.stringify(presets.locked_identities) === JSON.stringify(manifest.locked_identities)
);
check(
  "all presets carry a fold-subset certificate",
  presets.presets.every((p) => p.certified && p.new_vs_ridge_fold_count === 0),
  `${presets.presets.length} presets`
);
check(
  "frozen projection parameters in manifest",
  manifest.frozen_parameters.projection.attenuation === 0.75 &&
    manifest.frozen_parameters.projection.max_iterations === 64 &&
    manifest.frozen_parameters.projection.uniform_steps === 1001
);
check(
  "locked deployment identities present",
  manifest.locked_identities.base_model_sha256 ===
    "932db0a879a15ee51d8d6d52fb76d380609ed6ee02f85f8ae78afe9e11e8cfd3" &&
    manifest.locked_identities.gate_model_sha256 ===
      "9185d1baac0934b5fdb437ce451296132eba68a83d8f62d1bafe95a1aaa4f972" &&
    manifest.locked_identities.projection_signature ===
      "4a54c0535cff7d6e14fe8d944510e5831a803244892f770ac0d67f8706688e84"
);

// ----------------------------------------------------- 3. admission policy
const matrix = manifest.sliders.matrix_6x27;
const ood = manifest.ood;
const mk = (vals) => slidersToControls(vals, matrix);

const inCtrl = mk([0.3, 0.1, 0, 0, 0, 0]);
const inRes = admitControls(inCtrl, ood);
check("admission: consultation-range edit is within evaluated range", inRes.band === "in" && inRes.scale === 1);

// find a near-boundary request by scaling the extreme preset up
const extreme = mk([0.95, -0.85, 0.9, 0.85, -0.9, 0.65]);
const chiExtreme = controlChi(extreme, ood.per_dim_std);
const nearFactor = (ood.policy.within_evaluated_range * 1.1) / chiExtreme;
const nearCtrl = Float32Array.from(extreme, (v) => v * nearFactor);
const nearRes = admitControls(nearCtrl, ood);
check(
  "admission: near-boundary request scaled back exactly to threshold",
  nearRes.band === "near" &&
    Math.abs(controlChi(nearRes.effectiveCtrl, ood.per_dim_std) - ood.policy.within_evaluated_range) < 1e-6
);

const outFactor = (ood.policy.near_boundary * 1.2) / chiExtreme;
const outCtrl = Float32Array.from(extreme, (v) => v * outFactor);
const outRes = admitControls(outCtrl, ood);
check("admission: outside-range request is withheld", outRes.band === "out" && outRes.withheld);

// The two UI boundary-demonstration presets must land in their advertised
// admission bands (they exist to show scale-back and withholding live).
{
  const { DEMO_EDITS } = await import("./src/engine/demoEdits.js");
  for (const demo of DEMO_EDITS) {
    const res = admitControls(mk(demo.sliders), ood);
    const chi = controlChi(mk(demo.sliders), ood.per_dim_std);
    check(
      `boundary demo preset lands in band "${demo.expectBand}": ${demo.key}`,
      res.band === demo.expectBand &&
        demo.sliders.every((v) => Math.abs(v) <= 2.0),
      `chi=${chi.toFixed(2)}`
    );
  }
}

// Export-session acceptance: the report builder must produce a complete,
// serialisable payload (the browser download is a one-line wrapper on it).
{
  const { buildSessionReport, REPORT_REQUIRED_KEYS } = await import(
    "./src/engine/sessionReport.js"
  );
  const report = buildSessionReport({
    identities: manifest.locked_identities,
    sourceKey: "mean_neutral_roi_of_training_identities_v1",
    certifiedBackend: "in_browser_frozen_pipeline",
    session: { sliders: {}, admission: { chi: 0 } },
    plans: [],
  });
  const roundTrip = JSON.parse(JSON.stringify(report));
  check(
    "export session: report payload carries every required section",
    REPORT_REQUIRED_KEYS.every((k) => roundTrip[k] !== undefined) &&
      roundTrip.deployment.locked_identities.base_model_sha256 ===
        manifest.locked_identities.base_model_sha256 &&
      roundTrip.disclaimer.includes("Not a medical device")
  );
}

// Source-adaptive bridge-smoothing row: the browser formula on the mean
// source must reproduce the manifest's stored (mean-source) row.
{
  const stored = matrix[manifest.sliders.keys.indexOf("bridgeSmoothing")];
  const row = humpRow(source, ridge.landmarks, manifest.sliders.roles);
  let err = 0;
  for (let i = 0; i < 27; i++) err = Math.max(err, Math.abs(row[i] - stored[i]));
  check(
    "bridge smoothing: browser humpRow matches manifest mean-source row",
    err < 1e-3,
    `maxErr=${err.toExponential(2)}`
  );
}

// -------------------- 4. display-frame smoothness at the handle vertices
// The pure learned anchor must be locally smooth at every landmark: the
// landmark displacement must stay close to its one-ring neighbour mean
// (no triangular single-vertex anchor artefacts in the displayed field).
{
  const facesFlat = Int32Array.from(ridge.faces.flat());
  const rings = landmarkOneRings(facesFlat, ridge.landmarks);
  const delta = ridgeAnchorDelta(mk([0.8, 0.6, 0.9, -0.7, 0.5, -0.6]), ctx);
  let worst = 0;
  ridge.landmarks.forEach((lv, j) => {
    const ring = rings[j];
    for (let d = 0; d < 3; d++) {
      let acc = 0;
      for (const nb of ring) acc += delta[nb * 3 + d];
      worst = Math.max(worst, Math.abs(delta[lv * 3 + d] - acc / ring.length));
    }
  });
  check(
    "display smoothness: anchor field has no single-vertex bump at any handle",
    worst < 0.15,
    `max |landmark - ring mean| = ${worst.toFixed(4)} units`
  );

  // ROI boundary rim (taper weight 0) must be exactly frozen for any edit.
  let rimMoved = 0;
  let rimCount = 0;
  for (let v = 0; v < ridge.n_vertices; v++) {
    if (ctx.taper[v] !== 0) continue;
    rimCount++;
    if (Math.hypot(delta[v * 3], delta[v * 3 + 1], delta[v * 3 + 2]) > 1e-9) rimMoved++;
  }
  check(
    "display taper: ROI boundary rim is exactly frozen",
    rimCount > 100 && rimMoved === 0,
    `${rimCount} rim vertices, moved=${rimMoved}`
  );
}

// sanity: anchor at zero controls leaves the source unchanged
const zeroDelta = ridgeAnchorDelta(new Float32Array(27), ctx);
check("edit-relative frame: zero edit produces zero displacement", zeroDelta.every((v) => v === 0));

// descriptive: source self-flip is zero
check(
  "source mesh has no self-flips",
  normalFlipPct(source, applyDelta(source, zeroDelta), ridge.faces) === 0
);

// -------------------------- 5. full-head context (public training mean)
{
  const raw = J("./public/bundle/head_mean.json");
  const head = {
    vertices: new Float32Array(raw.vertices),
    roiIndices: Int32Array.from(raw.roi_indices),
    collar: {
      v: Int32Array.from(raw.collar.v),
      anchor: Int32Array.from(raw.collar.anchor),
      w: Float32Array.from(raw.collar.w),
    },
  };
  check(
    "manifest hash: head_mean.json",
    sha("./public/bundle/head_mean.json") === manifest.artifacts["head_mean.json"]
  );
  const testDelta = ridgeAnchorDelta(mk([0.6, 0.4, -0.4, -0.3, 0.2, 0.1]), ctx);
  const deformed = deformHead(head, testDelta);
  const editable = new Set([...head.roiIndices, ...head.collar.v]);
  let movedLocked = 0;
  let movedEditable = 0;
  const nHead = head.vertices.length / 3;
  for (let v = 0; v < nHead; v++) {
    const d = Math.hypot(
      deformed[v * 3] - head.vertices[v * 3],
      deformed[v * 3 + 1] - head.vertices[v * 3 + 1],
      deformed[v * 3 + 2] - head.vertices[v * 3 + 2]
    );
    if (editable.has(v)) { if (d > 1e-9) movedEditable++; }
    else if (d > 1e-9) movedLocked++;
  }
  check(
    "full-head scatter: vertices outside ROI+collar are locked",
    movedLocked === 0,
    `moved editable=${movedEditable}, locked-that-moved=${movedLocked}/${nHead - editable.size}`
  );
  // The grafted nose region must coincide with the displayed ROI source.
  let graftErr = 0;
  for (let i = 0; i < head.roiIndices.length; i++) {
    const hv = head.roiIndices[i];
    graftErr = Math.max(
      graftErr,
      Math.abs(head.vertices[hv * 3] - source[i * 3]),
      Math.abs(head.vertices[hv * 3 + 1] - source[i * 3 + 1]),
      Math.abs(head.vertices[hv * 3 + 2] - source[i * 3 + 2])
    );
  }
  check("head graft: nose region equals the public ROI source", graftErr < 1e-3,
    `maxErr=${graftErr.toExponential(2)}`);
}

// -------------------- 6. in-browser certified pipeline golden conformance
{
  const { parseCertifiedBundle, certifiedLocal } = await import("./src/engine/certifiedLocal.js");
  const rawCb = J("./public/bundle/certified_browser.json");
  const goldenCert = J("./public/bundle/golden_certified.json");
  check(
    "manifest hash: certified_browser.json",
    sha("./public/bundle/certified_browser.json") === manifest.artifacts["certified_browser.json"]
  );
  check(
    "certified browser identity lock",
    JSON.stringify(rawCb.locked_identities) === JSON.stringify(manifest.locked_identities)
  );
  const cb = parseCertifiedBundle(rawCb);
  const certCtx = {
    source,
    facesFlat: Int32Array.from(ridge.faces.flat()),
    landmarks: ridge.landmarks,
    wCtrl: ctx.wCtrl,
    taper: ctx.taper,
  };
  const tol = goldenCert.tolerances;
  let worst = { rmse: 0, vert: 0, retention: 0, iters: 0, gate: 0 };
  for (const gc of goldenCert.cases) {
    const res = certifiedLocal(Float64Array.from(gc.controls), certCtx, cb);
    let acc = 0;
    for (let i = 0; i < res.certifiedDelta.length; i++) acc += res.certifiedDelta[i] ** 2;
    const rmse = Math.sqrt(acc / ridge.n_vertices);
    const rmseErr = Math.abs(rmse - gc.certified_rmse);
    let vertErr = 0;
    gc.certified_first_vertices.forEach((vv, i) => {
      vertErr = Math.max(
        vertErr,
        Math.abs(res.certifiedDelta[i * 3] - vv[0]),
        Math.abs(res.certifiedDelta[i * 3 + 1] - vv[1]),
        Math.abs(res.certifiedDelta[i * 3 + 2] - vv[2])
      );
    });
    const retErr = Math.abs(res.retention - gc.retention);
    const iterErr = Math.abs(res.iterations - gc.iterations);
    const gateErr = Math.abs(res.gateMean - gc.gate_mean);
    worst = {
      rmse: Math.max(worst.rmse, rmseErr),
      vert: Math.max(worst.vert, vertErr),
      retention: Math.max(worst.retention, retErr),
      iters: Math.max(worst.iters, iterErr),
      gate: Math.max(worst.gate, gateErr),
    };
    check(
      `certified browser conformance: ${gc.name}`,
      res.certified === gc.certified &&
        res.status === gc.status &&
        rmseErr <= tol.certified_rmse_abs &&
        vertErr <= tol.vertex_abs &&
        retErr <= tol.retention_abs &&
        iterErr <= tol.iterations_abs &&
        gateErr <= tol.gate_mean_abs,
      `rmseErr=${rmseErr.toExponential(2)} vertErr=${vertErr.toExponential(2)} iters=${res.iterations}/${gc.iterations}`
    );
  }
  console.log(
    `      certified worst: rmse=${worst.rmse.toExponential(2)} vert=${worst.vert.toExponential(2)} ` +
      `ret=${worst.retention.toExponential(2)} gate=${worst.gate.toExponential(2)}`
  );

  // Edit-relative display: at zero controls the certified display is exactly
  // zero (no static residual bias pattern).
  const zeroRes = certifiedLocal(new Float64Array(27), certCtx, cb);
  check(
    "certified display: zero edit shows exactly zero displacement",
    zeroRes.certifiedDelta.every((v) => v === 0)
  );

  // RB-SR must deliver strictly more of the requested handle motion than the
  // pure Ridge anchor on every non-zero golden case (the correction the demo
  // exists to demonstrate).
  let allBetter = true;
  let summary = [];
  for (const gc of goldenCert.cases) {
    if (gc.controls.every((v) => v === 0)) continue;
    const res = certifiedLocal(Float64Array.from(gc.controls), certCtx, cb);
    if (!(res.realisationCertified > res.realisationAnchor)) allBetter = false;
    summary.push(`${gc.name}: ${res.realisationAnchor.toFixed(2)}->${res.realisationCertified.toFixed(2)}`);
  }
  check(
    "RB-SR correction: certified realises more of the requested edit than Ridge",
    allBetter
  );
  console.log(`      realisation (ridge->rbsr): ${summary.join("  ")}`);
}

// -------------------- 7. uploaded-source rebind conformance (OBJ path)
{
  const { parseCertifiedBundle, certifiedLocal } = await import("./src/engine/certifiedLocal.js");
  const { parseMesh, classifyUpload, rebindSource } = await import("./src/engine/uploadSource.js");
  const { readFileSync } = await import("node:fs");
  const goldenUp = J("./public/bundle/golden_upload.json");
  check(
    "manifest hash: upload_support.bin",
    sha("./public/bundle/upload_support.bin") === manifest.artifacts["upload_support.bin"]
  );
  check(
    "manifest hash: golden_upload.json",
    sha("./public/bundle/golden_upload.json") === manifest.artifacts["golden_upload.json"]
  );

  // Rebuild the deterministic perturbed fixture (same formula as
  // deploy/export_upload_support.py::perturbed_source).
  const V = ridge.n_vertices;
  const mean = [0, 0, 0];
  for (let v = 0; v < V; v++) for (let d = 0; d < 3; d++) mean[d] += source[v * 3 + d];
  for (let d = 0; d < 3; d++) mean[d] /= V;
  const fixture = new Float64Array(V * 3);
  for (let v = 0; v < V; v++) {
    const cx = source[v * 3] - mean[0];
    const cy = source[v * 3 + 1] - mean[1];
    const cz = source[v * 3 + 2] - mean[2];
    fixture[v * 3] = source[v * 3] + 0.8 * Math.sin(0.11 * cy) * Math.cos(0.07 * cz);
    fixture[v * 3 + 1] = source[v * 3 + 1] + 0.6 * Math.sin(0.09 * cx + 1.3);
    fixture[v * 3 + 2] = source[v * 3 + 2] + 0.7 * Math.cos(0.10 * cx - 0.5) * Math.sin(0.08 * cy);
  }

  // Exercise the real upload path: serialise as Wavefront OBJ (the format
  // FaceScape registered meshes ship in), parse via the unified entry point,
  // classify.
  const rows = ["# rhinoform verify fixture"];
  for (let v = 0; v < V; v++) {
    rows.push(`v ${fixture[v * 3]} ${fixture[v * 3 + 1]} ${fixture[v * 3 + 2]}`);
  }
  const objText = rows.join("\n") + "\n";
  const parsed = parseMesh(new TextEncoder().encode(objText).buffer, "fixture.obj");
  const { kind, roi } = classifyUpload(parsed, source, null);
  check("upload: OBJ fixture parses and classifies as ROI", kind === "roi" && roi.length === V * 3);

  // Bind the fixture with the shipped support tensors (node-side loader).
  const buf = readFileSync("./public/bundle/upload_support.bin");
  const all = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const dir = manifest.upload_support.tensors;
  const grab = (name) => {
    const { offset, shape } = dir[name];
    const len = shape.reduce((a, b) => a * b, 1);
    return all.subarray(offset, offset + len);
  };
  const support = {
    pcaMean: grab("pca_mean"),
    pcaComponents: grab("pca_components"),
    ridgeW: grab("ridge_w"),
    ridgeB: grab("ridge_b"),
    condMean: grab("cond_mean"),
    condStd: grab("cond_std"),
    gateCenter: grab("gate_center"),
    gateScale: grab("gate_scale")[0],
  };
  const cb0 = parseCertifiedBundle(J("./public/bundle/certified_browser.json"));
  const cb = rebindSource(cb0, support, roi);
  const upCtx = {
    source: new Float32Array(roi),
    facesFlat: Int32Array.from(ridge.faces.flat()),
    landmarks: ridge.landmarks,
    wCtrl: ctx.wCtrl,
    taper: ctx.taper,
  };
  const tol = goldenUp.tolerances;
  for (const gc of goldenUp.cases) {
    const res = certifiedLocal(Float64Array.from(gc.controls), upCtx, cb);
    let acc = 0;
    for (let i = 0; i < res.certifiedDelta.length; i++) acc += res.certifiedDelta[i] ** 2;
    const rmse = Math.sqrt(acc / V);
    const rmseErr = Math.abs(rmse - gc.certified_rmse);
    let vertErr = 0;
    gc.certified_first_vertices.forEach((vv, i) => {
      vertErr = Math.max(
        vertErr,
        Math.abs(res.certifiedDelta[i * 3] - vv[0]),
        Math.abs(res.certifiedDelta[i * 3 + 1] - vv[1]),
        Math.abs(res.certifiedDelta[i * 3 + 2] - vv[2])
      );
    });
    const retErr = Math.abs(res.retention - gc.retention);
    const iterErr = Math.abs(res.iterations - gc.iterations);
    const gateErr = Math.abs(res.gateMean - gc.gate_mean);
    check(
      `uploaded-source conformance: ${gc.name}`,
      res.certified === gc.certified &&
        rmseErr <= tol.certified_rmse_abs &&
        vertErr <= tol.vertex_abs &&
        retErr <= tol.retention_abs &&
        iterErr <= tol.iterations_abs &&
        gateErr <= tol.gate_mean_abs,
      `rmseErr=${rmseErr.toExponential(2)} vertErr=${vertErr.toExponential(2)} iters=${res.iterations}/${gc.iterations}`
    );
  }

  // Guardrails: wrong vertex count and off-scale units must be rejected.
  let rejectedCount = false;
  try {
    classifyUpload({ vertices: new Float64Array(300), count: 100 }, source, null);
  } catch { rejectedCount = true; }
  let rejectedScale = false;
  try {
    const mm = new Float64Array(roi.length);
    for (let i = 0; i < roi.length; i++) mm[i] = roi[i] * 25.4; // wrong units
    classifyUpload({ vertices: mm, count: V }, source, null);
  } catch { rejectedScale = true; }
  check("upload: wrong vertex count is rejected", rejectedCount);
  check("upload: off-scale units are rejected", rejectedScale);
}

console.log(failures === 0 ? "\nALL CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
