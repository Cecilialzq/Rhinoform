// In-browser certified RB-SR pipeline (exact port of the frozen Chapter-4
// chain, weights from certified_browser.json, hash-locked in the manifest):
//
//   cond(c) -> deterministic CVAE prior-mean field -> spatial gate
//   -> exact 9-handle contract -> ridge-relative fold-subset projection
//
// The projection loop runs on the browser-computed fields, so the fold-subset
// certificate reported here is true for exactly the mesh being displayed.
// Conformance against the authoritative Python runtime is enforced by
// golden_certified.json in `npm run verify`.

import {
  landmarkOneRings,
  vertexOneRings,
  smoothField,
  applyTaper,
  displaySmooth,
} from "./geometry.js";

function silu(x) {
  return x / (1 + Math.exp(-x));
}

// y = W x + b, W stored row-major (nOut x nIn)
function linear(W, b, x, nOut, nIn, out) {
  for (let j = 0; j < nOut; j++) {
    let acc = b[j];
    const base = j * nIn;
    for (let k = 0; k < nIn; k++) acc += W[base + k] * x[k];
    out[j] = acc;
  }
  return out;
}

function layernorm(x, g, b, n) {
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) {
    const d = x[i] - mean;
    variance += d * d;
  }
  variance /= n;
  const inv = 1 / Math.sqrt(variance + 1e-5);
  for (let i = 0; i < n; i++) x[i] = (x[i] - mean) * inv * g[i] + b[i];
  return x;
}

export function parseCertifiedBundle(raw) {
  const F = (a) => Float64Array.from(a);
  return {
    nVertices: raw.n_vertices,
    featureDim: raw.feature_dim,
    vertexFeatures: F(raw.vertex_features),
    cond0: F(raw.cond0_norm),
    invStd27: F(raw.inv_std27),
    r0Raw: F(raw.r0_raw),
    sourceNormalized: F(raw.source_normalized),
    cvae: Object.fromEntries(Object.entries(raw.cvae).map(([k, v]) => [k, F(v)])),
    gate: Object.fromEntries(Object.entries(raw.gate).map(([k, v]) => [k, F(v)])),
    projection: raw.projection,
  };
}

function buildCond(cb, controls) {
  const cond = Float64Array.from(cb.cond0);
  for (let i = 0; i < 27; i++) cond[i] += controls[i] * cb.invStd27[i];
  return cond;
}

// Deterministic CVAE prior-mean dense field (absolute frame), (V*3).
function cvaeField(cb, cond) {
  const { cvae } = cb;
  const h = new Float64Array(128);
  linear(cvae.prior_w, cvae.prior_b, cond, 128, 91, h);
  for (let i = 0; i < 128; i++) h[i] = silu(h[i]);
  layernorm(h, cvae.prior_ln_g, cvae.prior_ln_b, 128);
  const z = new Float64Array(8);
  linear(cvae.pmu_w, cvae.pmu_b, h, 8, 128, z);

  const condz = new Float64Array(99);
  condz.set(cond, 0);
  condz.set(z, 91);

  const V = cb.nVertices;
  const Fd = cb.featureDim; // 15; field input = [vf(15), condz(99)] = 114
  // Shared part of layer 0: columns Fd..114 applied to condz
  const shared = new Float64Array(128);
  for (let j = 0; j < 128; j++) {
    let acc = cvae.f0_b[j];
    const base = j * 114 + Fd;
    for (let k = 0; k < 99; k++) acc += cvae.f0_w[base + k] * condz[k];
    shared[j] = acc;
  }
  const out = new Float64Array(V * 3);
  const x1 = new Float64Array(128);
  const x2 = new Float64Array(128);
  for (let v = 0; v < V; v++) {
    const vf = v * Fd;
    for (let j = 0; j < 128; j++) {
      let acc = shared[j];
      const base = j * 114;
      for (let k = 0; k < Fd; k++) acc += cvae.f0_w[base + k] * cb.vertexFeatures[vf + k];
      x1[j] = silu(acc);
    }
    layernorm(x1, cvae.f2_g, cvae.f2_b, 128);
    linear(cvae.f3_w, cvae.f3_b, x1, 128, 128, x2);
    for (let j = 0; j < 128; j++) x2[j] = silu(x2[j]);
    layernorm(x2, cvae.f5_g, cvae.f5_b, 128);
    for (let d = 0; d < 3; d++) {
      let acc = cvae.f6_b[d];
      const base = d * 128;
      for (let k = 0; k < 128; k++) acc += cvae.f6_w[base + k] * x2[k];
      out[v * 3 + d] = acc;
    }
  }
  return out;
}

// Spatial gate values in [0,1], one per vertex. anchor = unenforced ridge
// (absolute), residual = cvae - anchor, exactly as predict_gate_batches.
function gateField(cb, cond, anchor, residual) {
  const { gate } = cb;
  const V = cb.nVertices;
  const Fd = cb.featureDim; // gate input = [vf(15), cond(91), src(3), anchor(3), resid(3)] = 115
  const shared = new Float64Array(64);
  for (let j = 0; j < 64; j++) {
    let acc = gate.b0_b[j];
    const base = j * 115 + Fd;
    for (let k = 0; k < 91; k++) acc += gate.b0_w[base + k] * cond[k];
    shared[j] = acc;
  }
  const out = new Float64Array(V);
  const x1 = new Float64Array(64);
  const x2 = new Float64Array(64);
  for (let v = 0; v < V; v++) {
    const vf = v * Fd;
    for (let j = 0; j < 64; j++) {
      let acc = shared[j];
      const base = j * 115;
      for (let k = 0; k < Fd; k++) acc += gate.b0_w[base + k] * cb.vertexFeatures[vf + k];
      const tail = base + Fd + 91;
      acc += gate.b0_w[tail] * cb.sourceNormalized[v * 3]
           + gate.b0_w[tail + 1] * cb.sourceNormalized[v * 3 + 1]
           + gate.b0_w[tail + 2] * cb.sourceNormalized[v * 3 + 2]
           + gate.b0_w[tail + 3] * anchor[v * 3]
           + gate.b0_w[tail + 4] * anchor[v * 3 + 1]
           + gate.b0_w[tail + 5] * anchor[v * 3 + 2]
           + gate.b0_w[tail + 6] * residual[v * 3]
           + gate.b0_w[tail + 7] * residual[v * 3 + 1]
           + gate.b0_w[tail + 8] * residual[v * 3 + 2];
      x1[j] = silu(acc);
    }
    layernorm(x1, gate.b2_g, gate.b2_b, 64);
    linear(gate.b3_w, gate.b3_b, x1, 64, 64, x2);
    for (let j = 0; j < 64; j++) x2[j] = silu(x2[j]);
    layernorm(x2, gate.b5_g, gate.b5_b, 64);
    let acc = gate.logit_b[0];
    for (let k = 0; k < 64; k++) acc += gate.logit_w[k] * x2[k];
    out[v] = 1 / (1 + Math.exp(-acc));
  }
  return out;
}

// Per-face orientation mask (signed 2D Jacobian determinant in the source
// tangent frame < 0), exact port of signed_fold_indicator.
function foldMask(source, delta, faces, out) {
  const nF = faces.length / 3;
  const EPS = 1e-12;
  for (let f = 0; f < nF; f++) {
    const a = faces[f * 3], b = faces[f * 3 + 1], c = faces[f * 3 + 2];
    const sax = source[a * 3], say = source[a * 3 + 1], saz = source[a * 3 + 2];
    const se1x = source[b * 3] - sax, se1y = source[b * 3 + 1] - say, se1z = source[b * 3 + 2] - saz;
    const se2x = source[c * 3] - sax, se2y = source[c * 3 + 1] - say, se2z = source[c * 3 + 2] - saz;
    const eax = sax + delta[a * 3], eay = say + delta[a * 3 + 1], eaz = saz + delta[a * 3 + 2];
    const ee1x = source[b * 3] + delta[b * 3] - eax;
    const ee1y = source[b * 3 + 1] + delta[b * 3 + 1] - eay;
    const ee1z = source[b * 3 + 2] + delta[b * 3 + 2] - eaz;
    const ee2x = source[c * 3] + delta[c * 3] - eax;
    const ee2y = source[c * 3 + 1] + delta[c * 3 + 1] - eay;
    const ee2z = source[c * 3 + 2] + delta[c * 3 + 2] - eaz;

    const l1 = Math.max(Math.sqrt(se1x * se1x + se1y * se1y + se1z * se1z), EPS);
    const ux = se1x / l1, uy = se1y / l1, uz = se1z / l1;
    const nx = se1y * se2z - se1z * se2y;
    const ny = se1z * se2x - se1x * se2z;
    const nz = se1x * se2y - se1y * se2x;
    const nn = Math.max(Math.sqrt(nx * nx + ny * ny + nz * nz), EPS);
    const wx = nx / nn, wy = ny / nn, wz = nz / nn;
    const vx = wy * uz - wz * uy;
    const vy = wz * ux - wx * uz;
    const vz = wx * uy - wy * ux;
    const A = ee1x * ux + ee1y * uy + ee1z * uz;
    const B = ee1x * vx + ee1y * vy + ee1z * vz;
    const C = ee2x * ux + ee2y * uy + ee2z * uz;
    const D = ee2x * vx + ee2y * vy + ee2z * vz;
    out[f] = A * D - B * C < 0 ? 1 : 0;
  }
  return out;
}

// Full certified RB-SR chain in the browser. `ctx` is the loaded bundle
// (source, faces, landmarks, wCtrl); `cb` the parsed certified bundle.
// Returns the same result contract as the live service / preset library,
// with certifiedDelta in the edit-relative display frame.
export function certifiedLocal(controls, ctx, cb, options = {}) {
  const t0 = performance.now();
  const V = cb.nVertices;
  const D = V * 3;
  const faces = ctx.facesFlat;
  const landmarks = ctx.landmarks;
  const source = ctx.source;

  const cond = buildCond(cb, controls);
  const cvae = cvaeField(cb, cond);

  // Unenforced absolute ridge R(c) = r0_raw + (c/sigma) @ W_ctrl
  const anchorAbs = Float64Array.from(cb.r0Raw);
  for (let k = 0; k < 27; k++) {
    const c = controls[k];
    if (c === 0) continue;
    const base = k * D;
    for (let i = 0; i < D; i++) anchorAbs[i] += c * ctx.wCtrl[base + i];
  }
  const gateResidual = new Float64Array(D);
  for (let i = 0; i < D; i++) gateResidual[i] = cvae[i] - anchorAbs[i];
  const gate = gateField(cb, cond, anchorAbs, gateResidual);

  // Exact-handle ridge and the gated, handle-neutralised residual
  const ridgeFixed = Float64Array.from(anchorAbs);
  for (let j = 0; j < landmarks.length; j++) {
    const lv = landmarks[j];
    ridgeFixed[lv * 3] = controls[j * 3];
    ridgeFixed[lv * 3 + 1] = controls[j * 3 + 1];
    ridgeFixed[lv * 3 + 2] = controls[j * 3 + 2];
  }
  const residual = new Float64Array(D);
  for (let v = 0; v < V; v++) {
    const g = gate[v];
    residual[v * 3] = g * (cvae[v * 3] - ridgeFixed[v * 3]);
    residual[v * 3 + 1] = g * (cvae[v * 3 + 1] - ridgeFixed[v * 3 + 1]);
    residual[v * 3 + 2] = g * (cvae[v * 3 + 2] - ridgeFixed[v * 3 + 2]);
  }
  for (const lv of landmarks) {
    residual[lv * 3] = 0;
    residual[lv * 3 + 1] = 0;
    residual[lv * 3 + 2] = 0;
  }
  const tProposal = options.collectTimings ? performance.now() : 0;

  // Ridge-relative fold-subset projection (local attenuation, then uniform)
  const { attenuation, max_iterations, uniform_steps } = cb.projection;
  const nFaces = faces.length / 3;
  const ridgeFold = foldMask(source, ridgeFixed, faces, new Uint8Array(nFaces));
  const fold = new Uint8Array(nFaces);
  const weights = new Float64Array(V).fill(1);
  for (const lv of landmarks) weights[lv] = 0;
  const delta = new Float64Array(D);

  const composeDelta = (w) => {
    for (let v = 0; v < V; v++) {
      const wv = w[v];
      delta[v * 3] = ridgeFixed[v * 3] + wv * residual[v * 3];
      delta[v * 3 + 1] = ridgeFixed[v * 3 + 1] + wv * residual[v * 3 + 1];
      delta[v * 3 + 2] = ridgeFixed[v * 3 + 2] + wv * residual[v * 3 + 2];
    }
  };
  const newFoldFaces = () => {
    const bad = [];
    for (let f = 0; f < nFaces; f++) if (fold[f] && !ridgeFold[f]) bad.push(f);
    return bad;
  };

  let certified = false;
  let status = "";
  let iterations = 0;
  for (let iter = 0; iter <= max_iterations; iter++) {
    composeDelta(weights);
    foldMask(source, delta, faces, fold);
    const bad = newFoldFaces();
    if (bad.length === 0) {
      certified = true;
      status = "local_fold_subset_certified";
      iterations = iter;
      break;
    }
    if (iter === max_iterations) break;
    const badVerts = new Set();
    for (const f of bad) {
      badVerts.add(faces[f * 3]);
      badVerts.add(faces[f * 3 + 1]);
      badVerts.add(faces[f * 3 + 2]);
    }
    for (const v of badVerts) weights[v] *= attenuation;
    for (const lv of landmarks) weights[lv] = 0;
  }
  if (!certified) {
    const uw = new Float64Array(V);
    for (let s = 0; s < uniform_steps; s++) {
      const alpha = 1 - s / (uniform_steps - 1);
      uw.fill(alpha);
      for (const lv of landmarks) uw[lv] = 0;
      composeDelta(uw);
      foldMask(source, delta, faces, fold);
      if (newFoldFaces().length === 0) {
        certified = true;
        status = "local_failed_uniform_fold_subset_certified";
        iterations = max_iterations;
        break;
      }
    }
  }
  const tProjection = options.collectTimings ? performance.now() : 0;

  // Retention and fold bookkeeping (same definitions as safe_fusion)
  let residNorm = 0;
  let diffNorm = 0;
  for (let i = 0; i < D; i++) {
    residNorm += residual[i] * residual[i];
    const d = delta[i] - ridgeFixed[i];
    diffNorm += d * d;
  }
  residNorm = Math.sqrt(residNorm);
  const retention = residNorm <= 1e-12
    ? 1
    : Math.min(Math.max(Math.sqrt(diffNorm) / residNorm, 0), 1);
  let ridgeFoldCount = 0;
  let projectedFoldCount = 0;
  let newVsRidgeFoldCount = 0;
  for (let f = 0; f < nFaces; f++) {
    if (ridgeFold[f]) ridgeFoldCount++;
    if (fold[f]) projectedFoldCount++;
    if (fold[f] && !ridgeFold[f]) newVsRidgeFoldCount++;
  }

  // Display frame (matches the Python runtime, display convention v4):
  //   certifiedDelta = taper * ((anchorAbs - R(0)) + smooth(resid(c) - resid(0)))
  // 1. resid = delta - ridgeFixed, with the nine landmark-pinned values
  //    filled by their one-ring neighbour mean (no single-vertex dimple);
  // 2. minus the zero-controls residual resid(0) (the CVAE's edit-
  //    independent bias field, which would otherwise show as a static bump
  //    pattern along the dorsum); cached per certified bundle / source;
  // 3. DISPLAY_SMOOTH_K rounds of Laplacian smoothing (high-frequency
  //    proposal noise only -- the correction is low/mid frequency);
  // 4. ROI boundary taper, then rim-pinned smoothing (diffuses the strain
  //    the taper concentrates in its band; rim stays exactly frozen).
  //    At zero controls the display is exactly zero.
  const resid = new Float64Array(D);
  for (let i = 0; i < D; i++) resid[i] = delta[i] - ridgeFixed[i];
  const rings = landmarkOneRings(ctx.facesFlat, ctx.landmarks);
  for (let j = 0; j < ctx.landmarks.length; j++) {
    const lv = ctx.landmarks[j];
    const ring = rings[j];
    for (let d = 0; d < 3; d++) {
      let acc = 0;
      for (const nb of ring) acc += resid[nb * 3 + d];
      resid[lv * 3 + d] = acc / ring.length;
    }
  }
  let isZeroEdit = true;
  for (let k = 0; k < 27; k++) {
    if (controls[k] !== 0) {
      isZeroEdit = false;
      break;
    }
  }
  if (isZeroEdit) {
    cb.resid0Filled = resid.slice();
  } else if (!cb.resid0Filled) {
    certifiedLocal(new Float64Array(27), ctx, cb); // caches cb.resid0Filled
  }
  const certifiedDelta = new Float32Array(D);
  if (!isZeroEdit) {
    const editResid = new Float64Array(D);
    for (let i = 0; i < D; i++) editResid[i] = resid[i] - cb.resid0Filled[i];
    if (!cb.vertexRings) cb.vertexRings = vertexOneRings(ctx.facesFlat, V);
    const smoothed = smoothField(editResid, cb.vertexRings);
    for (let i = 0; i < D; i++) {
      certifiedDelta[i] = anchorAbs[i] - cb.r0Raw[i] + smoothed[i];
    }
    applyTaper(certifiedDelta, ctx.taper);
    certifiedDelta.set(displaySmooth(certifiedDelta, ctx));
  }

  // Requested handle motion delivered along the request direction, for the
  // anchor display and the certified display (research panel).
  const anchorDisp = new Float64Array(D);
  for (let i = 0; i < D; i++) anchorDisp[i] = anchorAbs[i] - cb.r0Raw[i];
  applyTaper(anchorDisp, ctx.taper);
  anchorDisp.set(displaySmooth(anchorDisp, ctx));
  let reqSq = 0, achAnchor = 0, achCert = 0;
  for (let j = 0; j < landmarks.length; j++) {
    const lv = landmarks[j];
    for (let d = 0; d < 3; d++) {
      const req = controls[j * 3 + d];
      reqSq += req * req;
      achAnchor += req * anchorDisp[lv * 3 + d];
      achCert += req * certifiedDelta[lv * 3 + d];
    }
  }
  const realisationAnchor = reqSq > 1e-12 ? achAnchor / reqSq : 1;
  const realisationCertified = reqSq > 1e-12 ? achCert / reqSq : 1;

  // Gate summary stats for the research panel
  const sortedGate = Float64Array.from(gate).sort();
  const gateMean = gate.reduce((a, b) => a + b, 0) / V;

  const tEnd = performance.now();
  return {
    available: true,
    backend: "browser",
    certified,
    status,
    retention,
    iterations,
    ridgeFoldCount,
    projectedFoldCount,
    newVsRidgeFoldCount,
    gateP50: sortedGate[Math.floor(V * 0.5)],
    gateP95: sortedGate[Math.floor(V * 0.95)],
    gateMean,
    residualRms: residNorm / Math.sqrt(V),
    realisationAnchor,
    realisationCertified,
    computeMs: tEnd - t0,
    stageTimingsMs: options.collectTimings
      ? {
          rawProposal: tProposal - t0,
          certificateProjection: tProjection - tProposal,
          displayPreparation: tEnd - tProjection,
        }
      : undefined,
    certifiedDelta,
  };
}
