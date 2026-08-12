// Geometry and diagnostic utilities for the deployed Rhinoform runtime.
// The ROI lives in the model frame (x lateral, y vertical, z antero-posterior),
// in uncalibrated FaceScape model units.

// controls = sliders @ M, with the frozen (6 x 27) mapping from the bundle
// manifest — the identical matrix is used by the Python runtimes.
export function slidersToControls(sliderValues, matrix) {
  const ctrl = new Float32Array(27);
  for (let s = 0; s < matrix.length; s++) {
    const w = sliderValues[s];
    if (!w) continue;
    const row = matrix[s];
    for (let k = 0; k < 27; k++) ctrl[k] += w * row[k];
  }
  return ctrl;
}

// Source-adaptive bridge-smoothing slider row (27 entries). Mirrors
// rhinoform_runtime.bridge_hump_row exactly: straight line from the radix
// landmark to the tip landmark of the *bound* source; each dorsum landmark's
// vertical deviation from that line (at its projection ratio t along
// radix->tip) is the detected hump/scoop, and the row writes -dev to its y
// component so slider=+1 flattens the detected profile deviation.
export function humpRow(source, landmarks, roles) {
  const P = (j) => [
    source[landmarks[j] * 3],
    source[landmarks[j] * 3 + 1],
    source[landmarks[j] * 3 + 2],
  ];
  const rad = P(roles.radix);
  const tip = P(roles.tip);
  const ax = [tip[0] - rad[0], tip[1] - rad[1], tip[2] - rad[2]];
  const n2 = ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2];
  const row = new Array(27).fill(0);
  for (const j of roles.dorsum) {
    const p = P(j);
    const t = ((p[0] - rad[0]) * ax[0] + (p[1] - rad[1]) * ax[1] + (p[2] - rad[2]) * ax[2]) / n2;
    const lineY = rad[1] + t * (tip[1] - rad[1]);
    row[j * 3 + 1] = -(p[1] - lineY);
  }
  return row;
}

export function scaleControls(ctrl, s) {
  const out = new Float32Array(ctrl.length);
  for (let i = 0; i < ctrl.length; i++) out[i] = ctrl[i] * s;
  return out;
}

// Ridge anchor in the edit-relative display frame: the PURE learned field
//   delta(c) = smooth_pinned(taper * ((c / sigma_c) @ W_c))
// with the z-score already folded into the exported operator. The taper is
// the ROI boundary taper (0 on the rim, smoothstep to 1 at 5 units geodesic
// distance) so the ROI edge never follows an edit; every landmark sits at
// weight ~1. The rim-pinned smoothing diffuses the strain the taper
// concentrates inside its band (no wrinkle ring at the ROI boundary).
// Native amplitude, no display-frame handle enforcement.
export function ridgeAnchorDelta(ctrl, ctx) {
  const D = ctx.vertexDim; // 3 * nVertices
  const W = ctx.wCtrl;
  const delta = new Float32Array(D);
  for (let k = 0; k < 27; k++) {
    const c = ctrl[k];
    if (c === 0) continue;
    const base = k * D;
    for (let v = 0; v < D; v++) delta[v] += c * W[base + v];
  }
  applyTaper(delta, ctx.taper);
  const smoothed = displaySmooth(delta, ctx);
  const out = new Float32Array(D);
  out.set(smoothed);
  return out;
}

// Rim-pinned smoothing of a displayed (tapered) delta. Mirrors the Python
// runtime's smooth_field_pinned applied in ridge_anchor / certified_rbsr.
// Caches the vertex one-rings and the rim index list on the ctx object.
export function displaySmooth(delta, ctx) {
  const V = ctx.taper.length;
  if (!ctx._vertexRings) ctx._vertexRings = vertexOneRings(ctx.facesFlat, V);
  if (!ctx._rimIdx) {
    const rim = [];
    for (let v = 0; v < V; v++) if (ctx.taper[v] === 0) rim.push(v);
    ctx._rimIdx = Int32Array.from(rim);
  }
  return smoothFieldPinned(delta, ctx._vertexRings, ctx._rimIdx);
}

// Multiply a dense (V*3) displacement field by a per-vertex weight, in place.
export function applyTaper(delta, taper) {
  const V = delta.length / 3;
  for (let v = 0; v < V; v++) {
    const w = taper[v];
    if (w === 1) continue;
    delta[v * 3] *= w;
    delta[v * 3 + 1] *= w;
    delta[v * 3 + 2] *= w;
  }
  return delta;
}

// Sorted one-ring neighbour lists for every vertex. Mirrors
// rhinoform_runtime.vertex_adjacency.
export function vertexOneRings(facesFlat, nVertices) {
  const sets = Array.from({ length: nVertices }, () => new Set());
  for (let f = 0; f < facesFlat.length; f += 3) {
    const a = facesFlat[f], b = facesFlat[f + 1], c = facesFlat[f + 2];
    sets[a].add(b).add(c);
    sets[b].add(a).add(c);
    sets[c].add(a).add(b);
  }
  return sets.map((s) => Int32Array.from(Array.from(s).sort((x, y) => x - y)));
}

// k rounds of uniform Laplacian smoothing of a per-vertex (V*3) field:
// f <- f + lam * (one-ring mean - f). Mirrors rhinoform_runtime.smooth_field.
export const DISPLAY_SMOOTH_K = 20;
export const DISPLAY_SMOOTH_LAMBDA = 0.5;

export function smoothField(field, rings, k = DISPLAY_SMOOTH_K, lam = DISPLAY_SMOOTH_LAMBDA) {
  const V = rings.length;
  let f = Float64Array.from(field);
  let g = new Float64Array(V * 3);
  for (let it = 0; it < k; it++) {
    for (let v = 0; v < V; v++) {
      const ring = rings[v];
      let sx = 0, sy = 0, sz = 0;
      for (let j = 0; j < ring.length; j++) {
        const nb = ring[j] * 3;
        sx += f[nb];
        sy += f[nb + 1];
        sz += f[nb + 2];
      }
      const inv = 1 / ring.length;
      const b = v * 3;
      g[b] = f[b] + lam * (sx * inv - f[b]);
      g[b + 1] = f[b + 1] + lam * (sy * inv - f[b + 1]);
      g[b + 2] = f[b + 2] + lam * (sz * inv - f[b + 2]);
    }
    const tmp = f;
    f = g;
    g = tmp;
  }
  return f;
}

// Laplacian smoothing with the pinIdx vertices re-clamped to their input
// values after every round (Dirichlet boundary). Mirrors
// rhinoform_runtime.smooth_field_pinned.
export const DISPLAY_EDGE_SMOOTH_K = 12;

export function smoothFieldPinned(
  field,
  rings,
  pinIdx,
  k = DISPLAY_EDGE_SMOOTH_K,
  lam = DISPLAY_SMOOTH_LAMBDA
) {
  const V = rings.length;
  let f = Float64Array.from(field);
  let g = new Float64Array(V * 3);
  const pinned = new Float64Array(pinIdx.length * 3);
  for (let i = 0; i < pinIdx.length; i++) {
    const b = pinIdx[i] * 3;
    pinned[i * 3] = f[b];
    pinned[i * 3 + 1] = f[b + 1];
    pinned[i * 3 + 2] = f[b + 2];
  }
  for (let it = 0; it < k; it++) {
    for (let v = 0; v < V; v++) {
      const ring = rings[v];
      let sx = 0, sy = 0, sz = 0;
      for (let j = 0; j < ring.length; j++) {
        const nb = ring[j] * 3;
        sx += f[nb];
        sy += f[nb + 1];
        sz += f[nb + 2];
      }
      const inv = 1 / ring.length;
      const b = v * 3;
      g[b] = f[b] + lam * (sx * inv - f[b]);
      g[b + 1] = f[b + 1] + lam * (sy * inv - f[b + 1]);
      g[b + 2] = f[b + 2] + lam * (sz * inv - f[b + 2]);
    }
    const tmp = f;
    f = g;
    g = tmp;
    for (let i = 0; i < pinIdx.length; i++) {
      const b = pinIdx[i] * 3;
      f[b] = pinned[i * 3];
      f[b + 1] = pinned[i * 3 + 1];
      f[b + 2] = pinned[i * 3 + 2];
    }
  }
  return f;
}

// Sorted one-ring neighbour vertex ids of each landmark vertex (landmark
// itself excluded). Mirrors rhinoform_runtime.landmark_one_rings.
export function landmarkOneRings(facesFlat, landmarks) {
  return landmarks.map((lv) => {
    const ring = new Set();
    for (let f = 0; f < facesFlat.length; f += 3) {
      const a = facesFlat[f], b = facesFlat[f + 1], c = facesFlat[f + 2];
      if (a === lv || b === lv || c === lv) {
        ring.add(a);
        ring.add(b);
        ring.add(c);
      }
    }
    ring.delete(lv);
    return Array.from(ring).sort((x, y) => x - y);
  });
}

export function applyDelta(source, delta) {
  const out = new Float32Array(source.length);
  for (let i = 0; i < source.length; i++) out[i] = source[i] + delta[i];
  return out;
}

export function addDelta(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

export function displacementMag(delta) {
  const n = delta.length / 3;
  const perV = new Float32Array(n);
  let max = 1e-6;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const dx = delta[i * 3], dy = delta[i * 3 + 1], dz = delta[i * 3 + 2];
    const m = Math.sqrt(dx * dx + dy * dy + dz * dz);
    perV[i] = m;
    sum += m;
    if (m > max) max = m;
  }
  return { perV, max, mean: sum / n };
}

function faceNormal(V, a, b, c) {
  const ax = V[a * 3], ay = V[a * 3 + 1], az = V[a * 3 + 2];
  const bx = V[b * 3], by = V[b * 3 + 1], bz = V[b * 3 + 2];
  const cx = V[c * 3], cy = V[c * 3 + 1], cz = V[c * 3 + 2];
  const ux = bx - ax, uy = by - ay, uz = bz - az;
  const vx = cx - ax, vy = cy - ay, vz = cz - az;
  return [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx];
}

// % of faces whose orientation reverses between source and edited meshes
// (browser-side descriptive proxy of the paper's signed-fold rule).
export function normalFlipPct(source, edited, faces) {
  let flipped = 0;
  for (let f = 0; f < faces.length; f++) {
    const [a, b, c] = faces[f];
    const ns = faceNormal(source, a, b, c);
    const ne = faceNormal(edited, a, b, c);
    const dot = ns[0] * ne[0] + ns[1] * ne[1] + ns[2] * ne[2];
    if (dot < 0) flipped++;
  }
  return (100 * flipped) / faces.length;
}

// P95 relative edge-length strain + per-vertex max strain.
export function edgeStrain(source, edited, faces) {
  const n = source.length / 3;
  const perV = new Float32Array(n);
  const strains = [];
  const edge = (a, b) => {
    const s = Math.hypot(
      source[a * 3] - source[b * 3],
      source[a * 3 + 1] - source[b * 3 + 1],
      source[a * 3 + 2] - source[b * 3 + 2]
    );
    const e = Math.hypot(
      edited[a * 3] - edited[b * 3],
      edited[a * 3 + 1] - edited[b * 3 + 1],
      edited[a * 3 + 2] - edited[b * 3 + 2]
    );
    const st = s > 1e-6 ? Math.abs(e - s) / s : 0;
    strains.push(st);
    if (st > perV[a]) perV[a] = st;
    if (st > perV[b]) perV[b] = st;
  };
  for (let f = 0; f < faces.length; f++) {
    const [a, b, c] = faces[f];
    edge(a, b);
    edge(b, c);
    edge(c, a);
  }
  strains.sort((x, y) => x - y);
  const p95 = strains.length ? strains[Math.floor(strains.length * 0.95)] : 0;
  return { p95, perV };
}

// Scatter a ROI-local displacement field onto the full-head context mesh:
// ROI vertices receive their delta exactly; collar vertices receive the
// weighted delta of their anchor ROI vertex; every other vertex is locked.
export function deformHead(head, delta) {
  const out = new Float32Array(head.vertices);
  const { roiIndices, collar } = head;
  for (let i = 0; i < roiIndices.length; i++) {
    const hv = roiIndices[i];
    out[hv * 3] += delta[i * 3];
    out[hv * 3 + 1] += delta[i * 3 + 1];
    out[hv * 3 + 2] += delta[i * 3 + 2];
  }
  for (let j = 0; j < collar.v.length; j++) {
    const hv = collar.v[j];
    const a = collar.anchor[j];
    const w = collar.w[j];
    out[hv * 3] += w * delta[a * 3];
    out[hv * 3 + 1] += w * delta[a * 3 + 1];
    out[hv * 3 + 2] += w * delta[a * 3 + 2];
  }
  return out;
}

// chi = || c / sigma ||_2 in the frozen validation control frame. This is the
// input-applicability statistic, not a safety or confidence score.
export function controlChi(ctrl, perDimStd) {
  let acc = 0;
  for (let i = 0; i < 27; i++) {
    const z = ctrl[i] / (perDimStd[i] || 1e-6);
    acc += z * z;
  }
  return Math.sqrt(acc);
}
