#!/usr/bin/env node
/*
 * COMPUTE-ONLY per-edit latency microbenchmark for the Rhinoform runtime.
 *
 * This measures the algorithmic cost of one edit as the browser computes it:
 *   control(27) -> dense = control . W_ctrl -> scatter into head ROI ->
 *   collar blend -> recompute O(N) risk metrics (normal flips, edge-strain p95).
 * It EXCLUDES the WebGL/Three.js render cost, so it is a *lower bound* on true
 * per-frame latency. The genuine end-to-end browser frame time is measured
 * separately by demo/measure_browser_latency_playwright.js (run locally).
 *
 * Colab-runnable (Node only, no browser). Reads engine.json / head.json read-only.
 * Writes a single JSON report to --out. No fabrication: it times real arithmetic
 * on the frozen exported operator and mesh.
 *
 * Usage:
 *   node measure_browser_compute_latency.js \
 *     --engine ../demo/public/engine.json --head ../demo/public/head.json \
 *     --iters 1000 --out browser_compute_latency.json
 */
"use strict";
const fs = require("fs");

function arg(name, def) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
}

function findKey(obj, name, depth = 0) {
  if (obj == null || depth > 4) return undefined;
  if (typeof obj === "object" && !Array.isArray(obj)) {
    if (name in obj) return obj[name];
    for (const k of Object.keys(obj)) {
      const v = findKey(obj[k], name, depth + 1);
      if (v !== undefined) return v;
    }
  }
  return undefined;
}

function percentile(sorted, p) {
  if (sorted.length === 0) return null;
  const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
  return sorted[idx];
}

function main() {
  const enginePath = arg("engine", "../demo/public/engine.json");
  const headPath = arg("head", "../demo/public/head.json");
  const iters = parseInt(arg("iters", "1000"), 10);
  const outPath = arg("out", "browser_compute_latency.json");

  const engine = JSON.parse(fs.readFileSync(enginePath, "utf-8"));
  const head = JSON.parse(fs.readFileSync(headPath, "utf-8"));

  // ---- discover the linear operator ----
  let wctrl = findKey(engine, "w_ctrl");
  const ctrlStd = findKey(engine, "ctrl_std");
  let controlDim = findKey(engine, "control_dim") || findKey(engine, "n_control");
  let vertexDim = findKey(engine, "vertex_dim");
  if (wctrl === undefined) {
    console.error("Could not find w_ctrl in engine.json. Top-level keys:", Object.keys(engine));
    process.exit(2);
  }
  // normalise W to rows [controlDim][vertexDim]
  let W;
  if (Array.isArray(wctrl[0])) {
    W = wctrl;
    controlDim = controlDim || W.length;
    vertexDim = vertexDim || W[0].length;
  } else {
    // flat: length = controlDim * vertexDim
    controlDim = controlDim || 27;
    vertexDim = vertexDim || wctrl.length / controlDim;
    W = [];
    for (let i = 0; i < controlDim; i++) {
      W.push(wctrl.slice(i * vertexDim, (i + 1) * vertexDim));
    }
  }
  const nVerts = Math.floor(vertexDim / 3);

  // ---- discover head/ROI ----
  let vertices = findKey(head, "vertices");
  const roiIndices = findKey(head, "roi_indices");
  let roiFaces = findKey(head, "roi_faces");
  const collar = findKey(head, "collar");
  if (!vertices || !roiIndices || !roiFaces) {
    console.error("head.json missing vertices/roi_indices/roi_faces. Keys:", Object.keys(head));
    process.exit(2);
  }
  // Normalise flat arrays to nested triplets if needed.
  function toTriplets(arr, stride) {
    if (arr.length === 0 || Array.isArray(arr[0])) return arr; // already nested
    const out = [];
    for (let i = 0; i < arr.length; i += stride) out.push(arr.slice(i, i + stride));
    return out;
  }
  vertices = toTriplets(vertices, 3);          // [[x,y,z], ...]
  roiFaces = toTriplets(roiFaces, 3);          // [[a,b,c], ...]
  // ROI source positions (local order matches roi_indices)
  const roiSrc = roiIndices.map((gi) => vertices[gi]);
  // detect whether roi_faces are ROI-local or global indices
  let maxFace = 0;
  for (const f of roiFaces) for (const v of f) if (v > maxFace) maxFace = v;
  let faces = roiFaces;
  if (maxFace >= roiSrc.length) {
    const g2l = new Map();
    roiIndices.forEach((gi, li) => g2l.set(gi, li));
    faces = roiFaces.map((f) => f.map((v) => (g2l.has(v) ? g2l.get(v) : 0)));
  }
  // unique edges from faces
  const edgeSet = new Set();
  const edges = [];
  for (const f of faces) {
    for (let e = 0; e < 3; e++) {
      const a = f[e], b = f[(e + 1) % 3];
      const key = a < b ? `${a}_${b}` : `${b}_${a}`;
      if (!edgeSet.has(key)) { edgeSet.add(key); edges.push([a, b]); }
    }
  }
  // precompute source edge lengths and source face normals
  const e0 = edges.map(([a, b]) => {
    const pa = roiSrc[a], pb = roiSrc[b];
    return Math.hypot(pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]) || 1e-12;
  });
  function faceNormal(P, f) {
    const a = P[f[0]], b = P[f[1]], c = P[f[2]];
    const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
    const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
    return [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx];
  }
  const srcNormals = faces.map((f) => faceNormal(roiSrc, f));

  // collar weights (optional smoothing); if absent, no-op
  const collarV = collar && collar.v ? collar.v : null;
  const collarW = collar && collar.w ? collar.w : null;

  // ---- timed loop ----
  const times = [];
  const rng = () => (Math.random() * 2 - 1);
  // warmup
  for (let w = 0; w < Math.min(20, iters); w++) editOnce();
  for (let it = 0; it < iters; it++) {
    const t0 = process.hrtime.bigint();
    editOnce();
    const t1 = process.hrtime.bigint();
    times.push(Number(t1 - t0) / 1e6); // ms
  }

  function editOnce() {
    // 1. control vector (27), small random edit
    const ctrl = new Float64Array(controlDim);
    for (let i = 0; i < controlDim; i++) ctrl[i] = 0.3 * rng();
    // 2. dense = ctrl . W  -> length vertexDim
    const dense = new Float64Array(vertexDim);
    for (let i = 0; i < controlDim; i++) {
      const ci = ctrl[i]; if (ci === 0) continue;
      const Wi = W[i];
      for (let j = 0; j < vertexDim; j++) dense[j] += ci * Wi[j];
    }
    // 3. edited ROI positions
    const edited = new Array(nVerts);
    for (let v = 0; v < nVerts; v++) {
      edited[v] = [roiSrc[v][0] + dense[3 * v], roiSrc[v][1] + dense[3 * v + 1], roiSrc[v][2] + dense[3 * v + 2]];
    }
    // 3b. collar blend (light) if present
    if (collarV && collarW) {
      for (let k = 0; k < collarV.length; k++) {
        const idx = collarV[k]; const w = collarW[k];
        if (idx < nVerts) {
          edited[idx][0] = roiSrc[idx][0] + w * dense[3 * idx];
          edited[idx][1] = roiSrc[idx][1] + w * dense[3 * idx + 1];
          edited[idx][2] = roiSrc[idx][2] + w * dense[3 * idx + 2];
        }
      }
    }
    // 4a. normal-flip fraction
    let flips = 0;
    for (let fi = 0; fi < faces.length; fi++) {
      const n = faceNormal(edited, faces[fi]); const s = srcNormals[fi];
      const dot = n[0] * s[0] + n[1] * s[1] + n[2] * s[2];
      if (dot < 0) flips++;
    }
    const flipPct = (100 * flips) / faces.length;
    // 4b. edge-strain p95
    const strain = new Float64Array(edges.length);
    for (let ei = 0; ei < edges.length; ei++) {
      const [a, b] = edges[ei]; const pa = edited[a], pb = edited[b];
      const l = Math.hypot(pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]);
      strain[ei] = Math.abs(l - e0[ei]) / e0[ei];
    }
    const ss = Array.from(strain).sort((x, y) => x - y);
    void flipPct; void percentile(ss, 95); // consume so JIT can't elide
  }

  times.sort((a, b) => a - b);
  const report = {
    label: "COMPUTE-ONLY per-edit latency (excludes WebGL render). Lower bound on true frame time.",
    engine: enginePath, head: headPath,
    control_dim: controlDim, vertex_dim: vertexDim, n_roi_vertices: nVerts,
    n_faces: faces.length, n_edges: edges.length, iters: iters,
    median_ms: percentile(times, 50), p95_ms: percentile(times, 95),
    p99_ms: percentile(times, 99), min_ms: times[0], max_ms: times[times.length - 1],
    node_version: process.version,
    note: "True end-to-end browser latency (with GPU render) must be measured with "
        + "demo/measure_browser_latency_playwright.js on a machine with a browser.",
  };
  fs.writeFileSync(outPath, JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
}

main();
