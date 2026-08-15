#!/usr/bin/env node

// Steady-state compute benchmark for the exact certified browser pipeline.
// This intentionally excludes React/Three.js rendering; rendering is a
// separate stage and must not be inferred from these numbers.

import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { certifiedLocal, parseCertifiedBundle } from "../../demo/src/engine/certifiedLocal.js";
import { ridgeAnchorDelta, slidersToControls } from "../../demo/src/engine/geometry.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "../..");
const BUNDLE = path.join(ROOT, "demo/public/bundle");
const WARMUP = Number(process.env.RHINOFORM_BENCH_WARMUP ?? 100);
const MEASURED = Number(process.env.RHINOFORM_BENCH_RUNS ?? 1000);
const OUT = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(HERE, "results/browser_node_certified.json");

const readJson = (name) => JSON.parse(fs.readFileSync(path.join(BUNDLE, name), "utf8"));
const sha256 = (name) => crypto.createHash("sha256").update(fs.readFileSync(path.join(BUNDLE, name))).digest("hex");

function quantile(sorted, q) {
  if (!sorted.length) return Number.NaN;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  const w = pos - lo;
  return sorted[lo] * (1 - w) + sorted[hi] * w;
}

function stats(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const round = (x) => Number(x.toFixed(3));
  return {
    median_ms: round(quantile(sorted, 0.5)),
    p95_ms: round(quantile(sorted, 0.95)),
    p99_ms: round(quantile(sorted, 0.99)),
    mean_ms: round(mean),
    max_ms: round(sorted.at(-1)),
  };
}

const loadStart = performance.now();
const manifest = readJson("manifest.json");
const ridge = readJson("ridge_browser.json");
const presets = readJson("certified_presets.json");
const rawCertified = readJson("certified_browser.json");
const cb = parseCertifiedBundle(rawCertified);
const ctx = {
  vertexDim: ridge.n_vertices * 3,
  wCtrl: Float32Array.from(ridge.w_ctrl),
  landmarks: ridge.landmarks,
  taper: Float64Array.from(ridge.display_taper.weights),
  facesFlat: Int32Array.from(ridge.faces.flat()),
  source: new Float32Array(ridge.source.flat()),
};
const loadAndParseMs = performance.now() - loadStart;

const panel = [];
for (const preset of presets.presets) {
  for (const scale of [1.0, 0.5, 1.5]) {
    const sliders = preset.sliders.map((x) => x * scale);
    panel.push({
      key: `${preset.key}@${scale}`,
      controls: slidersToControls(sliders, manifest.sliders.matrix_6x27),
    });
  }
}

// The first zero-edit invocation fills the frozen display-bias cache. It is
// deliberately outside both warmup and measurement.
certifiedLocal(new Float64Array(27), ctx, cb);
for (let i = 0; i < WARMUP; i++) {
  const controls = panel[i % panel.length].controls;
  ridgeAnchorDelta(controls, ctx);
  certifiedLocal(controls, ctx, cb);
}

const memoryBaseline = process.memoryUsage();
let peakRss = memoryBaseline.rss;
let peakHeapUsed = memoryBaseline.heapUsed;
const ridgeMs = [];
const proposalMs = [];
const projectionMs = [];
const displayMs = [];
const e2eMs = [];
const iterations = [];
const retentions = [];
const statuses = {};
let certifiedCount = 0;
let maxNewVsRidgeFolds = 0;

for (let i = 0; i < MEASURED; i++) {
  const controls = panel[i % panel.length].controls;
  let t = performance.now();
  ridgeAnchorDelta(controls, ctx);
  ridgeMs.push(performance.now() - t);

  const result = certifiedLocal(controls, ctx, cb, { collectTimings: true });
  e2eMs.push(result.computeMs);
  proposalMs.push(result.stageTimingsMs.rawProposal);
  projectionMs.push(result.stageTimingsMs.certificateProjection);
  displayMs.push(result.stageTimingsMs.displayPreparation);
  iterations.push(result.iterations);
  retentions.push(result.retention);
  certifiedCount += Number(result.certified);
  statuses[result.status] = (statuses[result.status] ?? 0) + 1;
  maxNewVsRidgeFolds = Math.max(maxNewVsRidgeFolds, result.newVsRidgeFoldCount);

  const mem = process.memoryUsage();
  peakRss = Math.max(peakRss, mem.rss);
  peakHeapUsed = Math.max(peakHeapUsed, mem.heapUsed);
}

const iterSorted = [...iterations].sort((a, b) => a - b);
const retentionMean = retentions.reduce((a, b) => a + b, 0) / retentions.length;
const result = {
  schema: "rhinoform_browser_compute_benchmark_v1",
  generated_utc: new Date().toISOString(),
  scope: "Node/V8 execution of the exact certified browser compute pipeline; rendering excluded",
  locked_identities: manifest.locked_identities,
  artifact_sha256: {
    manifest_json: sha256("manifest.json"),
    ridge_browser_json: sha256("ridge_browser.json"),
    certified_browser_json: sha256("certified_browser.json"),
    certified_presets_json: sha256("certified_presets.json"),
  },
  environment: {
    platform: process.platform,
    release: os.release(),
    arch: process.arch,
    cpu_model: os.cpus()[0]?.model ?? "unknown",
    logical_cpu_count: os.cpus().length,
    total_memory_mb: Number((os.totalmem() / 2 ** 20).toFixed(1)),
    node: process.version,
    v8: process.versions.v8,
    batch_size: 1,
  },
  protocol: {
    warmup_runs: WARMUP,
    measured_runs: MEASURED,
    request_panel: "11 frozen presets x {1.0, 0.5, 1.5} slider amplitude, round-robin",
    panel_size: panel.length,
  },
  cold_start: { bundle_load_and_parse_ms: Number(loadAndParseMs.toFixed(3)) },
  steady_state_compute: {
    ridge_anchor: stats(ridgeMs),
    raw_rbsr_proposal: stats(proposalMs),
    certificate_projection: stats(projectionMs),
    display_preparation: stats(displayMs),
    certified_rbsr_end_to_end: stats(e2eMs),
  },
  certificate: {
    success_rate: certifiedCount / MEASURED,
    status_counts: statuses,
    iterations: {
      mean: Number((iterations.reduce((a, b) => a + b, 0) / iterations.length).toFixed(2)),
      p95: quantile(iterSorted, 0.95),
      max: Math.max(...iterations),
    },
    retention: {
      mean: Number(retentionMean.toFixed(4)),
      min: Number(Math.min(...retentions).toFixed(4)),
    },
    max_new_vs_ridge_fold_count: maxNewVsRidgeFolds,
  },
  memory: {
    scope: "Node process memory sampled after each measured request; not browser-tab, renderer-process, or GPU peak memory",
    baseline_rss_mb: Number((memoryBaseline.rss / 2 ** 20).toFixed(1)),
    max_observed_rss_mb: Number((peakRss / 2 ** 20).toFixed(1)),
    baseline_heap_used_mb: Number((memoryBaseline.heapUsed / 2 ** 20).toFixed(1)),
    max_observed_heap_used_mb: Number((peakHeapUsed / 2 ** 20).toFixed(1)),
  },
};

fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, `${JSON.stringify(result, null, 2)}\n`, "utf8");
console.log(JSON.stringify(result, null, 2));
console.error(`written ${OUT}`);
