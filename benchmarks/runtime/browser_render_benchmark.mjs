#!/usr/bin/env node

// Browser presentation benchmark for the built Rhinoform UI. The certified
// numerical pipeline is measured separately by browser_certified_benchmark.mjs.
// Here we measure the Ridge interaction-to-paint path and steady frame pacing.

import { execFileSync, spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import puppeteer from "../../demo/node_modules/puppeteer-core/lib/puppeteer/puppeteer-core.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "../..");
const DEMO = path.join(ROOT, "demo");
const OUT = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(HERE, "results/browser_chrome_render.json");
const CHROME = process.env.CHROME_PATH ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = Number(process.env.RHINOFORM_BENCH_PORT ?? 4517);
const WARMUP = Number(process.env.RHINOFORM_RENDER_WARMUP ?? 100);
const MEASURED = Number(process.env.RHINOFORM_RENDER_RUNS ?? 1000);

function quantile(sorted, q) {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos), w = pos - lo;
  return sorted[lo] * (1 - w) + sorted[hi] * w;
}

function stats(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const r = (x) => Number(x.toFixed(3));
  return {
    median_ms: r(quantile(sorted, 0.5)),
    p95_ms: r(quantile(sorted, 0.95)),
    p99_ms: r(quantile(sorted, 0.99)),
    mean_ms: r(mean),
    max_ms: r(sorted.at(-1)),
  };
}

async function waitForServer(url) {
  for (let i = 0; i < 100; i++) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`demo server did not become ready: ${url}`);
}

function processMemory(rootPid) {
  const text = execFileSync("ps", ["-axo", "pid=,ppid=,rss=,command="], { encoding: "utf8" });
  const rows = text.split("\n").map((line) => {
    const match = line.trim().match(/^(\d+)\s+(\d+)\s+(\d+)\s+(.*)$/);
    return match ? { pid: +match[1], ppid: +match[2], rssKb: +match[3], command: match[4] } : null;
  }).filter(Boolean);
  const descendants = new Set([rootPid]);
  let changed = true;
  while (changed) {
    changed = false;
    for (const row of rows) {
      if (descendants.has(row.ppid) && !descendants.has(row.pid)) {
        descendants.add(row.pid);
        changed = true;
      }
    }
  }
  const selected = rows.filter((row) => descendants.has(row.pid));
  const byRoleKb = { browser: 0, renderer: 0, gpu: 0, utility: 0, other: 0 };
  for (const row of selected) {
    let role = "other";
    if (row.pid === rootPid) role = "browser";
    else if (row.command.includes("--type=renderer")) role = "renderer";
    else if (row.command.includes("--type=gpu-process")) role = "gpu";
    else if (row.command.includes("--type=utility")) role = "utility";
    byRoleKb[role] += row.rssKb;
  }
  return {
    totalMb: selected.reduce((sum, row) => sum + row.rssKb, 0) / 1024,
    byRoleMb: Object.fromEntries(Object.entries(byRoleKb).map(([k, v]) => [k, v / 1024])),
  };
}

if (!fs.existsSync(CHROME)) throw new Error(`Chrome executable not found: ${CHROME}`);
const server = spawn("npm", ["run", "dev", "--", "--port", String(PORT), "--strictPort"], {
  cwd: DEMO,
  stdio: ["ignore", "pipe", "pipe"],
});
let browser;
try {
  const url = `http://127.0.0.1:${PORT}`;
  await waitForServer(url);
  browser = await puppeteer.launch({
    headless: "new",
    executablePath: CHROME,
    protocolTimeout: 600_000,
    args: ["--no-first-run", "--no-default-browser-check"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1460, height: 1200, deviceScaleFactor: 1 });
  await page.goto(url, { waitUntil: "networkidle2", timeout: 90_000 });
  await page.waitForSelector("canvas", { timeout: 90_000 });
  await page.waitForSelector(".slider input", { timeout: 90_000 });
  await page.addStyleTag({ content: "*,*::before,*::after{animation:none!important;transition:none!important}" });

  const cdp = await browser.target().createCDPSession();
  const systemInfo = await cdp.send("SystemInfo.getInfo");
  const rootPid = browser.process().pid;
  let peak = processMemory(rootPid);
  const sampler = setInterval(() => {
    try {
      const current = processMemory(rootPid);
      peak.totalMb = Math.max(peak.totalMb, current.totalMb);
      for (const role of Object.keys(peak.byRoleMb)) {
        peak.byRoleMb[role] = Math.max(peak.byRoleMb[role], current.byRoleMb[role]);
      }
    } catch {}
  }, 250);

  const measured = await page.evaluate(async ({ warmup, runs }) => {
    const raf = () => new Promise((resolve) => requestAnimationFrame(resolve));
    for (let i = 0; i < warmup; i++) await raf();
    const frameIntervals = [];
    let previous = await raf();
    for (let i = 0; i < runs; i++) {
      const current = await raf();
      frameIntervals.push(current - previous);
      previous = current;
    }

    const input = document.querySelector(".slider input");
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
    const ridgeInputToPaint = [];
    for (let i = 0; i < warmup + runs; i++) {
      const t0 = performance.now();
      setter.call(input, String(i % 2 === 0 ? 0.12 : -0.12));
      input.dispatchEvent(new Event("input", { bubbles: true }));
      await raf();
      await raf();
      if (i >= warmup) ridgeInputToPaint.push(performance.now() - t0);
    }
    return { frameIntervals, ridgeInputToPaint };
  }, { warmup: WARMUP, runs: MEASURED });
  clearInterval(sampler);

  const pageMetrics = await page.metrics();
  const gpu = systemInfo.gpu?.devices?.[0] ?? {};
  const result = {
    schema: "rhinoform_browser_render_benchmark_v1",
    generated_utc: new Date().toISOString(),
    scope: "Headless Chrome presentation frame pacing and Ridge slider input-to-second-animation-frame; certified compute measured separately",
    environment: {
      platform: process.platform,
      release: os.release(),
      arch: process.arch,
      cpu_model: os.cpus()[0]?.model ?? "unknown",
      logical_cpu_count: os.cpus().length,
      total_memory_mb: Number((os.totalmem() / 2 ** 20).toFixed(1)),
      node: process.version,
      chrome: await browser.version(),
      viewport_css_px: [1460, 1200],
      device_scale_factor: 1,
      gpu_device: gpu.deviceString ?? "unknown",
      gpu_vendor: gpu.vendorString ?? "unknown",
      gpu_driver: gpu.driverVendor ?? "unknown",
    },
    protocol: { warmup_runs: WARMUP, measured_runs: MEASURED, batch_size: 1 },
    rendering: {
      presentation_frame_interval: stats(measured.frameIntervals),
      ridge_input_to_second_animation_frame: stats(measured.ridgeInputToPaint),
      note: "The second-animation-frame boundary is a conservative next-paint proxy; it includes DOM/React scheduling, Ridge update and browser presentation, but excludes the 450-ms certified debounce and certified compute.",
    },
    memory: {
      scope: "Peak observed RSS sampled across the Chrome browser process tree at 250-ms intervals; roles may overlap in time and are not additive peak instants",
      chrome_process_tree_max_observed_rss_mb: Number(peak.totalMb.toFixed(1)),
      max_observed_rss_by_process_role_mb: Object.fromEntries(Object.entries(peak.byRoleMb).map(([k, v]) => [k, Number(v.toFixed(1))])),
      js_heap_used_after_benchmark_mb: Number((pageMetrics.JSHeapUsedSize / 2 ** 20).toFixed(1)),
      gpu_memory_note: "Chrome/macOS exposes no stable per-tab peak GPU-memory counter through this benchmark; no GPU-memory claim is made.",
    },
  };
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, `${JSON.stringify(result, null, 2)}\n`, "utf8");
  console.log(JSON.stringify(result, null, 2));
  console.error(`written ${OUT}`);
} finally {
  if (browser) await browser.close();
  server.kill("SIGTERM");
}
