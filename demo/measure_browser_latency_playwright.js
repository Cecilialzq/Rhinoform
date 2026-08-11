#!/usr/bin/env node
/*
 * TRUE end-to-end per-edit browser latency for the Rhinoform demo, measured with a
 * real browser (Playwright / Chromium). This is the figure Colab CANNOT produce
 * (no DOM/WebGL). Run it LOCALLY. It complements the compute-only Node microbench
 * (code/measure_browser_compute_latency.js).
 *
 * What it measures: for each simulated edit it sets a semantic slider, then times
 * from the input event to the next painted animation frame (requestAnimationFrame),
 * i.e. interaction -> rendered frame, including the Three.js/WebGL render. Reports
 * median / p95 / p99 over N edits.
 *
 * It does NOT read or modify any frozen result; it only drives the local demo UI.
 *
 * Prerequisites (see LOCAL_BROWSER_LATENCY_README.md):
 *   cd demo && npm install && npm run build      # produces dist/
 *   npm install -D playwright && npx playwright install chromium
 *
 * Usage (serve dist/ yourself, or let this script serve it):
 *   node measure_browser_latency_playwright.js --serve dist --iters 500 --out browser_latency.json
 *   # or against a running dev server:
 *   node measure_browser_latency_playwright.js --url http://127.0.0.1:5173 --iters 500
 */
"use strict";
const fs = require("fs");
const http = require("http");
const path = require("path");

function arg(name, def) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
}

const MIME = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml" };

function serveDir(dir) {
  const root = path.resolve(dir);
  const server = http.createServer((req, res) => {
    let p = decodeURIComponent(req.url.split("?")[0]);
    if (p === "/") p = "/index.html";
    const fp = path.join(root, p);
    if (!fp.startsWith(root) || !fs.existsSync(fp) || fs.statSync(fp).isDirectory()) {
      // SPA fallback
      const idx = path.join(root, "index.html");
      if (fs.existsSync(idx)) { res.writeHead(200, { "Content-Type": "text/html" }); return res.end(fs.readFileSync(idx)); }
      res.writeHead(404); return res.end("not found");
    }
    res.writeHead(200, { "Content-Type": MIME[path.extname(fp)] || "application/octet-stream" });
    res.end(fs.readFileSync(fp));
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve({ server, port: server.address().port })));
}

async function main() {
  let playwright;
  try {
    playwright = require("playwright");
  } catch (e) {
    console.error("Playwright not installed. Run: npm install -D playwright && npx playwright install chromium");
    process.exit(3);
  }
  const iters = parseInt(arg("iters", "500"), 10);
  const outPath = arg("out", "browser_latency.json");
  const serveDirArg = arg("serve", "");
  let url = arg("url", "");
  let served = null;
  if (!url) {
    const dir = serveDirArg || "dist";
    if (!fs.existsSync(dir)) { console.error(`No --url and serve dir '${dir}' not found. Build with: npm run build`); process.exit(3); }
    served = await serveDir(dir);
    url = `http://127.0.0.1:${served.port}/`;
    console.log(`[serve] ${dir} at ${url}`);
  }

  const browser = await playwright.chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  await page.goto(url, { waitUntil: "networkidle" });
  await page.waitForTimeout(1500); // let the 3D scene initialise

  // Measure interaction->frame latency by driving range sliders.
  const result = await page.evaluate(async (N) => {
    function nextFrame() { return new Promise((r) => requestAnimationFrame(() => r(performance.now()))); }
    const sliders = Array.from(document.querySelectorAll('input[type="range"]'));
    const samples = [];
    if (sliders.length === 0) {
      // Fallback: measure raw frame cadence so the run still yields a number.
      for (let i = 0; i < N; i++) { const a = await nextFrame(); const b = await nextFrame(); samples.push(b - a); }
      return { mode: "frame_cadence_no_sliders", samples, n_sliders: 0 };
    }
    for (let i = 0; i < N; i++) {
      const s = sliders[i % sliders.length];
      const min = parseFloat(s.min || "0"), max = parseFloat(s.max || "1");
      const val = min + Math.random() * (max - min);
      const t0 = performance.now();
      s.value = String(val);
      s.dispatchEvent(new Event("input", { bubbles: true }));
      s.dispatchEvent(new Event("change", { bubbles: true }));
      const painted = await nextFrame();
      samples.push(painted - t0);
    }
    return { mode: "slider_input_to_frame", samples, n_sliders: sliders.length };
  }, iters);

  await browser.close();
  if (served) served.server.close();

  const s = result.samples.slice().sort((a, b) => a - b);
  const pct = (p) => (s.length ? s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))] : null);
  const report = {
    label: "TRUE end-to-end browser per-edit latency (interaction -> painted frame, incl. WebGL render).",
    mode: result.mode, n_sliders: result.n_sliders, iters: iters,
    median_ms: pct(50), p95_ms: pct(95), p99_ms: pct(99), min_ms: s[0], max_ms: s[s.length - 1],
    url, note: result.mode === "frame_cadence_no_sliders"
      ? "No range sliders found; reported raw frame cadence. Adapt the selector to your UI controls for interaction latency."
      : "Measured from slider input event to the next animation frame.",
  };
  fs.writeFileSync(outPath, JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
}

main().catch((e) => { console.error(e); process.exit(1); });
