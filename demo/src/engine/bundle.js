// Frozen deployment bundle loader + local research-service probe.
//
// The bundle (public/bundle/) is the single source of truth for model
// identity, the browser Ridge operator, the OOD policy, the slider mapping
// and the precomputed certified preset library. The UI never hard-codes
// model parameters.

const SERVICE_URL = "http://127.0.0.1:8321";
const SERVICE_PROBE_TIMEOUT_MS = 1200;

async function grabJson(url, progress) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url} -> HTTP ${r.status}`);
  if (!progress || !r.body) return r.json();
  // Stream the body so the loading screen can show real byte progress.
  const total = Number(r.headers.get("content-length")) || 0;
  progress.addTotal(url, total);
  const reader = r.body.getReader();
  const chunks = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    progress.addLoaded(url, value.length);
  }
  return JSON.parse(await new Blob(chunks).text());
}

function makeProgress(onProgress) {
  if (!onProgress) return null;
  const totals = new Map();
  const loadeds = new Map();
  const emit = () => {
    let loaded = 0;
    let total = 0;
    for (const v of loadeds.values()) loaded += v;
    for (const v of totals.values()) total += v;
    onProgress({ loaded, total });
  };
  return {
    addTotal(url, bytes) {
      totals.set(url, bytes);
      emit();
    },
    addLoaded(url, bytes) {
      loadeds.set(url, (loadeds.get(url) || 0) + bytes);
      emit();
    },
  };
}

async function probeService() {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), SERVICE_PROBE_TIMEOUT_MS);
    const r = await fetch(`${SERVICE_URL}/health`, { signal: controller.signal });
    clearTimeout(timer);
    if (!r.ok) return null;
    const health = await r.json();
    return health && health.ok ? health : null;
  } catch {
    return null;
  }
}

export async function loadBundle(onProgress) {
  const progress = makeProgress(onProgress);
  const [manifest, ridge, presets, certifiedRaw, service] = await Promise.all([
    grabJson("/bundle/manifest.json", progress),
    grabJson("/bundle/ridge_browser.json", progress),
    grabJson("/bundle/certified_presets.json", progress),
    grabJson("/bundle/certified_browser.json", progress).catch(() => null),
    probeService(),
  ]);

  // Fail closed: certified presets must carry the same locked identities as
  // the manifest, otherwise only the Ridge anchor is offered.
  const identityMatch =
    JSON.stringify(presets.locked_identities) === JSON.stringify(manifest.locked_identities);
  const serviceIdentityMatch =
    !service || JSON.stringify(service.identity) === JSON.stringify(manifest.locked_identities);

  // Full-head display context: a training-mean head (population statistic)
  // shipped with the public bundle. No individual scan is redistributed.
  let head = null;
  try {
    const raw = await grabJson("/bundle/head_mean.json", progress);
    head = {
      vertices: new Float32Array(raw.vertices),
      facesFlat: Int32Array.from(raw.faces),
      roiIndices: Int32Array.from(raw.roi_indices),
      collar: {
        v: Int32Array.from(raw.collar.v),
        anchor: Int32Array.from(raw.collar.anchor),
        w: Float32Array.from(raw.collar.w),
      },
      meta: raw.meta,
    };
  } catch {
    head = null; // ROI-only display is the designed fallback
  }

  // In-browser certified pipeline: only enabled when its locked identities
  // match the manifest (fail closed to the Ridge anchor otherwise).
  let certifiedBrowser = null;
  if (
    certifiedRaw &&
    JSON.stringify(certifiedRaw.locked_identities) === JSON.stringify(manifest.locked_identities)
  ) {
    const { parseCertifiedBundle } = await import("./certifiedLocal.js");
    certifiedBrowser = parseCertifiedBundle(certifiedRaw);
  }

  const source = new Float32Array(ridge.source.flat());
  return {
    manifest,
    head,
    certifiedBrowser,
    presets: identityMatch ? presets.presets : [],
    presetsIdentityMatch: identityMatch,
    service: serviceIdentityMatch ? service : null,
    serviceIdentityMismatch: Boolean(service) && !serviceIdentityMatch,
    serviceUrl: SERVICE_URL,
    source,
    faces: ridge.faces,
    facesFlat: Int32Array.from(ridge.faces.flat()),
    landmarks: ridge.landmarks,
    subunits: ridge.subunits,
    vertexDim: ridge.n_vertices * 3,
    nVertices: ridge.n_vertices,
    wCtrl: Float32Array.from(ridge.w_ctrl),
    taper: Float64Array.from(ridge.display_taper.weights),
    sliderMatrix: manifest.sliders.matrix_6x27,
    sliderKeys: manifest.sliders.keys,
    sliderRoles: manifest.sliders.roles,
    ood: manifest.ood,
    identities: manifest.locked_identities,
    sourceNote: ridge.source_note,
  };
}

export async function requestCertified(controls, timeoutMs = 4000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const r = await fetch(`${SERVICE_URL}/certify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ controls: Array.from(controls) }),
      signal: controller.signal,
    });
    clearTimeout(timer);
    if (!r.ok) return { ok: false, error: `HTTP ${r.status}` };
    return await r.json();
  } catch (err) {
    clearTimeout(timer);
    const timedOut = err && err.name === "AbortError";
    return { ok: false, error: timedOut ? "timeout" : String(err), timedOut };
  }
}
