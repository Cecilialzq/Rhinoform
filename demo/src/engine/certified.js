// CertifiedRBSRAdapter: resolves a certified RB-SR enhancement for the
// current effective controls, from one of three backends:
//
//   1. live      — the local research service (authoritative Python runtime,
//                  any control vector), when it is reachable and its model
//                  identity matches the bundle;
//   2. browser   — the in-browser port of the frozen certified pipeline
//                  (any control vector, fixed public mean source), verified
//                  against Python golden references;
//   3. precomputed — the frozen preset library (exact preset sliders only).
//
// All backends return the same result contract. When none can serve the
// request the caller stays on the Ridge anchor — that is the designed
// fallback, not an error state.

import { requestCertified } from "./bundle.js";
import { certifiedLocal } from "./certifiedLocal.js";

const PRESET_EPS = 1e-6;

export function findPreset(sliderValues, presets) {
  outer: for (const preset of presets) {
    for (let i = 0; i < 6; i++) {
      if (Math.abs((sliderValues[i] || 0) - preset.sliders[i]) > PRESET_EPS) continue outer;
    }
    return preset;
  }
  return null;
}

function fromPreset(preset) {
  return {
    available: true,
    backend: "precomputed",
    certified: preset.certified,
    status: preset.status,
    retention: preset.retention,
    iterations: preset.iterations,
    ridgeFoldCount: preset.ridge_fold_count,
    projectedFoldCount: preset.projected_fold_count,
    newVsRidgeFoldCount: preset.new_vs_ridge_fold_count,
    gateP50: preset.gate_p50,
    gateP95: preset.gate_p95,
    residualRms: preset.residual_rms,
    realisationAnchor: preset.realisation_anchor ?? null,
    realisationCertified: preset.realisation_certified ?? null,
    computeMs: null,
    certifiedDelta: new Float32Array(preset.certified_delta.flat()),
  };
}

function fromService(payload) {
  return {
    available: true,
    backend: "live",
    certified: payload.certified,
    status: payload.status,
    retention: payload.retention,
    iterations: payload.iterations,
    ridgeFoldCount: payload.ridge_fold_count,
    projectedFoldCount: payload.projected_fold_count,
    newVsRidgeFoldCount: payload.new_vs_ridge_fold_count,
    gateP50: payload.gate_p50,
    gateP95: payload.gate_p95,
    residualRms: payload.residual_rms,
    realisationAnchor: payload.realisation_anchor ?? null,
    realisationCertified: payload.realisation_certified ?? null,
    computeMs: payload.compute_ms,
    certifiedDelta: new Float32Array(payload.certified_delta.flat()),
  };
}

// Resolve a certified enhancement. `sliderValues` identify presets; `controls`
// are the effective (post-admission) raw controls sent to the live service.
export async function resolveCertified({ sliderValues, controls, ctx, timeoutMs = 4000 }) {
  if (ctx.service) {
    const payload = await requestCertified(controls, timeoutMs);
    if (payload.ok) return fromService(payload);
    // Service failure falls through to the in-browser pipeline (if present)
    // before resolving to the anchor.
    if (!ctx.certifiedBrowser) {
      return {
        available: false,
        backend: "live",
        fallbackReason: payload.timedOut ? "timeout" : "service_error",
        error: payload.error,
      };
    }
  }
  if (ctx.certifiedBrowser) {
    // Yield once so the "computing" state paints before the dense compute.
    await new Promise((r) => setTimeout(r, 0));
    try {
      return certifiedLocal(controls, ctx, ctx.certifiedBrowser);
    } catch (err) {
      return { available: false, backend: "browser", fallbackReason: "browser_error", error: String(err) };
    }
  }
  const preset = findPreset(sliderValues, ctx.presets);
  if (preset) return fromPreset(preset);
  return { available: false, backend: "precomputed", fallbackReason: "no_precomputed_match" };
}
