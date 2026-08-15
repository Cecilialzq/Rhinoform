import React, { useEffect, useMemo, useRef, useState } from "react";
import Studio3D from "./Studio3D.jsx";
import {
  slidersToControls,
  humpRow,
  ridgeAnchorDelta,
  applyDelta,
  addDelta,
  deformHead,
  displacementMag,
  normalFlipPct,
  edgeStrain,
} from "../engine/geometry.js";
import { loadBundle } from "../engine/bundle.js";
import { admitControls } from "../engine/admission.js";
import { resolveCertified } from "../engine/certified.js";
import { parseMesh, loadUploadSupport, classifyUpload, rebindSource } from "../engine/uploadSource.js";
import { DEMO_EDITS } from "../engine/demoEdits.js";
import { buildSessionReport, downloadJson } from "../engine/sessionReport.js";

const SLIDERS = [
  { key: "tipRotation", label: "Tip rotation" },
  { key: "tipProjection", label: "Tip projection" },
  { key: "dorsumHeight", label: "Dorsum height" },
  { key: "alarWidth", label: "Alar width" },
  { key: "bridgeSmoothing", label: "Bridge smoothing" },
  { key: "tipWidth", label: "Tip width" },
];
const ZERO = Object.fromEntries(SLIDERS.map((s) => [s.key, 0]));

const DEBOUNCE_MS = 450;
const SERVICE_TIMEOUT_MS = 4000;
const CONSTRAINED_RETENTION = 0.98;

// Product state machine. Exactly one state is active at a time.
const STATES = {
  idle: {
    label: "Idle",
    tone: "quiet",
    text: "Move a control to explore an edit. The Ridge anchor responds instantly.",
  },
  editing: {
    label: "Editing · Anchor preview",
    tone: "anchor",
    text: "Showing the instant Ridge anchor for the current edit.",
  },
  computing: {
    label: "Computing certified enhancement",
    tone: "anchor",
    text: "The anchor stays interactive while Certified RB-SR runs in the background.",
  },
  certified: {
    label: "Certified enhancement available",
    tone: "good",
    text: "The learned enhancement passed the anchor-relative fold-subset certificate and replaced the anchor preview.",
  },
  constrained: {
    label: "Enhancement constrained",
    tone: "caution",
    text: "The certificate required attenuating part of the learned residual before it could be shown.",
  },
  fallback: {
    label: "Anchor fallback",
    tone: "caution",
    text: "Certified enhancement was unavailable for this request, so the preview stays on the deterministic Ridge anchor.",
  },
  outside: {
    label: "Request outside evaluated range",
    tone: "stop",
    text: "This edit exceeds the control range the system was evaluated on. The preview is withheld rather than extrapolated.",
  },
};

function magnitudeWord(v) {
  const a = Math.abs(v);
  if (a < 0.08) return "—";
  if (a < 0.4) return "subtle";
  if (a < 0.8) return "moderate";
  return "pronounced";
}

function shortHash(h) {
  return h ? `${h.slice(0, 8)}…` : "—";
}

export default function Studio() {
  const [ctx, setCtx] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [sliders, setSliders] = useState(ZERO);
  const [mode, setMode] = useState("consult"); // consult | research
  const [stage, setStage] = useState("after"); // after | before | delta | risk
  const [view, setView] = useState("head"); // head | roi (head only when context available)
  const [layer, setLayer] = useState("auto"); // auto | anchor | certified (research compare)
  const [presetKey, setPresetKey] = useState("");
  const [certState, setCertState] = useState({ phase: "none" }); // none|pending|done|failed
  const [plans, setPlans] = useState([]);
  const [customSource, setCustomSource] = useState(null); // {name, kind} | null
  const [uploadError, setUploadError] = useState(null);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [loadProgress, setLoadProgress] = useState(null); // {loaded, total} bytes
  const requestId = useRef(0);
  const debounceTimer = useRef(0);
  const originalCtx = useRef(null);
  const uploadSupport = useRef(null);
  const fileInput = useRef(null);

  const beginLoad = () => {
    setLoadError(null);
    setLoadProgress(null);
    loadBundle(setLoadProgress)
      .then((loaded) => {
        originalCtx.current = loaded;
        setCtx(loaded);
        // Open on a certified preset so the certified RB-SR channel is visible
        // immediately (especially on the public build, where free slider edits
        // only show the Ridge anchor).
        const initial = loaded.presets.find((p) => p.key === "combined_refinement");
        if (initial) {
          setPresetKey(initial.key);
          setSliders(Object.fromEntries(SLIDERS.map((s, i) => [s.key, initial.sliders[i]])));
        }
      })
      .catch((err) => setLoadError(String(err && err.message ? err.message : err)));
  };

  useEffect(beginLoad, []);

  const sliderValues = useMemo(() => SLIDERS.map((s) => sliders[s.key]), [sliders]);
  const dirty = useMemo(() => sliderValues.some((v) => Math.abs(v) > 0.001), [sliderValues]);

  // ---------------------------------------------------------------- anchor
  // Effective slider matrix: the bridge-smoothing row is source-adaptive
  // (hump removal computed on the currently bound source, gain-calibrated
  // against the learned operator), so it is recomputed here whenever the
  // source changes (upload / restore); the remaining rows come from the
  // frozen manifest matrix (pure learned field, native amplitude).
  const sliderMatrix = useMemo(() => {
    if (!ctx) return null;
    const m = ctx.sliderMatrix.map((row) => row.slice());
    const bs = ctx.sliderKeys.indexOf("bridgeSmoothing");
    if (bs >= 0) m[bs] = humpRow(ctx.source, ctx.landmarks, ctx.sliderRoles);
    return m;
  }, [ctx]);

  const admission = useMemo(() => {
    if (!ctx) return null;
    const requested = slidersToControls(sliderValues, sliderMatrix);
    return { requested, ...admitControls(requested, ctx.ood) };
  }, [ctx, sliderMatrix, sliderValues]);

  const anchor = useMemo(() => {
    if (!ctx || !admission || admission.withheld) return null;
    const delta = ridgeAnchorDelta(admission.effectiveCtrl, ctx);
    const edited = applyDelta(ctx.source, delta);
    const strain = edgeStrain(ctx.source, edited, ctx.faces);
    const disp = displacementMag(delta);
    return {
      delta,
      edited,
      flipPct: normalFlipPct(ctx.source, edited, ctx.faces),
      strainP95: strain.p95,
      strainPerV: strain.perV,
      mag: disp.perV,
      magMax: disp.max,
    };
  }, [ctx, admission]);

  // ------------------------------------------------------------- certified
  useEffect(() => {
    if (!ctx || !admission) return;
    requestId.current += 1;
    const rid = requestId.current;
    clearTimeout(debounceTimer.current);

    if (admission.withheld || !dirty) {
      setCertState({ phase: "none" });
      return;
    }
    // Live service and the in-browser pipeline certify any effective
    // controls; the preset library only matches exact preset sliders.
    setCertState({ phase: ctx.service || ctx.certifiedBrowser ? "pending" : "none" });
    debounceTimer.current = setTimeout(async () => {
      const result = await resolveCertified({
        sliderValues,
        controls: admission.effectiveCtrl,
        ctx,
        timeoutMs: SERVICE_TIMEOUT_MS,
      });
      if (rid !== requestId.current) return; // stale
      if (result.available) setCertState({ phase: "done", result });
      else if (result.fallbackReason === "no_precomputed_match") setCertState({ phase: "none" });
      else setCertState({ phase: "failed", reason: result.fallbackReason });
    }, DEBOUNCE_MS);
    return () => clearTimeout(debounceTimer.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ctx, admission, dirty]);

  const certified = certState.phase === "done" ? certState.result : null;

  const certifiedMesh = useMemo(() => {
    if (!ctx || !certified || !anchor) return null;
    // Live backend returns the certified delta already in the display frame.
    // The precomputed library stores the same frame; but its anchor is the
    // preset's own anchor, which equals ours because presets are matched on
    // exact slider values (scale=1 admission for all presets).
    const delta = certified.certifiedDelta;
    const edited = applyDelta(ctx.source, delta);
    const strain = edgeStrain(ctx.source, edited, ctx.faces);
    const disp = displacementMag(delta);
    return {
      delta,
      edited,
      flipPct: normalFlipPct(ctx.source, edited, ctx.faces),
      strainP95: strain.p95,
      strainPerV: strain.perV,
      mag: disp.perV,
      magMax: disp.max,
    };
  }, [ctx, certified, anchor]);

  // ---------------------------------------------------------------- state
  const runtimeState = useMemo(() => {
    if (!admission) return "idle";
    if (admission.withheld) return "outside";
    if (!dirty) return "idle";
    if (certState.phase === "pending") return "computing";
    if (certState.phase === "failed") return "fallback";
    if (certified) {
      const uniform = certified.status.includes("uniform");
      if (uniform || certified.retention < CONSTRAINED_RETENTION) return "constrained";
      return "certified";
    }
    return "editing";
  }, [admission, dirty, certState, certified]);

  const showCertified =
    Boolean(certifiedMesh) &&
    (layer === "certified" || (layer === "auto" && (runtimeState === "certified" || runtimeState === "constrained")));
  const active = showCertified ? certifiedMesh : anchor;

  // Full-head display context (training-mean head from the public bundle). The
  // ROI-local displacement of the active layer is scattered onto the head
  // with collar blending; everything outside ROI+collar stays locked.
  const useHead = Boolean(ctx && ctx.head) && view === "head";
  const headEdited = useMemo(() => {
    if (!useHead) return null;
    if (!active || (admission && admission.withheld)) return ctx.head.vertices;
    return deformHead(ctx.head, active.delta);
  }, [useHead, ctx, active, admission]);

  // ------------------------------------------------------------------- UI
  if (!ctx) {
    const loadedMb = loadProgress ? (loadProgress.loaded / 1e6).toFixed(1) : null;
    const pct =
      loadProgress && loadProgress.total > 0
        ? Math.min(100, Math.round((loadProgress.loaded / loadProgress.total) * 100))
        : null;
    return (
      <div className="studio-shell studio-shell--loading">
        {loadError ? (
          <div className="studio-load-error" role="alert">
            <span>Could not load the deployment bundle.</span>
            <p>{loadError}</p>
            <button className="studio-btn primary" onClick={beginLoad}>
              Retry download
            </button>
            <p className="hint">
              The bundle is served with 24-hour caching; a retry resumes from the
              browser cache where possible.
            </p>
          </div>
        ) : (
          <div className="studio-loading" role="status" aria-live="polite">
            <span>Loading the frozen deployment bundle…</span>
            <div
              className="load-progress"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={pct == null ? undefined : pct}
              aria-label="Deployment bundle download"
            >
              <div
                className="load-progress__fill"
                style={{ width: pct == null ? "8%" : `${Math.max(4, pct)}%` }}
              />
            </div>
            <p>
              {loadedMb ? `${loadedMb} MB fetched${pct != null ? ` · ${pct}%` : ""}` : "Connecting…"}
              {" · "}core bundle ≈ 8 MB on first visit, then cached for 24 h (the 7 MB
              upload-support pack is fetched only when a custom mesh is bound).
            </p>
          </div>
        )}
      </div>
    );
  }

  const set = (k, v) => {
    setSliders((s) => ({ ...s, [k]: v }));
    setPresetKey("");
  };

  const applyPreset = (key) => {
    setPresetKey(key);
    if (!key) return;
    const demo = DEMO_EDITS.find((p) => p.key === key);
    const preset = demo || ctx.presets.find((p) => p.key === key);
    if (!preset) return;
    // Boundary demonstrations live in the research slider range (±2) so the
    // admission state machine (scale-back / withholding) can be shown live.
    if (demo && mode !== "research") setMode("research");
    setSliders(Object.fromEntries(SLIDERS.map((s, i) => [s.key, preset.sliders[i]])));
  };

  const adoptLimitedEdit = () => {
    // Single source of truth: write the admitted (scaled) request back into
    // the sliders so requested == effective afterwards. A 0.5% margin keeps
    // the rounded slider values strictly inside the evaluated range.
    if (!admission || admission.scale >= 1) return;
    const s = admission.scale * 0.995;
    setSliders((prev) =>
      Object.fromEntries(SLIDERS.map(({ key }) => [key, Number((prev[key] * s).toFixed(3))]))
    );
    setPresetKey("");
  };

  const reset = () => {
    setSliders(ZERO);
    setPresetKey("");
  };

  // ------------------------------------------------- custom source (OBJ)
  const handleUpload = async (file) => {
    if (!file) return;
    setUploadBusy(true);
    setUploadError(null);
    try {
      const base = originalCtx.current;
      const parsed = parseMesh(await file.arrayBuffer(), file.name);
      const { kind, roi } = classifyUpload(parsed, base.source, base.head);
      if (!uploadSupport.current) {
        uploadSupport.current = await loadUploadSupport(base.manifest);
      }
      const cb = base.certifiedBrowser
        ? rebindSource(base.certifiedBrowser, uploadSupport.current, roi)
        : null;
      const head =
        kind === "head" && base.head
          ? { ...base.head, vertices: new Float32Array(parsed.vertices), meta: { custom: true } }
          : null;
      setCtx({
        ...base,
        source: new Float32Array(roi),
        certifiedBrowser: cb,
        head,
        presets: [],            // preset library is bound to the mean source
        service: null,          // the local service is bound to its own source
        sourceNote: `uploaded source: ${file.name} (${kind === "head" ? "full head" : "nasal ROI"})`,
      });
      setCustomSource({ name: file.name, kind });
      setView(head ? "head" : "roi");
      setSliders(ZERO);
      setPresetKey("");
    } catch (err) {
      setUploadError(String(err && err.message ? err.message : err));
    } finally {
      setUploadBusy(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  };

  const restoreMeanSource = () => {
    setCtx(originalCtx.current);
    setCustomSource(null);
    setUploadError(null);
    setView(originalCtx.current.head ? "head" : "roi");
    setSliders(ZERO);
    setPresetKey("");
  };

  const snapshot = () => ({
    sliders: { ...sliders },
    requested_controls: Array.from(admission.requested),
    effective_controls: admission.effectiveCtrl ? Array.from(admission.effectiveCtrl) : null,
    admission: {
      chi: Number(admission.chi.toFixed(4)),
      band: admission.band,
      scale: Number(admission.scale.toFixed(4)),
      within_threshold: ctx.ood.policy.within_evaluated_range,
    },
    runtime_state: STATES[runtimeState].label,
    anchor_metrics: anchor
      ? {
          normal_flip_pct: Number(anchor.flipPct.toFixed(4)),
          edge_strain_p95: Number(anchor.strainP95.toFixed(4)),
        }
      : null,
    certificate: certified
      ? {
          backend: certified.backend,
          certified: certified.certified,
          status: certified.status,
          retention: certified.retention,
          projection_iterations: certified.iterations,
          ridge_fold_count: certified.ridgeFoldCount,
          certified_fold_count: certified.projectedFoldCount,
          new_folds_vs_anchor: certified.newVsRidgeFoldCount,
          compute_ms: certified.computeMs,
        }
      : null,
  });

  const savePlan = () => {
    const name = `Plan ${String.fromCharCode(65 + plans.length)}`;
    setPlans((p) => [...p, { name, state: runtimeState, ...snapshot() }]);
  };

  const exportReport = () => {
    const payload = buildSessionReport({
      identities: ctx.identities,
      sourceKey: ctx.service ? ctx.service.source_key : "mean_neutral_roi_of_training_identities_v1",
      certifiedBackend: ctx.service
        ? "live_local_service"
        : ctx.certifiedBrowser
        ? "in_browser_frozen_pipeline"
        : "precomputed_presets",
      session: snapshot(),
      plans,
    });
    downloadJson(payload, "rhinoform-session.json");
  };

  const st = STATES[runtimeState];
  const chiPct = Math.min(100, (admission.chi / ctx.ood.policy.near_boundary) * 100);
  const chiColor =
    admission.band === "in" ? "var(--ok)" : admission.band === "near" ? "var(--caution)" : "var(--risk)";
  const sliderRange = mode === "research" ? 2 : 1;

  return (
    <div className="studio-shell">
      {/* LEFT — request */}
      <div className="studio-col">
        <div className="mode-switch" role="group" aria-label="Interface mode">
          <button
            className={mode === "consult" ? "on" : ""}
            aria-pressed={mode === "consult"}
            onClick={() => setMode("consult")}
          >
            Consultation
          </button>
          <button
            className={mode === "research" ? "on" : ""}
            aria-pressed={mode === "research"}
            onClick={() => setMode("research")}
          >
            Research
          </button>
        </div>

        <div className="panel-block">
          <h4>Preset edits</h4>
          <select
            className="preset-select"
            aria-label="Preset edit"
            value={presetKey}
            onChange={(e) => applyPreset(e.target.value)}
          >
            <option value="">Custom edit (sliders)</option>
            {ctx.presets.length > 0 && (
              <optgroup label="Certified presets">
                {ctx.presets.map((p) => (
                  <option key={p.key} value={p.key}>
                    {p.label}
                  </option>
                ))}
              </optgroup>
            )}
            <optgroup label="Boundary demonstrations (research range)">
              {DEMO_EDITS.map((p) => (
                <option key={p.key} value={p.key}>
                  {p.label}
                </option>
              ))}
            </optgroup>
          </select>
          <p className="hint">
            {ctx.service
              ? "Live mode: any edit is certified on demand by the local runtime."
              : ctx.certifiedBrowser
              ? "Any edit is certified in the browser by the frozen pipeline; presets are curated starting points."
              : "Certified enhancements are precomputed for these presets; free slider edits show the exact Ridge anchor."}
          </p>
        </div>

        <div className="panel-block">
          <h4>Source model</h4>
          <p className="hint">
            {customSource
              ? `Active: ${customSource.name} (${customSource.kind === "head" ? "full head" : "nasal ROI"})`
              : "Active: population-mean example (training-identity statistic)."}
          </p>
          <input
            ref={fileInput}
            type="file"
            name="source-mesh"
            aria-label="Source mesh file (OBJ or PLY, registered to the frozen correspondence)"
            accept=".obj,.ply"
            style={{ display: "none" }}
            onChange={(e) => handleUpload(e.target.files && e.target.files[0])}
          />
          <button
            className="studio-btn"
            disabled={uploadBusy}
            onClick={() => fileInput.current && fileInput.current.click()}
          >
            {uploadBusy ? "Binding source…" : "Upload source (OBJ)"}
          </button>
          {customSource && (
            <button className="studio-btn" onClick={restoreMeanSource}>
              Restore mean example
            </button>
          )}
          {uploadError && <p className="hint hint-error">Upload rejected: {uploadError}</p>}
          <p className="hint">
            Accepts OBJ (or PLY) meshes registered to the frozen correspondence: nasal ROI
            (3,934 vertices) or full head (26,317 vertices), in uncalibrated model units — the
            FaceScape registered meshes ship as OBJ. Certified inference then
            runs on the uploaded identity; presets and the local service stay bound to the mean example.
          </p>
        </div>

        <div className="panel-block">
          <h4>Semantic controls</h4>
          {SLIDERS.map((s) => (
            <div className="slider" key={s.key}>
              <div className="row">
                <label className="name" htmlFor={`slider-${s.key}`}>
                  {s.label}
                </label>
                <span className="val">
                  {mode === "research" ? sliders[s.key].toFixed(2) : magnitudeWord(sliders[s.key])}
                </span>
              </div>
              <input
                id={`slider-${s.key}`}
                type="range"
                min={-sliderRange}
                max={sliderRange}
                step={0.01}
                value={sliders[s.key]}
                aria-valuetext={`${s.label}: ${sliders[s.key].toFixed(2)}`}
                onChange={(e) => set(s.key, Number(e.target.value))}
              />
            </div>
          ))}
          {mode === "research" && (
            <p className="hint">
              Research range is ±2 so the evaluated-range boundary can be demonstrated (see the two
              boundary-demonstration presets); the consultation range (±1) stays inside it.
            </p>
          )}
        </div>

        <button className="studio-btn" onClick={reset}>
          Reset edit
        </button>
      </div>

      {/* CENTER — stage */}
      <div className="studio-stage">
        {/* One compact control bar directly above the canvas: stage tabs,
            head/roi view switch and (research) result-layer switch, so the
            eye never leaves the model while switching. */}
        <div className="stage-bar">
          <div className="stage-tabs" role="group" aria-label="Stage display">
            {["after", "before", "delta", "risk"].map((t) => (
              <button
                key={t}
                className={stage === t ? "on" : ""}
                aria-pressed={stage === t}
                onClick={() => setStage(t)}
              >
                {t}
              </button>
            ))}
          </div>
          {ctx.head && (
            <span className="view-switch" role="group" aria-label="Mesh view">
              {["head", "roi"].map((v) => (
                <button
                  key={v}
                  className={view === v ? "on" : ""}
                  aria-pressed={view === v}
                  onClick={() => setView(v)}
                >
                  {v === "head" ? "full head" : "roi"}
                </button>
              ))}
            </span>
          )}
          {mode === "research" && (
            <span className="layer-switch stage-layer" role="group" aria-label="Result layer">
              {["auto", "anchor", "certified"].map((l) => (
                <button
                  key={l}
                  className={layer === l ? "on" : ""}
                  aria-pressed={layer === l}
                  disabled={l === "certified" && !certifiedMesh}
                  onClick={() => setLayer(l)}
                >
                  {l}
                </button>
              ))}
            </span>
          )}
        </div>
        {useHead ? (
          <Studio3D
            source={ctx.head.vertices}
            edited={headEdited}
            facesFlat={ctx.head.facesFlat}
            roiIndices={ctx.head.roiIndices}
            mode={stage}
            delta={active ? { mag: active.mag, max: active.magMax } : null}
            risk={active ? { strainPerV: active.strainPerV } : null}
          />
        ) : (
          <Studio3D
            source={ctx.source}
            edited={active && !admission.withheld ? active.edited : ctx.source}
            facesFlat={ctx.facesFlat}
            roiIndices={null}
            mode={stage}
            delta={active ? { mag: active.mag, max: active.magMax } : null}
            risk={active ? { strainPerV: active.strainPerV } : null}
          />
        )}
        <div className="stage-legend">
          {stage === "delta" && (
            <>
              <span>
                <i style={{ background: "#2b2a26" }} />
                no change
              </span>
              <span>
                <i style={{ background: "#d7a23a" }} />
                moderate
              </span>
              <span>
                <i style={{ background: "#c8502f" }} />
                largest displacement
              </span>
            </>
          )}
          {stage === "risk" && (
            <>
              <span>
                <i style={{ background: "#3c6f57" }} />
                low strain
              </span>
              <span>
                <i style={{ background: "#9c7a23" }} />
                elevated
              </span>
              <span>
                <i style={{ background: "#a8402b" }} />
                high strain
              </span>
            </>
          )}
          {(stage === "after" || stage === "before") && (
            <span className="legend-note">
              {showCertified ? "Showing: certified enhancement" : "Showing: Ridge anchor"}
              {stage === "before" ? " · displaying the unedited source" : ""}
            </span>
          )}
          <span className="legend-note stage-scope-note">
            {useHead
              ? "Training-mean head context · ROI + collar deform · no individual scan"
              : ctx.service
              ? `Live certified runtime · ${ctx.service.source_key}`
              : "Population-mean nasal region · no individual scan"}
          </span>
        </div>
      </div>

      {/* RIGHT — assurance readouts */}
      <div className="studio-col">
        <div className={`state ${st.tone}`}>
          <div className="label">
            <span className="dot" />
            {st.label}
          </div>
          <p>{st.text}</p>
          {runtimeState === "constrained" && certified && (
            <p className="state-extra">
              {Math.round(certified.retention * 100)}% of the learned residual retained after projection
              ({certified.iterations} attenuation rounds).
            </p>
          )}
          {admission.band === "near" && !admission.withheld && (
            <>
              <p className="state-extra">
                Request scaled to {Math.round(admission.scale * 100)}% to stay within the evaluated control
                range. The preview and all metrics use the scaled edit.
              </p>
              <button className="studio-btn primary state-btn" onClick={adoptLimitedEdit}>
                Adopt limited edit into sliders
              </button>
            </>
          )}
          {runtimeState === "outside" && (
            <button className="studio-btn primary state-btn" onClick={reset}>
              Withdraw request
            </button>
          )}
        </div>

        <div className="meter">
          <div className="row">
            <span>{mode === "consult" ? "Distance from evaluated range" : "Control χ (validation frame)"}</span>
            <span>
              {mode === "consult"
                ? admission.band === "in"
                  ? "Within"
                  : admission.band === "near"
                  ? "Near boundary"
                  : "Outside"
                : admission.chi.toFixed(2)}
            </span>
          </div>
          <div className="track">
            <div className="fill" style={{ width: `${Math.max(4, chiPct)}%`, background: chiColor }} />
          </div>
          <div className="row small">
            <span>evaluated ≤ {ctx.ood.policy.within_evaluated_range.toFixed(2)}</span>
            <span>withheld &gt; {ctx.ood.policy.near_boundary.toFixed(2)}</span>
          </div>
        </div>

        {mode === "research" ? (
          <div className="panel-block">
            <h4>Runtime metrics</h4>
            <div className="metrics-grid">
              <div className="cell">
                <span>Anchor flips</span>
                <strong>{anchor ? `${anchor.flipPct.toFixed(2)}%` : "—"}</strong>
              </div>
              <div className="cell">
                <span>Anchor strain p95</span>
                <strong>{anchor ? anchor.strainP95.toFixed(3) : "—"}</strong>
              </div>
              <div className="cell">
                <span>Certificate</span>
                <strong>
                  {certified ? (certified.certified ? "PASS" : "FAIL") : certState.phase === "pending" ? "…" : "—"}
                </strong>
              </div>
              <div className="cell">
                <span>Residual retention</span>
                <strong>{certified ? `${(certified.retention * 100).toFixed(1)}%` : "—"}</strong>
              </div>
              <div className="cell">
                <span>Edit realised: Ridge → RB-SR</span>
                <strong>
                  {certified && certified.realisationAnchor != null
                    ? `${Math.round(certified.realisationAnchor * 100)}% → ${Math.round(certified.realisationCertified * 100)}%`
                    : "—"}
                </strong>
              </div>
              <div className="cell">
                <span>Projection rounds</span>
                <strong>{certified ? certified.iterations : "—"}</strong>
              </div>
              <div className="cell">
                <span>Folds anchor → certified</span>
                <strong>
                  {certified ? `${certified.ridgeFoldCount} → ${certified.projectedFoldCount}` : "—"}
                </strong>
              </div>
              <div className="cell">
                <span>New folds vs anchor</span>
                <strong>{certified ? certified.newVsRidgeFoldCount : "—"}</strong>
              </div>
              <div className="cell">
                <span>Certified latency</span>
                <strong>{certified && certified.computeMs != null ? `${Math.round(certified.computeMs)} ms` : "—"}</strong>
              </div>
            </div>
            <div className="identity-block">
              <div>
                <span>base</span> {shortHash(ctx.identities.base_model_sha256)}
              </div>
              <div>
                <span>gate</span> {shortHash(ctx.identities.gate_model_sha256)}
              </div>
              <div>
                <span>projection</span> {shortHash(ctx.identities.projection_signature)}
              </div>
              <div>
                <span>backend</span>{" "}
                {ctx.service
                  ? "live local service"
                  : ctx.certifiedBrowser
                  ? "in-browser frozen pipeline"
                  : "precomputed presets"}
              </div>
            </div>
            <p className="hint">
              Coordinates are uncalibrated FaceScape model units. Flip and strain values are descriptive
              geometric diagnostics; the certificate is a fold-subset statement relative to the Ridge anchor,
              not a clinical guarantee.
            </p>
          </div>
        ) : (
          <div className="panel-block readout">
            <h4>What changes</h4>
            {SLIDERS.filter((s) => Math.abs(sliders[s.key]) > 0.08).map((s) => (
              <div className="big" key={s.key}>
                {s.label} <span className="unit">· {magnitudeWord(sliders[s.key])}</span>
              </div>
            ))}
            {!dirty && <p className="hint">Move a control to see the nasal region respond in 3D.</p>}
            <p className="hint">
              This preview supports discussion of nasal morphology. It does not predict a surgical outcome.
            </p>
          </div>
        )}

        <div className="panel-block">
          <h4>Session plans</h4>
          <div className="plans">
            {plans.length === 0 && <span className="hint">No saved plans yet.</span>}
            {plans.map((p) => (
              <div className="plan-card" key={p.name}>
                <div>
                  <div className="nm">{p.name}</div>
                  <div className="meta">
                    χ {p.admission.chi.toFixed(2)}
                    {p.certificate ? ` · retention ${(p.certificate.retention * 100).toFixed(0)}%` : " · anchor"}
                  </div>
                </div>
                <span className={`tag ${p.state}`}>{STATES[p.state] ? STATES[p.state].label.split("·")[0].trim() : p.state}</span>
              </div>
            ))}
          </div>
          <div className="btn-grid">
            <button className="studio-btn" onClick={savePlan} disabled={!dirty || admission.withheld}>
              Save plan
            </button>
            <button className="studio-btn primary" onClick={exportReport}>
              Export session
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
