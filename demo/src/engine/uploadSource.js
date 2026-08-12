// Upload path: bind an arbitrary *registered* source mesh (OBJ, or PLY for
// backward compatibility) to the deployed pipeline.
//
// Requirements are structural, not statistical: the mesh must be in the
// frozen FaceScape TU correspondence — either the nasal ROI (3,934 vertices)
// or a full head (26,317 vertices, ROI extracted via the frozen index map).
// Coordinates must be in the same uncalibrated model units.
//
// Rebinding recomputes exactly the source-dependent quantities of the
// certified pipeline (source PCA code -> condition, absolute R(0), gate
// source normalisation); every weight stays the hash-locked export. The
// rebind path is pinned to the authoritative Python runtime by
// golden_upload.json in `npm run verify`.

const ROI_VERTICES = 3934;
const HEAD_VERTICES = 26317;

// ---------------------------------------------------------------- PLY parse
const PLY_TYPES = {
  char: 1, int8: 1, uchar: 1, uint8: 1,
  short: 2, int16: 2, ushort: 2, uint16: 2,
  int: 4, int32: 4, uint: 4, uint32: 4, float: 4, float32: 4,
  double: 8, float64: 8,
};

export function parsePly(buffer) {
  const bytes = new Uint8Array(buffer);
  const headEnd = new TextDecoder("ascii")
    .decode(bytes.slice(0, Math.min(bytes.length, 65536)))
    .indexOf("end_header");
  if (headEnd < 0) throw new Error("not a PLY file (no end_header)");
  const headerText = new TextDecoder("ascii").decode(bytes.slice(0, headEnd));
  const lines = headerText.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  if (lines[0] !== "ply") throw new Error("not a PLY file");

  let format = null;
  const elements = [];
  let current = null;
  for (const line of lines.slice(1)) {
    const parts = line.split(/\s+/);
    if (parts[0] === "format") format = parts[1];
    else if (parts[0] === "element") {
      current = { name: parts[1], count: parseInt(parts[2], 10), props: [] };
      elements.push(current);
    } else if (parts[0] === "property" && current) {
      if (parts[1] === "list") current.props.push({ list: true, countType: parts[2], itemType: parts[3], name: parts[4] });
      else current.props.push({ list: false, type: parts[1], name: parts[2] });
    }
  }
  const vertexElem = elements.find((e) => e.name === "vertex");
  if (!vertexElem) throw new Error("PLY has no vertex element");
  const xi = vertexElem.props.findIndex((p) => p.name === "x");
  const yi = vertexElem.props.findIndex((p) => p.name === "y");
  const zi = vertexElem.props.findIndex((p) => p.name === "z");
  if (xi < 0 || yi < 0 || zi < 0) throw new Error("PLY vertex element lacks x/y/z");

  const n = vertexElem.count;
  const out = new Float64Array(n * 3);

  // find where the vertex data starts: directly after "end_header\n"
  let dataStart = headEnd + "end_header".length;
  while (bytes[dataStart] === 13 || bytes[dataStart] === 10) dataStart++;

  if (format === "ascii") {
    const text = new TextDecoder("ascii").decode(bytes.slice(dataStart));
    const tokens = text.split(/\s+/).filter(Boolean);
    // vertex element is emitted first only if declared first; enforce that
    if (elements[0].name !== "vertex") throw new Error("PLY: vertex element must come first");
    const stride = vertexElem.props.length;
    for (let v = 0; v < n; v++) {
      const base = v * stride;
      out[v * 3] = parseFloat(tokens[base + xi]);
      out[v * 3 + 1] = parseFloat(tokens[base + yi]);
      out[v * 3 + 2] = parseFloat(tokens[base + zi]);
    }
  } else if (format === "binary_little_endian") {
    if (elements[0].name !== "vertex") throw new Error("PLY: vertex element must come first");
    if (vertexElem.props.some((p) => p.list)) throw new Error("PLY: list property inside vertex element is unsupported");
    const view = new DataView(buffer);
    const sizes = vertexElem.props.map((p) => PLY_TYPES[p.type]);
    if (sizes.some((s) => !s)) throw new Error("PLY: unsupported vertex property type");
    const offsets = [];
    let stride = 0;
    for (const s of sizes) { offsets.push(stride); stride += s; }
    const read = (pos, propIdx) => {
      const p = vertexElem.props[propIdx];
      const at = pos + offsets[propIdx];
      if (p.type === "float" || p.type === "float32") return view.getFloat32(at, true);
      if (p.type === "double" || p.type === "float64") return view.getFloat64(at, true);
      throw new Error("PLY: x/y/z must be float or double");
    };
    for (let v = 0; v < n; v++) {
      const pos = dataStart + v * stride;
      out[v * 3] = read(pos, xi);
      out[v * 3 + 1] = read(pos, yi);
      out[v * 3 + 2] = read(pos, zi);
    }
  } else {
    throw new Error(`PLY format '${format}' is unsupported (ascii / binary_little_endian only)`);
  }
  for (let i = 0; i < out.length; i++) {
    if (!Number.isFinite(out[i])) throw new Error("PLY contains non-finite coordinates");
  }
  return { vertices: out, count: n };
}

// ---------------------------------------------------------------- OBJ parse
// Wavefront OBJ vertex positions: "v x y z" lines. Everything else
// (vt/vn/f/g/usemtl/comments) is ignored — connectivity comes from the
// frozen correspondence, not from the file.
export function parseObj(buffer) {
  const text = new TextDecoder("utf-8").decode(buffer);
  const verts = [];
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line.startsWith("v ") && !line.startsWith("v\t")) continue;
    const parts = line.split(/\s+/);
    if (parts.length < 4) throw new Error(`OBJ vertex line has fewer than 3 coordinates: "${line}"`);
    verts.push(parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]));
  }
  if (verts.length === 0) throw new Error("not an OBJ mesh (no 'v x y z' vertex lines found)");
  const out = Float64Array.from(verts);
  for (let i = 0; i < out.length; i++) {
    if (!Number.isFinite(out[i])) throw new Error("OBJ contains non-finite coordinates");
  }
  return { vertices: out, count: out.length / 3 };
}

// Unified entry point: dispatch on the file extension.
export function parseMesh(buffer, filename) {
  const ext = String(filename || "").toLowerCase().split(".").pop();
  if (ext === "obj") return parseObj(buffer);
  if (ext === "ply") return parsePly(buffer);
  throw new Error(`unsupported mesh format ".${ext}" (expected .obj or .ply)`);
}

// ------------------------------------------------------- support tensor blob
export async function loadUploadSupport(manifest) {
  const meta = manifest.upload_support;
  if (!meta) throw new Error("bundle has no upload support");
  const r = await fetch("/bundle/upload_support.bin", { cache: "force-cache" });
  if (!r.ok) throw new Error(`upload_support.bin -> HTTP ${r.status}`);
  const buf = await r.arrayBuffer();
  const expected = manifest.artifacts["upload_support.bin"];
  const digest = await crypto.subtle.digest("SHA-256", buf);
  const hex = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
  if (hex !== expected) throw new Error("upload_support.bin hash mismatch (fail closed)");
  const all = new Float32Array(buf);
  const grab = (name) => {
    const { offset, shape } = meta.tensors[name];
    const len = shape.reduce((a, b) => a * b, 1);
    return { data: all.subarray(offset, offset + len), shape };
  };
  return {
    pcaMean: grab("pca_mean").data,
    pcaComponents: grab("pca_components").data,   // (64, 11802) row-major
    ridgeW: grab("ridge_w").data,                 // (91, 11802) row-major
    ridgeB: grab("ridge_b").data,
    condMean: grab("cond_mean").data,
    condStd: grab("cond_std").data,
    gateCenter: grab("gate_center").data,
    gateScale: grab("gate_scale").data[0],
  };
}

// -------------------------------------------------------------- source bind
// Validate the uploaded mesh and classify it as ROI or full head.
export function classifyUpload(parsed, meanSource, headCtx) {
  if (parsed.count !== ROI_VERTICES && parsed.count !== HEAD_VERTICES) {
    throw new Error(
      `mesh has ${parsed.count} vertices; expected ${ROI_VERTICES} (frozen ROI) ` +
      `or ${HEAD_VERTICES} (registered full head)`
    );
  }
  const kind = parsed.count === ROI_VERTICES ? "roi" : "head";
  let roi;
  if (kind === "roi") {
    roi = parsed.vertices;
  } else {
    if (!headCtx) throw new Error("full-head upload requires the head context map");
    roi = new Float64Array(ROI_VERTICES * 3);
    for (let i = 0; i < ROI_VERTICES; i++) {
      const hv = headCtx.roiIndices[i];
      roi[i * 3] = parsed.vertices[hv * 3];
      roi[i * 3 + 1] = parsed.vertices[hv * 3 + 1];
      roi[i * 3 + 2] = parsed.vertices[hv * 3 + 2];
    }
  }
  // Scale sanity in the uncalibrated model frame: ROI bbox diagonal must stay
  // within 2x of the population mean's (units mismatch fails loudly here).
  const diag = (V, n) => {
    const mn = [Infinity, Infinity, Infinity];
    const mx = [-Infinity, -Infinity, -Infinity];
    for (let v = 0; v < n; v++) {
      for (let d = 0; d < 3; d++) {
        const x = V[v * 3 + d];
        if (x < mn[d]) mn[d] = x;
        if (x > mx[d]) mx[d] = x;
      }
    }
    return Math.hypot(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]);
  };
  const dU = diag(roi, ROI_VERTICES);
  const dM = diag(meanSource, ROI_VERTICES);
  if (dU < dM / 2 || dU > dM * 2) {
    throw new Error(
      `ROI bounding-box diagonal ${dU.toFixed(1)} is far from the calibrated frame ` +
      `(expected about ${dM.toFixed(1)} model units); check units/registration`
    );
  }
  return { kind, roi };
}

// Recompute the source-dependent context of the certified browser pipeline.
// Returns a new `cb` object; all weight arrays are shared by reference.
export function rebindSource(cb, support, roi) {
  const D = ROI_VERTICES * 3;
  // 64-D source PCA code: (flat - mean) @ components^T
  const code = new Float64Array(64);
  for (let k = 0; k < 64; k++) {
    let acc = 0;
    const row = k * D;
    for (let i = 0; i < D; i++) acc += (roi[i] - support.pcaMean[i]) * support.pcaComponents[row + i];
    code[k] = acc;
  }
  // Normalised zero-edit condition and the raw R(0) prediction
  const cond0 = new Float64Array(91);
  for (let i = 0; i < 27; i++) cond0[i] = (0 - support.condMean[i]) / support.condStd[i];
  for (let k = 0; k < 64; k++) cond0[27 + k] = (code[k] - support.condMean[27 + k]) / support.condStd[27 + k];
  const r0 = Float64Array.from(support.ridgeB);
  for (let j = 0; j < 91; j++) {
    const c = cond0[j];
    if (c === 0) continue;
    const row = j * D;
    for (let i = 0; i < D; i++) r0[i] += c * support.ridgeW[row + i];
  }
  // Gate source normalisation
  const srcNorm = new Float64Array(D);
  for (let v = 0; v < ROI_VERTICES; v++) {
    for (let d = 0; d < 3; d++) {
      srcNorm[v * 3 + d] = (roi[v * 3 + d] - support.gateCenter[d]) / support.gateScale;
    }
  }
  // resid0Filled is the cached zero-controls display residual of the OLD
  // source -- it must be recomputed for the rebound source.
  return { ...cb, cond0, r0Raw: r0, sourceNormalized: srcNorm, sourceCode: code, resid0Filled: null };
}
