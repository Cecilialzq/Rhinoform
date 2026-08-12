// Session-report assembly and download, kept as pure/verifiable pieces so
// verify.mjs can accept the export path automatically (the browser download
// itself is a one-line anchor click around buildSessionReport's output).

export function buildSessionReport({ identities, sourceKey, certifiedBackend, session, plans }) {
  return {
    product:
      "RHINOFORM — population-informed sparse-control nasal preview (research prototype)",
    generated: new Date().toISOString(),
    deployment: {
      locked_identities: identities,
      source_key: sourceKey,
      certified_backend: certifiedBackend,
      units: "uncalibrated FaceScape model units (not millimetres)",
    },
    session,
    plans,
    disclaimer:
      "Research prototype for morphology communication. Not a medical device. Outputs do not predict surgical or anatomical results. The fold-subset certificate is a geometric statement relative to the Ridge anchor, not a clinical safety guarantee.",
  };
}

// Fields any exported session must carry; asserted in verify.mjs and cheap
// to keep in sync with buildSessionReport above.
export const REPORT_REQUIRED_KEYS = [
  "product",
  "generated",
  "deployment",
  "session",
  "plans",
  "disclaimer",
];

export function downloadJson(payload, filename) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  // Attach to the document so synthetic clicks are honoured by every browser
  // (and visible to automated acceptance tooling), then clean up.
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
