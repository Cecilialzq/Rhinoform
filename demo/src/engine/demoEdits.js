// Boundary demonstrations: UI-level preset edits that exercise the two
// admission outcomes that in-range presets cannot show live:
//
//   demo_scaleback  chi ~= 8.0  -> "near boundary": the request is scaled
//                                  back exactly onto the evaluated threshold
//                                  (7.49) and the scaled edit is certified.
//   demo_withheld   chi ~= 9.9  -> "outside range": the preview is withheld
//                                  rather than extrapolated.
//
// They are intentionally NOT part of the certified preset library: the
// admission policy, not a precomputed result, is what they demonstrate. The
// slider values exceed the consultation range (+-1) by design and live in
// the research range (+-2); applying one switches the studio to research
// mode. chi values below are computed against the frozen OOD statistics in
// manifest.json (per_dim_std); verify.mjs asserts the band classification.
export const DEMO_EDITS = [
  {
    key: "demo_scaleback",
    label: "Boundary demo — auto scale-back (χ≈8.0)",
    sliders: [1.37, -1.24, 1.24, 1.24, -1.24, 0.96],
    expectBand: "near",
  },
  {
    key: "demo_withheld",
    label: "Boundary demo — withheld request (χ≈9.9)",
    sliders: [1.7, -1.53, 1.53, 1.53, -1.53, 1.19],
    expectBand: "out",
  },
];
