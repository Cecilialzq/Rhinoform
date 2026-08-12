// Input-applicability policy (OOD policy).
//
// This module answers ONE question: is the requested edit inside the range
// of controls the system was evaluated on? It limits or withholds requests,
// and keeps requested vs effective controls as separate facts.
//
// It is deliberately separate from the residual certificate, which concerns
// whether Certified RB-SR adds folds relative to the Ridge anchor. The two
// must never be blended into a single "safety score".

import { controlChi, scaleControls } from "./geometry.js";

export function admitControls(requestedCtrl, ood) {
  const perDimStd = ood.per_dim_std;
  const within = ood.policy.within_evaluated_range;
  const near = ood.policy.near_boundary;
  const chi = controlChi(requestedCtrl, perDimStd);

  if (chi <= within) {
    return {
      band: "in",
      chi,
      effectiveChi: chi,
      scale: 1,
      effectiveCtrl: requestedCtrl,
      withheld: false,
    };
  }
  if (chi <= near) {
    // Scale the request back onto the evaluated boundary. chi is homogeneous
    // of degree 1, so the scaled request lands exactly at the threshold.
    const scale = within / chi;
    return {
      band: "near",
      chi,
      effectiveChi: within,
      scale,
      effectiveCtrl: scaleControls(requestedCtrl, scale),
      withheld: false,
    };
  }
  return {
    band: "out",
    chi,
    effectiveChi: chi,
    scale: 0,
    effectiveCtrl: null,
    withheld: true,
  };
}
