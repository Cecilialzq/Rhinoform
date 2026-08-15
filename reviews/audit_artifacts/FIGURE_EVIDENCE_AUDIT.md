# Rendered-figure evidence audit

## Scope and authority

This audit covers every unique image asset actually included by
`/Users/lizequan/Desktop/Rhinoform_Overleaf_Submission_副本/main.tex`: 35 assets,
with no missing files.  Each asset was rendered and its visible text, formulae,
numbers, units, legends, arrows and panel relationships were checked against the
frozen repository evidence.  The remote authority at the start of the audit was
commit `35577895ef20c007351e6213d08f6d5629d9987b` on
`agent/runtime-evidence-sync`; the commit containing this ledger adds the
one-line interface correction and the machine-readable audit artefacts.

Machine-readable companions:

- `figure_asset_inventory.csv`: exact report path, byte size and SHA-256.
- `figure_evidence_chain.csv`: claim role and repository evidence paths.
- `audit_figure_assets.py` and `build_figure_evidence_chain.py`: deterministic
  inventory/mapping scripts.

## Modifications supported by a complete evidence chain

Only the five conflicts below were modified.  All changes preserve the existing
composition and alter only the contradicted label or scope statement.

| Asset | Original content | Exact evidence | Why a change was necessary | Minimal correction and verification |
|---|---|---|---|---|
| Figure 3.1 studio screenshot | “in calibrated model units” | `demo/src/engine/uploadSource.js` states that uploaded coordinates use the same **uncalibrated** model units; `demo/src/App.jsx` gives the mandatory “uncalibrated model units” framing; the surrounding report uses the same claim boundary. | The screenshot contradicted both the upload contract and the responsible-use statement. | Changed only `calibrated` to `uncalibrated` in `demo/src/components/Studio.jsx` and in the captured line. `tools/patch_studio_units.py` hash-checks the source raster before changing that line. All other screenshot pixels and displayed example values are retained. |
| Figure 3.2 communication sequence | “Re-impose exact handles; submit D_raw” | `demo/src/engine/certifiedLocal.js` first constructs the gated residual, zeroes it at landmarks (lines 230–249 in the audited checkout), then passes that handle-neutralised field to the certificate. The tensor contract and Algorithm 4.1 call the resulting field `D_hard`. | A post-projection object cannot still be labelled `D_raw`; the old label collapsed two distinct algorithm states. | Replaced the arrow text with “Neutralise gated residual at handles; submit D_hard”. No actor, arrow, timing or branch changed. The standalone TikZ source recompiles successfully. |
| Figure 3.3 software architecture | Absolute claim that inference “remain[s] local to the browser” | `demo/src/engine/certified.js` tries an optional configured local research service, then falls through to the in-browser certified path; the public standalone path does not require the service. | The old unqualified sentence was too broad when a development service is configured. | Added only the scope “On the standalone public path”. The browser boundary, modules and arrows are unchanged. The standalone TikZ source recompiles successfully. |
| Figure 4.2 framework legend | “Zeroed (new fold vs Ridge)” represented a local certificate outcome | `demo/src/engine/certifiedLocal.js` identifies incident vertices of newly folded faces and performs `weights[v] *= attenuation` with attenuation 0.75; only the uniform fallback reaches alpha=0, the guaranteed Ridge endpoint. The Python implementation and Algorithm 4.1 implement the same rule. | The old legend falsely implied a discrete local-zeroing branch. | Replaced only that legend line with “Ridge endpoint (uniform fallback)”. The original artwork is embedded unchanged beneath a vector text overlay; the exact-handle legend remains present. |
| Appendix Figure A.1 | Values 0.803, 1.223, 0.759, 1.851, 0.741 and 1.059 were labelled “RMSE” | `build_redrawn_figures.py` computes these annotations as `mean(certified_rbsr_vertex_error)`. The six private NPZ files were recovered from the licensed Drive package and matched the SHA-256 values in `QUALITATIVE_CASES_EVIDENCE.json` byte-for-byte. `QUALITATIVE_CASE_SELECTION_FREEZE.json` separately records free-ROI RMSE for selection. | The displayed numbers were valid but the metric name was wrong; replacing the values with selection RMSE would misdescribe what the plot generator computed. | Kept all six numbers and changed only “RMSE” to “Mean error”; the caption now distinguishes mean per-vertex Euclidean error from selection free-ROI RMSE. The vector figure was regenerated from the hash-matched NPZs. |

Changed-asset digest trail:

| Asset | Original SHA-256 | Corrected SHA-256 |
|---|---|---|
| Figure 3.1 | `d838f1654593c2d3a7e950a497df9baf09015c6326fe2990e4e2741691c4cb17` | `8030d127deb35b9897797543b79e4640ba0c7fc5fb2b5a8d4b0cb9a11bb484f8` |
| Figure 3.2 | `9a5484b767ad54303636448212aed95b07a9c8c0d52cee0fb58f1e6162d704e7` | `3f2d005400cb68495360b0d811d2d40849eac0b4e06e1424a7fb81c5b72afdce` |
| Figure 3.3 | `c89410642a8cb4940b011cf409f4a97fd320e3667ad1c30e0fcaa028fb5288e7` | `f809a5948697416b6f3631a3900bd93ad0a55d353efee80c3378ff543acc41bf` |
| Figure 4.2 | `e5d098410d5aa8eb0d1892f9ad7a3b91e32efb59959355e994ed8826a4f4e954` | `fbc126aeb28da1a621cc308119fb62a2e1e84ba0953f733a8c30723d3a31c846` |
| Appendix Figure A.1 | `2c20f1d3af41a88ddc89ba214138aee76b94db5be0dc824208e5b90af1fe3201` | `a3f116e57e233e581a5064b5a18ee392e36a1a09359099d2bd52827e1f23b9c8` |

Figure 3.4 was not modified; its source and final SHA-256 remain
`5a91a5807f3b7bc912dab34c9bd30a32e52dca981bb48630455107fd4247015e`.

## Complete per-asset result

“Pass” means the rendered content, caption role and supporting evidence agree.
“Corrected” means the evidence-backed minimal change above was applied.

| # | Rendered asset | Result | Evidence checked |
|---:|---|---|---|
| 1 | `fig_1_1_clinical_gap.pdf` | Pass | Sparse verbal request → 9 handles/27-D controls → 3,934×3 dense field → certified enhancement/Ridge fallback; no clinical prediction claim. |
| 2 | `fig_2_1_task_taxonomy.pdf` | Pass | Literature/task taxonomy and sparse-to-dense role; no experimental value. |
| 3 | `fig_4_1_task_geometry.pdf` | Pass | `roi/vertices.json`, `roi/subunits.json`, bundle landmarks and 3,934-vertex ROI contract. |
| 4 | `fig_4_2_rbsr_framework.pdf` | Corrected | `demo/src/engine/certifiedLocal.js`, `rhinoform/safe_fusion.py`, `scripts/evaluation/rbsr_ridge_fold_projection.py`, projection freeze. |
| 5 | `fig_5_1_evaluation_protocol.pdf` | Pass | 4,830 validation pairs (70×69), 9,900 primary pairs (100×99), identity-disjoint split and eight-method protocol. |
| 6 | `eval_02_subunit_matrix_review_v1.pdf` | Pass | `subunit_means_all_methods.csv` and `region_topology_audit.json`; Hybrid omission is disclosed as an internal ablation. |
| 7 | `eval_03_spatial_operating_points_review_v1.pdf` | Pass | Frozen spatial NPZ sidecars, publication-figure ledger and the four displayed operating points. |
| 8 | `eval_06_qualitative_extremes_review_v1.pdf` | Pass | Deterministic roles 05/06, mandatory failure case, shared scales and descriptive/post-hoc boundary. |
| 9 | `eval_01_operating_points_review_v1.pdf` | Pass | `main_results_9900_pairs.csv`; method values, axes and bubble encoding agree. Hybrid omission is explicit. |
| 10 | `eval_05_noise_robustness_review_v1.pdf` | Pass | `noise_robustness_summary.csv` and strict-noise protocol/evidence; validation/post-hoc scope retained. |
| 11 | `fig_a_1_qualitative_all.pdf` | Corrected | Six SHA-matched private NPZs, qualitative selection freeze and evidence JSON; annotation computation verified in the generator. |
| 12 | `fig_a_3_subunit_strain.pdf` | Pass | All eight methods and five regions agree with `subunit_means_all_methods.csv`; the multi-method regional comparison warrants a chart. |
| 13 | `imperial-logo.pdf` | Pass / non-scientific | Template branding only. |
| 14 | `morphable_3dmm.pdf` | Pass / external | Prior-work illustration, used only with its cited source. |
| 15 | `pca_shape_model.pdf` | Pass / external | Prior-work illustration, used only with its cited source. |
| 16 | `flame_model.png` | Pass / external | Prior-work illustration, used only with its cited source. |
| 17 | `flame_model_joints.png` | Pass / external | Prior-work illustration, used only with its cited source. |
| 18 | `facescape.pdf` | Pass / external | Dataset/prior-work illustration, not presented as project output. |
| 19 | `laplacian_editing.pdf` | Pass / external | Prior-work illustration, not a Rhinoform result. |
| 20 | `bilaplacian.pdf` | Pass / external | Prior-work illustration, not a Rhinoform result. |
| 21 | `arap_deformation.pdf` | Pass / external | Prior-work illustration, not a Rhinoform result. |
| 22 | `cvae_architecture.pdf` | Pass / external | Background illustration; project-specific architecture is stated separately. |
| 23 | `fig_3_1_studio_modes_final.pdf` | Corrected | `Studio.jsx`, `App.jsx`, `uploadSource.js`, bundle state and screenshot example-value boundary. |
| 24 | `fig_3_2_software_communication_sequence_final.pdf` | Corrected | Six sliders → 27-D controls; independent applicability; immediate Ridge; asynchronous proposer/gate; handle-neutralised hard field; certificate; fail-closed Ridge. |
| 25 | `fig_3_3_software_architecture_final.pdf` | Corrected | `bundle.js`, `certified.js`, `certifiedLocal.js`, `Studio.jsx`; optional development service distinguished from standalone public path. |
| 26 | `fig_3_4_runtime_state_machine_final.pdf` | Pass | 450-ms debounce, applicability retain/scale/withhold, retention threshold 0.98, constrained uniform/lower-retention path, error/timeout Ridge fallback. |
| 27 | `fig_4_3_learning_architecture.pdf` | Pass | 91-D condition, 8-D latent, 15-D vertex features, 114-D decoder input, 115-D gate input, dense proposal → residual → gate → hard handles → certificate. |
| 28 | `fig_4_4_stage1_anchor_evidence.pdf` | Pass | `component_ablation_validation_4830_pairs.csv` and frozen anchor configuration. |
| 29 | `fig_4_5_stage2_residual_evidence.pdf` | Pass | Validation alpha sweep and fixed Hybrid ablation; no vertex-level conclusion attributed to the scalar sweep. |
| 30 | `fig_4_6_stage3_gate_evidence.pdf` | Pass | Gate training history and component ablation: raw-gate accuracy/regularity trade-off and no feasible soft-orientation checkpoint. |
| 31 | `fig_4_7_stage3_spatial_mechanism.pdf` | Pass | Gate/spatial diagnostics; spatial heterogeneity is not misrepresented as a hard certificate. |
| 32 | `fig_4_8_stage4_certificate_evidence.pdf` | Pass | Validation projection CSV/freeze: certificate rate 1.0, mean iterations ≈13.38, mean retention ≈0.9917, max local rounds 64. |
| 33 | `eval_00_primary_paired_effects_review_v1.pdf` | Pass | `primary_rbsr_vs_ridge_statistics.csv`; mean effects, clustered CIs and six-family Holm-adjusted p-values. |
| 34 | `eval_04_classical_spatial_tail_review_v1.pdf` | Pass | `spatial_tail_diagnostics_all_methods.csv`; aggregation level and classical-method scope agree. |
| 35 | `eval_07_personalisation_scatter_review_v1.pdf` | Pass | Personalisation ablation evidence; source-conditioning contribution is not called individual-outcome prediction. |

## Part A figure-related conclusions

- A1/A22/A27: all 35 rendered assets were read directly; five label/scope
  conflicts were corrected and the remaining 30 passed.
- A3–A11: proposal, gate, hard-handle and certificate figures now match the
  implementation order and the exact local-attenuation/uniform-fallback rule.
- A16: the runtime table separates compute-only browser timings, rendering and
  the recovered A100 LAMM forward-only timing; it does not rank unlike hardware.
- A20: no project figure makes a physical-unit claim for the uncalibrated
  FaceScape coordinates; the stale UI word was corrected.
- A21: no figure conflates the primary CVAE RB-SR pipeline with the
  supplementary PCA-128 deterministic-field capacity experiment.
- A23/A24: explanatory system figures remain caption-led; core quantitative and
  stage-development figures are interpreted in the body.  The retained
  multi-method/subunit bar chart carries regional comparison information and is
  not a two-value chart better represented as a table.

No figure was redrawn wholesale.  The original Figure 3 composition, software
communication diagram, architecture diagram and state machine remain in place.

## Additional aligned prose correction

The Chapter 3 explanation previously said gate admission was decided “per
region”.  `demo/src/engine/certifiedLocal.js` stores and applies one gate weight
per vertex, and the method equation defines `g(v)`.  The wording was therefore
minimally corrected to “spatially, through per-vertex weights”; no method claim,
number or figure geometry changed.

The abstract previously called the nine controls “FaceScape-calibrated”
landmarks.  `data/manifest.json`, `roi/landmarks.json` and the upload contract
support registered topology and fixed landmark indices, while
`demo/src/engine/uploadSource.js` explicitly defines the coordinates as
uncalibrated model units.  The phrase was minimally changed to “registered
FaceScape semantic nasal landmarks” to remove the unsupported calibration
implication.
