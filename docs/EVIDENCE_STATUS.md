# Evidence status and report claim gate

This file is the controlling evidence map. The public result surface contains
the primary confirmation, the supplementary matched LAMM comparison and the
five-subunit post-hoc analysis. Other tracked files are implementation,
data-construction or exact-source provenance, not additional paper result sets.

| Claim or component | Status | Canonical evidence | Permitted wording |
|---|---|---|---|
| Final-rerun split: 576 train, 70 validation, 100 test identities | verified | `results/rbsr_final_rerun_holdout_v1/protocol/final_rerun_holdout_split_manifest.json` | clean identity-disjoint internal final-rerun split |
| Same 9,900 ordered test pairs for all eight primary methods | verified | primary pair CSVs and `FINAL_CONFIRMATION_EVIDENCE.json` | matched internal comparison |
| 3,934-vertex ROI, nine exact controls and 3,925 scored free vertices | verified | `roi/vertices.json`; strict scorer | exact evaluation protocol |
| PCA-64 certified RB-SR primary result | verified internal final rerun | `results/rbsr_final_rerun_holdout_v1/rbsr/seed20260609/one_shot_test/` | primary certified RB-SR result |
| RB-SR improves Ridge ROI RMSE and target-relative new flip | verified and directly tested | `direct_paired_statistics/paired_primary_rbsr_vs_ridge.*` | Holm-corrected direct RB-SR-versus-Ridge improvement |
| RB-SR edge strain versus Ridge | no significant difference | same direct statistics | comparable to Ridge; do not claim significant improvement |
| Seven primary baselines: Ridge, CVAE, Hybrid, LAMM, Laplacian, bi-Laplacian and ARAP | verified matched outputs | primary result directory and `docs/final_tables/main_results_9900_pairs.csv` | all evaluated under the same split, ordered pairs and strict scorer |
| Validation component analysis | verified validation-only | `docs/final_tables/component_ablation_validation_4830_pairs.csv` | illustrates anchor, proposer, learned gate and hard-certificate roles; not a test headline |
| PCA-128 calibrated RB-SR beats frozen LAMM on RMSE, new flip and edge strain | verified supplementary post-hoc matched evidence; **no hard certificate run** | `results/supplemental_rbsr_lamm_dominance_search_v1/release_evidence/` (`projection_certificate`, `projection_freeze`, `projection_selected` are null) | supplementary observed trade-off evidence; do not call this point certified |
| Statistical significance of all three PCA-128-versus-LAMM core metrics | verified | `release_evidence/analysis/rbsr_vs_lamm_core_paired_statistics.*` | matched-pair significant within this supplementary analysis |
| Primary result-tree freeze | verified Drive snapshot | `RESULT_TREE_MANIFEST.json`; `FINAL_RESULT_FREEZE_AUDIT.json` | exact Drive result files are frozen |
| Supplementary release freeze | verified Drive snapshot | `release_evidence/RELEASE_MANIFEST.json` | exact supplementary evidence files are frozen |
| Public GitHub source release | verified | public `Cecilialzq/Rhinoform` repository, annotated evidence tags and passing CI | public reproducible source release |
| Five-subunit, robustness and qualitative post-hoc analyses | verified secondary post-hoc evidence | `results/rbsr_final_rerun_holdout_v1/posthoc_subunit_analysis_v1/` | secondary analysis; not a new blind confirmation |
| Apple M3 certified browser compute distribution | verified deployment evidence | `benchmarks/runtime/browser_certified_benchmark.mjs`; `benchmarks/runtime/results/browser_node_certified.json` | Ridge is the interactive anchor/fallback; certified RB-SR is an asynchronous enhanced preview |
| Chrome presentation distribution and sampled process RSS | verified single-device deployment evidence | `benchmarks/runtime/browser_render_benchmark.mjs`; `benchmarks/runtime/results/browser_chrome_render.json` | report separately from compute; RSS is process-tree observation, not per-tab or GPU peak memory |
| LAMM deployment latency | verified batch-one A100 forward-only record | `benchmarks/runtime/results/lamm_a100_batch1_forward.json` | 100 warmups/1,000 runs; 37.085/39.093/41.931 ms median/p95/p99; pre-staged GPU inputs; do not rank against M3 paths |
| Retraining scope | GitHub alone is insufficient, but the documented code supports retraining when the user supplies licensed FaceScape data, the pinned upstream LAMM source and CUDA; author-side licensed retraining was executed | release runbook, private checkpoint preflight and recovered Colab A100 notebook | exact result replay and browser conformance require only GitHub; fresh retraining requires external licensed inputs; exact author-artifact reproduction additionally requires the retained checkpoints |

## Claim boundaries

- The primary evidence is an **internal identity-disjoint held-out-from-final-retrain** result. It is not an external dataset and must not be described as historically untouched.
- The supplementary PCA-128 result is **post-hoc**. It cannot replace the primary PCA-64 certified model or be described as blind confirmation.
- The release freezes PCA-64/lambda 300 and now includes post-hoc PCA and supplemental lambda diagnostics under `results/baseline_validation/`, but those records explicitly are not the original joint selection run; describe the final pair as frozen operating constants, not as a publicly auditable optimum.
- The browser benchmark separates Node/V8 compute from Chrome presentation. Do not sum their quantiles or relabel the headless-Chrome process-tree RSS as per-tab/GPU memory.
- `normal_flip_pct` means target-relative new-flip percentage. It is not absolute flip percentage.
- Landmark RMSE is structurally zero because the nine controls are hard-fixed; it is not evidence of statistical improvement.
- A passing Drive manifest freezes artifacts but does not substitute for a Git commit.

## Report exclusion rule

Do not cite archived iterations, historical headline rows, old split results, or unexecuted post-hoc studies in the final report. If a claim is not supported by one of the rows above, either remove it or run a newly declared study.
