# Final experiment map

## 1. Primary: certified RB-SR internal final rerun

- Notebook: `notebooks/Rhinoform_RBSR_internal_final_rerun_holdout_colab.ipynb`
- Result root: `results/rbsr_final_rerun_holdout_v1/`
- Split: 576 train, 70 validation and 100 test identities; identity-disjoint.
- Evaluation: all 9,900 ordered non-self test pairs.
- Primary RB-SR operating point: source PCA dimension 64, Ridge anchor lambda 300, residual proposer plus learned gate, and frozen ridge-fold projection attenuation 0.75.
- Selection: validation only; test accessed once after all models and evaluation code were frozen.
- Primary hypothesis: certified RB-SR has lower ROI RMSE than matched Ridge, no higher target-relative new flip, and certificate rate 1.0.
- Outcome: passed. Direct paired statistics support the RMSE and new-flip improvements.

This experiment supplies the main method row, all seven baseline rows and the deployment-oriented conclusion.

## 2. Supplementary: PCA-128 RB-SR versus frozen LAMM

- Notebook: `notebooks/Rhinoform_RBSR_LAMM_dominance_search_colab.ipynb`
- Result root: `results/supplemental_rbsr_lamm_dominance_search_v1/`
- Split, ROI, strict scorer and ordered test pairs: identical to the primary final rerun.
- LAMM: reused frozen checkpoint and pair-level result; no LAMM retraining.
- Selected RB-SR configuration: source PCA dimension 128, Ridge lambda 1, validation-selected logit offset beta 6.
- Evidence type: supplementary post-hoc matched comparison.
- Outcome on 9,900 test pairs: RB-SR is lower on ROI RMSE, target-relative new flip and edge-strain p95; the three matched-pair tests remain significant after the stored correction/robustness procedure.

This experiment supports the narrower statement that additional representation capacity can move RB-SR to a point that dominates LAMM on these three metrics under the matched protocol. It does not establish that increasing PCA dimension always improves every split or that PCA-128 is the primary deployed model.

## Why the two experiments must not be numerically merged

They share evaluation data and scorer but use different RB-SR model configurations and different evidential roles. Present them as:

- the PCA-64 certified model in the main table; and
- the PCA-128 calibrated model in a separately labelled supplementary table or trade-off figure.

Do not select the best metric from each configuration to create a synthetic row.

## Existing ablation sufficient for the final report

The validation-only component table in `docs/final_tables/component_ablation_validation_4830_pairs.csv` shows:

| Component state | ROI RMSE | New flip % | Edge-strain p95 |
|---|---:|---:|---:|
| Ridge anchor | 1.124925 | 0.354069 | 0.217822 |
| CVAE proposer | 1.396638 | 1.366797 | 0.238795 |
| Fixed Hybrid | 1.118288 | 0.474226 | 0.208082 |
| Raw RB-SR gate | 1.112194 | 0.557925 | 0.214604 |
| Certified RB-SR | 1.112320 | 0.251323 | 0.216445 |

The raw gate gains accuracy but increases new flips; the hard certificate retains nearly all of the RMSE gain while reducing new flips below Ridge with a certificate rate of 1.0. This is the appropriate final-protocol explanation of the safety component.

## Retained post-hoc evaluation evidence

The frozen `posthoc_subunit_analysis_v1` tree contains the five nasal-subunit
breakdown, robustness/stress analyses, qualitative cases and failure-oriented
evidence generated after the primary freeze. These analyses use the frozen
methods and must remain labelled secondary post-hoc; they do not change either
headline model or reopen test-time selection.

## Claims that would require a new experiment

- multi-seed stability or variance across training seeds;
- causal attribution to every individual loss coefficient;
- robustness outside the retained frozen noise protocol;
- subunit claims beyond the retained five-subunit analysis;
- external-dataset generalisation; or
- measured deployment latency/memory for the full learned path.

None of these is required to support the current main and supplementary conclusions, provided the report does not make those extra claims.
