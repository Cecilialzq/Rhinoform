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

The final-rerun policy freezes and hash-locks PCA-64 and lambda 300. The public
release now includes a post-hoc PCA sweep at lambda 100 and a supplemental
validation lambda grid at PCA-16 under `results/baseline_validation/`; their
protocol fields explicitly state that they are not the original joint selection
run. The released result is therefore reproducible at the frozen operating
point; the release is not evidence that PCA-64/lambda 300 is an independently
auditable optimum.

## 2. Supplementary: PCA-128 RB-SR versus frozen LAMM

- Notebook: `notebooks/Rhinoform_RBSR_LAMM_dominance_search_colab.ipynb`
- Result root: `results/supplemental_rbsr_lamm_dominance_search_v1/`
- Split, ROI, strict scorer and ordered test pairs: identical to the primary final rerun.
- LAMM: reused frozen checkpoint and pair-level result; no LAMM retraining.
- Selected RB-SR configuration: source PCA dimension 128, Ridge lambda 1, validation-selected logit offset beta 6.
- Unlike the primary CVAE proposer with a budget-constrained gate, this point uses a deterministic neural-field residual proposer trained for 500 epochs, followed by unconstrained accuracy-first gate training.
- No hard projection certificate was run: `projection_certificate`, `projection_freeze` and `projection_selected` are all null in the released test record.
- Evidence type: supplementary post-hoc matched comparison.
- Outcome on 9,900 test pairs: RB-SR is lower on ROI RMSE, target-relative new flip and edge-strain p95; the three matched-pair tests remain significant after the stored correction/robustness procedure.

This experiment supports the narrower statement that a changed, calibrated
anchored configuration can move to a point that dominates the stored LAMM
outputs on these three observed metrics under the matched protocol. The
calibrated gate is almost closed (mean admission 0.0021; no vertex above 0.5),
so the result is anchor-dominated. Because PCA dimension, Ridge penalty,
proposer, gate objective and calibration all change, the experiment does not
isolate representation capacity, demonstrate certified capacity, establish
that increasing PCA dimension always improves every split, or make PCA-128 the
primary deployed model.

## Why the two experiments must not be numerically merged

They share evaluation data and scorer but use different RB-SR model configurations and different evidential roles. Present them as:

- the PCA-64 certified model in the main table; and
- the PCA-128 calibrated, non-certified model in a separately labelled supplementary table or trade-off figure.

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

The subunit notebook cell 23 applies a runtime-only zero-noise replay repair:
it authenticates the frozen zero-noise rows by artifact hash and exact pair identity,
then treats finite cross-GPU order-statistic differences as diagnostics. It does
not change model weights, noise draws, metric definitions, the operating point,
split or pair order, or test-data access.

## Claims that would require a new experiment

- multi-seed stability or variance across training seeds;
- causal attribution to every individual loss coefficient;
- robustness outside the retained frozen noise protocol;
- subunit claims beyond the retained five-subunit analysis;
- external-dataset generalisation; or
- a same-hardware LAMM-versus-public-browser deployment comparison or a stable browser GPU peak-memory figure.

The repository does now include a 100-warmup/1,000-run Apple M3 benchmark of the
exact browser bundle and a separate Chrome presentation benchmark under
`benchmarks/runtime/`. Those records support Ridge as the interactive fallback
and certified RB-SR as an asynchronous enhanced preview. A recovered
batch-size-one A100 LAMM forward record is also committed, but it does not fill
the same-hardware comparison or browser GPU-memory gaps above.
