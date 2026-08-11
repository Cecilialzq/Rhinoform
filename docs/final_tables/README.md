# Final paper tables

These tables are derived without retraining from the two frozen experiment families retained by the submission repository.

## Main internal final-rerun results

| Method | ROI RMSE | New flip % | Edge strain p95 | Absolute flip % | Missed flip % |
|---|---:|---:|---:|---:|---:|
| Certified RB-SR | 1.059931 | 0.226894 | 0.203530 | 0.509588 | 1.737834 |
| Ridge | 1.073914 | 0.331663 | 0.203410 | 0.714613 | 1.637579 |
| CVAE | 1.319489 | 1.351909 | 0.238776 | 1.935778 | 1.436659 |
| Hybrid | 1.065372 | 0.442395 | 0.195009 | 0.869027 | 1.593895 |
| LAMM | 0.971544 | 0.878723 | 0.319585 | 1.820195 | 1.079057 |
| Laplacian | 1.642036 | 0.559700 | 0.177864 | 0.894414 | 1.685814 |
| bi-Laplacian | 2.007143 | 1.488462 | 0.283021 | 2.007751 | 1.501239 |
| ARAP | 1.696876 | 0.053687 | 0.061726 | 0.107984 | 1.966231 |

The main table is the clean internal identity-disjoint final rerun over the same 9,900 ordered test pairs. It is not an external or historically untouched confirmation.

## Validation-only component ablation

| Method | ROI RMSE | New flip % | Edge strain p95 |
|---|---:|---:|---:|
| Ridge anchor | 1.124925 | 0.354069 | 0.217822 |
| CVAE proposer | 1.396638 | 1.366797 | 0.238795 |
| Fixed Hybrid | 1.118288 | 0.474226 | 0.208082 |
| Raw RB-SR gate | 1.112194 | 0.557925 | 0.214604 |
| Certified RB-SR | 1.112320 | 0.251323 | 0.216445 |

This component table is validation-only (4,830 ordered pairs). It shows the accuracy--safety role of the learned residual and the frozen ridge-fold certificate without reopening or retuning on test.

## Supplementary post-hoc LAMM comparison

The PCA-128 calibrated RB-SR result is supplementary post-hoc matched evidence on the same 9,900 ordered pairs. It must not be merged with the PCA-64 certified primary model or described as a blind confirmatory headline.
