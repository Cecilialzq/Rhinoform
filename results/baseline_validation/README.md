# Supplemental baseline-validation records

These files were recovered from the project Google Drive and are committed so
that the public evidence package no longer omits them. They are not the original
joint selection record for the final PCA-64, ridge-lambda-300 operating point.

- `source_pca_dim_sweep/` is explicitly labelled
  `canonical_strict_posthoc_supplemental`. It compares PCA dimensions 0, 8, 16,
  32, and 64 at lambda 100 on the frozen 9,900-pair test panel.
- `ridge_lambda_selection/` is explicitly labelled
  `supplemental_validation_sweep_not_original_training_selection`. It compares
  lambdas at PCA dimension 16, not the final PCA-64 configuration.
- `constants/` is a contemporaneous implementation inventory for that
  supplemental branch. Some values (including PCA-16/lambda-100) describe the
  branch, not the final deployed bundle.

Accordingly these records document nearby behaviour and improve provenance,
but they do not prove that PCA-64/lambda-300 was the public-package optimum.
The final deployed values remain reproducible frozen settings whose original
development selection sweep is not in the public release.
