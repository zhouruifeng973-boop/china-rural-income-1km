# Sensitivity analysis excluding 533 labels pending numeric verification

## Design

The frozen v1.5-no-road baseline retained all 8,373 county-year labels. The
sensitivity variant used only the 7,840 labels whose county names and numeric
values had been verified against the linked source item. Province folds,
features, pixel-sampling rules, model settings and random seeds were held fixed.
Both variants were scored on the same 7,753 verified county-year records with
valid rural-pixel panels. Pending records were excluded from scoring in both
variants.

## Overall metrics

| Variant | n | R2 | MAE | RMSE | MAPE (%) | Bias |
|---|---:|---:|---:|---:|---:|---:|
| all_8373_labels | 7,753 | 0.3959 | 3676.6 | 5222.2 | 19.67 | -946.9 |
| verified_7840_labels_only | 7,753 | 0.3860 | 3707.1 | 5264.8 | 19.85 | -918.8 |

## Paired comparison

- Delta R2 (verified-only minus all labels): -0.0099.
- Delta MAPE: +0.183 percentage points.
- Pearson correlation between paired predictions: 0.99787.
- Spearman correlation between paired predictions: 0.99842.
- Mean absolute paired difference: 164.5 2019-CNY, or 0.933% of the baseline mean prediction.

## Interpretation

The pre-specified aggregate stability checks returned `sensitivity_pass`.
Excluding the 533 pending records caused only small changes in province-transfer
validation metrics and very high paired prediction agreement. This result controls
the risk associated with retaining source-mapped but not yet numerically verified
training labels. It does not constitute pixel-level accuracy validation.
