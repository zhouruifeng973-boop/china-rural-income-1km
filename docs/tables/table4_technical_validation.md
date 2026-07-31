# Table 4. Technical-validation metrics

**Table 4. County-level validation metrics after rural-population-weighted aggregation.** Pixel predictions were aggregated back to counties using rural-population weights and compared with observed county-level rural disposable income. MAE, RMSE, and bias are in 2019 constant yuan per person per year; MAPE is a percentage. These metrics do not represent pixel-level validation. Province calibration is a product-generation constraint, so after-calibration results describe the constrained product rather than fully independent validation.

| Validation experiment | n | R² | MAE | RMSE | MAPE | Bias |
|---|---:|---:|---:|---:|---:|---:|
| 2020 leave-county five-fold, strict mask | 1,682 | 0.680 | 2,545.6 | 3,450.1 | 15.10% | -264.4 |
| 2020 leave-county five-fold, broad mask | 1,682 | 0.697 | 2,497.5 | 3,358.1 | 14.96% | -230.8 |
| 2020 leave-province, strict mask, before calibration | 1,682 | 0.463 | 3,168.8 | 4,470.4 | 18.87% | -526.8 |
| 2020 leave-province, strict mask, after calibration | 1,682 | 0.670 | 2,532.3 | 3,500.9 | 15.16% | -323.1 |
| 2019–2023 leave-one-year-out, strict mask, before calibration | 8,275 | 0.685 | 2,783.6 | 3,762.2 | 15.35% | -315.3 |
| 2019–2023 leave-one-year-out, strict mask, after calibration | 8,275 | 0.751 | 2,511.2 | 3,343.6 | 13.97% | -336.0 |

Source: `tables/validation_summary.csv`.
