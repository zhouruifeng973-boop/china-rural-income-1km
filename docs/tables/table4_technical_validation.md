# Table 4. Technical-validation metrics

**Table 4. County-level validation metrics after rural-population-weighted aggregation.** Pixel predictions were aggregated back to counties using rural-population weights and compared with observed county-level rural disposable income. MAE, RMSE, and bias are in 2019 constant yuan per person per year; MAPE is a percentage. These metrics do not represent pixel-level validation. Province calibration is a product-generation constraint, so after-calibration results describe the constrained product rather than fully independent validation.

| Validation experiment | n | R² | MAE | RMSE | MAPE | Bias |
|---|---:|---:|---:|---:|---:|---:|
| 2020 five-fold leave-province, no-road, before calibration | 1,682 | 0.442 | 3,292.6 | 4,555.6 | 19.72% | -616.3 |
| 2019–2023 leave-one-year-out, no-road, before calibration | 8,275 | 0.689 | 2,789.8 | 3,734.4 | 15.49% | -290.8 |
| 2019–2023 five-fold GroupKFold by province, no-road, before calibration | 8,275 | 0.398 | 3,661.4 | 5,194.3 | 19.64% | -894.5 |
| 2019–2023 five-fold GroupKFold by province, no-road, after operational province-year calibration | 8,275 | 0.678 | 2,859.3 | 3,798.6 | 15.93% | -420.6 |

Sources: `docs/validation/v1.5_no_road/` and the aggregate validation tables in the formal v1.5 no-road package. The after-calibration row is a final-product workflow diagnostic, not fully independent validation.
