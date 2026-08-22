# v1.5 no-road validation records

This directory contains public-safe aggregate diagnostics for the canonical v1.5 workflow. County-level held-out predictions are deliberately excluded because they would reproduce non-redistributed county training labels.

## Road-feature removal

`NO_ROAD_SENSITIVITY_ASSESSMENT.md` and the associated CSV/JSON files compare the historical road-enabled experiment with the regenerated no-road workflow. The canonical v1.5 model, annual maps, change maps and ensemble exclude `road_density` because the available OpenStreetMap snapshot dates to 2026, after the 2019–2023 study period.

## Multi-year leave-province validation

All 8,275 available county-year observations from 2019–2023 were assigned to five folds using province as the grouping variable. Every province's observations across all years remain in one test fold. The pre-calibration diagnostic measures model-only spatial transfer; the post-calibration diagnostic applies the official province-year constraint used by the released product.

| Result | R² | MAPE |
|---|---:|---:|
| Model-only, before province-year calibration | 0.398 | 19.64% |
| Operational product, after province-year calibration | 0.678 | 15.93% |

The after-calibration result is not fully independent of official province statistics and does not establish pixel-level accuracy. Province-year scaling does not change within-province county ranks. Both audit JSON files report `PASS`.

## Publication boundary

Included here: aggregate overall, year and fold metrics; calibration summaries; audit reports.

Excluded here: county-level predictions, residuals, county income labels, administrative assignment boundaries and other restricted inputs.
