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

## Pending-label sensitivity

The `pending_533/` directory contains aggregate fixed-fold results comparing the frozen all-label model with a verified-only model that excludes the 533 records still pending numeric verification. Both variants are scored on the same verified evaluation set. County-level predictions and income values are not included.

## County-uniform and parameter sensitivity

The `spatial_robustness/` directory contains the county-uniform baseline summary and aggregate reruns of the v1.5 no-road workflow across regularisation coefficients 0.0001, 0.001 and 0.01 and per-county pixel caps 64, 128 and 256. Annual and change-map comparisons are reported as aggregate statistics only. The experiments assess robustness, not independent 1 km accuracy.

## Publication boundary

Included here: aggregate overall, year and fold metrics; calibration summaries; audit reports; held-out province assignments; pending-label sensitivity; county-uniform and parameter-sensitivity summaries; the versioned run configuration and random-seed record.

Excluded here: county-level predictions, residuals, county income labels, administrative assignment boundaries and other restricted inputs.
