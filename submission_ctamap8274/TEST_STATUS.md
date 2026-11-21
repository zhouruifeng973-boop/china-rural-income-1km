# Test status 2026-11-21

Independent Windows CPython 3.11 venv installed without system site packages.
pip check passed. Newly generated synthetic data: 20 counties, five provinces,
five years, 100 county-year units, 27 predictors plus year, 128/64 MLP, 150 epochs.

Passed: feature alignment, population weighting, province CSV conversion, panel
creation, legacy/new feature compatibility, missing-feature rejection, county
encoding equivalence and wrong-pixel rejection, final model fit/save/reload,
five annual predictions, calibrated weighted means (0.01 yuan readback tolerance),
NoData, change-map writing, five leave-year folds, five single-year province folds,
five multi-year province folds, five ensemble models and 15 ensemble rasters.

Actual-input checks: rebuilt CTAmap has zero different county pixels; public
province tables convert to 155 rows. Earlier cropped quality/risk and bounded
change adapter tests passed. Synthetic outputs do not replace scientific results.

Not established: remote Earth Engine execution, historical source byte identity,
full national CLI reproduction, or clean-environment four-variant sensitivity
orchestration. This is production-function integration, not full reproduction.
