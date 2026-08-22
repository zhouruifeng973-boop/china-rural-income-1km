# v1.5-no-road release notes

## Canonical workflow change

- Removed `road_density`, which was derived from an OpenStreetMap snapshot obtained in 2026 and therefore post-dated the 2019–2023 study period.
- Retrained the canonical model and five-member ensemble without the road feature.
- Regenerated the five annual income maps and two 2019–2023 change maps.
- Retained 27 spatial covariates plus the scaled year variable.

## Validation change

- Added five-fold GroupKFold validation over all 8,275 county-year observations from 2019–2023.
- Province is the grouping variable; all years from each held-out province stay in the same test fold.
- Model-only spatial-transfer result before province-year calibration: R² = 0.398; MAPE = 19.64%.
- Operational product result after official province-year calibration: R² = 0.678; MAPE = 15.93%.
- The calibrated result is a product-level diagnostic and is not presented as fully independent validation or pixel-level accuracy.

## Public repository boundary

- Added only aggregate validation results and audit records.
- County-level labels, held-out county predictions, residual records, administrative assignment boundaries, raw workbooks and other restricted inputs remain excluded.
- Both the multi-year fold audit and the product-calibration audit report `PASS`.

## Frozen data package

- Formal package name: `china_rural_income_1km_2019_2023_v1.5-no-road.zip`.
- Package inventory: 111 files, including 67 GeoTIFFs.
- Added aggregate multi-year leave-province metrics at overall, fold, year and fold-by-year levels.
- Added held-out province assignments, province-year constraint comparisons, the versioned workflow configuration and explicit random seeds.
- Formal manifest: `metadata/v1_5_no_road_manifest.json`.
- Reserved dataset DOI: `10.5281/zenodo.21711838`.
