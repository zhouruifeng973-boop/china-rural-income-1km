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

- Added only aggregate validation results, provenance metadata and audit records.
- County-level labels, held-out county predictions, residual records, administrative assignment boundaries, raw workbooks and other restricted inputs remain excluded.
- Both the multi-year fold audit and the product-calibration audit report `PASS`.

## Provenance and robustness evidence

- Added a public-safe 8,373-row record-to-source crosswalk and a 95-item source catalogue without county income values.
- Retained transparent bibliographic-title or local-source-name disclosure for 55 source items whose exact table or URL locator could not be recovered.
- Added the fixed-fold experiment excluding 533 records pending numeric verification. R² changed by -0.0099, MAPE by +0.183 percentage points and paired predictions correlated at r = 0.99787.
- Added the county-uniform baseline, which preserves county population-weighted means and removes within-county variation.
- Added no-road parameter sensitivity for regularisation values 0.0001, 0.001 and 0.01 and per-county sampling caps 64, 128 and 256.
- County-level validation was stable across tested parameters, while within-county and change-map patterns were more sensitive to the sampling cap; the evidence is not described as independent pixel-level validation.

## Frozen data package

- Formal package name: `china_rural_income_1km_2019_2023_v1.5-no-road.zip`.
- Package inventory: 155 files, including 67 GeoTIFFs.
- Added aggregate multi-year leave-province metrics at overall, fold, year and fold-by-year levels.
- Added held-out province assignments, province-year constraint comparisons, the versioned workflow configuration and explicit random seeds.
- Formal manifest: `metadata/v1_5_no_road_manifest.json`.
- ZIP SHA-256: `da380c406d04696cf988cc6f6000ab8f21b6f0ebb40e6ffe0763d942e98dbf21`.
- Reserved dataset DOI: `10.5281/zenodo.21711838`.
