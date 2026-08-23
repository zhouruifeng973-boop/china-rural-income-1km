# Reproducible workflow for the 1 km rural-income dataset of China, 2019–2023

Current formal release: **v1.5-no-road**.

This repository contains the code, configuration and documentation used to generate and technically validate a 1 km gridded dataset of rural residents' per capita disposable income in mainland China for 2019–2023.

## Scope

The workflow combines county-level weak supervision, remote-sensing and spatial covariates, rural-population-weighted county aggregation, province-year calibration, change analysis, quality layers and five-member ensemble uncertainty. Income values are expressed in 2019 constant Chinese yuan per person per year. The canonical v1.5 model uses 27 spatial covariates plus a scaled year variable and excludes `road_density`.

## Resolution of post-period road-data leakage

An earlier experimental workflow used a road-density layer derived from an OpenStreetMap snapshot obtained in 2026. Because that snapshot post-dates the 2019–2023 study period, v1.5 removes `road_density` from model training, leave-year and leave-province validation, nationwide prediction, and the five-member ensemble. The canonical annual maps and change products were regenerated without this feature. The historical road-enabled scripts and configuration remain only for provenance and are not the recommended workflow.

## Reproducibility boundary

The public code repository does not contain the author-compiled county-level income table, non-redistributed administrative assignment boundaries, raw National Bureau of Statistics workbooks, or other third-party inputs with unresolved redistribution rights. The 2026 OpenStreetMap snapshot and its derived road-density raster are neither redistributed nor used by the v1.5 workflow. Full raw-input reproduction requires the province-year statistical-yearbook source crosswalk and locally supplied boundary data. Public-product auditing and downstream reuse do not require those inputs.

## Installation

```bash
python -m venv .venv
python -m pip install -r requirements_reproducibility.txt
```

## Safe starting point

Both pipeline entry points run in dry-run mode unless `--execute` is supplied:

```bash
python scripts/check_reproducibility_inputs.py
python scripts/run_v1_5_no_road_pipeline.py
```

Before an actual v1.5 model run, review `config/workflow_v1.5_no_road_core.json` and the expected-input documentation. The corrected county-level target named by that configuration is a locally supplied restricted input and is deliberately absent from this repository.

After checking all local inputs, execute the complete workflow with:

```bash
python scripts/run_v1_5_no_road_pipeline.py --execute
```

## Publication-map boundaries

Figures 2, 3 and 5 use the 2019 China ADM0 and ADM1 layers from geoBoundaries only as cartographic context. Download the `gbOpen` China ADM0 and ADM1 all-format archives and extract the shapefiles to:

```text
data/public_boundaries/geoboundaries_CHN_2019/ADM0/geoBoundaries-CHN-ADM0.shp
data/public_boundaries/geoboundaries_CHN_2019/ADM1/geoBoundaries-CHN-ADM1.shp
```

The source boundary archives are deliberately not committed. The display layers are independent of the restricted county geometry used for modelling, county aggregation and calibration. The released product covers mainland China; Hong Kong, Macao and Taiwan are outside product coverage. No South China Sea or nine-dash-line layer is used in these figures.

Boundary citation: Runfola, D. et al. *geoBoundaries: A global database of political administrative boundaries*. PLoS ONE 15, e0231866 (2020). https://doi.org/10.1371/journal.pone.0231866.

## Main stages

1. Statistical and spatial preprocessing.
2. Weakly supervised downscaling and county-level validation.
3. Leave-province and leave-year validation.
4. Province-level calibration.
5. Annual and change-map generation.
6. Quality and uncertainty products.
7. Public-package construction, raster/product audit and reproducibility audit.
8. Deterministic ZIP construction followed by an independent extraction and SHA-256 audit.

## Multi-year spatial-transfer validation

The v1.5 workflow adds a five-fold GroupKFold experiment using all available 2019–2023 county-year observations. Province is the grouping variable, so all five years from a held-out province remain in the same test fold and no county label from that province enters model fitting.

| Diagnostic | R² | MAPE | Interpretation |
|---|---:|---:|---|
| Before province-year calibration | 0.398 | 19.64% | Independent spatial transfer of the no-road model |
| After operational province-year calibration | 0.678 | 15.93% | Final-product workflow diagnostic; not fully independent validation |

The after-calibration result uses only the official province-year constraint available in the operational product workflow; it does not use held-out county labels to calculate the calibration factor. Positive within-province scaling leaves county ranks unchanged. Neither result constitutes pixel-level validation. Public-safe aggregate results and audit reports are stored under `docs/validation/v1.5_no_road/`; record-level held-out county predictions are not redistributed.

The Stage 17–18 reconciliation scripts document how selected official-yearbook checks were applied without embedding the checked income values. They require non-redistributed local workbooks and correction tables. The public repository therefore supports method inspection and product auditing, while a complete from-raw-data rerun remains conditional on access to the restricted inputs.

## Training-label provenance and pending-value sensitivity

The public-safe provenance index maps all 8,373 county-year training records to 95 source items without redistributing county income values. Forty source items have an exact table or page-level locator and 55 are transparently identified by the available bibliographic title or local source name because a more precise locator could not be recovered. Of the 8,373 records, 7,840 have completed county-name and numeric verification and 533 remain flagged as pending numeric verification.

A fixed-fold sensitivity analysis excluded those 533 records from fitting and evaluated both variants on the same 7,753 verified observations with valid pixel panels. Relative to the all-label baseline, R² changed by -0.0099 and MAPE by +0.183 percentage points; paired predictions correlated at r = 0.99787. Aggregate evidence is stored under `docs/validation/v1.5_no_road/pending_533/`, and the public-safe source crosswalk is under `docs/provenance/v2.4/`.

## County-uniform baseline and parameter sensitivity

The county-uniform baseline preserves every county's population-weighted mean while removing within-county variation. The canonical surface has a median within-county population-weighted coefficient of variation of 24.55–25.52% across 2019–2023. This quantifies the spatial structure added by the model but is not independent 1 km accuracy validation.

The no-road workflow was rerun with output-regularisation coefficients of 0.0001, 0.001 and 0.01 and per-county pixel-sampling caps of 64, 128 and 256, using fixed province folds and random seeds. Across tested configurations, the maximum absolute change in R² was 0.0212 and the maximum absolute change in MAPE was 0.398 percentage points. Annual-map spatial agreement remained high, but within-county ordering and especially 2019–2023 change patterns were more sensitive to the pixel-sampling cap. These results support county-level robustness while requiring cautious interpretation of individual pixels and local change estimates. Public aggregate results and manifests are stored under `docs/validation/v1.5_no_road/spatial_robustness/`; variant rasters and county-level predictions are excluded.

## Data product

The corresponding formal frozen gridded-data package has been deposited in a restricted Zenodo peer-review record under the following reserved DOI:

`https://doi.org/10.5281/zenodo.21711838`

The final peer-review package contains 155 files, including 67 GeoTIFFs and aggregate evidence for multi-year leave-province validation, province-year-constrained diagnostics, training-label sensitivity, the county-uniform baseline and parameter sensitivity. The frozen ZIP is named `china_rural_income_1km_2019_2023_v1.5-no-road.zip` and has SHA-256 `da380c406d04696cf988cc6f6000ab8f21b6f0ebb40e6ffe0763d942e98dbf21`. The DOI will be registered and the record made public when the Zenodo draft is formally published.

## Licence and citation

The original project code is released under the MIT License. The author order is Shaohua Wu (corresponding author) and Ruifeng Zhou. Ruifeng Zhou's ORCID is https://orcid.org/0009-0009-9678-2927; Shaohua Wu does not currently have an ORCID identifier. The source repository is https://github.com/zhouruifeng973-boop/china-rural-income-1km. The valid citation metadata is stored in `CITATION.cff`; an archived code DOI must be added after the code release by using `CITATION.cff.template` as a reminder.
