# Reproducible workflow for the 1 km rural-income dataset of China, 2019–2023

Current release candidate: **v1.5-no-road**.

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

## Data product

The corresponding gridded data product has been uploaded to a restricted Zenodo draft under the following reserved DOI:

`https://doi.org/10.5281/zenodo.21711838`

The DOI is not registered and the dataset is not publicly accessible until the Zenodo record is published after the remaining third-party-rights review.

## Licence and citation

The original project code is released under the MIT License. The author order is Shaohua Wu (corresponding author) and Ruifeng Zhou. Ruifeng Zhou's ORCID is https://orcid.org/0009-0009-9678-2927; Shaohua Wu does not currently have an ORCID identifier. The source repository is https://github.com/zhouruifeng973-boop/china-rural-income-1km. The valid citation metadata is stored in `CITATION.cff`; an archived code DOI must be added after the code release by using `CITATION.cff.template` as a reminder.
