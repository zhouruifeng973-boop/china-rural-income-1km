# Reproducible workflow for the 1 km rural-income dataset of China, 2019–2023

This repository contains the code, configuration and documentation used to generate and technically validate a 1 km gridded dataset of rural residents' per capita disposable income in mainland China for 2019–2023.

## Scope

The workflow combines county-level weak supervision, remote-sensing and spatial covariates, rural-population-weighted county aggregation, province-year calibration, change analysis, quality layers and five-member ensemble uncertainty. Income values are expressed in 2019 constant Chinese yuan per person per year.

## Reproducibility boundary

The public code repository does not contain the author-compiled county-level income table, purchased administrative boundaries, raw National Bureau of Statistics workbooks, raw OpenStreetMap PBF files, or other third-party inputs with unresolved redistribution rights. Full raw-input reproduction requires the province-year statistical-yearbook source crosswalk and locally supplied boundary data. Public-product auditing and downstream reuse do not require those inputs.

## Installation

```bash
python -m venv .venv
python -m pip install -r requirements_reproducibility.txt
```

## Safe starting point

The pipeline entry point runs in dry-run mode unless `--execute` is supplied:

```bash
python scripts/check_reproducibility_inputs.py
python scripts/run_reproducible_pipeline.py
```

Before an actual model run, review `config/workflow_v1.0_draft.yaml` and the expected-input documentation.

## Main stages

1. Statistical and spatial preprocessing.
2. Weakly supervised downscaling and county-level validation.
3. Leave-province and leave-year validation.
4. Province-level calibration.
5. Annual and change-map generation.
6. Quality and uncertainty products.
7. Public-package construction and audit.

An optional internal sensitivity script tests the treatment of 90 audited 2022 income records. It requires the non-redistributed county-level income inputs and does not expose those records in the public code package.

## Data product

The corresponding gridded data product has been uploaded to a restricted Zenodo draft under the following reserved DOI:

`https://doi.org/10.5281/zenodo.21711838`

The DOI is not registered and the dataset is not publicly accessible until the Zenodo record is published after the remaining third-party-rights review.

## Licence and citation

The original project code is released under the MIT License. The author order is Shaohua Wu (corresponding author) and Ruifeng Zhou. Ruifeng Zhou's ORCID is https://orcid.org/0009-0009-9678-2927; Shaohua Wu does not currently have an ORCID identifier. The source repository is https://github.com/zhouruifeng973-boop/china-rural-income-1km and remains private during submission preparation. The valid draft citation metadata is stored in `CITATION.cff`; an archived code DOI must be added after the code release by using `CITATION.cff.template` as a reminder.
