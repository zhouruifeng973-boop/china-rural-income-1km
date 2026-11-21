# CTAmap 8274 reproducibility workflow

Use this directory at the manuscript's fixed commit, not legacy root scripts.
Data version: 20261119-ctamap8274; reserved DOI: 10.5281/zenodo.21711838.
Code license: MIT. Data licenses are separate. Scientific products are unchanged.

## Input preparation

Replace uppercase paths below. Outputs must be new files/directories.

1. Extract the review package as PACKAGE; use its final 8274-row training table.
2. Rebuild county IDs:

```text
python rebuild_county_id.py --boundary PACKAGE/reproducibility/boundary/T2021_county.gpkg --grid PACKAGE/reproducibility/boundary/grid.json --output FEATURES/county_id.tif
```

3. Run gee_export_features.js in Earth Engine after uploading the licensed
boundary as your own asset and replacing YOUR_PROJECT. Start export tasks and
download all feature files to EXPORTS. Source scaling is retained. Historical
end-exclusive date cutoffs are preserved. Remote Earth Engine execution was not
tested here; access authorization and upstream collection availability are required.
4. Run gee_export_smod.js and obtain the raw 2020 smod_code export. Obtain the
matching annual WorldPop population-count files specified in the archive's source
documentation. Do not substitute population density or another product variant.
5. Align features and build population weights (repeat weights for 2019–2023):

```text
python prepare_inputs.py features --county-id FEATURES/county_id.tif --exports EXPORTS --output FEATURES
python prepare_inputs.py weights --county-id FEATURES/county_id.tif --smod SMOD.tif --population WORLDPOP_2019.tif --year 2019 --output PROCESSED/rural_weights
python prepare_inputs.py province --package PACKAGE --output PROCESSED/province_income_2019_2023_deflated.csv
python prepare_public_panels.py --package PACKAGE --features FEATURES --weights PROCESSED/rural_weights --county-id FEATURES/county_id.tif --prepare --output PANELS
```

Features use bilinear alignment; classification uses nearest neighbor; population
counts use sum resampling, classes 11/12/13 and positive county IDs. Invalid or
negative population is zeroed. Province conversion copies official targets from
the five archived calibration tables, without re-deflation. Panel preparation
applies source scaling once and writes 27 spatial columns; year is added later.
County validation compares fresh boundary rasterization pixel by pixel, including
grid/CRS/NoData, rather than requiring identical TIFF compression.

## Training and products

```text
python run_submission.py --stage validation --package PACKAGE --panels PANELS --features FEATURES --processed PROCESSED --output NEW_RUN --execute
```

Repeat with stages multiyear, core, products, ensemble. Core precedes products
and ensemble. Use the same NEW_RUN; existing stage outputs are rejected. Default
mode only checks inputs. Ensemble reporting requires inputs/outputs on one drive.
Additional adapters: run_auxiliary.py, create_bounded_changes.py,
run_uniform_baseline.py and run_sensitivity.py; inspect --help for explicit paths.
No existing national result is overwritten by default.

## Independent software test

```text
python -m venv .venv
.venv/Scripts/python.exe -m pip install numpy==2.2.6 pandas==2.3.3 rasterio==1.4.4 scikit-learn==1.7.2 torch==2.9.1 geopandas==1.1.2
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe test_generated_pipeline.py --output NEW_TEST_DIRECTORY
```

Tested with Windows CPython 3.11.9. The test generates source rasters, boundaries
and labels from scratch; it does not load author panels or trained models.
It exercises 150-epoch training, five-year prediction, calibration, changes,
leave-year/province functions and five-member ensemble. See TEST_STATUS.md.
Synthetic metrics are not scientific evidence. Full national CLI reproduction,
remote exports and the complete sensitivity launcher remain unverified.
