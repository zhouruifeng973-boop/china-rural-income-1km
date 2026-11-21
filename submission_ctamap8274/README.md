# CTAmap 8,274-record code candidate

This is a local publication candidate, not a published or fully reproduced release.
Data version: 20261119-ctamap8274. Reserved data DOI: 10.5281/zenodo.21711838.
Code: MIT. Data and boundary licenses remain separate.

## Inputs

Extract the corresponding Zenodo review package. Its final label table is
training_labels/county_training_labels_8274.csv. Do not reconstruct the cohort
from old predictions or apply the county recodes a second time.
The package contains CTAmap boundary and grid metadata; use rebuild_county_id.py
with --boundary, --grid, --output to create a NEW county-ID raster.
Other required inputs are the prepared feature GeoTIFFs, yearly strict rural
population weights and processed/province_income_2019_2023_deflated.csv.
These inputs are NOT all included in the Zenodo product package.
Feature filenames, band order and scaling are specified in prepare_public_panels.py.
Python dependencies: numpy, pandas, rasterio, scikit-learn, torch, geopandas, affine;
exact environment pinning and a fresh-environment installation test remain pending.

## Safe checks (replace CAPITALIZED path placeholders)

python check_submission_inputs.py --package PACKAGE
python prepare_public_panels.py --package PACKAGE --features FEATURES --weights WEIGHTS --county-id COUNTY_ID

The second command only checks inputs by default. Add --prepare --output NEW_PANELS
to build large panel files. It preserves the fixed labels and excludes roads.

python run_submission.py --stage validation --package PACKAGE --panels PANELS --features FEATURES --processed PROCESSED --output NEW_CHECK_DIR --check-parsers
python run_submission.py --stage validation --package PACKAGE --panels PANELS --features FEATURES --processed PROCESSED --output NEW_SMOKE_DIR --smoke

Parser checks exercise all four parsers and product paths. Smoke mode trains
only two epochs on 4,096 pixels from partial county bags, writes test-only CSVs,
and is NOT a scientific validation.

Add --execute to run the selected full stage explicitly. Run validation,
multiyear, core, products, and ensemble as separate commands, using the same
output directory for core/products/ensemble. Existing stage outputs are rejected;
there is no automatic overwrite or resume after a partial failure. Input path,
label and frozen-parameter identity are checked across stages. Core must precede
products and ensemble. A full stage can require substantial time, RAM and disk.
This execution dispatch is implemented but has NOT yet passed a complete run.
Ensemble inputs and output must be on the same filesystem drive for provenance
reporting; differing drives fail rather than silently redirect output.

requirements_observed_windows_py311.txt records versions observed on the test
machine. It is not a fully resolved dependency lock or a fresh-install guarantee.
Do not run scripts directly using their historical default paths/configuration.

## Quality and risk layers

python run_auxiliary.py --package PACKAGE --core CORE --features FEATURES --output NEW_AUXILIARY_OUTPUT

This defaults to read-only checks of the frozen labels, required files and grids.
Add --execute to generate 26 quality and 12 risk layers in a NEW directory.
This auxiliary adapter has passed preflight only, not a complete output run.
It does not generate county-uniform baseline maps or bounded change maps.
Parameter-sensitivity orchestration is available through run_sensitivity.py.
Supply --package --panels --features --processed --core --cv --config --output.
The config must contain final_maps and multiyear_leave_province_validation
settings matching the frozen baseline; parser-check workflow.json is suitable.
Default mode checks the final label hash, cohort size, baseline settings and 21
required input files. --execute runs four CV/map variants and their comparisons.
The internal county-ID and population-weight comparison paths now use the
explicit arguments. This adapter passed preflight only, not a full run.
Decision-narrative generation is excluded from the sensitivity adapter.

## County-uniform baseline and bounded change maps

python run_uniform_baseline.py --core CORE --county-id COUNTY_ID --weights WEIGHTS --output NEW_BASELINE_OUTPUT

Default mode checks inputs and grids. --execute writes baseline maps and reads
them back to verify county-constant values, NoData and population-weighted means
(absolute tolerance 0.01 yuan). --years can select a subset. A cropped test passed
for 2019 and 2023, not a full national rerun.

python create_bounded_changes.py --first BOUNDED_2019 --last BOUNDED_2023 --output NEW_CHANGE_OUTPUT

Default mode scans input grids, 3000–80000 bounds and common valid pixels.
--execute creates two float32 change maps, verifies formulas by readback and
checks input hashes remain unchanged. The full-input preflight passed, with
4,034,095 common valid pixels. This adapter has not yet executed an output test.

## What has and has not been tested

Passed locally: final package hashes/cohort; five-year grid and coverage checks;
sampled feature equivalence; two-epoch numerical test; all stage parser wiring;
cropped production calibration and change GeoTIFF readback.
Not completed: execution of all stages through the portable launcher, full
independent end-to-end reproduction, fresh-environment dependency validation,
full execution of the auxiliary-quality and sensitivity adapters, GitHub publication.
Historical development metadata legitimately records 8,275 units; final training
uses 8,274. The frozen development hyperparameters are unchanged.

Local tests and author machine paths are intentionally not bundled. No model,
raw training table, geographic source data or anonymous-review token is copied.


## Current test-status update (2026-11-21)

See TEST_STATUS.md and candidate_status.json for the latest checks. Bounded
change and quality/risk output functions now passed cropped execution tests;
earlier preflight-only descriptions refer to the earlier preparation stage.
Full national execution and clean-environment reproduction remain unverified.
This directory is a candidate, not a claim that all submission blockers are closed.
