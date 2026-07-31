# Expected local inputs

The repository intentionally contains no research input data. Users must supply inputs locally and update the paths expected by `config/workflow_v1.0_draft.yaml`.

Input categories include:

- author-compiled county-level rural disposable-income labels used for weak supervision, with a province-year statistical-yearbook source crosswalk;
- province-level rural disposable income and CPI statistics;
- WorldPop population rasters;
- GHS-SMOD settlement classes;
- annual remote-sensing, climate, terrain, land-cover, road and nighttime-light covariates;
- administrative geometries or a county-ID raster used for aggregation.

Purchased or licence-restricted inputs must not be committed to this repository. Run `scripts/check_reproducibility_inputs.py` before executing the full workflow.
