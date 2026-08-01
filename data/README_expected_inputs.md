# Expected local inputs

The repository intentionally contains no research input data. Users must supply inputs locally and update the paths expected by `config/workflow_v1.3_official44_corrected.yaml` (or use the retained historical `workflow_v1.0_draft.yaml`).

Input categories include:

- author-compiled county-level rural disposable-income labels used for weak supervision, with a province-year statistical-yearbook source crosswalk;
- province-level rural disposable income and CPI statistics;
- WorldPop population rasters;
- GHS-SMOD settlement classes;
- annual remote-sensing, climate, terrain, land-cover, road and nighttime-light covariates;
- administrative geometries or a county-ID raster used for aggregation.

Purchased or licence-restricted inputs must not be committed to this repository. Run `scripts/check_reproducibility_inputs.py` before executing the full workflow.

For v1.3, the configuration expects the locally supplied versioned target `data/corrections/county_income_2019_2023_deflated_official44_corrected_v3.csv`. Its filename and expected role are documented for reproducibility, but its records are not redistributed. The public release ZIP contains only derived gridded products, aggregate validation information and metadata.
