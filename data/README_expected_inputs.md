# Expected local inputs

The repository intentionally contains no research input data. Users must supply inputs locally and review the paths expected by `config/workflow_v1.5_no_road_core.json`. Historical configurations are retained for audit purposes.

Input categories include:

- author-compiled county-level rural disposable-income labels used for weak supervision, with a province-year statistical-yearbook source crosswalk;
- province-level rural disposable income and CPI statistics;
- WorldPop population rasters;
- GHS-SMOD settlement classes;
- annual remote-sensing, climate, terrain, land-cover and nighttime-light covariates;
- administrative geometries or a county-ID raster used for aggregation.

Purchased or licence-restricted inputs must not be committed to this repository. Run `scripts/check_reproducibility_inputs.py` before executing the full workflow.

For v1.5, the configuration expects the locally supplied versioned target `data/corrections/county_income_2019_2023_deflated_official77_corrected_v4.csv`. Its filename and expected role are documented for reproducibility, but its records are not redistributed. The public release ZIP contains only derived gridded products, aggregate validation information and metadata.

The final v1.5 workflow explicitly excludes `road_density`. The available OpenStreetMap road snapshot dates to 2026 and is therefore not required or permitted as an input for the 2019–2023 canonical model.
