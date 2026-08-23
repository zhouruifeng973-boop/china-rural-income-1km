# Administrative-boundary reproducibility limitation

The data-production workflow used one fixed 2020 county-boundary framework for all 2019–2023 study years. It determines pixel-to-county assignment, county population-weighted aggregation, weak-supervision units and province-grouped validation and is therefore a critical computational input, although it is not a model predictor.

The source page for this boundary did not provide a standardised open-licence statement. The source vectors, county-ID raster and boundary-derived intermediate files are not redistributed. The public repository and archive therefore do not support end-to-end reconstruction from the original administrative-boundary input alone. Public code can reproduce the subsequent preprocessing, modelling, validation, calibration and audit stages after the user supplies an equivalent county-to-pixel assignment under a licence available to that user. Exact numerical identity is not guaranteed when a different boundary vintage or geometry is substituted.

geoBoundaries ADM0/ADM1 layers used in manuscript figures are cartographic backgrounds only. They did not determine training units, aggregation or provincial calibration.
