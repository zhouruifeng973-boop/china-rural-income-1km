# Table 2. Released data records

**Table 2. Summary of released data-record groups.** Primary raster counts exclude auxiliary overview and XML sidecar files. All GeoTIFF products share the same 1 km analysis grid, spatial extent, and EPSG:4326 coordinate reference system. The canonical annual products use the strict rural mask, unbounded predictions, province-level calibration, and 2019 constant yuan.

| Folder | Product group | Primary GeoTIFFs | All files in folder | Format | Unit | NoData | Recommended use |
|---|---|---:|---:|---|---|---|---|
| data/income_maps/ | Canonical annual income maps | 5 | 15 | GeoTIFF | 2019 constant yuan person⁻¹ year⁻¹ | −9999 | Primary 2019–2023 rural-income surfaces; strict mask, unbounded, province calibrated. |
| data/change_maps/ | 2019–2023 change maps | 2 | 6 | GeoTIFF | 2019 constant yuan person⁻¹ year⁻¹; % | −9999 | Absolute and percentage change between the canonical 2019 and 2023 products. |
| data/sensitivity_bounded_maps/ | Bounded sensitivity products | 7 | 12 | GeoTIFF | 2019 constant yuan person⁻¹ year⁻¹; % | −9999 | Sensitivity analysis under the 3,000–80,000 yuan bounding rule; not the canonical product. |
| data/quality_layers/ | Quality-support layers | 26 | 27 | GeoTIFF | 0/1; factor; class; 2019 constant yuan | 255 or −9999 | Valid-pixel masks, training-province support, calibration factors/classes, and bounding sensitivity. |
| data/uncertainty_proxy_layers/ | Quality-risk proxy layers | 12 | 13 | GeoTIFF | 0–100 score; class | 255 or −9999 | Annual score/class layers and five-year mean/maximum scores; interpretation aid, not model SD. |
| data/ensemble_uncertainty_layers/ | Multi-seed ensemble layers | 15 | 16 | GeoTIFF | 2019 constant yuan; % | −9999 | Annual ensemble mean, standard deviation, and coefficient of variation from five members. |
| **Total raster data** | **Six raster product groups** | **67** | **89** | **GeoTIFF and sidecars** | — | — | Use the data dictionary to select the appropriate product. |

Supporting records: `tables/` contains 22 files (21 CSV, 1 MD); `metadata/` contains 12 files (10 CSV, 1 MD, 1 TXT); and the release root contains 9 Markdown documentation/version files. The complete package contains 132 files.

The release manifest records three county-level training/residual files as `exclude`; the files are absent from the public package as required by the release policy.

Source: `metadata/public_release_manifest_v1.0_draft.csv`, `metadata/data_dictionary.csv`, and the frozen public-release directory.
