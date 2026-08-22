#!/usr/bin/env python3
"""Generate Scientific Data Table 1 from the frozen source-data inventories."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = (
    ROOT
    / "outputs"
    / "public_release"
    / "china_rural_income_1km_2019_2023_v1.0-draft"
)
PUBLIC_INVENTORY = RELEASE / "metadata" / "source_data_inventory_public_sources_filled_draft.csv"
SCIENTIFIC_INVENTORY = RELEASE / "metadata" / "source_data_inventory_scientific_data_draft.csv"
TABLE_OUT = ROOT / "docs" / "tables" / "table1_input_datasets.md"
QA_OUT = ROOT / "docs" / "tables" / "table1_input_datasets_qa.json"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


ROWS = [
    {
        "dataset": "County-level rural income labels",
        "provider": "Compiled by the authors from provincial statistical yearbooks and officially published local statistical materials",
        "period_resolution": "2019–2023; county-year records",
        "role": "Weak-supervision targets for model fitting and county-level validation",
        "access_terms": "Cite the exact yearbook edition, table and page/source mapping; provenance crosswalk pending completion",
        "redistribution": "Author-compiled raw table and reconstructable residuals are withheld until the yearbook-source crosswalk is complete",
        "status": "provenance_pending",
    },
    {
        "dataset": "Provincial rural residents' per capita disposable income",
        "provider": "National Bureau of Statistics of China, National Data",
        "period_resolution": "2019–2023; province-year statistics",
        "role": "Province-year calibration constraint",
        "access_terms": "Official National Data table; cite the National Bureau of Statistics and access date (Ref. 1)",
        "redistribution": "No raw workbook; derived deflated values and calibration factors only",
        "status": "redistribution_caution",
    },
    {
        "dataset": "Provincial Consumer Price Index (previous year = 100)",
        "provider": "National Bureau of Statistics of China, National Data",
        "period_resolution": "2000–2024; 2019–2023 used; province-year statistics",
        "role": "Deflation of nominal income to 2019 constant yuan",
        "access_terms": "Official National Data table; cite the National Bureau of Statistics and access date (Ref. 2)",
        "redistribution": "No raw workbook; derived price indices and transformation metadata only",
        "status": "redistribution_caution",
    },
    {
        "dataset": "WorldPop Global 2 constrained population, R2025A UA v1",
        "provider": "WorldPop, University of Southampton",
        "period_resolution": "2019–2023; 1 km annual rasters",
        "role": "Rural-population weights and county/province aggregation",
        "access_terms": "CC BY 4.0; DOI: 10.5258/SOTON/WP00840 (Ref. 3)",
        "redistribution": "No raw input in this release; derived outputs retain attribution",
        "status": "frozen_public_terms",
    },
    {
        "dataset": "GHS-SMOD R2023A v2.0 Degree of Urbanisation",
        "provider": "European Commission Joint Research Centre, GHSL",
        "period_resolution": "2020; 1 km",
        "role": "Strict rural mask (classes 11/12/13) and broad-mask sensitivity (plus class 21)",
        "access_terms": "Open/free with acknowledgement; DOI: 10.2905/A0DF7A6F-49DE-46EA-9BDE-563437A6E2BA (Ref. 4)",
        "redistribution": "No raw input in this release; derived masks require acknowledgement",
        "status": "frozen_public_terms",
    },
    {
        "dataset": "VIIRS monthly stray-light-corrected DNB composites v1 (VCMSLCFG)",
        "provider": "Earth Observation Group, Colorado School of Mines",
        "period_resolution": "2019–2023; monthly, 463.83 m",
        "role": "Annual nighttime-light predictor of settlement and human activity",
        "access_terms": "Public domain; cite EOG and the product paper (Ref. 5)",
        "redistribution": "No raw input in this release; annual 1 km predictor retained only in the private workflow",
        "status": "frozen_public_terms",
    },
    {
        "dataset": "MOD13A3.061 Terra Vegetation Indices",
        "provider": "NASA LP DAAC at the USGS EROS Center",
        "period_resolution": "2019–2023; monthly, 1 km",
        "role": "Annual NDVI and EVI summary predictors",
        "access_terms": "No restrictions on subsequent use or redistribution; DOI: 10.5067/MODIS/MOD13A3.061 (Ref. 6)",
        "redistribution": "No raw input in this release; derived annual features used internally",
        "status": "frozen_public_terms",
    },
    {
        "dataset": "MOD17A3HGF.061 Terra annual gap-filled NPP",
        "provider": "NASA LP DAAC at the USGS EROS Center",
        "period_resolution": "2019–2023; annual, 500 m",
        "role": "Ecological-productivity predictor",
        "access_terms": "No restrictions on subsequent use or redistribution; DOI: 10.5067/MODIS/MOD17A3HGF.061 (Ref. 7)",
        "redistribution": "No raw input in this release; derived annual features used internally",
        "status": "frozen_public_terms",
    },
    {
        "dataset": "ESA WorldCover 2021 v200",
        "provider": "ESA WorldCover consortium",
        "period_resolution": "Static 2021 map; 10 m",
        "role": "Land-cover fractions for 1 km cells",
        "access_terms": "CC BY 4.0; DOI: 10.5281/zenodo.7254221 (Ref. 8)",
        "redistribution": "No raw input in this release; derived fractions require attribution",
        "status": "frozen_public_terms",
    },
    {
        "dataset": "ERA5-Land monthly aggregated reanalysis",
        "provider": "ECMWF / Copernicus Climate Change Service",
        "period_resolution": "2019–2023; monthly, approximately 11.1 km",
        "role": "Annual temperature, precipitation and solar-radiation predictors",
        "access_terms": "C3S/CAMS licence and acknowledgement; DOI: 10.24381/cds.68d2bb30 (Refs. 9, 10)",
        "redistribution": "No raw input in this release; derived features require Copernicus acknowledgement",
        "status": "frozen_public_terms",
    },
    {
        "dataset": "Harmonized Sentinel-2 MSI Level-2A surface reflectance",
        "provider": "European Union / ESA / Copernicus",
        "period_resolution": "2019–2023 summer composites; native bands at 10/20/60 m",
        "role": "Spectral bands and NDVI/NDWI/NDBI predictors",
        "access_terms": "Copernicus Sentinel Data Terms and Conditions; attribution required (Ref. 11)",
        "redistribution": "No raw input in this release; derived 1 km composites used internally",
        "status": "frozen_public_terms",
    },
    {
        "dataset": "NASA SRTMGL1 v003 digital elevation model",
        "provider": "NASA / USGS / JPL-Caltech",
        "period_resolution": "Static (February 2000 acquisition); 30 m",
        "role": "Elevation, slope and aspect predictors",
        "access_terms": "JPL public-use terms; cite Farr et al. (2007), DOI: 10.1029/2005RG000183 (Ref. 12)",
        "redistribution": "No raw input in this release; derived terrain variables used internally",
        "status": "frozen_public_terms",
    },
    {
        "dataset": "County/province/national administrative boundaries and county-ID raster",
        "provider": "Purchased/third-party boundary data; exact source/version documentation pending",
        "period_resolution": "Static vector boundaries; rasterised to the 1 km grid",
        "role": "Pixel-to-county assignment, aggregation and map context",
        "access_terms": "Restricted; exact source, citation, cartographic approval and permission must be confirmed",
        "redistribution": "No; source boundaries and reconstructable derivatives are excluded",
        "status": "restricted_pending",
    },
]


def main() -> None:
    public_rows = read_rows(PUBLIC_INVENTORY)
    scientific_rows = read_rows(SCIENTIFIC_INVENTORY)

    required_public = {
        "WorldPop population",
        "GHS-SMOD rural classification",
        "VIIRS nighttime lights",
        "MODIS vegetation indices",
        "MODIS NPP",
        "ESA WorldCover",
        "ERA5-Land climate",
        "Sentinel-2 surface reflectance",
        "SRTM terrain",
        "Province CPI",
        "Province rural income",
    }
    public_categories = {row["data_category"] for row in public_rows}
    missing_public = sorted(required_public - public_categories)
    scientific_categories = {row["data_category"] for row in scientific_rows}
    required_restricted = {"County income labels", "County and national boundaries"}
    missing_restricted = sorted(required_restricted - scientific_categories)
    if missing_public or missing_restricted:
        raise RuntimeError(
            f"Missing inventory rows: public={missing_public}; restricted={missing_restricted}"
        )

    lines = [
        "# Table 1. Input datasets",
        "",
        "**Table 1. Input statistical, demographic, remote-sensing and geospatial datasets used to generate the 1 km rural-income product.** Native raster and vector inputs were harmonised to the common 1 km analysis grid. County-level income labels were compiled by the authors from provincial statistical yearbooks; purchased administrative boundaries remain restricted and are not redistributed. No third-party raw input dataset is bundled in the public-release package.",
        "",
        "| Dataset / product | Provider | Period and native resolution | Role in workflow | Access, licence and citation | Redistribution in the public package |",
        "|---|---|---|---|---|---|",
    ]
    for row in ROWS:
        values = [
            row["dataset"],
            row["provider"],
            row["period_resolution"],
            row["role"],
            row["access_terms"],
            row["redistribution"],
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "Notes:",
            "",
            "1. All annual covariates were resampled or aggregated to the common 1 km grid before model fitting. Static layers were held constant across 2019–2023.",
            "2. National Bureau of Statistics source paths are: *Regional data → Provincial annual data → People's livelihood → Per capita disposable income → Rural residents' per capita disposable income (yuan)*, and *Price indices → Consumer Price Index (previous year = 100)*. The downloaded Excel workbooks are excluded until raw-table redistribution terms are confirmed.",
            "3. WorldPop population layers are licensed under CC BY 4.0. GHS-SMOD reuse requires acknowledgement. VIIRS EOG data are public domain. MODIS products have no subsequent-use or redistribution restriction. WorldCover is CC BY 4.0. ERA5-Land and Sentinel-2 require Copernicus acknowledgement. OpenStreetMap data are ODbL 1.0 and require attribution.",
            "4. The county-income row remains provenance-pending until each province-year source is linked to an exact statistical yearbook edition, table and page/URL. The administrative-boundary row remains restricted/pending until its purchased source, version, citation and publication permission are confirmed.",
            "",
            "Source inventories: `metadata/source_data_inventory_public_sources_filled_draft.csv` and `metadata/source_data_inventory_scientific_data_draft.csv` in the frozen public-release package.",
        ]
    )
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    status_counts: dict[str, int] = {}
    for row in ROWS:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    qa = {
        "table": "Table 1. Input datasets",
        "row_count": len(ROWS),
        "public_inventory_rows": len(public_rows),
        "scientific_inventory_rows": len(scientific_rows),
        "missing_required_public_categories": missing_public,
        "missing_required_restricted_categories": missing_restricted,
        "status_counts": status_counts,
        "submission_blockers": [
            "Complete the province-year statistical-yearbook source crosswalk for the author-compiled county-income labels.",
            "Supply exact source, version, citation and publication/redistribution permission for purchased administrative boundaries.",
        ],
        "raw_third_party_inputs_in_public_package": False,
    }
    QA_OUT.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
