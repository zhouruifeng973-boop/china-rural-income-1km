#!/usr/bin/env python3
"""Generate Scientific Data Tables 2-5 from the frozen public-release package."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = (
    ROOT
    / "outputs"
    / "public_release"
    / "china_rural_income_1km_2019_2023_v1.0-draft"
)
TABLES_OUT = ROOT / "docs" / "tables"

MANIFEST = RELEASE / "metadata" / "public_release_manifest_v1.0_draft.csv"
DATA_DICTIONARY = RELEASE / "metadata" / "data_dictionary.csv"
ANNUAL_SUMMARY = RELEASE / "tables" / "national_annual_summary_2019_2023.csv"
VALIDATION_SUMMARY = RELEASE / "tables" / "validation_summary.csv"
QUALITY_SUMMARY = RELEASE / "tables" / "uncertainty_proxy_summary_2019_2023.csv"
ENSEMBLE_SUMMARY = RELEASE / "tables" / "ensemble_uncertainty_summary_2019_2023.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt_num(value: str | float, decimals: int = 1) -> str:
    return f"{float(value):,.{decimals}f}"


def fmt_int(value: str | int) -> str:
    return f"{int(float(value)):,}"


def write_markdown(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def table2() -> tuple[str, dict[str, object]]:
    group_specs = [
        (
            "data/income_maps/",
            "Canonical annual income maps",
            "2019 constant yuan person⁻¹ year⁻¹",
            "−9999",
            "Primary 2019–2023 rural-income surfaces; strict mask, unbounded, province calibrated.",
        ),
        (
            "data/change_maps/",
            "2019–2023 change maps",
            "2019 constant yuan person⁻¹ year⁻¹; %",
            "−9999",
            "Absolute and percentage change between the canonical 2019 and 2023 products.",
        ),
        (
            "data/sensitivity_bounded_maps/",
            "Bounded sensitivity products",
            "2019 constant yuan person⁻¹ year⁻¹; %",
            "−9999",
            "Sensitivity analysis under the 3,000–80,000 yuan bounding rule; not the canonical product.",
        ),
        (
            "data/quality_layers/",
            "Quality-support layers",
            "0/1; factor; class; 2019 constant yuan",
            "255 or −9999",
            "Valid-pixel masks, training-province support, calibration factors/classes, and bounding sensitivity.",
        ),
        (
            "data/uncertainty_proxy_layers/",
            "Quality-risk proxy layers",
            "0–100 score; class",
            "255 or −9999",
            "Annual score/class layers and five-year mean/maximum scores; interpretation aid, not model SD.",
        ),
        (
            "data/ensemble_uncertainty_layers/",
            "Multi-seed ensemble layers",
            "2019 constant yuan; %",
            "−9999",
            "Annual ensemble mean, standard deviation, and coefficient of variation from five members.",
        ),
    ]

    rows = []
    for folder, product, unit, nodata, use in group_specs:
        directory = RELEASE / folder
        files = [path for path in directory.iterdir() if path.is_file()]
        tifs = [path for path in files if path.suffix.lower() == ".tif"]
        rows.append((folder, product, len(tifs), len(files), "GeoTIFF", unit, nodata, use))

    supporting = []
    for folder in ["tables", "metadata"]:
        files = [path for path in (RELEASE / folder).iterdir() if path.is_file()]
        extensions = Counter(path.suffix.lower().lstrip(".") or "none" for path in files)
        formats = ", ".join(f"{count} {ext.upper()}" for ext, count in sorted(extensions.items()))
        supporting.append((folder + "/", len(files), formats))
    root_docs = [path for path in RELEASE.iterdir() if path.is_file()]

    lines = [
        "# Table 2. Released data records",
        "",
        "**Table 2. Summary of released data-record groups.** Primary raster counts exclude auxiliary overview and XML sidecar files. All GeoTIFF products share the same 1 km analysis grid, spatial extent, and EPSG:4326 coordinate reference system. The canonical annual products use the strict rural mask, unbounded predictions, province-level calibration, and 2019 constant yuan.",
        "",
        "| Folder | Product group | Primary GeoTIFFs | All files in folder | Format | Unit | NoData | Recommended use |",
        "|---|---|---:|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    lines.extend(
        [
            "| **Total raster data** | **Six raster product groups** | **{}** | **{}** | **GeoTIFF and sidecars** | — | — | Use the data dictionary to select the appropriate product. |".format(
                sum(row[2] for row in rows), sum(row[3] for row in rows)
            ),
            "",
            "Supporting records: `tables/` contains {} files ({}); `metadata/` contains {} files ({}); and the release root contains {} Markdown documentation/version files. The complete package contains {} files.".format(
                supporting[0][1],
                supporting[0][2],
                supporting[1][1],
                supporting[1][2],
                len(root_docs),
                len([path for path in RELEASE.rglob("*") if path.is_file()]),
            ),
            "",
            "The release manifest records three county-level training/residual files as `exclude`; the files are absent from the public package as required by the release policy.",
            "",
            "Source: `metadata/public_release_manifest_v1.0_draft.csv`, `metadata/data_dictionary.csv`, and the frozen public-release directory.",
        ]
    )
    qa = {
        "raster_groups": [
            {"folder": row[0], "primary_geotiffs": row[2], "all_files": row[3]} for row in rows
        ],
        "total_geotiffs": sum(row[2] for row in rows),
        "total_data_folder_files": sum(row[3] for row in rows),
        "actual_package_files": len([path for path in RELEASE.rglob("*") if path.is_file()]),
    }
    return "\n".join(lines), qa


def table3() -> tuple[str, dict[str, object]]:
    rows = read_rows(ANNUAL_SUMMARY)
    lines = [
        "# Table 3. Annual income-map summary",
        "",
        "**Table 3. Summary statistics of the canonical annual rural-income maps.** Income values are expressed in 2019 constant yuan per person per year. Valid pixels are strict-mask rural pixels with valid rural-population weights. The population-weighted mean uses rural-population weights; annual growth is calculated from consecutive population-weighted means.",
        "",
        "| Year | Valid pixels | Unweighted mean | Median | Population-weighted mean | P1 | P99 | Weighted growth from previous year |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    qa_rows = []
    for row in rows:
        growth = "—" if not row["weighted_growth_from_previous_pct"] else fmt_num(row["weighted_growth_from_previous_pct"], 2) + "%"
        values = [
            row["year"],
            fmt_int(row["valid_pixels"]),
            fmt_num(row["unweighted_mean_2019yuan"]),
            fmt_num(row["median_2019yuan"]),
            fmt_num(row["population_weighted_mean_2019yuan"]),
            fmt_num(row["p01_2019yuan"]),
            fmt_num(row["p99_2019yuan"]),
            growth,
        ]
        lines.append("| " + " | ".join(values) + " |")
        qa_rows.append({key: row[key] for key in row})
    lines.extend(["", "Source: `tables/national_annual_summary_2019_2023.csv`."])
    return "\n".join(lines), {"rows": qa_rows}


def validation_label(row: dict[str, str]) -> str:
    labels = {
        "2020_leave_county_5fold_strict": "2020 leave-county five-fold, strict mask",
        "2020_leave_county_5fold_broad": "2020 leave-county five-fold, broad mask",
        "2020_leave_province_strict_before_calibration": "2020 leave-province, strict mask, before calibration",
        "2020_leave_province_strict_after_calibration": "2020 leave-province, strict mask, after calibration",
        "2019_2023_leave_one_year_out_strict_before_calibration": "2019–2023 leave-one-year-out, strict mask, before calibration",
        "2019_2023_leave_one_year_out_strict_after_calibration": "2019–2023 leave-one-year-out, strict mask, after calibration",
    }
    return labels[row["validation"]]


def table4() -> tuple[str, dict[str, object]]:
    rows = read_rows(VALIDATION_SUMMARY)
    lines = [
        "# Table 4. Technical-validation metrics",
        "",
        "**Table 4. County-level validation metrics after rural-population-weighted aggregation.** Pixel predictions were aggregated back to counties using rural-population weights and compared with observed county-level rural disposable income. MAE, RMSE, and bias are in 2019 constant yuan per person per year; MAPE is a percentage. These metrics do not represent pixel-level validation. Province calibration is a product-generation constraint, so after-calibration results describe the constrained product rather than fully independent validation.",
        "",
        "| Validation experiment | n | R² | MAE | RMSE | MAPE | Bias |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [
            validation_label(row),
            fmt_int(row["n"]),
            fmt_num(row["r2"], 3),
            fmt_num(row["mae"]),
            fmt_num(row["rmse"]),
            fmt_num(row["mape"], 2) + "%",
            fmt_num(row["bias"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "Source: `tables/validation_summary.csv`."])
    return "\n".join(lines), {"rows": rows}


def table5() -> tuple[str, dict[str, object]]:
    quality = {row["year"]: row for row in read_rows(QUALITY_SUMMARY)}
    ensemble = {row["year"]: row for row in read_rows(ENSEMBLE_SUMMARY)}
    years = sorted(set(quality) & set(ensemble))
    lines = [
        "# Table 5. Quality-risk and ensemble-uncertainty summary",
        "",
        "**Table 5. Annual quality-risk and ensemble-uncertainty indicators.** The quality-risk score is a 0–100 interpretation proxy combining training-province support, province-calibration magnitude, and bounded/unbounded sensitivity. High risk denotes scores of at least 50. Ensemble statistics summarize spread across five calibrated model members trained with different random seeds and pixel samples. Neither the quality-risk score nor ensemble spread is a direct pixel-level validation error or a full statistical confidence interval. STD is expressed in 2019 constant yuan per person per year; CV and high-risk share are percentages.",
        "",
        "| Year | Valid pixels | Risk score median | Risk score P95 | High-risk pixels | Ensemble members | STD median | STD P95 | CV median | CV P95 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    qa_rows = []
    for year in years:
        q = quality[year]
        e = ensemble[year]
        if int(q["valid_pixels"]) != int(e["valid_pixels"]):
            raise ValueError(f"Valid-pixel mismatch for {year}")
        values = [
            year,
            fmt_int(q["valid_pixels"]),
            fmt_num(q["score_median"]),
            fmt_num(q["score_p95"]),
            fmt_num(q["high_risk_pct"], 2) + "%",
            e["ensemble_members"],
            fmt_num(e["std_median"]),
            fmt_num(e["std_p95"]),
            fmt_num(e["cv_median_pct"], 2) + "%",
            fmt_num(e["cv_p95_pct"], 2) + "%",
        ]
        lines.append("| " + " | ".join(values) + " |")
        qa_rows.append({"year": year, "quality": q, "ensemble": e})
    lines.extend(
        [
            "",
            "Sources: `tables/uncertainty_proxy_summary_2019_2023.csv` and `tables/ensemble_uncertainty_summary_2019_2023.csv`.",
        ]
    )
    return "\n".join(lines), {"rows": qa_rows}


def main() -> None:
    required = [MANIFEST, DATA_DICTIONARY, ANNUAL_SUMMARY, VALIDATION_SUMMARY, QUALITY_SUMMARY, ENSEMBLE_SUMMARY]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing table input(s):\n" + "\n".join(missing))
    TABLES_OUT.mkdir(parents=True, exist_ok=True)

    outputs = {}
    qa = {"release": str(RELEASE.relative_to(ROOT)).replace("\\", "/"), "sources": [], "tables": {}}
    builders = [
        ("table2_data_records.md", table2),
        ("table3_annual_income_summary.md", table3),
        ("table4_technical_validation.md", table4),
        ("table5_quality_uncertainty.md", table5),
    ]
    for filename, builder in builders:
        content, table_qa = builder()
        path = TABLES_OUT / filename
        write_markdown(path, content)
        outputs[filename] = str(path.relative_to(ROOT)).replace("\\", "/")
        qa["tables"][filename] = table_qa

    manifest_rows = read_rows(MANIFEST)
    excluded = [row for row in manifest_rows if row["action"] == "exclude"]
    qa["manifest"] = {
        "include_rows": sum(row["action"] == "include" for row in manifest_rows),
        "exclude_rows": len(excluded),
        "excluded_paths_absent": all(not (RELEASE / row["relative_path"]).exists() for row in excluded),
    }
    qa["sources"] = [str(path.relative_to(ROOT)).replace("\\", "/") for path in required]
    qa["outputs"] = outputs
    qa_path = TABLES_OUT / "tables2_5_generation_qa.json"
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    readme = """# Scientific Data manuscript tables

This directory contains Tables 2–5 generated directly from the frozen public-release package.

- `table2_data_records.md`: released raster and supporting-record groups;
- `table3_annual_income_summary.md`: canonical annual-map statistics;
- `table4_technical_validation.md`: county-level validation metrics after population-weighted aggregation;
- `table5_quality_uncertainty.md`: quality-risk and five-member ensemble summaries;
- `tables2_5_generation_qa.json`: source and consistency checks.

Regenerate with `scripts/create_scientific_data_tables.py`. Table 1 is maintained separately because some source-data citation, licence, and purchased-data access statements remain subject to final review.
"""
    write_markdown(TABLES_OUT / "README.md", readme)
    print("Created Tables 2-5 in {}".format(TABLES_OUT))
    print("Created {}".format(qa_path))


if __name__ == "__main__":
    main()
