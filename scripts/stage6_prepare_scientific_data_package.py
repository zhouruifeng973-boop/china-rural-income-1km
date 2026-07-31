# -*- coding: utf-8 -*-
"""Prepare a first Scientific Data style data-product package.

This script does not modify the original Stage 5 outputs. It copies the
selected canonical rasters into a new Stage 6 package directory and creates
tables/metadata that make the dataset easier to inspect, cite, and later
deposit in a public repository.

Current decision:
    - Main annual maps: unbounded post-calibration strict rural maps.
    - Bounded maps: retained as sensitivity products, not the canonical data.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STAGE5 = ROOT / "outputs" / "stage5_final_maps"
STAGE3 = ROOT / "outputs" / "stage3_weak_supervision"
STAGE4 = ROOT / "outputs" / "stage4_province_calibration"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "stage6_scientific_data_product"

YEARS = [2019, 2020, 2021, 2022, 2023]


def ensure_dirs() -> None:
    for sub in [
        "data/income_maps",
        "data/change_maps",
        "data/sensitivity_bounded_maps",
        "tables",
        "metadata",
    ]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def copy_with_sidecars(src: Path, dst: Path) -> list[Path]:
    """Copy a raster and common ArcGIS sidecar files if they exist."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    copied = []
    shutil.copy2(src, dst)
    copied.append(dst)

    # Keep overview/statistics files because they make ArcMap display faster.
    for suffix in [".ovr", ".aux.xml", ".xml"]:
        sidecar = Path(str(src) + suffix)
        if sidecar.exists():
            sidecar_dst = Path(str(dst) + suffix)
            shutil.copy2(sidecar, sidecar_dst)
            copied.append(sidecar_dst)
    return copied


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def package_rasters() -> pd.DataFrame:
    records = []

    # Canonical annual maps: unbounded strict rural products.
    for year in YEARS:
        src = STAGE5 / f"rural_income_{year}_final_calibrated_unbounded_2019yuan_strict.tif"
        dst = OUT / "data" / "income_maps" / (
            f"china_rural_income_1km_{year}_2019yuan_strict_unbounded.tif"
        )
        for copied in copy_with_sidecars(src, dst):
            records.append(
                {
                    "role": "canonical_annual_income_map",
                    "year": year,
                    "source_file": str(src.relative_to(ROOT)),
                    "package_file": str(copied.relative_to(OUT)),
                }
            )

    change_specs = [
        (
            STAGE5 / "rural_income_change_2019_2023_absolute_unbounded_2019yuan.tif",
            OUT
            / "data"
            / "change_maps"
            / "china_rural_income_change_1km_2019_2023_absolute_2019yuan_strict_unbounded.tif",
            "canonical_absolute_change_map",
        ),
        (
            STAGE5 / "rural_income_change_2019_2023_percent_unbounded.tif",
            OUT
            / "data"
            / "change_maps"
            / "china_rural_income_change_1km_2019_2023_percent_strict_unbounded.tif",
            "canonical_percent_change_map",
        ),
    ]
    for src, dst, role in change_specs:
        for copied in copy_with_sidecars(src, dst):
            records.append(
                {
                    "role": role,
                    "year": "2019-2023",
                    "source_file": str(src.relative_to(ROOT)),
                    "package_file": str(copied.relative_to(OUT)),
                }
            )

    # Sensitivity maps: bounded 3000-80000 products, kept separate.
    for year in YEARS:
        src = STAGE5 / f"rural_income_{year}_final_calibrated_2019yuan_strict.tif"
        dst = OUT / "data" / "sensitivity_bounded_maps" / (
            f"china_rural_income_1km_{year}_2019yuan_strict_bounded_3000_80000.tif"
        )
        for copied in copy_with_sidecars(src, dst):
            records.append(
                {
                    "role": "sensitivity_bounded_annual_income_map",
                    "year": year,
                    "source_file": str(src.relative_to(ROOT)),
                    "package_file": str(copied.relative_to(OUT)),
                }
            )

    for src_name, dst_name, role in [
        (
            "rural_income_change_2019_2023_absolute_2019yuan.tif",
            "china_rural_income_change_1km_2019_2023_absolute_2019yuan_strict_bounded.tif",
            "sensitivity_bounded_absolute_change_map",
        ),
        (
            "rural_income_change_2019_2023_percent.tif",
            "china_rural_income_change_1km_2019_2023_percent_strict_bounded.tif",
            "sensitivity_bounded_percent_change_map",
        ),
    ]:
        src = STAGE5 / src_name
        dst = OUT / "data" / "sensitivity_bounded_maps" / dst_name
        for copied in copy_with_sidecars(src, dst):
            records.append(
                {
                    "role": role,
                    "year": "2019-2023",
                    "source_file": str(src.relative_to(ROOT)),
                    "package_file": str(copied.relative_to(OUT)),
                }
            )

    manifest = pd.DataFrame(records)
    manifest["bytes"] = manifest["package_file"].apply(lambda p: (OUT / p).stat().st_size)
    manifest["sha256"] = manifest["package_file"].apply(lambda p: sha256(OUT / p))
    manifest.to_csv(OUT / "metadata" / "file_manifest.csv", index=False, encoding="utf-8-sig")
    return manifest


def combine_calibration_tables() -> pd.DataFrame:
    frames = []
    for year in YEARS:
        path = STAGE5 / f"province_calibration_unbounded_{year}.csv"
        df = pd.read_csv(path)
        df.insert(0, "year", year)
        frames.append(df)
    calibration = pd.concat(frames, ignore_index=True)
    calibration.to_csv(
        OUT / "tables" / "province_calibration_factors_2019_2023_unbounded.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        calibration.groupby("year")
        .agg(
            province_count=("province_code", "count"),
            factor_min=("unbounded_factor", "min"),
            factor_median=("unbounded_factor", "median"),
            factor_max=("unbounded_factor", "max"),
            factor_mean=("unbounded_factor", "mean"),
            max_abs_final_vs_official_pct=(
                "final_vs_official_pct",
                lambda s: float(s.abs().max()),
            ),
        )
        .reset_index()
    )
    summary["provinces_factor_lt_0_8"] = calibration.groupby("year")[
        "unbounded_factor"
    ].apply(lambda s: int((s < 0.8).sum())).to_numpy()
    summary["provinces_factor_gt_1_2"] = calibration.groupby("year")[
        "unbounded_factor"
    ].apply(lambda s: int((s > 1.2).sum())).to_numpy()
    summary.to_csv(
        OUT / "tables" / "province_calibration_factor_summary_2019_2023.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return calibration


def copy_existing_tables() -> None:
    table_map = {
        STAGE5 / "national_annual_summary_2019_2023.csv": OUT
        / "tables"
        / "national_annual_summary_2019_2023.csv",
        STAGE5 / "pixel_change_summary_2019_2023.csv": OUT
        / "tables"
        / "pixel_change_summary_2019_2023.csv",
        STAGE5 / "raster_band_statistics.csv": OUT / "tables" / "raster_band_statistics_bounded.csv",
        PROCESSED / "province_income_2019_2023_deflated.csv": OUT
        / "tables"
        / "official_province_rural_income_2019_2023_2019yuan.csv",
        PROCESSED / "county_income_2019_2023_deflated.csv": OUT
        / "tables"
        / "county_training_income_2019_2023_2019yuan.csv",
        PROCESSED / "county_rural_weight_summary_2019_2023.csv": OUT
        / "tables"
        / "county_rural_weight_summary_2019_2023.csv",
    }
    for src, dst in table_map.items():
        if src.exists():
            shutil.copy2(src, dst)


def write_validation_summary() -> pd.DataFrame:
    records = []

    for mask in ["strict", "broad"]:
        path = STAGE3 / f"weak_supervised_2020_{mask}_overall_metrics.json"
        if path.exists():
            metrics = json.loads(path.read_text(encoding="utf-8"))
            records.append(
                {
                    "validation": f"2020_leave_county_5fold_{mask}",
                    "mask": mask,
                    "stage": "weak_supervised_downscaling",
                    "constraint": "no province calibration",
                    "n": metrics.get("n"),
                    "r2": metrics.get("r2"),
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "mape": metrics.get("mape"),
                    "bias": metrics.get("bias"),
                    "note": "R2/MAE are county-level metrics after population-weighted pixel aggregation.",
                }
            )

    for src, label in [
        (
            STAGE4 / "leave_province_2020_strict_summary.json",
            "2020_leave_province_strict_after_calibration",
        ),
        (
            STAGE4 / "leave_year_2019_2023_strict_summary.json",
            "2019_2023_leave_one_year_out_strict_after_calibration",
        ),
    ]:
        if src.exists():
            summary = json.loads(src.read_text(encoding="utf-8"))
            before = summary.get("before", {})
            after = summary.get("after", {})
            records.append(
                {
                    "validation": label.replace("_after_calibration", "_before_calibration"),
                    "mask": "strict",
                    "stage": "province_constraint_validation",
                    "constraint": "before province calibration",
                    "n": before.get("n"),
                    "r2": before.get("r2"),
                    "mae": before.get("mae"),
                    "rmse": before.get("rmse"),
                    "mape": before.get("mape"),
                    "bias": before.get("bias"),
                    "note": summary.get("note"),
                }
            )
            records.append(
                {
                    "validation": label,
                    "mask": "strict",
                    "stage": "province_constraint_validation",
                    "constraint": "after province calibration",
                    "n": after.get("n"),
                    "r2": after.get("r2"),
                    "mae": after.get("mae"),
                    "rmse": after.get("rmse"),
                    "mape": after.get("mape"),
                    "bias": after.get("bias"),
                    "note": summary.get("note"),
                }
            )

    validation = pd.DataFrame(records)
    validation.to_csv(OUT / "tables" / "validation_summary.csv", index=False, encoding="utf-8-sig")
    return validation


def write_data_dictionary() -> None:
    records = [
        {
            "file_group": "income_maps",
            "field_or_band": "band 1",
            "meaning": "Predicted rural residents' per capita disposable income",
            "unit": "2019 constant yuan/person/year",
            "nodata": "-9999",
            "notes": "Canonical version uses strict rural mask and unbounded post-calibration values.",
        },
        {
            "file_group": "change_maps_absolute",
            "field_or_band": "band 1",
            "meaning": "2023 income minus 2019 income",
            "unit": "2019 constant yuan/person/year",
            "nodata": "-9999",
            "notes": "Calculated on pixels valid in both 2019 and 2023.",
        },
        {
            "file_group": "change_maps_percent",
            "field_or_band": "band 1",
            "meaning": "Percent change from 2019 to 2023",
            "unit": "%",
            "nodata": "-9999",
            "notes": "100 * (income_2023 - income_2019) / income_2019.",
        },
        {
            "file_group": "province_calibration_factors",
            "field_or_band": "unbounded_factor",
            "meaning": "Province-year multiplier applied to initial pixel predictions",
            "unit": "dimensionless",
            "nodata": "",
            "notes": "Population-weighted province mean equals official province rural income after calibration.",
        },
        {
            "file_group": "validation_summary",
            "field_or_band": "r2, mae, rmse, mape, bias",
            "meaning": "County-level validation metrics after population-weighted aggregation",
            "unit": "income metrics in 2019 yuan unless MAPE/R2",
            "nodata": "",
            "notes": "These are not pixel-level validation metrics.",
        },
        {
            "file_group": "quality_valid_rural_pixel_mask",
            "field_or_band": "band 1",
            "meaning": "Valid rural income pixel flag",
            "unit": "0/1",
            "nodata": "255",
            "notes": "1 means the annual income map has a valid strict-rural pixel; 0 means invalid or outside the rural-population mask.",
        },
        {
            "file_group": "quality_common_valid_rural_pixel_mask",
            "field_or_band": "band 1",
            "meaning": "Common valid pixel flag across 2019-2023",
            "unit": "0/1",
            "nodata": "255",
            "notes": "1 means valid in all five annual income maps.",
        },
        {
            "file_group": "quality_province_calibration_factor",
            "field_or_band": "band 1",
            "meaning": "Province-year calibration factor assigned to each valid pixel",
            "unit": "dimensionless",
            "nodata": "-9999",
            "notes": "The factor is constant within each province-year.",
        },
        {
            "file_group": "quality_calibration_factor_class",
            "field_or_band": "band 1",
            "meaning": "Calibration magnitude class",
            "unit": "class code",
            "nodata": "255",
            "notes": "1=factor 0.9-1.1; 2=factor 0.8-1.2 but outside class 1; 3=factor <0.8 or >1.2.",
        },
        {
            "file_group": "quality_training_province_support_flag",
            "field_or_band": "band 1",
            "meaning": "Whether the pixel's province has county-level training labels in that year",
            "unit": "0/1",
            "nodata": "255",
            "notes": "1=province has county training labels; 0=province is extrapolated using province constraint only.",
        },
        {
            "file_group": "quality_bounded_unbounded_absdiff",
            "field_or_band": "band 1",
            "meaning": "Absolute difference between bounded sensitivity map and unbounded canonical map",
            "unit": "2019 constant yuan/person/year",
            "nodata": "-9999",
            "notes": "Larger values indicate pixels more sensitive to the 3000-80000 yuan bounding decision.",
        },
    ]
    pd.DataFrame(records).to_csv(
        OUT / "metadata" / "data_dictionary.csv", index=False, encoding="utf-8-sig"
    )


def write_source_inventory() -> None:
    records = [
        {
            "category": "target labels",
            "source": "County-level rural per capita disposable income",
            "local_path": "data/county_rural_income_2019.csv ... data/county_rural_income_2023.csv",
            "role": "Weak-supervised training labels after CPI deflation.",
            "publication_note": "Author-compiled from provincial statistical yearbooks; complete the province-year source crosswalk and cite exact editions/tables before release.",
        },
        {
            "category": "constraint labels",
            "source": "Province-level rural per capita disposable income",
            "local_path": "data/2005-2024年居民人均可支配收入与消费支出.xlsx",
            "role": "Province-year calibration constraint.",
            "publication_note": "Report official statistical source and license/permission.",
        },
        {
            "category": "price adjustment",
            "source": "Consumer Price Index, 2000-2024",
            "local_path": "data/居民消费价格指数2000-2024.xls",
            "role": "Convert nominal income to 2019 constant yuan.",
            "publication_note": "Report source and CPI base construction.",
        },
        {
            "category": "rural population weights",
            "source": "WorldPop 1 km population and GHS-SMOD 2020 rural classification",
            "local_path": "data/processed/rural_weights/",
            "role": "Define valid rural pixels and aggregate pixels to county/province means.",
            "publication_note": "Cite WorldPop and GHSL/JRC products; check licenses.",
        },
        {
            "category": "predictors",
            "source": "Nighttime light, vegetation, NPP, land cover, terrain, climate, Sentinel-2, road density",
            "local_path": "features/",
            "role": "1 km remote-sensing/spatial predictors for pixel-level income prediction.",
            "publication_note": "List exact product names, versions, access dates, and licenses.",
        },
    ]
    pd.DataFrame(records).to_csv(
        OUT / "metadata" / "source_data_inventory.csv", index=False, encoding="utf-8-sig"
    )


def write_readme(manifest: pd.DataFrame, calibration: pd.DataFrame, validation: pd.DataFrame) -> None:
    unbounded_summary = json.loads((STAGE5 / "unbounded_maps_summary.json").read_text(encoding="utf-8"))
    years = pd.DataFrame(unbounded_summary["years"])
    change = unbounded_summary["change"]
    created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Stage 6 Scientific Data-style package draft",
        "",
        "本目录是面向 Scientific Data / 数据论文整理的第一版数据产品包草稿。",
        "它不会替代原始 `outputs/stage5_final_maps`，而是把当前建议作为主版本的数据、敏感性版本、校准表、验证表和元数据集中整理。",
        "",
        "## 1. 主版本锁定",
        "",
        "- 主数据：2019—2023 年中国大陆 1 km 农村居民人均可支配收入预测图。",
        "- 单位：2019 年不变价人民币，元/人/年。",
        "- 农村定义：strict rural mask，即 GHS-SMOD 11/12/13 且 WorldPop 农村人口权重 > 0 的像元。",
        "- 主版本：`unbounded`，即省级校准后不再施加 3000—80000 元上下限。",
        "- bounded 版本仅作为敏感性分析保留在 `data/sensitivity_bounded_maps/`。",
        "- NoData：-9999，表示无有效农村人口或不在本研究统计口径内。",
        "",
        "## 2. 目录结构",
        "",
        "```text",
        "data/income_maps/                 主年度收入栅格",
        "data/change_maps/                 2019—2023 变化栅格",
        "data/sensitivity_bounded_maps/    bounded 敏感性栅格",
        "tables/                           统计表、校准表、验证摘要",
        "metadata/                         文件清单、数据字典、数据来源清单",
        "```",
        "",
        "## 3. 主图统计",
        "",
        "| year | valid_pixels | min | median | p99 | max | factor_min | factor_max |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in years.iterrows():
        lines.append(
            "| {year} | {valid_pixels:,} | {min:.1f} | {median:.1f} | {p99:.1f} | {max:.1f} | {factor_min:.3f} | {factor_max:.3f} |".format(
                **row.to_dict()
            )
        )

    lines.extend(
        [
            "",
            "## 4. 2019—2023 变化",
            "",
            f"- 共同有效像元数：{change['common_pixels']:,}",
            f"- 绝对变化中位数：{change['median_absolute_change']:.1f} 元/人/年",
            f"- 相对变化中位数：{change['median_percent_change']:.2f}%",
            "",
            "## 5. 关键验证结果",
            "",
            "| validation | constraint | R2 | MAE | RMSE | MAPE |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in validation.iterrows():
        lines.append(
            "| {validation} | {constraint} | {r2:.3f} | {mae:.1f} | {rmse:.1f} | {mape:.2f} |".format(
                **row.to_dict()
            )
        )

    factor_summary = (
        calibration.groupby("year")["unbounded_factor"]
        .agg(["min", "median", "max"])
        .reset_index()
    )
    lines.extend(
        [
            "",
            "## 6. 省级校准因子范围",
            "",
            "| year | min | median | max |",
            "|---:|---:|---:|---:|",
        ]
    )
    for _, row in factor_summary.iterrows():
        lines.append(
            f"| {int(row['year'])} | {row['min']:.3f} | {row['median']:.3f} | {row['max']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 7. 当前还没有完成的 Scientific Data 补强项",
            "",
            "- 栅格型质量图层：有效农村像元 mask、省级校准因子栅格、不确定性/质量标记图层。",
            "- 多随机种子或集成模型的不确定性估计。",
            "- 完整数据来源许可证与可再发布权限核查。",
            "- 可复现代码清理：把 stage2—stage6 脚本整理为统一入口和配置文件。",
            "- 公共仓库沉淀：Zenodo/Figshare/OSF 等 DOI 数据仓库版本。",
            "",
            "## 8. 重要解释边界",
            "",
            "- R²、MAE 等验证指标是像元预测经农村人口权重回聚合到县级后，与县级真实收入比较得到的，不是像元级真实精度。",
            "- 省级校准后的省级均值与官方统计一致是约束条件，不是独立验证结果。",
            "- 空白像元表示没有有效农村人口权重，不应被插值为农村收入。",
            "- 港澳台未纳入最终主产品，因为没有同口径省级约束。",
            "",
            "## 9. 文件完整性",
            "",
            f"- 文件数：{len(manifest)}",
            "- SHA256 校验码见 `metadata/file_manifest.csv`。",
            f"- 生成时间：{created}",
        ]
    )

    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    manifest = package_rasters()
    calibration = combine_calibration_tables()
    copy_existing_tables()
    validation = write_validation_summary()
    write_data_dictionary()
    write_source_inventory()
    write_readme(manifest, calibration, validation)

    print("Stage 6 package prepared:")
    print(OUT)
    print("Files in manifest:", len(manifest))
    print("Canonical annual maps:", len(manifest[manifest["role"] == "canonical_annual_income_map"]))
    print("Validation rows:", len(validation))


if __name__ == "__main__":
    main()
