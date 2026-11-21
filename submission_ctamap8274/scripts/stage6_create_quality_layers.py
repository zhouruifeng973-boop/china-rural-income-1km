# -*- coding: utf-8 -*-
"""Create raster quality layers for the Stage 6 data package.

The script expects rasterio to be available either in the active Python
environment or in the project-local `.codex_pydeps` directory.

Quality layers created:
    - valid rural pixel mask by year
    - common valid mask for 2019-2023
    - province calibration factor map by year
    - calibration factor quality class by year
    - training-province support flag by year
    - bounded vs unbounded absolute-difference map by year
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

LOCAL_DEPS = Path(__file__).resolve().parents[1] / "stage6_pydeps"
# The bundled stage6_pydeps directory was built for CPython 3.12.  Do not
# shadow a working environment on other Python versions (notably 3.11).
if LOCAL_DEPS.exists() and sys.version_info[:2] == (3, 12):
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd
import rasterio


ROOT = Path(__file__).resolve().parents[1]
STAGE5 = ROOT / "outputs" / "stage5_final_maps"
PACKAGE = ROOT / "outputs" / "stage6_scientific_data_product"
QUALITY = PACKAGE / "data" / "quality_layers"
TABLES = PACKAGE / "tables"
METADATA = PACKAGE / "metadata"
FEATURES = ROOT / "features"
PROCESSED = ROOT / "data" / "processed"
TARGET = PROCESSED / "county_income_2019_2023_deflated.csv"

YEARS = [2019, 2020, 2021, 2022, 2023]
NODATA_FLOAT = -9999.0
NODATA_UINT8 = 255


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def annual_unbounded_path(year: int) -> Path:
    return STAGE5 / f"rural_income_{year}_final_calibrated_unbounded_2019yuan_strict.tif"


def annual_bounded_path(year: int) -> Path:
    return STAGE5 / f"rural_income_{year}_final_calibrated_2019yuan_strict.tif"


def quality_profile(src, dtype: str, nodata, predictor: int | None = None):
    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        dtype=dtype,
        count=1,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    if predictor is not None:
        profile["predictor"] = predictor
    else:
        profile.pop("predictor", None)
    return profile


def load_training_provinces() -> dict[int, set[int]]:
    county = pd.read_csv(TARGET)
    county["province_code"] = (county["county_code"].astype(int) // 10000) * 10000
    result = {}
    for year, group in county.groupby("year"):
        result[int(year)] = set(group["province_code"].astype(int).unique().tolist())
    return result


def load_factor_lookup(year: int) -> dict[int, float]:
    df = pd.read_csv(STAGE5 / f"province_calibration_unbounded_{year}.csv")
    return dict(zip(df["province_code"].astype(int), df["unbounded_factor"].astype(float)))


def classify_factor(factor: np.ndarray) -> np.ndarray:
    """Classify calibration factor magnitude.

    1 = small adjustment, factor within [0.9, 1.1]
    2 = moderate adjustment, factor within [0.8, 1.2] but outside class 1
    3 = large adjustment, factor outside [0.8, 1.2]
    255 = NoData / invalid pixel
    """
    out = np.full(factor.shape, NODATA_UINT8, dtype=np.uint8)
    valid = np.isfinite(factor) & (factor != NODATA_FLOAT)
    small = valid & (factor >= 0.9) & (factor <= 1.1)
    moderate = valid & ~small & (factor >= 0.8) & (factor <= 1.2)
    large = valid & ~(small | moderate)
    out[small] = 1
    out[moderate] = 2
    out[large] = 3
    return out


def write_year_quality_layers(year: int, training_provinces: dict[int, set[int]]) -> dict:
    factor_lookup = load_factor_lookup(year)
    training_codes = training_provinces.get(year, set())

    valid_path = QUALITY / f"quality_valid_rural_pixel_mask_{year}_strict.tif"
    factor_path = QUALITY / f"quality_province_calibration_factor_{year}_strict_unbounded.tif"
    factor_class_path = QUALITY / f"quality_calibration_factor_class_{year}_strict_unbounded.tif"
    support_path = QUALITY / f"quality_training_province_support_flag_{year}_strict.tif"
    diff_path = QUALITY / f"quality_bounded_unbounded_absdiff_{year}_2019yuan_strict.tif"

    stats = {
        "year": year,
        "valid_pixels": 0,
        "training_province_valid_pixels": 0,
        "extrapolated_province_valid_pixels": 0,
        "factor_class_1_pixels": 0,
        "factor_class_2_pixels": 0,
        "factor_class_3_pixels": 0,
        "bounded_unbounded_diff_gt_1_yuan_pixels": 0,
        "bounded_unbounded_diff_gt_100_yuan_pixels": 0,
        "bounded_unbounded_absdiff_sum": 0.0,
        "bounded_unbounded_absdiff_max": 0.0,
    }

    with rasterio.open(annual_unbounded_path(year)) as income, \
        rasterio.open(annual_bounded_path(year)) as bounded, \
        rasterio.open(FEATURES / "county_id.tif") as county, \
        rasterio.open(valid_path, "w", **quality_profile(income, "uint8", NODATA_UINT8)) as valid_dst, \
        rasterio.open(factor_path, "w", **quality_profile(income, "float32", NODATA_FLOAT, predictor=3)) as factor_dst, \
        rasterio.open(factor_class_path, "w", **quality_profile(income, "uint8", NODATA_UINT8)) as class_dst, \
        rasterio.open(support_path, "w", **quality_profile(income, "uint8", NODATA_UINT8)) as support_dst, \
        rasterio.open(diff_path, "w", **quality_profile(income, "float32", NODATA_FLOAT, predictor=3)) as diff_dst:

        if income.shape != county.shape:
            raise RuntimeError("income raster and county_id raster have different shapes")

        income_nodata = income.nodata if income.nodata is not None else NODATA_FLOAT
        bounded_nodata = bounded.nodata if bounded.nodata is not None else NODATA_FLOAT

        for _, window in income.block_windows(1):
            inc = income.read(1, window=window).astype(np.float32)
            bnd = bounded.read(1, window=window).astype(np.float32)
            cid = county.read(1, window=window).astype(np.int64)
            valid = np.isfinite(inc) & (inc != income_nodata)
            province = (cid // 10000) * 10000

            valid_block = np.where(valid, 1, 0).astype(np.uint8)
            valid_dst.write(valid_block, 1, window=window)

            factor_block = np.full(inc.shape, NODATA_FLOAT, dtype=np.float32)
            support_block = np.full(inc.shape, NODATA_UINT8, dtype=np.uint8)
            for code, factor in factor_lookup.items():
                take = valid & (province == code)
                if np.any(take):
                    factor_block[take] = np.float32(factor)
                    support_block[take] = 1 if code in training_codes else 0

            factor_dst.write(factor_block, 1, window=window)
            class_block = classify_factor(factor_block)
            class_dst.write(class_block, 1, window=window)
            support_dst.write(support_block, 1, window=window)

            common = valid & np.isfinite(bnd) & (bnd != bounded_nodata)
            diff = np.full(inc.shape, NODATA_FLOAT, dtype=np.float32)
            diff_values = np.abs(bnd[common] - inc[common]).astype(np.float32)
            diff[common] = diff_values
            diff_dst.write(diff, 1, window=window)

            stats["valid_pixels"] += int(valid.sum())
            stats["training_province_valid_pixels"] += int(((support_block == 1) & valid).sum())
            stats["extrapolated_province_valid_pixels"] += int(((support_block == 0) & valid).sum())
            stats["factor_class_1_pixels"] += int((class_block == 1).sum())
            stats["factor_class_2_pixels"] += int((class_block == 2).sum())
            stats["factor_class_3_pixels"] += int((class_block == 3).sum())
            if diff_values.size:
                stats["bounded_unbounded_diff_gt_1_yuan_pixels"] += int((diff_values > 1).sum())
                stats["bounded_unbounded_diff_gt_100_yuan_pixels"] += int((diff_values > 100).sum())
                stats["bounded_unbounded_absdiff_sum"] += float(diff_values.sum())
                stats["bounded_unbounded_absdiff_max"] = max(
                    stats["bounded_unbounded_absdiff_max"], float(diff_values.max())
                )

    if stats["valid_pixels"]:
        stats["training_province_valid_pixel_pct"] = (
            100.0 * stats["training_province_valid_pixels"] / stats["valid_pixels"]
        )
        stats["extrapolated_province_valid_pixel_pct"] = (
            100.0 * stats["extrapolated_province_valid_pixels"] / stats["valid_pixels"]
        )
        stats["factor_class_1_pct"] = 100.0 * stats["factor_class_1_pixels"] / stats["valid_pixels"]
        stats["factor_class_2_pct"] = 100.0 * stats["factor_class_2_pixels"] / stats["valid_pixels"]
        stats["factor_class_3_pct"] = 100.0 * stats["factor_class_3_pixels"] / stats["valid_pixels"]
        stats["bounded_unbounded_absdiff_mean"] = (
            stats["bounded_unbounded_absdiff_sum"] / stats["valid_pixels"]
        )
    else:
        stats["training_province_valid_pixel_pct"] = np.nan
        stats["extrapolated_province_valid_pixel_pct"] = np.nan
        stats["factor_class_1_pct"] = np.nan
        stats["factor_class_2_pct"] = np.nan
        stats["factor_class_3_pct"] = np.nan
        stats["bounded_unbounded_absdiff_mean"] = np.nan

    return stats


def write_common_valid_mask() -> dict:
    out_path = QUALITY / "quality_common_valid_rural_pixel_mask_2019_2023_strict.tif"
    with rasterio.open(annual_unbounded_path(2019)) as ref:
        profile = quality_profile(ref, "uint8", NODATA_UINT8)
        nodata_by_year = {}
        sources = []
        try:
            for year in YEARS:
                src = rasterio.open(annual_unbounded_path(year))
                sources.append(src)
                nodata_by_year[year] = src.nodata if src.nodata is not None else NODATA_FLOAT

            common_count = 0
            ever_count = 0
            with rasterio.open(out_path, "w", **profile) as dst:
                for _, window in ref.block_windows(1):
                    masks = []
                    for year, src in zip(YEARS, sources):
                        arr = src.read(1, window=window).astype(np.float32)
                        masks.append(np.isfinite(arr) & (arr != nodata_by_year[year]))
                    stack = np.stack(masks, axis=0)
                    common = np.all(stack, axis=0)
                    ever = np.any(stack, axis=0)
                    common_count += int(common.sum())
                    ever_count += int(ever.sum())
                    dst.write(np.where(common, 1, 0).astype(np.uint8), 1, window=window)
        finally:
            for src in sources:
                src.close()

    return {
        "year": "2019-2023",
        "valid_pixels": common_count,
        "ever_valid_pixels": ever_count,
        "common_valid_share_of_ever_valid_pct": 100.0 * common_count / ever_count,
    }


def update_manifest() -> None:
    records = []
    for path in sorted(QUALITY.glob("*.tif")):
        records.append(
            {
                "role": "quality_layer",
                "year": "",
                "source_file": "",
                "package_file": str(path.relative_to(PACKAGE)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(records).to_csv(
        METADATA / "quality_layer_manifest.csv", index=False, encoding="utf-8-sig"
    )


def write_quality_readme(summary: pd.DataFrame, common: dict) -> None:
    lines = [
        "# Quality layers",
        "",
        "这些图层用于说明主数据产品的有效范围、约束强度和外推风险。",
        "",
        "## 文件说明",
        "",
        "- `quality_valid_rural_pixel_mask_YYYY_strict.tif`：1=该年有效农村收入像元，0=非有效农村像元。",
        "- `quality_common_valid_rural_pixel_mask_2019_2023_strict.tif`：1=2019—2023五年均有效，0=至少一年无效。",
        "- `quality_province_calibration_factor_YYYY_strict_unbounded.tif`：该像元所在省份当年的省级校准因子。",
        "- `quality_calibration_factor_class_YYYY_strict_unbounded.tif`：1=校准幅度小，因子0.9—1.1；2=中等，0.8—1.2但不在1类；3=较大，小于0.8或大于1.2；255=NoData。",
        "- `quality_training_province_support_flag_YYYY_strict.tif`：1=该像元所在省份有县级训练标签，0=该省属于全国外推区，255=NoData。",
        "- `quality_bounded_unbounded_absdiff_YYYY_2019yuan_strict.tif`：bounded版本与unbounded主版本的绝对差异，单位为2019年不变价元/人/年。",
        "",
        "## 摘要",
        "",
        f"- 2019—2023共同有效像元：{common['valid_pixels']:,}",
        f"- 五年中至少一年有效像元：{common['ever_valid_pixels']:,}",
        f"- 共同有效占至少一年有效比例：{common['common_valid_share_of_ever_valid_pct']:.2f}%",
        "",
        "| year | valid_pixels | training_province_pct | extrapolated_province_pct | class1_pct | class2_pct | class3_pct | mean_absdiff |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {year} | {valid_pixels:,} | {training_province_valid_pixel_pct:.2f} | {extrapolated_province_valid_pixel_pct:.2f} | {factor_class_1_pct:.2f} | {factor_class_2_pct:.2f} | {factor_class_3_pct:.2f} | {bounded_unbounded_absdiff_mean:.1f} |".format(
                year=int(row["year"]),
                valid_pixels=int(row["valid_pixels"]),
                training_province_valid_pixel_pct=float(row["training_province_valid_pixel_pct"]),
                extrapolated_province_valid_pixel_pct=float(row["extrapolated_province_valid_pixel_pct"]),
                factor_class_1_pct=float(row["factor_class_1_pct"]),
                factor_class_2_pct=float(row["factor_class_2_pct"]),
                factor_class_3_pct=float(row["factor_class_3_pct"]),
                bounded_unbounded_absdiff_mean=float(row["bounded_unbounded_absdiff_mean"]),
            )
        )
    (QUALITY / "README_quality_layers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global STAGE5, PACKAGE, QUALITY, TABLES, METADATA, TARGET
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage5-root", type=Path, default=STAGE5)
    parser.add_argument("--package-root", type=Path, default=PACKAGE)
    parser.add_argument("--target-path", type=Path, default=TARGET)
    args = parser.parse_args()
    STAGE5 = args.stage5_root.resolve()
    PACKAGE = args.package_root.resolve()
    TARGET = args.target_path.resolve()
    QUALITY = PACKAGE / "data" / "quality_layers"
    TABLES = PACKAGE / "tables"
    METADATA = PACKAGE / "metadata"
    if not TARGET.is_file():
        raise FileNotFoundError(TARGET)
    TABLES.mkdir(parents=True, exist_ok=True)
    METADATA.mkdir(parents=True, exist_ok=True)
    QUALITY.mkdir(parents=True, exist_ok=True)
    training_provinces = load_training_provinces()
    summaries = []
    for year in YEARS:
        print(f"Creating quality layers for {year} ...", flush=True)
        summaries.append(write_year_quality_layers(year, training_provinces))

    print("Creating common valid mask for 2019-2023 ...", flush=True)
    common = write_common_valid_mask()
    summary = pd.DataFrame(summaries)
    summary.to_csv(TABLES / "quality_layer_summary_2019_2023.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([common]).to_csv(
        TABLES / "quality_common_valid_mask_summary_2019_2023.csv",
        index=False,
        encoding="utf-8-sig",
    )
    update_manifest()
    write_quality_readme(summary, common)
    print("Quality layers written to:")
    print(QUALITY)
    print(summary[[
        "year",
        "valid_pixels",
        "training_province_valid_pixel_pct",
        "extrapolated_province_valid_pixel_pct",
        "factor_class_3_pct",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
