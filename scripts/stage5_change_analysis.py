# -*- coding: utf-8 -*-
"""Create 2019-2023 change products from calibrated Stage 5 maps."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import rasterio

from stage3_weak_supervised_downscaling import ROOT

OUT = ROOT / "outputs" / "stage5_final_maps"
PROCESSED = ROOT / "data" / "processed"
YEARS = [2019, 2020, 2021, 2022, 2023]
NODATA = -9999.0


def map_path(year):
    return OUT / f"rural_income_{year}_final_calibrated_2019yuan_strict.tif"


def load_map(year):
    with rasterio.open(map_path(year)) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    return arr, profile


def valid_values(arr):
    return np.isfinite(arr) & (arr != NODATA)


def interval_metrics(a, b, start, end):
    common = valid_values(a) & valid_values(b)
    delta = b[common].astype(np.float64) - a[common].astype(np.float64)
    pct = 100 * delta / a[common].astype(np.float64)
    return {
        "start_year": start,
        "end_year": end,
        "common_pixels": int(common.sum()),
        "mean_change_yuan": float(delta.mean()),
        "median_change_yuan": float(np.median(delta)),
        "p10_change_yuan": float(np.quantile(delta, 0.10)),
        "p90_change_yuan": float(np.quantile(delta, 0.90)),
        "mean_change_pct": float(pct.mean()),
        "median_change_pct": float(np.median(pct)),
        "positive_pixel_pct": float(100 * np.mean(delta > 0)),
        "negative_pixel_pct": float(100 * np.mean(delta < 0)),
    }


def write_change_maps(first, last, profile):
    common = valid_values(first) & valid_values(last)
    absolute = np.full(first.shape, NODATA, dtype=np.float32)
    percent = np.full(first.shape, NODATA, dtype=np.float32)
    absolute[common] = last[common] - first[common]
    percent[common] = 100 * absolute[common] / first[common]
    profile.update(dtype="float32", count=1, nodata=NODATA, compress="deflate",
                   predictor=3, tiled=True, blockxsize=256, blockysize=256,
                   BIGTIFF="IF_SAFER")
    abs_path = OUT / "rural_income_change_2019_2023_absolute_2019yuan.tif"
    pct_path = OUT / "rural_income_change_2019_2023_percent.tif"
    with rasterio.open(abs_path, "w", **profile) as dst:
        dst.write(absolute, 1)
    with rasterio.open(pct_path, "w", **profile) as dst:
        dst.write(percent, 1)
    return abs_path, pct_path


def main():
    annual_rows = []
    interval_rows = []
    arrays = {}
    profile = None

    for year in YEARS:
        arr, current_profile = load_map(year)
        if profile is None:
            profile = current_profile
        elif (current_profile["height"], current_profile["width"], current_profile["crs"],
              current_profile["transform"]) != (
              profile["height"], profile["width"], profile["crs"], profile["transform"]):
            raise RuntimeError(f"Grid mismatch in {year}")
        arrays[year] = arr
        valid = valid_values(arr)
        with rasterio.open(
            PROCESSED / "rural_weights" / f"rural_pop_weight_{year}_strict.tif"
        ) as src:
            weights = src.read(1).astype(np.float64)
        use = valid & np.isfinite(weights) & (weights > 0)
        values = arr[use].astype(np.float64)
        w = weights[use]
        annual_rows.append({
            "year": year,
            "valid_pixels": int(use.sum()),
            "unweighted_mean_2019yuan": float(values.mean()),
            "median_2019yuan": float(np.median(values)),
            "population_weighted_mean_2019yuan": float(np.sum(values * w) / np.sum(w)),
            "p01_2019yuan": float(np.quantile(values, 0.01)),
            "p99_2019yuan": float(np.quantile(values, 0.99)),
        })

    for start, end in zip(YEARS[:-1], YEARS[1:]):
        interval_rows.append(interval_metrics(arrays[start], arrays[end], start, end))
    interval_rows.append(interval_metrics(arrays[2019], arrays[2023], 2019, 2023))

    abs_path, pct_path = write_change_maps(arrays[2019], arrays[2023], profile)
    annual = pd.DataFrame(annual_rows)
    annual["weighted_growth_from_previous_pct"] = (
        annual["population_weighted_mean_2019yuan"].pct_change() * 100
    )
    intervals = pd.DataFrame(interval_rows)
    annual.to_csv(OUT / "national_annual_summary_2019_2023.csv", index=False,
                  encoding="utf-8-sig")
    intervals.to_csv(OUT / "pixel_change_summary_2019_2023.csv", index=False,
                     encoding="utf-8-sig")

    total = interval_rows[-1]
    report = f"""# 2019—2023 全国农村收入 1 km 变化汇总

- 产品单位：2019 年不变价元/人·年。
- 空间范围：中国大陆 31 个省级地区中 strict 农村掩膜且农村人口权重大于 0 的像元。
- 2019—2023 共同有效像元：{total['common_pixels']:,}。
- 像元收入变化中位数：{total['median_change_yuan']:.0f} 元（{total['median_change_pct']:.2f}%）。
- 收入上升像元比例：{total['positive_pixel_pct']:.2f}%；下降像元比例：{total['negative_pixel_pct']:.2f}%。
- 绝对变化图：`{abs_path.name}`。
- 百分比变化图：`{pct_path.name}`。

逐年全国人口加权均值和像元变化分布分别见 `national_annual_summary_2019_2023.csv` 与 `pixel_change_summary_2019_2023.csv`。变化图只在两个端点年份均有有效农村人口权重的共同像元上计算。
"""
    (OUT / "stage5_change_summary.md").write_text(report, encoding="utf-8")
    print(json.dumps({"annual": annual_rows, "intervals": interval_rows},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
