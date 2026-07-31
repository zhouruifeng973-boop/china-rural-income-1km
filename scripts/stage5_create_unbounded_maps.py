# -*- coding: utf-8 -*-
"""Create sensitivity maps without post-calibration income clipping.

The validated Stage 5 initial predictions are retained. A single unrestricted
positive multiplier is applied per province-year so the population-weighted
province mean matches the official statistic. No 3000/80000 clipping is
applied after calibration.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from stage3_weak_supervised_downscaling import ROOT
from stage5_final_maps import NODATA, official_income, output_profile, province_index

FEATURES = ROOT / "features"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "stage5_final_maps"
YEARS = [2019, 2020, 2021, 2022, 2023]


def initial_path(year):
    return OUT / f"rural_income_{year}_initial_2019yuan_strict.tif"


def unbounded_path(year):
    return OUT / f"rural_income_{year}_final_calibrated_unbounded_2019yuan_strict.tif"


def add_stats_tags(path):
    with rasterio.open(path, "r+") as src:
        values = src.read(1, masked=True).compressed().astype(np.float64)
        tags = {
            "STATISTICS_MINIMUM": f"{values.min():.10g}",
            "STATISTICS_MAXIMUM": f"{values.max():.10g}",
            "STATISTICS_MEAN": f"{values.mean():.10g}",
            "STATISTICS_STDDEV": f"{values.std():.10g}",
            "STATISTICS_VALID_PERCENT": f"{100 * len(values) / (src.width * src.height):.10g}",
        }
        src.update_tags(1, **tags)
    return values, tags


def make_year(year, row_block=128):
    official = official_income(year).reset_index(drop=True)
    codes = official["province_code"].to_numpy(np.int64)
    weighted_sum = np.zeros(len(official), dtype=np.float64)
    weight_sum = np.zeros(len(official), dtype=np.float64)
    weight_path = PROCESSED / "rural_weights" / f"rural_pop_weight_{year}_strict.tif"

    # First pass: calculate the unrestricted multiplicative factor.
    with rasterio.open(FEATURES / "county_id.tif") as county, \
         rasterio.open(weight_path) as weight_src, \
         rasterio.open(initial_path(year)) as initial:
        for row in range(0, county.height, row_block):
            h = min(row_block, county.height - row)
            window = Window(0, row, county.width, h)
            county_arr = county.read(1, window=window).astype(np.int64)
            weights = weight_src.read(1, window=window).astype(np.float64)
            raw = initial.read(1, window=window).astype(np.float64)
            pcode = (county_arr // 10000) * 10000
            pidx, mainland = province_index(pcode, codes)
            valid = (raw != NODATA) & (weights > 0) & mainland
            ids = pidx[valid]
            np.add.at(weighted_sum, ids, raw[valid] * weights[valid])
            np.add.at(weight_sum, ids, weights[valid])

    official["initial_aggregated_2019yuan"] = weighted_sum / np.maximum(weight_sum, 1e-9)
    official["unbounded_factor"] = (
        official["rural_income_2019yuan"] / official["initial_aggregated_2019yuan"]
    )
    factors = official["unbounded_factor"].to_numpy(np.float64)
    final_sum = np.zeros(len(official), dtype=np.float64)
    final_weight = np.zeros(len(official), dtype=np.float64)

    # Second pass: multiply without clipping and write a separate output.
    with rasterio.open(FEATURES / "county_id.tif") as county, \
         rasterio.open(weight_path) as weight_src, \
         rasterio.open(initial_path(year)) as initial, \
         rasterio.open(unbounded_path(year), "w", **output_profile(county)) as dst:
        for row in range(0, county.height, row_block):
            h = min(row_block, county.height - row)
            window = Window(0, row, county.width, h)
            county_arr = county.read(1, window=window).astype(np.int64)
            weights = weight_src.read(1, window=window).astype(np.float64)
            raw = initial.read(1, window=window).astype(np.float32)
            pcode = (county_arr // 10000) * 10000
            pidx, mainland = province_index(pcode, codes)
            valid = (raw != NODATA) & (weights > 0) & mainland
            block = np.full(raw.shape, NODATA, dtype=np.float32)
            ids = pidx[valid]
            values = raw[valid].astype(np.float64) * factors[ids]
            block[valid] = values.astype(np.float32)
            np.add.at(final_sum, ids, values * weights[valid])
            np.add.at(final_weight, ids, weights[valid])
            dst.write(block, 1, window=window)

    official["final_aggregated_2019yuan"] = final_sum / np.maximum(final_weight, 1e-9)
    official["final_vs_official_pct"] = 100 * (
        official["final_aggregated_2019yuan"] / official["rural_income_2019yuan"] - 1
    )
    values, tags = add_stats_tags(unbounded_path(year))
    official.to_csv(OUT / f"province_calibration_unbounded_{year}.csv", index=False,
                    encoding="utf-8-sig")
    return {
        "year": year,
        "file": unbounded_path(year).name,
        "valid_pixels": int(len(values)),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
        "max_abs_province_error_pct": float(official["final_vs_official_pct"].abs().max()),
        "factor_min": float(factors.min()),
        "factor_max": float(factors.max()),
        "statistics_tags": tags,
    }


def make_change_maps():
    with rasterio.open(unbounded_path(2019)) as first_src, \
         rasterio.open(unbounded_path(2023)) as last_src:
        first = first_src.read(1).astype(np.float32)
        last = last_src.read(1).astype(np.float32)
        profile = first_src.profile.copy()
    common = (first != NODATA) & (last != NODATA)
    absolute = np.full(first.shape, NODATA, dtype=np.float32)
    percent = np.full(first.shape, NODATA, dtype=np.float32)
    absolute[common] = last[common] - first[common]
    percent[common] = 100 * absolute[common] / first[common]
    abs_path = OUT / "rural_income_change_2019_2023_absolute_unbounded_2019yuan.tif"
    pct_path = OUT / "rural_income_change_2019_2023_percent_unbounded.tif"
    profile.update(dtype="float32", count=1, nodata=NODATA, compress="deflate",
                   predictor=3, tiled=True, blockxsize=256, blockysize=256,
                   BIGTIFF="IF_SAFER")
    with rasterio.open(abs_path, "w", **profile) as dst:
        dst.write(absolute, 1)
    with rasterio.open(pct_path, "w", **profile) as dst:
        dst.write(percent, 1)
    abs_values, _ = add_stats_tags(abs_path)
    pct_values, _ = add_stats_tags(pct_path)
    return {
        "absolute_file": abs_path.name,
        "percent_file": pct_path.name,
        "common_pixels": int(common.sum()),
        "median_absolute_change": float(np.median(abs_values)),
        "median_percent_change": float(np.median(pct_values)),
    }


def main():
    summaries = []
    for year in YEARS:
        print(f"Creating unbounded post-calibration map for {year} ...")
        summary = make_year(year)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    change = make_change_maps()
    result = {
        "scope": "No clipping after province calibration; validated base model retained.",
        "years": summaries,
        "change": change,
    }
    (OUT / "unbounded_maps_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
