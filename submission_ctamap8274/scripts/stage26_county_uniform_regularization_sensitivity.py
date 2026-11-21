# -*- coding: utf-8 -*-
"""County-uniform null baseline and output-regularization sensitivity.

The experiment is deliberately isolated under ``outputs/stage26_spatial_validation``.
It does not overwrite the frozen v1.3 maps or the public release package.

Two complementary questions are addressed:

1. How much within-county spatial information is present relative to a null
   surface that preserves each county's population-weighted mean exactly?
2. Is the inferred within-county ordering stable when the explicit pixel-output
   shrinkage coefficient is changed to 0.5x and 2x its frozen value?

The tests quantify spatial information and robustness. They are not direct
pixel-accuracy validation because no national 1-km income observations exist.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
# Use installed packages of the active interpreter (cached wheels may differ).

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import torch

import stage3_weak_supervised_downscaling as weak
import stage3_spatiotemporal_validation as validation
import stage5_final_maps as final_maps
import stage5_create_unbounded_maps as unbounded


OUT = ROOT / "outputs" / "stage26_spatial_validation"
BASELINE_OUT = OUT / "county_uniform_baseline"
REG_OUT = OUT / "regularization_sensitivity"
CANONICAL = ROOT / "outputs" / "official44_corrected_full_rerun_v3" / "stage5_final_maps"
TARGET = ROOT / "data" / "corrections" / "county_income_2019_2023_deflated_official44_corrected_v3.csv"
COUNTY_PATH = ROOT / "features" / "county_id.tif"
WEIGHT_DIR = ROOT / "data" / "processed" / "rural_weights"
YEARS = [2019, 2020, 2021, 2022, 2023]
SENSITIVITY_YEARS = [2019, 2020, 2023]
LAMBDAS = [0.0005, 0.001, 0.002]
FROZEN_LAMBDA = 0.001
NODATA = -9999.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_loader(target_path: Path):
    def load_targets(year: int) -> pd.DataFrame:
        df = pd.read_csv(target_path)
        df = df[df["year"].astype(int) == year].copy()
        df["county_code"] = df["county_code"].astype(int)
        df = df.dropna(subset=["rural_income_2019yuan", "province_std"])
        return df[[
            "year", "county_code", "county", "city", "province_std",
            "rural_income_nominal", "rural_income_2019yuan",
        ]]
    return load_targets


def map_path(root: Path, year: int) -> Path:
    return root / f"rural_income_{year}_final_calibrated_unbounded_2019yuan_strict.tif"


def aligned(*sources: rasterio.io.DatasetReader) -> bool:
    first = sources[0]
    return all(
        src.shape == first.shape and src.crs == first.crs and src.transform == first.transform
        for src in sources[1:]
    )


def output_profile(source: rasterio.io.DatasetReader) -> dict:
    profile = source.profile.copy()
    profile.update(
        driver="GTiff", dtype="float32", count=1, nodata=NODATA,
        compress="deflate", predictor=3, tiled=True, blockxsize=256,
        blockysize=256, BIGTIFF="IF_SAFER",
    )
    return profile


def create_county_uniform_baseline(year: int) -> tuple[pd.DataFrame, dict]:
    source_path = map_path(CANONICAL, year)
    weight_path = WEIGHT_DIR / f"rural_pop_weight_{year}_strict.tif"
    output_path = BASELINE_OUT / f"county_uniform_baseline_{year}_2019yuan_strict.tif"
    BASELINE_OUT.mkdir(parents=True, exist_ok=True)

    with rasterio.open(COUNTY_PATH) as county, rasterio.open(weight_path) as weight, rasterio.open(source_path) as source:
        if not aligned(county, weight, source):
            raise RuntimeError(f"Grid mismatch for {year}")
        max_code = int(county.read(1, out_shape=(max(1, county.height // 8), max(1, county.width // 8)), resampling=rasterio.enums.Resampling.nearest).max())
        max_code = max(max_code, 700000)
        weighted_sum = np.zeros(max_code + 1, dtype=np.float64)
        weight_sum = np.zeros(max_code + 1, dtype=np.float64)
        value_sum = np.zeros(max_code + 1, dtype=np.float64)
        pixel_count = np.zeros(max_code + 1, dtype=np.int64)

        for _, window in source.block_windows(1):
            c = county.read(1, window=window).astype(np.int64)
            w = weight.read(1, window=window).astype(np.float64)
            v = source.read(1, window=window).astype(np.float64)
            valid = (c > 0) & (c <= max_code) & np.isfinite(w) & (w > 0) & np.isfinite(v) & (v != source.nodata)
            codes = c[valid]
            np.add.at(weighted_sum, codes, v[valid] * w[valid])
            np.add.at(weight_sum, codes, w[valid])
            np.add.at(value_sum, codes, v[valid])
            np.add.at(pixel_count, codes, 1)

        county_mean = np.full(max_code + 1, np.nan, dtype=np.float64)
        good = weight_sum > 0
        county_mean[good] = weighted_sum[good] / weight_sum[good]
        weighted_sq = np.zeros(max_code + 1, dtype=np.float64)
        weighted_abs = np.zeros(max_code + 1, dtype=np.float64)
        sq = np.zeros(max_code + 1, dtype=np.float64)
        abs_differences: list[np.ndarray] = []
        weighted_abs_total = 0.0
        total_weight = 0.0

        with rasterio.open(output_path, "w", **output_profile(source)) as destination:
            for _, window in source.block_windows(1):
                c = county.read(1, window=window).astype(np.int64)
                w = weight.read(1, window=window).astype(np.float64)
                v = source.read(1, window=window).astype(np.float64)
                valid = (c > 0) & (c <= max_code) & np.isfinite(w) & (w > 0) & np.isfinite(v) & (v != source.nodata)
                block = np.full(c.shape, NODATA, dtype=np.float32)
                codes = c[valid]
                baseline = county_mean[codes]
                block[valid] = baseline.astype(np.float32)
                destination.write(block, 1, window=window)
                difference = v[valid] - baseline
                np.add.at(weighted_sq, codes, w[valid] * difference**2)
                np.add.at(weighted_abs, codes, w[valid] * np.abs(difference))
                unweighted_mean = value_sum[codes] / np.maximum(pixel_count[codes], 1)
                np.add.at(sq, codes, (v[valid] - unweighted_mean) ** 2)
                abs_differences.append(np.abs(difference).astype(np.float32))
                weighted_abs_total += float(np.sum(np.abs(difference) * w[valid]))
                total_weight += float(np.sum(w[valid]))

    codes = np.flatnonzero(good)
    weighted_sd = np.sqrt(weighted_sq[codes] / weight_sum[codes])
    unweighted_sd = np.sqrt(sq[codes] / np.maximum(pixel_count[codes], 1))
    county_table = pd.DataFrame({
        "year": year,
        "county_code": codes,
        "rural_pixel_count": pixel_count[codes],
        "rural_population_weight": weight_sum[codes],
        "product_county_weighted_mean_2019yuan": county_mean[codes],
        "uniform_baseline_value_2019yuan": county_mean[codes],
        "product_within_county_weighted_sd_yuan": weighted_sd,
        "product_within_county_weighted_cv_pct": 100.0 * weighted_sd / np.maximum(county_mean[codes], 1.0),
        "product_within_county_unweighted_sd_yuan": unweighted_sd,
        "product_within_county_weighted_mad_from_uniform_yuan": weighted_abs[codes] / weight_sum[codes],
        "baseline_within_county_sd_yuan": 0.0,
    })
    all_abs = np.concatenate(abs_differences) if abs_differences else np.array([], dtype=np.float32)
    summary = {
        "year": year,
        "definition": "Each valid rural pixel receives its county's population-weighted mean from the canonical product.",
        "county_count": int(len(county_table)),
        "valid_pixels": int(county_table["rural_pixel_count"].sum()),
        "max_abs_county_mean_difference_yuan": 0.0,
        "median_county_weighted_cv_pct": float(county_table["product_within_county_weighted_cv_pct"].median()),
        "p90_county_weighted_cv_pct": float(county_table["product_within_county_weighted_cv_pct"].quantile(0.90)),
        "median_county_weighted_mad_yuan": float(county_table["product_within_county_weighted_mad_from_uniform_yuan"].median()),
        "population_weighted_mean_abs_pixel_difference_yuan": weighted_abs_total / max(total_weight, 1e-9),
        "pixel_abs_difference_median_yuan": float(np.median(all_abs)),
        "pixel_abs_difference_p90_yuan": float(np.quantile(all_abs, 0.90)),
        "pixel_abs_difference_p95_yuan": float(np.quantile(all_abs, 0.95)),
        "source_map": str(source_path.relative_to(ROOT)).replace("\\", "/"),
        "baseline_map": str(output_path.relative_to(ROOT)).replace("\\", "/"),
    }
    return county_table, summary


def run_uniform_baselines() -> tuple[pd.DataFrame, pd.DataFrame]:
    county_tables = []
    summaries = []
    for year in YEARS:
        print(f"[uniform baseline] {year}", flush=True)
        table, summary = create_county_uniform_baseline(year)
        county_tables.append(table)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        gc.collect()
    county_df = pd.concat(county_tables, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    county_df.to_csv(BASELINE_OUT / "county_uniform_baseline_county_metrics_2019_2023.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(BASELINE_OUT / "county_uniform_baseline_summary_2019_2023.csv", index=False, encoding="utf-8-sig")
    return county_df, summary_df


def rank_values(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)


def spearman_values(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    return float(np.corrcoef(rank_values(x), rank_values(y))[0, 1])


def overlap_fraction(reference: np.ndarray, candidate: np.ndarray, upper: bool) -> float:
    n = len(reference)
    k = max(1, int(math.ceil(0.2 * n)))
    if upper:
        ref_idx = np.argpartition(reference, n - k)[n - k:]
        cand_idx = np.argpartition(candidate, n - k)[n - k:]
    else:
        ref_idx = np.argpartition(reference, k - 1)[:k]
        cand_idx = np.argpartition(candidate, k - 1)[:k]
    return float(len(np.intersect1d(ref_idx, cand_idx, assume_unique=False)) / k)


def compare_maps(reference_path: Path, candidate_path: Path, year: int, value_lambda: float) -> tuple[dict, pd.DataFrame]:
    weight_path = WEIGHT_DIR / f"rural_pop_weight_{year}_strict.tif"
    with rasterio.open(reference_path) as reference, rasterio.open(candidate_path) as candidate, rasterio.open(COUNTY_PATH) as county, rasterio.open(weight_path) as weight:
        if not aligned(reference, candidate, county, weight):
            raise RuntimeError(f"Sensitivity grid mismatch for {year}, lambda={value_lambda}")
        ref = reference.read(1).astype(np.float32)
        cand = candidate.read(1).astype(np.float32)
        codes = county.read(1).astype(np.int64)
        weights = weight.read(1).astype(np.float32)
        valid = (
            np.isfinite(ref) & np.isfinite(cand) & (ref != reference.nodata) &
            (cand != candidate.nodata) & (codes > 0) & np.isfinite(weights) & (weights > 0)
        )
    ref_v = ref[valid].astype(np.float64)
    cand_v = cand[valid].astype(np.float64)
    code_v = codes[valid]
    weight_v = weights[valid].astype(np.float64)
    rng = np.random.default_rng(26000 + year)
    take = np.arange(len(ref_v))
    if len(take) > 250000:
        take = np.sort(rng.choice(take, size=250000, replace=False))
    sample_ref = ref_v[take]
    sample_cand = cand_v[take]
    difference = cand_v - ref_v

    county_rows = []
    order = np.argsort(code_v, kind="stable")
    sorted_codes = code_v[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1], True])
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        idx = order[start:stop]
        if len(idx) < 10:
            continue
        if len(idx) > 300:
            idx = idx[np.linspace(0, len(idx) - 1, 300, dtype=int)]
        a = ref_v[idx]
        b = cand_v[idx]
        county_rows.append({
            "year": year,
            "output_regularization": value_lambda,
            "county_code": int(code_v[idx[0]]),
            "sampled_rural_pixels": int(len(idx)),
            "within_county_spearman": spearman_values(a, b),
            "top20_overlap_fraction": overlap_fraction(a, b, True),
            "bottom20_overlap_fraction": overlap_fraction(a, b, False),
        })
    county_df = pd.DataFrame(county_rows)
    summary = {
        "year": year,
        "output_regularization": value_lambda,
        "reference_regularization": FROZEN_LAMBDA,
        "common_valid_pixels": int(len(ref_v)),
        "sampled_pixel_spearman": spearman_values(sample_ref, sample_cand),
        "sampled_top20_overlap_fraction": overlap_fraction(sample_ref, sample_cand, True),
        "sampled_bottom20_overlap_fraction": overlap_fraction(sample_ref, sample_cand, False),
        "mean_absolute_difference_yuan": float(np.mean(np.abs(difference))),
        "median_absolute_difference_yuan": float(np.median(np.abs(difference))),
        "p95_absolute_difference_yuan": float(np.quantile(np.abs(difference), 0.95)),
        "population_weighted_mean_absolute_difference_yuan": float(np.sum(np.abs(difference) * weight_v) / np.sum(weight_v)),
        "median_county_spearman": float(county_df["within_county_spearman"].median()),
        "p10_county_spearman": float(county_df["within_county_spearman"].quantile(0.10)),
        "median_county_top20_overlap_fraction": float(county_df["top20_overlap_fraction"].median()),
        "median_county_bottom20_overlap_fraction": float(county_df["bottom20_overlap_fraction"].median()),
    }
    return summary, county_df


def lambda_slug(value: float) -> str:
    return f"lambda_{value:.4f}".replace(".", "p")


def run_regularization_sensitivity(epochs: int, pixels_per_county: int, row_block: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not TARGET.exists():
        raise FileNotFoundError(TARGET)
    weak.load_targets = target_loader(TARGET)
    validation.load_pixel_panel = weak.load_pixel_panel
    validation.fit_fold_model = weak.fit_fold_model
    validation.aggregate_prediction = weak.aggregate_prediction
    validation.sample_pixels_by_county = weak.sample_pixels_by_county

    cv_rows = []
    map_summaries = []
    county_stability = []
    generated_roots: dict[float, Path] = {}

    for value_lambda in LAMBDAS:
        run_root = REG_OUT / lambda_slug(value_lambda)
        generated_roots[value_lambda] = run_root
        run_root.mkdir(parents=True, exist_ok=True)
        print(f"[regularization] lambda={value_lambda:.4g}: leave-province validation", flush=True)
        predictions, folds, overall = validation.run_leave_province(
            2020, "strict", 5, pixels_per_county, epochs, 0.001, value_lambda
        )
        predictions.to_csv(run_root / "leave_province_2020_county_predictions.csv", index=False, encoding="utf-8-sig")
        folds.to_csv(run_root / "leave_province_2020_fold_metrics.csv", index=False, encoding="utf-8-sig")
        (run_root / "leave_province_2020_overall_metrics.json").write_text(
            json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        cv_rows.append(overall)
        del predictions, folds
        gc.collect()

        print(f"[regularization] lambda={value_lambda:.4g}: joint model", flush=True)
        final_maps.OUT = run_root
        final_maps.MODEL_DIR = run_root / "model"
        model, imputer, scaler, _ = final_maps.train_and_save(
            pixels_per_county, epochs, 0.001, value_lambda
        )
        unbounded.OUT = run_root
        for year in SENSITIVITY_YEARS:
            print(f"[regularization] lambda={value_lambda:.4g}: map {year}", flush=True)
            final_maps.predict_initial(year, model, imputer, scaler, row_block)
            unbounded.make_year(year, row_block)
        del model, imputer, scaler
        gc.collect()

    reference_root = generated_roots[FROZEN_LAMBDA]
    for value_lambda in LAMBDAS:
        if value_lambda == FROZEN_LAMBDA:
            continue
        for year in SENSITIVITY_YEARS:
            summary, county_df = compare_maps(
                map_path(reference_root, year), map_path(generated_roots[value_lambda], year), year, value_lambda
            )
            map_summaries.append(summary)
            county_stability.append(county_df)
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    cv_df = pd.DataFrame(cv_rows)
    map_df = pd.DataFrame(map_summaries)
    county_df = pd.concat(county_stability, ignore_index=True)
    cv_df.to_csv(REG_OUT / "regularization_leave_province_2020_summary.csv", index=False, encoding="utf-8-sig")
    map_df.to_csv(REG_OUT / "regularization_map_stability_summary.csv", index=False, encoding="utf-8-sig")
    county_df.to_csv(REG_OUT / "regularization_county_spatial_stability.csv", index=False, encoding="utf-8-sig")
    return cv_df, map_df, county_df


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return {
        "n": int(len(y_true)),
        "r2": float(1.0 - np.sum(residual**2) / denominator),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mape": float(np.mean(np.abs(residual) / y_true) * 100.0),
        "bias": float(np.mean(y_pred - y_true)),
    }


def postprocess_calibrated_cv() -> pd.DataFrame:
    official_path = ROOT / "data" / "processed" / "province_income_2019_2023_deflated.csv"
    official = pd.read_csv(official_path)
    official = official[official["year"].astype(int) == 2020][
        ["province_std", "rural_income_2019yuan"]
    ].rename(columns={"rural_income_2019yuan": "official_province_income_2019yuan"})
    rows = []
    for value_lambda in LAMBDAS:
        run_root = REG_OUT / lambda_slug(value_lambda)
        path = run_root / "leave_province_2020_county_predictions.csv"
        pred = pd.read_csv(path)
        province_rows = []
        for province, group in pred.groupby("province_std", sort=True):
            weights = group["rural_weight_sum"].to_numpy(float)
            values = group["pred_weak_2019yuan"].to_numpy(float)
            province_rows.append({
                "province_std": province,
                "covered_counties": int(len(group)),
                "covered_rural_weight": float(np.sum(weights)),
                "predicted_province_mean_2019yuan": float(np.sum(values * weights) / np.sum(weights)),
            })
        factors = pd.DataFrame(province_rows).merge(
            official, on="province_std", how="left", validate="one_to_one"
        )
        if factors["official_province_income_2019yuan"].isna().any():
            raise RuntimeError(f"Missing official 2020 province constraint for lambda={value_lambda}")
        factors["calibration_factor"] = (
            factors["official_province_income_2019yuan"] /
            factors["predicted_province_mean_2019yuan"]
        )
        pred = pred.merge(
            factors[["province_std", "calibration_factor"]],
            on="province_std", how="left", validate="many_to_one",
        )
        pred["pred_calibrated_2019yuan"] = pred["pred_weak_2019yuan"] * pred["calibration_factor"]
        before = metric_dict(pred["true_2019yuan"], pred["pred_weak_2019yuan"])
        after = metric_dict(pred["true_2019yuan"], pred["pred_calibrated_2019yuan"])
        rows.append({
            "output_regularization": value_lambda,
            **{f"{key}_before": val for key, val in before.items()},
            **{f"{key}_after": val for key, val in after.items()},
            "factor_min": float(factors["calibration_factor"].min()),
            "factor_median": float(factors["calibration_factor"].median()),
            "factor_max": float(factors["calibration_factor"].max()),
            "interpretation": "After metrics use held-out province-year official income constraints and are not fully independent validation.",
        })
        pred.to_csv(run_root / "leave_province_2020_calibrated_predictions.csv", index=False, encoding="utf-8-sig")
        factors.to_csv(run_root / "leave_province_2020_calibration_factors.csv", index=False, encoding="utf-8-sig")
    result = pd.DataFrame(rows)
    result.to_csv(REG_OUT / "regularization_leave_province_2020_before_after.csv", index=False, encoding="utf-8-sig")
    return result


def audit_stage26_outputs() -> pd.DataFrame:
    rows = []
    max_code = 700000
    for year in YEARS:
        source_path = map_path(CANONICAL, year)
        baseline_path = BASELINE_OUT / f"county_uniform_baseline_{year}_2019yuan_strict.tif"
        weight_path = WEIGHT_DIR / f"rural_pop_weight_{year}_strict.tif"
        source_sum = np.zeros(max_code + 1, dtype=np.float64)
        baseline_sum = np.zeros(max_code + 1, dtype=np.float64)
        den = np.zeros(max_code + 1, dtype=np.float64)
        mask_mismatch = 0
        with rasterio.open(source_path) as source, rasterio.open(baseline_path) as baseline, rasterio.open(COUNTY_PATH) as county, rasterio.open(weight_path) as weight:
            grid_ok = aligned(source, baseline, county, weight)
            for _, window in source.block_windows(1):
                src = source.read(1, window=window).astype(np.float64)
                base = baseline.read(1, window=window).astype(np.float64)
                codes = county.read(1, window=window).astype(np.int64)
                weights = weight.read(1, window=window).astype(np.float64)
                src_valid = np.isfinite(src) & (src != source.nodata) & (codes > 0) & (codes <= max_code) & (weights > 0)
                base_valid = np.isfinite(base) & (base != baseline.nodata) & (codes > 0) & (codes <= max_code) & (weights > 0)
                mask_mismatch += int(np.count_nonzero(src_valid != base_valid))
                valid = src_valid & base_valid
                c = codes[valid]
                w = weights[valid]
                np.add.at(source_sum, c, src[valid] * w)
                np.add.at(baseline_sum, c, base[valid] * w)
                np.add.at(den, c, w)
        good = den > 0
        source_mean = source_sum[good] / den[good]
        baseline_mean = baseline_sum[good] / den[good]
        max_difference = float(np.max(np.abs(source_mean - baseline_mean)))
        rows.append({
            "audit_group": "county_uniform_baseline",
            "year": year,
            "output_regularization": np.nan,
            "grid_aligned": grid_ok,
            "valid_mask_mismatch_pixels": mask_mismatch,
            "county_count": int(np.sum(good)),
            "max_abs_county_weighted_mean_difference_yuan": max_difference,
            "passed": bool(grid_ok and mask_mismatch == 0 and max_difference < 0.01),
            "file": str(baseline_path.relative_to(ROOT)).replace("\\", "/"),
        })
    for value_lambda in LAMBDAS:
        run_root = REG_OUT / lambda_slug(value_lambda)
        for year in SENSITIVITY_YEARS:
            candidate_path = map_path(run_root, year)
            reference_path = map_path(REG_OUT / lambda_slug(FROZEN_LAMBDA), year)
            with rasterio.open(reference_path) as reference, rasterio.open(candidate_path) as candidate:
                grid_ok = aligned(reference, candidate)
                valid_mismatch = 0
                valid_count = 0
                for _, window in reference.block_windows(1):
                    ref = reference.read(1, window=window)
                    cand = candidate.read(1, window=window)
                    ref_valid = np.isfinite(ref) & (ref != reference.nodata)
                    cand_valid = np.isfinite(cand) & (cand != candidate.nodata)
                    valid_mismatch += int(np.count_nonzero(ref_valid != cand_valid))
                    valid_count += int(np.count_nonzero(ref_valid & cand_valid))
            rows.append({
                "audit_group": "regularization_map",
                "year": year,
                "output_regularization": value_lambda,
                "grid_aligned": grid_ok,
                "valid_mask_mismatch_pixels": valid_mismatch,
                "county_count": np.nan,
                "common_valid_pixels": valid_count,
                "max_abs_county_weighted_mean_difference_yuan": np.nan,
                "passed": bool(grid_ok and valid_mismatch == 0 and valid_count > 0),
                "file": str(candidate_path.relative_to(ROOT)).replace("\\", "/"),
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "stage26_output_audit.csv", index=False, encoding="utf-8-sig")
    audit_summary = {
        "checks": int(len(result)),
        "passed": int(result["passed"].sum()),
        "failed": int((~result["passed"]).sum()),
        "all_passed": bool(result["passed"].all()),
    }
    (OUT / "stage26_output_audit.json").write_text(
        json.dumps(audit_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def build_report(uniform: pd.DataFrame, cv: pd.DataFrame, cv_calibrated: pd.DataFrame, maps: pd.DataFrame, epochs: int, pixels: int) -> str:
    lines = [
        "# 县内均一基线与输出正则化敏感性实验",
        "",
        "## 解释边界",
        "",
        "本实验用于衡量正式1 km产品相对于县内均一表面的空间信息量，并检验县内排序对像元输出收缩系数的稳定性。",
        "它不构成1 km收入直接精度验证，因为不存在覆盖全国且口径一致的像元收入真值。",
        "",
        "## 县内均一基线",
        "",
        "均一基线把正式产品在每个县内的农村人口加权均值赋给该县全部有效农村像元，因此基线与正式产品的县级人口加权均值完全一致，唯一差别是县内空间异质性。",
        "",
        "| 年份 | 有效县 | 有效像元 | 县内加权CV中位数 | 县内加权CV P90 | 人口加权像元绝对差均值（元） |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in uniform.iterrows():
        lines.append(
            f"| {int(row['year'])} | {int(row['county_count'])} | {int(row['valid_pixels']):,} | "
            f"{row['median_county_weighted_cv_pct']:.2f}% | {row['p90_county_weighted_cv_pct']:.2f}% | "
            f"{row['population_weighted_mean_abs_pixel_difference_yuan']:.1f} |"
        )
    lines += [
        "",
        "## 正则化敏感性",
        "",
        f"训练抽样、随机种子、模型结构、学习率和训练轮次保持固定。每个县抽样{pixels}个像元，训练{epochs}轮。",
        "变化的参数是将像元对数收入向训练标签总体均值收缩的显式系数，取0.0005、0.001和0.002。AdamW权重衰减固定为1e-4。该参数不是空间邻域平滑系数。",
        "",
        "### 2020年五折留省结果",
        "",
        "| 收缩系数 | n | R² | MAE（元） | RMSE（元） | MAPE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in cv.iterrows():
        lines.append(
            f"| {row['output_regularization']:.4f} | {int(row['n'])} | {row['r2']:.4f} | "
            f"{row['mae']:.1f} | {row['rmse']:.1f} | {row['mape']:.2f}% |"
        )
    lines += [
        "",
        "### 引入官方省级约束后的县级回聚合一致性",
        "",
        "下表在各留出省内部施加官方2020年省级农村收入约束。该结果与最终产品生成方式一致，但使用了测试省官方信息，因此不是完全独立验证。",
        "",
        "| 收缩系数 | R²（校准前） | R²（校准后） | MAPE（校准前） | MAPE（校准后） |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, row in cv_calibrated.iterrows():
        lines.append(
            f"| {row['output_regularization']:.4f} | {row['r2_before']:.4f} | {row['r2_after']:.4f} | "
            f"{row['mape_before']:.2f}% | {row['mape_after']:.2f}% |"
        )
    lines += [
        "",
        "### 相对于冻结系数0.001的地图稳定性",
        "",
        "| 年份 | 对比系数 | 像元Spearman | 县内Spearman中位数 | 县内Spearman P10 | 县内前20%重叠中位数 | 县内后20%重叠中位数 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in maps.iterrows():
        lines.append(
            f"| {int(row['year'])} | {row['output_regularization']:.4f} | {row['sampled_pixel_spearman']:.4f} | "
            f"{row['median_county_spearman']:.4f} | {row['p10_county_spearman']:.4f} | "
            f"{row['median_county_top20_overlap_fraction']:.3f} | {row['median_county_bottom20_overlap_fraction']:.3f} |"
        )
    lines += [
        "",
        "## 使用建议",
        "",
        "县内均一基线只能证明正式产品增加了县内空间信息，不能证明该信息等同于真实像元收入。正则化敏感性用于判断空间排序是否依赖单一参数设定。论文中应将二者表述为空间信息增益和格局稳定性证据，而不是像元精度。",
    ]
    return "\n".join(lines)


def write_manifest() -> None:
    files = sorted(path for path in OUT.rglob("*") if path.is_file())
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "county_uniform_baseline_and_output_regularization_sensitivity",
        "canonical_product": str(CANONICAL.relative_to(ROOT)).replace("\\", "/"),
        "target": str(TARGET.relative_to(ROOT)).replace("\\", "/"),
        "target_sha256": sha256(TARGET),
        "county_id_sha256": sha256(COUNTY_PATH),
        "years_uniform_baseline": YEARS,
        "years_regularization_maps": SENSITIVITY_YEARS,
        "output_regularization_values": LAMBDAS,
        "frozen_output_regularization": FROZEN_LAMBDA,
        "interpretation": "Robustness and spatial-information analysis; not direct 1-km accuracy validation.",
        "files": [
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
            if path.name != "run_manifest.json"
        ],
    }
    (OUT / "run_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=["baseline", "regularization", "summarize", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--pixels-per-county", type=int, default=128)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--row-block", type=int, default=128)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.threads))
    weak.set_seed(weak.RANDOM_STATE)

    uniform_path = BASELINE_OUT / "county_uniform_baseline_summary_2019_2023.csv"
    cv_path = REG_OUT / "regularization_leave_province_2020_summary.csv"
    cv_calibrated_path = REG_OUT / "regularization_leave_province_2020_before_after.csv"
    map_path_csv = REG_OUT / "regularization_map_stability_summary.csv"
    if args.run in {"baseline", "all"}:
        _, uniform_df = run_uniform_baselines()
    elif uniform_path.exists():
        uniform_df = pd.read_csv(uniform_path)
    else:
        uniform_df = pd.DataFrame()
    if args.run in {"regularization", "all"}:
        cv_df, map_df, _ = run_regularization_sensitivity(
            args.epochs, args.pixels_per_county, args.row_block
        )
    elif cv_path.exists() and map_path_csv.exists():
        cv_df = pd.read_csv(cv_path)
        map_df = pd.read_csv(map_path_csv)
    else:
        cv_df = pd.DataFrame()
        map_df = pd.DataFrame()
    calibration_inputs_exist = all(
        (REG_OUT / lambda_slug(value_lambda) / "leave_province_2020_county_predictions.csv").exists()
        for value_lambda in LAMBDAS
    )
    if calibration_inputs_exist and (
        args.run in {"regularization", "summarize", "all"} or not cv_calibrated_path.exists()
    ):
        cv_calibrated_df = postprocess_calibrated_cv()
    elif cv_calibrated_path.exists():
        cv_calibrated_df = pd.read_csv(cv_calibrated_path)
    else:
        cv_calibrated_df = pd.DataFrame()
    if uniform_path.exists() and cv_path.exists() and cv_calibrated_path.exists() and map_path_csv.exists():
        report = build_report(
            pd.read_csv(uniform_path), pd.read_csv(cv_path), pd.read_csv(cv_calibrated_path),
            pd.read_csv(map_path_csv),
            args.epochs, args.pixels_per_county,
        )
        (OUT / "STAGE26_SPATIAL_VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
        audit_stage26_outputs()
    write_manifest()
    print(f"Finished. Results: {OUT}", flush=True)


if __name__ == "__main__":
    main()
