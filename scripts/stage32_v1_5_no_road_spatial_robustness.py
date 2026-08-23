"""v1.5-no-road county-uniform and parameter-sensitivity experiments.

The canonical configuration is output_regularization=0.001 and 128 sampled
pixels per county. Four one-factor-at-a-time variants are evaluated:
regularization 0.0001 and 0.01 at 128 pixels, and 64 and 256 pixels at the
canonical regularization. Province folds, seeds, targets, masks and the
road-density exclusion are held fixed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for dependency_path in reversed([
    ROOT / ".codex_pydeps_stage8",
    ROOT / ".codex_pydeps",
    ROOT / "stage6_pydeps",
]):
    if dependency_path.exists():
        sys.path.insert(0, str(dependency_path))

import numpy as np
import pandas as pd
import rasterio
from scipy.stats import spearmanr
import torch

from feature_ablation import patch_final_map_features, patch_weak_pixel_loader
from repro_config import deep_get, int_list, load_config
import stage3_weak_supervised_downscaling as weak
import stage3_spatiotemporal_validation as validation
import stage5_final_maps as final_maps
import stage5_create_unbounded_maps as unbounded
import stage26_county_uniform_regularization_sensitivity as legacy_spatial
import stage27_multiyear_leave_province_validation as stage27


CONFIG = ROOT / "config" / "workflow_v1.5_no_road_core.json"
OUT = ROOT / "outputs" / "stage32_v1_5_no_road_spatial_robustness"
UNIFORM_OUT = OUT / "county_uniform_baseline"
VARIANT_OUT = OUT / "parameter_sensitivity"
CANONICAL_MAP_ROOT = ROOT / "outputs" / "sensitivity_no_road_v1_4" / "stage5_final_maps"
CANONICAL_CV_ROOT = ROOT / "outputs" / "multiyear_leave_province_v1_5_no_road"
CANONICAL_PREDICTIONS = (
    CANONICAL_CV_ROOT
    / "multiyear_leave_province_2019_2023_strict_county_year_predictions.csv"
)
CANONICAL_MANIFEST = (
    CANONICAL_CV_ROOT
    / "multiyear_leave_province_2019_2023_strict_manifest.json"
)
YEARS = [2019, 2020, 2021, 2022, 2023]
CANONICAL_LAMBDA = 0.001
CANONICAL_PIXELS = 128
VARIANTS = {
    "regularization_0p0001_pixels128": {"output_regularization": 0.0001, "pixels_per_county": 128},
    "regularization_0p01_pixels128": {"output_regularization": 0.01, "pixels_per_county": 128},
    "regularization_0p001_pixels64": {"output_regularization": 0.001, "pixels_per_county": 64},
    "regularization_0p001_pixels256": {"output_regularization": 0.001, "pixels_per_county": 256},
}
NODATA = -9999.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def annual_map(root: Path, year: int) -> Path:
    return root / f"rural_income_{year}_final_calibrated_unbounded_2019yuan_strict.tif"


def change_map(root: Path, kind: str) -> Path:
    if kind == "absolute":
        return root / "rural_income_change_2019_2023_absolute_unbounded_2019yuan.tif"
    return root / "rural_income_change_2019_2023_percent_unbounded.tif"


def prepare_no_road_modules(target_path: Path, excluded_features: list[str]) -> None:
    weak.load_targets = stage27.target_loader(target_path)
    if "road_density" not in excluded_features:
        raise RuntimeError("The v1.5 robustness workflow must exclude road_density")
    if "road_density" in weak.load_pixel_panel(2019, "strict")["band_names"]:
        patch_weak_pixel_loader(weak, excluded_features)
    validation.load_pixel_panel = weak.load_pixel_panel
    validation.load_targets = weak.load_targets
    if "road_density" in final_maps.BAND_NAMES:
        patch_final_map_features(final_maps, excluded_features)


def run_uniform() -> None:
    expected = [annual_map(CANONICAL_MAP_ROOT, year) for year in YEARS]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    legacy_spatial.OUT = OUT
    legacy_spatial.BASELINE_OUT = UNIFORM_OUT
    legacy_spatial.CANONICAL = CANONICAL_MAP_ROOT
    legacy_spatial.TARGET = ROOT / "data" / "corrections" / "county_income_2019_2023_deflated_official77_corrected_v4.csv"
    legacy_spatial.run_uniform_baselines()
    summary = pd.read_csv(UNIFORM_OUT / "county_uniform_baseline_summary_2019_2023.csv")
    public_summary = summary.drop(columns=["source_map", "baseline_map"], errors="ignore")
    public_summary.to_csv(
        UNIFORM_OUT / "county_uniform_baseline_summary_public.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest = {
        "experiment": "county_uniform_baseline_v1.5_no_road",
        "canonical_maps": "outputs/sensitivity_no_road_v1_4/stage5_final_maps",
        "canonical_map_hashes": {
            str(year): sha256(annual_map(CANONICAL_MAP_ROOT, year)) for year in YEARS
        },
        "definition": (
            "Each valid rural pixel receives the population-weighted county mean of "
            "the canonical v1.5-no-road product. County means are preserved exactly."
        ),
        "interpretation": (
            "Descriptive null surface for within-county spatial information; not an "
            "independent pixel-accuracy benchmark."
        ),
        "years": YEARS,
        "road_density_used": False,
        "county_level_detail_or_baseline_rasters_for_public_release": False,
    }
    (UNIFORM_OUT / "county_uniform_baseline_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def metric_dict(frame: pd.DataFrame) -> dict[str, float | int]:
    return weak.metric_dict(
        frame["true_2019yuan"].to_numpy(float),
        frame["pred_weak_2019yuan"].to_numpy(float),
    )


def grouped_metrics(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    key = columns[0] if len(columns) == 1 else columns
    for values, group in frame.groupby(key, dropna=False):
        values = (values,) if len(columns) == 1 else values
        row = {name: value for name, value in zip(columns, values)}
        row.update(metric_dict(group))
        rows.append(row)
    return pd.DataFrame(rows)


def fold_definitions() -> list[tuple[int, set[str]]]:
    manifest = json.loads(CANONICAL_MANIFEST.read_text(encoding="utf-8"))
    return [
        (int(row["fold"]), set(map(str, row["held_out_provinces"])))
        for row in manifest["fold_assignments"]
    ]


def run_cv_variant(name: str, settings: dict[str, float | int], config: dict) -> None:
    root = VARIANT_OUT / name
    overall_path = root / "multiyear_leave_province_overall_metrics.json"
    if overall_path.exists():
        print(f"[cv] {name}: existing output retained", flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    years = int_list(
        deep_get(config, "model.multiyear_leave_province_validation.years"), YEARS
    )
    mask = str(deep_get(config, "model.multiyear_leave_province_validation.mask", "strict"))
    epochs = int(deep_get(config, "model.multiyear_leave_province_validation.epochs", 150))
    lr = float(deep_get(config, "model.multiyear_leave_province_validation.lr", 0.001))
    pixels = int(settings["pixels_per_county"])
    value_lambda = float(settings["output_regularization"])

    print(f"[cv] {name}: loading fixed samples", flush=True)
    sampled = {
        year: stage27.sample_year(
            year, mask, pixels, seed=weak.RANDOM_STATE + 50_000 + year
        )
        for year in years
    }
    predictions = []
    training_rows = []
    for fold, held_out in fold_definitions():
        print(f"[cv] {name}: fold {fold}/5", flush=True)
        model, imputer, scaler, training = stage27.fit_fold(
            sampled,
            years,
            held_out,
            epochs,
            lr,
            seed=weak.RANDOM_STATE + 27_000 + fold,
            output_regularization=value_lambda,
        )
        training.update({
            "fold": fold,
            "held_out_provinces": ";".join(sorted(held_out)),
        })
        training_rows.append(training)
        for year in years:
            panel = weak.load_pixel_panel(year, mask)
            test_groups, pred, weight_sum, pixel_count = stage27.aggregate_held_out(
                model, imputer, scaler, panel, year, held_out
            )
            meta = panel["county_meta"].iloc[test_groups].copy()
            meta["year"] = year
            meta["fold"] = fold
            meta["true_2019yuan"] = panel["target"][test_groups]
            meta["pred_weak_2019yuan"] = pred
            meta["rural_weight_sum"] = weight_sum
            meta["rural_pixel_count"] = pixel_count
            predictions.append(meta)
            del panel
            gc.collect()
        del model, imputer, scaler
        gc.collect()

    prediction = pd.concat(predictions, ignore_index=True)
    overall = metric_dict(prediction)
    overall.update({
        "variant": name,
        "years": years,
        "folds": 5,
        "pixels_per_county": pixels,
        "output_regularization": value_lambda,
        "road_density_used": False,
    })
    internal = root / "restricted_internal"
    internal.mkdir(exist_ok=True)
    prediction.to_csv(
        internal / "county_year_predictions.csv", index=False, encoding="utf-8-sig"
    )
    grouped_metrics(prediction, ["fold"]).to_csv(
        root / "fold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    grouped_metrics(prediction, ["year"]).to_csv(
        root / "year_metrics.csv", index=False, encoding="utf-8-sig"
    )
    grouped_metrics(prediction, ["fold", "year"]).to_csv(
        root / "fold_year_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(training_rows).to_csv(
        root / "training_counts_by_fold.csv", index=False, encoding="utf-8-sig"
    )
    overall_path.write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "variant": name,
        "config": "config/workflow_v1.5_no_road_core.json",
        "config_sha256": sha256(CONFIG),
        "canonical_fold_manifest": str(CANONICAL_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "canonical_fold_manifest_sha256": sha256(CANONICAL_MANIFEST),
        "fixed_fold_assignments": [
            {"fold": fold, "held_out_provinces": sorted(held)}
            for fold, held in fold_definitions()
        ],
        "years": years,
        "mask": mask,
        "pixels_per_county": pixels,
        "epochs": epochs,
        "learning_rate": lr,
        "output_regularization": value_lambda,
        "excluded_features": ["road_density"],
        "random_seed_policy": {
            "pixel_sampling": "42 + 50000 + year",
            "model_fit": "42 + 27000 + fold",
        },
        "county_predictions_publicly_released": False,
    }
    (root / "cv_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_all_cv(config: dict) -> None:
    for name, settings in VARIANTS.items():
        run_cv_variant(name, settings, config)


def run_map_variant(name: str, settings: dict[str, float | int], config: dict) -> None:
    root = VARIANT_OUT / name / "maps"
    expected = [annual_map(root, year) for year in YEARS]
    expected += [change_map(root, "absolute"), change_map(root, "percent")]
    if all(path.exists() for path in expected):
        print(f"[maps] {name}: existing maps retained", flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    pixels = int(settings["pixels_per_county"])
    value_lambda = float(settings["output_regularization"])
    epochs = int(deep_get(config, "model.final_maps.epochs", 150))
    lr = float(deep_get(config, "model.final_maps.lr", 0.001))
    row_block = int(deep_get(config, "model.final_maps.row_block", 128))
    print(f"[maps] {name}: fitting joint model", flush=True)
    final_maps.OUT = root
    final_maps.MODEL_DIR = root / "model"
    unbounded.OUT = root
    model, imputer, scaler, model_meta = final_maps.train_and_save(
        pixels, epochs, lr, value_lambda
    )
    for year in YEARS:
        print(f"[maps] {name}: {year}", flush=True)
        final_maps.predict_initial(year, model, imputer, scaler, row_block)
        unbounded.make_year(year, row_block)
        gc.collect()
    change_summary = unbounded.make_change_maps()
    del model, imputer, scaler
    gc.collect()
    manifest = {
        "variant": name,
        "model": model_meta,
        "change_maps": change_summary,
        "excluded_features": ["road_density"],
        "province_year_calibration": "unbounded multiplicative factor",
        "canonical_comparison_maps": str(CANONICAL_MAP_ROOT.relative_to(ROOT)).replace("\\", "/"),
    }
    (root / "map_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_all_maps(config: dict) -> None:
    for name, settings in VARIANTS.items():
        run_map_variant(name, settings, config)


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    return float(spearmanr(x, y).statistic)


def top_bottom_overlap(x: np.ndarray, y: np.ndarray, upper: bool, share: float = 0.1) -> float:
    n = len(x)
    k = max(1, int(math.ceil(share * n)))
    if upper:
        left = np.argpartition(x, n - k)[n - k:]
        right = np.argpartition(y, n - k)[n - k:]
    else:
        left = np.argpartition(x, k - 1)[:k]
        right = np.argpartition(y, k - 1)[:k]
    return float(len(np.intersect1d(left, right)) / k)


def compare_annual_maps(
    reference_path: Path,
    candidate_path: Path,
    year: int,
    variant: str,
    settings: dict[str, float | int],
) -> tuple[dict[str, object], pd.DataFrame]:
    county_path = ROOT / "features" / "county_id.tif"
    weight_path = ROOT / "data" / "processed" / "rural_weights" / f"rural_pop_weight_{year}_strict.tif"
    with rasterio.open(reference_path) as ref_src, rasterio.open(candidate_path) as cand_src, rasterio.open(county_path) as county_src, rasterio.open(weight_path) as weight_src:
        if not (
            ref_src.shape == cand_src.shape == county_src.shape == weight_src.shape
            and ref_src.transform == cand_src.transform == county_src.transform == weight_src.transform
        ):
            raise RuntimeError(f"Grid mismatch for {variant}, {year}")
        ref = ref_src.read(1).astype(np.float32)
        cand = cand_src.read(1).astype(np.float32)
        codes = county_src.read(1).astype(np.int64)
        weights = weight_src.read(1).astype(np.float32)
        valid = (
            np.isfinite(ref) & np.isfinite(cand)
            & (ref != ref_src.nodata) & (cand != cand_src.nodata)
            & (codes > 0) & np.isfinite(weights) & (weights > 0)
        )
    a = ref[valid].astype(np.float64)
    b = cand[valid].astype(np.float64)
    c = codes[valid]
    w = weights[valid].astype(np.float64)
    rng = np.random.default_rng(32000 + year)
    take = np.arange(len(a))
    if len(take) > 250_000:
        take = np.sort(rng.choice(take, 250_000, replace=False))
    sa, sb = a[take], b[take]
    difference = b - a

    county_rows = []
    order = np.argsort(c, kind="stable")
    sorted_codes = c[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1], True])
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        idx = order[start:stop]
        if len(idx) < 10:
            continue
        full_idx = idx
        wa = w[full_idx]
        av = a[full_idx]
        bv = b[full_idx]
        ma = float(np.sum(av * wa) / np.sum(wa))
        mb = float(np.sum(bv * wa) / np.sum(wa))
        sda = float(np.sqrt(np.sum(wa * (av - ma) ** 2) / np.sum(wa)))
        sdb = float(np.sqrt(np.sum(wa * (bv - mb) ** 2) / np.sum(wa)))
        if len(idx) > 300:
            idx = idx[np.linspace(0, len(idx) - 1, 300, dtype=int)]
        county_rows.append({
            "variant": variant,
            "year": year,
            "county_code": int(c[idx[0]]),
            "sampled_rural_pixels": int(len(idx)),
            "within_county_spearman": rank_correlation(a[idx], b[idx]),
            "top10_overlap_fraction": top_bottom_overlap(a[idx], b[idx], True),
            "bottom10_overlap_fraction": top_bottom_overlap(a[idx], b[idx], False),
            "reference_weighted_cv_pct": 100.0 * sda / max(ma, 1.0),
            "candidate_weighted_cv_pct": 100.0 * sdb / max(mb, 1.0),
        })
    county = pd.DataFrame(county_rows)
    cv_valid = (
        np.isfinite(county["reference_weighted_cv_pct"])
        & np.isfinite(county["candidate_weighted_cv_pct"])
    )
    summary = {
        "variant": variant,
        "year": year,
        "output_regularization": settings["output_regularization"],
        "pixels_per_county": settings["pixels_per_county"],
        "common_valid_pixels": int(len(a)),
        "sampled_pixel_pearson": float(np.corrcoef(sa, sb)[0, 1]),
        "sampled_pixel_spearman": rank_correlation(sa, sb),
        "sampled_top10_overlap_fraction": top_bottom_overlap(sa, sb, True),
        "sampled_bottom10_overlap_fraction": top_bottom_overlap(sa, sb, False),
        "mean_absolute_difference_yuan": float(np.mean(np.abs(difference))),
        "median_absolute_difference_yuan": float(np.median(np.abs(difference))),
        "p95_absolute_difference_yuan": float(np.quantile(np.abs(difference), 0.95)),
        "population_weighted_mean_absolute_difference_yuan": float(
            np.sum(np.abs(difference) * w) / np.sum(w)
        ),
        "median_county_within_spearman": float(county["within_county_spearman"].median()),
        "p10_county_within_spearman": float(county["within_county_spearman"].quantile(0.1)),
        "median_county_top10_overlap_fraction": float(county["top10_overlap_fraction"].median()),
        "median_county_bottom10_overlap_fraction": float(county["bottom10_overlap_fraction"].median()),
        "county_cv_pearson": float(np.corrcoef(
            county.loc[cv_valid, "reference_weighted_cv_pct"],
            county.loc[cv_valid, "candidate_weighted_cv_pct"],
        )[0, 1]),
        "median_reference_county_cv_pct": float(county["reference_weighted_cv_pct"].median()),
        "median_candidate_county_cv_pct": float(county["candidate_weighted_cv_pct"].median()),
    }
    del ref, cand, codes, weights, a, b, c, w
    gc.collect()
    return summary, county


def compare_change_maps(
    reference_path: Path,
    candidate_path: Path,
    kind: str,
    variant: str,
    settings: dict[str, float | int],
) -> dict[str, object]:
    with rasterio.open(reference_path) as ref_src, rasterio.open(candidate_path) as cand_src:
        if ref_src.shape != cand_src.shape or ref_src.transform != cand_src.transform:
            raise RuntimeError(f"Change-map grid mismatch: {variant}, {kind}")
        ref = ref_src.read(1).astype(np.float32)
        cand = cand_src.read(1).astype(np.float32)
        valid = (
            np.isfinite(ref) & np.isfinite(cand)
            & (ref != ref_src.nodata) & (cand != cand_src.nodata)
        )
    a = ref[valid].astype(np.float64)
    b = cand[valid].astype(np.float64)
    rng = np.random.default_rng(32999 if kind == "absolute" else 33000)
    take = np.arange(len(a))
    if len(take) > 250_000:
        take = np.sort(rng.choice(take, 250_000, replace=False))
    sa, sb = a[take], b[take]
    difference = b - a
    result = {
        "variant": variant,
        "change_kind": kind,
        "output_regularization": settings["output_regularization"],
        "pixels_per_county": settings["pixels_per_county"],
        "common_valid_pixels": int(len(a)),
        "sampled_pearson": float(np.corrcoef(sa, sb)[0, 1]),
        "sampled_spearman": rank_correlation(sa, sb),
        "top10_overlap_fraction": top_bottom_overlap(sa, sb, True),
        "bottom10_overlap_fraction": top_bottom_overlap(sa, sb, False),
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
        "median_absolute_difference": float(np.median(np.abs(difference))),
        "p95_absolute_difference": float(np.quantile(np.abs(difference), 0.95)),
    }
    del ref, cand, a, b
    gc.collect()
    return result


def compare_cv_predictions(name: str, settings: dict[str, float | int]) -> dict[str, object]:
    canonical = pd.read_csv(CANONICAL_PREDICTIONS)
    candidate_path = VARIANT_OUT / name / "restricted_internal" / "county_year_predictions.csv"
    candidate = pd.read_csv(candidate_path)
    keys = ["year", "county_code", "fold"]
    paired = canonical.merge(
        candidate,
        on=keys,
        suffixes=("_canonical", "_candidate"),
        validate="one_to_one",
    )
    a = paired["pred_weak_2019yuan_canonical"].to_numpy(float)
    b = paired["pred_weak_2019yuan_candidate"].to_numpy(float)
    canonical_metric = metric_dict(pd.DataFrame({
        "true_2019yuan": paired["true_2019yuan_canonical"],
        "pred_weak_2019yuan": paired["pred_weak_2019yuan_canonical"],
    }))
    candidate_metric = metric_dict(pd.DataFrame({
        "true_2019yuan": paired["true_2019yuan_candidate"],
        "pred_weak_2019yuan": paired["pred_weak_2019yuan_candidate"],
    }))
    return {
        "variant": name,
        "output_regularization": settings["output_regularization"],
        "pixels_per_county": settings["pixels_per_county"],
        "common_test_records": int(len(paired)),
        "canonical_r2": canonical_metric["r2"],
        "candidate_r2": candidate_metric["r2"],
        "delta_r2_candidate_minus_canonical": candidate_metric["r2"] - canonical_metric["r2"],
        "canonical_mape": canonical_metric["mape"],
        "candidate_mape": candidate_metric["mape"],
        "delta_mape_candidate_minus_canonical": candidate_metric["mape"] - canonical_metric["mape"],
        "paired_prediction_pearson": float(np.corrcoef(a, b)[0, 1]),
        "paired_prediction_spearman": rank_correlation(a, b),
        "mean_absolute_prediction_difference": float(np.mean(np.abs(b - a))),
    }


def run_comparisons() -> None:
    annual_rows = []
    county_parts = []
    change_rows = []
    cv_rows = []
    for name, settings in VARIANTS.items():
        root = VARIANT_OUT / name / "maps"
        cv_rows.append(compare_cv_predictions(name, settings))
        for year in YEARS:
            summary, county = compare_annual_maps(
                annual_map(CANONICAL_MAP_ROOT, year),
                annual_map(root, year),
                year,
                name,
                settings,
            )
            annual_rows.append(summary)
            county_parts.append(county)
        for kind in ["absolute", "percent"]:
            change_rows.append(compare_change_maps(
                change_map(CANONICAL_MAP_ROOT, kind),
                change_map(root, kind),
                kind,
                name,
                settings,
            ))
    pd.DataFrame(cv_rows).to_csv(
        OUT / "parameter_sensitivity_cv_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(annual_rows).to_csv(
        OUT / "parameter_sensitivity_annual_map_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(county_parts, ignore_index=True).to_csv(
        OUT / "parameter_sensitivity_county_spatial_stability_internal.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(change_rows).to_csv(
        OUT / "parameter_sensitivity_change_map_summary.csv", index=False, encoding="utf-8-sig"
    )


def build_summary(config: dict) -> None:
    uniform = pd.read_csv(UNIFORM_OUT / "county_uniform_baseline_summary_public.csv")
    cv = pd.read_csv(OUT / "parameter_sensitivity_cv_summary.csv")
    annual = pd.read_csv(OUT / "parameter_sensitivity_annual_map_summary.csv")
    change = pd.read_csv(OUT / "parameter_sensitivity_change_map_summary.csv")
    decision = {
        "experiment": "v1.5_no_road_spatial_robustness",
        "county_uniform_baseline_is_accuracy_validation": False,
        "parameter_variants": VARIANTS,
        "checks": {
            "maximum_absolute_delta_r2": float(cv["delta_r2_candidate_minus_canonical"].abs().max()),
            "maximum_absolute_delta_mape_points": float(cv["delta_mape_candidate_minus_canonical"].abs().max()),
            "minimum_annual_pixel_pearson": float(annual["sampled_pixel_pearson"].min()),
            "minimum_annual_pixel_spearman": float(annual["sampled_pixel_spearman"].min()),
            "minimum_median_county_within_spearman": float(annual["median_county_within_spearman"].min()),
            "minimum_county_cv_pearson": float(annual["county_cv_pearson"].min()),
            "minimum_change_map_spearman": float(change["sampled_spearman"].min()),
        },
        "interpretation_scope": (
            "Robustness of county-level validation and within-county spatial patterns; "
            "not independent 1 km accuracy validation."
        ),
    }
    (OUT / "spatial_robustness_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "version": "v1.5-no-road",
        "config": "config/workflow_v1.5_no_road_core.json",
        "config_sha256": sha256(CONFIG),
        "canonical_target": deep_get(config, "paths.corrected_county_income"),
        "canonical_map_root": str(CANONICAL_MAP_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "canonical_cv_predictions": str(CANONICAL_PREDICTIONS.relative_to(ROOT)).replace("\\", "/"),
        "canonical_cv_predictions_sha256": sha256(CANONICAL_PREDICTIONS),
        "canonical_fold_manifest_sha256": sha256(CANONICAL_MANIFEST),
        "excluded_features": ["road_density"],
        "canonical_configuration": {
            "output_regularization": CANONICAL_LAMBDA,
            "pixels_per_county": CANONICAL_PIXELS,
        },
        "variants": VARIANTS,
        "years": YEARS,
        "public_release_policy": (
            "Only aggregate summary tables, manifests and this decision record are "
            "added to the frozen package. Variant rasters and county-level tables remain internal."
        ),
    }
    (OUT / "spatial_robustness_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# v1.5无道路版本县内均一基线与参数敏感性",
        "",
        "县内均一基线保持每个县的人口加权平均值不变，仅移除县内像元差异。",
        "它用于说明机器学习表面包含多少县内空间信息，不是像元精度验证。",
        "",
        "## 县内均一基线",
        "",
        "| 年份 | 县数 | 有效像元 | 县内加权CV中位数(%) | 县内加权CV P90(%) | 人口加权像元绝对差(元) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in uniform.itertuples(index=False):
        lines.append(
            f"| {int(row.year)} | {int(row.county_count)} | {int(row.valid_pixels):,} | "
            f"{row.median_county_weighted_cv_pct:.3f} | {row.p90_county_weighted_cv_pct:.3f} | "
            f"{row.population_weighted_mean_abs_pixel_difference_yuan:.1f} |"
        )
    lines.extend([
        "",
        "## 固定五折留省验证敏感性",
        "",
        "| 配置 | λ | 每县像元上限 | R2 | ΔR2 | MAPE(%) | ΔMAPE | 预测Pearson |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in cv.itertuples(index=False):
        lines.append(
            f"| {row.variant} | {row.output_regularization:g} | {int(row.pixels_per_county)} | "
            f"{row.candidate_r2:.4f} | {row.delta_r2_candidate_minus_canonical:+.4f} | "
            f"{row.candidate_mape:.2f} | {row.delta_mape_candidate_minus_canonical:+.2f} | "
            f"{row.paired_prediction_pearson:.5f} |"
        )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "这些实验评估县级验证指标和县内空间结构对参数选择的稳健性。",
        "没有乡镇、村庄或住户尺度真实收入时，结果不能解释为1 km像元精度。",
    ])
    (OUT / "SPATIAL_ROBUSTNESS_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["uniform", "cv", "maps", "compare", "summarize", "all"],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(CONFIG)
    target_path = stage27.resolve_path(deep_get(config, "paths.corrected_county_income"))
    excluded_features = list(deep_get(config, "model.feature_ablation.exclude_features", []))
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(deep_get(config, "model.final_maps.threads", 8)))
    prepare_no_road_modules(target_path, excluded_features)
    if args.phase in {"uniform", "all"}:
        run_uniform()
    if args.phase in {"cv", "all"}:
        run_all_cv(config)
    if args.phase in {"maps", "all"}:
        run_all_maps(config)
    if args.phase in {"compare", "all"}:
        run_comparisons()
    if args.phase in {"summarize", "all"}:
        build_summary(config)


if __name__ == "__main__":
    main()
