# -*- coding: utf-8 -*-
"""Train the final five-year weak-supervision model and create calibrated maps.

Outputs are restricted to mainland province codes with positive strict rural
population weights. Income is expressed in constant 2019 yuan/person/year.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import torch

from repro_config import deep_get, int_list, load_config
from feature_ablation import patch_final_map_features, patch_weak_pixel_loader
import stage3_weak_supervised_downscaling as weak_module
import stage3_spatiotemporal_validation as spatiotemporal_module
from stage3_weak_supervised_downscaling import ROOT, IncomeMLP, RANDOM_STATE, set_seed
from stage3_spatiotemporal_validation import (
    YEARS,
    add_year_column,
    fit_bag_model,
    sample_year_bags,
)


FEATURES = ROOT / "features"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "stage5_final_maps"
MODEL_DIR = OUT / "model"
NODATA = -9999.0
LOWER_INCOME = 3000.0
UPPER_INCOME = 80000.0

BAND_NAMES = [
    "ntl", "ndvi_mean", "ndvi_max", "ndvi_std", "evi_mean", "npp",
    "tree_frac", "shrub_frac", "grass_frac", "crop_frac", "built_frac",
    "bare_frac", "water_frac", "wetland_frac", "dem", "slope", "aspect",
    "temp_mean", "precip_sum", "solar_mean", "B2", "B3", "B4", "B8",
    "ndvi_s2", "ndwi", "ndbi", "road_density",
]
SCALE_BANDS = {
    "ndvi_mean", "ndvi_max", "ndvi_std", "evi_mean", "npp",
    "B2", "B3", "B4", "B8",
}


def feature_specs(year: int):
    return [
        (FEATURES / f"ntl_{year}_1km.tif", ["ntl"]),
        (FEATURES / f"vegetation_{year}_1km.tif",
         ["ndvi_mean", "ndvi_max", "ndvi_std", "evi_mean"]),
        (FEATURES / f"npp_{year}_1km.tif", ["npp"]),
        (FEATURES / f"landcover_{year}_1km.tif",
         ["tree_frac", "shrub_frac", "grass_frac", "crop_frac", "built_frac",
          "bare_frac", "water_frac", "wetland_frac"]),
        (FEATURES / "terrain_1km.tif", ["dem", "slope", "aspect"]),
        (FEATURES / f"climate_{year}_1km.tif",
         ["temp_mean", "precip_sum", "solar_mean"]),
        (FEATURES / f"sentinel2_{year}_summer_1km.tif",
         ["B2", "B3", "B4", "B8", "ndvi_s2", "ndwi", "ndbi"]),
        (FEATURES / "road_density_1km.tif", ["road_density"]),
    ]


def official_income(year: int) -> pd.DataFrame:
    path = PROCESSED / "province_income_2019_2023_deflated.csv"
    df = pd.read_csv(path)
    df = df[(df["year"].astype(int) == year)].copy()
    df = df.dropna(subset=["province_code", "province_std", "rural_income_2019yuan"])
    df["province_code"] = df["province_code"].astype(int)
    # Mainland 31 province-level regions only.
    df = df[df["province_code"] < 710000]
    return df[["province_code", "province_std", "rural_income_2019yuan",
               "rural_income_nominal", "price_index_2019_base"]].sort_values("province_code")


def train_and_save(
    pixels_per_county: int,
    epochs: int,
    lr: float,
    output_regularization: float = 0.001,
    hidden1: int = 64,
    hidden2: int = 32,
    dropout: float = 0.05,
    excluded_features: list[str] | None = None,
):
    print("Preparing final training samples from 2019-2023 ...")
    samples = {}
    reference_bands = None
    for year in YEARS:
        samples[year] = sample_year_bags(
            year, "strict", pixels_per_county, RANDOM_STATE + 500 + year
        )
        bands = samples[year]["band_names"]
        if reference_bands is None:
            reference_bands = bands
        elif bands != reference_bands:
            raise RuntimeError(f"Feature order differs in {year}.")
    if list(reference_bands) != BAND_NAMES:
        raise RuntimeError(f"Unexpected training feature order: {reference_bands}")

    print("Training final joint model on all five years ...")
    model, imputer, scaler, stats = fit_bag_model(
        samples,
        YEARS,
        epochs=epochs,
        lr=lr,
        seed=RANDOM_STATE + 500,
        output_regularization=output_regularization,
        hidden1=hidden1,
        hidden2=hidden2,
        dropout=dropout,
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "final_joint_model_state.pt")
    with (MODEL_DIR / "preprocessing.pkl").open("wb") as f:
        pickle.dump({"imputer": imputer, "scaler": scaler}, f)
    meta = {
        "training_years": YEARS,
        "mask": "strict",
        "income_unit": "2019 constant yuan/person/year",
        "pixels_per_county": pixels_per_county,
        "epochs": epochs,
        "learning_rate": lr,
        "output_regularization": float(output_regularization),
        "hidden1": hidden1,
        "hidden2": hidden2,
        "dropout": dropout,
        "excluded_features": list(excluded_features or []),
        "input_band_names": BAND_NAMES + ["year_scaled"],
        "n_features": len(BAND_NAMES) + 1,
        "init_log_income": model.init_log_income,
        **stats,
    }
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return model, imputer, scaler, meta


def load_model():
    meta = json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    model = IncomeMLP(
        meta["n_features"],
        meta["init_log_income"],
        hidden1=int(meta.get("hidden1", 64)),
        hidden2=int(meta.get("hidden2", 32)),
        dropout=float(meta.get("dropout", 0.05)),
    )
    state = torch.load(MODEL_DIR / "final_joint_model_state.pt", map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state)
    model.eval()
    with (MODEL_DIR / "preprocessing.pkl").open("rb") as f:
        prep = pickle.load(f)
    return model, prep["imputer"], prep["scaler"], meta


def province_index(province_codes: np.ndarray, official_codes: np.ndarray):
    idx = np.searchsorted(official_codes, province_codes)
    safe = np.minimum(idx, len(official_codes) - 1)
    valid = (idx < len(official_codes)) & (official_codes[safe] == province_codes)
    return idx, valid


def read_features_for_valid(open_specs, window, valid_flat):
    columns = []
    names = []
    for src, band_names in open_specs:
        for band_idx, name in enumerate(band_names, start=1):
            arr = src.read(band_idx, window=window).astype(np.float32)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            if name in SCALE_BANDS:
                arr *= np.float32(0.0001)
            columns.append(arr.ravel()[valid_flat])
            names.append(name)
    if names != BAND_NAMES:
        raise RuntimeError(f"Prediction feature order mismatch: {names}")
    return np.column_stack(columns).astype(np.float32, copy=False)


def output_profile(ref):
    profile = ref.profile.copy()
    profile.update(
        driver="GTiff", dtype="float32", count=1, nodata=NODATA,
        compress="deflate", predictor=3, tiled=True, blockxsize=256,
        blockysize=256, BIGTIFF="IF_SAFER",
    )
    return profile


@torch.no_grad()
def predict_initial(year, model, imputer, scaler, row_block):
    initial_path = OUT / f"rural_income_{year}_initial_2019yuan_strict.tif"
    weight_path = PROCESSED / "rural_weights" / f"rural_pop_weight_{year}_strict.tif"
    official = official_income(year)
    official_codes = official["province_code"].to_numpy(np.int64)
    province_sum = np.zeros(len(official), dtype=np.float64)
    province_weight = np.zeros(len(official), dtype=np.float64)
    pixel_count = np.zeros(len(official), dtype=np.int64)

    with ExitStack() as stack:
        county = stack.enter_context(rasterio.open(FEATURES / "county_id.tif"))
        weight_src = stack.enter_context(rasterio.open(weight_path))
        open_specs = []
        for path, names in feature_specs(year):
            if not path.exists():
                raise FileNotFoundError(path)
            src = stack.enter_context(rasterio.open(path))
            if src.count != len(names) or src.shape != county.shape:
                raise RuntimeError(f"Raster mismatch: {path}")
            open_specs.append((src, names))
        dst = stack.enter_context(rasterio.open(initial_path, "w", **output_profile(county)))

        for row in range(0, county.height, row_block):
            height = min(row_block, county.height - row)
            window = Window(0, row, county.width, height)
            county_arr = county.read(1, window=window).astype(np.int64)
            weights = weight_src.read(1, window=window).astype(np.float32)
            province_codes = (county_arr // 10000) * 10000
            pidx, in_mainland = province_index(province_codes, official_codes)
            valid = (county_arr > 0) & np.isfinite(weights) & (weights > 0) & in_mainland
            valid_flat = valid.ravel()
            block = np.full(county_arr.shape, NODATA, dtype=np.float32)
            if np.any(valid_flat):
                X = read_features_for_valid(open_specs, window, valid_flat)
                X = add_year_column(X, year)
                X = scaler.transform(imputer.transform(X)).astype(np.float32)
                pred = torch.exp(model(torch.from_numpy(X))).cpu().numpy().astype(np.float32)
                block.ravel()[valid_flat] = pred
                ids = pidx.ravel()[valid_flat]
                w = weights.ravel()[valid_flat].astype(np.float64)
                np.add.at(province_sum, ids, pred.astype(np.float64) * w)
                np.add.at(province_weight, ids, w)
                np.add.at(pixel_count, ids, 1)
            dst.write(block, 1, window=window)
            print(f"  {year} initial rows {row + height}/{county.height}", flush=True)

    official["predicted_initial_2019yuan"] = province_sum / np.maximum(province_weight, 1e-9)
    official["strict_rural_weight"] = province_weight
    official["rural_pixel_count"] = pixel_count
    official["unbounded_calibration_factor"] = (
        official["rural_income_2019yuan"] / official["predicted_initial_2019yuan"]
    )
    return initial_path, official


def solve_bounded_factor(raw, weights, target, lower=LOWER_INCOME, upper=UPPER_INCOME):
    """Find alpha so weighted_mean(clip(raw * alpha)) equals target."""
    raw = np.asarray(raw, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    def aggregate(alpha):
        values = np.clip(raw * alpha, lower, upper)
        return float(np.sum(values * weights) / np.sum(weights))

    lo, hi = 0.0, 1.0
    while aggregate(hi) < target and hi < 1024.0:
        hi *= 2.0
    if aggregate(hi) < target:
        raise RuntimeError(f"Cannot reach province target {target} within bounds.")
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if aggregate(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def calibrate_map(year, initial_path, province, row_block):
    final_path = OUT / f"rural_income_{year}_final_calibrated_2019yuan_strict.tif"
    weight_path = PROCESSED / "rural_weights" / f"rural_pop_weight_{year}_strict.tif"
    codes = province["province_code"].to_numpy(np.int64)
    # Collect the valid initial values once so a bounded calibration factor can
    # be solved independently for each province.
    raw_parts = [[] for _ in range(len(province))]
    weight_parts = [[] for _ in range(len(province))]
    with rasterio.open(FEATURES / "county_id.tif") as county, \
         rasterio.open(weight_path) as weight_src, \
         rasterio.open(initial_path) as initial:
        for row in range(0, county.height, row_block):
            height = min(row_block, county.height - row)
            window = Window(0, row, county.width, height)
            county_arr = county.read(1, window=window).astype(np.int64)
            weights = weight_src.read(1, window=window).astype(np.float32)
            raw = initial.read(1, window=window).astype(np.float32)
            province_codes = (county_arr // 10000) * 10000
            pidx, in_mainland = province_index(province_codes, codes)
            valid = (raw != NODATA) & (weights > 0) & in_mainland
            for idx in np.unique(pidx[valid]):
                take = valid & (pidx == idx)
                raw_parts[int(idx)].append(raw[take])
                weight_parts[int(idx)].append(weights[take])

    factors = np.zeros(len(province), dtype=np.float64)
    for idx in range(len(province)):
        raw_values = np.concatenate(raw_parts[idx])
        weight_values = np.concatenate(weight_parts[idx])
        factors[idx] = solve_bounded_factor(
            raw_values,
            weight_values,
            float(province.iloc[idx]["rural_income_2019yuan"]),
        )
    province["calibration_factor"] = factors
    del raw_parts, weight_parts
    gc.collect()

    final_sum = np.zeros(len(province), dtype=np.float64)
    final_weight = np.zeros(len(province), dtype=np.float64)
    sample_values = []
    value_min, value_max = math.inf, -math.inf
    value_sum, value_count = 0.0, 0
    lower_clipped, upper_clipped = 0, 0

    with rasterio.open(FEATURES / "county_id.tif") as county, \
         rasterio.open(weight_path) as weight_src, \
         rasterio.open(initial_path) as initial, \
         rasterio.open(final_path, "w", **output_profile(county)) as dst:
        for row in range(0, county.height, row_block):
            height = min(row_block, county.height - row)
            window = Window(0, row, county.width, height)
            county_arr = county.read(1, window=window).astype(np.int64)
            weights = weight_src.read(1, window=window).astype(np.float32)
            raw = initial.read(1, window=window).astype(np.float32)
            province_codes = (county_arr // 10000) * 10000
            pidx, in_mainland = province_index(province_codes, codes)
            valid = (raw != NODATA) & (weights > 0) & in_mainland
            block = np.full(raw.shape, NODATA, dtype=np.float32)
            ids = pidx[valid]
            unbounded = raw[valid].astype(np.float64) * factors[ids]
            lower_clipped += int(np.sum(unbounded < LOWER_INCOME))
            upper_clipped += int(np.sum(unbounded > UPPER_INCOME))
            calibrated = np.clip(unbounded, LOWER_INCOME, UPPER_INCOME)
            block[valid] = calibrated.astype(np.float32)
            w = weights[valid].astype(np.float64)
            np.add.at(final_sum, ids, calibrated * w)
            np.add.at(final_weight, ids, w)
            if len(calibrated):
                value_min = min(value_min, float(calibrated.min()))
                value_max = max(value_max, float(calibrated.max()))
                value_sum += float(calibrated.sum())
                value_count += len(calibrated)
                sample_values.append(calibrated[::100].astype(np.float32))
            dst.write(block, 1, window=window)

    province["final_aggregated_2019yuan"] = final_sum / np.maximum(final_weight, 1e-9)
    province["final_vs_official_pct"] = 100 * (
        province["final_aggregated_2019yuan"] / province["rural_income_2019yuan"] - 1
    )
    sample = np.concatenate(sample_values) if sample_values else np.array([])
    stats = {
        "year": year,
        "unit": "2019 constant yuan/person/year",
        "mask": "strict rural population weight > 0, mainland 31 provinces",
        "valid_pixels": int(value_count),
        "min": value_min,
        "mean": value_sum / max(value_count, 1),
        "max": value_max,
        "sample_p01": float(np.quantile(sample, 0.01)) if len(sample) else None,
        "sample_p50": float(np.quantile(sample, 0.50)) if len(sample) else None,
        "sample_p99": float(np.quantile(sample, 0.99)) if len(sample) else None,
        "lower_bound": LOWER_INCOME,
        "upper_bound": UPPER_INCOME,
        "lower_clipped_pixels": lower_clipped,
        "upper_clipped_pixels": upper_clipped,
        "max_abs_province_constraint_error_pct": float(
            province["final_vs_official_pct"].abs().max()
        ),
        "initial_map": str(initial_path),
        "final_map": str(final_path),
    }
    province.to_csv(OUT / f"province_calibration_{year}.csv", index=False,
                    encoding="utf-8-sig")
    (OUT / f"map_quality_{year}.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return final_path, stats


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(ROOT / "config" / "workflow_v1.0_draft.yaml"))
    pre_args, _ = pre_parser.parse_known_args()
    cfg = load_config(pre_args.config)
    defaults = deep_get(cfg, "model.final_maps", {})
    default_years = int_list(defaults.get("years"), deep_get(cfg, "project.study_years", YEARS))

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--years", nargs="+", type=int, default=default_years)
    parser.add_argument("--reuse-model", action="store_true", default=bool(defaults.get("reuse_model", False)))
    parser.add_argument("--pixels-per-county", type=int, default=int(defaults.get("pixels_per_county", 128)))
    parser.add_argument("--epochs", type=int, default=int(defaults.get("epochs", 150)))
    parser.add_argument("--lr", type=float, default=float(defaults.get("lr", 1e-3)))
    parser.add_argument(
        "--output-regularization",
        type=float,
        default=float(defaults.get("output_regularization", 0.001)),
        help="Coefficient for shrinkage of pixel log-income towards the training mean.",
    )
    parser.add_argument("--hidden1", type=int, default=int(defaults.get("hidden1", 64)))
    parser.add_argument("--hidden2", type=int, default=int(defaults.get("hidden2", 32)))
    parser.add_argument("--dropout", type=float, default=float(defaults.get("dropout", 0.05)))
    parser.add_argument(
        "--frozen-config",
        type=Path,
        default=Path("outputs/develop_leave_county_mlp/frozen_mlp_config.json"),
        help="Frozen development configuration; its parameters override defaults.",
    )
    parser.add_argument("--threads", type=int, default=int(defaults.get("threads", 8)))
    parser.add_argument("--row-block", type=int, default=int(defaults.get("row_block", 128)))
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    excluded_features = list(
        deep_get(config, "model.feature_ablation.exclude_features", [])
    )
    if excluded_features:
        # Keep final training samples and standalone raster prediction on the
        # same feature set as the v1.5 no-road validation workflows.
        patch_weak_pixel_loader(weak_module, excluded_features)
        patch_weak_pixel_loader(spatiotemporal_module, excluded_features)
        patch_final_map_features(sys.modules[__name__], excluded_features)
    frozen_params = None
    if args.frozen_config and not args.frozen_config.is_absolute():
        args.frozen_config = ROOT / args.frozen_config
    if args.frozen_config and args.frozen_config.is_file():
        frozen = json.loads(args.frozen_config.read_text(encoding="utf-8"))
        params = frozen.get("parameters", frozen)
        frozen_params = params
        args.pixels_per_county = int(params.get("pixels_per_unit", args.pixels_per_county))
        args.epochs = int(params.get("epochs", args.epochs))
        args.lr = float(params.get("lr", args.lr))
        args.output_regularization = float(
            params.get("output_regularization", args.output_regularization)
        )
        args.hidden1 = int(params.get("hidden1", args.hidden1))
        args.hidden2 = int(params.get("hidden2", args.hidden2))
        args.dropout = float(params.get("dropout", args.dropout))
        print(f"Using frozen MLP configuration: {args.frozen_config}")
    if args.reuse_model and frozen_params is not None:
        metadata_path = MODEL_DIR / "model_metadata.json"
        if not metadata_path.is_file():
            print("Saved model metadata not found; retraining with frozen configuration.")
            args.reuse_model = False
        else:
            old_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected = {
                "hidden1": int(frozen_params.get("hidden1", 64)),
                "hidden2": int(frozen_params.get("hidden2", 32)),
                "dropout": float(frozen_params.get("dropout", 0.05)),
            }
            actual = {
                "hidden1": int(old_meta.get("hidden1", 64)),
                "hidden2": int(old_meta.get("hidden2", 32)),
                "dropout": float(old_meta.get("dropout", 0.05)),
            }
            if expected != actual:
                print(
                    "Saved model architecture differs from frozen configuration; "
                    "retraining final model."
                )
                args.reuse_model = False
    invalid_years = sorted(set(args.years) - set(YEARS))
    if invalid_years:
        raise ValueError(f"Years outside final study period: {invalid_years}")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.threads))
    set_seed(RANDOM_STATE)

    if args.reuse_model:
        model, imputer, scaler, meta = load_model()
        print("Loaded saved final joint model.")
    else:
        model, imputer, scaler, meta = train_and_save(
            args.pixels_per_county,
            args.epochs,
            args.lr,
            args.output_regularization,
            args.hidden1,
            args.hidden2,
            args.dropout,
            excluded_features,
        )
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    all_stats = []
    for year in args.years:
        print(f"\nPredicting {year} initial map ...")
        initial_path, province = predict_initial(
            year, model, imputer, scaler, args.row_block
        )
        print(f"Calibrating {year} to official province income ...")
        final_path, stats = calibrate_map(year, initial_path, province, args.row_block)
        all_stats.append(stats)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        print(f"Saved {final_path}")
        gc.collect()

    (OUT / "run_summary.json").write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
