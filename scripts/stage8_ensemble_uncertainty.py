# -*- coding: utf-8 -*-
"""Create true multi-seed ensemble uncertainty layers.

This stage repeats the final weak-supervised downscaling workflow with several
random seeds. For each member, pixel predictions are calibrated to the official
province-year rural income using the same unbounded calibration rule as the
canonical Stage 6 product. The final uncertainty layers are computed from the
calibrated member predictions:

  - ensemble mean, 2019 constant yuan/person/year
  - ensemble standard deviation, 2019 constant yuan/person/year
  - coefficient of variation, percent

The output is a model-ensemble uncertainty estimate. It is different from the
Stage 7 quality-risk index, which is a rule-based caution layer.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

ROOT_HINT = Path(__file__).resolve().parents[1]
LOCAL_DEP_PATHS = [
    ROOT_HINT / ".codex_pydeps_stage8",
    ROOT_HINT / ".codex_pydeps",
    ROOT_HINT / "stage6_pydeps",
]
for LOCAL_DEPS in reversed(LOCAL_DEP_PATHS):
    if LOCAL_DEPS.exists():
        sys.path.insert(0, str(LOCAL_DEPS))

import rasterio
from rasterio.windows import Window
import torch

from repro_config import deep_get, int_list, load_config
from stage3_spatiotemporal_validation import YEARS, add_year_column, fit_bag_model, sample_year_bags
from stage3_weak_supervised_downscaling import ROOT, set_seed
from stage5_final_maps import (
    BAND_NAMES,
    NODATA,
    feature_specs,
    official_income,
    output_profile,
    province_index,
    read_features_for_valid,
)


PACKAGE = ROOT / "outputs" / "stage6_scientific_data_product"
OUT = PACKAGE / "data" / "ensemble_uncertainty_layers"
TABLES = PACKAGE / "tables"
METADATA = PACKAGE / "metadata"
MODEL_OUT = ROOT / "outputs" / "stage8_ensemble_uncertainty" / "models"
FEATURES = ROOT / "features"
PROCESSED = ROOT / "data" / "processed"


def parse_seed_list(text: str, members: int) -> list[int]:
    if text:
        seeds = [int(x.strip()) for x in text.split(",") if x.strip()]
    else:
        seeds = [1042 + 997 * i for i in range(members)]
    if len(seeds) != members:
        raise ValueError(f"--members={members}, but {len(seeds)} seeds were provided.")
    return seeds


def mean_path(year: int) -> Path:
    return OUT / f"china_rural_income_1km_{year}_ensemble_mean_2019yuan_strict_unbounded.tif"


def std_path(year: int) -> Path:
    return OUT / f"china_rural_income_1km_{year}_ensemble_std_2019yuan_strict_unbounded.tif"


def cv_path(year: int) -> Path:
    return OUT / f"china_rural_income_1km_{year}_ensemble_cv_percent_strict_unbounded.tif"


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def train_member(
    member_id: int,
    seed: int,
    pixels_per_county: int,
    epochs: int,
    lr: float,
) -> dict:
    print(f"\nTraining ensemble member {member_id} with seed={seed} ...", flush=True)
    set_seed(seed)
    samples = {}
    reference_bands = None
    for year in YEARS:
        samples[year] = sample_year_bags(
            year, "strict", pixels_per_county, seed + 100_000 + year
        )
        bands = samples[year]["band_names"]
        if reference_bands is None:
            reference_bands = bands
        elif bands != reference_bands:
            raise RuntimeError(f"Feature order differs in {year}.")
    if list(reference_bands) != BAND_NAMES:
        raise RuntimeError(f"Unexpected training feature order: {reference_bands}")

    model, imputer, scaler, stats = fit_bag_model(
        samples, YEARS, epochs=epochs, lr=lr, seed=seed
    )
    del samples
    gc.collect()

    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT / f"member_{member_id:02d}_state.pt")
    with (MODEL_OUT / f"member_{member_id:02d}_preprocessing.pkl").open("wb") as f:
        pickle.dump({"imputer": imputer, "scaler": scaler}, f)

    meta = {
        "member_id": member_id,
        "seed": seed,
        "pixels_per_county": pixels_per_county,
        "epochs": epochs,
        "learning_rate": lr,
        "n_features": len(BAND_NAMES) + 1,
        "init_log_income": float(model.init_log_income),
        **stats,
    }
    (MODEL_OUT / f"member_{member_id:02d}_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Finished member {member_id}: final_loss={stats['final_loss']:.6f}",
        flush=True,
    )
    return {"model": model.eval(), "imputer": imputer, "scaler": scaler, "meta": meta}


@torch.no_grad()
def predict_for_member(model, imputer, scaler, X_year: np.ndarray) -> np.ndarray:
    X_scaled = scaler.transform(imputer.transform(X_year)).astype(np.float32)
    return torch.exp(model(torch.from_numpy(X_scaled))).cpu().numpy().astype(np.float64)


@torch.no_grad()
def calculate_member_factors(year: int, members: list[dict], row_block: int) -> tuple[pd.DataFrame, np.ndarray]:
    official = official_income(year).reset_index(drop=True)
    official_codes = official["province_code"].to_numpy(np.int64)
    province_sum = np.zeros((len(members), len(official)), dtype=np.float64)
    province_weight = np.zeros(len(official), dtype=np.float64)
    weight_path = PROCESSED / "rural_weights" / f"rural_pop_weight_{year}_strict.tif"

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

        for row in range(0, county.height, row_block):
            height = min(row_block, county.height - row)
            window = Window(0, row, county.width, height)
            county_arr = county.read(1, window=window).astype(np.int64)
            weights = weight_src.read(1, window=window).astype(np.float32)
            province_codes = (county_arr // 10000) * 10000
            pidx, in_mainland = province_index(province_codes, official_codes)
            valid = (county_arr > 0) & np.isfinite(weights) & (weights > 0) & in_mainland
            valid_flat = valid.ravel()
            if np.any(valid_flat):
                X = read_features_for_valid(open_specs, window, valid_flat)
                X_year = add_year_column(X, year)
                ids = pidx.ravel()[valid_flat]
                w = weights.ravel()[valid_flat].astype(np.float64)
                np.add.at(province_weight, ids, w)
                for member_idx, member in enumerate(members):
                    pred = predict_for_member(
                        member["model"], member["imputer"], member["scaler"], X_year
                    )
                    np.add.at(province_sum[member_idx], ids, pred * w)
            print(f"  {year} factor pass rows {row + height}/{county.height}", flush=True)

    initial = province_sum / np.maximum(province_weight.reshape(1, -1), 1e-9)
    factors = (
        official["rural_income_2019yuan"].to_numpy(np.float64).reshape(1, -1)
        / np.maximum(initial, 1e-9)
    )

    rows = []
    for member_idx, member in enumerate(members):
        for province_idx, official_row in official.iterrows():
            rows.append(
                {
                    "year": year,
                    "member_id": member["meta"]["member_id"],
                    "seed": member["meta"]["seed"],
                    "province_code": int(official_row["province_code"]),
                    "province_std": official_row["province_std"],
                    "official_rural_income_2019yuan": float(official_row["rural_income_2019yuan"]),
                    "initial_aggregated_2019yuan": float(initial[member_idx, province_idx]),
                    "unbounded_calibration_factor": float(factors[member_idx, province_idx]),
                    "strict_rural_weight": float(province_weight[province_idx]),
                }
            )
    return pd.DataFrame(rows), factors


def write_stats_tags(path: Path) -> dict:
    with rasterio.open(path, "r+") as src:
        values = src.read(1, masked=True).compressed().astype(np.float64)
        if values.size == 0:
            return {}
        tags = {
            "STATISTICS_MINIMUM": f"{values.min():.10g}",
            "STATISTICS_MAXIMUM": f"{values.max():.10g}",
            "STATISTICS_MEAN": f"{values.mean():.10g}",
            "STATISTICS_STDDEV": f"{values.std():.10g}",
            "STATISTICS_VALID_PERCENT": f"{100 * len(values) / (src.width * src.height):.10g}",
        }
        src.update_tags(1, **tags)
        return tags


@torch.no_grad()
def create_year_layers(year: int, members: list[dict], factors: np.ndarray, row_block: int) -> dict:
    official = official_income(year).reset_index(drop=True)
    official_codes = official["province_code"].to_numpy(np.int64)
    weight_path = PROCESSED / "rural_weights" / f"rural_pop_weight_{year}_strict.tif"

    all_std = []
    all_cv = []
    all_mean = []
    valid_pixels = 0

    with ExitStack() as stack:
        county = stack.enter_context(rasterio.open(FEATURES / "county_id.tif"))
        weight_src = stack.enter_context(rasterio.open(weight_path))
        open_specs = []
        for path, names in feature_specs(year):
            src = stack.enter_context(rasterio.open(path))
            open_specs.append((src, names))

        profile = output_profile(county)
        mean_dst = stack.enter_context(rasterio.open(mean_path(year), "w", **profile))
        std_dst = stack.enter_context(rasterio.open(std_path(year), "w", **profile))
        cv_dst = stack.enter_context(rasterio.open(cv_path(year), "w", **profile))

        for row in range(0, county.height, row_block):
            height = min(row_block, county.height - row)
            window = Window(0, row, county.width, height)
            county_arr = county.read(1, window=window).astype(np.int64)
            weights = weight_src.read(1, window=window).astype(np.float32)
            province_codes = (county_arr // 10000) * 10000
            pidx, in_mainland = province_index(province_codes, official_codes)
            valid = (county_arr > 0) & np.isfinite(weights) & (weights > 0) & in_mainland
            valid_flat = valid.ravel()

            mean_block = np.full(county_arr.shape, NODATA, dtype=np.float32)
            std_block = np.full(county_arr.shape, NODATA, dtype=np.float32)
            cv_block = np.full(county_arr.shape, NODATA, dtype=np.float32)

            if np.any(valid_flat):
                X = read_features_for_valid(open_specs, window, valid_flat)
                X_year = add_year_column(X, year)
                ids = pidx.ravel()[valid_flat]
                member_preds = []
                for member_idx, member in enumerate(members):
                    pred = predict_for_member(
                        member["model"], member["imputer"], member["scaler"], X_year
                    )
                    member_preds.append(pred * factors[member_idx, ids])
                stack_pred = np.vstack(member_preds)
                mean_values = stack_pred.mean(axis=0)
                std_values = stack_pred.std(axis=0, ddof=1) if len(members) > 1 else np.zeros_like(mean_values)
                cv_values = 100.0 * std_values / np.maximum(mean_values, 1.0)

                mean_block.ravel()[valid_flat] = mean_values.astype(np.float32)
                std_block.ravel()[valid_flat] = std_values.astype(np.float32)
                cv_block.ravel()[valid_flat] = cv_values.astype(np.float32)

                all_mean.append(mean_values.astype(np.float32))
                all_std.append(std_values.astype(np.float32))
                all_cv.append(cv_values.astype(np.float32))
                valid_pixels += int(mean_values.size)

            mean_dst.write(mean_block, 1, window=window)
            std_dst.write(std_block, 1, window=window)
            cv_dst.write(cv_block, 1, window=window)
            print(f"  {year} output pass rows {row + height}/{county.height}", flush=True)

    for path in [mean_path(year), std_path(year), cv_path(year)]:
        write_stats_tags(path)

    mean_values = np.concatenate(all_mean) if all_mean else np.array([], dtype=np.float32)
    std_values = np.concatenate(all_std) if all_std else np.array([], dtype=np.float32)
    cv_values = np.concatenate(all_cv) if all_cv else np.array([], dtype=np.float32)
    return {
        "year": year,
        "ensemble_members": len(members),
        "valid_pixels": valid_pixels,
        "mean_income_median": float(np.median(mean_values)) if mean_values.size else np.nan,
        "mean_income_p01": float(np.quantile(mean_values, 0.01)) if mean_values.size else np.nan,
        "mean_income_p99": float(np.quantile(mean_values, 0.99)) if mean_values.size else np.nan,
        "std_median": float(np.median(std_values)) if std_values.size else np.nan,
        "std_p90": float(np.quantile(std_values, 0.90)) if std_values.size else np.nan,
        "std_p95": float(np.quantile(std_values, 0.95)) if std_values.size else np.nan,
        "std_max": float(std_values.max()) if std_values.size else np.nan,
        "cv_median_pct": float(np.median(cv_values)) if cv_values.size else np.nan,
        "cv_p90_pct": float(np.quantile(cv_values, 0.90)) if cv_values.size else np.nan,
        "cv_p95_pct": float(np.quantile(cv_values, 0.95)) if cv_values.size else np.nan,
        "cv_max_pct": float(cv_values.max()) if cv_values.size else np.nan,
    }


def update_manifest() -> None:
    records = []
    for path in sorted(OUT.glob("*.tif")):
        records.append(
            {
                "role": "ensemble_uncertainty_layer",
                "package_file": str(path.relative_to(PACKAGE)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(records).to_csv(
        METADATA / "ensemble_uncertainty_layer_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )


def append_data_dictionary() -> None:
    path = METADATA / "data_dictionary.csv"
    add = pd.DataFrame(
        [
            {
                "file_group": "ensemble_mean_income",
                "field_or_band": "band 1",
                "meaning": "Mean of calibrated predictions from the multi-seed ensemble",
                "unit": "2019 constant yuan/person/year",
                "nodata": "-9999",
                "notes": "Strict rural mask, unbounded province calibration.",
            },
            {
                "file_group": "ensemble_prediction_std",
                "field_or_band": "band 1",
                "meaning": "Standard deviation of calibrated predictions across ensemble members",
                "unit": "2019 constant yuan/person/year",
                "nodata": "-9999",
                "notes": "This is a model/sampling ensemble uncertainty estimate, not observed pixel error.",
            },
            {
                "file_group": "ensemble_prediction_cv",
                "field_or_band": "band 1",
                "meaning": "Coefficient of variation of calibrated predictions across ensemble members",
                "unit": "%",
                "nodata": "-9999",
                "notes": "100 * ensemble standard deviation / ensemble mean.",
            },
        ]
    )
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["file_group"].isin(add["file_group"])]
        pd.concat([old, add], ignore_index=True).to_csv(
            path, index=False, encoding="utf-8-sig"
        )


def write_readme(member_meta: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Multi-seed ensemble uncertainty layers",
        "",
        "These layers estimate prediction uncertainty by repeating the final weak-supervised downscaling workflow with multiple random seeds and pixel samples.",
        "",
        "Each ensemble member is calibrated to official province-year rural income using the same unbounded calibration rule as the canonical annual income maps. The annual mean, standard deviation, and coefficient of variation are then calculated from calibrated member predictions.",
        "",
        "## Files",
        "",
        "- `*_ensemble_mean_2019yuan_strict_unbounded.tif`: ensemble mean income.",
        "- `*_ensemble_std_2019yuan_strict_unbounded.tif`: ensemble standard deviation.",
        "- `*_ensemble_cv_percent_strict_unbounded.tif`: coefficient of variation in percent.",
        "",
        "## Member Settings",
        "",
        "| member_id | seed | sampled_pixels | train_county_years | final_loss |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, row in member_meta.iterrows():
        lines.append(
            f"| {int(row['member_id'])} | {int(row['seed'])} | {int(row['sampled_pixels']):,} | {int(row['train_county_years']):,} | {float(row['final_loss']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Annual Summary",
            "",
            "| year | valid_pixels | std_median | std_p95 | cv_median_pct | cv_p95_pct |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {int(row['year'])} | {int(row['valid_pixels']):,} | {float(row['std_median']):.1f} | {float(row['std_p95']):.1f} | {float(row['cv_median_pct']):.2f} | {float(row['cv_p95_pct']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The standard deviation and coefficient of variation represent model/sampling ensemble spread after province calibration. They should be described as ensemble prediction uncertainty, not as direct pixel-level validation error.",
            "",
            "The Stage 7 quality-risk index remains useful because it captures extrapolation support, calibration magnitude, and bounded/unbounded sensitivity. Stage 8 captures stochastic model uncertainty.",
        ]
    )
    (OUT / "README_ensemble_uncertainty_layers.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(ROOT / "config" / "workflow_v1.0_draft.yaml"))
    pre_args, _ = pre_parser.parse_known_args()
    cfg = load_config(pre_args.config)
    defaults = deep_get(cfg, "model.ensemble_uncertainty", {})
    default_years = int_list(defaults.get("years"), YEARS)
    default_seeds = defaults.get("seeds", "")
    if isinstance(default_seeds, (list, tuple)):
        default_seeds = ",".join(str(int(x)) for x in default_seeds)

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--members", type=int, default=int(defaults.get("members", 5)))
    parser.add_argument("--seeds", type=str, default=str(default_seeds or ""))
    parser.add_argument("--years", nargs="+", type=int, default=default_years)
    parser.add_argument("--pixels-per-county", type=int, default=int(defaults.get("pixels_per_county", 128)))
    parser.add_argument("--epochs", type=int, default=int(defaults.get("epochs", 150)))
    parser.add_argument("--lr", type=float, default=float(defaults.get("lr", 1e-3)))
    parser.add_argument("--threads", type=int, default=int(defaults.get("threads", 8)))
    parser.add_argument("--row-block", type=int, default=int(defaults.get("row_block", 128)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    invalid_years = sorted(set(args.years) - set(YEARS))
    if invalid_years:
        raise ValueError(f"Years outside final study period: {invalid_years}")
    if args.members < 2:
        raise ValueError("At least two ensemble members are required to estimate standard deviation.")

    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    METADATA.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.threads))
    seeds = parse_seed_list(args.seeds, args.members)

    print("=" * 72)
    print("Stage 8 multi-seed ensemble uncertainty")
    print("=" * 72)
    print(json.dumps({**vars(args), "resolved_seeds": seeds}, ensure_ascii=False, indent=2))

    members = [
        train_member(i + 1, seed, args.pixels_per_county, args.epochs, args.lr)
        for i, seed in enumerate(seeds)
    ]
    member_meta = pd.DataFrame([m["meta"] for m in members])
    member_meta.to_csv(
        TABLES / "ensemble_member_metadata.csv", index=False, encoding="utf-8-sig"
    )

    calibration_frames = []
    summaries = []
    for year in args.years:
        print(f"\nCalculating ensemble calibration factors for {year} ...", flush=True)
        calibration, factors = calculate_member_factors(year, members, args.row_block)
        calibration_frames.append(calibration)
        print(f"Creating ensemble uncertainty layers for {year} ...", flush=True)
        summaries.append(create_year_layers(year, members, factors, args.row_block))
        gc.collect()

    calibration_all = pd.concat(calibration_frames, ignore_index=True)
    calibration_all.to_csv(
        TABLES / "ensemble_province_calibration_factors_2019_2023.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = pd.DataFrame(summaries)
    summary.to_csv(
        TABLES / "ensemble_uncertainty_summary_2019_2023.csv",
        index=False,
        encoding="utf-8-sig",
    )

    update_manifest()
    append_data_dictionary()
    write_readme(member_meta, summary)

    print("\nStage 8 finished.")
    print(OUT)
    print(summary[["year", "std_median", "std_p95", "cv_median_pct", "cv_p95_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
