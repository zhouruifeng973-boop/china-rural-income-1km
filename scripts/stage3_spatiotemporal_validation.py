# -*- coding: utf-8 -*-
"""Stage 3 spatial and temporal validation for weak-supervised downscaling.

This script intentionally writes to a new subdirectory and does not overwrite
the original Stage 3 leave-county validation.

Validation schemes
------------------
1. leave-province: split 2020 counties by province, so every test province is
   completely absent from model fitting.
2. leave-year: train on four years and predict the omitted year. The same
   county may occur in other training years; this is a temporal, not spatial,
   generalisation test.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import torch

from repro_config import deep_get, load_config
from stage3_weak_supervised_downscaling import (
    OUT,
    RANDOM_STATE,
    IncomeMLP,
    aggregate_prediction,
    fit_fold_model,
    load_pixel_panel,
    metric_dict,
    sample_pixels_by_county,
    set_seed,
)


VALIDATION_OUT = OUT / "spatiotemporal_validation"
YEARS = [2019, 2020, 2021, 2022, 2023]


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def run_leave_province(
    year: int,
    mask_name: str,
    folds: int,
    pixels_per_county: int,
    epochs: int,
    lr: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    panel = load_pixel_panel(year, mask_name)
    provinces = panel["county_meta"]["province_std"].astype(str).to_numpy()
    if len(np.unique(provinces)) < folds:
        raise ValueError("Number of province groups is smaller than n_splits.")

    splitter = GroupKFold(n_splits=folds)
    dummy = np.zeros(len(panel["county_codes"]))
    pred_rows: list[pd.DataFrame] = []
    fold_rows: list[dict] = []

    for fold, (train_ids, test_ids) in enumerate(
        splitter.split(dummy, panel["target"], groups=provinces), start=1
    ):
        held_out = sorted(np.unique(provinces[test_ids]).tolist())
        print(
            f"\nProvince fold {fold}/{folds}: train counties={len(train_ids)}, "
            f"test counties={len(test_ids)}, held out={', '.join(held_out)}"
        )
        model, imputer, scaler, train_stats = fit_fold_model(
            panel["X"],
            panel["weights"],
            panel["groups"],
            panel["group_indices"],
            panel["target"],
            train_group_ids=train_ids,
            pixels_per_county=pixels_per_county,
            epochs=epochs,
            lr=lr,
            seed=RANDOM_STATE + 100 + fold,
        )
        pred, weight_sum, pixel_count = aggregate_prediction(
            model,
            imputer,
            scaler,
            panel["X"],
            panel["groups"],
            panel["weights"],
            test_ids,
        )
        true = panel["target"][test_ids]
        metrics = metric_dict(true, pred)
        metrics.update(
            {
                "year": year,
                "mask": mask_name,
                "fold": fold,
                "held_out_provinces": ";".join(held_out),
                "province_count": len(held_out),
                "pixels_per_county": pixels_per_county,
                "epochs": epochs,
                **train_stats,
            }
        )
        fold_rows.append(metrics)
        print(
            f"  Province fold {fold}: R2={metrics['r2']:.3f}, "
            f"MAE={metrics['mae']:.0f}, RMSE={metrics['rmse']:.0f}"
        )

        fold_meta = panel["county_meta"].iloc[test_ids].copy()
        fold_meta["fold"] = fold
        fold_meta["true_2019yuan"] = true
        fold_meta["pred_weak_2019yuan"] = pred
        fold_meta["rural_weight_sum"] = weight_sum
        fold_meta["rural_pixel_count"] = pixel_count
        pred_rows.append(fold_meta)

    predictions = pd.concat(pred_rows, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_rows)
    overall = metric_dict(
        predictions["true_2019yuan"].to_numpy(float),
        predictions["pred_weak_2019yuan"].to_numpy(float),
    )
    overall.update(
        {
            "scheme": "leave_province",
            "year": year,
            "mask": mask_name,
            "folds": folds,
            "province_count": int(len(np.unique(provinces))),
            "pixels_per_county": pixels_per_county,
            "epochs": epochs,
        }
    )
    return predictions, fold_metrics, overall


def sample_year_bags(
    year: int,
    mask_name: str,
    pixels_per_county: int,
    seed: int,
) -> dict:
    """Load one full year, retain a compact pixel sample for later fitting."""
    panel = load_pixel_panel(year, mask_name)
    all_group_ids = np.arange(len(panel["county_codes"]), dtype=np.int64)
    selected = sample_pixels_by_county(
        panel["group_indices"],
        panel["weights"],
        all_group_ids,
        pixels_per_county,
        np.random.default_rng(seed),
    )
    result = {
        "year": year,
        "X": panel["X"][selected].copy(),
        "weights": panel["weights"][selected].copy(),
        "groups": panel["groups"][selected].copy(),
        "target": panel["target"].copy(),
        "band_names": list(panel["band_names"]),
        "county_count": len(panel["county_codes"]),
    }
    del panel
    gc.collect()
    return result


def add_year_column(X: np.ndarray, year: int) -> np.ndarray:
    # Known temporal covariate, centred and scaled to roughly [-1, 1].
    year_scaled = np.float32((year - 2021.0) / 2.0)
    return np.column_stack(
        [X.astype(np.float32, copy=False), np.full(len(X), year_scaled, dtype=np.float32)]
    )


def fit_bag_model(
    sampled_by_year: dict[int, dict],
    train_years: list[int],
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[IncomeMLP, SimpleImputer, StandardScaler, dict]:
    """Fit one model to county-year bags sampled from multiple years."""
    X_parts = []
    weight_parts = []
    group_parts = []
    target_parts = []
    group_offset = 0

    for year in train_years:
        sample = sampled_by_year[year]
        X_parts.append(add_year_column(sample["X"], year))
        weight_parts.append(sample["weights"])
        group_parts.append(sample["groups"] + group_offset)
        target_parts.append(sample["target"])
        group_offset += len(sample["target"])

    X = np.concatenate(X_parts, axis=0)
    weights = np.concatenate(weight_parts).astype(np.float32, copy=False)
    groups = np.concatenate(group_parts).astype(np.int64, copy=False)
    targets = np.concatenate(target_parts).astype(np.float32, copy=False)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X)).astype(np.float32)
    del X, X_parts
    gc.collect()

    set_seed(seed)
    x_t = torch.from_numpy(X_scaled)
    w_t = torch.from_numpy(weights)
    g_t = torch.from_numpy(groups)
    log_y_t = torch.log(torch.from_numpy(targets))

    model = IncomeMLP(
        X_scaled.shape[1], init_log_income=float(np.log(np.median(targets)))
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    mean_log_y = log_y_t.mean()
    last_loss = math.nan

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        log_pred = model(x_t)
        pred = torch.exp(log_pred)
        num = torch.zeros(len(targets), dtype=torch.float32)
        den = torch.zeros(len(targets), dtype=torch.float32)
        num.index_add_(0, g_t, pred * w_t)
        den.index_add_(0, g_t, w_t)
        county_year_pred = num / torch.clamp(den, min=1e-6)
        main_loss = torch.mean(
            (torch.log(torch.clamp(county_year_pred, min=1.0)) - log_y_t) ** 2
        )
        regularizer = 0.001 * torch.mean((log_pred - mean_log_y) ** 2)
        loss = main_loss + regularizer
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        last_loss = float(loss.detach().cpu().item())
        if epoch in {1, epochs} or epoch % max(50, epochs // 4) == 0:
            print(f"    epoch {epoch:4d}/{epochs}: loss={last_loss:.6f}")

    stats = {
        "sampled_pixels": int(len(X_scaled)),
        "train_county_years": int(len(targets)),
        "final_loss": last_loss,
    }
    return model, imputer, scaler, stats


@torch.no_grad()
def aggregate_year_panel(
    model: IncomeMLP,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    panel: dict,
    year: int,
    batch_size: int = 200_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    group_count = len(panel["county_codes"])
    sums = np.zeros(group_count, dtype=np.float64)
    den = np.zeros(group_count, dtype=np.float64)
    counts = np.zeros(group_count, dtype=np.int64)

    for start in range(0, len(panel["X"]), batch_size):
        stop = min(start + batch_size, len(panel["X"]))
        X_chunk = add_year_column(panel["X"][start:stop], year)
        X_scaled = scaler.transform(imputer.transform(X_chunk)).astype(np.float32)
        pred = torch.exp(model(torch.from_numpy(X_scaled))).cpu().numpy().astype(np.float64)
        groups = panel["groups"][start:stop]
        weights = panel["weights"][start:stop].astype(np.float64)
        np.add.at(sums, groups, pred * weights)
        np.add.at(den, groups, weights)
        np.add.at(counts, groups, 1)

    return (
        (sums / np.maximum(den, 1e-9)).astype(np.float32),
        den.astype(np.float32),
        counts,
    )


def run_leave_year(
    mask_name: str,
    pixels_per_county: int,
    epochs: int,
    lr: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    print("\nPreparing fixed county-pixel samples for all five years ...")
    sampled: dict[int, dict] = {}
    reference_bands: list[str] | None = None
    for year in YEARS:
        sampled[year] = sample_year_bags(
            year,
            mask_name,
            pixels_per_county,
            seed=RANDOM_STATE + year,
        )
        bands = sampled[year]["band_names"]
        if reference_bands is None:
            reference_bands = bands
        elif bands != reference_bands:
            raise RuntimeError(f"Feature bands differ in {year}.")

    pred_rows: list[pd.DataFrame] = []
    year_rows: list[dict] = []
    for fold, test_year in enumerate(YEARS, start=1):
        train_years = [year for year in YEARS if year != test_year]
        print(
            f"\nYear fold {fold}/5: train={train_years}, held out={test_year}"
        )
        model, imputer, scaler, train_stats = fit_bag_model(
            sampled,
            train_years,
            epochs=epochs,
            lr=lr,
            seed=RANDOM_STATE + 200 + fold,
        )
        test_panel = load_pixel_panel(test_year, mask_name)
        pred, weight_sum, pixel_count = aggregate_year_panel(
            model, imputer, scaler, test_panel, test_year
        )
        true = test_panel["target"]
        metrics = metric_dict(true, pred)
        metrics.update(
            {
                "scheme": "leave_year",
                "test_year": test_year,
                "train_years": ";".join(str(x) for x in train_years),
                "mask": mask_name,
                "pixels_per_county": pixels_per_county,
                "epochs": epochs,
                **train_stats,
            }
        )
        year_rows.append(metrics)
        print(
            f"  Held-out {test_year}: R2={metrics['r2']:.3f}, "
            f"MAE={metrics['mae']:.0f}, RMSE={metrics['rmse']:.0f}, "
            f"Bias={metrics['bias']:.0f}"
        )

        meta = test_panel["county_meta"].copy()
        meta["test_year"] = test_year
        meta["true_2019yuan"] = true
        meta["pred_weak_2019yuan"] = pred
        meta["rural_weight_sum"] = weight_sum
        meta["rural_pixel_count"] = pixel_count
        pred_rows.append(meta)
        del test_panel, model, imputer, scaler
        gc.collect()

    predictions = pd.concat(pred_rows, ignore_index=True)
    year_metrics = pd.DataFrame(year_rows)
    overall = metric_dict(
        predictions["true_2019yuan"].to_numpy(float),
        predictions["pred_weak_2019yuan"].to_numpy(float),
    )
    overall.update(
        {
            "scheme": "leave_year",
            "years": YEARS,
            "mask": mask_name,
            "pixels_per_county": pixels_per_county,
            "epochs": epochs,
            "note": "Overall R2 pools all years; per-year R2 is the main temporal diagnostic.",
        }
    )
    return predictions, year_metrics, overall


def save_leave_province(
    predictions: pd.DataFrame, fold_metrics: pd.DataFrame, overall: dict
) -> None:
    year = overall["year"]
    mask_name = overall["mask"]
    prefix = VALIDATION_OUT / f"leave_province_{year}_{mask_name}"
    predictions.to_csv(
        Path(f"{prefix}_county_predictions.csv"), index=False, encoding="utf-8-sig"
    )
    fold_metrics.to_csv(
        Path(f"{prefix}_fold_metrics.csv"), index=False, encoding="utf-8-sig"
    )
    write_json(Path(f"{prefix}_overall_metrics.json"), overall)


def save_leave_year(
    predictions: pd.DataFrame, year_metrics: pd.DataFrame, overall: dict
) -> None:
    mask_name = overall["mask"]
    prefix = VALIDATION_OUT / f"leave_year_2019_2023_{mask_name}"
    predictions.to_csv(
        Path(f"{prefix}_county_predictions.csv"), index=False, encoding="utf-8-sig"
    )
    year_metrics.to_csv(
        Path(f"{prefix}_year_metrics.csv"), index=False, encoding="utf-8-sig"
    )
    write_json(Path(f"{prefix}_overall_metrics.json"), overall)


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config" / "workflow_v1.0_draft.yaml"))
    pre_args, _ = pre_parser.parse_known_args()
    cfg = load_config(pre_args.config)
    defaults = deep_get(cfg, "model.spatiotemporal_validation", {})
    main_mask = deep_get(cfg, "project.main_mask", "strict")

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--run", choices=["province", "year", "all"], default=str(defaults.get("run", "all")))
    parser.add_argument("--year", type=int, default=int(defaults.get("year", 2020)))
    parser.add_argument("--mask", choices=["strict", "broad"], default=str(defaults.get("mask", main_mask)))
    parser.add_argument("--province-folds", type=int, default=int(defaults.get("province_folds", 5)))
    parser.add_argument("--province-pixels", type=int, default=int(defaults.get("province_pixels", 128)))
    parser.add_argument("--year-pixels", type=int, default=int(defaults.get("year_pixels", 64)))
    parser.add_argument("--epochs", type=int, default=int(defaults.get("epochs", defaults.get("epochs_spatiotemporal_validation", 150))))
    parser.add_argument("--lr", type=float, default=float(defaults.get("lr", 1e-3)))
    parser.add_argument("--threads", type=int, default=int(defaults.get("threads", 8)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    VALIDATION_OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.threads))
    set_seed(RANDOM_STATE)
    print("=" * 72)
    print("Stage 3 spatial and temporal validation")
    print("=" * 72)
    print(vars(args))

    if args.run in {"province", "all"}:
        province_result = run_leave_province(
            args.year,
            args.mask,
            args.province_folds,
            args.province_pixels,
            args.epochs,
            args.lr,
        )
        save_leave_province(*province_result)
        print("\nLeave-province overall:")
        print(json.dumps(province_result[2], ensure_ascii=False, indent=2))

    if args.run in {"year", "all"}:
        year_result = run_leave_year(
            args.mask,
            args.year_pixels,
            args.epochs,
            args.lr,
        )
        save_leave_year(*year_result)
        print("\nLeave-year overall:")
        print(json.dumps(year_result[2], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
