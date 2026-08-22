"""Five-fold 2019-2023 leave-province-out validation.

Every province is assigned to exactly one fold. All county-year observations
from a held-out province are excluded from fitting, so the experiment measures
multi-year spatial transfer without allowing the same province to enter the
training set in another year.
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
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import torch

from feature_ablation import patch_weak_pixel_loader
from repro_config import deep_get, int_list, load_config
import stage3_weak_supervised_downscaling as weak
from stage3_spatiotemporal_validation import add_year_column


DEFAULT_CONFIG = ROOT / "config" / "workflow_v1.5_no_road_core.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def target_loader(target_path: Path):
    def load_targets(year: int) -> pd.DataFrame:
        frame = pd.read_csv(target_path)
        frame = frame[frame["year"].astype(int) == year].copy()
        frame["county_code"] = frame["county_code"].astype(int)
        frame = frame.dropna(subset=["rural_income_2019yuan", "province_std"])
        return frame[[
            "year", "county_code", "county", "city", "province_std",
            "rural_income_nominal", "rural_income_2019yuan",
        ]]
    return load_targets


def sample_year(year: int, mask: str, pixels_per_county: int, seed: int) -> dict:
    panel = weak.load_pixel_panel(year, mask)
    group_ids = np.arange(len(panel["county_codes"]), dtype=np.int64)
    selected = weak.sample_pixels_by_county(
        panel["group_indices"],
        panel["weights"],
        group_ids,
        pixels_per_county,
        np.random.default_rng(seed),
    )
    result = {
        "year": year,
        "X": panel["X"][selected].copy(),
        "weights": panel["weights"][selected].copy(),
        "groups": panel["groups"][selected].copy(),
        "target": panel["target"].copy(),
        "county_meta": panel["county_meta"].copy(),
        "county_codes": panel["county_codes"].copy(),
        "band_names": list(panel["band_names"]),
    }
    del panel
    gc.collect()
    return result


def fit_fold(
    sampled: dict[int, dict],
    years: list[int],
    held_out_provinces: set[str],
    epochs: int,
    lr: float,
    seed: int,
    output_regularization: float,
):
    X_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    group_offset = 0

    for year in years:
        sample = sampled[year]
        provinces = sample["county_meta"]["province_std"].astype(str).to_numpy()
        train_groups = np.flatnonzero(~np.isin(provinces, list(held_out_provinces)))
        lookup = np.full(len(provinces), -1, dtype=np.int64)
        lookup[train_groups] = np.arange(len(train_groups), dtype=np.int64)
        pixel_mask = np.isin(sample["groups"], train_groups)
        X_parts.append(add_year_column(sample["X"][pixel_mask], year))
        weight_parts.append(sample["weights"][pixel_mask])
        group_parts.append(lookup[sample["groups"][pixel_mask]] + group_offset)
        target_parts.append(sample["target"][train_groups])
        group_offset += len(train_groups)

    X = np.concatenate(X_parts, axis=0)
    weights = np.concatenate(weight_parts).astype(np.float32, copy=False)
    groups = np.concatenate(group_parts).astype(np.int64, copy=False)
    targets = np.concatenate(target_parts).astype(np.float32, copy=False)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X)).astype(np.float32)
    del X, X_parts, weight_parts, group_parts, target_parts
    gc.collect()

    weak.set_seed(seed)
    x_t = torch.from_numpy(X_scaled)
    w_t = torch.from_numpy(weights)
    g_t = torch.from_numpy(groups)
    log_y_t = torch.log(torch.from_numpy(targets))
    model = weak.IncomeMLP(
        X_scaled.shape[1], init_log_income=float(np.log(np.median(targets)))
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    mean_log_y = log_y_t.mean()
    last_loss = math.nan
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        log_pred = model(x_t)
        pixel_pred = torch.exp(log_pred)
        numerator = torch.zeros(len(targets), dtype=torch.float32)
        denominator = torch.zeros(len(targets), dtype=torch.float32)
        numerator.index_add_(0, g_t, pixel_pred * w_t)
        denominator.index_add_(0, g_t, w_t)
        county_year_pred = numerator / torch.clamp(denominator, min=1e-6)
        main_loss = torch.mean(
            (torch.log(torch.clamp(county_year_pred, min=1.0)) - log_y_t) ** 2
        )
        regularizer = output_regularization * torch.mean((log_pred - mean_log_y) ** 2)
        loss = main_loss + regularizer
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())
        if epoch in {1, epochs} or epoch % max(50, epochs // 4) == 0:
            print(f"    epoch {epoch:4d}/{epochs}: loss={last_loss:.6f}", flush=True)

    return model.eval(), imputer, scaler, {
        "sampled_pixels": int(len(X_scaled)),
        "train_county_years": int(len(targets)),
        "final_loss": last_loss,
    }


@torch.no_grad()
def aggregate_held_out(
    model,
    imputer,
    scaler,
    panel: dict,
    year: int,
    held_out_provinces: set[str],
    batch_size: int = 200_000,
):
    provinces = panel["county_meta"]["province_std"].astype(str).to_numpy()
    test_groups = np.flatnonzero(np.isin(provinces, list(held_out_provinces)))
    lookup = np.full(len(provinces), -1, dtype=np.int64)
    lookup[test_groups] = np.arange(len(test_groups), dtype=np.int64)
    pixel_indices = np.flatnonzero(np.isin(panel["groups"], test_groups))
    sums = np.zeros(len(test_groups), dtype=np.float64)
    denominator = np.zeros(len(test_groups), dtype=np.float64)
    counts = np.zeros(len(test_groups), dtype=np.int64)
    for start in range(0, len(pixel_indices), batch_size):
        index = pixel_indices[start:start + batch_size]
        X_year = add_year_column(panel["X"][index], year)
        X_scaled = scaler.transform(imputer.transform(X_year)).astype(np.float32)
        pred = torch.exp(model(torch.from_numpy(X_scaled))).cpu().numpy().astype(np.float64)
        local_group = lookup[panel["groups"][index]]
        weights = panel["weights"][index].astype(np.float64)
        np.add.at(sums, local_group, pred * weights)
        np.add.at(denominator, local_group, weights)
        np.add.at(counts, local_group, 1)
    return (
        test_groups,
        (sums / np.maximum(denominator, 1e-9)).astype(np.float32),
        denominator.astype(np.float32),
        counts,
    )


def metrics(frame: pd.DataFrame) -> dict:
    return weak.metric_dict(
        frame["true_2019yuan"].to_numpy(float),
        frame["pred_weak_2019yuan"].to_numpy(float),
    )


def grouped_metrics(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    key = groups[0] if len(groups) == 1 else groups
    for values, group in frame.groupby(key, dropna=False):
        values = (values,) if len(groups) == 1 else values
        row = {name: value for name, value in zip(groups, values)}
        row.update(metrics(group))
        rows.append(row)
    return pd.DataFrame(rows)


def within_year_r2(frame: pd.DataFrame) -> float:
    true = frame["true_2019yuan"].astype(float)
    pred = frame["pred_weak_2019yuan"].astype(float)
    true_centered = true - frame.assign(_true=true).groupby("year")["_true"].transform("mean")
    pred_centered = pred - frame.assign(_pred=pred).groupby("year")["_pred"].transform("mean")
    denominator = float(np.sum(np.square(true_centered)))
    return float(1.0 - np.sum(np.square(true_centered - pred_centered)) / denominator)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    settings = deep_get(config, "model.multiyear_leave_province_validation", {})
    years = int_list(settings.get("years"), [2019, 2020, 2021, 2022, 2023])
    mask = str(settings.get("mask", "strict"))
    folds = int(settings.get("folds", 5))
    pixels_per_county = int(settings.get("pixels_per_county", 128))
    epochs = int(settings.get("epochs", 150))
    lr = float(settings.get("lr", 0.001))
    threads = int(settings.get("threads", 8))
    output_regularization = float(settings.get("output_regularization", 0.001))
    output = resolve_path(settings.get("output_root", "outputs/multiyear_leave_province_v1_5_no_road"))
    target_path = resolve_path(deep_get(config, "paths.corrected_county_income"))
    excluded_features = list(deep_get(config, "model.feature_ablation.exclude_features", []))
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Versioned output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    weak.load_targets = target_loader(target_path)
    patch_weak_pixel_loader(weak, excluded_features)
    torch.set_num_threads(max(1, threads))

    print("Preparing fixed multi-year county-pixel samples ...", flush=True)
    sampled: dict[int, dict] = {}
    observation_rows = []
    reference_bands = None
    for year in years:
        sampled[year] = sample_year(
            year, mask, pixels_per_county, seed=weak.RANDOM_STATE + 50_000 + year
        )
        bands = sampled[year]["band_names"]
        if reference_bands is None:
            reference_bands = bands
        elif bands != reference_bands:
            raise RuntimeError(f"Feature order differs in {year}")
        meta = sampled[year]["county_meta"]
        observation_rows.append(pd.DataFrame({
            "year": year,
            "province_std": meta["province_std"].astype(str).to_numpy(),
        }))
    if any(name in reference_bands for name in excluded_features):
        raise RuntimeError(f"Excluded features remain: {excluded_features}")

    observations = pd.concat(observation_rows, ignore_index=True)
    splitter = GroupKFold(n_splits=folds)
    fold_definitions = []
    for fold, (_, test_index) in enumerate(
        splitter.split(
            np.zeros(len(observations)),
            np.zeros(len(observations)),
            groups=observations["province_std"],
        ),
        start=1,
    ):
        held = sorted(observations.iloc[test_index]["province_std"].unique().tolist())
        fold_definitions.append((fold, set(held)))

    prediction_parts = []
    fold_training_rows = []
    for fold, held_out in fold_definitions:
        print(
            f"\nFold {fold}/{folds}: held out all years for {', '.join(sorted(held_out))}",
            flush=True,
        )
        model, imputer, scaler, training = fit_fold(
            sampled,
            years,
            held_out,
            epochs,
            lr,
            seed=weak.RANDOM_STATE + 27_000 + fold,
            output_regularization=output_regularization,
        )
        training.update({
            "fold": fold,
            "held_out_provinces": ";".join(sorted(held_out)),
            "held_out_province_count": len(held_out),
        })
        fold_training_rows.append(training)
        for year in years:
            panel = weak.load_pixel_panel(year, mask)
            test_groups, pred, weight_sum, pixel_count = aggregate_held_out(
                model, imputer, scaler, panel, year, held_out
            )
            meta = panel["county_meta"].iloc[test_groups].copy()
            meta["year"] = year
            meta["fold"] = fold
            meta["held_out_provinces"] = ";".join(sorted(held_out))
            meta["true_2019yuan"] = panel["target"][test_groups]
            meta["pred_weak_2019yuan"] = pred
            meta["rural_weight_sum"] = weight_sum
            meta["rural_pixel_count"] = pixel_count
            prediction_parts.append(meta)
            print(
                f"  {year}: n={len(meta)}, R2={metrics(meta)['r2']:.3f}, "
                f"MAPE={metrics(meta)['mape']:.2f}%",
                flush=True,
            )
            del panel
            gc.collect()
        del model, imputer, scaler
        gc.collect()

    predictions = pd.concat(prediction_parts, ignore_index=True)
    fold_metrics = grouped_metrics(predictions, ["fold"])
    fold_metrics = fold_metrics.merge(pd.DataFrame(fold_training_rows), on="fold", how="left")
    year_metrics = grouped_metrics(predictions, ["year"])
    fold_year_metrics = grouped_metrics(predictions, ["fold", "year"])
    province_metrics = grouped_metrics(predictions, ["province_std"])
    province_year_metrics = grouped_metrics(predictions, ["province_std", "year"])
    overall = metrics(predictions)
    overall.update({
        "scheme": "multiyear_leave_province",
        "years": years,
        "mask": mask,
        "folds": folds,
        "province_count": int(predictions["province_std"].nunique()),
        "county_year_observation_count": int(len(predictions)),
        "pixels_per_county": pixels_per_county,
        "epochs": epochs,
        "learning_rate": lr,
        "output_regularization": output_regularization,
        "within_year_centered_r2": within_year_r2(predictions),
        "excluded_features": excluded_features,
        "input_band_names": reference_bands + ["year_scaled"],
        "province_year_leakage": False,
        "note": "Pooled R2 includes between-year income differences; per-year and within-year-centered R2 are the primary spatial-transfer diagnostics.",
    })

    prefix = output / "multiyear_leave_province_2019_2023_strict"
    predictions.to_csv(Path(f"{prefix}_county_year_predictions.csv"), index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(Path(f"{prefix}_fold_metrics.csv"), index=False, encoding="utf-8-sig")
    year_metrics.to_csv(Path(f"{prefix}_year_metrics.csv"), index=False, encoding="utf-8-sig")
    fold_year_metrics.to_csv(Path(f"{prefix}_fold_year_metrics.csv"), index=False, encoding="utf-8-sig")
    province_metrics.to_csv(Path(f"{prefix}_province_metrics.csv"), index=False, encoding="utf-8-sig")
    province_year_metrics.to_csv(Path(f"{prefix}_province_year_metrics.csv"), index=False, encoding="utf-8-sig")
    Path(f"{prefix}_overall_metrics.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = {
        "version": deep_get(config, "project.version"),
        "scheme": "multiyear_leave_province",
        "config": str(args.config.resolve()),
        "target": str(target_path.relative_to(ROOT)).replace("\\", "/"),
        "target_sha256": sha256(target_path),
        "fold_assignments": [
            {"fold": fold, "held_out_provinces": sorted(held)}
            for fold, held in fold_definitions
        ],
        "settings": {
            "years": years,
            "mask": mask,
            "folds": folds,
            "pixels_per_county": pixels_per_county,
            "epochs": epochs,
            "lr": lr,
            "threads": threads,
            "output_regularization": output_regularization,
            "excluded_features": excluded_features,
            "input_band_names": reference_bands + ["year_scaled"],
        },
        "overall_metrics": overall,
        "historical_outputs_overwritten": False,
    }
    Path(f"{prefix}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 2019—2023年五折留省验证",
        "",
        "每一折将若干省份在2019—2023年的全部县级观测整体留出。模型训练阶段不接触被留出省份任何年份的县级标签。",
        "",
        "## 总体结果",
        "",
        f"- 县—年观测数：{overall['n']:,}。",
        f"- Pooled R²：{overall['r2']:.4f}。",
        f"- 年内中心化R²：{overall['within_year_centered_r2']:.4f}。",
        f"- MAE：{overall['mae']:.1f}元。",
        f"- RMSE：{overall['rmse']:.1f}元。",
        f"- MAPE：{overall['mape']:.2f}%。",
        "",
        "Pooled R²同时包含年度总体增长信号；分年R²和年内中心化R²更直接衡量空间外推能力。",
        "",
        "## 分年结果",
        "",
        year_metrics.to_csv(index=False),
    ]
    Path(f"{prefix}_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

