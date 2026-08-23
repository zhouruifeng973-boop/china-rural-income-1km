"""Sensitivity test for the 533 labels pending record-level numeric verification.

Both variants use the same province folds and the same verified test records.
The baseline predictions come from the frozen v1.5-no-road validation run. The
sensitivity variant excludes all pending records from model fitting. County
income values and county-level predictions remain internal and are not copied
to the public release package.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
from collections import defaultdict
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
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch

from feature_ablation import patch_weak_pixel_loader
from repro_config import deep_get, int_list, load_config
import stage3_weak_supervised_downscaling as weak
from stage3_spatiotemporal_validation import add_year_column
import stage27_multiyear_leave_province_validation as stage27


CONFIG = ROOT / "config" / "workflow_v1.5_no_road_core.json"
INDEX = (
    ROOT
    / "outputs"
    / "provenance_batch_20260822"
    / "county_training_label_provenance_index_v2.4_jiangsu_corrected.csv"
)
BASELINE_ROOT = ROOT / "outputs" / "multiyear_leave_province_v1_5_no_road"
BASELINE_PREDICTIONS = (
    BASELINE_ROOT
    / "multiyear_leave_province_2019_2023_strict_county_year_predictions.csv"
)
BASELINE_MANIFEST = (
    BASELINE_ROOT
    / "multiyear_leave_province_2019_2023_strict_manifest.json"
)
OUT = ROOT / "outputs" / "stage31_pending_533_label_sensitivity"
VERIFIED_STATUSES = {
    "record_value_verified",
    "exact_source_item_and_value_verified",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_dict(true: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    return weak.metric_dict(np.asarray(true, dtype=float), np.asarray(pred, dtype=float))


def fit_fold_verified_only(
    sampled: dict[int, dict],
    years: list[int],
    held_out_provinces: set[str],
    verified_keys: set[tuple[int, int]],
    epochs: int,
    lr: float,
    seed: int,
    output_regularization: float,
):
    x_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    group_offset = 0

    for year in years:
        sample = sampled[year]
        meta = sample["county_meta"]
        provinces = meta["province_std"].astype(str).to_numpy()
        county_codes = meta["county_code"].astype(int).to_numpy()
        verified_mask = np.asarray(
            [(int(year), int(code)) in verified_keys for code in county_codes],
            dtype=bool,
        )
        eligible = (~np.isin(provinces, list(held_out_provinces))) & verified_mask
        train_groups = np.flatnonzero(eligible)
        lookup = np.full(len(provinces), -1, dtype=np.int64)
        lookup[train_groups] = np.arange(len(train_groups), dtype=np.int64)
        pixel_mask = np.isin(sample["groups"], train_groups)
        x_parts.append(add_year_column(sample["X"][pixel_mask], year))
        weight_parts.append(sample["weights"][pixel_mask])
        group_parts.append(lookup[sample["groups"][pixel_mask]] + group_offset)
        target_parts.append(sample["target"][train_groups])
        group_offset += len(train_groups)

    x = np.concatenate(x_parts, axis=0)
    weights = np.concatenate(weight_parts).astype(np.float32, copy=False)
    groups = np.concatenate(group_parts).astype(np.int64, copy=False)
    targets = np.concatenate(target_parts).astype(np.float32, copy=False)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(imputer.fit_transform(x)).astype(np.float32)
    del x, x_parts, weight_parts, group_parts, target_parts
    gc.collect()

    weak.set_seed(seed)
    x_t = torch.from_numpy(x_scaled)
    w_t = torch.from_numpy(weights)
    g_t = torch.from_numpy(groups)
    log_y_t = torch.log(torch.from_numpy(targets))
    model = weak.IncomeMLP(
        x_scaled.shape[1], init_log_income=float(np.log(np.median(targets)))
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
        if epoch in {1, epochs} or epoch % max(50, epochs // 3) == 0:
            print(f"    epoch {epoch:4d}/{epochs}: loss={last_loss:.6f}", flush=True)

    return model.eval(), imputer, scaler, {
        "sampled_pixels": int(len(x_scaled)),
        "train_verified_county_years": int(len(targets)),
        "final_loss": last_loss,
    }


def summarize_by(frame: pd.DataFrame, columns: list[str], variant: str) -> pd.DataFrame:
    rows = []
    key = columns[0] if len(columns) == 1 else columns
    for values, group in frame.groupby(key, dropna=False):
        values = (values,) if len(columns) == 1 else values
        row = {name: value for name, value in zip(columns, values)}
        row.update(metric_dict(group["true_2019yuan"], group["pred_weak_2019yuan"]))
        row["variant"] = variant
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Versioned output is not empty: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    config = load_config(CONFIG)
    settings = deep_get(config, "model.multiyear_leave_province_validation", {})
    years = int_list(settings.get("years"), [2019, 2020, 2021, 2022, 2023])
    mask = str(settings.get("mask", "strict"))
    pixels_per_county = int(settings.get("pixels_per_county", 128))
    epochs = int(settings.get("epochs", 150))
    lr = float(settings.get("lr", 0.001))
    threads = int(settings.get("threads", 8))
    output_regularization = float(settings.get("output_regularization", 0.001))
    target_path = stage27.resolve_path(deep_get(config, "paths.corrected_county_income"))
    excluded_features = list(deep_get(config, "model.feature_ablation.exclude_features", []))

    index = pd.read_csv(INDEX, dtype={"county_code": str})
    index["county_code"] = index["county_code"].astype(int)
    verified = index["source_verification_status"].isin(VERIFIED_STATUSES)
    verified_keys = set(
        zip(index.loc[verified, "year"].astype(int), index.loc[verified, "county_code"])
    )
    pending_keys = set(
        zip(index.loc[~verified, "year"].astype(int), index.loc[~verified, "county_code"])
    )
    if len(index) != 8373 or len(verified_keys) != 7840 or len(pending_keys) != 533:
        raise RuntimeError(
            f"Unexpected provenance counts: {len(index)}, {len(verified_keys)}, {len(pending_keys)}"
        )

    baseline = pd.read_csv(BASELINE_PREDICTIONS)
    baseline["county_code"] = baseline["county_code"].astype(int)
    baseline["year"] = baseline["year"].astype(int)
    baseline = baseline[
        [
            (int(year), int(code)) in verified_keys
            for year, code in zip(baseline["year"], baseline["county_code"])
        ]
    ].copy()
    baseline["variant"] = "all_8373_labels"

    manifest = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    fold_definitions = [
        (int(row["fold"]), set(map(str, row["held_out_provinces"])))
        for row in manifest["fold_assignments"]
    ]

    weak.load_targets = stage27.target_loader(target_path)
    patch_weak_pixel_loader(weak, excluded_features)
    torch.set_num_threads(max(1, threads))

    print("Preparing the same fixed full-label pixel samples ...", flush=True)
    sampled: dict[int, dict] = {}
    reference_bands = None
    for year in years:
        sampled[year] = stage27.sample_year(
            year, mask, pixels_per_county, seed=weak.RANDOM_STATE + 50_000 + year
        )
        bands = sampled[year]["band_names"]
        if reference_bands is None:
            reference_bands = bands
        elif bands != reference_bands:
            raise RuntimeError(f"Feature order differs in {year}")
    if any(name in reference_bands for name in excluded_features):
        raise RuntimeError(f"Excluded features remain: {excluded_features}")

    prediction_parts = []
    training_rows = []
    for fold, held_out in fold_definitions:
        print(
            f"\nFold {fold}/5 verified-only fitting; held out: {', '.join(sorted(held_out))}",
            flush=True,
        )
        model, imputer, scaler, training = fit_fold_verified_only(
            sampled,
            years,
            held_out,
            verified_keys,
            epochs,
            lr,
            seed=weak.RANDOM_STATE + 27_000 + fold,
            output_regularization=output_regularization,
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
            meta["county_code"] = meta["county_code"].astype(int)
            keep = np.asarray(
                [
                    (int(year), int(code)) in verified_keys
                    for code in meta["county_code"].to_numpy()
                ],
                dtype=bool,
            )
            meta = meta.loc[keep].copy()
            meta["year"] = year
            meta["fold"] = fold
            meta["true_2019yuan"] = panel["target"][test_groups][keep]
            meta["pred_weak_2019yuan"] = pred[keep]
            meta["rural_weight_sum"] = weight_sum[keep]
            meta["rural_pixel_count"] = pixel_count[keep]
            prediction_parts.append(meta)
            result = metric_dict(meta["true_2019yuan"], meta["pred_weak_2019yuan"])
            print(
                f"  {year}: verified test n={len(meta)}, "
                f"R2={result['r2']:.3f}, MAPE={result['mape']:.2f}%",
                flush=True,
            )
            del panel
            gc.collect()
        del model, imputer, scaler
        gc.collect()

    sensitivity = pd.concat(prediction_parts, ignore_index=True)
    sensitivity["variant"] = "verified_7840_labels_only"
    keys = ["year", "county_code", "fold"]
    paired = baseline.merge(
        sensitivity,
        on=keys,
        suffixes=("_baseline", "_verified_only"),
        validate="one_to_one",
    )
    if len(paired) != len(baseline) or len(paired) != len(sensitivity):
        raise RuntimeError(
            f"Common-test mismatch: paired={len(paired)}, baseline={len(baseline)}, "
            f"verified_only={len(sensitivity)}"
        )

    overall_rows = []
    for variant, frame in [
        ("all_8373_labels", baseline),
        ("verified_7840_labels_only", sensitivity),
    ]:
        row = {"variant": variant}
        row.update(metric_dict(frame["true_2019yuan"], frame["pred_weak_2019yuan"]))
        overall_rows.append(row)
    overall = pd.DataFrame(overall_rows)
    fold_metrics = pd.concat([
        summarize_by(baseline, ["fold"], "all_8373_labels"),
        summarize_by(sensitivity, ["fold"], "verified_7840_labels_only"),
    ], ignore_index=True)
    year_metrics = pd.concat([
        summarize_by(baseline, ["year"], "all_8373_labels"),
        summarize_by(sensitivity, ["year"], "verified_7840_labels_only"),
    ], ignore_index=True)

    baseline_pred = paired["pred_weak_2019yuan_baseline"].astype(float)
    verified_pred = paired["pred_weak_2019yuan_verified_only"].astype(float)
    delta = verified_pred - baseline_pred
    pearson = float(np.corrcoef(baseline_pred, verified_pred)[0, 1])
    spearman = float(spearmanr(baseline_pred, verified_pred).statistic)
    prediction_summary = {
        "common_verified_test_records": int(len(paired)),
        "pearson_prediction_correlation": pearson,
        "spearman_prediction_correlation": spearman,
        "mean_signed_difference_verified_only_minus_all": float(delta.mean()),
        "mean_absolute_difference": float(delta.abs().mean()),
        "median_absolute_difference": float(delta.abs().median()),
        "p95_absolute_difference": float(delta.abs().quantile(0.95)),
        "max_absolute_difference": float(delta.abs().max()),
        "mean_absolute_difference_percent_of_baseline_mean_prediction": float(
            delta.abs().mean() / baseline_pred.mean() * 100.0
        ),
    }

    baseline_metrics = overall.iloc[0]
    verified_metrics = overall.iloc[1]
    deltas = {
        "delta_r2_verified_only_minus_all": float(
            verified_metrics.r2 - baseline_metrics.r2
        ),
        "delta_mape_percentage_points_verified_only_minus_all": float(
            verified_metrics.mape - baseline_metrics.mape
        ),
        "delta_mae_verified_only_minus_all": float(
            verified_metrics.mae - baseline_metrics.mae
        ),
        "delta_rmse_verified_only_minus_all": float(
            verified_metrics.rmse - baseline_metrics.rmse
        ),
    }
    thresholds = {
        "maximum_absolute_delta_r2": 0.03,
        "maximum_absolute_delta_mape_percentage_points": 1.0,
        "minimum_prediction_pearson_correlation": 0.98,
        "maximum_mean_absolute_prediction_difference_percent": 5.0,
    }
    checks = {
        "r2_stable": abs(deltas["delta_r2_verified_only_minus_all"])
        <= thresholds["maximum_absolute_delta_r2"],
        "mape_stable": abs(deltas["delta_mape_percentage_points_verified_only_minus_all"])
        <= thresholds["maximum_absolute_delta_mape_percentage_points"],
        "prediction_correlation_stable": pearson
        >= thresholds["minimum_prediction_pearson_correlation"],
        "prediction_difference_stable": prediction_summary[
            "mean_absolute_difference_percent_of_baseline_mean_prediction"
        ]
        <= thresholds["maximum_mean_absolute_prediction_difference_percent"],
    }
    decision = {
        "status": "sensitivity_pass" if all(checks.values()) else "sensitivity_requires_review",
        "checks": checks,
        "thresholds": thresholds,
        "metric_deltas": deltas,
        "prediction_comparison": prediction_summary,
        "interpretation_scope": (
            "County-level multi-year province-transfer validation. This experiment "
            "does not constitute pixel-level accuracy validation."
        ),
    }

    overall.to_csv(OUT / "pending_533_overall_metrics.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(OUT / "pending_533_fold_metrics.csv", index=False, encoding="utf-8-sig")
    year_metrics.to_csv(OUT / "pending_533_year_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(training_rows).to_csv(
        OUT / "pending_533_training_counts_by_fold.csv", index=False, encoding="utf-8-sig"
    )
    (OUT / "pending_533_prediction_comparison_summary.json").write_text(
        json.dumps(prediction_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "pending_533_sensitivity_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run_manifest = {
        "experiment": "pending_533_county_training_label_sensitivity",
        "config": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": sha256(CONFIG),
        "provenance_index": str(INDEX.relative_to(ROOT)).replace("\\", "/"),
        "provenance_index_sha256": sha256(INDEX),
        "baseline_predictions_internal": str(BASELINE_PREDICTIONS.relative_to(ROOT)).replace("\\", "/"),
        "baseline_predictions_sha256": sha256(BASELINE_PREDICTIONS),
        "baseline_manifest": str(BASELINE_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "baseline_manifest_sha256": sha256(BASELINE_MANIFEST),
        "target_file_internal_not_redistributed": str(target_path.relative_to(ROOT)).replace("\\", "/"),
        "target_sha256": sha256(target_path),
        "total_labels": int(len(index)),
        "verified_labels": int(len(verified_keys)),
        "pending_labels_excluded_from_sensitivity_fit": int(len(pending_keys)),
        "common_verified_test_records": int(len(paired)),
        "years": years,
        "mask": mask,
        "folds": len(fold_definitions),
        "fold_assignments": [
            {"fold": fold, "held_out_provinces": sorted(held)}
            for fold, held in fold_definitions
        ],
        "pixels_per_county": pixels_per_county,
        "epochs": epochs,
        "learning_rate": lr,
        "output_regularization": output_regularization,
        "excluded_features": excluded_features,
        "random_seed_policy": {
            "pixel_sampling": "42 + 50000 + year",
            "model_fit": "42 + 27000 + fold",
        },
        "fixed_test_policy": (
            "Both variants are scored on the same 7,753 source-verified county-year "
            "records with valid pixel panels. Pending records are excluded from "
            "scoring in both variants."
        ),
        "county_values_or_predictions_publicly_released": False,
    }
    (OUT / "pending_533_run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    overall_table_lines = [
        "| variant | n | R2 | MAE | RMSE | MAPE (%) | bias |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall.itertuples(index=False):
        overall_table_lines.append(
            f"| {row.variant} | {row.n:,} | {row.r2:.4f} | {row.mae:.1f} | "
            f"{row.rmse:.1f} | {row.mape:.2f} | {row.bias:.1f} |"
        )
    lines = [
        "# 533条待数值核验标签排除敏感性实验",
        "",
        "## 设计",
        "",
        "- 基准模型保留全部8,373条标签。",
        "- 敏感性模型仅使用7,840条已完成县名和数值核验的标签。",
        "- 两个版本采用相同的省份分组、特征、像元抽样规则、模型参数和随机种子。",
        "- 两个版本均只在具有有效像元面板的7,753条已核验县—年记录上评分。",
        "",
        "## 总体结果",
        "",
        *overall_table_lines,
        "",
        "## 差异",
        "",
        f"- ΔR²：{deltas['delta_r2_verified_only_minus_all']:+.4f}。",
        f"- ΔMAPE：{deltas['delta_mape_percentage_points_verified_only_minus_all']:+.3f}个百分点。",
        f"- 两版本预测Pearson相关：{pearson:.5f}。",
        f"- 两版本预测Spearman相关：{spearman:.5f}。",
        f"- 平均绝对预测差：{prediction_summary['mean_absolute_difference']:.1f}元，"
        f"占基准平均预测的{prediction_summary['mean_absolute_difference_percent_of_baseline_mean_prediction']:.3f}%。",
        "",
        "## 判定",
        "",
        f"- `{decision['status']}`。",
        "- 本实验控制县级训练标签核验状态风险，但不能替代1 km像元级真实收入验证。",
    ]
    (OUT / "pending_533_sensitivity_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
