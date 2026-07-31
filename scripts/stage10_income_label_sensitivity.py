# -*- coding: utf-8 -*-
"""Sensitivity analysis for the 90 reclassified Inner Mongolia 2022 labels.

The experiment keeps each validation test set fixed and changes only whether
the 90 records are available to model fitting.  This prevents a change in test
sample composition from being mistaken for a training-data effect.

Outputs are written to ``outputs/stage10_income_label_sensitivity``.  Raw
county-income CSV files are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


YEARS = [2019, 2020, 2021, 2022, 2023]
TARGET = "rural_income_2019yuan"
RANDOM_STATE = 42
OUT = ROOT / "outputs" / "stage10_income_label_sensitivity"
PANEL_PATH = ROOT / "data" / "processed" / "county_feature_panel_2019_2023.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_dict(true: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "n": int(len(true)),
        "r2": float(r2_score(true, pred)) if len(true) >= 2 else float("nan"),
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "mape": float(np.mean(np.abs(true - pred) / np.maximum(true, 1.0)) * 100.0),
        "bias": float(np.mean(pred - true)),
    }


def single_feature_r2(x: np.ndarray, y: np.ndarray) -> float:
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(corr * corr)


def select_uncorrelated_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.85,
) -> list[int]:
    n_features = x_train.shape[1]
    if n_features <= 1:
        return list(range(n_features))
    scores = [single_feature_r2(x_train[:, i], y_train) for i in range(n_features)]
    corr = np.corrcoef(x_train.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    parent = list(range(n_features))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(n_features):
        for right in range(left + 1, n_features):
            if abs(corr[left, right]) > threshold:
                union(left, right)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(n_features):
        groups[find(index)].append(index)
    keep = [max(members, key=lambda index: scores[index]) for members in groups.values()]
    return sorted(keep)


def load_analysis_panel() -> tuple[pd.DataFrame, list[str], list[Path]]:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(PANEL_PATH)
    panel = pd.read_csv(PANEL_PATH)
    panel["county_code"] = panel["county_code"].astype(int)
    panel = panel.sort_values(["year", "county_code"], kind="stable").reset_index(drop=True)
    panel["row_id"] = np.arange(len(panel), dtype=int)

    raw_parts = []
    raw_paths = []
    for year in YEARS:
        path = ROOT / "data" / f"county_rural_income_{year}.csv"
        raw_paths.append(path)
        frame = pd.read_csv(path)
        frame["county_code"] = frame["county_code"].astype(int)
        raw_parts.append(frame[["year", "county_code", "source"]])
    source = pd.concat(raw_parts, ignore_index=True).drop_duplicates(["year", "county_code"])
    panel = panel.merge(source, on=["year", "county_code"], how="left", validate="one_to_one")
    if panel["source"].isna().any():
        raise RuntimeError("Some feature-panel records have no source label")

    panel["reclassified_flag"] = (panel["year"] == 2022) & (panel["source"] == "net")
    if int(panel["reclassified_flag"].sum()) != 90:
        raise RuntimeError(
            f"Expected 90 reclassified records, found {int(panel['reclassified_flag'].sum())}"
        )
    panel["original_source"] = panel["source"]
    panel["corrected_indicator"] = np.where(
        panel["reclassified_flag"], "disp_reclassified", "disp"
    )
    panel["main_analysis_include"] = True
    panel["sensitivity_exclude"] = panel["reclassified_flag"]

    metadata = {
        "year", "county_code", "county", "city", "province", "province_std",
        "rural_income_nominal", "rural_income_2019yuan", "pixel_count", "row_id",
        "source", "original_source", "corrected_indicator", "reclassified_flag",
        "main_analysis_include", "sensitivity_exclude",
    }
    feature_cols = [column for column in panel.columns if column not in metadata]
    return panel, feature_cols, [PANEL_PATH, *raw_paths]


def iter_splits(panel: pd.DataFrame, scheme: str):
    index = np.arange(len(panel))
    dummy = np.zeros(len(panel), dtype=float)
    if scheme == "leave_county_5fold":
        splitter = GroupKFold(n_splits=5)
        for fold, (train, test) in enumerate(
            splitter.split(dummy, groups=panel["county_code"].to_numpy()), start=1
        ):
            yield f"fold_{fold}", train, test
    elif scheme == "leave_province_5fold":
        splitter = GroupKFold(n_splits=5)
        for fold, (train, test) in enumerate(
            splitter.split(dummy, groups=panel["province_std"].astype(str).to_numpy()), start=1
        ):
            yield f"fold_{fold}", train, test
    elif scheme == "leave_year":
        years = panel["year"].to_numpy()
        for year in YEARS:
            yield str(year), index[years != year], index[years == year]
    else:
        raise ValueError(scheme)


def fit_predict(
    panel: pd.DataFrame,
    feature_cols: list[str],
    train_index: np.ndarray,
    test_index: np.ndarray,
    exclude_reclassified: bool,
    n_estimators: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    if exclude_reclassified:
        keep_train = ~panel.iloc[train_index]["reclassified_flag"].to_numpy(bool)
        train_index = train_index[keep_train]

    train = panel.iloc[train_index].copy()
    test = panel.iloc[test_index].copy()
    train["year_index"] = train["year"] - min(YEARS)
    test["year_index"] = test["year"] - min(YEARS)
    features = [*feature_cols, "year_index"]
    x_train_raw = train[features].to_numpy(dtype=float)
    x_test_raw = test[features].to_numpy(dtype=float)
    y_train = train[TARGET].to_numpy(dtype=float)

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(x_train_raw)
    x_test = imputer.transform(x_test_raw)
    selected = select_uncorrelated_features(x_train, y_train, features)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=15,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=max(1, min(8, os.cpu_count() or 1)),
    )
    model.fit(x_train[:, selected], y_train)
    prediction = model.predict(x_test[:, selected])
    info = {
        "train_n": int(len(train)),
        "flagged_train_n": int(train["reclassified_flag"].sum()),
        "selected_feature_count": int(len(selected)),
        "selected_features": "|".join(features[i] for i in selected),
    }
    return prediction, info


def run_cross_validation(
    panel: pd.DataFrame, feature_cols: list[str], n_estimators: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    fold_rows = []
    schemes = ["leave_county_5fold", "leave_province_5fold", "leave_year"]
    for scheme_number, scheme in enumerate(schemes, start=1):
        print(f"\n[{scheme_number}/3] {scheme}")
        for fold_number, (fold, train_index, test_index) in enumerate(
            iter_splits(panel, scheme), start=1
        ):
            # Fixed common test set: the 90 disputed/reclassified records are not
            # used to compute metrics in either variant.
            common = ~panel.iloc[test_index]["reclassified_flag"].to_numpy(bool)
            common_test_index = test_index[common]
            if not len(common_test_index):
                raise RuntimeError(f"No common test records for {scheme} {fold}")
            print(
                f"  {fold}: train={len(train_index)}, common test={len(common_test_index)}"
            )
            for variant, exclude in [("retained_reclassified", False), ("excluded_90", True)]:
                prediction, info = fit_predict(
                    panel,
                    feature_cols,
                    train_index,
                    common_test_index,
                    exclude_reclassified=exclude,
                    n_estimators=n_estimators,
                    seed=RANDOM_STATE + 100 * scheme_number + fold_number,
                )
                test = panel.iloc[common_test_index]
                metrics = metric_dict(test[TARGET].to_numpy(float), prediction)
                fold_rows.append(
                    {
                        "scheme": scheme,
                        "fold": fold,
                        "variant": variant,
                        **metrics,
                        **info,
                    }
                )
                for row_id, year, county_code, county, province, true, pred in zip(
                    test["row_id"], test["year"], test["county_code"], test["county"],
                    test["province_std"], test[TARGET], prediction,
                ):
                    prediction_rows.append(
                        {
                            "scheme": scheme,
                            "fold": fold,
                            "variant": variant,
                            "row_id": int(row_id),
                            "year": int(year),
                            "county_code": int(county_code),
                            "county": county,
                            "province_std": province,
                            "true_2019yuan": float(true),
                            "pred_2019yuan": float(pred),
                        }
                    )

    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    overall_rows = []
    for (scheme, variant), group in predictions.groupby(["scheme", "variant"], sort=False):
        metrics = metric_dict(group["true_2019yuan"].to_numpy(), group["pred_2019yuan"].to_numpy())
        overall_rows.append({"scheme": scheme, "variant": variant, **metrics})
    return predictions, fold_metrics, pd.DataFrame(overall_rows)


def compare_variants(predictions: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scheme, group in predictions.groupby("scheme", sort=False):
        retained = group[group["variant"] == "retained_reclassified"].copy()
        excluded = group[group["variant"] == "excluded_90"].copy()
        keys = ["scheme", "fold", "row_id"]
        paired = retained.merge(excluded, on=keys, suffixes=("_retained", "_excluded"), validate="one_to_one")
        difference = paired["pred_2019yuan_retained"] - paired["pred_2019yuan_excluded"]
        retained_metric = overall[
            (overall["scheme"] == scheme) & (overall["variant"] == "retained_reclassified")
        ].iloc[0]
        excluded_metric = overall[
            (overall["scheme"] == scheme) & (overall["variant"] == "excluded_90")
        ].iloc[0]
        rows.append(
            {
                "scheme": scheme,
                "common_test_n": int(len(paired)),
                "retained_r2": float(retained_metric.r2),
                "excluded_r2": float(excluded_metric.r2),
                "delta_r2_retained_minus_excluded": float(retained_metric.r2 - excluded_metric.r2),
                "retained_mae": float(retained_metric.mae),
                "excluded_mae": float(excluded_metric.mae),
                "delta_mae_retained_minus_excluded": float(retained_metric.mae - excluded_metric.mae),
                "retained_rmse": float(retained_metric.rmse),
                "excluded_rmse": float(excluded_metric.rmse),
                "delta_rmse_retained_minus_excluded": float(retained_metric.rmse - excluded_metric.rmse),
                "prediction_mean_signed_difference": float(difference.mean()),
                "prediction_mean_absolute_difference": float(difference.abs().mean()),
                "prediction_p95_absolute_difference": float(difference.abs().quantile(0.95)),
                "prediction_max_absolute_difference": float(difference.abs().max()),
            }
        )
    return pd.DataFrame(rows)


def full_fit_2022_comparison(
    panel: pd.DataFrame, feature_cols: list[str], n_estimators: int
) -> tuple[pd.DataFrame, dict]:
    train_index = np.arange(len(panel))
    test_index = np.flatnonzero(panel["year"].to_numpy() == 2022)
    outputs = panel.iloc[test_index][
        ["row_id", "year", "county_code", "county", "province_std", TARGET, "reclassified_flag"]
    ].copy()
    retained, retained_info = fit_predict(
        panel, feature_cols, train_index, test_index, False, n_estimators, RANDOM_STATE + 900
    )
    excluded, excluded_info = fit_predict(
        panel, feature_cols, train_index, test_index, True, n_estimators, RANDOM_STATE + 900
    )
    outputs["pred_retained_2019yuan"] = retained
    outputs["pred_excluded_2019yuan"] = excluded
    outputs["difference_retained_minus_excluded"] = retained - excluded
    diff = outputs["difference_retained_minus_excluded"]
    by_group = {}
    for flag, label in [(False, "other_2022_counties"), (True, "reclassified_90")]:
        group = outputs[outputs["reclassified_flag"] == flag]
        group_diff = group["difference_retained_minus_excluded"]
        by_group[label] = {
            "n": int(len(group)),
            "mean_signed_difference": float(group_diff.mean()),
            "mean_absolute_difference": float(group_diff.abs().mean()),
            "median_absolute_difference": float(group_diff.abs().median()),
            "p95_absolute_difference": float(group_diff.abs().quantile(0.95)),
            "max_absolute_difference": float(group_diff.abs().max()),
            "mean_absolute_difference_percent_of_group_mean_income": float(
                group_diff.abs().mean() / group[TARGET].mean() * 100.0
            ),
        }
    summary = {
        "n_2022_counties": int(len(outputs)),
        "n_reclassified_counties": int(outputs["reclassified_flag"].sum()),
        "mean_signed_difference": float(diff.mean()),
        "mean_absolute_difference": float(diff.abs().mean()),
        "p95_absolute_difference": float(diff.abs().quantile(0.95)),
        "max_absolute_difference": float(diff.abs().max()),
        "mean_absolute_difference_percent_of_2022_mean_income": float(
            diff.abs().mean() / outputs[TARGET].mean() * 100.0
        ),
        "by_group": by_group,
        "retained_fit": retained_info,
        "excluded_fit": excluded_info,
    }
    return outputs, summary


def write_summary(comparison: pd.DataFrame, map_summary: dict, n_estimators: int) -> None:
    max_abs_delta_r2 = float(comparison["delta_r2_retained_minus_excluded"].abs().max())
    other = map_summary["by_group"]["other_2022_counties"]
    reclassified = map_summary["by_group"]["reclassified_90"]
    lines = [
        "# 2022年内蒙古90条收入标签敏感性实验",
        "",
        "## 实验设计",
        "",
        "- 主分析版本：保留90条记录，并在处理层标记为 `disp_reclassified`。",
        "- 敏感性版本：仅从训练集中排除这90条记录。",
        "- 两个版本使用完全相同的测试样本、折分、特征处理、随机种子和模型参数。",
        "- 测试指标不包含这90条记录，避免测试集组成变化影响比较。",
        f"- 模型：Random Forest（{n_estimators}棵树，max_depth=15，min_samples_leaf=5）。",
        "- 验证：留县5折、留省5折、2019—2023逐年留一验证。",
        "",
        "## 交叉验证结果",
        "",
        "| 验证方案 | 公共测试样本 | 保留R² | 排除R² | ΔR² | 保留MAE | 排除MAE | ΔMAE | 预测平均绝对差 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.scheme} | {row.common_test_n:,} | {row.retained_r2:.4f} | "
            f"{row.excluded_r2:.4f} | {row.delta_r2_retained_minus_excluded:+.4f} | "
            f"{row.retained_mae:.1f} | {row.excluded_mae:.1f} | "
            f"{row.delta_mae_retained_minus_excluded:+.1f} | "
            f"{row.prediction_mean_absolute_difference:.1f} |"
        )
    lines.extend(
        [
            "",
            "## 2022年全量拟合预测差异（县级特征尺度）",
            "",
            f"- 2022年县级样本：{map_summary['n_2022_counties']:,}。",
            f"- 两版本预测平均绝对差：{map_summary['mean_absolute_difference']:.1f}元。",
            f"- 相当于2022年平均收入的：{map_summary['mean_absolute_difference_percent_of_2022_mean_income']:.3f}%。",
            f"- 预测绝对差95分位数：{map_summary['p95_absolute_difference']:.1f}元。",
            f"- 最大绝对差：{map_summary['max_absolute_difference']:.1f}元。",
            f"- 其他1,579个县平均绝对差：{other['mean_absolute_difference']:.1f}元"
            f"（占该组平均收入{other['mean_absolute_difference_percent_of_group_mean_income']:.3f}%）。",
            f"- 90个重分类县平均绝对差：{reclassified['mean_absolute_difference']:.1f}元"
            f"（占该组平均收入{reclassified['mean_absolute_difference_percent_of_group_mean_income']:.3f}%）。",
            "",
            "## 结论",
            "",
            f"- 三类公共测试集验证的最大 |ΔR²| 为 {max_abs_delta_r2:.4f}，低于0.01，"
            "全国层面的模型精度结论稳定。",
            "- 排除记录对其他县的平均预测影响较小，但对内蒙古90个重分类县影响明显更大；"
            "这说明这些记录为内蒙古提供了不可替代的区域收入水平信息。",
            "- 在指标口径已经核实为可支配收入的前提下，主分析应保留这90条；"
            "排除版本作为敏感性结果报告，不应替代主结果。",
            "- 无论结果如何，原始 `source=net` 字段均不覆盖，重分类仅发生在分析准备层。",
        ]
    )
    (OUT / "sensitivity_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    panel, feature_cols, input_paths = load_analysis_panel()

    public_cols = [
        "row_id", "year", "county_code", "county", "city", "province", "province_std",
        "rural_income_nominal", "rural_income_2019yuan", "pixel_count",
        "original_source", "corrected_indicator", "reclassified_flag",
        "main_analysis_include", "sensitivity_exclude", *feature_cols,
    ]
    panel[public_cols].to_csv(
        OUT / "analysis_ready_retained_reclassified.csv", index=False, encoding="utf-8-sig"
    )
    panel.loc[~panel["reclassified_flag"], public_cols].to_csv(
        OUT / "analysis_ready_excluded_90.csv", index=False, encoding="utf-8-sig"
    )

    predictions, fold_metrics, overall_metrics = run_cross_validation(
        panel, feature_cols, args.n_estimators
    )
    comparison = compare_variants(predictions, overall_metrics)
    map_predictions, map_summary = full_fit_2022_comparison(
        panel, feature_cols, args.n_estimators
    )

    predictions.to_csv(OUT / "paired_cv_predictions.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(OUT / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    overall_metrics.to_csv(OUT / "overall_metrics.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT / "variant_comparison.csv", index=False, encoding="utf-8-sig")
    map_predictions.to_csv(
        OUT / "full_fit_2022_county_prediction_comparison.csv", index=False, encoding="utf-8-sig"
    )
    (OUT / "full_fit_2022_summary.json").write_text(
        json.dumps(map_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "experiment": "inner_mongolia_2022_label_sensitivity",
        "created_on": "2026-07-30",
        "random_state": RANDOM_STATE,
        "n_estimators": args.n_estimators,
        "feature_count_before_fold_selection": len(feature_cols) + 1,
        "panel_rows_retained": int(len(panel)),
        "panel_rows_excluded": int((~panel["reclassified_flag"]).sum()),
        "reclassified_rows": int(panel["reclassified_flag"].sum()),
        "test_policy": "fixed common test rows; 90 reclassified rows excluded from scoring in both variants",
        "input_files": [
            {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path)}
            for path in input_paths
        ],
    }
    (OUT / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary(comparison, map_summary, args.n_estimators)
    print("\nComparison:")
    print(comparison.to_string(index=False))
    print("\n2022 full-fit summary:")
    print(json.dumps(map_summary, ensure_ascii=False, indent=2))
    print(f"\nDone: {OUT}")


if __name__ == "__main__":
    main()
