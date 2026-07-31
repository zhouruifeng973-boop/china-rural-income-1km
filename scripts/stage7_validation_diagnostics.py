# -*- coding: utf-8 -*-
"""Create additional validation diagnostics for the data-product package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STAGE3 = ROOT / "outputs" / "stage3_weak_supervision"
STAGE4 = ROOT / "outputs" / "stage4_province_calibration"
PACKAGE = ROOT / "outputs" / "stage6_scientific_data_product"
TABLES = PACKAGE / "tables"


def metrics(true, pred) -> dict:
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    valid = np.isfinite(true) & np.isfinite(pred)
    true = true[valid]
    pred = pred[valid]
    resid = pred - true
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return {
        "n": int(len(true)),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mape": float(np.mean(np.abs(resid / true)) * 100),
        "bias": float(np.mean(resid)),
        "true_mean": float(np.mean(true)),
        "pred_mean": float(np.mean(pred)),
    }


def grouped_metrics(df: pd.DataFrame, group_cols: list[str], pred_col: str, label: str) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["prediction"] = label
        row.update(metrics(group["true_2019yuan"], group[pred_col]))
        rows.append(row)
    return pd.DataFrame(rows)


def create_leave_year_diagnostics() -> None:
    path = STAGE4 / "leave_year_2019_2023_strict_calibrated_predictions.csv"
    df = pd.read_csv(path)
    by_year = pd.concat(
        [
            grouped_metrics(df, ["year"], "pred_weak_2019yuan", "before_calibration"),
            grouped_metrics(df, ["year"], "pred_calibrated_2019yuan", "after_calibration"),
        ],
        ignore_index=True,
    )
    by_year.to_csv(
        TABLES / "validation_leave_year_by_year_before_after.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_province = pd.concat(
        [
            grouped_metrics(df, ["province_std"], "pred_weak_2019yuan", "before_calibration"),
            grouped_metrics(df, ["province_std"], "pred_calibrated_2019yuan", "after_calibration"),
        ],
        ignore_index=True,
    )
    by_province.to_csv(
        TABLES / "validation_leave_year_by_province_before_after.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_year_province = pd.concat(
        [
            grouped_metrics(
                df, ["year", "province_std"], "pred_weak_2019yuan", "before_calibration"
            ),
            grouped_metrics(
                df, ["year", "province_std"], "pred_calibrated_2019yuan", "after_calibration"
            ),
        ],
        ignore_index=True,
    )
    by_year_province.to_csv(
        TABLES / "validation_leave_year_by_year_province_before_after.csv",
        index=False,
        encoding="utf-8-sig",
    )


def create_leave_province_diagnostics() -> None:
    path = STAGE4 / "leave_province_2020_strict_calibrated_predictions.csv"
    df = pd.read_csv(path)
    by_province = pd.concat(
        [
            grouped_metrics(df, ["province_std"], "pred_weak_2019yuan", "before_calibration"),
            grouped_metrics(df, ["province_std"], "pred_calibrated_2019yuan", "after_calibration"),
        ],
        ignore_index=True,
    )
    by_province.to_csv(
        TABLES / "validation_leave_province_2020_by_province_before_after.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_fold = pd.concat(
        [
            grouped_metrics(df, ["fold"], "pred_weak_2019yuan", "before_calibration"),
            grouped_metrics(df, ["fold"], "pred_calibrated_2019yuan", "after_calibration"),
        ],
        ignore_index=True,
    )
    by_fold.to_csv(
        TABLES / "validation_leave_province_2020_by_fold_before_after.csv",
        index=False,
        encoding="utf-8-sig",
    )


def create_county_validation_residuals() -> None:
    """Write compact residual tables for later plotting or appendix checks."""
    for src, dst in [
        (
            STAGE4 / "leave_year_2019_2023_strict_calibrated_predictions.csv",
            TABLES / "validation_leave_year_county_residuals.csv",
        ),
        (
            STAGE4 / "leave_province_2020_strict_calibrated_predictions.csv",
            TABLES / "validation_leave_province_2020_county_residuals.csv",
        ),
    ]:
        df = pd.read_csv(src)
        keep = [
            "county_code",
            "year",
            "county",
            "city",
            "province_std",
            "true_2019yuan",
            "pred_weak_2019yuan",
            "pred_calibrated_2019yuan",
            "rural_weight_sum",
            "rural_pixel_count",
            "calibration_factor",
        ]
        keep = [c for c in keep if c in df.columns]
        out = df[keep].copy()
        out["residual_before"] = out["pred_weak_2019yuan"] - out["true_2019yuan"]
        out["residual_after"] = out["pred_calibrated_2019yuan"] - out["true_2019yuan"]
        out["abs_error_before"] = out["residual_before"].abs()
        out["abs_error_after"] = out["residual_after"].abs()
        out["ape_before_pct"] = 100 * out["abs_error_before"] / out["true_2019yuan"]
        out["ape_after_pct"] = 100 * out["abs_error_after"] / out["true_2019yuan"]
        out.to_csv(dst, index=False, encoding="utf-8-sig")


def write_readme() -> None:
    text = """# Validation diagnostics

这些表是在已有 Stage 3/Stage 4 验证结果基础上进一步拆分得到的，不重新训练模型。

## 表格说明

- `validation_leave_year_by_year_before_after.csv`：2019—2023 留一年验证，按年份统计校准前后误差。
- `validation_leave_year_by_province_before_after.csv`：2019—2023 留一年验证，按省份统计校准前后误差。
- `validation_leave_year_by_year_province_before_after.csv`：2019—2023 留一年验证，按年份—省份统计误差。
- `validation_leave_province_2020_by_province_before_after.csv`：2020 留省验证，按省份统计校准前后误差。
- `validation_leave_province_2020_by_fold_before_after.csv`：2020 留省验证，按折统计校准前后误差。
- `validation_leave_year_county_residuals.csv`：留一年验证的县级残差明细。
- `validation_leave_province_2020_county_residuals.csv`：2020 留省验证的县级残差明细。

## 解释边界

这些指标仍然是县级回聚合后的验证指标，不是像元级真实精度。省级校准后的省级均值一致是约束，不应被表述为独立验证。
"""
    (TABLES / "README_validation_diagnostics.md").write_text(text, encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    create_leave_year_diagnostics()
    create_leave_province_diagnostics()
    create_county_validation_residuals()
    write_readme()
    print("Validation diagnostics written to:")
    print(TABLES)


if __name__ == "__main__":
    main()
