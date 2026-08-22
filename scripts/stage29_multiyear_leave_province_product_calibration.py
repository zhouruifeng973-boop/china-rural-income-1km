"""Apply operational province-year constraints to multi-year held-out predictions.

This is a product-level diagnostic, not fully independent model validation. The
county labels remain held out. Calibration uses only the official province-year
income constraint that is also used to generate the released data product.
"""

from __future__ import annotations

import hashlib
import json
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


RUN = ROOT / "outputs" / "multiyear_leave_province_v1_5_no_road"
PREFIX = RUN / "multiyear_leave_province_2019_2023_strict"
PREDICTIONS = Path(f"{PREFIX}_county_year_predictions.csv")
OFFICIAL = ROOT / "data" / "processed" / "province_income_2019_2023_deflated.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    value = values.to_numpy(float)
    weight = weights.to_numpy(float)
    keep = np.isfinite(value) & np.isfinite(weight) & (weight > 0)
    return float(np.sum(value[keep] * weight[keep]) / np.sum(weight[keep]))


def metrics(frame: pd.DataFrame, prediction: str) -> dict:
    true = frame["true_2019yuan"].to_numpy(float)
    pred = frame[prediction].to_numpy(float)
    residual = pred - true
    denominator = np.sum(np.square(true - true.mean()))
    return {
        "n": int(len(frame)),
        "r2": float(1.0 - np.sum(np.square(residual)) / denominator),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mape": float(100.0 * np.mean(np.abs(residual) / np.maximum(np.abs(true), 1e-9))),
        "bias": float(np.mean(residual)),
    }


def grouped_metrics(frame: pd.DataFrame, groups: list[str], prediction: str) -> pd.DataFrame:
    rows = []
    key = groups[0] if len(groups) == 1 else groups
    for values, group in frame.groupby(key, sort=True, dropna=False):
        values = (values,) if len(groups) == 1 else values
        row = {name: value for name, value in zip(groups, values)}
        row.update(metrics(group, prediction))
        rows.append(row)
    return pd.DataFrame(rows)


def centered_r2(frame: pd.DataFrame, prediction: str, groups: list[str]) -> float:
    true = frame["true_2019yuan"].astype(float)
    pred = frame[prediction].astype(float)
    true_centered = true - frame.assign(_v=true).groupby(groups)["_v"].transform("mean")
    pred_centered = pred - frame.assign(_v=pred).groupby(groups)["_v"].transform("mean")
    denominator = float(np.sum(np.square(true_centered)))
    return float(1.0 - np.sum(np.square(true_centered - pred_centered)) / denominator)


def main() -> None:
    pred = pd.read_csv(PREDICTIONS, encoding="utf-8-sig")
    pred["year"] = pred["year"].astype(int)
    official = pd.read_csv(OFFICIAL, encoding="utf-8-sig")
    official["year"] = official["year"].astype(int)
    official = official[["year", "province_std", "rural_income_2019yuan"]].rename(
        columns={"rural_income_2019yuan": "official_province_income_2019yuan"}
    )

    factor_rows = []
    for (year, province), group in pred.groupby(["year", "province_std"], sort=True):
        factor_rows.append({
            "year": int(year),
            "province_std": province,
            "fold": int(group["fold"].iloc[0]),
            "county_count": int(len(group)),
            "covered_rural_weight": float(group["rural_weight_sum"].sum()),
            "predicted_covered_county_mean_2019yuan": weighted_mean(
                group["pred_weak_2019yuan"], group["rural_weight_sum"]
            ),
        })
    factors = pd.DataFrame(factor_rows).merge(
        official, on=["year", "province_std"], how="left", validate="one_to_one"
    )
    if factors["official_province_income_2019yuan"].isna().any():
        missing = factors.loc[
            factors["official_province_income_2019yuan"].isna(), ["year", "province_std"]
        ]
        raise RuntimeError(f"Missing official province-year constraints:\n{missing}")
    factors["calibration_factor"] = (
        factors["official_province_income_2019yuan"]
        / factors["predicted_covered_county_mean_2019yuan"]
    )
    factors["prediction_gap_before_pct"] = 100.0 * (
        factors["predicted_covered_county_mean_2019yuan"]
        / factors["official_province_income_2019yuan"] - 1.0
    )

    pred = pred.merge(
        factors[["year", "province_std", "calibration_factor"]],
        on=["year", "province_std"], how="left", validate="many_to_one",
    )
    pred["pred_calibrated_2019yuan"] = (
        pred["pred_weak_2019yuan"] * pred["calibration_factor"]
    )

    before = metrics(pred, "pred_weak_2019yuan")
    after = metrics(pred, "pred_calibrated_2019yuan")
    before["within_year_centered_r2"] = centered_r2(
        pred, "pred_weak_2019yuan", ["year"]
    )
    after["within_year_centered_r2"] = centered_r2(
        pred, "pred_calibrated_2019yuan", ["year"]
    )
    before["within_province_year_centered_r2"] = centered_r2(
        pred, "pred_weak_2019yuan", ["province_std", "year"]
    )
    after["within_province_year_centered_r2"] = centered_r2(
        pred, "pred_calibrated_2019yuan", ["province_std", "year"]
    )

    year_before = grouped_metrics(pred, ["year"], "pred_weak_2019yuan")
    year_after = grouped_metrics(pred, ["year"], "pred_calibrated_2019yuan")
    year_comparison = year_before.merge(
        year_after, on=["year", "n"], suffixes=("_before", "_after"), validate="one_to_one"
    )
    fold_before = grouped_metrics(pred, ["fold"], "pred_weak_2019yuan")
    fold_after = grouped_metrics(pred, ["fold"], "pred_calibrated_2019yuan")
    fold_comparison = fold_before.merge(
        fold_after, on=["fold", "n"], suffixes=("_before", "_after"), validate="one_to_one"
    )
    held = pred[["fold", "held_out_provinces"]].drop_duplicates()
    fold_comparison = fold_comparison.merge(held, on="fold", validate="one_to_one")
    fold_year_before = grouped_metrics(pred, ["fold", "year"], "pred_weak_2019yuan")
    fold_year_after = grouped_metrics(pred, ["fold", "year"], "pred_calibrated_2019yuan")
    fold_year_comparison = fold_year_before.merge(
        fold_year_after,
        on=["fold", "year", "n"],
        suffixes=("_before", "_after"),
        validate="one_to_one",
    )

    summary = {
        "scheme": "2019_2023_multiyear_groupkfold_by_province_product_calibration",
        "county_year_records": int(len(pred)),
        "province_count": int(pred["province_std"].nunique()),
        "province_year_count": int(len(factors)),
        "before_calibration": before,
        "after_calibration": after,
        "change": {
            "r2": after["r2"] - before["r2"],
            "mae": after["mae"] - before["mae"],
            "rmse": after["rmse"] - before["rmse"],
            "mape_percentage_points": after["mape"] - before["mape"],
        },
        "calibration_factor": {
            "min": float(factors["calibration_factor"].min()),
            "median": float(factors["calibration_factor"].median()),
            "max": float(factors["calibration_factor"].max()),
        },
        "interpretation": (
            "After-calibration metrics are product-level diagnostics using official province-year "
            "constraints available in the operational workflow. They are not fully independent "
            "model-validation metrics and do not validate within-province pixel-scale accuracy."
        ),
        "calibration_scope_note": (
            "Factors are computed from rural-population-weighted predictions for the held-out "
            "counties represented in the validation panel. The released raster workflow computes "
            "factors from all valid rural pixels in each province-year."
        ),
        "input_sha256": {
            PREDICTIONS.name: sha256(PREDICTIONS),
            OFFICIAL.name: sha256(OFFICIAL),
        },
    }

    stem = RUN / "multiyear_leave_province_2019_2023_strict_product_calibrated"
    pred.to_csv(Path(f"{stem}_county_year_predictions.csv"), index=False, encoding="utf-8-sig")
    factors.to_csv(Path(f"{stem}_province_year_factors.csv"), index=False, encoding="utf-8-sig")
    year_comparison.to_csv(Path(f"{stem}_year_comparison.csv"), index=False, encoding="utf-8-sig")
    fold_comparison.to_csv(Path(f"{stem}_fold_comparison.csv"), index=False, encoding="utf-8-sig")
    fold_year_comparison.to_csv(
        Path(f"{stem}_fold_year_comparison.csv"), index=False, encoding="utf-8-sig"
    )
    Path(f"{stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 2019—2023年多年份留省验证：省级校准前后对照",
        "",
        "县级标签在每折中仍按省份整体留出。校准后结果仅使用最终产品生产流程中可获得的官方省—年农村收入约束。",
        "",
        "| 指标 | 校准前 | 校准后 | 变化 |",
        "|---|---:|---:|---:|",
        f"| R² | {before['r2']:.3f} | {after['r2']:.3f} | {after['r2']-before['r2']:+.3f} |",
        f"| MAE（元） | {before['mae']:.0f} | {after['mae']:.0f} | {after['mae']-before['mae']:+.0f} |",
        f"| RMSE（元） | {before['rmse']:.0f} | {after['rmse']:.0f} | {after['rmse']-before['rmse']:+.0f} |",
        f"| MAPE（%） | {before['mape']:.2f} | {after['mape']:.2f} | {after['mape']-before['mape']:+.2f} |",
        f"| 年内中心化R² | {before['within_year_centered_r2']:.3f} | {after['within_year_centered_r2']:.3f} | {after['within_year_centered_r2']-before['within_year_centered_r2']:+.3f} |",
        f"| 省—年内中心化R² | {before['within_province_year_centered_r2']:.3f} | {after['within_province_year_centered_r2']:.3f} | {after['within_province_year_centered_r2']-before['within_province_year_centered_r2']:+.3f} |",
        "",
        "校准后指标用于评价最终产品工作流，而不是独立模型精度。乘法校准不能增加省内县级排序信息，也不能代替1 km真实观测验证。",
    ]
    Path(f"{stem}_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

