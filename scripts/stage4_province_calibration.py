# -*- coding: utf-8 -*-
"""Stage 4: calibrate Stage 3 predictions to official province income."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from stage3_weak_supervised_downscaling import ROOT, metric_dict

IN_DIR = ROOT / "outputs" / "stage3_weak_supervision" / "spatiotemporal_validation"
OUT_DIR = ROOT / "outputs" / "stage4_province_calibration"
OFFICIAL_CSV = ROOT / "data" / "processed" / "province_income_2019_2023_deflated.csv"


def weighted_mean(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float(np.sum(v[keep] * w[keep]) / np.sum(w[keep]))


def calibrate(pred, official, label):
    pred = pred.copy()
    pred["year"] = pred["year"].astype(int)
    keys = ["year", "province_std"]
    rows = []
    for (year, province), g in pred.groupby(keys, sort=True):
        rows.append({
            "year": int(year),
            "province_std": province,
            "county_count": len(g),
            "covered_rural_weight": float(g["rural_weight_sum"].sum()),
            "predicted_province_mean_2019yuan": weighted_mean(
                g["pred_weak_2019yuan"], g["rural_weight_sum"]),
            "observed_county_weighted_mean_2019yuan": weighted_mean(
                g["true_2019yuan"], g["rural_weight_sum"]),
        })
    province = pd.DataFrame(rows).merge(official, on=keys, how="left", validate="one_to_one")
    if province["official_province_income_2019yuan"].isna().any():
        missing = province.loc[province["official_province_income_2019yuan"].isna(), keys]
        raise RuntimeError(f"Missing official province income:\n{missing}")

    province["calibration_factor"] = (
        province["official_province_income_2019yuan"]
        / province["predicted_province_mean_2019yuan"]
    )
    province["prediction_gap_before_pct"] = 100 * (
        province["predicted_province_mean_2019yuan"]
        / province["official_province_income_2019yuan"] - 1
    )
    province["official_vs_covered_counties_pct"] = 100 * (
        province["official_province_income_2019yuan"]
        / province["observed_county_weighted_mean_2019yuan"] - 1
    )

    pred = pred.merge(province[keys + ["calibration_factor"]], on=keys,
                      how="left", validate="many_to_one")
    pred["pred_calibrated_2019yuan"] = (
        pred["pred_weak_2019yuan"] * pred["calibration_factor"]
    )
    before = metric_dict(pred["true_2019yuan"].to_numpy(float),
                         pred["pred_weak_2019yuan"].to_numpy(float))
    after = metric_dict(pred["true_2019yuan"].to_numpy(float),
                        pred["pred_calibrated_2019yuan"].to_numpy(float))

    group_rows = []
    for (year, province_name), g in pred.groupby(keys, sort=True):
        mb = metric_dict(g["true_2019yuan"].to_numpy(float),
                         g["pred_weak_2019yuan"].to_numpy(float))
        ma = metric_dict(g["true_2019yuan"].to_numpy(float),
                         g["pred_calibrated_2019yuan"].to_numpy(float))
        group_rows.append({
            "year": int(year), "province_std": province_name, "n": len(g),
            "r2_before": mb["r2"], "r2_after": ma["r2"],
            "mae_before": mb["mae"], "mae_after": ma["mae"],
            "bias_before": mb["bias"], "bias_after": ma["bias"],
            "spearman_before": g["true_2019yuan"].corr(
                g["pred_weak_2019yuan"], method="spearman"),
            "spearman_after": g["true_2019yuan"].corr(
                g["pred_calibrated_2019yuan"], method="spearman"),
        })

    summary = {
        "label": label,
        "county_records": int(len(pred)),
        "province_years": int(len(province)),
        "before": before,
        "after": after,
        "r2_change": after["r2"] - before["r2"],
        "mae_change": after["mae"] - before["mae"],
        "factor_min": float(province["calibration_factor"].min()),
        "factor_median": float(province["calibration_factor"].median()),
        "factor_max": float(province["calibration_factor"].max()),
        "province_mape_before": float(province["prediction_gap_before_pct"].abs().mean()),
        "province_mape_after": 0.0,
        "note": "The zero post-calibration province MAPE is a constraint, not validation accuracy."
    }
    return pred, province, pd.DataFrame(group_rows), summary


def save(name, result):
    pred, province, groups, summary = result
    pred.to_csv(OUT_DIR / f"{name}_calibrated_predictions.csv", index=False,
                encoding="utf-8-sig")
    province.to_csv(OUT_DIR / f"{name}_province_factors.csv", index=False,
                    encoding="utf-8-sig")
    groups.to_csv(OUT_DIR / f"{name}_province_metrics.csv", index=False,
                  encoding="utf-8-sig")
    (OUT_DIR / f"{name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(spatial, temporal):
    s, t = spatial[3], temporal[3]
    report = f"""# Stage 4：省级统计约束校准诊断

校准公式：省内所有预测值乘以“官方省级农村收入 / 模型人口加权省级收入”。统一正比例缩放不会改变省内县级或像元级排序。

| 验证 | 指标 | 校准前 | 校准后 | 变化 |
|---|---|---:|---:|---:|
| 2020 留省 | R² | {s['before']['r2']:.3f} | {s['after']['r2']:.3f} | {s['r2_change']:+.3f} |
| 2020 留省 | MAE（元） | {s['before']['mae']:.0f} | {s['after']['mae']:.0f} | {s['mae_change']:+.0f} |
| 2020 留省 | MAPE | {s['before']['mape']:.2f}% | {s['after']['mape']:.2f}% | {s['after']['mape']-s['before']['mape']:+.2f} 个百分点 |
| 2019—2023 留一年 | R² | {t['before']['r2']:.3f} | {t['after']['r2']:.3f} | {t['r2_change']:+.3f} |
| 2019—2023 留一年 | MAE（元） | {t['before']['mae']:.0f} | {t['after']['mae']:.0f} | {t['mae_change']:+.0f} |
| 2019—2023 留一年 | MAPE | {t['before']['mape']:.2f}% | {t['after']['mape']:.2f}% | {t['after']['mape']-t['before']['mape']:+.2f} 个百分点 |

2020 留省校准系数范围为 {s['factor_min']:.3f}—{s['factor_max']:.3f}，中位数 {s['factor_median']:.3f}。省级误差校准后为零是数学约束，不是独立精度；是否有效应以县级留省指标是否改善为准。
"""
    (OUT_DIR / "stage4_province_calibration_summary.md").write_text(report, encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    official = pd.read_csv(OFFICIAL_CSV)
    official["year"] = official["year"].astype(int)
    official = official[["year", "province_std", "rural_income_2019yuan"]].rename(
        columns={"rural_income_2019yuan": "official_province_income_2019yuan"})

    spatial = calibrate(pd.read_csv(
        IN_DIR / "leave_province_2020_strict_county_predictions.csv"),
        official, "leave_province_2020_strict")
    temporal = calibrate(pd.read_csv(
        IN_DIR / "leave_year_2019_2023_strict_county_predictions.csv"),
        official, "leave_year_2019_2023_strict")
    save("leave_province_2020_strict", spatial)
    save("leave_year_2019_2023_strict", temporal)
    write_report(spatial, temporal)
    print(json.dumps({"spatial": spatial[3], "temporal": temporal[3]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
