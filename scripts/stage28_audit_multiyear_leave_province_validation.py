"""Independent audit of the multi-year leave-province validation artefacts."""

from __future__ import annotations

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


def metric_dict(frame: pd.DataFrame) -> dict:
    true = frame["true_2019yuan"].to_numpy(float)
    pred = frame["pred_weak_2019yuan"].to_numpy(float)
    residual = pred - true
    return {
        "n": int(len(frame)),
        "r2": float(1.0 - np.sum(np.square(residual)) / np.sum(np.square(true - true.mean()))),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mape": float(100.0 * np.mean(np.abs(residual) / np.maximum(np.abs(true), 1e-9))),
        "bias": float(np.mean(residual)),
    }


def close(a: float, b: float, tolerance: float = 1e-6) -> bool:
    """Allow harmless CSV round-trip differences from float32 predictions."""
    return bool(abs(float(a) - float(b)) <= tolerance * max(1.0, abs(float(a)), abs(float(b))))


def main() -> None:
    predictions = pd.read_csv(Path(f"{PREFIX}_county_year_predictions.csv"), encoding="utf-8-sig")
    fold_metrics = pd.read_csv(Path(f"{PREFIX}_fold_metrics.csv"), encoding="utf-8-sig")
    year_metrics = pd.read_csv(Path(f"{PREFIX}_year_metrics.csv"), encoding="utf-8-sig")
    manifest = json.loads(Path(f"{PREFIX}_manifest.json").read_text(encoding="utf-8"))
    reported = json.loads(Path(f"{PREFIX}_overall_metrics.json").read_text(encoding="utf-8"))
    recomputed = metric_dict(predictions)

    province_fold_counts = predictions.groupby("province_std")["fold"].nunique()
    province_year_fold_counts = predictions.groupby(["province_std", "year"])["fold"].nunique()
    duplicates = int(predictions.duplicated(["year", "county_code"]).sum())
    manifest_assignment = {
        province: int(item["fold"])
        for item in manifest["fold_assignments"]
        for province in item["held_out_provinces"]
    }
    observed_assignment = (
        predictions[["province_std", "fold"]]
        .drop_duplicates()
        .set_index("province_std")["fold"]
        .astype(int)
        .to_dict()
    )
    metric_names = ("r2", "mae", "rmse", "mape", "bias")
    overall_match = all(close(recomputed[name], reported[name]) for name in metric_names)

    year_matches = []
    for _, row in year_metrics.iterrows():
        recalculated = metric_dict(predictions[predictions["year"].astype(int) == int(row["year"])])
        year_matches.append(all(close(recalculated[name], row[name]) for name in metric_names))

    checks = {
        "expected_county_year_records": 8275,
        "actual_county_year_records": int(len(predictions)),
        "unique_provinces": int(predictions["province_std"].nunique()),
        "unique_years": sorted(predictions["year"].astype(int).unique().tolist()),
        "fold_count": int(predictions["fold"].nunique()),
        "duplicate_year_county_records": duplicates,
        "every_province_occurs_in_one_fold": bool((province_fold_counts == 1).all()),
        "every_province_year_occurs_in_one_fold": bool((province_year_fold_counts == 1).all()),
        "manifest_assignment_matches_predictions": manifest_assignment == observed_assignment,
        "overall_metrics_recompute_match": overall_match,
        "all_year_metrics_recompute_match": bool(all(year_matches)),
        "road_density_absent": "road_density" not in manifest["settings"]["input_band_names"],
        "input_feature_count_including_year": len(manifest["settings"]["input_band_names"]),
        "all_fold_predictions_finite": bool(
            np.isfinite(predictions[["true_2019yuan", "pred_weak_2019yuan"]].to_numpy(float)).all()
        ),
        "all_fold_metrics_finite": bool(
            np.isfinite(fold_metrics[["r2", "mae", "rmse", "mape", "bias"]].to_numpy(float)).all()
        ),
    }
    passed = (
        checks["actual_county_year_records"] == checks["expected_county_year_records"]
        and checks["unique_provinces"] == 21
        and checks["unique_years"] == [2019, 2020, 2021, 2022, 2023]
        and checks["fold_count"] == 5
        and checks["duplicate_year_county_records"] == 0
        and checks["every_province_occurs_in_one_fold"]
        and checks["every_province_year_occurs_in_one_fold"]
        and checks["manifest_assignment_matches_predictions"]
        and checks["overall_metrics_recompute_match"]
        and checks["all_year_metrics_recompute_match"]
        and checks["road_density_absent"]
        and checks["input_feature_count_including_year"] == 28
        and checks["all_fold_predictions_finite"]
        and checks["all_fold_metrics_finite"]
    )
    checks["audit_status"] = "PASS" if passed else "FAIL"
    (RUN / "stage28_audit.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    weakest_r2 = fold_metrics.loc[fold_metrics["r2"].idxmin()]
    highest_mape = fold_metrics.loc[fold_metrics["mape"].idxmax()]
    lines = [
        "# 2019—2023年多年份留省验证审查与解释",
        "",
        f"**审查结果：{checks['audit_status']}**",
        "",
        "## 验证设计",
        "",
        "- 2019—2023年全部8,275条有效县—年观测共同参与五折GroupKFold。",
        "- 分组变量为省份，同一省份所有年份只属于一个测试折。",
        "- 每折训练均完全排除测试省份五年的县级标签。",
        "- 模型使用27个空间特征和1个年份变量，未使用road_density。",
        "- 结果是省级校准前的纯模型空间外推能力，不利用测试省份的县级观测或省级均值修正预测。",
        "",
        "## 总体结果",
        "",
        f"- Pooled R²：{reported['r2']:.3f}。",
        f"- 年内中心化R²：{reported['within_year_centered_r2']:.3f}。",
        f"- MAE：{reported['mae']:.0f}元。",
        f"- RMSE：{reported['rmse']:.0f}元。",
        f"- MAPE：{reported['mape']:.2f}%。",
        f"- Bias：{reported['bias']:.0f}元。",
        "",
        "## 分年结果",
        "",
        "| 年份 | n | R² | MAE（元） | RMSE（元） | MAPE（%） | Bias（元） |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in year_metrics.iterrows():
        lines.append(
            f"| {int(row['year'])} | {int(row['n'])} | {row['r2']:.3f} | "
            f"{row['mae']:.0f} | {row['rmse']:.0f} | {row['mape']:.2f} | {row['bias']:.0f} |"
        )
    lines.extend([
        "",
        "## 分折异质性",
        "",
        "| 折 | 留出省份 | n | R² | MAPE（%） | Bias（元） |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for _, row in fold_metrics.iterrows():
        lines.append(
            f"| {int(row['fold'])} | {row['held_out_provinces'].replace(';', '、')} | "
            f"{int(row['n'])} | {row['r2']:.3f} | {row['mape']:.2f} | {row['bias']:.0f} |"
        )
    lines.extend([
        "",
        "## 结论边界",
        "",
        "- 升级后的验证设计覆盖了真实的跨省外推场景，原先仅有2020年留省验证的设计短板已经解决。",
        "- 总体MAPE低于20%，但Pooled R²为0.398，没有达到预设的0.45—0.55范围；不能为了结果美观改写数值。",
        "- 各年总体R²为0.329—0.370，说明跨省空间外推能力为中等且弱于时间迁移能力。",
        f"- 最低R²折为第{int(weakest_r2['fold'])}折（R²={weakest_r2['r2']:.3f}）；"
        f"最高MAPE折为第{int(highest_mape['fold'])}折（MAPE={highest_mape['mape']:.2f}%）。",
        "- 结果支持数据集用于宏观和区域格局分析，但无标签省份的县级和1 km局地解释必须结合质量图层并保持谨慎。",
        "- 该实验验证21个具有至少一年县级标签的省份之间的迁移；对其余完全无标签省份仍属于外推风险提示，而不是直接观测验证。",
    ])
    (RUN / "MULTIYEAR_LEAVE_PROVINCE_ASSESSMENT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

