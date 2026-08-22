"""Quantify validation and spatial stability after removing road_density.

This script is intentionally separate from model fitting so that the comparison
criteria and the comparison artefacts remain auditable.
"""

from __future__ import annotations

import argparse
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
import rasterio


YEARS = (2019, 2020, 2021, 2022, 2023)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def common_values(reference: Path, ablation: Path):
    with rasterio.open(reference) as left, rasterio.open(ablation) as right:
        if left.shape != right.shape or left.transform != right.transform or left.crs != right.crs:
            raise RuntimeError(f"Raster grids differ: {reference} vs {ablation}")
        for _, window in left.block_windows(1):
            x = left.read(1, window=window, masked=True).filled(np.nan).astype(np.float64)
            y = right.read(1, window=window, masked=True).filled(np.nan).astype(np.float64)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.any():
                yield x[valid], y[valid]


def raster_metrics(reference: Path, ablation: Path) -> dict:
    n = 0
    sx = sy = sxx = syy = sxy = 0.0
    abs_parts = []
    ref_parts = []
    for x, y in common_values(reference, ablation):
        n += x.size
        sx += float(x.sum())
        sy += float(y.sum())
        sxx += float(np.dot(x, x))
        syy += float(np.dot(y, y))
        sxy += float(np.dot(x, y))
        abs_parts.append(np.abs(y - x).astype(np.float32))
        ref_parts.append(x.astype(np.float32))
    if n == 0:
        raise RuntimeError(f"No common valid pixels: {reference} vs {ablation}")
    numerator = n * sxy - sx * sy
    denominator = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    correlation = float(numerator / denominator) if denominator > 0 else np.nan
    absolute = np.concatenate(abs_parts)
    reference_values = np.concatenate(ref_parts)
    median_reference = float(np.median(reference_values))
    result = {
        "common_valid_pixels": int(n),
        "pearson_r": correlation,
        "reference_mean": float(sx / n),
        "ablation_mean": float(sy / n),
        "mean_difference": float((sy - sx) / n),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(absolute, dtype=np.float64)))),
        "median_absolute_difference": float(np.median(absolute)),
        "p95_absolute_difference": float(np.percentile(absolute, 95)),
        "reference_median": median_reference,
        "nmae_pct_of_reference_median": float(100.0 * absolute.mean() / median_reference),
        "pixels_abs_difference_gt_1000_pct": float(100.0 * np.mean(absolute > 1000.0)),
        "pixels_abs_difference_gt_3000_pct": float(100.0 * np.mean(absolute > 3000.0)),
    }
    return result


def metric_comparison(reference_path: Path, ablation_path: Path, scheme: str) -> dict:
    reference = read_json(reference_path)
    ablation = read_json(ablation_path)
    row = {"validation_scheme": scheme, "reference_n": reference["n"], "ablation_n": ablation["n"]}
    for name in ("r2", "mae", "rmse", "mape", "bias"):
        row[f"reference_{name}"] = float(reference[name])
        row[f"ablation_{name}"] = float(ablation[name])
        row[f"delta_{name}"] = float(ablation[name] - reference[name])
    return row


def dataframe_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for values in frame.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{float(value):.4f}")
            else:
                rendered.append(str(value))
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join([header, divider, *rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=ROOT / "outputs" / "official77_corrected_full_rerun_v4",
    )
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=ROOT / "outputs" / "sensitivity_no_road_v1_4",
    )
    args = parser.parse_args()
    reference_root = args.reference_root.resolve()
    ablation_root = args.ablation_root.resolve()
    out = ablation_root / "assessment"
    out.mkdir(parents=True, exist_ok=True)

    reference_validation = reference_root / "stage3_weak_supervision" / "spatiotemporal_validation"
    ablation_validation = ablation_root / "stage3_weak_supervision" / "spatiotemporal_validation"
    reference_leave_province = reference_validation / "leave_province_2020_strict_overall_metrics.json"
    if not reference_leave_province.exists():
        # The official77 correction affects 2022 only, so the frozen v1.4 package
        # intentionally reuses the already completed 2020 leave-province result.
        reference_leave_province = (
            ROOT
            / "outputs"
            / "official77_corrected_stage8_v4"
            / "package"
            / "tables"
            / "leave_province_2020_strict_overall_metrics.json"
        )
    validation_rows = [
        metric_comparison(
            reference_validation / "leave_year_2019_2023_strict_overall_metrics.json",
            ablation_validation / "leave_year_2019_2023_strict_overall_metrics.json",
            "leave_year_2019_2023",
        ),
        metric_comparison(
            reference_leave_province,
            ablation_validation / "leave_province_2020_strict_overall_metrics.json",
            "leave_province_2020",
        ),
    ]
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(out / "no_road_validation_comparison.csv", index=False, encoding="utf-8-sig")

    reference_maps = reference_root / "stage5_final_maps"
    ablation_maps = ablation_root / "stage5_final_maps"
    spatial_rows = []
    for year in YEARS:
        name = f"rural_income_{year}_final_calibrated_unbounded_2019yuan_strict.tif"
        row = {"product": f"income_{year}", "unit": "2019 CNY person-1 year-1"}
        row.update(raster_metrics(reference_maps / name, ablation_maps / name))
        spatial_rows.append(row)
    for product, name, unit in (
        (
            "change_2019_2023_absolute",
            "rural_income_change_2019_2023_absolute_unbounded_2019yuan.tif",
            "2019 CNY person-1 year-1",
        ),
        (
            "change_2019_2023_percent",
            "rural_income_change_2019_2023_percent_unbounded.tif",
            "percent",
        ),
    ):
        row = {"product": product, "unit": unit}
        row.update(raster_metrics(reference_maps / name, ablation_maps / name))
        spatial_rows.append(row)
    spatial = pd.DataFrame(spatial_rows)
    spatial.to_csv(out / "no_road_spatial_comparison.csv", index=False, encoding="utf-8-sig")

    # Criteria were fixed before the final spatial statistics were inspected.
    annual = spatial[spatial["product"].str.startswith("income_")]
    validation_pass = bool(
        (validation["delta_r2"].abs() <= 0.02).all()
        and (validation["delta_mape"].abs() <= 1.0).all()
    )
    annual_pattern_pass = bool(
        (annual["pearson_r"] >= 0.95).all()
        and (annual["nmae_pct_of_reference_median"] <= 20.0).all()
    )
    change_absolute = spatial[spatial["product"] == "change_2019_2023_absolute"].iloc[0]
    change_pattern_pass = bool(change_absolute["pearson_r"] >= 0.80)
    overall_pass = validation_pass and annual_pattern_pass and change_pattern_pass

    decision = {
        "excluded_feature": "road_density",
        "reference_model": str(reference_root.relative_to(ROOT)).replace("\\", "/"),
        "ablation_model": str(ablation_root.relative_to(ROOT)).replace("\\", "/"),
        "predefined_criteria": {
            "absolute_delta_r2_max": 0.02,
            "absolute_delta_mape_percentage_points_max": 1.0,
            "annual_map_pearson_r_min": 0.95,
            "annual_map_nmae_pct_of_reference_median_max": 20.0,
            "absolute_change_map_pearson_r_min": 0.80,
        },
        "validation_stability_pass": validation_pass,
        "annual_spatial_pattern_pass": annual_pattern_pass,
        "change_spatial_pattern_pass": change_pattern_pass,
        "overall_sensitivity_pass": overall_pass,
        "interpretation": (
            "The 2026 OSM road-density covariate is not materially influential under the predefined criteria; "
            "the current core product may be retained with this sensitivity analysis."
            if overall_pass
            else "At least one predefined stability criterion failed; the core product should be regenerated "
            "without the post-period road-density covariate rather than relying only on disclosure."
        ),
    }
    (out / "no_road_sensitivity_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# road_density未来信息泄漏敏感性评估",
        "",
        "本实验使用相同的v1.4县级标签、随机种子、抽样量、网络结构、训练轮数和省级校准流程，唯一变化是从训练及全国预测中删除2026年OSM `road_density`。",
        "",
        "## 预先设定的判据",
        "",
        "- 两类县级验证的绝对R²变化不超过0.02，MAPE变化不超过1个百分点。",
        "- 五幅年度图与核心产品的Pearson相关系数均不低于0.95，且相对核心图中位数的NMAE均不超过20%。",
        "- 2019至2023年绝对变化图的Pearson相关系数不低于0.80。",
        "",
        "## 县级验证",
        "",
        dataframe_markdown(validation),
        "",
        "## 空间产品比较",
        "",
        dataframe_markdown(spatial),
        "",
        "## 判定",
        "",
        f"- 县级验证稳定性：{'通过' if validation_pass else '未通过'}。",
        f"- 年度空间格局稳定性：{'通过' if annual_pattern_pass else '未通过'}。",
        f"- 2019至2023年变化格局稳定性：{'通过' if change_pattern_pass else '未通过'}。",
        f"- 总体判定：{'通过' if overall_pass else '未通过'}。",
        f"- 建议：{decision['interpretation']}",
    ]
    (out / "NO_ROAD_SENSITIVITY_ASSESSMENT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

