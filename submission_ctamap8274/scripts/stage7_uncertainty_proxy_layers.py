# -*- coding: utf-8 -*-
"""Create uncertainty-proxy / quality-risk layers for the Stage 6 package.

These layers are not a substitute for a true multi-seed ensemble uncertainty
estimate. They combine three already defensible risk signals:

1. Province support: whether the pixel is inside a province with county-level
   training labels in that year.
2. Calibration strength: how far the province-year calibration factor is from 1.
3. Bound sensitivity: how different the bounded sensitivity product is from the
   unbounded canonical product.

The resulting score is 0-100, where larger values mean higher quality risk /
interpretation caution.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

LOCAL_DEPS = Path(__file__).resolve().parents[1] / "stage6_pydeps"
# The bundled stage6_pydeps directory was built for CPython 3.12.  Do not
# shadow a working environment on other Python versions (notably 3.11).
if LOCAL_DEPS.exists() and sys.version_info[:2] == (3, 12):
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd
import rasterio


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "outputs" / "stage6_scientific_data_product"
INCOME = PACKAGE / "data" / "income_maps"
QUALITY = PACKAGE / "data" / "quality_layers"
OUT = PACKAGE / "data" / "uncertainty_proxy_layers"
TABLES = PACKAGE / "tables"
METADATA = PACKAGE / "metadata"

YEARS = [2019, 2020, 2021, 2022, 2023]
NODATA_FLOAT = -9999.0
NODATA_UINT8 = 255


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def income_path(year: int) -> Path:
    return INCOME / f"china_rural_income_1km_{year}_2019yuan_strict_unbounded.tif"


def support_path(year: int) -> Path:
    return QUALITY / f"quality_training_province_support_flag_{year}_strict.tif"


def factor_path(year: int) -> Path:
    return QUALITY / f"quality_province_calibration_factor_{year}_strict_unbounded.tif"


def diff_path(year: int) -> Path:
    return QUALITY / f"quality_bounded_unbounded_absdiff_{year}_2019yuan_strict.tif"


def score_path(year: int) -> Path:
    return OUT / f"uncertainty_proxy_score_{year}_strict.tif"


def class_path(year: int) -> Path:
    return OUT / f"uncertainty_proxy_class_{year}_strict.tif"


def float_profile(src):
    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=NODATA_FLOAT,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    return profile


def uint8_profile(src):
    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        nodata=NODATA_UINT8,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    profile.pop("predictor", None)
    return profile


def classify(score: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full(score.shape, NODATA_UINT8, dtype=np.uint8)
    out[valid & (score < 25)] = 1
    out[valid & (score >= 25) & (score < 50)] = 2
    out[valid & (score >= 50)] = 3
    return out


def create_year(year: int) -> dict:
    scores = []
    class_counts = {1: 0, 2: 0, 3: 0}
    component_sums = {
        "support_score": 0.0,
        "factor_score": 0.0,
        "sensitivity_score": 0.0,
    }
    valid_count = 0

    with rasterio.open(income_path(year)) as income_src, \
        rasterio.open(support_path(year)) as support_src, \
        rasterio.open(factor_path(year)) as factor_src, \
        rasterio.open(diff_path(year)) as diff_src, \
        rasterio.open(score_path(year), "w", **float_profile(income_src)) as score_dst, \
        rasterio.open(class_path(year), "w", **uint8_profile(income_src)) as class_dst:

        income_nodata = income_src.nodata if income_src.nodata is not None else NODATA_FLOAT
        factor_nodata = factor_src.nodata if factor_src.nodata is not None else NODATA_FLOAT
        diff_nodata = diff_src.nodata if diff_src.nodata is not None else NODATA_FLOAT

        for _, window in income_src.block_windows(1):
            income = income_src.read(1, window=window).astype(np.float32)
            support = support_src.read(1, window=window).astype(np.uint8)
            factor = factor_src.read(1, window=window).astype(np.float32)
            diff = diff_src.read(1, window=window).astype(np.float32)

            valid = np.isfinite(income) & (income != income_nodata)
            valid &= np.isfinite(factor) & (factor != factor_nodata)

            # Component 1: extrapolation support risk.
            support_score = np.zeros(income.shape, dtype=np.float32)
            support_score[valid & (support == 0)] = 30.0
            support_score[valid & (support == 1)] = 0.0

            # Component 2: calibration-factor magnitude risk.
            # abs(log(factor)) = 0 at factor=1; a multiplicative deviation
            # around 1.5 receives the maximum 40 points.
            safe_factor = np.where(valid, np.maximum(factor, 1e-6), 1.0)
            factor_score = 40.0 * np.abs(np.log(safe_factor)) / np.log(1.5)
            factor_score = np.clip(factor_score, 0.0, 40.0).astype(np.float32)
            factor_score[~valid] = 0.0

            # Component 3: sensitivity to the bounded-vs-unbounded decision.
            # A 5% difference or larger receives the maximum 30 points.
            diff_valid = valid & np.isfinite(diff) & (diff != diff_nodata)
            sensitivity_pct = np.zeros(income.shape, dtype=np.float32)
            sensitivity_pct[diff_valid] = 100.0 * diff[diff_valid] / np.maximum(
                income[diff_valid], 1.0
            )
            sensitivity_score = np.clip(30.0 * sensitivity_pct / 5.0, 0.0, 30.0)
            sensitivity_score[~valid] = 0.0

            score = np.full(income.shape, NODATA_FLOAT, dtype=np.float32)
            score[valid] = np.clip(
                support_score[valid] + factor_score[valid] + sensitivity_score[valid],
                0.0,
                100.0,
            )
            score_dst.write(score, 1, window=window)

            cls = classify(score, valid)
            class_dst.write(cls, 1, window=window)

            v = score[valid]
            if v.size:
                scores.append(v.astype(np.float32))
                valid_count += int(v.size)
                for key, arr in [
                    ("support_score", support_score),
                    ("factor_score", factor_score),
                    ("sensitivity_score", sensitivity_score),
                ]:
                    component_sums[key] += float(arr[valid].sum())
                for c in [1, 2, 3]:
                    class_counts[c] += int((cls == c).sum())

    values = np.concatenate(scores) if scores else np.array([], dtype=np.float32)
    stats = {
        "year": year,
        "valid_pixels": valid_count,
        "score_mean": float(values.mean()) if values.size else np.nan,
        "score_median": float(np.median(values)) if values.size else np.nan,
        "score_p90": float(np.quantile(values, 0.90)) if values.size else np.nan,
        "score_p95": float(np.quantile(values, 0.95)) if values.size else np.nan,
        "score_max": float(values.max()) if values.size else np.nan,
        "low_risk_pixels": class_counts[1],
        "moderate_risk_pixels": class_counts[2],
        "high_risk_pixels": class_counts[3],
        "low_risk_pct": 100.0 * class_counts[1] / valid_count if valid_count else np.nan,
        "moderate_risk_pct": 100.0 * class_counts[2] / valid_count if valid_count else np.nan,
        "high_risk_pct": 100.0 * class_counts[3] / valid_count if valid_count else np.nan,
        "mean_support_component": component_sums["support_score"] / valid_count
        if valid_count
        else np.nan,
        "mean_factor_component": component_sums["factor_score"] / valid_count
        if valid_count
        else np.nan,
        "mean_sensitivity_component": component_sums["sensitivity_score"] / valid_count
        if valid_count
        else np.nan,
    }
    return stats


def create_multiyear_summary_layers() -> dict:
    mean_path = OUT / "uncertainty_proxy_score_2019_2023_mean_strict.tif"
    max_path = OUT / "uncertainty_proxy_score_2019_2023_max_strict.tif"

    with rasterio.open(score_path(2019)) as ref:
        profile = float_profile(ref)
        nodata = ref.nodata if ref.nodata is not None else NODATA_FLOAT
        all_values_mean = []
        all_values_max = []
        valid_count = 0
        with rasterio.open(mean_path, "w", **profile) as mean_dst, \
            rasterio.open(max_path, "w", **profile) as max_dst:
            sources = [rasterio.open(score_path(year)) for year in YEARS]
            try:
                for _, window in ref.block_windows(1):
                    arrays = []
                    valids = []
                    for src in sources:
                        arr = src.read(1, window=window).astype(np.float32)
                        arrays.append(arr)
                        valids.append(np.isfinite(arr) & (arr != nodata))
                    stack = np.stack(arrays, axis=0)
                    valid_stack = np.stack(valids, axis=0)
                    common = np.all(valid_stack, axis=0)
                    mean_arr = np.full(stack.shape[1:], NODATA_FLOAT, dtype=np.float32)
                    max_arr = np.full(stack.shape[1:], NODATA_FLOAT, dtype=np.float32)
                    mean_arr[common] = np.mean(stack[:, common], axis=0)
                    max_arr[common] = np.max(stack[:, common], axis=0)
                    mean_dst.write(mean_arr, 1, window=window)
                    max_dst.write(max_arr, 1, window=window)
                    if np.any(common):
                        all_values_mean.append(mean_arr[common])
                        all_values_max.append(max_arr[common])
                        valid_count += int(common.sum())
            finally:
                for src in sources:
                    src.close()

    mean_values = np.concatenate(all_values_mean)
    max_values = np.concatenate(all_values_max)
    return {
        "common_valid_pixels": valid_count,
        "mean_score_mean": float(mean_values.mean()),
        "mean_score_median": float(np.median(mean_values)),
        "mean_score_p95": float(np.quantile(mean_values, 0.95)),
        "max_score_mean": float(max_values.mean()),
        "max_score_median": float(np.median(max_values)),
        "max_score_p95": float(np.quantile(max_values, 0.95)),
    }


def update_manifest() -> None:
    records = []
    for path in sorted(OUT.glob("*.tif")):
        records.append(
            {
                "role": "uncertainty_proxy_layer",
                "package_file": str(path.relative_to(PACKAGE)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(records).to_csv(
        METADATA / "uncertainty_proxy_layer_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )


def write_readme(summary: pd.DataFrame, multi: dict) -> None:
    lines = [
        "# Uncertainty proxy / quality-risk layers",
        "",
        "这些图层是“不确定性代理”或“质量风险指数”，不是多随机种子模型集成得到的真实预测标准差。",
        "它们用于先把外推风险、校准幅度、bounded/unbounded敏感性整合为可制图、可解释的风险层。",
        "",
        "## 评分构成",
        "",
        "- 省份支持风险：外推省份 +30 分，有县级训练标签省份 +0 分。",
        "- 省级校准风险：根据 `abs(log(calibration_factor))` 计分，最高 40 分。",
        "- 上下限敏感性风险：bounded 与 unbounded 差异达到或超过 5% 时最高 30 分。",
        "",
        "总分范围为 0—100，分值越高，解释时越需要谨慎。",
        "",
        "## 风险等级",
        "",
        "- 1：低风险，score < 25。",
        "- 2：中等风险，25 ≤ score < 50。",
        "- 3：高风险，score ≥ 50。",
        "- 255：NoData。",
        "",
        "## 年度摘要",
        "",
        "| year | valid_pixels | mean | median | p95 | low_pct | moderate_pct | high_pct |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {year} | {valid_pixels:,} | {score_mean:.1f} | {score_median:.1f} | {score_p95:.1f} | {low_risk_pct:.2f} | {moderate_risk_pct:.2f} | {high_risk_pct:.2f} |".format(
                year=int(row["year"]),
                valid_pixels=int(row["valid_pixels"]),
                score_mean=float(row["score_mean"]),
                score_median=float(row["score_median"]),
                score_p95=float(row["score_p95"]),
                low_risk_pct=float(row["low_risk_pct"]),
                moderate_risk_pct=float(row["moderate_risk_pct"]),
                high_risk_pct=float(row["high_risk_pct"]),
            )
        )
    lines.extend(
        [
            "",
            "## 五年综合层",
            "",
            f"- 共同有效像元：{multi['common_valid_pixels']:,}",
            f"- 五年平均风险分数中位数：{multi['mean_score_median']:.1f}",
            f"- 五年平均风险分数95分位：{multi['mean_score_p95']:.1f}",
            f"- 五年最大风险分数中位数：{multi['max_score_median']:.1f}",
            f"- 五年最大风险分数95分位：{multi['max_score_p95']:.1f}",
            "",
            "## 使用边界",
            "",
            "论文中应称其为 quality-risk index 或 uncertainty proxy。",
            "不要把它写成真正的模型预测标准差。真正的模型不确定性仍需要后续多随机种子/模型集成实验。",
        ]
    )
    (OUT / "README_uncertainty_proxy_layers.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def append_data_dictionary() -> None:
    path = METADATA / "data_dictionary.csv"
    add = pd.DataFrame(
        [
            {
                "file_group": "uncertainty_proxy_score",
                "field_or_band": "band 1",
                "meaning": "Composite quality-risk score based on province support, calibration strength, and bounded/unbounded sensitivity",
                "unit": "0-100 score",
                "nodata": "-9999",
                "notes": "Higher values indicate greater interpretation caution; this is not a true ensemble standard deviation.",
            },
            {
                "file_group": "uncertainty_proxy_class",
                "field_or_band": "band 1",
                "meaning": "Classified quality-risk score",
                "unit": "class code",
                "nodata": "255",
                "notes": "1=low risk (<25); 2=moderate risk (25-50); 3=high risk (>=50).",
            },
        ]
    )
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["file_group"].isin(add["file_group"])]
        pd.concat([old, add], ignore_index=True).to_csv(
            path, index=False, encoding="utf-8-sig"
        )
    else:
        add.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    global PACKAGE, INCOME, QUALITY, OUT, TABLES, METADATA
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=PACKAGE)
    args = parser.parse_args()
    PACKAGE = args.package_root.resolve()
    INCOME = PACKAGE / "data" / "income_maps"
    QUALITY = PACKAGE / "data" / "quality_layers"
    OUT = PACKAGE / "data" / "uncertainty_proxy_layers"
    TABLES = PACKAGE / "tables"
    METADATA = PACKAGE / "metadata"
    TABLES.mkdir(parents=True, exist_ok=True)
    METADATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for year in YEARS:
        print(f"Creating uncertainty proxy layers for {year} ...", flush=True)
        summaries.append(create_year(year))
    summary = pd.DataFrame(summaries)
    summary.to_csv(
        TABLES / "uncertainty_proxy_summary_2019_2023.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Creating 2019-2023 mean/max uncertainty proxy layers ...", flush=True)
    multi = create_multiyear_summary_layers()
    pd.DataFrame([multi]).to_csv(
        TABLES / "uncertainty_proxy_multiyear_summary_2019_2023.csv",
        index=False,
        encoding="utf-8-sig",
    )
    update_manifest()
    write_readme(summary, multi)
    append_data_dictionary()
    print("Uncertainty proxy layers written to:")
    print(OUT)
    print(summary[["year", "score_median", "score_p95", "high_risk_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
