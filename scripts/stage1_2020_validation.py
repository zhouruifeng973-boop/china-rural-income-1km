"""第一阶段：修复2020多波段预测并做县级回聚合一致性诊断。

输出均写入 outputs/stage1_2020，不覆盖既有成果。

该脚本属于早期历史实验，包含后来被v1.5核心流程删除的road_density，
不得用于生成或验证v1.5最终数据产品。
"""

from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


YEAR = 2020
ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "features"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "stage1_2020"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_PATH = OUTPUT_DIR / "prediction_2020_rf_fixed.tif"
REAGG_PATH = OUTPUT_DIR / "county_reaggregation_2020.csv"
AUDIT_PATH = OUTPUT_DIR / "feature_audit_2020.csv"
MULTIYEAR_AUDIT_PATH = OUTPUT_DIR / "multiyear_raster_audit.csv"
PROVINCE_PATH = OUTPUT_DIR / "province_coverage_2020.csv"
SUMMARY_PATH = OUTPUT_DIR / "stage1_summary.md"
RURAL_BUILT_THRESHOLDS = (0.50, 0.10, 0.02)

SCALE_0001 = {
    "ndvi_mean", "ndvi_max", "ndvi_std", "evi_mean", "npp",
    "B2", "B3", "B4", "B8",
}


def feature_groups(year):
    """返回 (路径, 波段名称列表)。名称按文件内波段顺序定义。"""
    return [
        (FEATURE_DIR / f"ntl_{year}_1km.tif", ["ntl"]),
        (FEATURE_DIR / f"vegetation_{year}_1km.tif",
         ["ndvi_mean", "ndvi_max", "ndvi_std", "evi_mean"]),
        (FEATURE_DIR / f"npp_{year}_1km.tif", ["npp"]),
        (FEATURE_DIR / f"landcover_{year}_1km.tif",
         ["tree_frac", "shrub_frac", "grass_frac", "crop_frac",
          "built_frac", "bare_frac", "water_frac", "wetland_frac"]),
        (FEATURE_DIR / "terrain_1km.tif", ["dem", "slope", "aspect"]),
        (FEATURE_DIR / f"climate_{year}_1km.tif",
         ["temp_mean", "precip_sum", "solar_mean"]),
        (FEATURE_DIR / f"sentinel2_{year}_summer_1km.tif",
         ["B2", "B3", "B4", "B8", "ndvi_s2", "ndwi", "ndbi"]),
        (FEATURE_DIR / "road_density_1km.tif", ["road_density"]),
    ]


def same_grid(src, ref):
    return (
        src.width == ref.width
        and src.height == ref.height
        and src.crs == ref.crs
        and src.transform.almost_equals(ref.transform)
    )


def audit_multiyear_rasters():
    rows = []
    with rasterio.open(FEATURE_DIR / "ntl_2020_1km.tif") as ref:
        for year in range(2019, 2025):
            for path, names in feature_groups(year):
                row = {
                    "year": year,
                    "file": path.name,
                    "expected_bands": len(names),
                    "expected_names": "|".join(names),
                    "exists": path.exists(),
                }
                if path.exists():
                    with rasterio.open(path) as src:
                        row.update({
                            "actual_bands": src.count,
                            "descriptions": "|".join(x or "" for x in src.descriptions),
                            "width": src.width,
                            "height": src.height,
                            "crs": str(src.crs),
                            "aligned": same_grid(src, ref),
                            "band_count_ok": src.count == len(names),
                        })
                rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(MULTIYEAR_AUDIT_PATH, index=False, encoding="utf-8-sig")
    bad = result[(~result["exists"]) | (~result["aligned"].fillna(False)) |
                 (~result["band_count_ok"].fillna(False))]
    if len(bad):
        raise RuntimeError(f"多年栅格审计发现 {len(bad)} 个缺失/错位/波段数错误，详见 {MULTIYEAR_AUDIT_PATH}")
    return result


def build_band_sources(year, requested_bands):
    """建立特征名 -> (文件, 1-based波段号, 缩放系数) 映射。"""
    lookup = {}
    for path, names in feature_groups(year):
        if not path.exists():
            continue
        with rasterio.open(path) as src:
            if src.count != len(names):
                raise ValueError(f"{path.name}: 实际{src.count}波段，预期{len(names)}波段")
        for band_index, name in enumerate(names, start=1):
            lookup[name] = (path, band_index, 0.0001 if name in SCALE_0001 else 1.0)

    missing = [name for name in requested_bands if name not in lookup]
    if missing:
        raise ValueError(f"缺少训练所需特征: {missing}")
    return {name: lookup[name] for name in requested_bands}


def iter_windows(width, height, size=512):
    for row in range(0, height, size):
        for col in range(0, width, size):
            yield Window(col, row, min(size, width - col), min(size, height - row))


def metrics(y_true, y_pred):
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[keep], y_pred[keep]
    return {
        "n": int(keep.sum()),
        "r2": float(r2_score(yt, yp)),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "bias": float(np.mean(yp - yt)),
    }


def main():
    print("[1/5] 审计2019—2024特征栅格...")
    multiyear_audit = audit_multiyear_rasters()
    print(f"  通过：{len(multiyear_audit)} 个年度-文件组合均存在且对齐")

    print("[2/5] 加载2020县级训练数据并训练RF...")
    data = np.load(ROOT / "outputs" / "features_uncorrelated_2020.npz", allow_pickle=True)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.float64)
    train_codes = data["codes"].astype(np.int64)
    bands = [str(x) for x in data["bands"]]

    clean = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y, train_codes = X[clean], y[clean], train_codes[clean]
    feature_medians = np.median(X, axis=0)
    scaler = StandardScaler().fit(X)
    model = RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_leaf=10,
        random_state=42, n_jobs=-1,
    )
    model.fit(scaler.transform(X), y)
    direct_train_prediction = model.predict(scaler.transform(X))
    train_fit = metrics(y, direct_train_prediction)
    print(f"  县={len(y)}, 特征={len(bands)}, 训练拟合R2={train_fit['r2']:.3f}")

    band_sources = build_band_sources(YEAR, bands)
    ref_path = FEATURE_DIR / f"ntl_{YEAR}_1km.tif"
    county_path = FEATURE_DIR / "county_id.tif"
    pop_path = DATA_DIR / "worldpop" / "chn_pop_2020_CN_1km_R2025A_UA_v1.tif"

    print("[3/5] 逐块生成修复后的2020全国预测...")
    max_code = 900000
    pixel_count = np.zeros(max_code, dtype=np.int64)
    pred_sum = np.zeros(max_code, dtype=np.float64)
    pop_sum = np.zeros(max_code, dtype=np.float64)
    pred_pop_sum = np.zeros(max_code, dtype=np.float64)
    rural_count = {t: np.zeros(max_code, dtype=np.int64) for t in RURAL_BUILT_THRESHOLDS}
    rural_pred_sum = {t: np.zeros(max_code, dtype=np.float64) for t in RURAL_BUILT_THRESHOLDS}
    rural_pop_sum = {t: np.zeros(max_code, dtype=np.float64) for t in RURAL_BUILT_THRESHOLDS}
    rural_pred_pop_sum = {t: np.zeros(max_code, dtype=np.float64) for t in RURAL_BUILT_THRESHOLDS}
    stat_min = np.full(len(bands), np.inf)
    stat_max = np.full(len(bands), -np.inf)
    stat_sum = np.zeros(len(bands), dtype=np.float64)
    stat_count = np.zeros(len(bands), dtype=np.int64)
    stat_imputed = np.zeros(len(bands), dtype=np.int64)

    with ExitStack() as stack:
        ref = stack.enter_context(rasterio.open(ref_path))
        county_src = stack.enter_context(rasterio.open(county_path))
        if not same_grid(county_src, ref):
            raise ValueError("county_id.tif 与参考栅格未对齐")

        datasets = {}
        for path, _, _ in band_sources.values():
            if path not in datasets:
                src = stack.enter_context(rasterio.open(path))
                if not same_grid(src, ref):
                    raise ValueError(f"{path.name} 与参考栅格未对齐")
                datasets[path] = src

        pop_src = stack.enter_context(rasterio.open(pop_path))
        pop_vrt = stack.enter_context(WarpedVRT(
            pop_src, crs=ref.crs, transform=ref.transform,
            width=ref.width, height=ref.height,
            src_nodata=pop_src.nodata, nodata=0,
            resampling=Resampling.bilinear,
        ))

        profile = ref.profile.copy()
        profile.update(
            dtype="float32", count=1, nodata=-9999.0, compress="lzw",
            predictor=3, tiled=True, blockxsize=256, blockysize=256,
            BIGTIFF="IF_SAFER",
        )
        dst = stack.enter_context(rasterio.open(PREDICTION_PATH, "w", **profile))
        dst.set_band_description(1, "predicted_rural_income_per_capita_2020")

        windows = list(iter_windows(ref.width, ref.height))
        for number, window in enumerate(windows, start=1):
            codes_2d = county_src.read(1, window=window).astype(np.int64)
            valid_2d = (codes_2d > 0) & (codes_2d < max_code)
            out = np.full(codes_2d.shape, -9999.0, dtype=np.float32)
            if valid_2d.any():
                n = int(valid_2d.sum())
                features = np.empty((n, len(bands)), dtype=np.float32)
                for j, name in enumerate(bands):
                    path, band_index, scale = band_sources[name]
                    src = datasets[path]
                    arr = src.read(band_index, window=window).astype(np.float32)
                    values = arr[valid_2d] * scale
                    bad = ~np.isfinite(values)
                    if src.nodata is not None:
                        bad |= values == (src.nodata * scale)
                    stat_imputed[j] += int(bad.sum())
                    if bad.any():
                        values[bad] = feature_medians[j]
                    features[:, j] = values
                    stat_min[j] = min(stat_min[j], float(values.min()))
                    stat_max[j] = max(stat_max[j], float(values.max()))
                    stat_sum[j] += float(values.sum(dtype=np.float64))
                    stat_count[j] += len(values)

                pred = model.predict(scaler.transform(features)).astype(np.float32)
                pred = np.clip(pred, 1000, 80000)
                out[valid_2d] = pred

                codes = codes_2d[valid_2d]
                pop = pop_vrt.read(1, window=window).astype(np.float64)[valid_2d]
                pop[~np.isfinite(pop) | (pop < 0)] = 0
                pixel_count += np.bincount(codes, minlength=max_code)
                pred_sum += np.bincount(codes, weights=pred, minlength=max_code)
                pop_sum += np.bincount(codes, weights=pop, minlength=max_code)
                pred_pop_sum += np.bincount(codes, weights=pred * pop, minlength=max_code)
                built = features[:, bands.index("built_frac")]
                for threshold in RURAL_BUILT_THRESHOLDS:
                    rural = built <= threshold
                    rural_codes = codes[rural]
                    rural_pred = pred[rural]
                    rural_pop = pop[rural]
                    rural_count[threshold] += np.bincount(rural_codes, minlength=max_code)
                    rural_pred_sum[threshold] += np.bincount(
                        rural_codes, weights=rural_pred, minlength=max_code)
                    rural_pop_sum[threshold] += np.bincount(
                        rural_codes, weights=rural_pop, minlength=max_code)
                    rural_pred_pop_sum[threshold] += np.bincount(
                        rural_codes, weights=rural_pred * rural_pop, minlength=max_code)
            dst.write(out, 1, window=window)
            if number % 20 == 0 or number == len(windows):
                print(f"  {number}/{len(windows)} windows", flush=True)

    print("[4/5] 生成县级回聚合结果...")
    codes = np.flatnonzero(pixel_count > 0)
    result = pd.DataFrame({
        "county_code": codes,
        "pixel_count": pixel_count[codes],
        "population_weight": pop_sum[codes],
        "pred_area_mean": pred_sum[codes] / pixel_count[codes],
        "pred_population_weighted": np.divide(
            pred_pop_sum[codes], pop_sum[codes],
            out=np.full(len(codes), np.nan), where=pop_sum[codes] > 0,
        ),
    })
    for threshold in RURAL_BUILT_THRESHOLDS:
        suffix = str(threshold).replace(".", "p")
        result[f"pixel_count_built_le_{suffix}"] = rural_count[threshold][codes]
        result[f"pred_area_mean_built_le_{suffix}"] = np.divide(
            rural_pred_sum[threshold][codes], rural_count[threshold][codes],
            out=np.full(len(codes), np.nan), where=rural_count[threshold][codes] > 0,
        )
        result[f"pred_population_weighted_built_le_{suffix}"] = np.divide(
            rural_pred_pop_sum[threshold][codes], rural_pop_sum[threshold][codes],
            out=np.full(len(codes), np.nan), where=rural_pop_sum[threshold][codes] > 0,
        )

    income = pd.read_csv(DATA_DIR / "county_rural_income_2020.csv")
    income["county_code"] = income["county_code"].astype(np.int64)
    income_unique = income.drop_duplicates("county_code", keep="last").copy()
    keep_cols = ["county_code", "province", "city", "county", "income", "source"]
    result = result.merge(income_unique[keep_cols], on="county_code", how="left")
    direct = pd.DataFrame({
        "county_code": train_codes,
        "pred_county_feature_direct": direct_train_prediction,
        "in_training": True,
    })
    result = result.merge(direct, on="county_code", how="left")
    result["in_training"] = result["in_training"].eq(True)
    mapping = pd.read_csv(FEATURE_DIR / "county_code_mapping.csv")
    mapping = mapping.rename(columns={"县代码": "county_code", "省": "map_province",
                                      "市": "map_city", "县": "map_county"})
    mapping["county_code"] = mapping["county_code"].astype(np.int64)
    mapping = mapping.drop_duplicates("county_code", keep="first")
    result = result.merge(mapping, on="county_code", how="left")
    result.to_csv(REAGG_PATH, index=False, encoding="utf-8-sig")

    observed = result[result["income"].notna() & result["in_training"]].copy()
    direct_metrics = metrics(observed["income"].to_numpy(), observed["pred_county_feature_direct"].to_numpy())
    area_metrics = metrics(observed["income"].to_numpy(), observed["pred_area_mean"].to_numpy())
    pop_metrics = metrics(observed["income"].to_numpy(), observed["pred_population_weighted"].to_numpy())
    rural_metrics = []
    for threshold in RURAL_BUILT_THRESHOLDS:
        suffix = str(threshold).replace(".", "p")
        rural_metrics.append((
            threshold,
            metrics(observed["income"].to_numpy(),
                    observed[f"pred_area_mean_built_le_{suffix}"].to_numpy()),
            metrics(observed["income"].to_numpy(),
                    observed[f"pred_population_weighted_built_le_{suffix}"].to_numpy()),
        ))

    province = income.groupby("province", as_index=False).agg(
        label_rows=("county_code", "size"),
        label_unique_counties=("county_code", "nunique"),
    )
    matched = observed.groupby("province", as_index=False).agg(
        raster_matched_counties=("county_code", "nunique")
    )
    province = province.merge(matched, on="province", how="left").fillna(0)
    province.to_csv(PROVINCE_PATH, index=False, encoding="utf-8-sig")

    audit = pd.DataFrame({
        "band": bands,
        "source_file": [band_sources[b][0].name for b in bands],
        "source_band_index": [band_sources[b][1] for b in bands],
        "scale": [band_sources[b][2] for b in bands],
        "national_min": stat_min,
        "national_mean": np.divide(stat_sum, stat_count, where=stat_count > 0),
        "national_max": stat_max,
        "imputed_pixel_count": stat_imputed,
    })
    audit.to_csv(AUDIT_PATH, index=False, encoding="utf-8-sig")

    print("[5/5] 写入阶段报告...")
    summary = f"""# 第一阶段：2020年预测修复与回聚合诊断

## 数据审计

- 2019—2024年年度—文件组合：{len(multiyear_audit)}个，全部存在、波段数正确且与参考网格对齐。
- 参考网格：6815 × 3942，EPSG:4326。
- 2020年收入标签：{len(income)}行、{income_unique['county_code'].nunique()}个唯一代码，{income['province'].nunique()}个省级地区。
- 成功进入县级训练数据：{len(y)}个县，{len(bands)}个去共线特征。
- WorldPop与特征网格不完全一致；验证时已按参考网格动态重投影。

## 修复内容

- 多波段GeoTIFF按真实波段序号读取，不再把第1波段复制给多个特征。
- 全国预测采用512×512窗口逐块运行，避免一次性堆叠全国21维栅格。
- 非有限特征值使用训练县级中位数填补，不再统一填0。
- 原有预测文件未覆盖；修复结果：`{PREDICTION_PATH.relative_to(ROOT)}`。

## 县级回聚合一致性诊断

> 以下是全量模型的回聚合一致性检查，不是独立交叉验证精度。

| 聚合方式 | 县数 | R2 | MAE（元） | RMSE（元） | Bias（元） |
|---|---:|---:|---:|---:|---:|
| 县级特征直接拟合 | {direct_metrics['n']} | {direct_metrics['r2']:.4f} | {direct_metrics['mae']:.0f} | {direct_metrics['rmse']:.0f} | {direct_metrics['bias']:.0f} |
| 像元面积均值 | {area_metrics['n']} | {area_metrics['r2']:.4f} | {area_metrics['mae']:.0f} | {area_metrics['rmse']:.0f} | {area_metrics['bias']:.0f} |
| WorldPop人口加权 | {pop_metrics['n']} | {pop_metrics['r2']:.4f} | {pop_metrics['mae']:.0f} | {pop_metrics['rmse']:.0f} | {pop_metrics['bias']:.0f} |
{chr(10).join(f"| 建成比例≤{t:.2f}：面积均值 | {am['n']} | {am['r2']:.4f} | {am['mae']:.0f} | {am['rmse']:.0f} | {am['bias']:.0f} |{chr(10)}| 建成比例≤{t:.2f}：人口加权 | {pm['n']} | {pm['r2']:.4f} | {pm['mae']:.0f} | {pm['rmse']:.0f} | {pm['bias']:.0f} |" for t, am, pm in rural_metrics)}

人口加权使用总人口作为当前代理权重；它不是农村人口专用权重，后续仍需结合农村掩膜完善。
建成比例阈值仅用于诊断城市人口权重影响，不在本阶段替代正式农村掩膜。

## 阶段判读

- 县级特征直接拟合到像元面积回聚合，R2由{direct_metrics['r2']:.3f}降至{area_metrics['r2']:.3f}，说明县级关系直接迁移到像元后存在尺度不一致。
- 未加农村约束的总人口加权结果不可用（R2={pop_metrics['r2']:.3f}）；高密度城市人口把预测系统性抬高，平均偏差为{pop_metrics['bias']:.0f}元。
- 更严格地排除高建成比例像元后，人口加权结果逐步改善，但仍不优于面积均值，因此不能用总人口权重替代农村人口权重。
- NPP在全国县域有效像元中有{stat_imputed[bands.index('npp')]:,}个值需要填补，占{100 * stat_imputed[bands.index('npp')] / stat_count[bands.index('npp')]:.1f}%；这是全国外推需重点标记的不确定性来源。
- 本阶段完成了技术修复，但回聚合诊断尚未支持把该TIF作为最终论文产品；下一步应先构建正式农村人口/土地掩膜并处理尺度一致性。

## 输出

- `{PREDICTION_PATH.relative_to(ROOT)}`
- `{REAGG_PATH.relative_to(ROOT)}`
- `{AUDIT_PATH.relative_to(ROOT)}`
- `{MULTIYEAR_AUDIT_PATH.relative_to(ROOT)}`
- `{PROVINCE_PATH.relative_to(ROOT)}`
"""
    SUMMARY_PATH.write_text(summary, encoding="utf-8")

    print("\n完成")
    print(f"  预测图: {PREDICTION_PATH}")
    print(f"  面积回聚合: R2={area_metrics['r2']:.4f}, MAE={area_metrics['mae']:.0f}")
    print(f"  人口回聚合: R2={pop_metrics['r2']:.4f}, MAE={pop_metrics['mae']:.0f}")
    print(f"  报告: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
