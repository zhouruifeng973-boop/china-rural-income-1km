# -*- coding: utf-8 -*-
"""
Stage 2 data preprocessing.

Outputs:
  data/processed/province_income_2019_2023_deflated.csv
  data/processed/cpi_2019_2023_panel.csv
  data/processed/county_income_2019_2023_deflated.csv
  data/processed/worldpop_aligned/chn_pop_<year>_1km_aligned.tif
  data/processed/rural_weights/rural_pop_weight_<year>_<strict|broad>.tif
  data/processed/county_rural_weight_summary_2019_2023.csv
  outputs/stage2_preprocessed/preprocess_summary.md
"""

from __future__ import annotations

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FEATURES = ROOT / "features"
PROCESSED = DATA / "processed"
OUT = ROOT / "outputs" / "stage2_preprocessed"
YEARS = [2019, 2020, 2021, 2022, 2023]


PROVINCE_STD = {
    "北京": "北京市",
    "北京市": "北京市",
    "天津": "天津市",
    "天津市": "天津市",
    "河北": "河北省",
    "河北省": "河北省",
    "山西": "山西省",
    "山西省": "山西省",
    "内蒙古": "内蒙古自治区",
    "内蒙古自治区": "内蒙古自治区",
    "辽宁": "辽宁省",
    "辽宁省": "辽宁省",
    "吉林": "吉林省",
    "吉林省": "吉林省",
    "黑龙江": "黑龙江省",
    "黑龙江省": "黑龙江省",
    "上海": "上海市",
    "上海市": "上海市",
    "江苏": "江苏省",
    "江苏省": "江苏省",
    "浙江": "浙江省",
    "浙江省": "浙江省",
    "安徽": "安徽省",
    "安徽省": "安徽省",
    "福建": "福建省",
    "福建省": "福建省",
    "江西": "江西省",
    "江西省": "江西省",
    "山东": "山东省",
    "山东省": "山东省",
    "河南": "河南省",
    "河南省": "河南省",
    "湖北": "湖北省",
    "湖北省": "湖北省",
    "湖南": "湖南省",
    "湖南省": "湖南省",
    "广东": "广东省",
    "广东省": "广东省",
    "广西": "广西壮族自治区",
    "广西壮族自治区": "广西壮族自治区",
    "海南": "海南省",
    "海南省": "海南省",
    "重庆": "重庆市",
    "重庆市": "重庆市",
    "四川": "四川省",
    "四川省": "四川省",
    "贵州": "贵州省",
    "贵州省": "贵州省",
    "云南": "云南省",
    "云南省": "云南省",
    "西藏": "西藏自治区",
    "西藏自治区": "西藏自治区",
    "陕西": "陕西省",
    "陕西省": "陕西省",
    "甘肃": "甘肃省",
    "甘肃省": "甘肃省",
    "青海": "青海省",
    "青海省": "青海省",
    "宁夏": "宁夏回族自治区",
    "宁夏回族自治区": "宁夏回族自治区",
    "新疆": "新疆维吾尔自治区",
    "新疆维吾尔自治区": "新疆维吾尔自治区",
}


def standardize_province(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().replace(" ", "")
    return PROVINCE_STD.get(text, text)


def find_data_file(keyword: str, suffixes: tuple[str, ...]) -> Path:
    matches = [
        p
        for p in DATA.iterdir()
        if p.is_file() and keyword in p.name and p.suffix.lower() in suffixes
    ]
    if not matches:
        raise FileNotFoundError(f"Cannot find data file containing {keyword!r}")
    if len(matches) > 1:
        matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def load_province_income() -> pd.DataFrame:
    path = find_data_file("2005-2024", (".xlsx",))
    df = pd.read_excel(path)
    year_col = df.columns[0]
    code_col = df.columns[1]
    province_col = df.columns[2]
    rural_income_col = df.columns[5]
    rural_expense_col = df.columns[7]
    urban_income_col = df.columns[4]
    disposable_income_col = df.columns[3]

    panel = pd.DataFrame(
        {
            "year": pd.to_numeric(df[year_col], errors="coerce"),
            "province_code": df[code_col],
            "province_raw": df[province_col],
            "income_per_capita_nominal": pd.to_numeric(
                df[disposable_income_col], errors="coerce"
            ),
            "urban_income_nominal": pd.to_numeric(df[urban_income_col], errors="coerce"),
            "rural_income_nominal": pd.to_numeric(df[rural_income_col], errors="coerce"),
            "rural_expense_nominal": pd.to_numeric(df[rural_expense_col], errors="coerce"),
        }
    )
    panel = panel[panel["year"].isin(YEARS)].copy()
    panel["year"] = panel["year"].astype(int)
    panel["province_std"] = panel["province_raw"].map(standardize_province)
    panel["source_file"] = path.name
    return panel


def load_cpi_panel() -> pd.DataFrame:
    path = find_data_file("居民消费价格指数", (".xls", ".xlsx"))

    # The second sheet is the tidy panel sheet in the current file.
    cpi = pd.read_excel(path, sheet_name=1)
    prov_col = cpi.columns[0]
    year_col = cpi.columns[1]
    value_col = cpi.columns[2]

    panel = pd.DataFrame(
        {
            "province_raw": cpi[prov_col],
            "year": cpi[year_col].astype(str).str.extract(r"(\d{4})")[0],
            "cpi_yoy": pd.to_numeric(cpi[value_col], errors="coerce"),
        }
    )
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce")
    panel = panel[panel["year"].isin(YEARS)].copy()
    panel["year"] = panel["year"].astype(int)
    panel["province_std"] = panel["province_raw"].map(standardize_province)

    rows = []
    for province, g in panel.groupby("province_std"):
        yoy = dict(zip(g["year"], g["cpi_yoy"]))
        price_index = 1.0
        for year in YEARS:
            if year == 2019:
                price_index = 1.0
            else:
                price_index *= yoy[year] / 100.0
            rows.append(
                {
                    "province_std": province,
                    "year": year,
                    "cpi_yoy": yoy[year],
                    "price_index_2019_base": price_index,
                    "source_file": path.name,
                }
            )
    out = pd.DataFrame(rows)
    return out


def build_deflated_income_tables() -> dict:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    province_income = load_province_income()
    cpi_panel = load_cpi_panel()

    province_deflated = province_income.merge(
        cpi_panel[["province_std", "year", "cpi_yoy", "price_index_2019_base"]],
        on=["province_std", "year"],
        how="left",
    )
    for col in [
        "income_per_capita_nominal",
        "urban_income_nominal",
        "rural_income_nominal",
        "rural_expense_nominal",
    ]:
        province_deflated[col.replace("_nominal", "_2019yuan")] = (
            province_deflated[col] / province_deflated["price_index_2019_base"]
        )

    county_rows = []
    for year in YEARS:
        path = DATA / f"county_rural_income_{year}.csv"
        df = pd.read_csv(path)
        df["year"] = year
        df["county_code"] = df["county_code"].astype(int)
        df["province_std"] = df["province"].map(standardize_province)
        df = df.rename(columns={"income": "rural_income_nominal"})
        county_rows.append(df)
    county_panel = pd.concat(county_rows, ignore_index=True)
    county_deflated = county_panel.merge(
        cpi_panel[["province_std", "year", "cpi_yoy", "price_index_2019_base"]],
        on=["province_std", "year"],
        how="left",
    )
    county_deflated["rural_income_2019yuan"] = (
        county_deflated["rural_income_nominal"]
        / county_deflated["price_index_2019_base"]
    )

    cpi_panel.to_csv(PROCESSED / "cpi_2019_2023_panel.csv", index=False, encoding="utf-8-sig")
    province_deflated.to_csv(
        PROCESSED / "province_income_2019_2023_deflated.csv",
        index=False,
        encoding="utf-8-sig",
    )
    county_deflated.to_csv(
        PROCESSED / "county_income_2019_2023_deflated.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "province_rows": int(len(province_deflated)),
        "province_count": int(province_deflated["province_std"].nunique()),
        "province_missing_cpi": int(province_deflated["price_index_2019_base"].isna().sum()),
        "county_rows": int(len(county_deflated)),
        "county_count_union": int(county_deflated["county_code"].nunique()),
        "county_missing_cpi": int(county_deflated["price_index_2019_base"].isna().sum()),
    }


def reference_profile() -> dict:
    with rasterio.open(FEATURES / "county_id.tif") as ref:
        profile = ref.profile.copy()
        transform = ref.transform
        crs = ref.crs
        height, width = ref.height, ref.width
    return {
        "profile": profile,
        "transform": transform,
        "crs": crs,
        "height": height,
        "width": width,
    }


def align_worldpop(year: int, ref: dict) -> Path:
    out_dir = PROCESSED / "worldpop_aligned"
    out_dir.mkdir(parents=True, exist_ok=True)
    matches = sorted((DATA / "worldpop").glob(f"*{year}*.tif"))
    if not matches:
        raise FileNotFoundError(f"Missing WorldPop raster for {year}")
    src_path = matches[0]
    out_path = out_dir / f"chn_pop_{year}_1km_aligned.tif"

    profile = ref["profile"].copy()
    profile.update(
        dtype="float32",
        count=1,
        compress="lzw",
        nodata=0.0,
        driver="GTiff",
    )

    dst = np.zeros((ref["height"], ref["width"]), dtype=np.float32)
    with rasterio.open(src_path) as src:
        resampling = Resampling.sum if hasattr(Resampling, "sum") else Resampling.bilinear
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref["transform"],
            dst_crs=ref["crs"],
            dst_nodata=0.0,
            resampling=resampling,
        )

    dst[~np.isfinite(dst)] = 0
    dst[dst < 0] = 0
    with rasterio.open(out_path, "w", **profile) as dst_src:
        dst_src.write(dst, 1)
    return out_path


def make_rural_weights(ref: dict) -> dict:
    weight_dir = PROCESSED / "rural_weights"
    weight_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(DATA / "ghs_smod" / "rural_mask_2020_strict_1km_aligned.tif") as src:
        strict_mask = src.read(1, masked=False) > 0
    with rasterio.open(DATA / "ghs_smod" / "rural_mask_2020_broad_1km_aligned.tif") as src:
        broad_mask = src.read(1, masked=False) > 0
    with rasterio.open(FEATURES / "county_id.tif") as src:
        county_id = src.read(1).astype(np.int64)

    profile = ref["profile"].copy()
    profile.update(dtype="float32", count=1, compress="lzw", nodata=0.0, driver="GTiff")

    stats = []
    county_rows = []
    valid_county = county_id > 0
    max_code = int(county_id[valid_county].max())
    county_codes = np.unique(county_id[valid_county])

    for year in YEARS:
        pop_path = align_worldpop(year, ref)
        with rasterio.open(pop_path) as src:
            pop = src.read(1).astype(np.float32)
        pop[~np.isfinite(pop)] = 0
        pop[pop < 0] = 0

        for mask_name, mask in [("strict", strict_mask), ("broad", broad_mask)]:
            weight = np.where(mask & valid_county, pop, 0).astype(np.float32)
            out_path = weight_dir / f"rural_pop_weight_{year}_{mask_name}.tif"
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(weight, 1)

            weight_sum = np.bincount(
                county_id[valid_county].ravel(),
                weights=weight[valid_county].ravel(),
                minlength=max_code + 1,
            )
            pixel_count = np.bincount(
                county_id[valid_county].ravel(),
                weights=(mask & valid_county)[valid_county].astype(np.float64).ravel(),
                minlength=max_code + 1,
            )
            for code in county_codes:
                county_rows.append(
                    {
                        "year": year,
                        "mask": mask_name,
                        "county_code": int(code),
                        "rural_pop_weight_sum": float(weight_sum[int(code)]),
                        "rural_pixel_count": int(pixel_count[int(code)]),
                    }
                )
            stats.append(
                {
                    "year": year,
                    "mask": mask_name,
                    "aligned_population_total": float(pop[valid_county].sum()),
                    "rural_weight_total": float(weight.sum()),
                    "rural_pixel_count": int((mask & valid_county).sum()),
                    "output": str(out_path.relative_to(ROOT)),
                }
            )

    pd.DataFrame(county_rows).to_csv(
        PROCESSED / "county_rural_weight_summary_2019_2023.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {"raster_stats": stats, "county_weight_rows": int(len(county_rows))}


def grid_check(path: Path, ref: dict) -> bool:
    with rasterio.open(path) as src:
        return (
            src.width == ref["width"]
            and src.height == ref["height"]
            and src.crs == ref["crs"]
            and src.transform == ref["transform"]
        )


def write_summary(income_stats: dict, weight_stats: dict, ref: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raster_stats = pd.DataFrame(weight_stats["raster_stats"])
    raster_stats.to_csv(OUT / "rural_weight_raster_stats.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "years": YEARS,
        "reference_grid": {
            "width": ref["width"],
            "height": ref["height"],
            "crs": str(ref["crs"]),
            "transform": [round(v, 12) for v in tuple(ref["transform"])[:6]],
        },
        "income_stats": income_stats,
        "weight_stats": weight_stats,
    }
    (OUT / "stage2_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Stage 2 数据预处理汇总",
        "",
        "## 已生成的核心数据",
        "",
        "- `data/processed/cpi_2019_2023_panel.csv`",
        "- `data/processed/province_income_2019_2023_deflated.csv`",
        "- `data/processed/county_income_2019_2023_deflated.csv`",
        "- `data/processed/worldpop_aligned/chn_pop_2019_1km_aligned.tif` 等 5 个 WorldPop 对齐栅格",
        "- `data/processed/rural_weights/rural_pop_weight_<year>_<strict|broad>.tif` 共 10 个农村人口权重栅格",
        "- `data/processed/county_rural_weight_summary_2019_2023.csv`",
        "",
        "## 收入和 CPI",
        "",
        f"- 省级收入记录：{income_stats['province_rows']} 行，{income_stats['province_count']} 个省级地区。",
        f"- 县级收入记录：{income_stats['county_rows']} 行，县代码并集 {income_stats['county_count_union']} 个。",
        f"- 省级收入缺失 CPI 匹配：{income_stats['province_missing_cpi']} 行。",
        f"- 县级收入缺失 CPI 匹配：{income_stats['county_missing_cpi']} 行。",
        "- 不变价口径：以 2019 年为基期，2020—2023 年按省级 CPI 链式折算。",
        "",
        "## 农村人口权重",
        "",
        "- strict 权重：WorldPop × GHS-SMOD 2020 strict rural mask。",
        "- broad 权重：WorldPop × GHS-SMOD 2020 broad rural mask，用于敏感性分析。",
        "- WorldPop 已重投影/重采样到项目主网格，和 `features/county_id.tif` 一致。",
        "",
        "## 注意",
        "",
        "- GHS-SMOD 是城乡类型代理变量，不等同于国家统计局城乡划分；论文中应表述为农村空间约束或敏感性掩膜。",
        "- WorldPop 是人口总量栅格，经过 rural mask 后作为农村人口权重代理。",
    ]
    (OUT / "preprocess_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    print("[1/4] Building deflated income tables...")
    income_stats = build_deflated_income_tables()

    print("[2/4] Preparing reference grid...")
    ref = reference_profile()

    print("[3/4] Aligning WorldPop and creating rural population weights...")
    weight_stats = make_rural_weights(ref)

    print("[4/4] Writing summary...")
    write_summary(income_stats, weight_stats, ref)

    print("Done.")
    print(f"Summary: {OUT / 'preprocess_summary.md'}")


if __name__ == "__main__":
    main()
