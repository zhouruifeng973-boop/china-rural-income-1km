# -*- coding: utf-8 -*-
"""Audit the Scientific Data-style raster package for usability consistency.

This script checks:
1. raster spatial alignment: CRS, shape, transform, resolution;
2. NoData and value ranges;
3. filename convention;
4. manifest coverage;
5. README/data dictionary coverage.

It does not modify raster products.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEP_PATHS = [
    ROOT / ".codex_pydeps_stage8",
    ROOT / ".codex_pydeps",
    ROOT / "stage6_pydeps",
]
for path in reversed(LOCAL_DEP_PATHS):
    if path.exists():
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
import rasterio


PACKAGE = ROOT / "outputs" / "stage6_scientific_data_product"
DATA = PACKAGE / "data"
TABLES = PACKAGE / "tables"
METADATA = PACKAGE / "metadata"
AUDIT_CSV = TABLES / "final_data_product_raster_audit.csv"
AUDIT_MD = PACKAGE / "STAGE9_final_data_product_audit.md"


@dataclass
class RasterInfo:
    relpath: str
    group: str
    name: str
    width: int
    height: int
    count: int
    dtype: str
    crs: str
    transform: tuple[float, ...]
    res_x: float
    res_y: float
    nodata: float | None
    valid_pixels: int
    min_value: float | None
    max_value: float | None
    mean_value: float | None
    p01: float | None
    p50: float | None
    p99: float | None
    aligns_reference: bool
    nodata_status: str
    naming_status: str


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def tuple_transform(src: rasterio.DatasetReader) -> tuple[float, ...]:
    t = src.transform
    return (t.a, t.b, t.c, t.d, t.e, t.f, t.g, t.h, t.i)


def same_transform(a: tuple[float, ...], b: tuple[float, ...], tol: float = 1e-9) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def classify_expected_nodata(path: Path) -> str:
    name = path.name.lower()
    group = path.parent.name
    if "class" in name:
        return "255"
    if "support_flag" in name:
        return "255_or_-9999"
    if "mask" in name:
        return "255_or_-9999_or_0"
    if group == "quality_layers" and "absdiff" not in name and "factor" not in name:
        return "255_or_-9999_or_0"
    return "-9999"


def check_nodata(path: Path, nodata: float | None) -> str:
    expected = classify_expected_nodata(path)
    if nodata is None:
        return f"warn_no_nodata_declared_expected_{expected}"
    if expected == "-9999":
        return "ok" if math.isclose(float(nodata), -9999.0, abs_tol=1e-6) else f"warn_expected_-9999_got_{nodata}"
    if expected == "255":
        return "ok" if math.isclose(float(nodata), 255.0, abs_tol=1e-6) else f"warn_expected_255_got_{nodata}"
    allowed = []
    if "255" in expected:
        allowed.append(255.0)
    if "-9999" in expected:
        allowed.append(-9999.0)
    if "_0" in expected:
        allowed.append(0.0)
    return "ok" if any(math.isclose(float(nodata), x, abs_tol=1e-6) for x in allowed) else f"warn_expected_{expected}_got_{nodata}"


PATTERNS = {
    "income_maps": [
        re.compile(r"^china_rural_income_1km_(2019|2020|2021|2022|2023)_2019yuan_strict_unbounded\.tif$"),
    ],
    "change_maps": [
        re.compile(r"^china_rural_income_change_1km_2019_2023_(absolute_2019yuan|percent)_strict_unbounded\.tif$"),
    ],
    "sensitivity_bounded_maps": [
        re.compile(r"^china_rural_income_1km_(2019|2020|2021|2022|2023)_2019yuan_strict_bounded_3000_80000\.tif$"),
        re.compile(r"^china_rural_income_change_1km_2019_2023_(absolute_2019yuan|percent)_strict_bounded\.tif$"),
    ],
    "quality_layers": [
        re.compile(r"^quality_.+\.tif$"),
    ],
    "uncertainty_proxy_layers": [
        re.compile(r"^uncertainty_proxy_.+_strict\.tif$"),
    ],
    "ensemble_uncertainty_layers": [
        re.compile(r"^china_rural_income_1km_(2019|2020|2021|2022|2023)_ensemble_(mean_2019yuan|std_2019yuan|cv_percent)_strict_unbounded\.tif$"),
    ],
}


def check_naming(path: Path) -> str:
    group = path.parent.name
    patterns = PATTERNS.get(group, [])
    if any(p.match(path.name) for p in patterns):
        return "ok"
    return "warn_unexpected_name"


def read_stats(path: Path, ref_meta: dict | None) -> RasterInfo:
    group = path.parent.name
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
        valid = arr.compressed().astype(np.float64)
        transform = tuple_transform(src)
        crs = src.crs.to_string() if src.crs else ""
        if ref_meta is None:
            aligns = True
        else:
            aligns = (
                src.width == ref_meta["width"]
                and src.height == ref_meta["height"]
                and crs == ref_meta["crs"]
                and same_transform(transform, ref_meta["transform"])
            )
        if valid.size:
            min_value = float(valid.min())
            max_value = float(valid.max())
            mean_value = float(valid.mean())
            p01 = float(np.quantile(valid, 0.01))
            p50 = float(np.quantile(valid, 0.50))
            p99 = float(np.quantile(valid, 0.99))
        else:
            min_value = max_value = mean_value = p01 = p50 = p99 = None
        return RasterInfo(
            relpath=str(path.relative_to(PACKAGE)),
            group=group,
            name=path.name,
            width=src.width,
            height=src.height,
            count=src.count,
            dtype=src.dtypes[0],
            crs=crs,
            transform=transform,
            res_x=float(src.res[0]),
            res_y=float(src.res[1]),
            nodata=src.nodata,
            valid_pixels=int(valid.size),
            min_value=min_value,
            max_value=max_value,
            mean_value=mean_value,
            p01=p01,
            p50=p50,
            p99=p99,
            aligns_reference=aligns,
            nodata_status=check_nodata(path, src.nodata),
            naming_status=check_naming(path),
        )


def load_manifest_files() -> pd.DataFrame:
    frames = []
    for path in sorted(METADATA.glob("*manifest*.csv")):
        df = pd.read_csv(path)
        if "package_file" in df.columns:
            tmp = df[["package_file"]].copy()
            tmp["manifest"] = path.name
            frames.append(tmp)
    if not frames:
        return pd.DataFrame(columns=["package_file", "manifest"])
    out = pd.concat(frames, ignore_index=True)
    out["package_file_norm"] = out["package_file"].astype(str).str.replace("\\", "/", regex=False)
    return out


def audit_manifests(tif_paths: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    manifest = load_manifest_files()
    tif_rels = [str(p.relative_to(PACKAGE)).replace("\\", "/") for p in tif_paths]
    manifest_set = set(manifest["package_file_norm"].tolist())
    rows = []
    warnings = []
    for rel in tif_rels:
        covered = rel in manifest_set
        rows.append({"package_file": rel, "manifest_covered": covered})
        if not covered:
            warnings.append(rel)
    return pd.DataFrame(rows), warnings


def text_contains(path: Path, needles: list[str]) -> dict[str, bool]:
    if not path.exists():
        return {n: False for n in needles}
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    return {n: n.lower() in text for n in needles}


def audit_documentation() -> dict:
    data_dictionary = METADATA / "data_dictionary.csv"
    dd = pd.read_csv(data_dictionary) if data_dictionary.exists() else pd.DataFrame()
    if not dd.empty:
        dd_text = "\n".join(
            " ".join(str(v) for v in row) for row in dd.to_numpy()
        ).lower()
    else:
        dd_text = ""
    package_readme = PACKAGE / "README.md"
    docs = {
        "package_readme_exists": package_readme.exists(),
        "data_dictionary_exists": data_dictionary.exists(),
        "data_dictionary_mentions_income_maps": "income" in dd_text,
        "data_dictionary_mentions_quality_layers": "quality" in dd_text,
        "data_dictionary_mentions_uncertainty_proxy": "uncertainty" in dd_text,
        "data_dictionary_mentions_ensemble": "ensemble" in dd_text,
        "root_readme_mentions_bounded": False,
        "root_readme_mentions_unbounded": False,
        "root_readme_mentions_quality": False,
        "root_readme_mentions_ensemble": False,
        "quality_readme_exists": (DATA / "quality_layers" / "README_quality_layers.md").exists(),
        "uncertainty_proxy_readme_exists": (DATA / "uncertainty_proxy_layers" / "README_uncertainty_proxy_layers.md").exists(),
        "ensemble_readme_exists": (DATA / "ensemble_uncertainty_layers" / "README_ensemble_uncertainty_layers.md").exists(),
    }
    if package_readme.exists():
        text = package_readme.read_text(encoding="utf-8", errors="ignore").lower()
        docs["root_readme_mentions_bounded"] = "bounded" in text
        docs["root_readme_mentions_unbounded"] = "unbounded" in text
        docs["root_readme_mentions_quality"] = "quality" in text or "风险" in text
        docs["root_readme_mentions_ensemble"] = "ensemble" in text
    return docs


def write_csv(infos: list[RasterInfo], manifest_audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    manifest_map = dict(zip(manifest_audit["package_file"], manifest_audit["manifest_covered"]))
    for info in infos:
        d = info.__dict__.copy()
        d["package_file_norm"] = info.relpath.replace("\\", "/")
        d["manifest_covered"] = manifest_map.get(d["package_file_norm"], False)
        d.pop("transform", None)
        rows.append(d)
    df = pd.DataFrame(rows)
    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(AUDIT_CSV, index=False, encoding="utf-8-sig")
    return df


def fmt_bool(ok: bool) -> str:
    return "通过" if ok else "需处理"


def write_markdown(
    df: pd.DataFrame,
    manifest_missing: list[str],
    docs: dict,
    ref_relpath: str,
) -> None:
    total = len(df)
    groups = df.groupby("group").size().to_dict()
    align_ok = bool(df["aligns_reference"].all())
    docs_ok = all(bool(v) for v in docs.values())
    nodata_warn = df[~df["nodata_status"].eq("ok")]
    naming_warn = df[~df["naming_status"].eq("ok")]
    manifest_warn = df[~df["manifest_covered"]]

    lines = [
        "# Stage 9：最终数据产品可用性审查",
        "",
        "本报告检查 Scientific Data-style 数据包中 GeoTIFF 产品的空间一致性、NoData、值域、命名、manifest 覆盖和说明文件覆盖情况。本审查不修改任何 tif 主数据。",
        "",
        "## 1. 审查范围",
        "",
        f"- GeoTIFF 文件数：{total}",
        f"- 明细表：`tables/{AUDIT_CSV.name}`",
        "",
        "| group | tif_count |",
        "|---|---:|",
    ]
    for group, count in sorted(groups.items()):
        lines.append(f"| {group} | {count} |")

    lines.extend(
        [
            "",
            "## 2. 空间一致性",
            "",
            f"- CRS / shape / transform 与参考主图一致：{fmt_bool(align_ok)}",
        ]
    )
    ref_rows = df[df["relpath"].eq(ref_relpath)]
    ref = ref_rows.iloc[0] if len(ref_rows) else df.iloc[0]
    lines.extend(
        [
            f"- 参考文件：`{ref['relpath']}`",
            f"- CRS：`{ref['crs']}`",
            f"- 栅格尺寸：{int(ref['width'])} × {int(ref['height'])}",
            f"- 分辨率：{ref['res_x']} × {ref['res_y']}",
        ]
    )
    if not align_ok:
        bad = df[~df["aligns_reference"]]["relpath"].tolist()
        lines.append("")
        lines.append("未对齐文件：")
        for item in bad:
            lines.append(f"- `{item}`")

    lines.extend(
        [
            "",
            "## 3. NoData 检查",
            "",
            f"- NoData 状态异常/需人工确认文件数：{len(nodata_warn)}",
        ]
    )
    if len(nodata_warn):
        lines.append("")
        lines.append("| file | nodata | status |")
        lines.append("|---|---:|---|")
        for _, row in nodata_warn.iterrows():
            lines.append(f"| `{row['relpath']}` | {row['nodata']} | {row['nodata_status']} |")
    else:
        lines.append("- 连续型产品主要使用 -9999；分类/掩膜类产品使用 255、0 或 -9999 的设计均被识别为可接受。")

    lines.extend(
        [
            "",
            "## 4. 值域概览",
            "",
            "| group | min | median | p99 | max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for group, gdf in df.groupby("group"):
        lines.append(
            f"| {group} | {gdf['min_value'].min():.3f} | {gdf['p50'].median():.3f} | {gdf['p99'].median():.3f} | {gdf['max_value'].max():.3f} |"
        )

    lines.extend(
        [
            "",
            "## 5. 文件命名检查",
            "",
            f"- 命名不符合当前规则的 tif 数：{len(naming_warn)}",
        ]
    )
    if len(naming_warn):
        for item in naming_warn["relpath"].tolist():
            lines.append(f"- `{item}`")

    lines.extend(
        [
            "",
            "## 6. Manifest 覆盖检查",
            "",
            f"- 未被任一 manifest 覆盖的 tif 数：{len(manifest_warn)}",
        ]
    )
    if len(manifest_warn):
        for item in manifest_warn["relpath"].tolist():
            lines.append(f"- `{item}`")
    else:
        lines.append("- 所有 tif 均已被 `metadata/*manifest*.csv` 中至少一个清单覆盖。")

    lines.extend(
        [
            "",
            "## 7. README / Data Dictionary 覆盖检查",
            "",
            f"- 根 README 存在：{fmt_bool(bool(docs['package_readme_exists']))}",
            f"- data dictionary 存在：{fmt_bool(bool(docs['data_dictionary_exists']))}",
            f"- data dictionary 提到 income：{fmt_bool(bool(docs['data_dictionary_mentions_income_maps']))}",
            f"- data dictionary 提到 quality：{fmt_bool(bool(docs['data_dictionary_mentions_quality_layers']))}",
            f"- data dictionary 提到 uncertainty：{fmt_bool(bool(docs['data_dictionary_mentions_uncertainty_proxy']))}",
            f"- data dictionary 提到 ensemble：{fmt_bool(bool(docs['data_dictionary_mentions_ensemble']))}",
            f"- 根 README 提到 bounded：{fmt_bool(bool(docs['root_readme_mentions_bounded']))}",
            f"- 根 README 提到 unbounded：{fmt_bool(bool(docs['root_readme_mentions_unbounded']))}",
            f"- 根 README 提到 quality/risk：{fmt_bool(bool(docs['root_readme_mentions_quality']))}",
            f"- 根 README 提到 ensemble：{fmt_bool(bool(docs['root_readme_mentions_ensemble']))}",
            f"- quality layer README 存在：{fmt_bool(bool(docs['quality_readme_exists']))}",
            f"- uncertainty proxy README 存在：{fmt_bool(bool(docs['uncertainty_proxy_readme_exists']))}",
            f"- ensemble uncertainty README 存在：{fmt_bool(bool(docs['ensemble_readme_exists']))}",
            "",
            "## 8. 审查结论",
            "",
        ]
    )
    if align_ok and not len(nodata_warn) and not len(naming_warn) and not len(manifest_warn):
        lines.append("数据产品的栅格空间一致性、NoData、命名和 manifest 覆盖均通过自动审查。")
    else:
        lines.append("自动审查发现若干需要处理或人工确认的问题，详见上文对应章节。")
    lines.append("")
    if docs_ok:
        lines.append("README / data dictionary / 分项说明文件覆盖检查通过。当前数据包已经能区分主产品、变化产品、bounded 敏感性产品、quality-risk index 和 ensemble uncertainty 产品。")
    else:
        lines.append("建议后续优先处理 README/data dictionary 的文字覆盖问题，尤其是 Stage 8 新增的 ensemble uncertainty 与 Stage 7 quality-risk index 的关系说明，保证 Scientific Data 审稿人能清楚区分主产品、敏感性产品、质量风险产品和集成不确定性产品。")
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    tif_paths = sorted(DATA.rglob("*.tif"))
    if not tif_paths:
        raise FileNotFoundError(DATA)

    ref_path = DATA / "income_maps" / "china_rural_income_1km_2019_2019yuan_strict_unbounded.tif"
    if not ref_path.exists():
        ref_path = tif_paths[0]
    with rasterio.open(ref_path) as ref:
        ref_meta = {
            "width": ref.width,
            "height": ref.height,
            "crs": ref.crs.to_string() if ref.crs else "",
            "transform": tuple_transform(ref),
        }

    infos = [read_stats(path, ref_meta) for path in tif_paths]
    manifest_audit, manifest_missing = audit_manifests(tif_paths)
    df = write_csv(infos, manifest_audit)
    docs = audit_documentation()
    write_markdown(df, manifest_missing, docs, str(ref_path.relative_to(PACKAGE)))

    print(f"Audited {len(df)} tif files.")
    print(f"Alignment ok: {bool(df['aligns_reference'].all())}")
    print(f"NoData warnings: {int((~df['nodata_status'].eq('ok')).sum())}")
    print(f"Naming warnings: {int((~df['naming_status'].eq('ok')).sum())}")
    print(f"Manifest missing: {int((~df['manifest_covered']).sum())}")
    print(AUDIT_CSV)
    print(AUDIT_MD)


if __name__ == "__main__":
    main()
