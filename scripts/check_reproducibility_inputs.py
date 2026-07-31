"""Check input availability for the v1.0-draft reproducibility workflow.

This script is a Stage 0 preflight check. It does not run models and does not
modify raster products. It only checks whether expected inputs, intermediate
files, final package files, and documentation artifacts are present.

Outputs:
    reproducibility/input_check_report_v1.0_draft.csv
    reproducibility/input_check_report_v1.0_draft.md

Default behavior is non-strict: missing files are reported but the script exits
with code 0. Use --strict to fail when files required for full local
reproduction are missing.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

from repro_config import deep_get, int_list, load_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "workflow_v1.0_draft.yaml"
REPORT_DIR = ROOT / "reproducibility"
CSV_REPORT = REPORT_DIR / "input_check_report_v1.0_draft.csv"
MD_REPORT = REPORT_DIR / "input_check_report_v1.0_draft.md"


@dataclass(frozen=True)
class InputSpec:
    group: str
    path: str
    check_type: str
    required_for: str
    access_class: str
    redistribution_status: str
    role: str
    expected_count: int = 1
    note: str = ""


@dataclass
class CheckResult:
    group: str
    path: str
    check_type: str
    required_for: str
    access_class: str
    redistribution_status: str
    role: str
    expected_count: int
    found_count: int
    status: str
    note: str


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def add_yearly(
    specs: list[InputSpec],
    *,
    group: str,
    template: str,
    years: Iterable[int],
    required_for: str,
    access_class: str,
    redistribution_status: str,
    role: str,
    note: str = "",
) -> None:
    for year in years:
        specs.append(
            InputSpec(
                group=group,
                path=template.format(year=year),
                check_type="file",
                required_for=required_for,
                access_class=access_class,
                redistribution_status=redistribution_status,
                role=role,
                note=note,
            )
        )


def build_specs(cfg: dict) -> list[InputSpec]:
    years = int_list(deep_get(cfg, "project.study_years"), [2019, 2020, 2021, 2022, 2023])
    specs: list[InputSpec] = []

    for directory, role in [
        ("data", "Local input data directory."),
        ("features", "Aligned remote-sensing and spatial feature directory."),
        ("config", "Workflow configuration directory."),
        ("scripts", "Reproducible code directory."),
        ("outputs/stage6_scientific_data_product", "Frozen v1.0-draft data product directory."),
    ]:
        specs.append(
            InputSpec(
                "project_structure",
                directory,
                "dir",
                "full_and_partial_reproduction",
                "project_workspace",
                "not_a_data_source",
                role,
            )
        )

    add_yearly(
        specs,
        group="county_income_labels_private",
        template="data/county_rural_income_{year}.csv",
        years=years,
        required_for="full_reproduction_only",
        access_class="author_compiled_provenance_pending",
        redistribution_status="do_not_redistribute_without_rights_review",
        role="County-level weak-supervision labels for model training and validation.",
        note="Author-compiled from provincial statistical yearbooks; keep outside the current public release until the province-year source crosswalk is complete.",
    )
    specs.append(
        InputSpec(
            "county_income_labels_private",
            "data/county_rural_income_all.csv",
            "file",
            "full_reproduction_only",
            "author_compiled_provenance_pending",
            "do_not_redistribute_without_rights_review",
            "Merged county-level income labels.",
        )
    )

    for path in [
        "data/2020xianBJ/县.shp",
        "data/2020xianBJ/县.dbf",
        "data/2020xianBJ/县.shx",
        "data/2020xianBJ/县.prj",
    ]:
        specs.append(
            InputSpec(
                "county_boundary_private",
                path,
                "file",
                "full_reproduction_only",
                "purchased_boundary_data",
                "do_not_redistribute_without_rights_review",
                "County boundary source used for county IDs and aggregation geometry.",
                note="Temporarily excluded from public release by project policy.",
            )
        )

    add_yearly(
        specs,
        group="worldpop_raw_public",
        template="data/worldpop/chn_pop_{year}_CN_1km_R2025A_UA_v1.tif",
        years=years,
        required_for="full_reproduction",
        access_class="public_raw_data",
        redistribution_status="redistributable_with_attribution_cc_by_4_0",
        role="WorldPop population counts for rural population weights.",
    )
    for path, role in [
        ("data/ghs_smod/ghs_smod_2020_1km_aligned.tif", "GHS-SMOD urban-rural classification aligned to project grid."),
        ("data/ghs_smod/rural_mask_2020_strict_1km_aligned.tif", "Strict rural mask derived from GHS-SMOD."),
        ("data/ghs_smod/rural_mask_2020_broad_1km_aligned.tif", "Broad rural mask derived from GHS-SMOD sensitivity definition."),
    ]:
        specs.append(
            InputSpec(
                "ghs_smod_rural_definition",
                path,
                "file",
                "full_reproduction",
                "public_or_public_derived_data",
                "redistribution_terms_follow_source_license",
                role,
            )
        )

    for path, role in [
        ("data/2005-2024年居民人均可支配收入与消费支出.xlsx", "National Bureau of Statistics provincial rural income source workbook."),
        ("data/居民消费价格指数2000-2024.xls", "National Bureau of Statistics CPI source workbook for deflation."),
    ]:
        specs.append(
            InputSpec(
                "nbs_raw_workbooks",
                path,
                "file",
                "full_reproduction",
                "public_downloaded_statistical_workbook",
                "exclude_raw_workbook_from_public_package_use_processed_tables",
                role,
                note="Raw NBS workbooks are useful for full local reproduction; public package should prefer processed/cited tables.",
            )
        )

    specs.append(
        InputSpec(
            "osm_raw_or_derived",
            "data/osm/china-latest.osm.pbf",
            "file",
            "full_reproduction_optional_if_road_density_exists",
            "openstreetmap_raw_data",
            "do_not_redistribute_raw_pbf_without_odbl_review",
            "Raw OSM PBF used to derive road density.",
        )
    )
    specs.append(
        InputSpec(
            "osm_raw_or_derived",
            "features/road_density_1km.tif",
            "file",
            "full_reproduction",
            "derived_spatial_feature",
            "review_before_public_release",
            "Road density feature derived from OSM.",
        )
    )

    for feature in ["climate", "landcover", "npp", "ntl", "vegetation"]:
        add_yearly(
            specs,
            group="annual_feature_rasters",
            template=f"features/{feature}_{{year}}_1km.tif",
            years=years,
            required_for="full_reproduction",
            access_class="public_derived_feature",
            redistribution_status="redistribution_terms_follow_source_license",
            role=f"Annual {feature} feature raster aligned to project grid.",
        )
    add_yearly(
        specs,
        group="annual_feature_rasters",
        template="features/sentinel2_{year}_summer_1km.tif",
        years=years,
        required_for="full_reproduction",
        access_class="public_derived_feature",
        redistribution_status="redistribution_terms_follow_source_license",
        role="Annual summer Sentinel-2 composite feature raster aligned to project grid.",
    )
    for path, role in [
        ("features/terrain_1km.tif", "Static terrain feature raster."),
        ("features/county_id.tif", "County ID raster for pixel-to-county aggregation."),
        ("features/county_code_mapping.csv", "County ID/code mapping table."),
    ]:
        specs.append(
            InputSpec(
                "static_feature_rasters",
                path,
                "file",
                "full_reproduction",
                "derived_spatial_feature",
                "review_before_public_release",
                role,
            )
        )

    for path, role in [
        ("data/processed/county_feature_panel_2019_2023.csv", "County-year feature panel."),
        ("data/processed/county_income_2019_2023_deflated.csv", "County income deflated to 2019 yuan."),
        ("data/processed/province_income_2019_2023_deflated.csv", "Province rural income calibration constraints deflated to 2019 yuan."),
        ("data/processed/cpi_2019_2023_panel.csv", "Province CPI panel."),
        ("data/processed/county_rural_weight_summary_2019_2023.csv", "County rural population weight summary."),
    ]:
        specs.append(
            InputSpec(
                "processed_training_inputs",
                path,
                "file",
                "full_reproduction_and_method_audit",
                "processed_project_table",
                "public_release_depends_on_source_rights",
                role,
            )
        )
    add_yearly(
        specs,
        group="processed_worldpop_aligned",
        template="data/processed/worldpop_aligned/chn_pop_{year}_1km_aligned.tif",
        years=years,
        required_for="full_reproduction_and_method_audit",
        access_class="public_derived_data",
        redistribution_status="redistributable_with_attribution_cc_by_4_0",
        role="WorldPop raster aligned to project grid.",
    )
    for mask in ["strict", "broad"]:
        add_yearly(
            specs,
            group="processed_rural_weights",
            template=f"data/processed/rural_weights/rural_pop_weight_{{year}}_{mask}.tif",
            years=years,
            required_for="full_reproduction_and_method_audit",
            access_class="public_derived_data",
            redistribution_status="redistribution_terms_follow_source_license",
            role=f"Rural population weight raster using {mask} rural definition.",
        )

    for path, role in [
        ("requirements_reproducibility.txt", "Python dependency list for reproducibility."),
        ("config/workflow_v1.0_draft.yaml", "Human-readable workflow configuration."),
        ("scripts/run_reproducible_pipeline.py", "Safe dry-run/execute pipeline entry point."),
        ("scripts/repro_config.py", "Shared configuration loader."),
    ]:
        specs.append(
            InputSpec(
                "reproducibility_code",
                path,
                "file",
                "full_and_partial_reproduction",
                "project_code",
                "include_in_public_code_repository",
                role,
            )
        )

    product_files = [
        ("outputs/stage6_scientific_data_product/README.md", "Main data product README."),
        ("outputs/stage6_scientific_data_product/VERSION_v1.0_draft.md", "Frozen product version record."),
        ("outputs/stage6_scientific_data_product/STAGE9_final_data_product_audit.md", "Final raster/data-product audit report."),
        ("outputs/stage6_scientific_data_product/PUBLIC_SOURCE_DATA_CITATIONS_AND_LICENSES_FILLED.md", "Public source citation and license record."),
        ("outputs/stage6_scientific_data_product/metadata/file_manifest.csv", "Package file manifest."),
        ("outputs/stage6_scientific_data_product/metadata/data_dictionary.csv", "Data dictionary."),
        ("outputs/stage6_scientific_data_product/metadata/source_data_inventory_public_sources_filled_draft.csv", "Source data inventory."),
        ("outputs/stage6_scientific_data_product/metadata/excluded_raw_data_policy_v1.0_draft.md", "Policy for raw data excluded from public release."),
        ("outputs/stage6_scientific_data_product/tables/final_data_product_raster_audit.csv", "Raster audit table."),
        ("outputs/stage6_scientific_data_product/tables/validation_summary.csv", "Validation summary table."),
    ]
    for path, role in product_files:
        specs.append(
            InputSpec(
                "final_data_product_documentation",
                path,
                "file",
                "partial_reproduction_and_reuse",
                "public_data_product_metadata",
                "include_in_public_data_repository",
                role,
            )
        )

    raster_patterns = [
        ("outputs/stage6_scientific_data_product/data/income_maps/*.tif", 5, "Main annual strict-unbounded income maps."),
        ("outputs/stage6_scientific_data_product/data/change_maps/*.tif", 2, "2019-2023 absolute and percent change maps."),
        ("outputs/stage6_scientific_data_product/data/sensitivity_bounded_maps/*.tif", 7, "Bounded sensitivity maps."),
        ("outputs/stage6_scientific_data_product/data/quality_layers/*.tif", 26, "Quality layers."),
        ("outputs/stage6_scientific_data_product/data/uncertainty_proxy_layers/*.tif", 12, "Rule-based uncertainty proxy layers."),
        ("outputs/stage6_scientific_data_product/data/ensemble_uncertainty_layers/*.tif", 15, "Multi-seed ensemble uncertainty layers."),
    ]
    for pattern, expected, role in raster_patterns:
        specs.append(
            InputSpec(
                "final_data_product_rasters",
                pattern,
                "glob",
                "partial_reproduction_and_reuse",
                "public_data_product_raster",
                "include_in_public_data_repository_if_file_size_policy_allows",
                role,
                expected_count=expected,
            )
        )

    return specs


def evaluate(spec: InputSpec) -> CheckResult:
    target = ROOT / spec.path
    found_count = 0
    if spec.check_type == "file":
        found_count = 1 if target.is_file() else 0
    elif spec.check_type == "dir":
        found_count = 1 if target.is_dir() else 0
    elif spec.check_type == "glob":
        found_count = len(list(ROOT.glob(spec.path)))
    else:
        raise ValueError(f"Unknown check_type: {spec.check_type}")

    status = "OK" if found_count >= spec.expected_count else "MISSING"
    return CheckResult(
        group=spec.group,
        path=spec.path,
        check_type=spec.check_type,
        required_for=spec.required_for,
        access_class=spec.access_class,
        redistribution_status=spec.redistribution_status,
        role=spec.role,
        expected_count=spec.expected_count,
        found_count=found_count,
        status=status,
        note=spec.note,
    )


def write_csv(results: list[CheckResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(results[0]).keys()) if results else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))


def summarize(results: list[CheckResult]) -> dict[str, int]:
    summary = {
        "total": len(results),
        "ok": sum(1 for r in results if r.status == "OK"),
        "missing": sum(1 for r in results if r.status == "MISSING"),
        "full_required_missing": sum(
            1
            for r in results
            if r.status == "MISSING"
            and r.required_for
            in {
                "full_reproduction",
                "full_reproduction_only",
                "full_reproduction_and_method_audit",
                "full_and_partial_reproduction",
            }
        ),
        "private_or_rights_review": sum(
            1
            for r in results
            if "private" in r.access_class
            or "purchased" in r.access_class
            or "provenance" in r.access_class
            or "review" in r.redistribution_status
            or "do_not_redistribute" in r.redistribution_status
        ),
    }
    return summary


def write_markdown(results: list[CheckResult], path: Path, cfg: dict) -> None:
    summary = summarize(results)
    missing = [r for r in results if r.status == "MISSING"]
    by_group: dict[str, list[CheckResult]] = {}
    for row in results:
        by_group.setdefault(row.group, []).append(row)

    lines: list[str] = []
    lines.append("# v1.0-draft 复现输入数据检查报告")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- 项目根目录：`{ROOT}`")
    lines.append(f"- 配置文件：`{deep_get(cfg, '_meta.config_path', str(DEFAULT_CONFIG))}`")
    lines.append(
        "- 配置状态："
        + ("已读取 YAML 文件" if deep_get(cfg, "_meta.loaded_from_file", False) else "未读取 YAML，使用脚本内置默认值")
    )
    lines.append("")
    lines.append("## 1. 总体结果")
    lines.append("")
    lines.append("| 指标 | 数量 |")
    lines.append("|---|---:|")
    lines.append(f"| 检查项总数 | {summary['total']} |")
    lines.append(f"| 已找到 | {summary['ok']} |")
    lines.append(f"| 缺失 | {summary['missing']} |")
    lines.append(f"| 完整复现必需但缺失 | {summary['full_required_missing']} |")
    lines.append(f"| 涉及来源追溯/购买/需权利复核的数据项 | {summary['private_or_rights_review']} |")
    lines.append("")
    lines.append("解释：这个检查器不判断科学结果好坏，只回答“文件是否在预期位置”。作者整理但来源对照尚未补齐的县级收入、购买县界、原始 OSM PBF 等即使存在，也仍应按项目政策暂不进入公开仓库。")
    lines.append("")
    lines.append("## 2. 缺失项")
    lines.append("")
    if not missing:
        lines.append("未发现缺失项。")
    else:
        lines.append("| 分组 | 路径/模式 | 用途 | 复现层级 | 备注 |")
        lines.append("|---|---|---|---|---|")
        for r in missing:
            note = r.note.replace("|", "；")
            role = r.role.replace("|", "；")
            lines.append(f"| {r.group} | `{r.path}` | {role} | {r.required_for} | {note} |")
    lines.append("")
    lines.append("## 3. 分组明细")
    lines.append("")
    for group, rows in sorted(by_group.items()):
        ok = sum(1 for r in rows if r.status == "OK")
        miss = sum(1 for r in rows if r.status == "MISSING")
        lines.append(f"### {group}")
        lines.append("")
        lines.append(f"- 已找到：{ok}")
        lines.append(f"- 缺失：{miss}")
        lines.append("")
        lines.append("| 状态 | 路径/模式 | 找到数量 | 期望数量 | 公开/许可边界 |")
        lines.append("|---|---|---:|---:|---|")
        for r in rows:
            lines.append(
                f"| {r.status} | `{r.path}` | {r.found_count} | {r.expected_count} | {r.redistribution_status} |"
            )
        lines.append("")
    lines.append("## 4. 建议")
    lines.append("")
    lines.append("1. 若目标是本机完整复现，优先补齐 `full_reproduction` 与 `full_reproduction_only` 中缺失的输入。")
    lines.append("2. 若目标是公开数据仓库，优先保留 final data product、metadata、README、data dictionary、manifest、license/citation 清单。")
    lines.append("3. 对 `do_not_redistribute_without_rights_review` 或 `review_before_public_release` 项，不要直接上传到公开仓库。")
    lines.append("4. 每次移动文件或冻结新版本后，重新运行本脚本更新 CSV/Markdown 报告。")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to workflow YAML config.")
    parser.add_argument("--csv", default=str(CSV_REPORT), help="CSV report output path.")
    parser.add_argument("--md", default=str(MD_REPORT), help="Markdown report output path.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if full-reproduction required inputs are missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    results = [evaluate(spec) for spec in build_specs(cfg)]
    write_csv(results, Path(args.csv))
    write_markdown(results, Path(args.md), cfg)

    summary = summarize(results)
    print("Stage 0 reproducibility input check")
    print(f"  checked: {summary['total']}")
    print(f"  ok: {summary['ok']}")
    print(f"  missing: {summary['missing']}")
    print(f"  full-required missing: {summary['full_required_missing']}")
    print(f"  csv: {args.csv}")
    print(f"  md: {args.md}")

    if args.strict and summary["full_required_missing"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
