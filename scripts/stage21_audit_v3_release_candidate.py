"""Final technical audit for the v3 public release candidate."""

from __future__ import annotations

import argparse
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
import rasterio


PACKAGE = ROOT / "outputs" / "official44_corrected_stage8_v3" / "package"
OUT = ROOT / "outputs" / "stage21_v3_release_candidate_audit"
EXPECTED_GROUPS = {
    "income_maps": 5,
    "change_maps": 2,
    "sensitivity_bounded_maps": 7,
    "quality_layers": 26,
    "uncertainty_proxy_layers": 12,
    "ensemble_uncertainty_layers": 15,
}
SENSITIVE_NAME_TOKENS = (
    "county_predictions", "county_training_income", "county_income_",
    "corrected_target", ".xlsx", ".xls", ".pbf", ".shp",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_raster(path: Path) -> dict:
    with rasterio.open(path) as source:
        nodata = source.nodata
        valid_count = 0
        invalid_nonfinite = 0
        minimum = np.inf
        maximum = -np.inf
        observed_values: set[int] = set()
        categorical = source.dtypes[0] == "uint8"
        for _, window in source.block_windows(1):
            array = source.read(1, window=window)
            valid = np.ones(array.shape, dtype=bool)
            if nodata is not None:
                valid &= array != nodata
            invalid_nonfinite += int((valid & ~np.isfinite(array)).sum())
            valid &= np.isfinite(array)
            if valid.any():
                values = array[valid]
                valid_count += int(values.size)
                minimum = min(minimum, float(values.min()))
                maximum = max(maximum, float(values.max()))
                if categorical and len(observed_values) <= 20:
                    observed_values.update(int(x) for x in np.unique(values))
        return {
            "path": str(path.relative_to(PACKAGE)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "width": source.width,
            "height": source.height,
            "count": source.count,
            "dtype": source.dtypes[0],
            "crs": str(source.crs),
            "transform": list(source.transform)[:6],
            "nodata": nodata,
            "valid_pixels": valid_count,
            "valid_min": minimum if valid_count else None,
            "valid_max": maximum if valid_count else None,
            "invalid_nonfinite_pixels": invalid_nonfinite,
            "categorical_values": sorted(observed_values) if categorical else [],
        }


def main() -> None:
    global PACKAGE
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=PACKAGE)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    PACKAGE = args.package_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    checks: list[dict] = []
    def check(name: str, passed: bool, detail: str, severity: str = "blocking") -> None:
        checks.append({"check": name, "passed": bool(passed), "severity": severity, "detail": detail})

    raster_paths = sorted((PACKAGE / "data").glob("**/*.tif"))
    group_counts = {
        name: len(list((PACKAGE / "data" / name).glob("*.tif")))
        for name in EXPECTED_GROUPS
    }
    check("total_geotiff_count", len(raster_paths) == 67, f"found={len(raster_paths)}, expected=67")
    for group, expected in EXPECTED_GROUPS.items():
        check(f"group_count:{group}", group_counts[group] == expected, f"found={group_counts[group]}, expected={expected}")

    raster_rows = [audit_raster(path) for path in raster_paths]
    grid_keys = {
        (row["width"], row["height"], row["crs"], tuple(row["transform"]))
        for row in raster_rows
    }
    check("common_grid", len(grid_keys) == 1, f"unique_grids={len(grid_keys)}")
    check("single_band", all(row["count"] == 1 for row in raster_rows), "all rasters must have one band")
    check("positive_file_size", all(row["bytes"] > 0 for row in raster_rows), "all GeoTIFF files must be non-empty")
    check("valid_pixels_present", all(row["valid_pixels"] > 0 for row in raster_rows), "all layers must contain valid pixels")
    check("finite_valid_values", all(row["invalid_nonfinite_pixels"] == 0 for row in raster_rows), "valid pixels must not contain NaN/Inf")

    negative_forbidden = [
        row["path"] for row in raster_rows
        if any(token in row["path"] for token in ("income_maps/", "ensemble_std", "ensemble_cv"))
        and row["valid_min"] is not None and row["valid_min"] < 0
    ]
    check("nonnegative_income_and_ensemble_spread", not negative_forbidden, f"violations={negative_forbidden}")

    score_violations = [
        row["path"] for row in raster_rows
        if "uncertainty_proxy_score" in row["path"]
        and (row["valid_min"] < 0 or row["valid_max"] > 100)
    ]
    check("quality_risk_score_range", not score_violations, f"violations={score_violations}")

    manifest_path = PACKAGE / "metadata" / "v3_release_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing_manifest_files = []
    hash_mismatches = []
    for record in manifest["files"]:
        path = PACKAGE / record["path"]
        if not path.is_file():
            missing_manifest_files.append(record["path"])
        elif sha256(path) != record["sha256"]:
            hash_mismatches.append(record["path"])
    check("manifest_files_exist", not missing_manifest_files, f"missing={missing_manifest_files}")
    check("manifest_hashes_match", not hash_mismatches, f"mismatches={hash_mismatches}")

    all_files = [path for path in PACKAGE.rglob("*") if path.is_file()]
    sensitive = [
        str(path.relative_to(PACKAGE)).replace("\\", "/")
        for path in all_files
        if any(token.lower() in path.name.lower() for token in SENSITIVE_NAME_TOKENS)
    ]
    check("restricted_filename_exclusion", not sensitive, f"flagged={sensitive}")

    required = [
        PACKAGE / "README_V3_RELEASE_CANDIDATE.md",
        PACKAGE / "metadata" / "data_dictionary.csv",
        PACKAGE / "metadata" / "stage8_run_manifest.json",
        PACKAGE / "metadata" / "v3_release_candidate_manifest.json",
        PACKAGE / "tables" / "leave_year_2019_2023_strict_overall_metrics.json",
        PACKAGE / "tables" / "leave_province_2020_strict_overall_metrics.json",
    ]
    check("core_documentation_present", all(path.is_file() for path in required), f"missing={[str(p) for p in required if not p.is_file()]}")

    publication_required = [
        PACKAGE / "LICENSE_DATA_CC_BY_4.0.txt",
        PACKAGE / "CITATION.cff",
        PACKAGE / "PUBLIC_SOURCE_DATA_CITATIONS_AND_LICENSES.md",
    ]
    check(
        "publication_license_and_citation_files",
        all(path.is_file() for path in publication_required),
        f"missing={[p.name for p in publication_required if not p.is_file()]}",
        severity="publication_blocker",
    )

    pd.DataFrame(raster_rows).to_csv(output / "raster_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(checks).to_csv(output / "checklist.csv", index=False, encoding="utf-8-sig")
    blockers = [row for row in checks if not row["passed"]]
    report = {
        "package": str(PACKAGE),
        "geotiff_count": len(raster_paths),
        "group_counts": group_counts,
        "checks_passed": sum(row["passed"] for row in checks),
        "checks_total": len(checks),
        "unresolved": blockers,
        "technical_data_checks_passed": not any(
            not row["passed"] and row["severity"] == "blocking" for row in checks
        ),
        "ready_for_publication": not blockers,
    }
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# v3 发布候选包最终技术审查",
        "",
        f"- GeoTIFF：{len(raster_paths)}幅。",
        f"- 检查通过：{report['checks_passed']}/{report['checks_total']}。",
        f"- 栅格与数据技术检查：{'通过' if report['technical_data_checks_passed'] else '未通过'}。",
        f"- 可直接公开发布：{'是' if report['ready_for_publication'] else '否'}。",
        "",
        "## 未解决事项",
        "",
    ]
    if blockers:
        lines.extend(f"- [{row['severity']}] {row['check']}：{row['detail']}" for row in blockers)
    else:
        lines.append("- 无。")
    lines.extend([
        "",
        "县级收入标签、县级预测表、购买边界和原始工作簿未进入候选包。论文、图件和Zenodo尚未在本阶段更新。",
    ])
    (output / "AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
