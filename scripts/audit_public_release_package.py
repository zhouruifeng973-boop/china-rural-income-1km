"""Audit the conservative public-release package before upload.

The audit checks:
1. Required README/metadata files exist.
2. Files marked as included in the public-release manifest exist and match
   size/SHA-256.
3. Files marked as excluded in the manifest are absent from the release package.
4. No obvious county-level restricted tables slipped into the public package.
5. GeoTIFF files can be opened and share the same CRS, shape, transform, and
   single-band layout when rasterio is available.

Outputs are written inside the public release package:
    metadata/public_release_audit_v1.0_draft.csv
    PUBLIC_RELEASE_AUDIT_v1.0_draft.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_DIR = ROOT / "outputs" / "public_release" / "china_rural_income_1km_2019_2023_v1.0-draft"

LOCAL_DEP_PATHS = [
    ROOT / ".codex_pydeps_stage8",
    ROOT / ".codex_pydeps",
    ROOT / "stage6_pydeps",
]


REQUIRED_FILES = [
    "README.md",
    "VERSION_v1.0_draft.md",
    "PUBLIC_RELEASE_README_v1.0_draft.md",
    "PUBLIC_SOURCE_DATA_CITATIONS_AND_LICENSES_FILLED.md",
    "SOURCE_DATA_LICENSE_AND_CITATION_AUDIT.md",
    "STAGE7_uncertainty_proxy_and_validation_summary.md",
    "STAGE8_ensemble_uncertainty_summary.md",
    "STAGE9_final_data_product_audit.md",
    "metadata/data_dictionary.csv",
    "metadata/file_manifest.csv",
    "metadata/public_release_manifest_v1.0_draft.csv",
    "metadata/source_data_inventory_public_sources_filled_draft.csv",
    "metadata/excluded_raw_data_policy_v1.0_draft.md",
    "tables/validation_summary.csv",
    "tables/final_data_product_raster_audit.csv",
]


FORBIDDEN_FILES = [
    "tables/county_training_income_2019_2023_2019yuan.csv",
    "tables/validation_leave_province_2020_county_residuals.csv",
    "tables/validation_leave_year_county_residuals.csv",
]


FORBIDDEN_NAME_PATTERNS = [
    "county_training_income",
    "county_residuals",
    "county_rural_income_",
    "CNT_ResidentIncome",
    "中国各县域城乡居民收入",
    "2020xianBJ",
]


@dataclass
class AuditRow:
    category: str
    item: str
    status: str
    details: str


def add_local_dependency_paths() -> None:
    for path in reversed(LOCAL_DEP_PATHS):
        if path.exists():
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def audit_required_files(release_dir: Path) -> list[AuditRow]:
    rows: list[AuditRow] = []
    for item in REQUIRED_FILES:
        path = release_dir / item
        rows.append(
            AuditRow(
                "required_file",
                item,
                "OK" if path.is_file() else "FAIL",
                "present" if path.is_file() else "missing",
            )
        )
    return rows


def audit_manifest(release_dir: Path) -> list[AuditRow]:
    manifest_path = release_dir / "metadata" / "public_release_manifest_v1.0_draft.csv"
    if not manifest_path.is_file():
        return [
            AuditRow(
                "manifest",
                "metadata/public_release_manifest_v1.0_draft.csv",
                "FAIL",
                "manifest missing; cannot validate checksums",
            )
        ]

    rows: list[AuditRow] = []
    manifest = read_manifest(manifest_path)
    for entry in manifest:
        action = entry.get("action", "")
        relative_path = entry.get("relative_path", "")
        expected_bytes = int(entry.get("bytes", "0") or 0)
        expected_sha = entry.get("sha256", "")
        path = release_dir / relative_path

        if action == "include":
            if not path.is_file():
                rows.append(AuditRow("manifest_include", relative_path, "FAIL", "included file missing"))
                continue
            actual_bytes = path.stat().st_size
            actual_sha = sha256_file(path)
            if actual_bytes != expected_bytes:
                rows.append(
                    AuditRow(
                        "manifest_include",
                        relative_path,
                        "FAIL",
                        f"size mismatch expected={expected_bytes} actual={actual_bytes}",
                    )
                )
            elif actual_sha != expected_sha:
                rows.append(AuditRow("manifest_include", relative_path, "FAIL", "sha256 mismatch"))
            else:
                rows.append(AuditRow("manifest_include", relative_path, "OK", "size and sha256 match"))
        elif action == "exclude":
            rows.append(
                AuditRow(
                    "manifest_exclude",
                    relative_path,
                    "OK" if not path.exists() else "FAIL",
                    "absent as expected" if not path.exists() else "excluded file exists in release package",
                )
            )
        else:
            rows.append(AuditRow("manifest", relative_path or "<blank>", "FAIL", f"unknown action={action!r}"))
    return rows


def audit_forbidden_files(release_dir: Path) -> list[AuditRow]:
    rows: list[AuditRow] = []
    for item in FORBIDDEN_FILES:
        path = release_dir / item
        rows.append(
            AuditRow(
                "forbidden_file",
                item,
                "OK" if not path.exists() else "FAIL",
                "absent" if not path.exists() else "forbidden file present",
            )
        )

    for file in sorted(release_dir.rglob("*")):
        if not file.is_file():
            continue
        relative_path = rel(file, release_dir)
        for pattern in FORBIDDEN_NAME_PATTERNS:
            if pattern.lower() in relative_path.lower():
                rows.append(
                    AuditRow(
                        "forbidden_name_pattern",
                        relative_path,
                        "WARN",
                        f"name contains review keyword: {pattern}",
                    )
                )
    return rows


def load_rasterio() -> Any | None:
    add_local_dependency_paths()
    try:
        import rasterio  # type: ignore
    except Exception:
        return None
    return rasterio


def audit_rasters(release_dir: Path) -> list[AuditRow]:
    tif_files = sorted((release_dir / "data").rglob("*.tif")) if (release_dir / "data").is_dir() else []
    rows: list[AuditRow] = []
    if not tif_files:
        return [AuditRow("raster", "data/*.tif", "FAIL", "no GeoTIFF files found")]

    rasterio = load_rasterio()
    if rasterio is None:
        fallback_rows = audit_raster_audit_table(release_dir, len(tif_files))
        rows.extend(fallback_rows)
        rows.append(AuditRow("raster_count", "data/*.tif", "OK", f"{len(tif_files)} GeoTIFF files found"))
        return rows

    reference: dict[str, Any] | None = None
    opened = 0
    failed_open = 0
    mismatch = 0
    nodata_values: dict[str, int] = {}
    dtypes: dict[str, int] = {}

    for tif in tif_files:
        relative_path = rel(tif, release_dir)
        try:
            with rasterio.open(tif) as src:
                opened += 1
                profile = {
                    "crs": str(src.crs),
                    "width": src.width,
                    "height": src.height,
                    "transform": tuple(round(x, 12) for x in src.transform),
                    "count": src.count,
                }
                nodata_key = str(src.nodata)
                dtype_key = ",".join(src.dtypes)
                nodata_values[nodata_key] = nodata_values.get(nodata_key, 0) + 1
                dtypes[dtype_key] = dtypes.get(dtype_key, 0) + 1
                if reference is None:
                    reference = profile
                    rows.append(AuditRow("raster_reference", relative_path, "OK", str(profile)))
                elif profile != reference:
                    mismatch += 1
                    rows.append(AuditRow("raster_alignment", relative_path, "FAIL", f"mismatch: {profile}"))
        except Exception as exc:
            failed_open += 1
            rows.append(AuditRow("raster_open", relative_path, "FAIL", repr(exc)))

    rows.append(AuditRow("raster_count", "data/*.tif", "OK", f"{len(tif_files)} GeoTIFF files found"))
    rows.append(AuditRow("raster_open_summary", "data/*.tif", "OK" if failed_open == 0 else "FAIL", f"opened={opened}; failed={failed_open}"))
    rows.append(AuditRow("raster_alignment_summary", "data/*.tif", "OK" if mismatch == 0 else "FAIL", f"mismatched={mismatch}"))
    rows.append(AuditRow("raster_nodata_summary", "data/*.tif", "INFO", str(nodata_values)))
    rows.append(AuditRow("raster_dtype_summary", "data/*.tif", "INFO", str(dtypes)))
    return rows


def audit_raster_audit_table(release_dir: Path, actual_tif_count: int) -> list[AuditRow]:
    """Use the Stage 9 raster audit table as fallback when rasterio is unavailable."""
    table = release_dir / "tables" / "final_data_product_raster_audit.csv"
    if not table.is_file():
        return [
            AuditRow(
                "raster_fallback_audit",
                "tables/final_data_product_raster_audit.csv",
                "WARN",
                "rasterio is unavailable and Stage 9 raster audit table is missing",
            )
        ]

    with table.open("r", encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))

    def truthy(value: str) -> bool:
        return str(value).strip().lower() in {"true", "1", "yes", "ok"}

    align_fail = [r.get("package_file_norm", r.get("relpath", "")) for r in records if not truthy(r.get("aligns_reference", ""))]
    nodata_fail = [r.get("package_file_norm", r.get("relpath", "")) for r in records if str(r.get("nodata_status", "")).lower() != "ok"]
    naming_fail = [r.get("package_file_norm", r.get("relpath", "")) for r in records if str(r.get("naming_status", "")).lower() != "ok"]
    manifest_fail = [r.get("package_file_norm", r.get("relpath", "")) for r in records if not truthy(r.get("manifest_covered", ""))]

    rows: list[AuditRow] = []
    rows.append(
        AuditRow(
            "raster_dependency",
            "rasterio",
            "OK",
            "rasterio unavailable in current Python; used included Stage 9 raster audit table as fallback",
        )
    )
    rows.append(
        AuditRow(
            "raster_fallback_audit_count",
            "tables/final_data_product_raster_audit.csv",
            "OK" if len(records) == actual_tif_count else "WARN",
            f"table_rows={len(records)}; release_tif_files={actual_tif_count}",
        )
    )
    checks = [
        ("raster_fallback_alignment", align_fail),
        ("raster_fallback_nodata", nodata_fail),
        ("raster_fallback_naming", naming_fail),
        ("raster_fallback_manifest", manifest_fail),
    ]
    for category, failures in checks:
        rows.append(
            AuditRow(
                category,
                "tables/final_data_product_raster_audit.csv",
                "OK" if not failures else "FAIL",
                "all records pass" if not failures else "; ".join(failures[:10]),
            )
        )
    return rows


def write_csv(rows: list[AuditRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_markdown(rows: list[AuditRow], path: Path, release_dir: Path) -> None:
    fail = [r for r in rows if r.status == "FAIL"]
    warn = [r for r in rows if r.status == "WARN"]
    ok = [r for r in rows if r.status == "OK"]
    info = [r for r in rows if r.status == "INFO"]

    lines: list[str] = []
    lines.append("# v1.0-draft 公开发布包二次审查报告")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- 审查目录：`{release_dir}`")
    lines.append("")
    lines.append("## 1. 总体结果")
    lines.append("")
    lines.append("| 状态 | 数量 |")
    lines.append("|---|---:|")
    lines.append(f"| OK | {len(ok)} |")
    lines.append(f"| WARN | {len(warn)} |")
    lines.append(f"| FAIL | {len(fail)} |")
    lines.append(f"| INFO | {len(info)} |")
    lines.append("")
    if fail:
        lines.append("结论：未通过。需要先处理 FAIL 项再上传。")
    elif warn:
        lines.append("结论：基本通过，但存在 WARN 项，上传前建议人工确认。")
    else:
        lines.append("结论：通过。未发现阻塞公开发布的问题。")
    lines.append("")
    lines.append("## 2. FAIL / WARN 项")
    lines.append("")
    flagged = fail + warn
    if not flagged:
        lines.append("没有 FAIL 或 WARN 项。")
    else:
        lines.append("| 状态 | 类别 | 项目 | 说明 |")
        lines.append("|---|---|---|---|")
        for row in flagged:
            details = row.details.replace("|", "；")
            lines.append(f"| {row.status} | {row.category} | `{row.item}` | {details} |")
    lines.append("")
    lines.append("## 3. 信息项")
    lines.append("")
    if not info:
        lines.append("无。")
    else:
        lines.append("| 类别 | 项目 | 说明 |")
        lines.append("|---|---|---|")
        for row in info:
            details = row.details.replace("|", "；")
            lines.append(f"| {row.category} | `{row.item}` | {details} |")
    lines.append("")
    lines.append("## 4. 审查内容")
    lines.append("")
    lines.append("- 必需 README、版本、来源许可、data dictionary、manifest 是否存在；")
    lines.append("- public release manifest 中 include 文件是否存在、大小和 SHA-256 是否一致；")
    lines.append("- public release manifest 中 exclude 文件是否确实不在公开包内；")
    lines.append("- 县级购买收入相关文件名是否误入公开包；")
    lines.append("- GeoTIFF 是否可打开、CRS/分辨率/网格/波段数是否一致。")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", default=str(DEFAULT_RELEASE_DIR), help="Public release package directory.")
    parser.add_argument("--strict", action="store_true", help="Return exit code 1 when WARN or FAIL items exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_dir = Path(args.release_dir)
    if not release_dir.is_absolute():
        release_dir = ROOT / release_dir
    if not release_dir.is_dir():
        raise SystemExit(f"Release directory not found: {release_dir}")

    rows: list[AuditRow] = []
    rows.extend(audit_required_files(release_dir))
    rows.extend(audit_manifest(release_dir))
    rows.extend(audit_forbidden_files(release_dir))
    rows.extend(audit_rasters(release_dir))

    csv_path = release_dir / "metadata" / "public_release_audit_v1.0_draft.csv"
    md_path = release_dir / "PUBLIC_RELEASE_AUDIT_v1.0_draft.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, release_dir)

    fail_count = sum(1 for r in rows if r.status == "FAIL")
    warn_count = sum(1 for r in rows if r.status == "WARN")
    ok_count = sum(1 for r in rows if r.status == "OK")
    info_count = sum(1 for r in rows if r.status == "INFO")

    print("Public release package audit")
    print(f"  release_dir: {release_dir}")
    print(f"  OK: {ok_count}")
    print(f"  WARN: {warn_count}")
    print(f"  FAIL: {fail_count}")
    print(f"  INFO: {info_count}")
    print(f"  csv: {csv_path}")
    print(f"  md: {md_path}")

    if fail_count > 0 or (args.strict and warn_count > 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
