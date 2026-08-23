"""Build a public-safe provenance-audited revision of the v1.5-no-road package.

This script does not expose county income values, source file paths, source-file
hashes, or purchased/restricted source documents. It publishes a normalized
record-to-source crosswalk and a separate source-item catalogue.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_PACKAGE = (
    ROOT
    / "outputs"
    / "public_release_v1_5_no_road_final"
    / "china_rural_income_1km_2019_2023_v1.5-no-road"
)
LATEST_INDEX = (
    ROOT
    / "outputs"
    / "provenance_batch_20260822"
    / "county_training_label_provenance_index_v2.4_jiangsu_corrected.csv"
)
SOURCE_CATALOGUE = (
    ROOT
    / "outputs"
    / "provenance_batch_20260816"
    / "source_items_v2.0_zhejiang2019_indirect.csv"
)
LEGACY_PUBLIC_SOURCES = (
    ROOT
    / "outputs"
    / "stage6_scientific_data_product"
    / "metadata"
    / "county_training_label_source_items_public_v2.csv"
)
OUT_ROOT = ROOT / "outputs" / "public_release_v1_5_no_road_provenance_audited"
PACKAGE_NAME = "china_rural_income_1km_2019_2023_v1.5-no-road"
PACKAGE = OUT_ROOT / PACKAGE_NAME
ZIP_PATH = OUT_ROOT / f"{PACKAGE_NAME}-provenance-audited.zip"
CHECKSUM_PATH = OUT_ROOT / "SHA256SUMS_v1.5-no-road-provenance-audited.txt"
AUDIT_PATH = OUT_ROOT / "PROVENANCE_RELEASE_AUDIT.json"

VERIFIED_STATUSES = {
    "record_value_verified",
    "exact_source_item_and_value_verified",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_nonempty(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def build_public_metadata() -> dict[str, object]:
    metadata = PACKAGE / "metadata"
    latest = read_csv(LATEST_INDEX)
    source_rows = read_csv(SOURCE_CATALOGUE)
    legacy_rows = read_csv(LEGACY_PUBLIC_SOURCES)

    if len(latest) != 8373:
        raise RuntimeError(f"Unexpected record count: {len(latest)}")
    if any(not row["exact_source_item_id"].strip() for row in latest):
        raise RuntimeError("At least one record lacks exact_source_item_id")

    exact_ids = sorted({row["exact_source_item_id"].strip() for row in latest})
    source_lookup = {row["source_item_id"].strip(): row for row in source_rows}
    legacy_lookup = {row["source_item_id"].strip(): row for row in legacy_rows}
    missing_ids = [
        source_id
        for source_id in exact_ids
        if source_id not in source_lookup and source_id not in legacy_lookup
    ]
    if missing_ids:
        raise RuntimeError(f"Unresolved source ids: {missing_ids}")

    status_counts = Counter(row["source_verification_status"] for row in latest)
    verified_count = sum(
        count for status, count in status_counts.items() if status in VERIFIED_STATUSES
    )
    pending_count = len(latest) - verified_count
    if verified_count != 7840 or pending_count != 533:
        raise RuntimeError(
            f"Unexpected verification counts: verified={verified_count}, "
            f"pending={pending_count}"
        )

    linked_counts = Counter(row["exact_source_item_id"].strip() for row in latest)
    verified_by_source = Counter(
        row["exact_source_item_id"].strip()
        for row in latest
        if row["source_verification_status"] in VERIFIED_STATUSES
    )

    public_sources: list[dict[str, object]] = []
    locator_status_by_id: dict[str, str] = {}
    for source_id in exact_ids:
        current = source_lookup.get(source_id, {})
        legacy = legacy_lookup.get(source_id, {})
        publication_year = first_nonempty(
            current.get("publication_year"), legacy.get("publication_year")
        )
        table_or_section = first_nonempty(
            current.get("table_name"), legacy.get("table_or_section")
        )
        page_or_location = first_nonempty(legacy.get("page_or_location"))
        access_url = first_nonempty(current.get("access_url"), legacy.get("access_url"))
        access_date = first_nonempty(
            current.get("access_date"), legacy.get("access_date")
        )
        title = first_nonempty(current.get("title"), legacy.get("title"))
        if access_url and table_or_section:
            locator_status = "exact_table_or_section_and_url_available"
        elif table_or_section:
            locator_status = "exact_table_or_section_available"
        elif access_url:
            locator_status = "stable_url_available"
        else:
            locator_status = "bibliographic_title_or_local_source_name_only"
        locator_status_by_id[source_id] = locator_status
        notes = first_nonempty(current.get("rejection_reason"), legacy.get("notes"))
        if not notes and locator_status.endswith("only"):
            notes = (
                "The exact source item is mapped internally, but a stable public URL "
                "or separate page/table locator was not recoverable. No income value "
                "or restricted source document is redistributed."
            )
        public_sources.append({
            "source_item_id": source_id,
            "province": first_nonempty(current.get("province"), legacy.get("province")),
            "data_years": first_nonempty(current.get("years"), legacy.get("data_years")),
            "source_kind": first_nonempty(
                current.get("source_kind"), legacy.get("source_kind")
            ),
            "title": title,
            "publication_year": publication_year,
            "table_or_section": table_or_section,
            "page_or_location": page_or_location,
            "access_url": access_url,
            "access_date": access_date,
            "locator_status": locator_status,
            "verification_role": first_nonempty(
                current.get("status_hint"), legacy.get("verification_status")
            ),
            "linked_label_count": linked_counts[source_id],
            "numeric_verified_count": verified_by_source[source_id],
            "numeric_pending_count": linked_counts[source_id] - verified_by_source[source_id],
            "notes": notes,
        })

    public_records: list[dict[str, object]] = []
    for row in latest:
        source_id = row["exact_source_item_id"].strip()
        status = row["source_verification_status"]
        public_records.append({
            "provenance_record_id": row["provenance_record_id"],
            "year": row["year"],
            "province_code": row["province_code"],
            "province": row["province"],
            "city": row["city"],
            "county_code": row["county_code"],
            "county": row["county"],
            "source_item_id": source_id,
            "source_locator_status": locator_status_by_id[source_id],
            "standardized_indicator": row["standardized_indicator"],
            "indicator_correction": row["indicator_correction"],
            "numeric_match_status": row["numeric_match_status"],
            "source_verification_status": status,
            "numeric_value_verified": "yes" if status in VERIFIED_STATUSES else "no",
            "included_in_final_training": (
                "yes" if row["main_analysis_treatment"] == "included" else "no"
            ),
            "sensitivity_treatment": row["sensitivity_treatment"],
            "income_value_included_in_public_index": "no",
        })

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in latest:
        grouped[(row["province_code"], row["province"], row["year"])].append(row)
    province_year_rows: list[dict[str, object]] = []
    for (province_code, province, year), rows in sorted(grouped.items()):
        group_verified = sum(
            row["source_verification_status"] in VERIFIED_STATUSES for row in rows
        )
        source_ids = sorted({row["exact_source_item_id"] for row in rows})
        province_year_rows.append({
            "province_code": province_code,
            "province": province,
            "data_year": year,
            "indicator": "Rural residents' per capita disposable income",
            "training_record_count": len(rows),
            "source_mapped_count": len(rows),
            "county_name_and_value_verified_count": group_verified,
            "numeric_verification_pending_count": len(rows) - group_verified,
            "source_item_count": len(source_ids),
            "source_item_ids": ";".join(source_ids),
            "verification_status": (
                "all_record_values_verified"
                if group_verified == len(rows)
                else "source_mapped_with_some_record_values_pending"
            ),
            "redistribution_note": (
                "County income values and restricted source files are not included."
            ),
        })

    coverage_rows = [
        {
            "metric": "total_records",
            "records": len(latest),
            "share_of_all_records": 1.0,
            "definition": "All county-year labels entering the final training data flow",
        },
        {
            "metric": "exact_source_item_records",
            "records": len(latest),
            "share_of_all_records": 1.0,
            "definition": "Records linked to a source_item_id in the public source catalogue",
        },
        {
            "metric": "value_verified_records_all",
            "records": verified_count,
            "share_of_all_records": verified_count / len(latest),
            "definition": "County name and numeric value verified against the linked source",
        },
        {
            "metric": "numeric_verification_pending_records",
            "records": pending_count,
            "share_of_all_records": pending_count / len(latest),
            "definition": "Exact source mapped but record-level numeric verification remains pending",
        },
    ]
    for status, count in sorted(status_counts.items()):
        coverage_rows.append({
            "metric": f"status__{status}",
            "records": count,
            "share_of_all_records": count / len(latest),
            "definition": "Count by source_verification_status in the public record index",
        })

    record_name = "county_training_label_record_provenance_public_v2.4.csv"
    source_name = "county_training_label_source_items_public_v2.4.csv"
    coverage_name = "county_training_label_coverage_summary_v2.4.csv"
    province_year_name = (
        "county_training_label_province_year_provenance_public_v2.4.csv"
    )
    readme_name = "README_county_training_label_provenance_v2.4.md"

    old_summary = metadata / "county_training_label_province_year_provenance_public_v2.csv"
    if old_summary.exists():
        old_summary.unlink()

    write_csv(
        metadata / record_name,
        public_records,
        list(public_records[0].keys()),
    )
    write_csv(
        metadata / source_name,
        public_sources,
        list(public_sources[0].keys()),
    )
    write_csv(
        metadata / province_year_name,
        province_year_rows,
        list(province_year_rows[0].keys()),
    )
    write_csv(
        metadata / coverage_name,
        coverage_rows,
        list(coverage_rows[0].keys()),
    )

    locator_counts = Counter(row["locator_status"] for row in public_sources)
    readme = f"""# County training-label provenance, public v2.4

This metadata release provides a normalized, public-safe audit trail for the
county-year income labels used by the study. It contains no county income values,
restricted source documents, local file paths, or internal file hashes.

## Coverage

- Training-label records: {len(latest):,}.
- Records linked to an exact `source_item_id`: {len(latest):,} (100%).
- County names and numeric values verified: {verified_count:,} ({verified_count / len(latest) * 100:.2f}%).
- Records with numeric verification pending: {pending_count:,} ({pending_count / len(latest) * 100:.2f}%).
- Distinct source items referenced: {len(public_sources)}.
- Province-year groups: {len(province_year_rows)}.

## Relational structure

Join `{record_name}` to `{source_name}` using `source_item_id`. The record table
identifies the province, city, county, year, verification status, and final-training
treatment of every label. The source table stores bibliographic and locator metadata
once per source item. `{province_year_name}` provides the compact 98-group summary.

The field `numeric_value_verified=no` does not mean that the record lacks a source.
It means that the source item has been identified but that the county-name/value pair
has not yet been checked at record level. These {pending_count:,} records remain
explicitly distinguishable for risk assessment or verified-only sensitivity tests.

## Locator interpretation

Exact table/section names or stable URLs are preferred. A page number is not required
when an exact table/section or stable URL uniquely locates the source. For historical
items whose public URL or page/table locator could not be recovered, `locator_status`
is set to `bibliographic_title_or_local_source_name_only`; these cases are disclosed
rather than represented as fully public locators.

Source-locator status counts: {json.dumps(dict(locator_counts), ensure_ascii=False)}.
"""
    (metadata / readme_name).write_text(readme, encoding="utf-8")

    return {
        "record_file": record_name,
        "source_file": source_name,
        "province_year_file": province_year_name,
        "coverage_file": coverage_name,
        "readme_file": readme_name,
        "record_count": len(public_records),
        "source_item_count": len(public_sources),
        "province_year_group_count": len(province_year_rows),
        "verified_count": verified_count,
        "pending_count": pending_count,
        "locator_status_counts": dict(locator_counts),
    }


def update_package_readme(provenance: dict[str, object]) -> None:
    readme_path = PACKAGE / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    marker = "## County training-label provenance"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n"
    text += f"""

## County training-label provenance

The `metadata/` directory includes a public-safe record-level provenance crosswalk
for all {provenance['record_count']:,} county-year training labels and a separate
catalogue of {provenance['source_item_count']} source items. All records have an exact
source-item mapping; {provenance['verified_count']:,} record values are verified and
{provenance['pending_count']:,} remain explicitly flagged as numeric-verification
pending. County income values and restricted source documents are not redistributed.
See `metadata/{provenance['readme_file']}`.
"""
    readme_path.write_text(text, encoding="utf-8")


def update_release_summary(provenance: dict[str, object]) -> None:
    path = PACKAGE / "metadata" / "release_summary_v1.5.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["training_label_provenance"] = {
        "record_level_public_index_included": True,
        "record_count": provenance["record_count"],
        "exact_source_mapped_count": provenance["record_count"],
        "numeric_verified_count": provenance["verified_count"],
        "numeric_pending_count": provenance["pending_count"],
        "source_item_count": provenance["source_item_count"],
        "county_income_values_redistributed": False,
        "restricted_source_documents_redistributed": False,
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def write_manifest() -> None:
    path = PACKAGE / "metadata" / "v1_5_no_road_manifest.json"
    files = []
    for item in sorted(PACKAGE.rglob("*")):
        if not item.is_file() or item == path:
            continue
        files.append({
            "path": item.relative_to(PACKAGE).as_posix(),
            "bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        })
    manifest = {
        "version": "v1.5-no-road",
        "metadata_revision": "provenance-audited-v2.4",
        "release_status": "formal_frozen_peer_review_package",
        "package_root": PACKAGE_NAME,
        "file_count_excluding_manifest": len(files),
        "file_count_including_manifest": len(files) + 1,
        "geotiff_count": sum(row["path"].lower().endswith(".tif") for row in files),
        "files": files,
        "restricted_county_targets_included": False,
        "county_income_values_in_public_provenance": False,
        "record_level_public_provenance_included": True,
        "administrative_assignment_boundaries_included": False,
        "road_density_included": False,
        "source_products_overwritten": False,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_zip() -> None:
    with zipfile.ZipFile(
        ZIP_PATH,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for item in sorted(PACKAGE.rglob("*")):
            if item.is_file():
                archive.write(item, Path(PACKAGE_NAME) / item.relative_to(PACKAGE))
    CHECKSUM_PATH.write_text(
        f"{sha256_file(ZIP_PATH)}  {ZIP_PATH.name}\n", encoding="ascii"
    )


def verify(provenance: dict[str, object]) -> dict[str, object]:
    metadata = PACKAGE / "metadata"
    records = read_csv(metadata / str(provenance["record_file"]))
    sources = read_csv(metadata / str(provenance["source_file"]))
    source_ids = {row["source_item_id"] for row in sources}
    unmatched = sorted({row["source_item_id"] for row in records} - source_ids)
    exposed_headers = {
        name
        for name in records[0]
        if "source_value" in name
        or "label_value" in name
        or name in {"income", "income_value", "training_input_file", "file_path_internal"}
    }
    with zipfile.ZipFile(ZIP_PATH) as archive:
        bad_member = archive.testzip()
        zip_member_count = len(archive.namelist())
    manifest = json.loads(
        (metadata / "v1_5_no_road_manifest.json").read_text(encoding="utf-8")
    )
    checks = {
        "record_count_is_8373": len(records) == 8373,
        "source_item_count_is_95": len(sources) == 95,
        "all_record_source_ids_resolve": not unmatched,
        "no_income_value_headers_exposed": not exposed_headers,
        "geotiff_count_is_67": manifest["geotiff_count"] == 67,
        "zip_integrity_ok": bad_member is None,
        "manifest_and_package_counts_match": (
            manifest["file_count_including_manifest"]
            == len([item for item in PACKAGE.rglob("*") if item.is_file()])
        ),
    }
    if not all(checks.values()):
        raise RuntimeError({
            "checks": checks,
            "unmatched_source_ids": unmatched,
            "exposed_headers": sorted(exposed_headers),
            "bad_zip_member": bad_member,
        })
    result = {
        "package": str(PACKAGE),
        "zip": str(ZIP_PATH),
        "zip_bytes": ZIP_PATH.stat().st_size,
        "zip_sha256": sha256_file(ZIP_PATH),
        "zip_member_count": zip_member_count,
        "package_file_count": manifest["file_count_including_manifest"],
        "geotiff_count": manifest["geotiff_count"],
        "provenance": provenance,
        "checks": checks,
    }
    AUDIT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError(f"Refusing to overwrite existing output: {OUT_ROOT}")
    OUT_ROOT.mkdir(parents=True)
    shutil.copytree(BASE_PACKAGE, PACKAGE)
    provenance = build_public_metadata()
    update_package_readme(provenance)
    update_release_summary(provenance)
    write_manifest()
    write_zip()
    print(json.dumps(verify(provenance), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
