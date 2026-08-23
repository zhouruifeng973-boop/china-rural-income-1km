"""Build the v1.5.3-no-road peer-review package with provenance v2.5.

The script copies the audited 156-file v1.5 package, replaces the five v2.4
provenance artefacts one-for-one, updates release metadata, regenerates the
155-entry manifest (the manifest itself is the 156th file), and writes a ZIP
plus external checksum/audit records.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "outputs/public_release_v1_5_no_road_final_review_complete"
    / "china_rural_income_1km_2019_2023_v1.5-no-road"
)
PROV = ROOT / "outputs/provenance_completion_v2_5"
OUT = ROOT / "outputs/public_release_v1_5_3_no_road_provenance_v2_5"
PACKAGE_NAME = "china_rural_income_1km_2019_2023_v1.5.3-no-road"
PACKAGE = OUT / PACKAGE_NAME
ZIP_PATH = OUT / f"{PACKAGE_NAME}.zip"
MANIFEST = PACKAGE / "metadata/v1_5_no_road_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_text(path: Path, replacements: list[tuple[str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    for old, new in replacements:
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8", newline="\n")


def build_provenance_readme() -> str:
    return """# County training-label provenance, public v2.5

This metadata release provides a normalized, public-safe audit trail for the
county-year income labels used by the study. It contains no county income values,
restricted source documents, local file paths, or internal file hashes.

## Coverage

- Training-label records: 8,373.
- Records linked to an exact `source_item_id`: 8,373 (100%).
- County names and numeric values verified: 7,933 (94.75%).
- Records with numeric verification pending: 440 (5.25%).
- Distinct source items referenced: 95.
- Province-year groups: 98.
- Source items with an exact table or section locator: 95; 40 also provide a URL.

## Relational structure

Join `county_training_label_record_provenance_public_v2.5.csv` to
`county_training_label_source_items_public_v2.5.csv` using `source_item_id`.
The record table identifies the province, city, county, year, verification status,
and final-training treatment of every label. The source table stores bibliographic
and locator metadata once per source item.
`county_training_label_province_year_provenance_public_v2.5.csv` provides the
compact 98-group summary.

The field `numeric_value_verified=no` does not mean that the record lacks a source.
It means that the source item and exact table or section have been identified but
that the county-name/value pair remains unresolved at record level. These 440 records
remain explicitly distinguishable for audit and sensitivity analysis.

## Verification update

The v2.5 concentrated completion pass applied a conservative rule requiring a county
name or unique administrative-name stem and a source value within 1 yuan on the same
mapped source row. It newly verified 93 records without changing any training value.
The remaining 440 cases comprise 33 records whose mapped source file was unavailable,
304 for which the county was not located in the mapped extract, and 103 for which the
county was located but its value could not be reconciled. The previously frozen
experiment excluding all 533 then-pending records is retained as a conservative
sensitivity analysis; it removes a superset of the 440 records still pending in v2.5.
"""


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copytree(BASE, PACKAGE)

    metadata = PACKAGE / "metadata"
    old_files = [
        "county_training_label_coverage_summary_v2.4.csv",
        "county_training_label_province_year_provenance_public_v2.4.csv",
        "county_training_label_record_provenance_public_v2.4.csv",
        "county_training_label_source_items_public_v2.4.csv",
        "README_county_training_label_provenance_v2.4.md",
    ]
    for name in old_files:
        (metadata / name).unlink()

    copies = {
        "county_training_label_coverage_summary_v2.5.csv": "county_training_label_coverage_summary_v2.5.csv",
        "county_training_label_province_year_provenance_public_v2.5.csv": "county_training_label_province_year_provenance_public_v2.5.csv",
        "county_training_label_record_provenance_public_v2.5.csv": "county_training_label_record_provenance_public_v2.5.csv",
        "county_training_label_source_items_public_v2.5.csv": "county_training_label_source_items_public_v2.5.csv",
    }
    for source, destination in copies.items():
        shutil.copy2(PROV / source, metadata / destination)
    (metadata / "README_county_training_label_provenance_v2.5.md").write_text(
        build_provenance_readme(), encoding="utf-8", newline="\n"
    )

    release_summary_path = metadata / "release_summary_v1.5.json"
    summary = json.loads(release_summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "version": "v1.5.3-no-road",
            "value_verified_records": 7933,
            "value_verified_rate": 7933 / 8373,
            "value_pending_records": 440,
            "metadata_revision": "provenance-v2.5-concentrated-completion",
        }
    )
    provenance = summary["training_label_provenance"]
    provenance.update(
        {
            "numeric_verified_count": 7933,
            "numeric_pending_count": 440,
            "source_items_with_exact_table_or_section": 95,
            "source_items_with_url": 40,
            "labels_changed_in_v2_5_completion": 0,
        }
    )
    summary["pending_533_label_sensitivity"]["interpretation_after_v2_5"] = (
        "Conservative frozen sensitivity excluding a superset of the 440 records "
        "still pending numeric verification after the v2.5 completion pass."
    )
    release_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    replacements = [
        ("v1.5-no-road", "v1.5.3-no-road"),
        ("7,840", "7,933"),
        ("533 remain explicitly flagged as numeric-verification\npending", "440 remain explicitly flagged as numeric-verification\npending"),
        ("See `metadata/README_county_training_label_provenance_v2.4.md`.", "See `metadata/README_county_training_label_provenance_v2.5.md`."),
    ]
    replace_text(PACKAGE / "README.md", replacements)

    citation = PACKAGE / "CITATION.cff"
    if citation.exists():
        replace_text(citation, [("version: v1.5-no-road", "version: v1.5.3-no-road")])

    MANIFEST.unlink()
    files = sorted(p for p in PACKAGE.rglob("*") if p.is_file())
    if len(files) != 155:
        raise RuntimeError(f"Expected 155 files excluding manifest, found {len(files)}")
    manifest = {
        "version": "v1.5.3-no-road",
        "metadata_revision": "provenance-v2.5-concentrated-completion-20260823",
        "release_status": "formal_frozen_peer_review_package",
        "package_root": PACKAGE_NAME,
        "file_count_excluding_manifest": 155,
        "file_count_including_manifest": 156,
        "geotiff_count": sum(p.suffix.lower() == ".tif" for p in files),
        "files": [
            {
                "path": p.relative_to(PACKAGE).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
            }
            for p in files
        ],
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    all_files = [p for p in PACKAGE.rglob("*") if p.is_file()]
    counts = {
        "total_files": len(all_files),
        "geotiff_files": sum(p.suffix.lower() == ".tif" for p in all_files),
        "data_files": sum(1 for p in (PACKAGE / "data").rglob("*") if p.is_file()),
        "tables_files": sum(1 for p in (PACKAGE / "tables").rglob("*") if p.is_file()),
        "metadata_files": sum(1 for p in metadata.rglob("*") if p.is_file()),
        "root_files": sum(1 for p in PACKAGE.iterdir() if p.is_file()),
    }
    expected = {
        "total_files": 156,
        "geotiff_files": 67,
        "data_files": 70,
        "tables_files": 46,
        "metadata_files": 36,
        "root_files": 4,
    }
    if counts != expected:
        raise RuntimeError(f"Package-count mismatch: {counts} != {expected}")

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(all_files):
            archive.write(path, f"{PACKAGE_NAME}/{path.relative_to(PACKAGE).as_posix()}")

    audit = {
        "status": "PASS",
        "version": "v1.5.3-no-road",
        "package_counts": counts,
        "manifest_entries": len(manifest["files"]),
        "manifest_self_excluded": True,
        "training_labels": {
            "total": 8373,
            "source_mapped": 8373,
            "numeric_verified": 7933,
            "numeric_pending": 440,
            "exact_table_or_section_source_items": 95,
            "source_items_with_url": 40,
            "labels_changed": 0,
        },
        "zip": {
            "name": ZIP_PATH.name,
            "bytes": ZIP_PATH.stat().st_size,
            "sha256": sha256(ZIP_PATH),
        },
    }
    (OUT / "FINAL_RELEASE_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "SHA256SUMS_v1.5.3-no-road.txt").write_text(
        f"{audit['zip']['sha256']}  {ZIP_PATH.name}\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
