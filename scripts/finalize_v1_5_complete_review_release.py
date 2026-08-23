"""Build the final v1.5-no-road peer-review package with aggregate robustness evidence.

The script preserves the published 67 GeoTIFF products byte-for-byte. It adds only
aggregate county-uniform and parameter-sensitivity outputs, run manifests and fixed-fold
metrics. Variant rasters, county-level predictions and training targets stay internal.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "outputs" / "public_release_v1_5_no_road_provenance_audited"
PACKAGE_NAME = "china_rural_income_1km_2019_2023_v1.5-no-road"
SOURCE_PACKAGE = SOURCE_ROOT / PACKAGE_NAME
OUT_ROOT = ROOT / "outputs" / "public_release_v1_5_no_road_final_review_complete"
PACKAGE = OUT_ROOT / PACKAGE_NAME
ZIP_PATH = OUT_ROOT / f"{PACKAGE_NAME}.zip"
CHECKSUM_PATH = OUT_ROOT / "SHA256SUMS_v1.5-no-road.txt"
AUDIT_PATH = OUT_ROOT / "FINAL_RELEASE_AUDIT.json"
STAGE32 = ROOT / "outputs" / "stage32_v1_5_no_road_spatial_robustness"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def initialise() -> None:
    if not SOURCE_PACKAGE.is_dir():
        raise FileNotFoundError(SOURCE_PACKAGE)
    if OUT_ROOT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing final output: {OUT_ROOT}. "
            "Move it aside before rebuilding."
        )
    OUT_ROOT.mkdir(parents=True)
    shutil.copytree(SOURCE_PACKAGE, PACKAGE)


def copy_aggregate_evidence() -> list[str]:
    tables = PACKAGE / "tables"
    metadata = PACKAGE / "metadata"
    copied: list[str] = []

    root_tables = [
        "parameter_sensitivity_cv_summary.csv",
        "parameter_sensitivity_annual_map_summary.csv",
        "parameter_sensitivity_change_map_summary.csv",
    ]
    for name in root_tables:
        shutil.copy2(STAGE32 / name, tables / name)
        copied.append(f"tables/{name}")

    uniform_source = (
        STAGE32 / "county_uniform_baseline" / "county_uniform_baseline_summary_public.csv"
    )
    uniform_destination = tables / "county_uniform_baseline_summary_2019_2023.csv"
    shutil.copy2(uniform_source, uniform_destination)
    copied.append("tables/county_uniform_baseline_summary_2019_2023.csv")

    variants = [
        "regularization_0p0001_pixels128",
        "regularization_0p01_pixels128",
        "regularization_0p001_pixels64",
        "regularization_0p001_pixels256",
    ]
    aggregate_names = [
        "fold_metrics.csv",
        "fold_year_metrics.csv",
        "year_metrics.csv",
        "training_counts_by_fold.csv",
    ]
    for variant in variants:
        source = STAGE32 / "parameter_sensitivity" / variant
        for name in aggregate_names:
            destination_name = f"sensitivity_{variant}_{name}"
            shutil.copy2(source / name, tables / destination_name)
            copied.append(f"tables/{destination_name}")
        overall_name = f"sensitivity_{variant}_overall_metrics.json"
        shutil.copy2(source / "multiyear_leave_province_overall_metrics.json", metadata / overall_name)
        copied.append(f"metadata/{overall_name}")
        run_name = f"sensitivity_{variant}_run_manifest.json"
        shutil.copy2(source / "cv_run_manifest.json", metadata / run_name)
        copied.append(f"metadata/{run_name}")

    metadata_sources = {
        STAGE32 / "county_uniform_baseline" / "county_uniform_baseline_manifest.json":
            "county_uniform_baseline_manifest.json",
        STAGE32 / "spatial_robustness_decision.json": "spatial_robustness_decision.json",
        STAGE32 / "spatial_robustness_run_manifest.json": "spatial_robustness_run_manifest.json",
        STAGE32 / "SPATIAL_ROBUSTNESS_SUMMARY.md": "SPATIAL_ROBUSTNESS_SUMMARY.md",
    }
    for source, destination_name in metadata_sources.items():
        shutil.copy2(source, metadata / destination_name)
        copied.append(f"metadata/{destination_name}")
    return copied


def update_descriptive_metadata() -> None:
    readme_path = PACKAGE / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    marker = "## County-uniform baseline and parameter sensitivity"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n"
    text += """

## County-uniform baseline and parameter sensitivity

The county-uniform baseline preserves the population-weighted mean of every county
while removing within-county pixel differences. The canonical machine-learning surface
has a median within-county population-weighted coefficient of variation of 24.55-25.52%
across 2019-2023. This comparison describes the additional within-county spatial
structure in the gridded product; it is not independent 1 km accuracy validation.

The v1.5 no-road workflow was also rerun with output-regularisation coefficients of
0.0001 and 0.01 and with per-county pixel-sampling caps of 64, 128 and 256. Province
folds, random seeds, features and provincial calibration rules were fixed. Aggregate
cross-validation metrics, annual/change-map comparisons and run manifests are included
in `tables/` and `metadata/`. County-level predictions and variant rasters are not
redistributed. County-level validation was stable, whereas within-county pixel ordering
and especially 2019-2023 change patterns were more sensitive to the pixel-sampling cap.
Users should therefore avoid interpreting individual pixels as independently validated
income observations and should use the change maps together with quality layers.
"""
    readme_path.write_text(text, encoding="utf-8")

    summary_path = PACKAGE / "metadata" / "release_summary_v1.5.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    decision = json.loads(
        (STAGE32 / "spatial_robustness_decision.json").read_text(encoding="utf-8")
    )
    summary["county_uniform_and_parameter_sensitivity"] = {
        "included": True,
        "workflow": "v1.5-no-road",
        "county_uniform_is_pixel_accuracy_validation": False,
        "regularization_values": [0.0001, 0.001, 0.01],
        "pixels_per_county_values": [64, 128, 256],
        "maximum_absolute_delta_r2": decision["checks"]["maximum_absolute_delta_r2"],
        "maximum_absolute_delta_mape_percentage_points": decision["checks"][
            "maximum_absolute_delta_mape_points"
        ],
        "minimum_annual_pixel_spearman": decision["checks"][
            "minimum_annual_pixel_spearman"
        ],
        "minimum_change_map_spearman": decision["checks"]["minimum_change_map_spearman"],
        "variant_rasters_redistributed": False,
        "county_level_predictions_redistributed": False,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_manifest() -> dict[str, object]:
    path = PACKAGE / "metadata" / "v1_5_no_road_manifest.json"
    files = []
    for item in sorted(PACKAGE.rglob("*")):
        if not item.is_file() or item == path:
            continue
        files.append(
            {
                "path": item.relative_to(PACKAGE).as_posix(),
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    manifest = {
        "version": "v1.5-no-road",
        "metadata_revision": "review-evidence-complete-20260823",
        "release_status": "formal_frozen_peer_review_package",
        "package_root": PACKAGE_NAME,
        "file_count_excluding_manifest": len(files),
        "file_count_including_manifest": len(files) + 1,
        "geotiff_count": sum(row["path"].lower().endswith(".tif") for row in files),
        "files": files,
        "restricted_county_targets_included": False,
        "county_income_values_in_public_provenance": False,
        "county_level_predictions_included": False,
        "record_level_public_provenance_included": True,
        "pending_label_sensitivity_evidence_included": True,
        "county_uniform_aggregate_evidence_included": True,
        "parameter_sensitivity_aggregate_evidence_included": True,
        "variant_sensitivity_rasters_included": False,
        "administrative_assignment_boundaries_included": False,
        "road_density_included": False,
        "source_products_overwritten": False,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


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


def verify(copied: list[str], manifest: dict[str, object]) -> dict[str, object]:
    source_tifs = {
        item.relative_to(SOURCE_PACKAGE).as_posix(): sha256_file(item)
        for item in SOURCE_PACKAGE.rglob("*.tif")
    }
    final_tifs = {
        item.relative_to(PACKAGE).as_posix(): sha256_file(item)
        for item in PACKAGE.rglob("*.tif")
    }
    record_rows = read_csv(
        PACKAGE / "metadata" / "county_training_label_record_provenance_public_v2.4.csv"
    )
    source_rows = read_csv(
        PACKAGE / "metadata" / "county_training_label_source_items_public_v2.4.csv"
    )
    with zipfile.ZipFile(ZIP_PATH) as archive:
        bad = archive.testzip()
        zip_names = archive.namelist()
    forbidden = [
        item.relative_to(PACKAGE).as_posix()
        for item in PACKAGE.rglob("*")
        if item.is_file()
        and any(
            token in item.name.lower()
            for token in [
                "county_year_predictions",
                "county_spatial_stability_internal",
                "training_target",
                "variant_map",
            ]
        )
    ]
    release_candidate_hits = []
    for item in PACKAGE.rglob("*"):
        if item.is_file() and item.suffix.lower() in {".md", ".json", ".csv", ".txt", ".cff"}:
            try:
                content = item.read_text(encoding="utf-8").lower()
            except UnicodeDecodeError:
                continue
            if "release_candidate" in content or "release candidate" in content:
                release_candidate_hits.append(item.relative_to(PACKAGE).as_posix())
    checks = {
        "record_index_has_8373_rows": len(record_rows) == 8373,
        "source_catalogue_has_95_rows": len(source_rows) == 95,
        "all_67_geotiffs_byte_identical_to_prior_v1_5": source_tifs == final_tifs and len(final_tifs) == 67,
        "aggregate_evidence_present": all((PACKAGE / name).is_file() for name in copied),
        "restricted_outputs_absent": not forbidden,
        "release_candidate_wording_absent": not release_candidate_hits,
        "zip_integrity_ok": bad is None,
        "zip_and_manifest_member_counts_match": len(zip_names) == manifest["file_count_including_manifest"],
    }
    if not all(checks.values()):
        raise RuntimeError(
            {
                "checks": checks,
                "forbidden": forbidden,
                "release_candidate_hits": release_candidate_hits,
                "bad_zip_member": bad,
            }
        )
    result = {
        "package": str(PACKAGE),
        "zip": str(ZIP_PATH),
        "zip_bytes": ZIP_PATH.stat().st_size,
        "zip_sha256": sha256_file(ZIP_PATH),
        "package_file_count": manifest["file_count_including_manifest"],
        "geotiff_count": manifest["geotiff_count"],
        "record_provenance_rows": len(record_rows),
        "source_items": len(source_rows),
        "pending_533_sensitivity_status": "sensitivity_pass",
        "county_uniform_and_parameter_sensitivity_included": True,
        "checks": checks,
    }
    AUDIT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    initialise()
    copied = copy_aggregate_evidence()
    update_descriptive_metadata()
    manifest = write_manifest()
    write_zip()
    print(json.dumps(verify(copied, manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
