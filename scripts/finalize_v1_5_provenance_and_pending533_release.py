"""Add the aggregate pending-533 sensitivity evidence to the audited release."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs" / "public_release_v1_5_no_road_provenance_audited"
PACKAGE_NAME = "china_rural_income_1km_2019_2023_v1.5-no-road"
PACKAGE = OUT_ROOT / PACKAGE_NAME
ZIP_PATH = OUT_ROOT / f"{PACKAGE_NAME}-provenance-audited.zip"
CHECKSUM_PATH = OUT_ROOT / "SHA256SUMS_v1.5-no-road-provenance-audited.txt"
AUDIT_PATH = OUT_ROOT / "PROVENANCE_RELEASE_AUDIT.json"
SENSITIVITY = ROOT / "outputs" / "stage31_pending_533_label_sensitivity"
BASE_PACKAGE = (
    ROOT
    / "outputs"
    / "public_release_v1_5_no_road_final"
    / PACKAGE_NAME
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_summary() -> Path:
    overall = read_csv(SENSITIVITY / "pending_533_overall_metrics.csv")
    decision = json.loads(
        (SENSITIVITY / "pending_533_sensitivity_decision.json").read_text(encoding="utf-8")
    )
    comparison = decision["prediction_comparison"]
    deltas = decision["metric_deltas"]
    lines = [
        "# Sensitivity analysis excluding 533 labels pending numeric verification",
        "",
        "## Design",
        "",
        "The frozen v1.5-no-road baseline retained all 8,373 county-year labels. The",
        "sensitivity variant used only the 7,840 labels whose county names and numeric",
        "values had been verified against the linked source item. Province folds,",
        "features, pixel-sampling rules, model settings and random seeds were held fixed.",
        "Both variants were scored on the same 7,753 verified county-year records with",
        "valid rural-pixel panels. Pending records were excluded from scoring in both",
        "variants.",
        "",
        "## Overall metrics",
        "",
        "| Variant | n | R2 | MAE | RMSE | MAPE (%) | Bias |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            f"| {row['variant']} | {int(row['n']):,} | {float(row['r2']):.4f} | "
            f"{float(row['mae']):.1f} | {float(row['rmse']):.1f} | "
            f"{float(row['mape']):.2f} | {float(row['bias']):.1f} |"
        )
    lines.extend([
        "",
        "## Paired comparison",
        "",
        f"- Delta R2 (verified-only minus all labels): {deltas['delta_r2_verified_only_minus_all']:+.4f}.",
        f"- Delta MAPE: {deltas['delta_mape_percentage_points_verified_only_minus_all']:+.3f} percentage points.",
        f"- Pearson correlation between paired predictions: {comparison['pearson_prediction_correlation']:.5f}.",
        f"- Spearman correlation between paired predictions: {comparison['spearman_prediction_correlation']:.5f}.",
        f"- Mean absolute paired difference: {comparison['mean_absolute_difference']:.1f} 2019-CNY, "
        f"or {comparison['mean_absolute_difference_percent_of_baseline_mean_prediction']:.3f}% of the baseline mean prediction.",
        "",
        "## Interpretation",
        "",
        f"The pre-specified aggregate stability checks returned `{decision['status']}`.",
        "Excluding the 533 pending records caused only small changes in province-transfer",
        "validation metrics and very high paired prediction agreement. This result controls",
        "the risk associated with retaining source-mapped but not yet numerically verified",
        "training labels. It does not constitute pixel-level accuracy validation.",
    ])
    path = SENSITIVITY / "pending_533_sensitivity_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def copy_evidence(summary_path: Path) -> list[str]:
    table_files = [
        "pending_533_overall_metrics.csv",
        "pending_533_fold_metrics.csv",
        "pending_533_year_metrics.csv",
        "pending_533_training_counts_by_fold.csv",
    ]
    metadata_files = [
        "pending_533_prediction_comparison_summary.json",
        "pending_533_sensitivity_decision.json",
        "pending_533_run_manifest.json",
    ]
    copied = []
    for name in table_files:
        shutil.copy2(SENSITIVITY / name, PACKAGE / "tables" / name)
        copied.append(f"tables/{name}")
    for name in metadata_files:
        shutil.copy2(SENSITIVITY / name, PACKAGE / "metadata" / name)
        copied.append(f"metadata/{name}")
    destination = PACKAGE / "metadata" / "PENDING_533_LABEL_SENSITIVITY.md"
    shutil.copy2(summary_path, destination)
    copied.append("metadata/PENDING_533_LABEL_SENSITIVITY.md")
    return copied


def update_readmes() -> None:
    path = PACKAGE / "README.md"
    text = path.read_text(encoding="utf-8")
    marker = "## Pending-label sensitivity"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n"
    text += """

## Pending-label sensitivity

An additional fixed-fold sensitivity analysis excludes the 533 labels that have an
exact source mapping but remain pending record-level numeric verification. Both
variants are evaluated on the same 7,753 verified county-year observations with valid
pixel panels. Exclusion changes R2 by -0.0099 and MAPE by +0.183 percentage points;
paired predictions correlate at r = 0.99787. Aggregate results, fold/year metrics,
run settings and the decision record are included in `tables/` and `metadata/`.
County-level predictions and income values are not redistributed.
"""
    path.write_text(text, encoding="utf-8")

    summary_path = PACKAGE / "metadata" / "release_summary_v1.5.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["pending_533_label_sensitivity"] = {
        "included": True,
        "status": "sensitivity_pass",
        "baseline_label_count": 8373,
        "verified_only_label_count": 7840,
        "pending_label_count": 533,
        "common_verified_test_count": 7753,
        "delta_r2": -0.009913176187982375,
        "delta_mape_percentage_points": 0.18273568949070196,
        "paired_prediction_pearson_r": 0.9978723131080294,
        "county_values_or_predictions_redistributed": False,
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
        files.append({
            "path": item.relative_to(PACKAGE).as_posix(),
            "bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        })
    manifest = {
        "version": "v1.5-no-road",
        "metadata_revision": "provenance-audited-v2.4-plus-pending533-sensitivity",
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
    record_rows = read_csv(
        PACKAGE / "metadata" / "county_training_label_record_provenance_public_v2.4.csv"
    )
    source_rows = read_csv(
        PACKAGE / "metadata" / "county_training_label_source_items_public_v2.4.csv"
    )
    base_tifs = {
        item.relative_to(BASE_PACKAGE).as_posix(): sha256_file(item)
        for item in BASE_PACKAGE.rglob("*.tif")
    }
    new_tifs = {
        item.relative_to(PACKAGE).as_posix(): sha256_file(item)
        for item in PACKAGE.rglob("*.tif")
    }
    with zipfile.ZipFile(ZIP_PATH) as archive:
        bad = archive.testzip()
        zip_names = archive.namelist()
    forbidden_public_files = [
        item.relative_to(PACKAGE).as_posix()
        for item in PACKAGE.rglob("*")
        if item.is_file()
        and any(
            token in item.name.lower()
            for token in ["county_year_predictions", "residual", "training_target"]
        )
    ]
    checks = {
        "record_index_has_8373_rows": len(record_rows) == 8373,
        "source_catalogue_has_95_rows": len(source_rows) == 95,
        "all_geotiffs_byte_identical_to_previous_v1_5": base_tifs == new_tifs,
        "geotiff_count_is_67": len(new_tifs) == 67 == manifest["geotiff_count"],
        "sensitivity_evidence_files_present": all((PACKAGE / name).is_file() for name in copied),
        "no_county_predictions_or_training_targets_in_package": not forbidden_public_files,
        "zip_integrity_ok": bad is None,
        "zip_and_package_member_counts_match": len(zip_names) == manifest["file_count_including_manifest"],
    }
    if not all(checks.values()):
        raise RuntimeError({
            "checks": checks,
            "forbidden_public_files": forbidden_public_files,
            "bad_zip_member": bad,
        })
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
        "checks": checks,
    }
    AUDIT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    if not PACKAGE.exists():
        raise FileNotFoundError(PACKAGE)
    summary_path = write_summary()
    copied = copy_evidence(summary_path)
    update_readmes()
    manifest = write_manifest()
    write_zip()
    print(json.dumps(verify(copied, manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
