"""Stage public-safe v3 raster and aggregate validation products.

Restricted county targets and county-level prediction tables are deliberately
excluded. Existing source products are copied, never moved or overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
STAGE5 = ROOT / "outputs" / "official44_corrected_full_rerun_v3" / "stage5_final_maps"
VALIDATION = ROOT / "outputs" / "official44_corrected_full_rerun_v3" / "stage3_weak_supervision" / "spatiotemporal_validation"
PACKAGE = ROOT / "outputs" / "official44_corrected_stage8_v3" / "package"
YEARS = range(2019, 2024)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_checked(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise RuntimeError(f"Destination exists with different content: {destination}")
        return
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage5-root", type=Path, default=STAGE5)
    parser.add_argument("--validation-root", type=Path, default=VALIDATION)
    parser.add_argument("--package-root", type=Path, default=PACKAGE)
    args = parser.parse_args()
    stage5 = args.stage5_root.resolve()
    validation = args.validation_root.resolve()
    package = args.package_root.resolve()

    copies: list[tuple[Path, Path]] = []
    for year in YEARS:
        copies.extend([
            (
                stage5 / f"rural_income_{year}_final_calibrated_unbounded_2019yuan_strict.tif",
                package / "data" / "income_maps" / f"china_rural_income_1km_{year}_2019yuan_strict_unbounded.tif",
            ),
            (
                stage5 / f"rural_income_{year}_final_calibrated_2019yuan_strict.tif",
                package / "data" / "sensitivity_bounded_maps" / f"china_rural_income_1km_{year}_2019yuan_strict_bounded_3000_80000.tif",
            ),
        ])
    copies.extend([
        (
            stage5 / "rural_income_change_2019_2023_absolute_unbounded_2019yuan.tif",
            package / "data" / "change_maps" / "china_rural_income_change_1km_2019_2023_absolute_2019yuan_strict_unbounded.tif",
        ),
        (
            stage5 / "rural_income_change_2019_2023_percent_unbounded.tif",
            package / "data" / "change_maps" / "china_rural_income_change_1km_2019_2023_percent_strict_unbounded.tif",
        ),
        (
            stage5 / "rural_income_change_2019_2023_absolute_2019yuan.tif",
            package / "data" / "sensitivity_bounded_maps" / "china_rural_income_change_1km_2019_2023_absolute_2019yuan_strict_bounded.tif",
        ),
        (
            stage5 / "rural_income_change_2019_2023_percent.tif",
            package / "data" / "sensitivity_bounded_maps" / "china_rural_income_change_1km_2019_2023_percent_strict_bounded.tif",
        ),
    ])

    aggregate_validation = [
        "leave_year_2019_2023_strict_overall_metrics.json",
        "leave_year_2019_2023_strict_year_metrics.csv",
        "leave_province_2020_strict_overall_metrics.json",
        "leave_province_2020_strict_fold_metrics.csv",
        "leave_province_2020_strict_manifest.json",
    ]
    for name in aggregate_validation:
        copies.append((validation / name, package / "tables" / name))
    copies.append((
        ROOT / "outputs" / "stage6_scientific_data_product" / "PUBLIC_SOURCE_DATA_CITATIONS_AND_LICENSES_FILLED.md",
        package / "PUBLIC_SOURCE_DATA_CITATIONS_AND_LICENSES.md",
    ))

    for source, destination in copies:
        copy_checked(source, destination)

    records = []
    manifest_path = package / "metadata" / "v3_release_candidate_manifest.json"
    for path in sorted(p for p in package.rglob("*") if p.is_file() and p != manifest_path):
        records.append({
            "path": str(path.relative_to(package)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = {
        "version": "v1.3-official44-corrected-release-candidate",
        "package_root": str(package),
        "files": records,
        "restricted_county_targets_included": False,
        "county_level_predictions_included": False,
        "source_products_overwritten": False,
    }
    metadata = package / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"copied_or_verified": len(copies), "package_files": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
