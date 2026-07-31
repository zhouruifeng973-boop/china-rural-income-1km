"""Build a conservative public-release package for v1.0-draft.

Default mode is dry-run: the script prints what would be copied or excluded but
does not create the release directory. Use --execute to copy files.

The release policy is intentionally conservative. Final rasters, documentation,
metadata, and aggregate validation summaries are included. County-level tables
that may reveal purchased/third-party income labels are excluded until the user
clears their redistribution rights.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "outputs" / "stage6_scientific_data_product"
DEFAULT_DEST = ROOT / "outputs" / "public_release" / "china_rural_income_1km_2019_2023_v1.0-draft"


EXCLUDE_RULES: list[tuple[str, str]] = [
    (
        "tables/county_training_income_2019_2023_2019yuan.csv",
        "Potentially contains county-level purchased/third-party income labels.",
    ),
    (
        "tables/validation_leave_province_2020_county_residuals.csv",
        "County-level validation residuals may reveal purchased/third-party income labels.",
    ),
    (
        "tables/validation_leave_year_county_residuals.csv",
        "County-level validation residuals may reveal purchased/third-party income labels.",
    ),
    (
        "**/__pycache__/**",
        "Python cache file.",
    ),
    (
        "**/*.pyc",
        "Python bytecode cache.",
    ),
    (
        "**/*.lock",
        "Local lock/cache file.",
    ),
]


@dataclass
class ManifestRow:
    action: str
    relative_path: str
    bytes: int
    sha256: str
    reason: str


def normalize_relative(path: Path, source: Path) -> str:
    return path.relative_to(source).as_posix()


def matches(pattern: str, relative_path: str) -> bool:
    return fnmatch.fnmatch(relative_path, pattern) or fnmatch.fnmatch(relative_path, pattern.replace("/", "\\"))


def exclusion_reason(relative_path: str) -> str | None:
    for pattern, reason in EXCLUDE_RULES:
        if matches(pattern, relative_path):
            return reason
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(source: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for file in sorted(source.rglob("*")):
        if not file.is_file():
            continue
        relative_path = normalize_relative(file, source)
        reason = exclusion_reason(relative_path)
        size = file.stat().st_size
        checksum = sha256_file(file)
        if reason:
            rows.append(
                ManifestRow(
                    action="exclude",
                    relative_path=relative_path,
                    bytes=size,
                    sha256=checksum,
                    reason=reason,
                )
            )
        else:
            rows.append(
                ManifestRow(
                    action="include",
                    relative_path=relative_path,
                    bytes=size,
                    sha256=checksum,
                    reason="Allowed by public release policy.",
                )
            )
    return rows


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_readme(rows: list[ManifestRow], dest: Path, source: Path) -> None:
    included = [r for r in rows if r.action == "include"]
    excluded = [r for r in rows if r.action == "exclude"]
    total_bytes = sum(r.bytes for r in included)
    excluded_bytes = sum(r.bytes for r in excluded)

    lines: list[str] = []
    lines.append("# Public release package manifest")
    lines.append("")
    lines.append(f"- Built at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Source product: `{source}`")
    lines.append(f"- Release directory: `{dest}`")
    lines.append(f"- Included files: {len(included)}")
    lines.append(f"- Excluded files: {len(excluded)}")
    lines.append(f"- Included size: {total_bytes / 1024 / 1024:.2f} MB")
    lines.append(f"- Excluded size: {excluded_bytes / 1024 / 1024:.2f} MB")
    lines.append("")
    lines.append("This public package was built with a conservative redistribution policy.")
    lines.append("County-level income labels and county-level residual tables are excluded until redistribution rights are cleared.")
    lines.append("")
    lines.append("## Excluded files")
    lines.append("")
    if not excluded:
        lines.append("No files were excluded.")
    else:
        lines.append("| File | Reason |")
        lines.append("|---|---|")
        for row in excluded:
            lines.append(f"| `{row.relative_path}` | {row.reason} |")
    lines.append("")
    lines.append("## Checksums")
    lines.append("")
    lines.append("See `metadata/public_release_manifest_v1.0_draft.csv` for file sizes and SHA-256 hashes.")
    lines.append("")

    (dest / "PUBLIC_RELEASE_README_v1.0_draft.md").write_text("\n".join(lines), encoding="utf-8")


def copy_included_files(rows: list[ManifestRow], source: Path, dest: Path, overwrite: bool) -> None:
    for row in rows:
        if row.action != "include":
            continue
        src = source / row.relative_path
        dst = dest / row.relative_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not overwrite:
            continue
        shutil.copy2(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Frozen data product directory.")
    parser.add_argument("--dest", default=str(DEFAULT_DEST), help="Public release output directory.")
    parser.add_argument("--execute", action="store_true", help="Copy included files and write release manifest.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in --dest when executing.")
    parser.add_argument("--show-files", action="store_true", help="Print every include/exclude decision.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    dest = Path(args.dest)
    if not source.is_absolute():
        source = ROOT / source
    if not dest.is_absolute():
        dest = ROOT / dest
    if not source.is_dir():
        raise SystemExit(f"Source directory not found: {source}")

    rows = build_manifest(source)
    included = [r for r in rows if r.action == "include"]
    excluded = [r for r in rows if r.action == "exclude"]
    included_bytes = sum(r.bytes for r in included)
    excluded_bytes = sum(r.bytes for r in excluded)

    print("Public release package builder")
    print(f"  source: {source}")
    print(f"  dest: {dest}")
    print(f"  mode: {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print(f"  include files: {len(included)} ({included_bytes / 1024 / 1024:.2f} MB)")
    print(f"  exclude files: {len(excluded)} ({excluded_bytes / 1024 / 1024:.2f} MB)")

    if args.show_files:
        for row in rows:
            print(f"  {row.action.upper():7} {row.relative_path} -- {row.reason}")

    if not args.execute:
        print("  dry-run only; add --execute to build the package.")
        return

    if dest.exists() and any(dest.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Destination exists and is not empty: {dest}\n"
            "Use --overwrite to update existing files, or choose a new --dest."
        )

    copy_included_files(rows, source, dest, args.overwrite)
    manifest_path = dest / "metadata" / "public_release_manifest_v1.0_draft.csv"
    write_manifest(rows, manifest_path)
    write_readme(rows, dest, source)
    print(f"  wrote manifest: {manifest_path}")
    print(f"  wrote readme: {dest / 'PUBLIC_RELEASE_README_v1.0_draft.md'}")


if __name__ == "__main__":
    main()
