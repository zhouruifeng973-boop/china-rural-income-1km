#!/usr/bin/env python3
"""Build and independently audit the v1.3 public release ZIP.

This stage is intentionally standard-library only.  It creates a stable archive
layout, calculates its SHA-256 digest, extracts it into an isolated directory,
and compares every extracted file with the staged public release candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "outputs" / "official44_corrected_stage8_v3" / "package"
DEFAULT_RELEASE_DIR = ROOT / "outputs" / "public_release_v3"
DEFAULT_AUDIT_DIR = ROOT / "outputs" / "stage23_v3_zip_audit"
ARCHIVE_STEM = "china_rural_income_1km_2019_2023_v1.3-official44-corrected"
FIXED_ZIP_TIME = (2026, 8, 1, 0, 0, 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(directory: Path) -> dict[str, dict[str, int | str]]:
    result: dict[str, dict[str, int | str]] = {}
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        rel = path.relative_to(directory).as_posix()
        result[rel] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def safely_reset_owned_directory(path: Path, allowed_parent: Path) -> None:
    resolved = path.resolve()
    parent = allowed_parent.resolve()
    if resolved == parent or parent not in resolved.parents:
        raise RuntimeError(f"Refusing to reset path outside owned audit directory: {resolved}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_zip(source: Path, zip_path: Path, files: list[str], force: bool) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        if not force:
            raise FileExistsError(f"Archive already exists (use --force): {zip_path}")
        zip_path.unlink()

    with zipfile.ZipFile(
        zip_path,
        "x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for rel in files:
            src = source / Path(rel)
            name = f"{ARCHIVE_STEM}/{rel}"
            info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            with src.open("rb") as reader, archive.open(info, "w", force_zip64=True) as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)


def validate_member_name(name: str) -> bool:
    posix = PurePosixPath(name)
    return not posix.is_absolute() and ".." not in posix.parts and "\\" not in name


def audit_internal_manifest(extracted_package: Path) -> tuple[bool, int, list[str]]:
    manifest_path = extracted_package / "metadata" / "v3_release_candidate_manifest.json"
    if not manifest_path.exists():
        return False, 0, ["metadata/v3_release_candidate_manifest.json is missing"]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("files", [])
    errors: list[str] = []
    for entry in entries:
        rel = entry.get("relative_path") or entry.get("path")
        expected = entry.get("sha256")
        if not rel or not expected:
            errors.append(f"Malformed manifest entry: {entry!r}")
            continue
        target = extracted_package / Path(rel)
        if not target.is_file():
            errors.append(f"Missing manifest file: {rel}")
        elif sha256_file(target) != expected:
            errors.append(f"Manifest hash mismatch: {rel}")
    return not errors, len(entries), errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    release_dir = args.release_dir.resolve()
    audit_dir = args.audit_dir.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Release candidate package not found: {source}")

    source_inventory = inventory(source)
    if not source_inventory:
        raise RuntimeError("Release candidate package is empty")

    zip_path = release_dir / f"{ARCHIVE_STEM}.zip"
    checksum_path = release_dir / "SHA256SUMS_v1.3.txt"
    if checksum_path.exists() and not args.force:
        raise FileExistsError(f"Checksum file already exists (use --force): {checksum_path}")

    build_zip(source, zip_path, list(source_inventory), args.force)
    zip_sha256 = sha256_file(zip_path)
    checksum_path.write_text(f"{zip_sha256}  {zip_path.name}\n", encoding="ascii")

    audit_dir.mkdir(parents=True, exist_ok=True)
    extract_root = audit_dir / "extracted"
    safely_reset_owned_directory(extract_root, audit_dir)

    with zipfile.ZipFile(zip_path, "r") as archive:
        names = archive.namelist()
        bad_crc_member = archive.testzip()
        safe_paths = all(validate_member_name(name) for name in names)
        if not safe_paths:
            raise RuntimeError("Archive contains an unsafe member path")
        expected_prefix = f"{ARCHIVE_STEM}/"
        prefix_ok = all(name.startswith(expected_prefix) for name in names)
        if not prefix_ok:
            raise RuntimeError("Archive layout does not use the expected root directory")
        for member in archive.infolist():
            destination = (extract_root / Path(member.filename)).resolve()
            if extract_root.resolve() not in destination.parents:
                raise RuntimeError(f"Unsafe extraction target: {member.filename}")
        archive.extractall(extract_root)

    extracted_package = extract_root / ARCHIVE_STEM
    extracted_inventory = inventory(extracted_package)
    missing = sorted(set(source_inventory) - set(extracted_inventory))
    extra = sorted(set(extracted_inventory) - set(source_inventory))
    mismatches = sorted(
        rel
        for rel in set(source_inventory) & set(extracted_inventory)
        if source_inventory[rel] != extracted_inventory[rel]
    )
    internal_manifest_ok, manifest_entry_count, manifest_errors = audit_internal_manifest(
        extracted_package
    )

    checks = {
        "zip_crc_integrity": bad_crc_member is None,
        "safe_member_paths": safe_paths,
        "expected_archive_root": prefix_ok,
        "archive_member_count_matches_source": len(names) == len(source_inventory),
        "no_missing_extracted_files": not missing,
        "no_extra_extracted_files": not extra,
        "all_extracted_hashes_and_sizes_match": not mismatches,
        "internal_release_manifest_matches": internal_manifest_ok,
        "checksum_sidecar_matches_zip": checksum_path.read_text(encoding="ascii").split()[0]
        == zip_sha256,
    }
    ready = all(checks.values())
    generated = datetime.now(timezone.utc).isoformat()
    report = {
        "generated_at_utc": generated,
        "source_package": str(source),
        "archive": str(zip_path),
        "archive_size_bytes": zip_path.stat().st_size,
        "archive_sha256": zip_sha256,
        "checksum_file": str(checksum_path),
        "source_file_count": len(source_inventory),
        "archive_member_count": len(names),
        "extracted_file_count": len(extracted_inventory),
        "internal_manifest_entry_count": manifest_entry_count,
        "checks": checks,
        "missing_files": missing,
        "extra_files": extra,
        "mismatched_files": mismatches,
        "internal_manifest_errors": manifest_errors,
        "ready_for_zenodo_upload": ready,
    }
    (audit_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    status = "PASS" if ready else "FAIL"
    lines = [
        "# Stage 23 — v1.3 ZIP build and extraction audit",
        "",
        f"- Overall status: **{status}**",
        f"- Ready for Zenodo upload: **{str(ready).lower()}**",
        f"- Source files: {len(source_inventory)}",
        f"- Archive members: {len(names)}",
        f"- Extracted files: {len(extracted_inventory)}",
        f"- ZIP size: {zip_path.stat().st_size:,} bytes",
        f"- ZIP SHA-256: `{zip_sha256}`",
        f"- Archive: `{zip_path}`",
        f"- Extracted audit copy: `{extracted_package}`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- [{'x' if passed else ' '}] {name}" for name, passed in checks.items())
    lines += [
        "",
        "## Exceptions",
        "",
        f"- Missing: {len(missing)}",
        f"- Extra: {len(extra)}",
        f"- Hash/size mismatches: {len(mismatches)}",
        f"- Internal manifest errors: {len(manifest_errors)}",
        "",
        "This stage builds and audits the local release artifact only; it does not upload or publish it.",
    ]
    (audit_dir / "AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
