"""Apply verified Guangzhou/Luoyang workbook corrections to a new label version."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    args = parser.parse_args()

    targets = pd.read_csv(args.source)
    changes = pd.read_csv(args.corrections)
    if len(changes) != 15:
        raise RuntimeError(f"Expected 15 corrections, found {len(changes)}")
    if changes[["year", "county_code"]].duplicated().any():
        raise RuntimeError("Duplicate correction keys")
    if set(changes["verification_status"]) != {"VERIFIED_OFFICIAL_WORKBOOK"}:
        raise RuntimeError("Correction verification status is incomplete")

    targets["county_code"] = targets["county_code"].astype(int)
    changes["county_code"] = changes["county_code"].astype(int)
    indexed = targets.set_index(["year", "county_code"], drop=False)
    applied = []
    for change in changes.itertuples(index=False):
        key = (int(change.year), int(change.county_code))
        if key not in indexed.index:
            raise RuntimeError(f"Target record missing: {key}")
        old = float(indexed.loc[key, "rural_income_nominal"])
        if abs(old - float(change.old_rural_income_nominal)) > 0.01:
            raise RuntimeError(f"Old-value guard failed for {key}")
        if str(indexed.loc[key, "county"]) != str(change.county):
            raise RuntimeError(f"County-name guard failed for {key}")
        corrected = float(change.corrected_rural_income_nominal)
        price_index = float(indexed.loc[key, "price_index_2019_base"])
        indexed.loc[key, "rural_income_nominal"] = corrected
        indexed.loc[key, "rural_income_2019yuan"] = corrected / price_index
        applied.append({
            "year": key[0], "county_code": key[1], "county": change.county,
            "city": change.city, "province": change.province,
            "old_rural_income_nominal": old,
            "corrected_rural_income_nominal": corrected,
            "change_yuan": corrected - old,
            "corrected_rural_income_2019yuan": corrected / price_index,
            "source_table": change.source_table,
            "verification_status": change.verification_status,
        })

    result = indexed.reset_index(drop=True).sort_values(["year", "county_code"])
    if len(result) != len(targets):
        raise RuntimeError("Output row count changed")
    if result[["year", "county_code"]].duplicated().any():
        raise RuntimeError("Output key uniqueness failed")
    if result[["rural_income_nominal", "rural_income_2019yuan"]].isna().any().any():
        raise RuntimeError("Output contains missing income values")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    audit_path = args.audit_dir / "guangzhou_luoyang_applied_corrections.csv"
    pd.DataFrame(applied).to_csv(audit_path, index=False, encoding="utf-8-sig")
    evidence = [{"path": str(path), "sha256": sha256(path)} for path in args.evidence]
    manifest = {
        "version": "v3-official44-corrected",
        "parent": str(args.source),
        "parent_sha256": sha256(args.source),
        "corrections": str(args.corrections),
        "corrections_sha256": sha256(args.corrections),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "audit": str(audit_path),
        "audit_sha256": sha256(audit_path),
        "corrections_applied": len(applied),
        "evidence_workbooks": evidence,
        "preservation_note": "The v2 target and all source workbooks were preserved unchanged.",
    }
    (args.audit_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
