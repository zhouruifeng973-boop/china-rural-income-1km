"""Reconcile official Guangzhou, Foshan and Luoyang workbooks with labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


GUANGZHOU_NAMES = {
    "白云": "白云区", "黄埔": "黄埔区", "番禺": "番禺区",
    "花都": "花都区", "南沙": "南沙区", "从化": "从化区", "增城": "增城区",
}


def extract_guangzhou(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Sheet1", header=None)
    years = [int(value) for value in raw.iloc[4, 2:7]]
    records = []
    for _, row in raw.iloc[6:18].iterrows():
        county = GUANGZHOU_NAMES.get(str(row.iloc[0]).strip())
        if not county:
            continue
        for offset, year in enumerate(years, start=2):
            value = pd.to_numeric(row.iloc[offset], errors="coerce")
            if pd.notna(value):
                records.append({"year": year, "county": county, "official_income": float(value)})
    result = pd.DataFrame(records)
    result["city"] = "广州市"
    result["province"] = "广东省"
    result["source_table"] = "广州统计年鉴2024 表8-11"
    # The source table mixes full-precision cells and values rounded to the
    # nearest yuan; differences up to one yuan are treated as presentation
    # rounding rather than label errors.
    result["tolerance_yuan"] = 1.0
    return result


def extract_luoyang(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="1", header=None)
    years = [int(str(value).replace("年", "")) for value in raw.iloc[2, 1:7]]
    records = []
    for _, row in raw.iloc[3:19].iterrows():
        county = str(row.iloc[0]).strip()
        if county == "洛阳市":
            continue
        for offset, year in enumerate(years, start=1):
            value = pd.to_numeric(row.iloc[offset], errors="coerce")
            if pd.notna(value):
                records.append({"year": year, "county": county, "official_income": float(value)})
    result = pd.DataFrame(records)
    result["city"] = "洛阳市"
    result["province"] = "河南省"
    result["source_table"] = "洛阳统计年鉴2024 表4-9"
    result["tolerance_yuan"] = 1.0
    return result


def extract_foshan(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="12-5", header=None)
    records = []
    for column in range(2, 7):
        county = str(raw.iloc[2, column]).strip()
        value = pd.to_numeric(raw.iloc[6, column], errors="coerce")
        if pd.notna(value):
            records.append({"year": 2023, "county": county, "official_income": float(value)})
    result = pd.DataFrame(records)
    result["city"] = "佛山市"
    result["province"] = "广东省"
    result["source_table"] = "佛山统计年鉴2024 表12-5"
    result["tolerance_yuan"] = 1.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--guangzhou", type=Path, required=True)
    parser.add_argument("--foshan", type=Path, required=True)
    parser.add_argument("--luoyang", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    official = pd.concat([
        extract_guangzhou(args.guangzhou),
        extract_foshan(args.foshan),
        extract_luoyang(args.luoyang),
    ], ignore_index=True)
    if official[["year", "county", "city", "province"]].duplicated().any():
        raise RuntimeError("Duplicate official year-county records")

    targets = pd.read_csv(args.targets)
    comparison = official.merge(
        targets[["year", "county_code", "county", "city", "province", "rural_income_nominal"]],
        on=["year", "county", "city", "province"], how="left", validate="one_to_one"
    )
    comparison["difference_yuan"] = comparison["rural_income_nominal"] - comparison["official_income"]
    comparison["absolute_difference_yuan"] = comparison["difference_yuan"].abs()
    comparison["status"] = "NOT_IN_TRAINING_PANEL"
    matched = comparison["rural_income_nominal"].notna()
    comparison.loc[matched, "status"] = "MATCH"
    mismatch = matched & (
        comparison["absolute_difference_yuan"] > comparison["tolerance_yuan"]
    )
    comparison.loc[mismatch, "status"] = "MISMATCH"

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output / "official_workbooks_vs_training_all.csv", index=False, encoding="utf-8-sig")
    candidates = comparison[comparison["status"] == "MISMATCH"].copy()
    candidates["county_code"] = candidates["county_code"].astype("Int64")
    candidates["old_rural_income_nominal"] = candidates["rural_income_nominal"]
    candidates["corrected_rural_income_nominal"] = candidates["official_income"]
    candidates["verification_status"] = "VERIFIED_OFFICIAL_WORKBOOK"
    candidates["reason"] = "县名对应的训练值与官方年鉴表不一致"
    candidates[[
        "year", "county_code", "county", "city", "province",
        "old_rural_income_nominal", "corrected_rural_income_nominal",
        "difference_yuan", "source_table", "verification_status", "reason"
    ]].to_csv(
        output / "official_workbooks_correction_candidates.csv",
        index=False, encoding="utf-8-sig"
    )

    summary = {
        "official_records": int(len(official)),
        "matched_training_records": int(matched.sum()),
        "matches": int((comparison["status"] == "MATCH").sum()),
        "mismatches": int(len(candidates)),
        "not_in_training_panel": int((comparison["status"] == "NOT_IN_TRAINING_PANEL").sum()),
        "mismatches_by_source": candidates.groupby("source_table").size().to_dict(),
    }
    (output / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nMismatches:\n")
    print(candidates[[
        "year", "county_code", "county", "city", "rural_income_nominal",
        "official_income", "difference_yuan", "source_table"
    ]].sort_values(["source_table", "year", "county"]).to_string(index=False))


if __name__ == "__main__":
    main()
