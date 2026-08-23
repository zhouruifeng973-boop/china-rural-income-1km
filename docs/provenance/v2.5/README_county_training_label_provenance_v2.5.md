# County training-label provenance, public v2.5

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
