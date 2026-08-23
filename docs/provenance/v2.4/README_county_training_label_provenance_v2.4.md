# County training-label provenance, public v2.4

This metadata release provides a normalized, public-safe audit trail for the
county-year income labels used by the study. It contains no county income values,
restricted source documents, local file paths, or internal file hashes.

## Coverage

- Training-label records: 8,373.
- Records linked to an exact `source_item_id`: 8,373 (100%).
- County names and numeric values verified: 7,840 (93.63%).
- Records with numeric verification pending: 533 (6.37%).
- Distinct source items referenced: 95.
- Province-year groups: 98.

## Relational structure

Join `county_training_label_record_provenance_public_v2.4.csv` to `county_training_label_source_items_public_v2.4.csv` using `source_item_id`. The record table
identifies the province, city, county, year, verification status, and final-training
treatment of every label. The source table stores bibliographic and locator metadata
once per source item. `county_training_label_province_year_provenance_public_v2.4.csv` provides the compact 98-group summary.

The field `numeric_value_verified=no` does not mean that the record lacks a source.
It means that the source item has been identified but that the county-name/value pair
has not yet been checked at record level. These 533 records remain
explicitly distinguishable for risk assessment or verified-only sensitivity tests.

## Locator interpretation

Exact table/section names or stable URLs are preferred. A page number is not required
when an exact table/section or stable URL uniquely locates the source. For historical
items whose public URL or page/table locator could not be recovered, `locator_status`
is set to `bibliographic_title_or_local_source_name_only`; these cases are disclosed
rather than represented as fully public locators.

Source-locator status counts: {"bibliographic_title_or_local_source_name_only": 55, "exact_table_or_section_and_url_available": 40}.
