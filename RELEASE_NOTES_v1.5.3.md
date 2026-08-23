# v1.5.3-no-road release notes

## Provenance v2.5 completion

- Retained source-item mappings for all 8,373 county-year training labels.
- Increased county-name and numeric verification from 7,840 to 7,933 records, or from 93.63% to 94.75%.
- Reduced records pending numeric verification from 533 to 440, or from 6.37% to 5.25%.
- Added exact table or section locators for all 95 source items; 40 also provide a URL.
- Applied a conservative same-row rule requiring a county name or unique administrative stem and a source value within 1 yuan.
- Changed no training-label value during the completion pass.
- Retained the prior experiment excluding 533 then-pending records as a conservative sensitivity test because it excludes a superset of the 440 records still pending.

## Frozen data package

- Formal package name: `china_rural_income_1km_2019_2023_v1.5.3-no-road.zip`.
- Package inventory: 156 files, including 67 GeoTIFFs.
- Directory inventory: 70 files under `data/`, 46 under `tables/`, 36 under `metadata/`, and four at the package root.
- Formal manifest: `metadata/v1_5_no_road_manifest.json`, with 155 entries excluding the manifest itself.
- ZIP SHA-256: `aa18b2a5c14b4151a4a5cba1a2f4b07ef31b29c3521b09f93c4351ea881c06e6`.
- Reserved dataset DOI: `10.5281/zenodo.21711838`.

The gridded products and validation metrics are unchanged from v1.5.2-no-road; this revision updates public provenance evidence and cross-carrier consistency.
