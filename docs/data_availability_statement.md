# Data Availability Statement 草稿

适用版本：`china_rural_income_1km_2019_2023_v1.5-no-road`
当前状态：公开发布包已生成并通过二次审查；ZIP 已上传到 Zenodo restricted draft，预留 DOI 为 `10.5281/zenodo.21711838`，尚未公开发布。

## 中文说明版

本研究生成的中国农村居民人均可支配收入 1 km 栅格数据产品，包括 2019—2023 年年度收入图、2019—2023 年变化图、质量控制图层、不确定性辅助图层、多随机种子集成不确定性图层、验证结果表、数据字典、文件清单和来源许可引用清单。公开发布包位于本地目录：

```text
outputs/public_release/china_rural_income_1km_2019_2023_v1.0-draft/
```

该公开包已经通过上传前二次审查：

```text
outputs/public_release/china_rural_income_1km_2019_2023_v1.0-draft/PUBLIC_RELEASE_AUDIT_v1.0_draft.md
```

二次审查结果为 `OK = 156, WARN = 0, FAIL = 0`，公开包中共包含 67 个 GeoTIFF 文件，总大小约 700 MB。公开包中的 `metadata/public_release_manifest_v1.0_draft.csv` 提供了文件大小和 SHA-256 校验值。

数据仓库记录：

```text
Zenodo reserved DOI: https://doi.org/10.5281/zenodo.21711838
Owner-only draft URL: https://zenodo.org/uploads/21711838
Intended public record URL after publication: https://zenodo.org/records/21711838
```

草稿网址仅供记录所有者管理，不应写入论文或作为审稿人访问链接。DOI 只有在 Zenodo 记录发布后才会正式注册并解析。

作者原创的 1 km 派生数据产品拟采用 Creative Commons Attribution 4.0 International（CC BY 4.0）许可证，但正式公开仍以完成第三方权利复核为前提。该许可证只覆盖公开包中的作者原创派生产品和作者编写的说明文档，不覆盖作者整理的县级收入工作源表、购买行政边界、国家统计局原始 Excel 或其他未纳入公开包的第三方材料。

本研究最终v1.5流程使用的公开源数据包括 WorldPop 人口数据、GHS-SMOD 城乡分类、VIIRS 夜间灯光、MODIS 植被指数、MODIS NPP、ESA WorldCover、ERA5-Land、Sentinel-2、SRTM以及国家统计局公开统计数据。早期实验中的2026年OpenStreetMap道路密度特征已从最终模型和数据产品中删除。各源数据的来源、引用和许可信息记录在：

```text
outputs/stage6_scientific_data_product/PUBLIC_SOURCE_DATA_CITATIONS_AND_LICENSES_FILLED.md
outputs/stage6_scientific_data_product/metadata/source_data_inventory_public_sources_filled_draft.csv
```

县级收入标签由作者根据各省统计年鉴及地方官方统计资料整理形成。工作源表 `CNT_ResidentIncome.xlsx` 以及能够重建县级真实值的训练表和残差表未包含在公开发布包中。`中国各县域城乡居民收入（2000-2022年）.xlsx` 是重复历史副本，不参与分析。非再分发行政边界、原始国家统计局 Excel 工作簿以及其他再分发条款尚待复核的第三方原始数据也未包含在公开包中。具体排除规则见：

```text
outputs/stage6_scientific_data_product/metadata/excluded_raw_data_policy_v1.0_draft.md
outputs/public_release/china_rural_income_1km_2019_2023_v1.0-draft/PUBLIC_RELEASE_README_v1.0_draft.md
```

如果审稿人或读者需要进行完整本地复现，需要自行取得相应权限的数据输入，并根据复现说明准备本地目录。若仅使用或复核公开数据产品，则不需要访问上述受限原始数据。

## English version for manuscript

The gridded data products generated in this study, including annual 1 km maps of rural residents' per capita disposable income in China for 2019–2023, 2019–2023 change maps, quality layers, rule-based uncertainty proxy layers, ensemble uncertainty layers, validation tables, metadata, data dictionary, file manifests, and source-data citation/license records, have been uploaded to a restricted Zenodo draft record under the reserved DOI https://doi.org/10.5281/zenodo.21711838. The DOI will be registered and the files will be made publicly accessible after completion of the remaining third-party rights review.

The author-created 1 km derived data products will be released under the Creative Commons Attribution 4.0 International (CC BY 4.0) licence, subject to completion of the remaining third-party rights review. The licence applies only to the derived products and author-created documentation included in the public package. It does not apply to the author-compiled county-income source table, non-redistributed administrative boundaries, raw National Bureau of Statistics workbooks, or any other excluded third-party source material.

The current pre-release package is versioned as `china_rural_income_1km_2019_2023_v1.0-draft`. The package contains 67 GeoTIFF files and associated tables and metadata, with a total size of approximately 700 MB. File sizes and SHA-256 checksums are provided in `metadata/public_release_manifest_v1.0_draft.csv`. A pre-release audit of the public package reported 156 checks passed, with no warnings or failures.

The public package includes the final data products and documentation required for data reuse and technical validation, including the annual income maps, change maps, bounded sensitivity maps, quality layers, uncertainty proxy layers, ensemble uncertainty layers, validation summary tables, raster audit tables, data dictionary, file manifest, and source-data citation/license inventory.

The final v1.5 workflow used WorldPop population data, GHS-SMOD settlement classification, VIIRS nighttime lights, MODIS vegetation indices, MODIS net primary productivity, ESA WorldCover, ERA5-Land, Sentinel-2, SRTM, and public statistical data from the National Bureau of Statistics of China. A road-density covariate derived from a 2026 OpenStreetMap snapshot and used in an earlier experiment was removed from final model training, validation and data generation. Source information, citations, and license/use terms are documented in the source-data inventory and citation/license files accompanying the data product.

County-level income labels were compiled by the authors from provincial statistical yearbooks and other officially published local statistical materials. The working source table (`CNT_ResidentIncome.xlsx`) and county-level training or residual tables that could reconstruct the observations are withheld from the current public package. The duplicate historical workbook `中国各县域城乡居民收入（2000-2022年）.xlsx` was not used in the analysis. Non-redistributed administrative boundaries, raw National Bureau of Statistics Excel workbooks, and other third-party raw inputs with uncertain redistribution terms are also excluded. Users who wish to reproduce the full modelling workflow from raw inputs should compile the statistical labels from the documented yearbook sources, obtain permitted boundary inputs independently, and follow the reproducibility instructions. Users who wish to reuse or technically inspect the released gridded products do not need access to these non-redistributed raw inputs.

## 投稿前需要补充的信息

- [x] 数据仓库名称：Zenodo；
- [x] 数据集预留 DOI：10.5281/zenodo.21711838；
- [x] 草稿管理 URL：https://zenodo.org/uploads/21711838；
- [ ] 数据集公开 URL（记录发布后核验 https://zenodo.org/records/21711838）；
- [ ] 数据仓库最终版本号；
- [ ] 若期刊要求，补充访问日期；
- [ ] 若老师决定公开更多县级派生表，需要重新审查此声明。
