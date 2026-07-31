# v1.0-draft 公开数据仓库发布结构草案

版本对象：`china_rural_income_1km_2019_2023_v1.0-draft`  
冻结数据产品目录：`outputs/stage6_scientific_data_product/`  
当前数据产品规模：约 131 个文件，约 703 MB  
用途：为 Scientific Data 风格数据仓库发布、审稿人复核、后续论文 Methods/Data Records/Usage Notes 写作提供边界。

## 1. 发布原则

这个项目建议拆成两层发布：

1. **公开数据产品包**：面向读者、审稿人和数据仓库，包含最终 tif、表格、元数据、README、许可引用说明。
2. **代码复现仓库**：面向方法复现，包含脚本、配置、requirements、复现说明、输入检查器；不包含购买数据、原始县级收入、购买县界、原始 OSM PBF 等有再分发风险的数据。

不建议把当前整个工作目录原样上传。这个目录里既有最终产品，也有购买数据、历史脚本、临时文件和原始大数据，直接上传会带来版权/许可风险，也会让仓库显得不干净。

## 2. 推荐公开数据产品包结构

建议把 `outputs/stage6_scientific_data_product/` 作为公开数据产品包的主体，整理成下面结构：

```text
china_rural_income_1km_2019_2023_v1.0-draft/
├── README.md
├── VERSION_v1.0_draft.md
├── PUBLIC_SOURCE_DATA_CITATIONS_AND_LICENSES_FILLED.md
├── SOURCE_DATA_LICENSE_AND_CITATION_AUDIT.md
├── STAGE7_uncertainty_proxy_and_validation_summary.md
├── STAGE8_ensemble_uncertainty_summary.md
├── STAGE9_final_data_product_audit.md
├── data/
│   ├── income_maps/
│   ├── change_maps/
│   ├── sensitivity_bounded_maps/
│   ├── quality_layers/
│   ├── uncertainty_proxy_layers/
│   └── ensemble_uncertainty_layers/
├── tables/
└── metadata/
```

### 2.1 必须公开的核心产品

| 类别 | 建议发布 | 理由 |
|---|---|---|
| `data/income_maps/` | 是 | 2019—2023 年 1 km 农村居民人均可支配收入主产品 |
| `data/change_maps/` | 是 | 2019—2023 绝对变化和百分比变化图 |
| `data/quality_layers/` | 是 | 解释哪些像元有效、哪些省份/区域更可靠 |
| `data/uncertainty_proxy_layers/` | 是 | 数据产品不确定性辅助解释 |
| `data/ensemble_uncertainty_layers/` | 是 | 多随机种子集成不确定性结果，更符合数据论文要求 |
| `metadata/data_dictionary.csv` | 是 | 解释每个文件、字段、单位、NoData、用途 |
| `metadata/file_manifest.csv` | 是 | 文件清单和校验信息，方便审稿人检查完整性 |
| `tables/validation_summary.csv` | 是 | 模型验证核心结果 |
| `tables/final_data_product_raster_audit.csv` | 是 | 坐标系、分辨率、NoData、范围一致性审查 |

### 2.2 建议公开但要解释清楚的辅助产品

| 类别 | 建议发布 | 注意事项 |
|---|---|---|
| `data/sensitivity_bounded_maps/` | 是 | 明确这是 bounded 敏感性版本，不是主版本 |
| `tables/county_training_income_2019_2023_2019yuan.csv` | 谨慎 | 如果包含购买县级收入或可反推购买数据，先不要公开；可改为仅公开聚合/脱敏验证表 |
| `tables/official_province_rural_income_2019_2023_2019yuan.csv` | 可公开 | 来源为国家统计局，需保留网页、指标、访问日期、处理方法 |
| `tables/county_rural_weight_summary_2019_2023.csv` | 谨慎 | 如果不含受限县界属性，一般可作为派生表；但发布前再做字段审查 |
| `metadata/source_data_inventory*.csv` | 是 | 用于说明所有源数据、许可、引用、再分发边界 |

## 3. 不应进入公开数据仓库的内容

以下内容当前不建议上传到公开数据仓库，也不建议放进公开 GitHub：

| 本地内容 | 当前处理建议 | 原因 |
|---|---|---|
| `data/county_rural_income_2019.csv` 至 `data/county_rural_income_2023.csv` | 不公开 | 县级收入原始/近原始标签来自购买或第三方数据 |
| `data/county_rural_income_all.csv` | 不公开 | 可能包含购买县级收入合集 |
| `data/CNT_ResidentIncome.xlsx` | 不公开 | 用户已说明为网上购买获得 |
| `data/中国各县域城乡居民收入（2000-2022年）.xlsx` | 不使用、仅内部归档 | 与 `CNT_ResidentIncome.xlsx` 的2000—2022记录重复，不参与当前分析 |
| `data/2020xianBJ/` | 不公开 | 购买县界或再分发边界未确认 |
| 原始国家统计局 Excel | 暂不随包公开 | 可提供来源网页、指标和处理后派生表；原始 Excel 是否再分发后续再确认 |
| `data/osm/china-latest.osm.pbf` | 不直接公开 | OSM 原始 PBF 受 ODbL 约束，需单独审查 |
| 临时文件、ArcMap 工程缓存、`.aux.xml` 是否保留 | 视数据仓库规则 | `.aux.xml` 和 `.ovr` 对 ArcMap 显示有帮助，但不是科学数值核心 |

一句大白话：公开包要发“我们产出的最终数据和说明”，不要把“训练时用到的购买原料”也倒进去。

## 4. 推荐代码复现仓库结构

代码仓库可以单独建一个干净目录，例如：

```text
rural-income-1km-china-reproducibility/
├── README.md
├── requirements_reproducibility.txt
├── config/
│   └── workflow_v1.0_draft.yaml
├── scripts/
│   ├── check_reproducibility_inputs.py
│   ├── run_reproducible_pipeline.py
│   ├── repro_config.py
│   ├── stage2_preprocess_data.py
│   ├── stage2_strict_baseline.py
│   ├── stage3_weak_supervised_downscaling.py
│   ├── stage3_spatiotemporal_validation.py
│   ├── stage4_province_calibration.py
│   ├── stage5_final_maps.py
│   ├── stage5_create_unbounded_maps.py
│   ├── stage5_change_analysis.py
│   ├── stage5_add_raster_statistics.py
│   ├── stage6_create_quality_layers.py
│   ├── stage6_prepare_scientific_data_package.py
│   ├── stage7_uncertainty_proxy_layers.py
│   ├── stage7_validation_diagnostics.py
│   ├── stage8_ensemble_uncertainty.py
│   └── stage9_data_product_audit.py
├── reproducibility/
│   ├── README_REPRODUCIBILITY.md
│   ├── RELEASE_PACKAGE_PLAN_v1.0_draft.md
│   ├── input_check_report_v1.0_draft.md
│   └── input_check_report_v1.0_draft.csv
└── docs/
    ├── data_availability_statement.md
    ├── code_availability_statement.md
    └── excluded_raw_data_policy_v1.0_draft.md
```

代码仓库里可以保留“空目录说明”，但不要把受限数据放进去。例如：

```text
data/
├── README_expected_inputs.md
├── worldpop/
├── ghs_smod/
├── processed/
└── private_inputs_not_redistributed/
```

其中 `private_inputs_not_redistributed/` 只放说明，不放真实数据。

## 5. 建议 `.gitignore` 规则

公开代码仓库建议加入以下忽略规则：

```gitignore
# private / purchased / restricted raw data
data/county_rural_income_*.csv
data/county_rural_income_all.csv
data/CNT_ResidentIncome.xlsx
data/2020xianBJ/
data/osm/*.pbf

# raw downloaded workbooks kept local unless redistribution is cleared
data/*居民人均可支配收入*
data/*消费价格指数*

# large intermediate rasters; publish final product through data repository instead
features/*.tif
data/processed/**/*.tif
outputs/stage5_final_maps/
outputs/stage8_ensemble_uncertainty/

# caches and local GIS sidecars
__pycache__/
*.pyc
*.lock
*.tmp
*.bak
```

注意：公开数据产品仓库和代码仓库的 `.gitignore` 可以不同。代码仓库应尽量轻；数据仓库应包含最终数据产品。

## 6. 发布前审查清单

正式上传前建议逐项确认：

- [ ] `metadata/file_manifest.csv` 覆盖所有公开发布文件；
- [ ] `metadata/data_dictionary.csv` 解释所有 tif、csv、NoData、单位、坐标系；
- [ ] `STAGE9_final_data_product_audit.md` 说明所有 tif 空间对齐；
- [ ] `PUBLIC_SOURCE_DATA_CITATIONS_AND_LICENSES_FILLED.md` 覆盖 WorldPop、GHS-SMOD、VIIRS、MODIS、ESA WorldCover、ERA5-Land、Sentinel-2、SRTM、OSM、国家统计局；
- [ ] 所有购买/私有数据未进入公开包；
- [ ] 所有原始县级收入和购买县界未进入公开包；
- [ ] 主产品 `strict_unbounded` 与辅助产品 `bounded`、`quality`、`uncertainty` 的关系在 README 中解释清楚；
- [ ] 变化图明确为 2019—2023、2019 年不变价；
- [ ] 港澳台是否排除、农村掩膜定义、NoData 含义写清楚；
- [ ] 代码仓库 dry-run 能从 `stage0_input_check` 打印完整流程；
- [ ] 数据仓库上传后，下载一次重新跑 `stage9_data_product_audit.py` 或至少复核 manifest/hash。

## 7. 论文中可对应的写法

这个发布结构对应 Scientific Data 论文中的几个部分：

| 论文部分 | 对应材料 |
|---|---|
| Data Records | `data/`、`metadata/data_dictionary.csv`、`metadata/file_manifest.csv` |
| Technical Validation | `tables/validation_summary.csv`、`STAGE7/8/9` 报告 |
| Usage Notes | `README.md`、quality layers、uncertainty layers 说明 |
| Code Availability | `scripts/`、`config/`、`requirements_reproducibility.txt` |
| Data Availability | 公开数据产品仓库 DOI + 受限原始数据说明 |

当前已起草的投稿声明文件：

- `docs/data_availability_statement.md`
- `docs/code_availability_statement.md`

## 8. 下一步建议

下一步可以做“发布包构建脚本”，但先不要真正上传：

```text
当前 frozen product
        ↓
按本计划复制允许公开的文件
        ↓
跳过私有/购买/需权利复核数据
        ↓
生成 public_release_manifest
        ↓
重新计算 hash 和文件大小
        ↓
形成待上传 zip/目录
```

建议脚本名：

```text
scripts/build_public_release_package.py
```

默认 dry-run，只打印将复制/排除哪些文件；确认后再加 `--execute` 生成真正的公开发布目录。
