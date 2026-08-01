# Script index for v1.0-draft reproducibility

本索引用于说明哪些脚本属于当前 `v1.0-draft` 数据产品复现主链，哪些属于历史实验或补充分析。

## 1. 主复现链脚本

| 顺序 | 脚本 | 作用 | 备注 |
|---:|---|---|---|
| 0 | `check_reproducibility_inputs.py` | 开跑前检查输入数据、派生文件、最终产品和再分发边界 | Stage 0 预检查，默认只报告 |
| 1 | `stage2_preprocess_data.py` | 数据预处理、收入平减、WorldPop/GHS-SMOD 对齐、农村人口权重 | 核心 |
| 2 | `stage2_strict_baseline.py` | strict 农村定义下的县级基线验证 | 核心验证 |
| 3 | `stage3_weak_supervised_downscaling.py` | 弱监督像元模型训练和县级回聚合验证 | 有 CLI 参数，已接入 `--config` |
| 4 | `stage3_spatiotemporal_validation.py` | 留省、留年验证 | 有 CLI 参数，已接入 `--config` |
| 5 | `stage4_province_calibration.py` | 省级校准 | 核心 |
| 6 | `stage5_final_maps.py` | 年度最终图生成 | 核心，已接入 `--config` |
| 7 | `stage5_create_unbounded_maps.py` | unbounded 主版本图生成 | 核心 |
| 8 | `stage5_change_analysis.py` | 2019—2023 变化图和统计 | 核心 |
| 9 | `stage5_add_raster_statistics.py` | raster statistics 补充 | 辅助 |
| 10 | `stage6_create_quality_layers.py` | quality layers | 核心 |
| 11 | `stage6_prepare_scientific_data_package.py` | Scientific Data 风格数据包整理 | 核心 |
| 12 | `stage7_uncertainty_proxy_layers.py` | uncertainty proxy layers | 核心 |
| 13 | `stage7_validation_diagnostics.py` | 补充分组验证诊断 | 核心 |
| 14 | `stage8_ensemble_uncertainty.py` | 多随机种子 ensemble uncertainty | 有 CLI 参数，已接入 `--config` |
| 15 | `stage9_data_product_audit.py` | 最终数据产品审查 | 核心审查 |

## 2. 补充敏感性验证脚本

| 脚本 | 作用 | 公开边界 |
|---|---|---|
| `stage10_income_label_sensitivity.py` | 比较2022年内蒙古90条重分类记录“保留/训练时排除”对留县、留省和留年验证的影响 | 输出含县级收入标签与预测，仅限内部复现和审查，不进入保守公开发布包 |
| `stage11_hebei_2021_label_sensitivity.py` | 对河北2021年14条疑似沿用上一年值的标签建立不覆盖原始文件的修正版，并在相同分折、模型与随机种子下比较留县和留省验证 | 修正版标签、县级预测和差异表含受限县级收入信息，仅限内部复现和审查；公开稿只可报告汇总敏感性结论 |
| `stage12_hebei14_full_rerun.py` | 用版本化标签运行留年验证、五年联合弱监督训练、年度图和变化图；支持修正版与同环境原标签对照的隔离输出 | 不覆盖 v1.0；县级预测仅限内部，汇总指标和派生栅格可用于版本决策 |
| `stage12_compare_correction_only.py` | 逐像元比较同环境修正版与原标签对照版，分离14条修正净效应和历史重跑漂移 | 生成汇总表、审计报告及差值GeoTIFF，不包含县级原始收入值 |
| `stage13_audit_corrected_stage8.py` | 审查修正版五成员Stage 8的15幅栅格、空间对齐、有效掩膜、校准精度、成员设置和文件哈希 | 生成机器可读审查JSON、逐层CSV和最终审查Markdown |
| `stage14_nationwide_income_label_audit.py` | 对修正版8,373条县—年标签做全国自动异常筛查和省—年覆盖审查，不自动改值 | 记录级候选含受限收入值，仅限内部；公开材料只可使用汇总统计 |
| `stage15_extract_hebei_ranked_pdf.py` | 按PDF文字坐标提取河北三栏县级收入排序表，保留位次和原始文字用于人工核验 | 仅适用于带文字层且版式匹配的单页表；提取结果需与整页视觉检查和数值集合比对 |
| `stage15_reconcile_hebei_2021_official.py` | 修复PDF内嵌字体导致的县名乱码，并将2021年河北官方排序表与训练标签逐县一对一对账 | 只生成差异清单和审计报告，不修改标签；记录级收入值仅限内部 |
| `stage16_apply_hebei_2021_2022_reconciliation.py` | 在旧版本上应用15条经官方年鉴逐项核验的河北2021—2022修正，重算2019年不变价收入并记录哈希 | 生成新版本，不覆盖任何原始或中间标签；必须通过旧值、县名、行数和唯一键保护检查 |

## 3. 历史审查/早期实验脚本

这些脚本保留以保证研究过程可追踪，但不作为 v1.0-draft 的主复现入口：

- `stage1_2020_validation.py`
- `train_baseline.py`
- `predict_21feat.py`
- `predict_nationwide.py`
- `predict_2024.py`
- `pseudo_prediction.py`
- `robustness_analysis.py`
- `ntl_importance.py`
- `morans_i.py`
- `gen_cv_csv.py`
- `change_maps.py`
- `clean_income.py`
- `collinearity_removal.py`
- `extract_features.py`
- `audit_data.py`
- `build_report_html.py`
- `add_road_density.py`
- `create_county_id.py`

## 4. GEE 数据导出脚本

这些脚本用于在 Google Earth Engine 中导出遥感/空间特征，不属于本地 Python 主链：

- `gee_export.js`
- `gee_county_features_2021_2024.js`
- `gee_export_ghs_smod_2020.js`

## 5. 不推荐作为正式复现入口

`run_all.py` 是早期一键脚本，会临时改写脚本文本中的年份再运行。为了保证论文复现过程透明、稳定、可审计，不建议继续把它作为正式入口。

后续应新建一个安全入口，例如：

```text
scripts/run_reproducible_pipeline.py
```

该入口只按顺序调用 stage 脚本，不动态修改任何源码。

当前状态：`scripts/run_reproducible_pipeline.py` 已新增，默认 dry-run，并已能读取 `config/workflow_v1.0_draft.yaml` 中的 stage3/stage5/stage8 参数，同时向全链路核心 stage 脚本传递 `--config`。

## 6. 发布包构建脚本

- `build_public_release_package.py`：根据 `reproducibility/RELEASE_PACKAGE_PLAN_v1.0_draft.md` 构建保守公开发布包。默认 dry-run，不复制文件；加 `--execute` 后才会生成 `outputs/public_release/china_rural_income_1km_2019_2023_v1.0-draft/`。
- `audit_public_release_package.py`：对已生成的公开发布包做上传前二次审查，检查 README/metadata、manifest/hash、敏感文件缺席情况和 GeoTIFF 空间一致性。

该脚本当前会排除可能包含购买/第三方县级收入信息的表：

- `tables/county_training_income_2019_2023_2019yuan.csv`
- `tables/validation_leave_province_2020_county_residuals.csv`
- `tables/validation_leave_year_county_residuals.csv`

## 7. 官方年鉴标签复核与版本化修正

- `stage17_reconcile_guangzhou_foshan_luoyang.py`：将广州、佛山和洛阳官方年鉴工作簿与版本化县级训练标签逐条对账。只生成内部差异表与修正候选，不改写标签；原始工作簿不进入公开发布包。
- `stage18_apply_guangzhou_luoyang_official_corrections.py`：将 15 条经官方工作簿核实的广州、洛阳修正应用到新版本，并重算 2019 年不变价收入。生成 v3 标签、修正审计表和哈希清单，不覆盖父版本与证据工作簿。
- `stage14_nationwide_income_label_audit.py --reviewed-records <CSV>`：可接收一个或多个官方工作簿对账结果，将 `MATCH` 记录标为已核实，避免真实极值继续进入未复核异常队列。
- `stage19_v3_leave_province_validation.py`：使用版本化县级收入标签执行2020年五折留省验证，并将预测、分折指标、总体指标及哈希清单写入对应版本的独立验证目录。
- `stage8_ensemble_uncertainty.py`：已支持从版本化配置读取标签、独立包目录和模型目录；v3运行生成5成员、5年份共15幅集成均值/标准差/CV栅格及独立哈希清单。
- `run_v3_final_pipeline.py`：v3最终链路的安全主入口，默认dry-run，按顺序组织完整重跑、留省验证、集成不确定性、候选包构建、质量图、风险图和最终审查。
- `stage22_audit_v3_code_reproducibility.py`：检查v3配置、冻结标签哈希、核心脚本语法、依赖声明、dry-run入口、数据审查状态和用户特定绝对路径。
- `stage23_build_and_audit_v3_zip.py`：以固定目录结构构建 v1.3 最终 ZIP，生成 SHA-256 校验文件，并在独立目录真实解压后逐文件比较大小与哈希，同时复核 ZIP 路径安全、CRC 和内部发布清单；只生成本地待上传文件，不执行 Zenodo 发布。
