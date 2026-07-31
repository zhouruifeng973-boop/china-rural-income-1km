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
