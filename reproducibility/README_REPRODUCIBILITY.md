# 代码可复现化路线图

版本目标：复现 `outputs/stage6_scientific_data_product/` 的 `v1.0-draft` 数据产品。  
当前状态：脚本已按 stage1—stage9 存在，但仍有部分硬编码路径、历史脚本和实验脚本。后续目标是把它们整理成清晰、可解释、可按顺序执行的研究流水线。

## 1. 复现边界

本项目的复现分为三层。

公开数据仓库发布边界见：

- `reproducibility/RELEASE_PACKAGE_PLAN_v1.0_draft.md`

投稿和数据仓库说明草稿见：

- `docs/scientific_data_paper_outline.md`
- `docs/scientific_data_manuscript_full_draft.md`
- `docs/scientific_data_manuscript_gap_checklist.md`
- `docs/scientific_data_figures_tables_plan.md`
- `docs/figures/figure1_workflow.svg`
- `docs/figures/figure1_workflow.png`
- `docs/figures/figure1_workflow_caption.md`
- `docs/figures/figure2_annual_income_2019_2023.png`
- `docs/figures/figure2_annual_income_2019_2023.tif`
- `docs/figures/figure2_annual_income_2019_2023_caption.md`
- `docs/figures/figure2_annual_income_2019_2023_qa.json`
- `scripts/create_figure2_annual_income.py`
- `docs/figures/figure3_income_change_2019_2023.png`
- `docs/figures/figure3_income_change_2019_2023.tif`
- `docs/figures/figure3_income_change_2019_2023_caption.md`
- `docs/figures/figure3_income_change_2019_2023_qa.json`
- `scripts/create_figure3_income_change.py`
- `docs/figures/figure4_technical_validation.png`
- `docs/figures/figure4_technical_validation.tif`
- `docs/figures/figure4_technical_validation_caption.md`
- `docs/figures/figure4_technical_validation_qa.json`
- `scripts/create_figure4_technical_validation.py`
- `docs/figures/figure5_quality_uncertainty.png`
- `docs/figures/figure5_quality_uncertainty.tif`
- `docs/figures/figure5_quality_uncertainty_caption.md`
- `docs/figures/figure5_quality_uncertainty_qa.json`
- `scripts/create_figure5_quality_uncertainty.py`
- `docs/tables/table1_input_datasets.md`
- `docs/tables/table1_input_datasets_qa.json`
- `docs/tables/table2_data_records.md`
- `docs/tables/table3_annual_income_summary.md`
- `docs/tables/table4_technical_validation.md`
- `docs/tables/table5_quality_uncertainty.md`
- `docs/tables/tables2_5_generation_qa.json`
- `scripts/create_scientific_data_tables.py`
- `scripts/create_table1_input_datasets.py`
- `docs/references_scientific_data_draft.md`
- `docs/scientific_data_background_summary_draft.md`
- `docs/scientific_data_methods_draft.md`
- `docs/scientific_data_data_records_draft.md`
- `docs/scientific_data_technical_validation_draft.md`
- `docs/scientific_data_usage_notes_draft.md`
- `docs/data_availability_statement.md`
- `docs/code_availability_statement.md`

### 1.1 完整复现

在用户具备所有原始输入数据的前提下，从预处理开始重建：

```text
原始统计数据 + 遥感/空间特征 + 人口/农村掩膜
        ↓
预处理与农村人口权重
        ↓
弱监督训练与验证
        ↓
省级校准
        ↓
最终 1 km 收入图
        ↓
质量图层/不确定性/审查
```

### 1.2 部分复现

若购买数据不能公开，则外部用户可以从公开的中间表、数据产品和元数据开始：

```text
公开派生表/产品
        ↓
复核数据字典、manifest、raster audit
        ↓
复核变化图、质量图层和不确定性图层
```

### 1.3 使用复现

用户不重跑模型，只使用数据产品：

```text
下载 income_maps/change_maps/quality_layers
        ↓
按 README 和 data dictionary 解释单位、NoData、农村定义
        ↓
在 GIS 或 Python/R 中分析
```

## 2. 当前推荐复现顺序

| 阶段 | 脚本 | 当前作用 | 当前状态 |
|---|---|---|---|
| Stage 0 | `scripts/check_reproducibility_inputs.py` | 开跑前检查输入数据、派生文件、最终产品和再分发边界 | 新增预检查脚本 |
| Stage 1 | `scripts/stage1_2020_validation.py` | 修复/审查 2020 初始预测与聚合验证 | 历史审查脚本，保留 |
| Stage 2 | `scripts/stage2_preprocess_data.py` | 整理省级收入、CPI、WorldPop、GHS-SMOD、农村权重 | 核心脚本 |
| Stage 2 | `scripts/stage2_strict_baseline.py` | 严格流程下的县级基线验证 | 核心验证脚本 |
| Stage 3 | `scripts/stage3_weak_supervised_downscaling.py` | 弱监督像元模型训练与县级回聚合验证 | 核心脚本，已有 CLI 参数 |
| Stage 3 | `scripts/stage3_spatiotemporal_validation.py` | 留省验证、留年验证 | 核心验证脚本，已有 CLI 参数 |
| Stage 4 | `scripts/stage4_province_calibration.py` | 省级校准与校准后验证 | 核心脚本 |
| Stage 5 | `scripts/stage5_final_maps.py` | 生成最终年度收入图 | 核心脚本 |
| Stage 5 | `scripts/stage5_create_unbounded_maps.py` | 生成 unbounded 主版本图 | 核心脚本 |
| Stage 5 | `scripts/stage5_change_analysis.py` | 生成 2019—2023 变化图和统计摘要 | 核心脚本 |
| Stage 5 | `scripts/stage5_add_raster_statistics.py` | 补充 raster statistics | 辅助脚本 |
| Stage 6 | `scripts/stage6_create_quality_layers.py` | 生成 quality layers | 核心脚本 |
| Stage 6 | `scripts/stage6_prepare_scientific_data_package.py` | 整理 Scientific Data 风格数据包 | 核心脚本 |
| Stage 7 | `scripts/stage7_uncertainty_proxy_layers.py` | 生成 uncertainty proxy layers | 核心脚本 |
| Stage 7 | `scripts/stage7_validation_diagnostics.py` | 补充分组验证诊断表 | 核心脚本 |
| Stage 8 | `scripts/stage8_ensemble_uncertainty.py` | 多随机种子 ensemble uncertainty | 核心脚本，已有 CLI 参数 |
| Stage 9 | `scripts/stage9_data_product_audit.py` | 最终数据产品审查 | 核心审查脚本 |
| Stage 10 | `scripts/stage10_income_label_sensitivity.py` | 2022年内蒙古90条收入标签保留/排除敏感性 | 补充验证；输出仅限内部使用 |

## 3. 暂不推荐作为主复现入口的脚本

以下脚本保留为历史实验或早期分析脚本，但不建议作为正式复现主入口：

- `scripts/run_all.py`：会临时改写脚本文本中的年份再运行，不适合数据论文复现；
- `scripts/predict_2024.py`：当前主数据产品不包含 2024；
- `scripts/predict_nationwide.py`、`scripts/predict_21feat.py`、`scripts/train_baseline.py`：属于早期建模路线；
- `scripts/pseudo_prediction.py`、`scripts/robustness_analysis.py`、`scripts/ntl_importance.py`、`scripts/morans_i.py`：可作为补充分析，不作为 v1.0-draft 主流水线。

## 4. 推荐命令顺序草案

以下命令是后续整理后的目标顺序。正式运行前应确认所有原始数据和依赖均已准备好。

请使用 Python 3.10 或更高版本运行复现入口，例如：

```powershell
python scripts\run_reproducible_pipeline.py
```

在 Windows 中应先用 `python --version` 确认当前解释器为 Python 3；不要默认假设 `py` 启动器指向正确版本。

新入口默认是 dry-run，只打印命令，不真正运行。若要实际执行，需要加 `--execute`。

注意：`config/workflow_v1.0_draft.yaml` 需要 `PyYAML` 才能被真正读取。若当前 Python 环境没有安装 `PyYAML`，新入口会退回内置默认参数，并在 dry-run 中提示 `using embedded defaults`。正式复现前建议先安装：

```powershell
pip install -r requirements_reproducibility.txt
```

正式跑模型前，建议先做 Stage 0 输入检查。默认只生成报告，不会因为购买数据或不可公开原始数据的边界问题而中断：

```powershell
python scripts/check_reproducibility_inputs.py
```

如果你是在本机准备“完整复现”，可以加 `--strict`，让缺失的完整复现必需输入直接返回失败：

```powershell
python scripts/check_reproducibility_inputs.py --strict
```

检查结果会写入：

- `reproducibility/input_check_report_v1.0_draft.md`
- `reproducibility/input_check_report_v1.0_draft.csv`

公开发布包构建脚本默认也是 dry-run，只显示哪些文件会被纳入公开包、哪些会被排除：

```powershell
python scripts/build_public_release_package.py
python scripts/build_public_release_package.py --show-files
```

确认无误后再执行：

```powershell
python scripts/build_public_release_package.py --execute
```

默认输出目录为：

- `outputs/public_release/china_rural_income_1km_2019_2023_v1.0-draft/`

生成公开包后，建议立刻做二次审查：

```powershell
python scripts/audit_public_release_package.py
```

审查结果会写入公开包内部：

- `PUBLIC_RELEASE_AUDIT_v1.0_draft.md`
- `metadata/public_release_audit_v1.0_draft.csv`

```powershell
python scripts/run_reproducible_pipeline.py
python scripts/run_reproducible_pipeline.py --execute --start stage7_uncertainty_proxy --stop stage9_audit
```

完整 stage 命令如下：

```powershell
python scripts/check_reproducibility_inputs.py
python scripts/stage2_preprocess_data.py
python scripts/stage2_strict_baseline.py
python scripts/stage3_weak_supervised_downscaling.py --year 2020 --mask strict --folds 5 --pixels-per-county 128 --epochs 180
python scripts/stage3_spatiotemporal_validation.py --run all --year 2020 --mask strict --province-folds 5 --province-pixels 128 --year-pixels 64 --epochs 150
python scripts/stage4_province_calibration.py
python scripts/stage5_final_maps.py
python scripts/stage5_create_unbounded_maps.py
python scripts/stage5_change_analysis.py
python scripts/stage5_add_raster_statistics.py
python scripts/stage6_create_quality_layers.py
python scripts/stage6_prepare_scientific_data_package.py
python scripts/stage7_uncertainty_proxy_layers.py
python scripts/stage7_validation_diagnostics.py
python scripts/stage8_ensemble_uncertainty.py --members 5 --seeds 1042,2039,3036,4033,5030 --years 2019 2020 2021 2022 2023 --pixels-per-county 128 --epochs 150 --threads 8
python scripts/stage9_data_product_audit.py
python scripts/stage10_income_label_sensitivity.py --n-estimators 300
```

Stage 10 不改变任何原始县级收入文件。它会在 `outputs/stage10_income_label_sensitivity/` 中生成两个分析准备版本，并固定公共测试样本比较留县、留省和留年结果。该目录中的分析准备表、逐县预测和交叉验证预测含县级收入信息，应作为受限内部复现材料保存，不应加入保守公开发布包。

## 5. 需要继续工程化的代码问题

下一阶段建议逐项处理：

1. 建立统一配置文件，减少脚本内硬编码路径；
2. 为核心 stage 脚本增加 `argparse` 参数；
3. 增加 `--dry-run` 或输入检查模式；
4. 增加统一日志输出；
5. 把依赖统一到 `requirements.txt`；
6. 写一个新的安全主入口，例如 `scripts/run_reproducible_pipeline.py`，只负责按顺序调用 stage 脚本，不动态改写任何脚本源码；
7. 为每个 stage 写清输入、输出和是否依赖非公开原始数据；
8. 最后重新运行 Stage 9 数据产品审查。

当前已完成的工程化进展：

- 新增 `config/workflow_v1.0_draft.yaml`；
- 新增 `scripts/repro_config.py`；
- 新增安全主入口 `scripts/run_reproducible_pipeline.py`；
- `stage3_weak_supervised_downscaling.py`、`stage3_spatiotemporal_validation.py`、`stage5_final_maps.py`、`stage8_ensemble_uncertainty.py` 已能通过 `--config` 读取默认参数；
- `scripts/run_reproducible_pipeline.py` 已能从配置生成 stage3/stage5/stage8 的命令，并会把 `--config` 传给全链路核心 stage 脚本；
- 默认 dry-run 已测试通过。

## 6. 当前复现注意事项

- 当前数据产品主版本为 `strict_unbounded`；
- 港澳台未纳入最终主产品；
- 空白像元表示无有效农村人口权重或不在研究口径内，不应插值解释为收入；
- 省级校准后省均值与官方统计一致是约束条件，不是独立验证；
- 县级 R²、MAE 等指标来自像元预测回聚合到县级后与县级真实收入比较，不是像元级真实误差；
- 购买数据和再分发不明确的原始数据不进入公开复现仓库。
