# road_density未来信息泄漏敏感性评估

本实验使用相同的v1.4县级标签、随机种子、抽样量、网络结构、训练轮数和省级校准流程，唯一变化是从训练及全国预测中删除2026年OSM `road_density`。

## 预先设定的判据

- 两类县级验证的绝对R²变化不超过0.02，MAPE变化不超过1个百分点。
- 五幅年度图与核心产品的Pearson相关系数均不低于0.95，且相对核心图中位数的NMAE均不超过20%。
- 2019至2023年绝对变化图的Pearson相关系数不低于0.80。

## 县级验证

| validation_scheme | reference_n | ablation_n | reference_r2 | ablation_r2 | delta_r2 | reference_mae | ablation_mae | delta_mae | reference_rmse | ablation_rmse | delta_rmse | reference_mape | ablation_mape | delta_mape | reference_bias | ablation_bias | delta_bias |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leave_year_2019_2023 | 8275 | 8275 | 0.6857 | 0.6890 | 0.0034 | 2778.0833 | 2789.7861 | 11.7028 | 3754.6417 | 3734.4280 | -20.2136 | 15.3274 | 15.4890 | 0.1615 | -320.1143 | -290.8455 | 29.2688 |
| leave_province_2020 | 1682 | 1682 | 0.4521 | 0.4418 | -0.0102 | 3209.9180 | 3292.5816 | 82.6637 | 4513.6295 | 4555.5924 | 41.9629 | 18.9375 | 19.7212 | 0.7837 | -602.3078 | -616.3186 | -14.0108 |

## 空间产品比较

| product | unit | common_valid_pixels | pearson_r | reference_mean | ablation_mean | mean_difference | mae | rmse | median_absolute_difference | p95_absolute_difference | reference_median | nmae_pct_of_reference_median | pixels_abs_difference_gt_1000_pct | pixels_abs_difference_gt_3000_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| income_2019 | 2019 CNY person-1 year-1 | 4038300 | 0.8392 | 12242.8773 | 12555.7576 | 312.8804 | 1942.6292 | 2937.6439 | 1335.2114 | 5721.1616 | 11277.5391 | 17.2256 | 60.4561 | 19.2097 |
| income_2020 | 2019 CNY person-1 year-1 | 4043058 | 0.8294 | 13009.3367 | 13408.8607 | 399.5241 | 2176.7375 | 3231.4260 | 1512.9297 | 6387.5562 | 11976.5664 | 18.1750 | 64.5343 | 23.2717 |
| income_2021 | 2019 CNY person-1 year-1 | 4044215 | 0.8313 | 13962.4920 | 14430.2969 | 467.8049 | 2345.3689 | 3439.5619 | 1652.0469 | 6800.3965 | 12780.1875 | 18.3516 | 67.2604 | 26.2686 |
| income_2022 | 2019 CNY person-1 year-1 | 4044185 | 0.8267 | 14521.2941 | 15042.6257 | 521.3316 | 2458.7981 | 3563.8158 | 1744.4824 | 7131.5981 | 13274.3682 | 18.5229 | 68.9538 | 28.1718 |
| income_2023 | 2019 CNY person-1 year-1 | 4043179 | 0.8166 | 15612.7591 | 16281.2467 | 668.4876 | 2729.0571 | 3902.6957 | 1981.8867 | 7736.2065 | 14288.1738 | 19.1001 | 72.4582 | 32.9065 |
| change_2019_2023_absolute | 2019 CNY person-1 year-1 | 4038291 | 0.4193 | 3373.4985 | 3728.4267 | 354.9282 | 2572.5859 | 3852.7050 | 1769.7041 | 7567.2383 | 2993.9756 | 85.9254 | 68.9972 | 29.3822 |
| change_2019_2023_percent | percent | 4038291 | 0.4125 | 30.1825 | 33.6877 | 3.5052 | 22.0543 | 31.4595 | 16.5125 | 60.3099 | 27.6374 | 79.7988 | 0.0010 | 0.0000 |

## 判定

- 县级验证稳定性：通过。
- 年度空间格局稳定性：未通过。
- 2019至2023年变化格局稳定性：未通过。
- 总体判定：未通过。
- 建议：At least one predefined stability criterion failed; the core product should be regenerated without the post-period road-density covariate rather than relying only on disclosure.

