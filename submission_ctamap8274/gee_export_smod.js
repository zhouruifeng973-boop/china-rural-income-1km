// ============================================================
// GEE：导出 GHS-SMOD 2020 城乡聚落分类及农村掩膜
// 数据：JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020
// 输出网格与本项目现有 6815 × 3942 特征栅格一致
// ============================================================

// 项目现有参考网格（来自 features/ntl_2020_1km.tif）
var TARGET_CRS = 'EPSG:4326';
var TARGET_TRANSFORM = [
  0.008983152841195215, 0, 73.55405546370642,
  0, -0.008983152841195215, 53.56654039204707
];
var TARGET_BOUNDS = [
  73.55405546370642, 18.15495189205553,
  134.7742420764518, 53.56654039204707
];
var region = ee.Geometry.Rectangle(TARGET_BOUNDS, TARGET_CRS, false);

// 2020年主分类栅格；分类数据默认采用最近邻重采样
var smod = ee.Image('JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020')
  .select('smod_code')
  .rename('smod_code')
  .clip(region)
  .unmask(-200)
  .toInt16();

// 严格农村：11 极低密度农村、12 低密度农村、13 农村聚落/村庄
var ruralStrict = smod.eq(11)
  .or(smod.eq(12))
  .or(smod.eq(13))
  .rename('rural_mask_strict')
  .toByte();

// 宽松农村：严格农村 + 21 城乡接合部；仅建议用于稳健性分析
var ruralBroad = ruralStrict
  .or(smod.eq(21))
  .rename('rural_mask_broad')
  .toByte();

// 可视化检查
var palette = [
  '7ab6f5', // 10 水体
  'cdf57a', // 11 极低密度农村
  'abcd66', // 12 低密度农村
  '375623', // 13 农村聚落
  'ffff00', // 21 城乡接合部
  'a87000', // 22 半密集城市聚落
  '732600', // 23 密集城市聚落
  'ff0000'  // 30 城市中心
];
Map.centerObject(region, 4);
Map.addLayer(smod.updateMask(smod.neq(-200)),
  {min: 10, max: 30, palette: palette}, 'GHS-SMOD 2020');
Map.addLayer(ruralStrict.selfMask(), {palette: ['32a852']},
  'Strict rural: 11/12/13', false);
Map.addLayer(ruralBroad.selfMask(), {palette: ['8bd646']},
  'Broad rural: +21', false);

print('GHS-SMOD 2020', smod);
print('目标区域', region);

// 任务1：原始城乡分类（建议必须导出）
Export.image.toDrive({
  image: smod,
  description: 'ghs_smod_2020_1km_aligned',
  folder: 'Rural_income',
  fileNamePrefix: 'ghs_smod_2020_1km_aligned',
  region: region,
  crs: TARGET_CRS,
  crsTransform: TARGET_TRANSFORM,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true, noData: -200}
});

// 任务2：严格农村二值掩膜（主分析）
Export.image.toDrive({
  image: ruralStrict,
  description: 'rural_mask_2020_strict_1km_aligned',
  folder: 'Rural_income',
  fileNamePrefix: 'rural_mask_2020_strict_1km_aligned',
  region: region,
  crs: TARGET_CRS,
  crsTransform: TARGET_TRANSFORM,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true, noData: 0}
});

// 任务3：包含城乡接合部的宽松掩膜（稳健性分析，可选）
Export.image.toDrive({
  image: ruralBroad,
  description: 'rural_mask_2020_broad_1km_aligned',
  folder: 'Rural_income',
  fileNamePrefix: 'rural_mask_2020_broad_1km_aligned',
  region: region,
  crs: TARGET_CRS,
  crsTransform: TARGET_TRANSFORM,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true, noData: 0}
});
