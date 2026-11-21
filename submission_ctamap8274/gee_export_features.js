// ============================================================
// GEE 脚本：2021-2024 年 1km 遥感特征导出
// 输出：每年 7 个 GeoTIFF + 1 个地形（静态）
// ============================================================

// Upload the licensed T2021_county.gpkg geometry as your own EE asset.
// Set this ID explicitly; there is no dependency on the author's private asset.
var counties = ee.FeatureCollection('projects/YOUR_PROJECT/assets/T2021_county');
var china = counties.geometry();

var YEARS = [2019, 2020, 2021, 2022, 2023];
// Preserve historical end-exclusive filterDate cutoffs (Dec 31 and Sep 30).
// Export source units. Python applies the 0.0001 factors exactly once.
var SCALE = 1000;  // 1km
var CRS = 'EPSG:4326';

Map.centerObject(china, 4);
print('县级数量:', counties.size());

// ====== 地形（静态，只导一次）======
var dem = ee.Image('USGS/SRTMGL1_003').clip(china);
var slope = ee.Terrain.slope(dem);
var aspect = ee.Terrain.aspect(dem);
var terrain = dem.rename('dem').addBands(slope.rename('slope')).addBands(aspect.rename('aspect'));

Export.image.toDrive({
  image: terrain.float(), description: 'terrain_1km',
  scale: SCALE, region: china.bounds(), crs: CRS, maxPixels: 1e12
});

// ====== 逐年导出 ======
YEARS.forEach(function(YEAR) {

  // 1. 夜间灯光
  var ntl = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
    .filterDate(YEAR + '-01-01', YEAR + '-12-31')
    .select('avg_rad').median().clip(china).rename('ntl');
  Export.image.toDrive({
    image: ntl.float(), description: 'ntl_' + YEAR + '_1km',
    scale: SCALE, region: china.bounds(), crs: CRS, maxPixels: 1e12
  });

  // 2. MODIS NDVI/EVI
  var mod13 = ee.ImageCollection('MODIS/061/MOD13A3')
    .filterDate(YEAR + '-01-01', YEAR + '-12-31').select(['NDVI','EVI']);
  var veg = mod13.select('NDVI').mean().clip(china).rename('ndvi_mean')
    .addBands(mod13.select('NDVI').max().clip(china).rename('ndvi_max'))
    .addBands(mod13.select('NDVI').reduce(ee.Reducer.stdDev()).clip(china).rename('ndvi_std'))
    .addBands(mod13.select('EVI').mean().clip(china).rename('evi_mean'));
  Export.image.toDrive({
    image: veg.float(), description: 'vegetation_' + YEAR + '_1km',
    scale: SCALE, region: china.bounds(), crs: CRS, maxPixels: 1e12
  });

  // 3. MODIS NPP
  var npp = ee.ImageCollection('MODIS/061/MOD17A3HGF')
    .filterDate(YEAR + '-01-01', YEAR + '-12-31')
    .select('Npp').mean().clip(china).rename('npp');
  Export.image.toDrive({
    image: npp.float(), description: 'npp_' + YEAR + '_1km',
    scale: SCALE, region: china.bounds(), crs: CRS, maxPixels: 1e12
  });

  // 4. 土地覆被（2021-2024 用 WorldCover v200）
  var wc = ee.ImageCollection('ESA/WorldCover/v200').first().clip(china);
  var classValues = [10,20,30,40,50,60,80,90];
  var classNames  = ['tree','shrub','grass','crop','built','bare','water','wetland'];
  var lcBands = classValues.map(function(val, i) {
    return wc.eq(val).reduceResolution({
      reducer: ee.Reducer.mean(), maxPixels: 65536
    }).reproject({crs: CRS, scale: SCALE}).rename(classNames[i] + '_frac');
  });
  Export.image.toDrive({
    image: ee.Image.cat(lcBands).clip(china).float(),
    description: 'landcover_' + YEAR + '_1km',
    scale: SCALE, region: china.bounds(), crs: CRS, maxPixels: 1e12
  });

  // 5. 气候
  var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
    .filterDate(YEAR + '-01-01', YEAR + '-12-31');
  var climate = era5.select('temperature_2m').mean()
      .subtract(273.15).clip(china).rename('temp_mean')
    .addBands(era5.select('total_precipitation_sum').sum().clip(china).rename('precip_sum'))
    .addBands(era5.select('surface_solar_radiation_downwards_sum').mean().clip(china).rename('solar_mean'));
  Export.image.toDrive({
    image: climate.float(), description: 'climate_' + YEAR + '_1km',
    scale: SCALE, region: china.bounds(), crs: CRS, maxPixels: 1e12
  });

  // 6. Sentinel-2 夏季合成
  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate(YEAR + '-06-01', YEAR + '-09-30')
    .filterBounds(china)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30));
  var addIndices = function(img) {
    return img.addBands(img.normalizedDifference(['B8','B4']).rename('ndvi'))
              .addBands(img.normalizedDifference(['B3','B8']).rename('ndwi'))
              .addBands(img.normalizedDifference(['B11','B8']).rename('ndbi'));
  };
  var s2f = s2.map(addIndices).median().clip(china)
    .select(['B2','B3','B4','B8','ndvi','ndwi','ndbi']);
  Export.image.toDrive({
    image: s2f.float(), description: 'sentinel2_' + YEAR + '_summer_1km',
    scale: SCALE, region: china.bounds(), crs: CRS, maxPixels: 1e12
  });

  print('Year ' + YEAR + ' exports submitted');
});
