"""Cropped-input test of production calibration/change functions, not a data product."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 home=Path(__file__).resolve().parent;sys.path.insert(0,str(home/'scripts'))
 import stage5_create_unbounded_maps as prod
 run=a.run.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 features=out/'features';processed=out/'processed';core=out/'core'
 features.mkdir();(processed/'rural_weights').mkdir(parents=True);core.mkdir()
 with rasterio.open(run/'core/rural_income_2019_initial_2019yuan_strict.tif') as s:
  win=None
  for row in range(0,s.height,256):
   for col in range(0,s.width,256):
    candidate=Window(col,row,min(256,s.width-col),min(256,s.height-row))
    x=s.read(1,window=candidate)
    if (x!=-9999).sum()>1000:win=candidate;break
   if win is not None:break
  if win is None:raise ValueError('No suitable test tile')
 def crop(source,dest):
  with rasterio.open(source) as s:
   arr=s.read(1,window=win);pr=s.profile.copy();pr.update(height=int(win.height),width=int(win.width),transform=s.window_transform(win))
  with rasterio.open(dest,'w',**pr) as d:d.write(arr,1)
  return arr,pr
 county,profile=crop(run/'features/county_id.tif',features/'county_id.tif')
 inputs={}
 for year in [2019,2023]:
  raw,_=crop(run/f'core/rural_income_{year}_initial_2019yuan_strict.tif',core/f'rural_income_{year}_initial_2019yuan_strict.tif')
  w,_=crop(run/f'processed/rural_weights/rural_pop_weight_{year}_strict.tif',processed/f'rural_weights/rural_pop_weight_{year}_strict.tif')
  inputs[year]=(raw,w)
 official=pd.read_csv(run/'processed/province_income_2019_2023_deflated.csv')
 def local_official(year):
  raw,w=inputs[year];valid=(raw!=-9999)&(w>0)&(county>0)
  codes=np.unique((county[valid]//10000)*10000)
  return official[(official.year==year)&official.province_code.isin(codes)].copy().sort_values('province_code')
 prod.OUT=core;prod.FEATURES=features;prod.PROCESSED=processed;prod.official_income=local_official
 reports=[];images={}
 for year in [2019,2023]:
  result=prod.make_year(year,row_block=64)
  raw,w=inputs[year];expected=np.full(raw.shape,-9999,dtype='float32');errors=[]
  for _,r in local_official(year).iterrows():
   valid=(raw!=-9999)&(w>0)&((county//10000)*10000==int(r.province_code))
   weight=w[valid].astype('float64');values=raw[valid].astype('float64')
   factor=float(r.rural_income_2019yuan)/np.average(values,weights=weight)
   expected[valid]=(values*factor).astype('float32')
  with rasterio.open(prod.unbounded_path(year)) as s:
   got=s.read(1);assert s.transform==profile['transform'] and s.crs==profile['crs'] and s.nodata==-9999 and s.dtypes==('float32',)
  assert np.array_equal(got,expected)
  for _,r in local_official(year).iterrows():
   valid=(got!=-9999)&((county//10000)*10000==int(r.province_code))
   errors.append(abs(np.average(got[valid].astype('float64'),weights=w[valid].astype('float64'))/float(r.rural_income_2019yuan)-1))
  assert max(errors)<1e-6
  images[year]=got;reports.append(dict(year=year,pixel_values_exact=True,max_relative_mean_error_after_float32=max(errors),valid_pixels=result['valid_pixels']))
 prod.make_change_maps()
 first,last=images[2019],images[2023];common=(first!=-9999)&(last!=-9999)
 for kind in ['absolute','percent']:
  path=core/('rural_income_change_2019_2023_absolute_unbounded_2019yuan.tif' if kind=='absolute' else 'rural_income_change_2019_2023_percent_unbounded.tif')
  expected=np.full(first.shape,-9999,dtype='float32');delta=last[common]-first[common]
  expected[common]=delta if kind=='absolute' else 100*delta/first[common]
  with rasterio.open(path) as s:
   assert s.transform==profile['transform'] and s.crs==profile['crs'] and s.nodata==-9999
   assert np.array_equal(s.read(1),expected)
 report=dict(passed=True,scope='Real cropped initial predictions; calibration means computed within crop, NOT national product or accuracy evidence',window=list(win.flatten()),years=reports,change_formulas_exact=True,georeferencing_checked=True,nodata_checked=True,production_functions_used=True,full_pipeline_reproduced=False)
 (out/'raster_smoke_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
