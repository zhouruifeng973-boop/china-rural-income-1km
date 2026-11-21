"""Explicit-path uniform-within-county baseline; read-only preflight by default."""
import argparse,json,os,sys
from pathlib import Path
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['core','county-id','weights','output']:p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--years',nargs='+',type=int,default=list(range(2019,2024)))
 p.add_argument('--execute',action='store_true');a=p.parse_args()
 if not a.years or len(set(a.years))!=len(a.years) or not set(a.years)<=set(range(2019,2024)):raise ValueError('Invalid years')
 sys.path.insert(0,str(Path(__file__).resolve().parent/'scripts'))
 import numpy as np
 import pandas as pd
 import rasterio
 import stage26_county_uniform_regularization_sensitivity as m
 m.CANONICAL=a.core.resolve();m.COUNTY_PATH=a.county_id.resolve();m.WEIGHT_DIR=a.weights.resolve();m.BASELINE_OUT=a.output.resolve();m.YEARS=a.years
 m.ROOT=Path(os.path.commonpath([str(m.CANONICAL),str(m.BASELINE_OUT)]))
 paths=[m.COUNTY_PATH]
 with rasterio.open(m.COUNTY_PATH) as c:
  for year in a.years:
   paths += [m.map_path(m.CANONICAL,year),m.WEIGHT_DIR/f'rural_pop_weight_{year}_strict.tif']
   with rasterio.open(paths[-2]) as v,rasterio.open(paths[-1]) as w:
    if not m.aligned(c,v,w):raise ValueError(f'Grid mismatch {year}')
 print(json.dumps(dict(preflight_passed=True,years=a.years,execute=a.execute)))
 if not a.execute:return
 m.BASELINE_OUT.mkdir(parents=True,exist_ok=False)
 lock={str(path):m.sha256(path) for path in paths}
 (m.BASELINE_OUT/'input_lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
 summaries=[]
 for year in a.years:
  table,summary=m.create_county_uniform_baseline(year)
  sums=np.zeros(700001);weights=np.zeros(700001)
  expected=table.set_index('county_code').product_county_weighted_mean_2019yuan.to_dict()
  with rasterio.open(m.COUNTY_PATH) as c,rasterio.open(m.WEIGHT_DIR/f'rural_pop_weight_{year}_strict.tif') as w,rasterio.open(m.map_path(m.CANONICAL,year)) as orig,rasterio.open(m.BASELINE_OUT/f'county_uniform_baseline_{year}_2019yuan_strict.tif') as b:
   assert m.aligned(c,w,orig,b)
   for _,win in b.block_windows(1):
    codes=c.read(1,window=win);pop=w.read(1,window=win).astype('float64');x=orig.read(1,window=win);v=b.read(1,window=win)
    mask=(codes>0)&(codes<=700000)&np.isfinite(pop)&(pop>0)&np.isfinite(x)&(x!=orig.nodata)
    assert np.array_equal(mask,v!=b.nodata)
    wanted=np.array([expected[int(code)] for code in codes[mask]],dtype='float32')
    assert np.array_equal(v[mask],wanted)
    np.add.at(sums,codes[mask],v[mask].astype('float64')*pop[mask]);np.add.at(weights,codes[mask],pop[mask])
  good=np.flatnonzero(weights>0)
  error=float(np.max(np.abs(sums[good]/weights[good]-np.array([expected[int(code)] for code in good]))))
  assert error<.01
  summary['max_abs_county_mean_difference_yuan']=error
  summary['persisted_raster_verified']=True
  summaries.append(summary)
  table.to_csv(m.BASELINE_OUT/f'county_metrics_{year}.csv',index=False)
 pd.DataFrame(summaries).to_csv(m.BASELINE_OUT/'county_uniform_baseline_summary.csv',index=False)
 assert all(m.sha256(Path(path))==h for path,h in lock.items())
 report=dict(passed=True,years=summaries,inputs_unchanged=True,scope='Descriptive within-county comparison, not independent predictive accuracy')
 (m.BASELINE_OUT/'baseline_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
