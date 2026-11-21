"""Create change maps from the final bounded annual products, without recalibration."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import rasterio
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['first','last','output']:p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--execute',action='store_true');a=p.parse_args()
 with rasterio.open(a.first) as f,rasterio.open(a.last) as l:
  if (f.shape,f.crs,f.transform)!=(l.shape,l.crs,l.transform):raise ValueError('Grid mismatch')
  if f.count!=1 or l.count!=1:raise ValueError('Expected single-band annual maps')
  count=0
  for _,win in f.block_windows(1):
   x=f.read(1,window=win);y=l.read(1,window=win)
   vx=np.isfinite(x)&(x!=f.nodata);vy=np.isfinite(y)&(y!=l.nodata)
   if np.any((x[vx]<3000)|(x[vx]>80000)) or np.any((y[vy]<3000)|(y[vy]>80000)):raise ValueError('Input is not bounded at 3000–80000')
   count+=int((vx&vy).sum())
  print(json.dumps(dict(preflight_passed=True,common_pixels=count,execute=a.execute)))
  if not a.execute:return
  a.output.mkdir(parents=True,exist_ok=False)
  lock={str(q.resolve()):sha(q) for q in [a.first,a.last]}
  profile=f.profile.copy();profile.update(dtype='float32',count=1,nodata=-9999,compress='deflate',predictor=3)
  paths=[a.output/'rural_income_change_2019_2023_absolute_bounded_2019yuan.tif',a.output/'rural_income_change_2019_2023_percent_bounded.tif']
  with rasterio.open(paths[0],'w',**profile) as d,rasterio.open(paths[1],'w',**profile) as p:
   for _,win in f.block_windows(1):
    x=f.read(1,window=win).astype('float64');y=l.read(1,window=win).astype('float64')
    v=np.isfinite(x)&np.isfinite(y)&(x!=f.nodata)&(y!=l.nodata)&(x>0)
    diff=np.full(x.shape,-9999,dtype='float32');pct=diff.copy()
    diff[v]=y[v]-x[v];pct[v]=100*(y[v]-x[v])/x[v]
    d.write(diff,1,window=win);p.write(pct,1,window=win)
  with rasterio.open(paths[0]) as d,rasterio.open(paths[1]) as p:
   assert (d.shape,d.crs,d.transform)==(f.shape,f.crs,f.transform)
   for _,win in f.block_windows(1):
    x=f.read(1,window=win).astype('float64');y=l.read(1,window=win).astype('float64')
    v=np.isfinite(x)&np.isfinite(y)&(x!=f.nodata)&(y!=l.nodata)&(x>0)
    diff=d.read(1,window=win);pct=p.read(1,window=win)
    assert np.array_equal(diff[v],(y[v]-x[v]).astype('float32'))
    assert np.array_equal(pct[v],(100*(y[v]-x[v])/x[v]).astype('float32'))
    assert np.all(diff[~v]==-9999) and np.all(pct[~v]==-9999)
  assert all(sha(Path(q))==h for q,h in lock.items())
  (a.output/'bounded_change_audit.json').write_text(json.dumps(dict(passed=True,common_pixels=count,inputs=lock,output_readback=True,input_files_unchanged=True),indent=2),encoding='utf-8')
if __name__=='__main__':main()
