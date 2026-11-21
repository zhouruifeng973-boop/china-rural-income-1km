"""Explicit-input panel preparation; default is read-only preflight, no training."""
from pathlib import Path
from contextlib import ExitStack
import argparse,csv,hashlib,json
import numpy as np
import rasterio
from rasterio.windows import Window

GROUPS=[('ntl_{y}_1km.tif',['ntl']),('vegetation_{y}_1km.tif',['ndvi_mean','ndvi_max','ndvi_std','evi_mean']),('npp_{y}_1km.tif',['npp']),('landcover_{y}_1km.tif',['tree_frac','shrub_frac','grass_frac','crop_frac','built_frac','bare_frac','water_frac','wetland_frac']),('terrain_1km.tif',['dem','slope','aspect']),('climate_{y}_1km.tif',['temp_mean','precip_sum','solar_mean']),('sentinel2_{y}_summer_1km.tif',['B2','B3','B4','B8','ndvi_s2','ndwi','ndbi'])]
NAMES=[n for _,names in GROUPS for n in names]
SCALE={'ndvi_mean','ndvi_max','ndvi_std','evi_mean','npp','B2','B3','B4','B8'}
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['package','features','weights','county-id']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--output',type=Path);p.add_argument('--prepare',action='store_true')
 a=p.parse_args(); package=a.package.resolve()
 label=package/'training_labels/county_training_labels_8274.csv'
 if sha(label)!='073dbc850899192449f042b6d3502cce4620d7f939331844c59e8fe1abdcc482':raise ValueError('Not the frozen final label file')
 with label.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
 if len(rows)!=8274 or len({(r['year'],r['county_code']) for r in rows})!=8274:raise ValueError('Invalid cohort')
 grid=json.loads((package/'reproducibility/boundary/grid.json').read_text(encoding='utf-8'))
 with rasterio.open(a.county_id) as ref:
  shape=ref.shape;transform=ref.transform;crs=ref.crs;county=ref.read(1)
 if shape!=(grid['height'],grid['width']) or not np.allclose(tuple(transform)[:6],grid['transform'],rtol=0,atol=1e-12) or crs!=rasterio.crs.CRS.from_string(grid['crs']):raise ValueError('County grid differs')
 # Header equality alone cannot establish county attribution equivalence.
 reference=package/'reproducibility/boundary/rebuild_verification.json'
 valid_hashes={'90befebebfe82b13c67e715ac2adbc3e0d604220493fd3cb723ee42f394d6b30','fc4d22bd53f5574742df893e5b4bf53a594239f37ff7a90584ae51d58afa6947'}
 if sha(a.county_id) not in valid_hashes:raise ValueError('County-ID encoding not audited; independently compare reconstructed pixel values first')
 report=[]
 for y in range(2019,2024):
  codes=np.array([int(r['county_code']) for r in rows if int(r['year'])==y])
  with ExitStack() as stack:
   wsrc=stack.enter_context(rasterio.open(a.weights/f'rural_pop_weight_{y}_strict.tif'))
   specs=[(stack.enter_context(rasterio.open(a.features/f.format(y=y))),names) for f,names in GROUPS]
   for s,n in [(wsrc,['weight'])]+specs:
    if s.shape!=shape or s.transform!=transform or s.crs!=crs or s.count!=len(n):raise ValueError(f'Grid/bands mismatch: {s.name}')
   w=wsrc.read(1);mask=np.isfinite(w)&(w>0)&np.isin(county,codes)
   if set(np.unique(county[mask]))!=set(codes):raise ValueError(f'Missing positive-weight counties: {y}')
   report.append(dict(year=y,county_years=len(codes),rural_pixels=int(mask.sum())))
 if a.prepare:
  if a.output is None:raise ValueError('--prepare requires a new --output directory')
  a.output.mkdir(parents=True,exist_ok=False)
  for rec in report:
   y=rec['year']; codes=np.array([int(r['county_code']) for r in rows if int(r['year'])==y])
   with ExitStack() as stack:
    wsrc=stack.enter_context(rasterio.open(a.weights/f'rural_pop_weight_{y}_strict.tif'));w=wsrc.read(1)
    mask=np.isfinite(w)&(w>0)&np.isin(county,codes)
    specs=[(stack.enter_context(rasterio.open(a.features/f.format(y=y))),names) for f,names in GROUPS]
    X=np.empty((rec['rural_pixels'],27),dtype='float32');offset=0
    for start in range(0,shape[0],128):
     h=min(128,shape[0]-start);take=mask[start:start+h].ravel();n=int(take.sum())
     if not n:continue
     columns=[]
     for s,names in specs:
      for b,name in enumerate(names,1):
       arr=s.read(b,window=Window(0,start,shape[1],h)).astype('float32')
       if s.nodata is not None:arr[arr==s.nodata]=np.nan
       if name in SCALE:arr*=np.float32(.0001)
       columns.append(arr.ravel()[take])
     X[offset:offset+n]=np.column_stack(columns);offset+=n
    if offset!=len(X):raise ValueError('Pixel count mismatch')
    np.savez(a.output/f'{y}.npz',X=X,codes=county[mask],weights=w[mask],band_names=np.array(NAMES))
  (a.output/'prepare_complete.json').write_text(json.dumps(dict(years=report,features=NAMES,label_sha256=sha(label)),indent=2),encoding='utf-8')
 print(json.dumps(dict(preflight_passed=True,training_started=False,panels_written=a.prepare,spatial_features=len(NAMES),years=report),indent=2))
if __name__=='__main__':main()
