"""Explicit-path quality/risk driver. Defaults to read-only checks."""
from pathlib import Path
import argparse,hashlib,json,sys
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['package','core','features','output']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--execute',action='store_true');a=p.parse_args()
 home=Path(__file__).resolve().parent;sys.path.insert(0,str(home/'scripts'))
 import pandas as pd
 import rasterio
 import stage6_create_quality_layers as q
 import stage7_uncertainty_proxy_layers as risk
 target=a.package.resolve()/'training_labels/county_training_labels_8274.csv'
 with target.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=='073dbc850899192449f042b6d3502cce4620d7f939331844c59e8fe1abdcc482'
 assert len(pd.read_csv(target))==8274
 paths=[a.features/'county_id.tif',target]
 for y in range(2019,2024):
  paths += [a.core/f'rural_income_{y}_final_calibrated_unbounded_2019yuan_strict.tif',a.core/f'rural_income_{y}_final_calibrated_2019yuan_strict.tif',a.core/f'province_calibration_unbounded_{y}.csv']
 for path in paths:
  if not path.is_file():raise FileNotFoundError(path)
 with rasterio.open(a.features/'county_id.tif') as ref:
  for path in paths:
   if path.suffix!='.tif':continue
   with rasterio.open(path) as s:assert (s.shape,s.crs,s.transform)==(ref.shape,ref.crs,ref.transform)
 q.FEATURES=a.features.resolve()
 risk.income_path=lambda y:a.core.resolve()/f'rural_income_{y}_final_calibrated_unbounded_2019yuan_strict.tif'
 print(json.dumps(dict(preflight_passed=True,inputs=len(paths),execute=a.execute,scope='26 quality and 12 risk layers; excludes county-uniform baseline and bounded changes'),indent=2))
 if not a.execute:return
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 def sha(path):
  with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
 locked={str(path.resolve()):sha(path) for path in paths}
 (out/'input_lock.json').write_text(json.dumps(locked,indent=2),encoding='utf-8')
 sys.argv=[q.__name__,'--stage5-root',str(a.core.resolve()),'--package-root',str(out),'--target-path',str(target)];q.main()
 sys.argv=[risk.__name__,'--package-root',str(out)];risk.main()
 assert len(list(q.QUALITY.glob('*.tif')))==26
 assert len(list(risk.OUT.glob('*.tif')))==12
 assert all(sha(Path(path))==h for path,h in locked.items())
 (out/'auxiliary_complete.json').write_text(json.dumps(dict(completed=True,quality_layers=26,risk_layers=12,input_hashes_unchanged=True),indent=2),encoding='utf-8')
if __name__=='__main__':main()
