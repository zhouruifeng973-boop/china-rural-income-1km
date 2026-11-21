"""Portable sensitivity driver; default preflight does not train."""
from pathlib import Path
import argparse,json,os,sys
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['package','panels','features','processed','core','cv','config','output']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--execute',action='store_true');a=p.parse_args()
 home=Path(__file__).resolve().parent;sys.path.insert(0,str(home/'scripts'))
 import numpy as np
 import stage32_v1_5_no_road_spatial_robustness as m
 target=a.package.resolve()/'training_labels/county_training_labels_8274.csv'
 if m.sha256(target)!='073dbc850899192449f042b6d3502cce4620d7f939331844c59e8fe1abdcc482':raise ValueError('Wrong labels')
 cfg=json.loads(a.config.read_text(encoding='utf-8'))
 for section in ['final_maps','multiyear_leave_province_validation']:
  for k,v in dict(hidden1=128,hidden2=64,dropout=.05,epochs=150,pixels_per_county=128,lr=.001,output_regularization=.001).items():
   if cfg['model'][section].get(k)!=v:raise ValueError(f'Baseline parameter differs: {section}.{k}')
 cfg['paths']['corrected_county_income']=str(target)
 load=m.stage27.target_loader(target)
 def panel(year,mask):
  assert mask=='strict'
  with np.load(a.panels/f'{year}.npz',allow_pickle=False) as d:X=d['X'];codes=d['codes'];weights=d['weights'];names=d['band_names'].tolist()
  unique=np.unique(codes);groups=np.searchsorted(unique,codes);order=np.argsort(groups,kind='stable');offset=np.r_[0,np.cumsum(np.bincount(groups))]
  meta=load(year).set_index('county_code').loc[unique].reset_index()
  return dict(X=X,codes=codes,weights=weights,groups=groups,group_indices=[order[offset[i]:offset[i+1]] for i in range(len(unique))],county_codes=unique,target=meta.rural_income_2019yuan.to_numpy(dtype='float32'),county_meta=meta,band_names=names)
 m.weak.load_pixel_panel=panel;m.validation.load_pixel_panel=panel
 for mod in [m.final_maps,m.unbounded]:mod.FEATURES=a.features.resolve();mod.PROCESSED=a.processed.resolve()
 m.COUNTY_PATH=a.features.resolve()/'county_id.tif';m.WEIGHT_DIR=a.processed.resolve()/'rural_weights'
 m.CANONICAL_MAP_ROOT=a.core.resolve();m.CANONICAL_CV_ROOT=a.cv.resolve()
 m.CANONICAL_PREDICTIONS=a.cv.resolve()/'multiyear_leave_province_2019_2023_strict_county_year_predictions.csv'
 m.CANONICAL_MANIFEST=a.cv.resolve()/'multiyear_leave_province_2019_2023_strict_manifest.json'
 m.OUT=a.output.resolve();m.VARIANT_OUT=m.OUT/'parameter_sensitivity'
 m.ROOT=Path(os.path.commonpath([str(a.core.resolve()),str(a.cv.resolve()),str(m.OUT)]))
 files=[target,m.COUNTY_PATH,m.CANONICAL_PREDICTIONS,m.CANONICAL_MANIFEST]
 for y in m.YEARS:files.extend([a.panels/f'{y}.npz',m.WEIGHT_DIR/f'rural_pop_weight_{y}_strict.tif',m.annual_map(m.CANONICAL_MAP_ROOT,y)])
 files.extend([m.change_map(m.CANONICAL_MAP_ROOT,k) for k in ['absolute','percent']])
 for f in files:
  if not f.is_file():raise FileNotFoundError(f)
 assert sum(len(load(y)) for y in m.YEARS)==8274
 print(json.dumps(dict(preflight_passed=True,variants=m.VARIANTS,files_checked=len(files),training_started=False),indent=2))
 if not a.execute:return
 m.OUT.mkdir(parents=True,exist_ok=False)
 m.CONFIG=m.OUT/'workflow.json';m.CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
 lock={str(f.resolve()):m.sha256(f) for f in files}
 (m.OUT/'input_lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
 m.torch.set_num_threads(8);m.prepare_no_road_modules(target,['road_density'])
 for name,settings in m.VARIANTS.items():
  m.run_cv_variant(name,settings,cfg);m.run_map_variant(name,settings,cfg)
 m.run_comparisons()
 assert all(m.sha256(Path(f))==h for f,h in lock.items())
 (m.OUT/'complete.json').write_text(json.dumps(dict(completed=True,variants=4,inputs_unchanged=True,scope='CV/maps/comparison only; not county-uniform baseline or decision narrative'),indent=2),encoding='utf-8')
if __name__=='__main__':main()
