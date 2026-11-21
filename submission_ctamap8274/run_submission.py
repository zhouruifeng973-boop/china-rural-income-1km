"""Explicit-path launcher. Default checks only; --execute starts the selected stage."""
import argparse,json,sys
from pathlib import Path
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--stage',choices=['validation','multiyear','core','products','ensemble'],required=True)
 for key in ['package','panels','features','processed','output']:p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--execute',action='store_true')
 p.add_argument('--smoke',action='store_true',help='Two-epoch partial-bag integration test, not validation')
 p.add_argument('--check-parsers',action='store_true',help='Write isolated config and test all parsers, never train')
 a=p.parse_args()
 home=Path(__file__).resolve().parent
 sys.path.insert(0,str(home/'scripts'))
 import numpy as np
 import stage3_weak_supervised_downscaling as weak
 import stage3_spatiotemporal_validation as val
 import stage27_multiyear_leave_province_validation as multi
 import stage5_final_maps as fm
 import stage5_create_unbounded_maps as products
 import stage8_ensemble_uncertainty as ensemble
 from feature_ablation import patch_weak_pixel_loader
 target=a.package.resolve()/'training_labels/county_training_labels_8274.csv'
 frozen=home/'config/historical_development_frozen_config.json'
 settings=json.loads(frozen.read_text(encoding='utf-8'))['parameters']
 required={'hidden1':128,'hidden2':64,'epochs':150,'pixels_per_unit':128,'lr':.001,'dropout':.05,'output_regularization':.001}
 if any(settings.get(k)!=v for k,v in required.items()):raise ValueError('Frozen parameters changed')
 load=multi.target_loader(target)
 def panel(year,mask):
  if mask!='strict':raise ValueError('Only frozen strict mask supported')
  with np.load(a.panels/f'{year}.npz',allow_pickle=False) as d:
   X=d['X'];codes=d['codes'];weights=d['weights'];names=d['band_names'].tolist()
  unique=np.unique(codes);groups=np.searchsorted(unique,codes);order=np.argsort(groups,kind='stable');offset=np.r_[0,np.cumsum(np.bincount(groups))]
  meta=load(year).set_index('county_code').loc[unique].reset_index()
  return dict(X=X,codes=codes,weights=weights,groups=groups,group_indices=[order[offset[i]:offset[i+1]] for i in range(len(unique))],county_codes=unique,target=meta.rural_income_2019yuan.to_numpy(dtype='float32'),county_meta=meta,band_names=names)
 weak.load_targets=load;weak.load_pixel_panel=panel;val.load_pixel_panel=panel
 # ROOT is used for provenance-relative target reporting in this module.
 multi.ROOT=a.package.resolve()
 out=a.output.resolve();cfg=out/'workflow.json'
 for m in [fm,products,ensemble]:
  m.FEATURES=a.features.resolve();m.PROCESSED=a.processed.resolve()
 fm.OUT=out/'core';fm.MODEL_DIR=fm.OUT/'model';products.OUT=fm.OUT
 common=['--config',str(cfg),'--frozen-config',str(frozen)]
 if a.stage=='validation':
  patch_weak_pixel_loader(val,['road_density']);module=val;args=common+['--run','all','--output-root',str(out/'validation')]
 elif a.stage=='multiyear':module=multi;args=common
 elif a.stage=='core':module=fm;args=common
 elif a.stage=='products':module=products;args=[]
 else:module=ensemble;args=common+['--target-path',str(target),'--package-root',str(out/'ensemble/package'),'--model-output',str(out/'ensemble/models'),'--reference-model-metadata',str(out/'core/model/model_metadata.json')]
 model=dict(mask='strict',epochs=150,lr=.001,hidden1=128,hidden2=64,dropout=.05,output_regularization=.001,pixels_per_county=128,threads=8)
 config={'project':{'version':'20261119-ctamap8274'},'paths':{'corrected_county_income':str(target),'parent_county_income':str(target)},'model':{'feature_ablation':{'exclude_features':['road_density']},'spatiotemporal_validation':dict(model,run='all',year=2020,province_folds=5,province_pixels=128,year_pixels=128),'multiyear_leave_province_validation':dict(model,years=list(range(2019,2024)),folds=5,output_root=str(out/'multiyear'))}}
 config['model']['final_maps']=dict(model,years=list(range(2019,2024)),reuse_model=False,row_block=128)
 config['model']['ensemble_uncertainty']=dict(model,years=list(range(2019,2024)),members=5,seeds='1042,2039,3036,4033,5030')
 for y in range(2019,2024):
  if not (a.panels/f'{y}.npz').is_file():raise FileNotFoundError(a.panels/f'{y}.npz')
 if sum(len(load(y)) for y in range(2019,2024))!=8274:raise ValueError('Wrong training cohort')
 if a.smoke:
  if a.execute or a.check_parsers:raise ValueError('Smoke mode must be used alone')
  from portable_smoke import run
  run(a.package.resolve(),a.panels.resolve(),out,required)
  return
 print(json.dumps(dict(stage=a.stage,module=module.__name__,arguments=args,parameters=required,training_started=False,check_scope='Imports, cohort count, parameter and module wiring only; no numerical execution'),indent=2))
 if a.check_parsers:
  if a.execute:raise ValueError('Parser check cannot be combined with execute')
  out.mkdir(parents=True,exist_ok=False)
  cfg.write_text(json.dumps(config,indent=2),encoding='utf-8')
  checks={};original_argv=sys.argv[:]
  commands={
   'validation':(val,common+['--run','all','--output-root',str(out/'validation')]),
   'multiyear':(multi,common),
   'core':(fm,common),
   'ensemble':(ensemble,common+['--target-path',str(target),'--package-root',str(out/'ensemble/package'),'--model-output',str(out/'ensemble/models'),'--reference-model-metadata',str(out/'core/model/model_metadata.json')])}
  try:
   for stage,(mod,argv) in commands.items():
    sys.argv=[mod.__name__]+argv;parsed=vars(mod.parse_args())
    if Path(parsed['config']).resolve()!=cfg:raise ValueError('Wrong workflow path')
    if Path(parsed['frozen_config']).resolve()!=frozen:raise ValueError('Wrong frozen configuration')
    if stage in ['validation','core','ensemble']:
     for key in ['epochs','lr','hidden1','hidden2','dropout','output_regularization']:
      if parsed[key]!=required[key]:raise ValueError(f'{stage}: {key} differs')
    if stage=='validation':
     assert parsed['province_pixels']==128 and parsed['year_pixels']==128
     assert Path(parsed['output_root']).resolve()==out/'validation'
    if stage=='ensemble':
     assert ensemble.parse_seed_list(parsed['seeds'],parsed['members'])==[1042,2039,3036,4033,5030]
     assert Path(parsed['package_root']).resolve()==out/'ensemble/package'
    checks[stage]=parsed
   assert products.OUT==out/'core' and fm.MODEL_DIR==out/'core/model'
   assert products.FEATURES==a.features.resolve() and products.PROCESSED==a.processed.resolve()
   checks['products']={'config_interface':'module globals, no command parser','output':str(products.OUT),'features':str(products.FEATURES),'processed':str(products.PROCESSED)}
  finally:sys.argv=original_argv
  report=dict(passed=True,training_started=False,numerical_pipeline_executed=False,checks=checks)
  (out/'parser_check.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
  print('All four parsers and product paths passed; no computation started')
  return
 if not a.execute:return
 import hashlib,os
 if not out.exists():out.mkdir(parents=True)
 marker=out/'.submission_run.json'
 identity={'package':str(a.package.resolve()),'panels':str(a.panels.resolve()),'features':str(a.features.resolve()),'processed':str(a.processed.resolve()),'labels_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'frozen_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest()}
 if marker.exists():
  if json.loads(marker.read_text())!=identity:raise ValueError('Run inputs differ from recorded identity')
 elif any(out.iterdir()):raise FileExistsError('Output must be empty or belong to this launcher')
 else:marker.write_text(json.dumps(identity,indent=2),encoding='utf-8')
 stages={'validation':out/'validation','multiyear':out/'multiyear','core':out/'core','ensemble':out/'ensemble'}
 if a.stage in stages and stages[a.stage].exists():raise FileExistsError('Stage output already exists; use a new run directory')
 if a.stage=='products':
  for y in range(2019,2024):
   if not products.initial_path(y).is_file():raise FileNotFoundError(products.initial_path(y))
   if products.unbounded_path(y).exists():raise FileExistsError(products.unbounded_path(y))
 if a.stage=='ensemble' and not (out/'core/model/model_metadata.json').is_file():raise FileNotFoundError('Run core before ensemble')
 # Existing reporting code requires package/target/output paths beneath ROOT.
 # Different-drive runs fail explicitly before computation, never silently reroute.
 ensemble.ROOT=Path(os.path.commonpath([str(a.package.resolve()),str(out)]))
 cfg.write_text(json.dumps(config,indent=2),encoding='utf-8')
 sys.argv=[module.__name__]+args
 module.main()
 (out/(a.stage+'_complete.json')).write_text(json.dumps({'completed':True,'stage':a.stage,'identity':identity},indent=2),encoding='utf-8')
if __name__=='__main__':main()
