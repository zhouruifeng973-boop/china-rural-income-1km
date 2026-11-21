"""Small real-input wiring test; partial bags are not scientific validation."""
import csv,hashlib,json,zipfile
import numpy as np
def head(z,name,n):
 with z.open(name+'.npy') as f:
  v=np.lib.format.read_magic(f);shape,order,dtype=np.lib.format._read_array_header(f,v)
  if order or dtype.hasobject:raise ValueError('Unsupported NPY layout')
  n=min(n,shape[0]);width=int(np.prod(shape[1:])) if len(shape)>1 else 1
  data=f.read(n*width*dtype.itemsize)
  return np.frombuffer(data,dtype=dtype).reshape((n,)+shape[1:]).copy()
def run(package,panels,output,settings):
 import torch
 import stage3_spatiotemporal_validation as val
 from stage27_multiyear_leave_province_validation import target_loader
 output.mkdir(parents=True,exist_ok=False)
 label=package/'training_labels/county_training_labels_8274.csv'
 with label.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 if h!='073dbc850899192449f042b6d3502cce4620d7f939331844c59e8fe1abdcc482':raise ValueError('Incorrect final labels')
 frozen_names=json.loads((package/'metadata/current_model_metadata.json').read_text(encoding='utf-8'))['input_band_names']
 names=[n for n in frozen_names if n not in ['road_density','year_scaled']]
 if len(names)!=27:raise ValueError('Expected 27 spatial variables')
 with zipfile.ZipFile(panels/'2019.npz') as z:
  stored=head(z,'band_names',100).tolist()
  X=head(z,'X',4096)[:,[stored.index(n) for n in names]]
  c=head(z,'codes',4096);w=head(z,'weights',4096)
 if len(X)!=len(c) or len(c)!=len(w) or not (w>0).all():raise ValueError('Invalid panel')
 codes,groups=np.unique(c,return_inverse=True)
 labels=target_loader(label)(2019).set_index('county_code')
 target=labels.loc[codes].rural_income_2019yuan.to_numpy(dtype='float32')
 sample={2019:dict(X=X,weights=w,groups=groups,target=target)}
 torch.set_num_threads(2)
 kwargs={k:settings[k] for k in ['lr','hidden1','hidden2','dropout','output_regularization']}
 model,imputer,scaler,stats=val.fit_bag_model(sample,[2019],epochs=2,seed=42,**kwargs)
 model.eval()
 with torch.no_grad():
  pred=model(torch.tensor(scaler.transform(imputer.transform(val.add_year_column(X,2019))),dtype=torch.float32)).numpy().reshape(-1)
 if not np.isfinite(pred).all():raise ValueError('Non-finite predictions')
 totals=np.bincount(groups,weights=w)
 agg=np.bincount(groups,weights=w*pred)/totals
 manual=np.array([np.sum(pred[groups==i].astype('float64')*w[groups==i])/np.sum(w[groups==i],dtype='float64') for i in range(len(codes))])
 if not np.allclose(agg,manual,rtol=1e-6):raise ValueError('Aggregation mismatch')
 with (output/'partial_bag_predictions_NOT_VALIDATION.csv').open('w',newline='',encoding='utf-8-sig') as f:
  writer=csv.writer(f);writer.writerow(['county_code','sample_pixels','sample_population_weight','partial_bag_prediction','full_county_label_reference'])
  for i,code in enumerate(codes):writer.writerow([int(code),int((groups==i).sum()),float(totals[i]),float(agg[i]),float(target[i])])
 with (output/'partial_bag_predictions_NOT_VALIDATION.csv').open(encoding='utf-8-sig') as f: reread=list(csv.DictReader(f))
 assert len(reread)==len(codes)
 assert np.allclose([float(r['partial_bag_prediction']) for r in reread],agg)
 report=dict(passed=True,entrypoint='run_submission.py --smoke',label_sha256=h,year=2019,pixels=len(X),partial_county_bags=len(codes),spatial_features=names,epochs=2,parameters=kwargs,aggregation_crosscheck=True,output_readback=True,model_saved=False,full_training_run=False,scope='First 4096 raster-order pixels, partial bags, training-set predictions only. Not accuracy validation. No province calibration or national GeoTIFF test.')
 (output/'smoke_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2))
