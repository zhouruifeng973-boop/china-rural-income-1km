"""New synthetic rasters -> inputs -> MLP -> calibration -> maps.

No existing NPZ, trained model, income product or private input is read.
Tests software integration, NOT national accuracy or Earth Engine availability.
"""
import argparse, hashlib, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import box
import torch

HOME=Path(__file__).resolve().parent
sys.path.insert(0,str(HOME/'scripts'))
from prepare_inputs import features, weights, province_table
from prepare_public_panels import GROUPS, NAMES, SCALE, build_panel
from check_county_content import check
from feature_ablation import patch_weak_pixel_loader, patch_final_map_features
import stage3_spatiotemporal_validation as val
import stage5_final_maps as fm
import stage5_create_unbounded_maps as products
import stage27_multiyear_leave_province_validation as multi
import stage8_ensemble_uncertainty as ensemble


def run(root):
    root.mkdir(parents=True,exist_ok=False)
    raw=root/'new_raw_inputs';raw.mkdir()
    feat=root/'features';feat.mkdir()
    proc=root/'processed';proc.mkdir()
    core=root/'core';core.mkdir()
    rng=np.random.default_rng(20261121)
    transform=from_origin(100,30,.01,.01)
    shape=(20,20); codes=np.zeros(shape,dtype='int32')
    polygons=[];county_codes=[];provinces=[]
    for i in range(5):
        for j in range(4):
            code=(11+i)*10000+101+j
            codes[i*4:(i+1)*4,j*5:(j+1)*5]=code
            polygons.append(box(100+j*.05,30-(i+1)*.04,100+(j+1)*.05,30-i*.04))
            county_codes.append(code);provinces.append((11+i)*10000)
    boundary=raw/'synthetic.gpkg'
    gpd.GeoDataFrame({'区划码':county_codes,'省级码':provinces},geometry=polygons,crs='EPSG:4326').to_file(boundary,driver='GPKG')
    grid=raw/'grid.json'
    grid.write_text(json.dumps(dict(width=20,height=20,transform=list(transform)[:6],crs='EPSG:4326',source_boundary_sha256=hashlib.sha256(boundary.read_bytes()).hexdigest())),encoding='utf-8')
    def raster(path,data,nodata=-9999,compression='lzw'):
        arr=np.asarray(data)
        if arr.ndim==2:arr=arr[None]
        with rasterio.open(path,'w',driver='GTiff',height=20,width=20,count=len(arr),dtype=str(arr.dtype),transform=transform,crs='EPSG:4326',nodata=nodata,compress=compression) as dst:dst.write(arr)
    county=feat/'county_id.tif';raster(county,codes,0)
    county_audit=check(county,boundary,grid)
    alternate=raw/'county_other_encoding.tif';raster(alternate,codes,0,'deflate')
    check(alternate,boundary,grid)
    wrong=codes.copy();wrong[0,0]=codes[-1,-1];raster(raw/'wrong.tif',wrong,0)
    try:check(raw/'wrong.tif',boundary,grid)
    except ValueError: county_audit['wrong_attribution_rejected']=True
    else:raise AssertionError('Changed attribution accepted')
    smod=np.full(shape,11,dtype='int16');smod[::4,::5]=30
    raster(raw/'smod.tif',smod,-200)
    tables=raw/'tables/province_calibration';tables.mkdir(parents=True)
    prepared={};population_audits=[]
    for y in range(2019,2024):
        for template,names in GROUPS:
            path=raw/template.format(y=y)
            if path.exists():continue
            data=rng.uniform(.2,.8,(len(names),*shape)).astype('float32')
            for b,name in enumerate(names):
                if name in SCALE:data[b]*=10000
            raster(path,data)
        pop=rng.uniform(1,100,shape).astype('float32');pop[0,0]=-1;pop[1,0]=np.nan
        raster(raw/f'population_{y}.tif',pop)
        population_audits.append(weights(county,raw/'smod.tif',raw/f'population_{y}.tif',proc/'rural_weights',y))
        rows=[]
        for i in range(5):
            income=18000+1000*i+500*(y-2019)
            rows.append(dict(province_code=(11+i)*10000,province_std=f'SYNTHETIC_{i}',rural_income_2019yuan=income,rural_income_nominal=income,price_index_2019_base=1))
        pd.DataFrame(rows).to_csv(tables/f'province_calibration_unbounded_{y}.csv',index=False)
    features(county,raw,feat)
    assert province_table(raw,proc/'province_income_2019_2023_deflated.csv')==25
    panels=root/'panels';panels.mkdir()
    for y in range(2019,2024):
        d=build_panel(feat,proc/'rural_weights',county,county_codes,y)
        np.savez(panels/f'{y}.npz',**d)
        unique,groups=np.unique(d['codes'],return_inverse=True)
        target=np.array([18000+1000*(int(c)//10000-11)+500*(y-2019)+(int(c)%100-2)*100 for c in unique],dtype='float32')
        order=np.argsort(groups,kind='stable');offset=np.r_[0,np.cumsum(np.bincount(groups))]
        prepared[y]=dict(X=d['X'],weights=d['weights'],groups=groups,target=target,
                         band_names=d['band_names'].tolist(),county_codes=unique,
                         group_indices=[order[offset[i]:offset[i+1]] for i in range(len(unique))])
        prepared[y]['county_meta']=pd.DataFrame({'county_code':unique,
            'province_std':[f'SYNTHETIC_{int(c)//10000-11}' for c in unique]})
    adapter=SimpleNamespace(load_pixel_panel=lambda y,m:dict(prepared[y]))
    patch_weak_pixel_loader(adapter,['road_density'])
    assert adapter.load_pixel_panel(2019,'strict')['X'].shape[1]==27
    # The legacy 28-column representation must yield exactly the same matrix.
    old=dict(prepared[2019]);old['X']=np.column_stack([old['X'],np.ones(len(old['X']))]);old['band_names']=NAMES+['road_density']
    legacy=SimpleNamespace(load_pixel_panel=lambda y,m:dict(old));patch_weak_pixel_loader(legacy,['road_density'])
    np.testing.assert_array_equal(legacy.load_pixel_panel(2019,'strict')['X'],prepared[2019]['X'])
    broken=dict(prepared[2019]);broken['band_names']=NAMES[:-1];broken['X']=broken['X'][:,:-1]
    bad=SimpleNamespace(load_pixel_panel=lambda y,m:broken);patch_weak_pixel_loader(bad,['road_density'])
    try:bad.load_pixel_panel(2019,'strict')
    except ValueError:pass
    else:raise AssertionError('Incomplete schema accepted')
    val.load_pixel_panel=adapter.load_pixel_panel
    torch.set_num_threads(2)
    fm.FEATURES=feat;fm.PROCESSED=proc;fm.OUT=core;fm.MODEL_DIR=core/'model'
    products.FEATURES=feat;products.PROCESSED=proc;products.OUT=core
    patch_final_map_features(fm,['road_density'])
    model,imputer,scaler,meta=fm.train_and_save(128,150,.001,.001,128,64,.05,['road_density'])
    model,imputer,scaler,_=fm.load_model()
    audits=[]
    for y in range(2019,2024):
        fm.predict_initial(y,model,imputer,scaler,128)
        audits.append(products.make_year(y))
        with rasterio.open(products.unbounded_path(y)) as src,rasterio.open(proc/'rural_weights'/f'rural_pop_weight_{y}_strict.tif') as wsrc:
            values=src.read(1);w=wsrc.read(1)
            assert np.all(values[w<=0]==-9999)
            official=fm.official_income(y)
            for row in official.itertuples():
                mask=(codes//10000*10000==row.province_code)&(w>0)
                mean=np.average(values[mask].astype('float64'),weights=w[mask])
                assert abs(mean-row.rural_income_2019yuan)<.01
    # Execute the production change-map writer, not a separate formula implementation.
    products.make_change_maps()
    cv_options=dict(mask_name='strict',pixels_per_county=128,epochs=150,lr=.001,
                    hidden1=128,hidden2=64,dropout=.05,output_regularization=.001)
    validation=root/'validation';validation.mkdir()
    for name,result in [('leave_year',val.run_leave_year(**cv_options)),
                        ('leave_province_2020',val.run_leave_province(year=2020,folds=5,**cv_options))]:
        result[0].to_csv(validation/f'{name}_SYNTHETIC.csv',index=False)
        assert np.isfinite(result[0]['pred_weak_2019yuan']).all()
    samples={y:val.sample_year_bags(y,'strict',128,42+y) for y in prepared}
    for y in samples:samples[y]['county_meta']=prepared[y]['county_meta']
    count=0
    for i in range(5):
        held={f'SYNTHETIC_{i}'}
        m,imp,scl,stats=multi.fit_fold(samples,list(prepared),held,150,.001,42+i,.001,128,64,.05)
        assert stats['train_county_years']==80
        for y in prepared:
            ids,pred,_,_=multi.aggregate_held_out(m,imp,scl,prepared[y],y,held)
            assert len(ids)==4 and np.isfinite(pred).all()
            count+=len(ids)
    assert count==100
    ensemble.FEATURES=feat;ensemble.PROCESSED=proc;ensemble.MODEL_OUT=root/'ensemble_models'
    ensemble.OUT=root/'ensemble_maps';ensemble.OUT.mkdir()
    ensemble.BAND_NAMES=list(fm.BAND_NAMES);ensemble.feature_specs=fm.feature_specs
    members=[ensemble.train_member(i+1,seed,128,150,.001,128,64,.05,.001)
             for i,seed in enumerate([1042,2039,3036,4033,5030])]
    for y in prepared:
        _,factors=ensemble.calculate_member_factors(y,members,128)
        ensemble.create_year_layers(y,members,factors,128)
        with rasterio.open(ensemble.std_path(y)) as src:
            data=src.read(1,masked=True).compressed()
            assert np.isfinite(data).all() and (data>=0).all()
    report=dict(passed=True,synthetic=True,county_years=100,years=list(range(2019,2024)),spatial_features=27,
                epochs=150,network=[28,128,64,1],county_audit=county_audit,population=population_audits,
                source='new deterministic synthetic rasters and labels',independent_environment=sys.prefix,
                checks=['feature alignment','population weights','province CSV conversion','27/28 feature compatibility',
                        'missing feature rejection','county encoding equivalence','county pixel change rejection',
                        'training and saved model reload','five annual predictions','province weighted means','NoData','change writer',
                        'five leave-year folds','five 2020 leave-province folds','five multi-year province folds','five-member ensemble, 15 layers'],
                scope='Synthetic production-function integration. Not Earth Engine execution, national reproduction, scientific accuracy validation or full CLI execution.')
    (root/'test_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
