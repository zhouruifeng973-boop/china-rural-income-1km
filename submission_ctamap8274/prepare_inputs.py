"""Prepare explicit local inputs. Never downloads data or overwrites outputs.

Feature exports retain source scaling; prepare_public_panels applies factors once.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from prepare_public_panels import GROUPS


def align(source, reference, method):
    with rasterio.open(source) as src:
        if src.crs is None:
            raise ValueError(f'Missing CRS: {source}')
        result = np.full((src.count, reference.height, reference.width), np.nan, np.float32)
        for band in range(src.count):
            data = src.read(band+1, masked=True).astype('float32').filled(np.nan)
            if method == Resampling.sum:
                data[~np.isfinite(data) | (data < 0)] = 0
            reproject(data, result[band], src_transform=src.transform, src_crs=src.crs,
                      src_nodata=np.nan, dst_transform=reference.transform,
                      dst_crs=reference.crs, dst_nodata=np.nan, resampling=method)
        return result


def write(path, data, ref):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, 'w', driver='GTiff', width=ref.width, height=ref.height,
                       count=len(data), dtype='float32', crs=ref.crs,
                       transform=ref.transform, nodata=-9999, compress='lzw') as dst:
        dst.write(np.where(np.isfinite(data), data, -9999).astype('float32'))


def province_table(package, output, years=range(2019, 2024)):
    columns = ['province_code','province_std','rural_income_2019yuan',
               'rural_income_nominal','price_index_2019_base']
    tables = []
    for year in years:
        path = Path(package)/'tables/province_calibration'/f'province_calibration_unbounded_{year}.csv'
        table = pd.read_csv(path)[columns].copy()
        if table.isna().any().any() or table.province_code.duplicated().any():
            raise ValueError(f'Incomplete/duplicate provincial data: {year}')
        if (table[columns[2:]] <= 0).any().any():
            raise ValueError('Income and price indices must be positive')
        table.insert(0, 'year', year)
        tables.append(table)
    result = pd.concat(tables, ignore_index=True)
    if Path(output).exists():
        raise FileExistsError(output)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding='utf-8-sig')
    return len(result)


def weights(county_id, smod, population, output, year):
    with rasterio.open(county_id) as ref:
        classes = align(smod, ref, Resampling.nearest)[0]
        pop = align(population, ref, Resampling.sum)[0]
        pop[~np.isfinite(pop) | (pop < 0)] = 0
        rural = np.isin(classes, [11,12,13]) & (ref.read(1) > 0)
        result = np.where(rural, pop, 0)
        write(Path(output)/f'rural_pop_weight_{year}_strict.tif', result[None], ref)
        return {'year':year, 'positive_pixels':int((result > 0).sum()),
                'aligned_population':float(pop.sum(dtype=np.float64)),
                'rural_population':float(result.sum(dtype=np.float64))}


def features(county_id, exports, output, years=range(2019,2024)):
    seen = set()
    with rasterio.open(county_id) as ref:
        for year in years:
            for template, names in GROUPS:
                filename = template.format(y=year)
                if filename in seen:
                    continue
                data = align(Path(exports)/filename, ref, Resampling.bilinear)
                if len(data) != len(names):
                    raise ValueError(f'Wrong band count: {filename}')
                write(Path(output)/filename, data, ref)
                with rasterio.open(Path(output)/filename, 'r+') as dst:
                    for i, name in enumerate(names,1): dst.set_band_description(i,name)
                seen.add(filename)
    return sorted(seen)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    t = sub.add_parser('province')
    t.add_argument('--package',required=True); t.add_argument('--output',required=True)
    w = sub.add_parser('weights')
    for key in ['county-id','smod','population','output']: w.add_argument('--'+key,required=True)
    w.add_argument('--year',type=int,required=True,choices=range(2019,2024))
    f = sub.add_parser('features')
    for key in ['county-id','exports','output']: f.add_argument('--'+key,required=True)
    a = p.parse_args()
    if a.command=='province': result=province_table(a.package,a.output)
    elif a.command=='weights': result=weights(a.county_id,a.smod,a.population,a.output,a.year)
    else: result=features(a.county_id,a.exports,a.output)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
