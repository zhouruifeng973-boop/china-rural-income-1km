"""Rebuild CTAmap county IDs using only boundary and explicit grid metadata."""
import argparse, hashlib, json
from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from affine import Affine

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--boundary',required=True);p.add_argument('--grid',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    m=json.loads(Path(a.grid).read_text(encoding='utf-8'))
    if Path(a.output).exists():raise FileExistsError('Use a new output path; existing products are never overwritten')
    assert sha(a.boundary)==m['source_boundary_sha256'],'Boundary differs from frozen input'
    g=gpd.read_file(a.boundary).to_crs(m['crs'])
    g['code']=g['区划码'].astype(int)
    assert not g.code.duplicated().any() and g.is_valid.all()
    g=g[(g['省级码'].astype(int)<710000)&(g.code>0)].copy()
    arr=rasterize(zip(g.geometry,g.code),out_shape=(m['height'],m['width']),transform=Affine(*m['transform']),fill=0,dtype='int32',all_touched=False)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with rasterio.open(a.output,'w',driver='GTiff',height=m['height'],width=m['width'],count=1,dtype='int32',crs=m['crs'],transform=Affine(*m['transform']),nodata=0,compress='lzw') as d:d.write(arr,1)
    print(json.dumps({'mainland_features':len(g),'output':a.output,'sha256':sha(a.output)},ensure_ascii=False))
if __name__=='__main__':main()
