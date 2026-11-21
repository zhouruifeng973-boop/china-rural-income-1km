"""Compare county attribution with a fresh rasterization, independent of TIFF encoding."""
import hashlib
import json
import numpy as np
import rasterio
import geopandas as gpd
from affine import Affine
from rasterio.features import rasterize


def check(county_path, boundary, grid):
    m=json.loads(grid.read_text(encoding='utf-8'))
    with open(boundary,'rb') as f:
        digest=hashlib.file_digest(f,'sha256').hexdigest()
    if digest != m['source_boundary_sha256']:
        raise ValueError('Boundary hash differs from declared source')
    g=gpd.read_file(boundary).to_crs(m['crs'])
    code=g['区划码'].astype('int64'); province=g['省级码'].astype('int64')
    if code.duplicated().any() or not g.is_valid.all():
        raise ValueError('Invalid boundary geometry or duplicate code')
    keep=(province<710000)&(code>0)
    expected=rasterize(zip(g.geometry[keep],code[keep]),out_shape=(m['height'],m['width']),
                       transform=Affine(*m['transform']),fill=0,dtype='int32',all_touched=False)
    with rasterio.open(county_path) as src:
        if src.shape != expected.shape or src.count != 1 or src.crs != rasterio.crs.CRS.from_string(m['crs']):
            raise ValueError('County grid/CRS/bands differ')
        if not np.allclose(tuple(src.transform)[:6],m['transform'],atol=1e-12,rtol=0):
            raise ValueError('County transform differs')
        actual=src.read(1)
        if not np.issubdtype(actual.dtype,np.integer) or src.nodata != 0:
            raise ValueError('County raster must have integer values and NoData=0')
    changed=int(np.count_nonzero(actual!=expected))
    if changed:
        raise ValueError(f'County attribution differs at {changed} pixels')
    return {'changed_pixels':changed,'content_sha256':hashlib.sha256(expected.astype('<i4').tobytes()).hexdigest(),
            'source_boundary_sha256':digest,'comparison':'pixel values, grid, CRS, integer type and NoData'}
