"""
This script exists simply to keep the notebooks
'Extract_training_data.ipynb' and 'Predict.ipynb' tidy by
not clutering them up with custom training data functions.
"""

import pyproj
import dask
import hdstats
import datacube
import numpy as np
import sys
import xarray as xr
import warnings
import dask.array as da
from datacube.utils.geometry import assign_crs
from datacube.testutils.io import rio_slurp_xarray
from odc.algo import randomize, reshape_for_geomedian, xr_reproject, xr_geomedian
from odc.algo._dask import reshape_yxbt

sys.path.append('../../Scripts')
from deafrica_bandindices import calculate_indices
from deafrica_temporal_statistics import xr_phenology, temporal_statistics
from deafrica_classificationtools import HiddenPrints
from deafrica_datahandling import load_ard

warnings.filterwarnings("ignore")

# Edited with Chad on 14 Dec 2020
def xr_tmad(ds, gm, axis='time', where=None, **kw):
    """
    :param ds: xr.Dataset|xr.DataArray|numpy array
    Other parameters:
    **kwargs -- passed on to pcm.gnmpcm
       maxiters   : int         1000
       eps        : float       0.0001
       num_threads: int| None   None
    """

    import hdstats
    def tmad(arr, gm, **kw):
        """
        arr: a high dimensional numpy array where the last dimension will be reduced. 
    
        returns: a numpy array with one less dimension than input.
        """
        nt = kw.pop('num_threads', None)
        emad = hdstats.emad_pcm(arr, gm, num_threads=nt)[:,:, np.newaxis]
        smad = hdstats.smad_pcm(arr, gm, num_threads=nt)[:,:, np.newaxis]
        bcmad = hdstats.bcmad_pcm(arr, gm, num_threads=nt)[:,:, np.newaxis]
        tmad=np.concatenate([emad, smad, bcmad], axis=-1)
        print(tmad.shape)
        return tmad
        # removed gm from the concatenate to see if it fixes dimension problem
        # numpy array (y, x, 3)

    def norm_input(ds, axis):
        if isinstance(ds, xr.DataArray):
            xx = ds
            if len(xx.dims) != 4:
                raise ValueError("Expect 4 dimensions on input: y,x,band,time")
            if axis is not None and xx.dims[3] != axis:
                raise ValueError(f"Can only reduce last dimension, expect: y,x,band,{axis}")
            return None, xx, xx.data
        elif isinstance(ds, xr.Dataset):
            xx = reshape_for_geomedian(ds, axis)
            return ds, xx, xx.data
        else:  # assume numpy or similar
            xx_data = ds
            if xx_data.ndim != 4:
                raise ValueError("Expect 4 dimensions on input: y,x,band,time")
            return None, None, xx_data

    kw.setdefault('nocheck', False)
    kw.setdefault('num_threads', 1)
    kw.setdefault('eps', 1e-6)

    ds, xx, xx_data = norm_input(ds, axis)
    is_dask = dask.is_dask_collection(xx_data)
    
    if where is not None:
        if is_dask:
            raise NotImplementedError("Dask version doesn't support output masking currently")

        if where.shape != xx_data.shape[:2]:
            raise ValueError("Shape for `where` parameter doesn't match")
        set_nan = ~where
    else:
        set_nan = None

    if is_dask:
#         print(gm)
#         print(xx_data)
        if xx_data.shape[-2:] != xx_data.chunksize[-2:]:
            xx_data = xx_data.rechunk(xx_data.chunksize[:2] + (-1, -1))
        print(xx_data.chunks[:-2] + (xx_data.chunks[-2][0]+3,))
        data = da.map_blocks(lambda x,g: tmad(x, g, **kw),
                             xx_data, gm,
                             name=randomize('tmads'),
                             dtype=xx_data.dtype, 
                             chunks=xx_data.chunks[:-2] + (xx_data.chunks[-2][0]+3,),
                             drop_axis=3)
    else:
        data = tmad(xx_data, gm, **kw)

    if set_nan is not None:
        data[set_nan, :] = np.nan

    if xx is None:
        return data
    print(data)
    
    #return data
    dims = xx.dims[:-1]
    #print(dims)
    cc = {k: xx.coords[k] for k in dims}
    #print(cc)
    cc[dims[-1]] = np.hstack([xx.coords[dims[-1]].values,['edev', 'sdev', 'bcdev']])
    #print(cc)
    xx_out = xr.DataArray(data, dims=dims, coords=cc)
    #print(xx_out)
    if ds is None:
        xx_out.attrs.update(xx.attrs)
        return xx_out

    ds_out = xx_out.to_dataset(dim='band')
    ds_out=ds_out.drop(['red', 'blue', 'green', 'nir', 'swir_1', 'swir_2', 'red_edge_1',
                'red_edge_2', 'red_edge_3'])

    return assign_crs(ds_out, crs=ds.geobox.crs)

def gm_two_seasons_annual_mads(ds):
    dc = datacube.Datacube(app='training')
    ds = ds / 10000
    ds1 = ds.sel(time=slice('2019-01', '2019-06'))
    ds2 = ds.sel(time=slice('2019-07', '2019-12')) 
    
    gm = dc.load(product='ga_s2_gm', like=ds, measurements = [
                'red', 'blue', 'green', 'nir', 'swir_1', 'swir_2', 'red_edge_1',
                'red_edge_2', 'red_edge_3'],
                 dask_chunks={'x':600,'y':700}
                ) 
    
    gm = reshape_for_geomedian(gm)
    gm = gm.squeeze().astype(np.float32).data
    # gm = da.from_array(gm, chunks=(750, 750,1))
    #should load the same spatial extent,crs etc.
    
    annual_tmad = xr_tmad(ds, gm) # write separate tmad function that calls on Kirill's geomedian    
    annual_tmad['sdev'] = -np.log(annual_tmad['sdev'])
    annual_tmad['bcdev'] = -np.log(annual_tmad['bcdev'])
    annual_tmad['edev'] = -np.log(annual_tmad['edev'])
    # this is an xarray holding only the annual tmads
    
    def fun(ds, era):
        #geomedian and tmads
        # Run geomedian and tmads for annual, then drop the gm and add the 6month gm
        gm = xr_geomedian(ds)
        gm = calculate_indices(gm,
                               index=['NDVI','LAI','MNDWI'],
                               drop=False,
                               normalise=False,
                               collection='s2')
        #rainfall climatology
        if era == '_S1':
            chirps = assign_crs(xr.open_rasterio('../data/CHIRPS/CHPclim_jan_jun_cumulative_rainfall.nc'),  crs='epsg:4326')
        if era == '_S2':
            chirps = assign_crs(xr.open_rasterio('../data/CHIRPS/CHPclim_jul_dec_cumulative_rainfall.nc'),  crs='epsg:4326')
        
        chirps = xr_reproject(chirps,ds.geobox,"bilinear")
        chirps = chirps.chunk({'x':1500,'y':1500})
        gm['rain'] = chirps
        
        for band in gm.data_vars:
            gm = gm.rename({band:band+era})
        
        return gm
        # returns the 6monthly geomedian
    
    epoch1 = fun(ds1, era='_S1')
    epoch2 = fun(ds2, era='_S2')
     
    #slope
    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    slope = rio_slurp_xarray(url_slope, gbox=ds.geobox)
    slope = slope.to_dataset(name='slope').chunk({'x':1500,'y':1500})
        
    result = xr.merge([epoch1,
                       epoch2,
                       annual_tmad,
                       slope],
                       compat='override')

    return result.squeeze()