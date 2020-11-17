
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

sys.path.append('../../Scripts')
from deafrica_bandindices import calculate_indices
from deafrica_temporal_statistics import xr_phenology, temporal_statistics
from deafrica_classificationtools import HiddenPrints
from deafrica_datahandling import load_ard

warnings.filterwarnings("ignore")

def hdstats_two_seasons(ds):
    dc = datacube.Datacube(app='training')
    ds = ds / 10000
    ds1 = ds.sel(time=slice('2019-01', '2019-06'))
    ds2 = ds.sel(time=slice('2019-07', '2019-12'))
    
    def fun(ds, era):
        
        #temporal stats
        ndvi = calculate_indices(ds,
                             index=['NDVI'],
                             drop=True,
                             collection='s2')
    
        ts = temporal_statistics(ndvi.NDVI,
                           stats=['f_mean', 'abs_change','discordance',
                                  'complexity','central_diff'])
        
        #geomedian and tmads
        gm_mads = xr_geomedian_tmad(ds)
        gm_mads = calculate_indices(gm_mads,
                               index=['NDVI', 'LAI'],
                               drop=False,
                               normalise=False,
                               collection='s2')
        gm_mads['edev'] = -np.log(gm_mads['edev'])
        gm_mads['sdev'] = -np.log(gm_mads['sdev'])
        gm_mads['bcdev'] = -np.log(gm_mads['bcdev'])
        
        #merge
        res = xr.merge([gm_mads,ts],compat='override')
        
        for band in res.data_vars:
            res = res.rename({band:band+era})
        
        return res
    
    epoch1 = fun(ds1, era='_S1')
    epoch2 = fun(ds2, era='_S2')
    
    #slope
    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    slope = rio_slurp_xarray(url_slope, gbox=ds.geobox)
    slope = slope.to_dataset(name='slope')
    
    result = xr.merge([epoch1,epoch2,slope],compat='override')
    result = assign_crs(result, crs=ds.geobox.crs)#.chunk({'x':2000,'y':2000})
    
    return result.squeeze()


def gm_mads_two_seasons(ds):
    dc = datacube.Datacube(app='training')
    ds = ds / 10000
    ds1 = ds.sel(time=slice('2019-01', '2019-06'))
    ds2 = ds.sel(time=slice('2019-07', '2019-12')) 
    
    def fun(ds, era):
        #geomedian and tmads
        gm_mads = xr_geomedian_tmad(ds)
        gm_mads = calculate_indices(gm_mads,
                               index=['NDVI','LAI','MNDWI'],
                               drop=False,
                               normalise=False,
                               collection='s2')
        
        gm_mads['sdev'] = -np.log(gm_mads['sdev'])
        gm_mads['bcdev'] = -np.log(gm_mads['bcdev'])
        gm_mads['edev'] = -np.log(gm_mads['edev'])
        
        #rainfall climatology
        if era == '_S1':
            chirps = assign_crs(xr.open_rasterio('../data/CHIRPS/CHPclim_jan_jun_cumulative_rainfall.nc'),  crs='epsg:4326')
        if era == '_S2':
            chirps = assign_crs(xr.open_rasterio('../data/CHIRPS/CHPclim_jul_dec_cumulative_rainfall.nc'),  crs='epsg:4326')
        
        chirps = xr_reproject(chirps,ds.geobox,"bilinear")
        chirps = chirps.chunk({'x':2000,'y':2000})
        gm_mads['rain'] = chirps
        
        for band in gm_mads.data_vars:
            gm_mads = gm_mads.rename({band:band+era})
        
        return gm_mads
    
    epoch1 = fun(ds1, era='_S1')
    epoch2 = fun(ds2, era='_S2')
    
    #slope
    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    slope = rio_slurp_xarray(url_slope, gbox=ds.geobox)
    slope = slope.to_dataset(name='slope').chunk({'x':2000,'y':2000})
    
    result = xr.merge([epoch1,
                       epoch2,
                       slope],compat='override')

    return result.squeeze()

def gm_mads_three_seasons(ds):  
# Note that 01_Extract_training_data is not Dask friendly
# Make sure to comment out all chunk notation
# Comment back in for 04_Predict
# First drafted 11 November 2020

    dc = datacube.Datacube(app='training')
    ds = ds / 10000
    ds1 = ds.sel(time=slice('2019-01', '2019-04'))
    ds2 = ds.sel(time=slice('2019-05', '2019-08')) 
    ds3 = ds.sel(time=slice('2019-09', '2019-12')) 
    
    def fun(ds, era):
        #geomedian and tmads
        gm_mads = xr_geomedian_tmad(ds)
        gm_mads = calculate_indices(gm_mads,
                               index=['NDVI','LAI','MNDWI'],
                               drop=False,
                               normalise=False,
                               collection='s2')
        
        gm_mads['sdev'] = -np.log(gm_mads['sdev'])
        gm_mads['bcdev'] = -np.log(gm_mads['bcdev'])
        gm_mads['edev'] = -np.log(gm_mads['edev'])
        
        #rainfall climatology
        if era == '_S1':
            chirps = assign_crs(xr.open_rasterio('../data/CHIRPS/CHPclim_jan_apr_cumulative_rainfall.nc'),  crs='epsg:4326')
        if era == '_S2':
            chirps = assign_crs(xr.open_rasterio('../data/CHIRPS/CHPclim_may_aug_cumulative_rainfall.nc'),  crs='epsg:4326')
        if era == '_S3':
            chirps = assign_crs(xr.open_rasterio('../data/CHIRPS/CHPclim_sep_dec_cumulative_rainfall.nc'),  crs='epsg:4326')
        
        chirps = xr_reproject(chirps,ds.geobox,"bilinear")
        chirps = chirps.chunk({'x':2000,'y':2000})
        gm_mads['rain'] = chirps
        
        for band in gm_mads.data_vars:
            gm_mads = gm_mads.rename({band:band+era})
        
        return gm_mads
    
    epoch1 = fun(ds1, era='_S1')
    epoch2 = fun(ds2, era='_S2')
    epoch3 = fun(ds3, era='_S3')
    
    #slope
    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    slope = rio_slurp_xarray(url_slope, gbox=ds.geobox)
    slope = slope.to_dataset(name='slope').chunk({'x':2000,'y':2000})
    
    result = xr.merge([epoch1,
                       epoch2,
                       epoch3,
                       slope],compat='override')

    return result.squeeze()

def gm_three_seasons(ds):
# With rainfall but no tmads
# Four-monthly intervals
# First drafted 12 November 2020

    dc = datacube.Datacube(app='training')
    ds = ds / 10000
    ds1 = ds.sel(time=slice('2019-01', '2019-04'))
    ds2 = ds.sel(time=slice('2019-05', '2019-08')) 
    ds3 = ds.sel(time=slice('2019-09', '2019-12')) 
    
    def fun(ds, era):
        #geomedian
        gm = xr_geomedian(ds)
        gm = calculate_indices(gm,
                               index=['NDVI', 'LAI'],
                               drop=False,
                               normalise=False,
                               collection='s2')
        
        if era == '_S1':
            chirps = assign_crs(xr.open_rasterio('../data/CHIRPS/CHPclim_jan_apr_cumulative_rainfall.nc'),  crs='epsg:4326')
        if era == '_S2':
            chirps = assign_crs(xr.open_rasterio('../data/CHIRPS/CHPclim_may_aug_cumulative_rainfall.nc'),  crs='epsg:4326')
        if era == '_S3':
            chirps = assign_crs(xr.open_rasterio('../data/CHIRPS/CHPclim_sep_dec_cumulative_rainfall.nc'),  crs='epsg:4326')
        
        chirps = xr_reproject(chirps,ds.geobox,"bilinear")
        chirps = chirps.chunk({'x':2000,'y':2000})
        gm['rain'] = chirps
         
        for band in gm.data_vars:
            gm = gm.rename({band:band+era})
            
        return gm
    
    epoch1 = fun(ds1, era='_S1')
    epoch2 = fun(ds2, era='_S2')
    epoch3 = fun(ds3, era='_S3')
    
    #slope
    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    slope = rio_slurp_xarray(url_slope, gbox=ds.geobox)
    slope = slope.to_dataset(name='slope').chunk({'x':2000,'y':2000})
    
    result = xr.merge([epoch1,
                       epoch2,
                       epoch3,
                       slope],compat='override')
    result = assign_crs(result, crs=ds.geobox.crs)
    
    return result.squeeze()

def simple_two_seasons(ds):
    dc = datacube.Datacube(app='training')
    ds = ds / 10000
    ds1 = ds.sel(time=slice('2019-01', '2019-06'))
    ds2 = ds.sel(time=slice('2019-07', '2019-12'))
    
    def fun(ds, era):
        #geomedian
        gm = xr_geomedian(ds)
        gm = calculate_indices(gm,
                               index=['NDVI', 'LAI'],
                               drop=False,
                               normalise=False,
                               collection='s2')
        
        for band in gm.data_vars:
            gm = gm.rename({band:band+era})
        
        return gm
    
    epoch1 = fun(ds1, era='_S1')
    epoch2 = fun(ds2, era='_S2')
    
    #slope
    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    slope = rio_slurp_xarray(url_slope, gbox=ds.geobox)
    slope = slope.to_dataset(name='slope')
    
    result = xr.merge([epoch1,epoch2,slope],compat='override')
    result = assign_crs(result, crs=ds.geobox.crs)
    
    return result.squeeze()


def xr_geomedian_tmad(ds, axis='time', where=None, **kw):
    """
    :param ds: xr.Dataset|xr.DataArray|numpy array
    Other parameters:
    **kwargs -- passed on to pcm.gnmpcm
       maxiters   : int         1000
       eps        : float       0.0001
       num_threads: int| None   None
    """

    import hdstats
    def gm_tmad(arr, **kw):
        """
        arr: a high dimensional numpy array where the last dimension will be reduced. 
    
        returns: a numpy array with one less dimension than input.
        """
        gm = hdstats.nangeomedian_pcm(arr, **kw)
        nt = kw.pop('num_threads', None)
        emad = hdstats.emad_pcm(arr, gm, num_threads=nt)[:,:, np.newaxis]
        smad = hdstats.smad_pcm(arr, gm, num_threads=nt)[:,:, np.newaxis]
        bcmad = hdstats.bcmad_pcm(arr, gm, num_threads=nt)[:,:, np.newaxis]
        return np.concatenate([gm, emad, smad, bcmad], axis=-1)


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
        if xx_data.shape[-2:] != xx_data.chunksize[-2:]:
            xx_data = xx_data.rechunk(xx_data.chunksize[:2] + (-1, -1))

        data = da.map_blocks(lambda x: gm_tmad(x, **kw),
                             xx_data,
                             name=randomize('geomedian'),
                             dtype=xx_data.dtype, 
                             chunks=xx_data.chunks[:-2] + (xx_data.chunks[-2][0]+3,),
                             drop_axis=3)
    else:
        data = gm_tmad(xx_data, **kw)

    if set_nan is not None:
        data[set_nan, :] = np.nan

    if xx is None:
        return data

    dims = xx.dims[:-1]
    cc = {k: xx.coords[k] for k in dims}
    cc[dims[-1]] = np.hstack([xx.coords[dims[-1]].values,['edev', 'sdev', 'bcdev']])
    xx_out = xr.DataArray(data, dims=dims, coords=cc)

    if ds is None:
        xx_out.attrs.update(xx.attrs)
        return xx_out

    ds_out = xx_out.to_dataset(dim='band')
    for b in ds.data_vars.keys():
        src, dst = ds[b], ds_out[b]
        dst.attrs.update(src.attrs)

    return assign_crs(ds_out, crs=ds.geobox.crs)


# import richdem as rd
# def xr_terrain(da, attribute=None):
#     """
#     Using the richdem package, calculates terrain attributes
#     on a DEM stored in memory as an xarray.DataArray 
    
#     Params
#     -------
#     da : xr.DataArray
#     attribute : str
#         One of the terrain attributes that richdem.TerrainAttribute()
#         has implemented. e.g. 'slope_riserun', 'slope_percentage', 'aspect'.
#         See all option here:  
#         https://richdem.readthedocs.io/en/latest/python_api.html#richdem.TerrainAttribute
        
#     """
#     #remove time if its there
#     da = da.squeeze()
#     #convert to richdem array
#     rda = rd.rdarray(da.data, no_data=da.attrs['nodata'])
#     #add projection and geotransform
#     rda.projection=pyproj.crs.CRS(da.attrs['crs']).to_wkt()
#     rda.geotransform = da.geobox.affine.to_gdal()
#     #calulate attribute
#     attrs = rd.TerrainAttribute(rda, attrib=attribute)

#     #return as xarray DataArray
#     return xr.DataArray(attrs,
#                         attrs=da.attrs,
#                         coords={'x':da.x, 'y':da.y},
#                         dims=['y', 'x'])

   
#     slope = dc.load(product='srtm', like=ds.geobox).squeeze()
#     slope = slope.elevation
#     slope = xr_terrain(slope, 'slope_riserun')
#     slope = slope.to_dataset(name='slope')#.chunk({'x':1500,'y':1500})

#     #rainfall climatology
#     chirps = assign_crs(xr.open_rasterio('../data/CHIRPS/CHPclim_annualSum.nc'),  crs='epsg:4326')
#     chirps = xr_reproject(chirps,ds.geobox,"bilinear")
#     chirps = chirps.to_dataset(name='rainfall').chunk({'x':2000,'y':2000})