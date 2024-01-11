import numpy as np
import netCDF4 as nc
import pandas as pd
import geopandas as gp
from shapely.geometry import Point
import time

start=time.time()
fn_dir = '/Users/ealteanau/Documents/SWORD_Dev/swot_data/temp_data/tiles/'
file = 'SWOT_L2_HR_PIXC_006_298_198L_20231113T043148_20231113T043159_PIB0_01.nc'
fn = fn_dir+file
tile = '298_198L'

pixc = nc.Dataset(fn)
ht = np.array(pixc.groups['pixel_cloud'].variables['height'][:])
geoid = np.array(pixc.groups['pixel_cloud'].variables['geoid'][:])
lat = np.array(pixc.groups['pixel_cloud'].variables['latitude'][:])
lon = np.array(pixc.groups['pixel_cloud'].variables['longitude'][:])
type = np.array(pixc.groups['pixel_cloud'].variables['classification'][:])
power = np.array(pixc.groups['pixel_cloud'].variables['coherent_power'][:])
sig0 = np.array(pixc.groups['pixel_cloud'].variables['sig0'][:])
sig0_qual = np.array(pixc.groups['pixel_cloud'].variables['sig0_qual'][:])
class_qual = np.array(pixc.groups['pixel_cloud'].variables['classification_qual'][:])
geo_qual = np.array(pixc.groups['pixel_cloud'].variables['geolocation_qual'][:])
frac = np.array(pixc.groups['pixel_cloud'].variables['water_frac'][:])

pixc_all = gp.GeoDataFrame([
    ht,
    geoid,
    lat,
    lon,
    type,
    power,
    sig0,
    sig0_qual,
    class_qual,
    geo_qual,
    frac,
]).T

#rename columns.
pixc_all.rename(
    columns={
        0:"ht",
        1:"geoid",
        2:"lat",
        3:"lon",
        4:"class",
        5:"power",
        6:"sig0",
        7:"sig0_qual",
        8:"class_qual",
        9:"geo_qual",
        10:"frac",
        },inplace=True)

pixc_all = pixc_all.apply(pd.to_numeric, errors='ignore') # nodes.dtypes
geom = gp.GeoSeries(map(Point, zip(lon, lat)))
pixc_all['geometry'] = geom
pixc_all = gp.GeoDataFrame(pixc_all)
pixc_all.set_geometry(col='geometry')
pixc_all = pixc_all.set_crs(4326, allow_override=True)

outgpkg = fn_dir+tile+'.gpkg' #fn_dir + 'gis_files/pixel_cloud_'+tile+'.gpkg'
pixc_all.to_file(outgpkg, driver='GPKG', layer='pixc')

end = time.time()
print('Time to Finish Files: ' + str(np.round((end-start)/60, 2)) + ' min')