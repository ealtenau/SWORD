import numpy as np
import netCDF4 as nc
import pandas as pd
import geopandas as gp
from shapely.geometry import Point
import time

start=time.time()
fn_dir = '/Users/ealtenau/Documents/SWORD_Dev/swot_data/Ganges/v2.0/pixel_cloud/'
file = 'SWOT_L2_HR_PIXC_008_495_193R_20231231T225617_20231231T225628_PIC0_01.nc'
fn = fn_dir+file

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
xtrack = np.array(pixc.groups['pixel_cloud'].variables['sig0_qual'][:])

# water = np.where((type == 4) & (class_qual == 0) & (sig0 > 30))[0]
water = np.where((type == 4) & (sig0 > 30))[0]
# water0 = np.where((type == 4) & (sig0 > 30))[0]
# water = water0[np.where(class_qual[water0] == 524288)[0]]
# water = np.append(water,water0[np.where(class_qual[water0] == 0)[0]])

ht_water = ht[water]
geoid_water = geoid[water]
lat_water = lat[water]
lon_water = lon[water]
type_water = type[water]
power_water = power[water]
sig0_water = sig0[water]
sig0_qual_water = sig0_qual[water]
class_qual_water = class_qual[water]
geo_qual_water = geo_qual[water]
frac_water = frac[water]

pixc_water = gp.GeoDataFrame([
    ht_water,
    geoid_water,
    lat_water,
    lon_water,
    type_water,
    power_water,
    sig0_water,
    sig0_qual_water,
    class_qual_water,
    geo_qual_water,
    frac_water,
]).T

#rename columns.
pixc_water.rename(
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

pixc_water = pixc_water.apply(pd.to_numeric, errors='ignore') # nodes.dtypes
geom = gp.GeoSeries(map(Point, zip(lon_water, lat_water)))
pixc_water['geometry'] = geom
pixc_water = gp.GeoDataFrame(pixc_water)
pixc_water.set_geometry(col='geometry')
pixc_water = pixc_water.set_crs(4326, allow_override=True)

outgpkg = fn_dir+file[:-3]+'_water_only.gpkg'
pixc_water.to_file(outgpkg, driver='GPKG', layer='pixc')

end = time.time()
print('Time to Finish Files: ' + str(np.round((end-start)/60, 2)) + ' min')