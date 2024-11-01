import numpy as np
import netCDF4 as nc
import pandas as pd
import geopandas as gp
from shapely.geometry import Point
import time
import os

start=time.time()
fn_dir = '/Users/ealtenau/Documents/SWORD_Dev/swot_data/Africa/pixc/'
files = os.listdir(fn_dir)
files = np.array([f for f in files if '.nc' in f])

###loop
for f in list(range(len(files))):
    print(f, len(files)-1)
    fn = fn_dir+files[f]

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
    xtrack = np.array(pixc.groups['pixel_cloud'].variables['cross_track'][:])

    keep = np.where(abs(xtrack) > 10000)[0]

    pixc_all = gp.GeoDataFrame([
        ht[keep],
        lat[keep],
        lon[keep],
        type[keep],
        sig0[keep],
    ]).T

    #rename columns.
    pixc_all.rename(
        columns={
            0:"ht",
            1:"lat",
            2:"lon",
            3:"class",
            4:"sig0",
            },inplace=True)

    pixc_all = pixc_all.apply(pd.to_numeric, errors='ignore') # nodes.dtypes
    geom = gp.GeoSeries(map(Point, zip(lon[keep], lat[keep])))
    pixc_all['geometry'] = geom
    pixc_all = gp.GeoDataFrame(pixc_all)
    pixc_all.set_geometry(col='geometry')
    pixc_all = pixc_all.set_crs(4326, allow_override=True)

    outgpkg = fn_dir+'gpkg/'+files[f][:-3]+'.gpkg'
    pixc_all.to_file(outgpkg, driver='GPKG', layer='pixc')

end = time.time()
print('Time to Finish Files: ' + str(np.round((end-start)/60, 2)) + ' min')