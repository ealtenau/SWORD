import numpy as np
import pandas as pd
import geopandas as gp
import netCDF4 as nc
from shapely.geometry import Point

fn = '/Users/ealteanau/Documents/SWORD_Dev/swot_data/Amazon/amazon_extdist_test/SWOT_L2_HR_PIXCVecRiver_529_022_159R_20230523T074555_20230523T074606_PGA1_01.nc'

pixc = nc.Dataset(fn)
keep = np.where(pixc.variables['height_vectorproc'][:] != -9999)[0]
df = gp.GeoDataFrame(np.array([pixc.variables['longitude_vectorproc'][keep], 
                            pixc.variables['latitude_vectorproc'][keep],
                            pixc.variables['height_vectorproc'][keep],
                            pixc.variables['reach_id'][keep],
                            pixc.variables['node_id'][keep],
                            pixc.variables['segmentation_label'][keep],
                            pixc.variables['distance_to_node'][keep],
                            pixc.variables['pixc_index'][keep]]).T)

df.rename(
    columns={
        0:'lon',
        1:'lat',
        2:'height',
        3:'reach_id',
        4:'node_id',
        5:'seg_label',
        6:'dist_to_node',
        7:'pixc_index'
        },inplace=True)

df = df.apply(pd.to_numeric, errors='ignore') # nodes.dtypes
geom = gp.GeoSeries(map(Point, zip(pixc.variables['longitude_vectorproc'][keep], pixc.variables['latitude_vectorproc'][keep])))
df['geometry'] = geom
df = gp.GeoDataFrame(df)
df.set_geometry(col='geometry')
df = df.set_crs(4326, allow_override=True)

#write file
df.to_file(fn[:-3]+'.gpkg', driver='GPKG', layer='points')

