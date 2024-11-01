import numpy as np
import pandas as pd
import geopandas as gp
import netCDF4 as nc
from shapely.geometry import Point

fn = '/Users/ealtenau/Documents/SWORD_Dev/swot_data/shifting_tests/412_079L/'\
    'SWOT_L2_HR_PIXCVec_011_412_079L_20240229T134222_20240229T134234_PIC0_01.nc'

pixc = nc.Dataset(fn)

rch_id = np.array(pixc.variables['reach_id'][:])
nd_id = np.array(pixc.variables['node_id'][:])

reach_id = np.zeros(len(rch_id))
for r in list(range(len(rch_id))):
    id = rch_id[r,0]+rch_id[r,1]+rch_id[r,2]+rch_id[r,3]+rch_id[r,4]+rch_id[r,5]+\
        rch_id[r,6]+rch_id[r,7]+rch_id[r,8]+rch_id[r,9]+rch_id[r,10]
    if len(id) > 0:
        reach_id[r] = int(id)
    else:
        reach_id[r] = -9999

node_id = np.zeros(len(nd_id))
for n in list(range(len(nd_id))):
    id = nd_id[n,0]+nd_id[n,1]+nd_id[n,2]+nd_id[n,3]+nd_id[n,4]+nd_id[n,5]+\
        nd_id[n,6]+nd_id[n,7]+nd_id[n,8]+nd_id[n,9]+nd_id[n,10]+nd_id[n,11]+nd_id[n,12]+nd_id[n,13]
    if len(id) > 0:
        node_id[r] = int(id)
    else:
        node_id[r] = -9999

df = gp.GeoDataFrame(np.array([pixc.variables['longitude_vectorproc'][:], 
                            pixc.variables['latitude_vectorproc'][:],
                            pixc.variables['height_vectorproc'][:],
                            reach_id,
                            node_id,
                            # pixc.variables['segmentation_label'][keep],
                            # pixc.variables['distance_to_node'][keep],
                            # pixc.variables['pixc_index'][keep]
                            ]).T)

df.rename(
    columns={
        0:'lon',
        1:'lat',
        2:'height',
        3:'reach_id',
        4:'node_id',
        },inplace=True)

df = df.apply(pd.to_numeric, errors='ignore') # nodes.dtypes
geom = gp.GeoSeries(map(Point, zip(pixc.variables['longitude_vectorproc'][:], pixc.variables['latitude_vectorproc'][:])))
df['geometry'] = geom
df = gp.GeoDataFrame(df)
df.set_geometry(col='geometry')
df = df.set_crs(4326, allow_override=True)

#write file
df.to_file(fn[:-3]+'.gpkg', driver='GPKG', layer='points')





