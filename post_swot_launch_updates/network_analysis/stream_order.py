import numpy as np
import netCDF4 as nc
import geopandas as gp
import geopy.distance
from shapely.geometry import LineString, Point
from geopandas import GeoSeries
import pandas as pd
import time
import argparse
import os
from scipy import spatial as sp
import matplotlib.pyplot as plt
from scipy import stats
from scipy import interpolate

region = 'NA'
version = 'v17a'
basin = 'hb74'
path_nc = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/pathways/'+region+'/'+basin+'_path_vars.nc'

data = nc.Dataset(path_nc,'r+')

paths = data.groups['centerlines'].variables['path_travel_frequency'][:]
order = data.groups['centerlines'].variables['path_order_by_length'][:]
main_side = data.groups['centerlines'].variables['main_side_chan'][:]
rchs = data.groups['centerlines'].variables['reach_id'][:]
x = data.groups['centerlines'].variables['x'][:]
y = data.groups['centerlines'].variables['y'][:]

unq_paths = np.unique(order)
norm_freq = np.zeros(len(paths))
for ord in list(range(len(unq_paths))):
    pts = np.where(order == unq_paths[ord])[0]
    if min(paths[pts]) > 1:
        norm_freq[pts] = paths[pts]/min(paths[pts])
    else:
        norm_freq[pts] = paths[pts]

strm_order = np.zeros(len(norm_freq))
normalize = np.where(norm_freq > 0)[0] 
strm_order[normalize] = (np.round(np.log(norm_freq[normalize])))+1
strm_order[np.where(norm_freq == 0)] = 1

### filter??








# if 'stream_order' in data.groups['centerlines'].variables.keys():
#     data.groups['centerlines'].variables['stream_order'][:] = strm_order
#     data.close()
# else:
#     stream_order = data.groups['centerlines'].createVariable(
#         'stream_order', 'i4', ('num_points',), fill_value=-9999.)
#     data.groups['centerlines'].variables['stream_order'][:] = strm_order
#     data.close()


np.unique(strm_order)
plt.figure(1)
plt.scatter(x,y,c=strm_order,cmap = 'rainbow', s = 5)
plt.show()

plt.figure(2)
plt.scatter(x,y,c=norm_freq,cmap = 'rainbow', s = 5)
plt.show()

one = np.where(order == 1)[0]
np.unique(norm_freq[one])