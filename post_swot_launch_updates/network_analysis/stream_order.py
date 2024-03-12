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
con_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/reach_geometry/'+region.lower()+'_sword_'+version+'_connectivity.nc'

data = nc.Dataset(path_nc,'r+')
con = nc.Dataset(con_dir)

paths = data.groups['centerlines'].variables['path_travel_frequency'][:]
order = data.groups['centerlines'].variables['path_order_by_length'][:]
main_side = data.groups['centerlines'].variables['main_side_chan'][:]
rchs = data.groups['centerlines'].variables['reach_id'][:]
x = data.groups['centerlines'].variables['x'][:]
y = data.groups['centerlines'].variables['y'][:]
ends = con.groups['centerlines'].variables['end_reach'][:]
con_rchs = con.groups['centerlines'].variables['reach_id'][:]

strm_order = np.zeros(len(paths))
normalize = np.where(paths > 0)[0] 
strm_order[normalize] = (np.round(np.log(paths[normalize])))+1
strm_order[np.where(paths == 0)] = -9999

### filter??
#split paths at junctions (create path segments)
#find all junction reaches
#order the junction reaches by path number
#loop through reaches and if the numbering doesn't work out update path segment


if 'stream_order' in data.groups['centerlines'].variables.keys():
    data.groups['centerlines'].variables['stream_order'][:] = strm_order
    data.close()
else:
    stream_order = data.groups['centerlines'].createVariable(
        'stream_order', 'i4', ('num_points',), fill_value=-9999.)
    data.groups['centerlines'].variables['stream_order'][:] = strm_order
    data.close()




'''

np.unique(strm_order)
plt.figure(1)
plt.scatter(x,y,c=strm_order,cmap = 'rainbow', s = 5)
plt.show()

plt.figure(2)
plt.scatter(x,y,c=paths,cmap = 'rainbow', s = 5)
plt.show()

one = np.where(order == 1)[0]
np.unique(paths[one])

'''