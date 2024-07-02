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

###############################################################################

def find_path_segs(order, paths):
    unq_paths = np.unique(order)
    unq_paths = unq_paths[unq_paths>0]
    cnt = 1
    path_segs = np.zeros(len(order))
    for p in list(range(len(unq_paths))):
        pth = np.where(order == unq_paths[p])[0]
        sections = np.unique(paths[pth])
        for s in list(range(len(sections))):
            sec = np.where(paths[pth] == sections[s])[0]
            path_segs[pth[sec]] = cnt
            cnt = cnt+1
    return path_segs

###############################################################################
###############################################################################
###############################################################################

region = 'OC'
version = 'v17'
basin = 'hb52'
path_nc = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/network_building/pathway_netcdfs/'+region+'/'+basin+'_path_vars.nc'
con_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/reach_geometry/'+region.lower()+'_sword_'+version+'_connectivity.nc'

# Read in data. 
data = nc.Dataset(path_nc,'r+')
con = nc.Dataset(con_dir)

paths = data.groups['centerlines'].variables['path_travel_frequency'][:]
order = data.groups['centerlines'].variables['path_order_by_length'][:]
main_side = data.groups['centerlines'].variables['main_side_chan'][:]
rchs = data.groups['centerlines'].variables['reach_id'][:]
x = data.groups['centerlines'].variables['x'][:]
y = data.groups['centerlines'].variables['y'][:]
cl_ids = data.groups['centerlines'].variables['cl_id'][:]
dist = data.groups['centerlines'].variables['dist_out_all'][:]
ends = con.groups['centerlines'].variables['end_reach'][:]
con_rchs = con.groups['centerlines'].variables['reach_id'][:]
con_cl_ids = con.groups['centerlines'].variables['cl_id'][:]

# Calculate starting stream order. 
strm_order = np.zeros(len(paths))
normalize = np.where(paths > 0)[0] 
strm_order[normalize] = (np.round(np.log(paths[normalize])))+1
strm_order[np.where(paths == 0)] = -9999

# Find path segments. 
path_segs = find_path_segs(order,paths)

if 'stream_order' in data.groups['centerlines'].variables.keys():
    data.groups['centerlines'].variables['stream_order'][:] = strm_order
    data.groups['centerlines'].variables['path_segments'][:] = path_segs
    data.close()
else:
    stream_order = data.groups['centerlines'].createVariable(
        'stream_order', 'i4', ('num_points',), fill_value=-9999.)
    data.groups['centerlines'].variables['stream_order'][:] = strm_order
    path_segments = data.groups['centerlines'].createVariable(
        'path_segments', 'i8', ('num_points',), fill_value=-9999.)
    data.groups['centerlines'].variables['path_segments'][:] = path_segs
    data.close()


print('Done')
# good = np.where(strm_order>0)[0]
# plt.figure(1)
# plt.scatter(x[good],y[good],c=strm_order[good],cmap = 'rainbow', s = 5)
# plt.show()

'''

good = np.where(strm_order>0)[0]
plt.figure(1)
plt.scatter(x[good],y[good],c=strm_order[good],cmap = 'rainbow', s = 5)
plt.show()





plt.figure(2)
plt.scatter(x,y,c='blue', s = 5)
plt.scatter(x[pth],y[pth],c='gold', s = 5)
plt.scatter(x[pth[junc[1]:junc[2]+1]],y[pth[junc[1]:junc[2]+1]], c='magenta', s=5)
plt.scatter(x[pth[junc[3]:junc[4]+1]],y[pth[junc[3]:junc[4]+1]], c='green', s=5)
# plt.scatter(x[pth[junc]],y[pth[junc]],c='black', s = 5)
plt.show()


plt.figure(3)
junc=np.where(ends == 3)[0]
plt.scatter(x,y,c=path_segs, cmap = 'rainbow', s = 5)
plt.scatter(x[junc],y[junc],c='black', s = 5)
plt.show()

one = np.where(order == 1)[0]
np.unique(paths[one])


plt.figure(4)
junc=np.where(ends == 3)[0]
plt.scatter(x,y,c=path_segs, cmap = 'rainbow', s = 5)
plt.scatter(x[seg],y[seg],c='black', s = 5)
plt.show()




df = pd.DataFrame(np.array([x,y,path_segs,new_strm_order,rchs[0,:]]).T)
df.to_csv('/Users/ealtenau/Desktop/path_seg_test.csv')
'''