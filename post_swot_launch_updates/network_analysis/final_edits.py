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
import matplotlib.pyplot as plt

region = 'SA'
version = 'v17'

nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'\
    +region.lower()+'_sword_'+version+'.nc'

sword = nc.Dataset(nc_fn, 'r+')

rch_edit_flag = sword.groups['reaches'].variables['edit_flag'][:]
node_edit_flag = sword.groups['nodes'].variables['edit_flag'][:]
ext_dist_coef = sword.groups['nodes'].variables['ext_dist_coef'][:]
cl_x = sword.groups['centerlines'].variables['x'][:]
cl_y = sword.groups['centerlines'].variables['y'][:]
cl_node_id = sword.groups['centerlines'].variables['node_id'][:]

print('Updating Edit Flag')
rch_erse = np.where(rch_edit_flag != '7')[0]
node_erse = np.where(node_edit_flag != '7')[0]
rch_edit_flag[rch_erse] = 'NaN'
node_edit_flag[node_erse] = 'NaN'
sword.groups['reaches'].variables['edit_flag'][:] = rch_edit_flag
sword.groups['nodes'].variables['edit_flag'][:] = node_edit_flag

print('Updating Centerline Node ID Neighbors')
cl_pts = np.vstack((cl_x, cl_y)).T
kdt = sp.cKDTree(cl_pts)
pt_dist, pt_ind = kdt.query(cl_pts, k = 4)

id_arr = cl_node_id[0,pt_ind]
row_sum = np.sum(abs(np.diff(id_arr, axis=1)), axis = 1)
# (len(np.where(row_sum > 0)[0])/len(row_sum))*100
update = np.where(row_sum > 0)[0]
cl_node_id[1:4,update] = id_arr[update,1:4].T
sword.groups['centerlines'].variables['node_id'][:] = cl_node_id
sword.close()

print('Done')