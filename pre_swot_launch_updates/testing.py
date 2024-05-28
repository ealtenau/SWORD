from __future__ import division
import numpy as np
from scipy import spatial as sp
import netCDF4 as nc
from pyproj import Proj
import time
import geopy.distance
import pandas as pd
import argparse
import re
import os 
import geopandas as gp
from shapely.geometry import Point
import matplotlib.pyplot as plt


sword_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/sa_sword_v17.nc'
sword = nc.Dataset(sword_dir,'r+')

reaches = np.array(sword.groups['reaches'].variables['reach_id'][:])
node_rchs = np.array(sword.groups['nodes'].variables['reach_id'][:])

# rchs1 = np.array([22320700693,22320700681,22320700663,22320700651,22320700644,22320700631,22320700623,22320700611,22320700603,22320700591,22320700583,22320700571,22320700563,22320700551,22320700541,22320700534,22320700523,22320700511,22320700503])
# rchs2 = np.array([24434001793,24434001801,24434001811,24434001821,24434001831,24434001841,24434001853,24434001861,24434001871,24434001883])
# val1 = 608042.31378
# val2 = 671642.5202143731

# rch_ind1 = np.where(np.in1d(reaches, rchs1)==True)[0]
# rch_ind2 = np.where(np.in1d(reaches, rchs2)==True)[0]
# node_ind1 = np.where(np.in1d(node_rchs, rchs1)==True)[0]
# node_ind2 = np.where(np.in1d(node_rchs, rchs2)==True)[0]

# sword.groups['reaches'].variables['dist_out'][rch_ind1] = sword.groups['reaches'].variables['dist_out'][rch_ind1]+val1
# sword.groups['nodes'].variables['dist_out'][node_ind1] = sword.groups['nodes'].variables['dist_out'][node_ind1]+val1
# sword.groups['reaches'].variables['dist_out'][rch_ind2] = sword.groups['reaches'].variables['dist_out'][rch_ind2]+val2
# sword.groups['nodes'].variables['dist_out'][node_ind2] = sword.groups['nodes'].variables['dist_out'][node_ind2]+val2

csv = pd.read_csv('/Users/ealtenau/Documents/SWORD_Dev/update_requests/v17/EU/volga_delta_changes.csv')
rchs = np.array(csv['reach_id'])
main_side = np.array(csv['main_side'])
paths = np.array(csv['path_order'])

keep = np.where((paths > 1)&(main_side==0))[0]
rchs_change = rchs[keep]

update = np.where(np.in1d(reaches, rchs_change)==True)[0]
update2 = np.where(np.in1d(node_rchs, rchs_change)==True)[0]

sword.groups['reaches'].variables['main_side'][update] = 2
sword.groups['nodes'].variables['main_side'][update2] = 2
sword.groups['reaches'].variables['stream_order'][update] = -9999
sword.groups['nodes'].variables['stream_order'][update2] = -9999

sword.close()


#################################################################################################
#################################################################################################
#################################################################################################

sword_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/eu_sword_v17.nc'
sword = nc.Dataset(sword_dir,'r+')

rch_nan = np.where(np.array(sword.groups['reaches'].variables['main_side'][:]) == 1)[0]
node_nan = np.where(np.array(sword.groups['nodes'].variables['main_side'][:]) == 1)[0]
sword.groups['reaches'].variables['path_freq'][rch_nan] = -9999
sword.groups['nodes'].variables['path_freq'][node_nan] = -9999
sword.groups['reaches'].variables['path_order'][rch_nan] = -9999
sword.groups['nodes'].variables['path_order'][node_nan] = -9999
sword.close()

np.min(np.array(sword.groups['reaches'].variables['path_freq'][:]))
np.min(np.array(sword.groups['reaches'].variables['path_order'][:]))

#################################################################################################
#################################################################################################
#################################################################################################

sword_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/sa_sword_v17.nc'
sword = nc.Dataset(sword_dir,'r+')

rch = 61538000271
cl_ids = np.array(sword.groups['centerlines'].variables['cl_id'][:])
cl_rchs = np.array(sword.groups['centerlines'].variables['reach_id'][:])
x = np.array(sword.groups['centerlines'].variables['x'][:])
y = np.array(sword.groups['centerlines'].variables['y'][:])

r = np.where(cl_rchs[0,:] == rch)[0]
sort_ids = np.argsort(cl_ids[r])
plt.plot(x[r[sort_ids]], y[r[sort_ids]])
plt.show()


#################################################################################################
#################################################################################################
#################################################################################################

sword_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/sa_sword_v17.nc'
sword = nc.Dataset(sword_dir)

cl_nodes = np.array(sword.groups['centerlines'].variables['node_id'][0,:])
cl_id = np.array(sword.groups['centerlines'].variables['cl_id'][:])
cl_x = np.array(sword.groups['centerlines'].variables['x'][:])
cl_y = np.array(sword.groups['centerlines'].variables['y'][:])
nodes = np.array(sword.groups['nodes'].variables['node_id'][:])
nodes_cl_id = np.array(sword.groups['nodes'].variables['cl_ids'][:])
nx = np.array(sword.groups['nodes'].variables['x'][:])
ny = np.array(sword.groups['nodes'].variables['y'][:])

cln = np.unique(cl_nodes)
nds = np.unique(nodes)
missing = np.where(np.in1d(nds, cln) == False)[0]
missed_nodes = nds[missing]

for ind in list(range(len(missed_nodes))):
    id1 = nodes_cl_id[0,np.where(nds == missed_nodes[ind])[0]]
    id2 = nodes_cl_id[1,np.where(nds == missed_nodes[ind])[0]]
    indexes = list(range(id1[0],id2[0]+1))
    cl_inds = np.where(np.in1d(cl_id, indexes) == True)[0]
    cl_nodes[cl_inds] = missed_nodes[ind]


pt = np.where(nds == nodes[0])[0]
test = np.where(cl_nodes == nodes[0])[0]
plt.scatter(cl_x[test], cl_y[test])
plt.scatter(nx[0], ny[0],c = 'red')
plt.show()

#################################################################################################
#################################################################################################
#################################################################################################

dir1 = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/shp/NA/'
dir2 = '/Users/ealtenau/Desktop/NA_v17_04092024/'
files1 = os.listdir(dir1)
files2 = os.listdir(dir2)
files1 = np.array([f for f in files1 if '.shp' in f])
files2 = np.array([f for f in files2 if '.shp' in f])
files1 = np.sort(np.array([f for f in files1 if 'reaches' in f]))
files2 = np.sort(np.array([f for f in files2 if 'reaches' in f]))

for ind in list(range(len(files1))):
    f1 = gp.read_file(dir1+files1[ind])
    f2 = gp.read_file(dir2+files2[ind])
    print(files1[ind], np.unique(f1['reach_id']-f2['reach_id']))

#################################################################################################
#################################################################################################
#################################################################################################

sword_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/na_sword_v17.nc'
sword = nc.Dataset(sword_dir)

