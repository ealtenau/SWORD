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

sword_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/eu_sword_v17.nc'
sword = nc.Dataset(sword_dir, 'r+')

rchs = sword.groups['centerlines'].variables['reach_id'][:]
cl_ids = sword.groups['centerlines'].variables['cl_id'][:]
x = sword.groups['centerlines'].variables['x'][:]
y = sword.groups['centerlines'].variables['y'][:]

rch = np.where(rchs[0,:] == 23250900135)[0]
sort = np.argsort(cl_ids[rch])

new_ids = np.copy(cl_ids[rch[sort]])
start = np.where(new_ids == 12248853)[0]
end = np.where(new_ids == 12248921)[0]
new_ids[start[0]:end[0]+1] = new_ids[start[0]:end[0]+1][::-1]

cl_ids[rch[sort]] = new_ids
sword.groups['centerlines'].variables['cl_id'][rch[sort]] = new_ids

sort_final = np.argsort(cl_ids[rch])
plt.plot(x[rch[sort_final]], y[rch[sort_final]], c='blue')
plt.show()

sword.close()


##################################################################################################

sword_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/shp/NA/na_sword_reaches_hb81_v17.shp'
sword = gp.read_file(sword_dir)

index = np.where((sword['type'] == 5) & (sword['main_side'] == 0))[0]
unq_paths = np.unique(sword['path_order'][index])
unq_paths = unq_paths[1::]
keep = np.where(np.in1d(sword['path_order'], unq_paths)==True)[0]
side = np.where(sword['main_side'] == 1)[0]

###works but need to be able to separate connected networks from one another...
# maybe remove any paths that are longer than a specific threshold???

plt.scatter(sword['x'], sword['y'], c = 'blue', s = 5)
plt.scatter(sword['x'][keep], sword['y'][keep], c = 'gold', s = 5)
plt.scatter(sword['x'][side], sword['y'][side], c = 'magenta', s = 5)
plt.show()