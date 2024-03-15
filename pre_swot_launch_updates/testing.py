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


sword_dir1 = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17_pre_edits/netcdf_pre_path_updates/na_sword_v17a.nc'
sword_dir2 = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/na_sword_v17.nc'

old = nc.Dataset(sword_dir1)
new = nc.Dataset(sword_dir2)

old_cl_rchs = old.groups['centerlines'].variables['reach_id'][:]
old_cl_ids = old.groups['centerlines'].variables['cl_id'][:]
old_cl_x = old.groups['centerlines'].variables['x'][:]
old_cl_y = old.groups['centerlines'].variables['y'][:]

new_cl_rchs = new.groups['centerlines'].variables['reach_id'][:]
new_cl_ids = new.groups['centerlines'].variables['cl_id'][:]
new_cl_x = new.groups['centerlines'].variables['x'][:]
new_cl_y = new.groups['centerlines'].variables['y'][:]


rch1 = np.where(old_cl_rchs[0,:] == 81210300051)[0]
rch2 = np.where(new_cl_rchs[0,:] == 81210300051)[0]

sort1 = np.argsort(old_cl_ids[rch1])
sort2 = np.argsort(new_cl_ids[rch2])

plt.plot(old_cl_x[rch1[sort1]], old_cl_y[rch1[sort1]], c='blue')
plt.show()

plt.plot(new_cl_x[rch2[sort2]], new_cl_y[rch2[sort2]], c='magenta')
plt.plot(old_cl_x[rch1[sort1]], old_cl_y[rch1[sort1]], c='blue')
plt.show()

plt.scatter(old_cl_x[rch1[sort1]], old_cl_y[rch1[sort1]], c=old_cl_ids[rch1[sort1]])
plt.show()

plt.scatter(new_cl_x[rch1[sort1]], new_cl_y[rch1[sort1]], c=new_cl_ids[rch2[sort2]])
plt.show()







old_cl_ids[rch1[sort1]]

new_ids_test = old_cl_ids[rch1[sort1]][::-1]
plt.scatter(new_cl_x[rch1[sort1]], new_cl_y[rch1[sort1]], c=new_ids_test)
plt.show()

plt.plot(new_cl_x[rch1[sort1]], new_cl_y[rch1[sort1]], c='blue')
plt.show()
