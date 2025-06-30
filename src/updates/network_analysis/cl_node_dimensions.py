"""
Filling in Centerline Node ID Neighbors (cl_node_dimensions.py).
===============================================================

This script fills in the node neighbor columns of the SWORD 
netCDF centerline dimension (i.e. sword.centerlines.node_id[1:4,:]).  
Currently the columns are filled with the closest neighboring 
based on a spatial query. 

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter region 
identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python cl_node_dimensions.py NA v17

"""

import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import netCDF4 as nc
import geopandas as gp
import pandas as pd
import argparse
import sys
import time
import matplotlib.pyplot as plt
from scipy import spatial as sp
from src.updates.sword import SWORD

start_all = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

#read data. 
sword = SWORD(main_dir, region, version)

print('Updating Centerline Node ID Neighbors')
cl_pts = np.vstack((sword.centerlines.x, sword.centerlines.y)).T
kdt = sp.cKDTree(cl_pts)
pt_dist, pt_ind = kdt.query(cl_pts, k = 4)

id_arr = sword.centerlines.node_id[0,pt_ind]
row_sum = np.sum(abs(np.diff(id_arr, axis=1)), axis = 1)
# (len(np.where(row_sum > 0)[0])/len(row_sum))*100
update = np.where(row_sum > 0)[0]
sword.centerlines.node_id[1:4,update] = id_arr[update,1:4].T

#updating the netcdf. 
print('Updating the NetCDF')
sword.save_nc()
end_all = time.time()
print('*** '+region + ' Done in: '+str(np.round((end_all-start_all)/60,2))+' mins ***')

