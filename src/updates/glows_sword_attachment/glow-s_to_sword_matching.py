"""
Matching SWORD IDs to GLOW-S IDs (glow-s_to_sword_matching.py).
===============================================================

This script matches and adds GLOW-S COMID ID attribute to SWORD. 

The is run at a regional/continental scale. Command line 
arguments required are the two-letter region identifier (i.e. NA), 
SWORD version (i.e. v17_glows), and the GLOW-S region identifier
(i.e. '1').

Execution example (terminal):
    python path/to/glow-s_to_sword_matching.py AF v17_glows 1

"""
import os
main_dir = os.getcwd()
import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt
import argparse 

start_all = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("continent", help="sword continental region", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("region", help="glows region", type = str)
args = parser.parse_args()

region = args.continent
version = args.version
glows_region = args.region

# region = 'AF'
# version = 'v17_glows'
# glows_region = '1'

sworddir = main_dir+'/data/outputs/Reaches_Nodes/'
glows_data_dir = main_dir+'/data/inputs/GLOW-S/'

swordpath = sworddir+version+'/'
sword_dir = swordpath+'netcdf/'+region.lower()+'_sword_'+version+'.nc'
wth_data_dir = glows_data_dir + 'GLOW-S_regions_merged/GLOW-S_region_'+glows_region+'_daywidth.parquet'
wth_node_dir = glows_data_dir + 'GLOW-S_crosssection_points/GLOW-S_region_'+glows_region+'.shp'

print('Reading in SWORD Data')
start = time.time()
sword = nc.Dataset(sword_dir,'r+')
nlon = np.array(sword.groups['nodes'].variables['x'][:])
nlat = np.array(sword.groups['nodes'].variables['y'][:])
nid = np.array(sword.groups['nodes'].variables['node_id'][:])
rid = np.array(sword.groups['reaches'].variables['reach_id'][:])
rlon = np.array(sword.groups['reaches'].variables['x'][:])
rlat = np.array(sword.groups['reaches'].variables['y'][:])
nrid = np.array(sword.groups['nodes'].variables['reach_id'][:])
nedit = np.array(sword.groups['nodes'].variables['edit_flag'][:])
nedit[:] = 'NaN'
if 'glows_river_id' in sword.groups['nodes'].variables.keys():
    node_glows_id = np.array(sword.groups['nodes'].variables['glows_river_id'][:])
end = time.time()
print(str(np.round((end-start),2))+' sec')

print('Reading in Width Cross Section Shp File')
start = time.time()
shp = gp.read_file(wth_node_dir)
wlon = np.array(shp['lon'])
wlat = np.array(shp['lat'])
wid = np.array(shp['riverID'])
end = time.time()
print(str(np.round((end-start)/60,2))+' min')

if 'glows_river_id' in sword.groups['nodes'].variables.keys():
    glows_ids = np.copy(node_glows_id)
else:
    glows_ids = np.copy(nedit)

print('Spatial Join')
start = time.time()
wth_pts = np.vstack((wlon, wlat)).T
sword_pts = np.vstack((nlon, nlat)).T
kdt = sp.cKDTree(wth_pts)
pt_dist, pt_ind = kdt.query(sword_pts, k = 1)
keep = np.where(pt_dist <= 0.0025)[0]
glows_ids[keep] = wid[pt_ind[keep]]
end = time.time()
print(str(np.round((end-start),2))+' sec')

print('Updating NetCDF')
start = time.time()
if 'glows_river_id' in sword.groups['nodes'].variables.keys():
    print('variable exists updating in netcdf')
    #nodes
    #glows river id population
    sword_glows_id = np.copy(node_glows_id)
    add = np.where(sword_glows_id != 'NaN')[0]
    glows_ids[add] = sword_glows_id[add]
    sword.groups['nodes'].variables['glows_river_id'][:] = glows_ids
    # sword.close()
else:
    #nodes
    #glows river id population
    print('creating and updating new variable in netcdf')
    sword.groups['nodes'].createVariable('glows_river_id', 'S50', ('num_nodes',))
    sword.groups['nodes'].variables['glows_river_id']._Encoding = 'ascii'
    sword.groups['nodes'].variables['glows_river_id'][:] = glows_ids
    # sword.close()
end = time.time()
print(str(np.round((end-start)/60,2))+' min')

print('DONE in: ')
end_all = time.time()
print(str(np.round((end_all-start_all)/60,2))+' min')
print(np.unique(sword.groups['nodes'].variables['glows_river_id'][:]))
sword.close()

### stats and plotting 
ndata = np.where(glows_ids != 'NaN')[0]
plt.scatter(nlon, nlat, c='black', s=5)
plt.scatter(nlon[ndata], nlat[ndata], c='red', s=5)
plt.show()