"""
Adding GLOW-S widths to SWORD (glow-s_to_sword_widths.py).
===============================================================

This script calculates and adds GLOW-S widths to SWORD. 

The script is run at a specified Pfafstetter level basin scale. 
Command line arguments required are the two-letter region 
identifier (i.e. NA), SWORD version (i.e. v17_glows), GLOW-S 
region identifier(i.e. '1'), and desired Pfafstetter basin level 
(i.e. 4).

Execution example (terminal):
    python path/to/glow-s_to_sword_widths.py AF v17_glows 1 4

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
parser.add_argument("sword_region", help="sword continental region", type = str)
parser.add_argument("sword_version", help="version", type = str)
parser.add_argument("glows_region", help="glows region", type = str)
parser.add_argument("basin_level", help="basin for subsetting", type = str)
args = parser.parse_args()

region = args.sword_region
version = args.sword_version
glows_region = args.glows_region
level = args.basin_level

# region = 'NA'
# version = 'v17_glows'
# glows_region = '7'
# level = '4'

sworddir = main_dir+'/data/outputs/Reaches_Nodes/'
glows_data_dir = main_dir+'/data/inputs/GLOW-S/'

swordpath = sworddir+version+'/'
sword_dir = swordpath+'netcdf/'+region.lower()+'_sword_'+version+'.nc'
wth_data_dir = glows_data_dir + 'GLOW-S_regions_merged/GLOW-S_region_'+glows_region+'_daywidth.parquet'

print('Reading in SWORD Data')
start = time.time()
sword = nc.Dataset(sword_dir,'r+')
nlon = np.array(sword.groups['nodes'].variables['x'][:])
nlat = np.array(sword.groups['nodes'].variables['y'][:])
nid = np.array(sword.groups['nodes'].variables['node_id'][:])
glows_ids = np.array(sword.groups['nodes'].variables['glows_river_id'][:]) #np.unique(glows_ids)
sword_comid = np.array(sword.groups['nodes'].variables['comid'][:]) 
#creating fill variables.
if 'glows_wth_1sig' in sword.groups['nodes'].variables.keys():
    node_glows_sig = np.array(sword.groups['nodes'].variables['glows_wth_1sig'][:])
    node_glows_wth_min = np.array(sword.groups['nodes'].variables['glows_wth_min'][:])
    node_glows_wth_med = np.array(sword.groups['nodes'].variables['glows_wth_med'][:])
    node_glow_wth_max = np.array(sword.groups['nodes'].variables['glows_wth_max'][:])
    node_glow_wth_nobs = np.array(sword.groups['nodes'].variables['glows_wth_nobs'][:])
else:
    node_glows_sig = np.repeat(-9999, len(nid))
    node_glows_wth_min = np.repeat(-9999, len(nid))
    node_glows_wth_med = np.repeat(-9999, len(nid))
    node_glow_wth_max = np.repeat(-9999, len(nid))
    node_glow_wth_nobs = np.repeat(-9999, len(nid))
end = time.time()
print(str(np.round((end-start),2))+' sec')

print('Reading in Width Parquet File')
start = time.time()
df = pd.read_parquet(wth_data_dir)
node = np.array(df['crossSxnID'])
wth = np.array(df['width'])
end = time.time()
print(str(np.round((end-start),2))+' sec')

#Here starts the inner basin loop
wth_comid = np.array([int(ind[1:9]) for ind in node])
sword_basin = np.array([int(str(ind)[0:int(level)]) for ind in nid])
unq_basins = np.unique(sword_basin)
for b in list(range(len(unq_basins))):
    print('STARTING BASIN ' + str(unq_basins[b]),'-', b, 'of', len(unq_basins)-1)
    basin_id = unq_basins[b]
    print('---> Dividing Data into Sub-Basins')
    start = time.time()
    #sword subsetting
    basin = np.where(sword_basin == int(basin_id))[0]
    null = np.where((glows_ids[basin] != 'NaN') & (node_glows_wth_med[basin] == -9999))[0]
    unq_ids = np.unique(glows_ids[basin[null]])
    sword_comid_sub = np.copy(sword_comid[basin])
    wth_min = np.copy(node_glows_wth_min[basin])
    wth_max = np.copy(node_glow_wth_max[basin])
    wth_median = np.copy(node_glows_wth_med[basin])
    wth_1sigma = np.copy(node_glows_sig[basin])
    wth_nobs = np.copy(node_glow_wth_nobs[basin])
    glows_ids_sub = np.copy(glows_ids[basin])
    #glows subsetting
    subset = np.where(np.in1d(wth_comid, sword_comid_sub) == True)[0]
    if len(subset) == 0:
        continue
    node_sub = node[subset]
    wth_sub = wth[subset]
    # end = time.time()
    # print(str(np.round((end-start),2))+' sec')
    
    print('---> Starting Node Calculations')
    # start = time.time()
    for n in list(range(len(unq_ids))):
        # print(n, len(unq_ids)-1)
        inds = np.where(node_sub == unq_ids[n])[0]
        if len(inds) == 0:
            continue
        fill = np.where(glows_ids_sub == unq_ids[n])[0]
        non_zero = np.where(wth_sub[inds] > 0)[0]
        wth_min[fill] = np.min(wth_sub[inds[non_zero]])
        wth_max[fill] = np.max(wth_sub[inds[non_zero]])
        wth_median[fill] = np.median(wth_sub[inds[non_zero]])
        wth_1sigma[fill] = np.std(wth_sub[inds[non_zero]])
        wth_nobs[fill] = len(non_zero)

    #filling in the continental data with basin values. 
    node_glows_sig[basin] = wth_1sigma
    node_glows_wth_min[basin] = wth_min
    node_glows_wth_med[basin] = wth_median
    node_glow_wth_max[basin] = wth_max
    node_glow_wth_nobs[basin] = wth_nobs
    end = time.time()
    print('---> DONE with Basin: '+ str(np.round((end-start),2))+' sec')
    #### end basin loop 

print('Updating NetCDF')
start = time.time()
if 'glows_wth_med' in sword.groups['nodes'].variables.keys():
    sword.groups['nodes'].variables['glows_wth_med'][:] = node_glows_wth_med
    sword.groups['nodes'].variables['glows_wth_min'][:] = node_glows_wth_min
    sword.groups['nodes'].variables['glows_wth_max'][:] = node_glow_wth_max
    sword.groups['nodes'].variables['glows_wth_1sig'][:] = node_glows_sig
    sword.groups['nodes'].variables['glows_wth_nobs'][:] = node_glow_wth_nobs
    sword.close()
else:
    #nodes
    sword.groups['nodes'].createVariable('glows_wth_med', 'f8', ('num_nodes',), fill_value=-9999.)
    sword.groups['nodes'].createVariable('glows_wth_min', 'f8', ('num_nodes',), fill_value=-9999.)
    sword.groups['nodes'].createVariable('glows_wth_max', 'f8', ('num_nodes',), fill_value=-9999.)
    sword.groups['nodes'].createVariable('glows_wth_1sig', 'f8', ('num_nodes',), fill_value=-9999.)
    sword.groups['nodes'].createVariable('glows_wth_nobs', 'i8', ('num_nodes',), fill_value=-9999.)
    sword.groups['nodes'].variables['glows_wth_med'][:] = node_glows_wth_med
    sword.groups['nodes'].variables['glows_wth_min'][:] = node_glows_wth_min
    sword.groups['nodes'].variables['glows_wth_max'][:] = node_glow_wth_max
    sword.groups['nodes'].variables['glows_wth_1sig'][:] = node_glows_sig
    sword.groups['nodes'].variables['glows_wth_nobs'][:] = node_glow_wth_nobs
    sword.close()
end = time.time()
print(str(np.round((end-start)/60,2))+' min')

print()
end_all = time.time()
print(region+' DONE in: ' + str(np.round((end_all-start_all)/3600,2))+' hrs')

### stats and plotting 
data = np.where(node_glows_wth_med>-9999)[0]
print(len(data)/len(nid)*100, '% Node Coverage') #89% node coverage in Ohio Basin. 

# plt.scatter(nlon, nlat, c='blue', s=5)
# plt.scatter(nlon[data], nlat[data], c='red', s=5)
# plt.show()

plt.scatter(nlon, nlat, c='black', s=5)
plt.scatter(nlon[data], nlat[data], c=np.log(node_glows_wth_med[data]), s=5, cmap='rainbow')
plt.show()

######### time comparisons: 
# start = time.time()
# filtered_df = df[df['crossSxnID'] == unq_ids[n]]
# end = time.time()
# print(str(np.round((end-start),2))+' sec') #5.5 sec

# import pyarrow.parquet as pq
# import pyarrow.compute as pc
# table = pq.read_table(wth_data_dir)
# start = time.time()
# filtered_table = table.filter(pc.equal(table['crossSxnID'], unq_ids[n]))
# end = time.time()
# print(str(np.round((end-start),2))+' sec')

# start = time.time()
# inds = np.where(node_sub == unq_ids[n])[0]
# end = time.time()
# print(str(np.round((end-start),2))+' sec')