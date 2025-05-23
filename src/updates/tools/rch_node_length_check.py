import os
main_dir = os.getcwd()
import numpy as np
import netCDF4 as nc
import geopandas as gp
from geopy import distance
import pandas as pd
import random

region = 'OC'
version = 'v18'

nc_fn = main_dir+'/data/outputs/Reaches_Nodes/'+version+'/netcdf/'\
    +region.lower()+'_sword_'+version+'.nc'

sword = nc.Dataset(nc_fn)
reaches = np.array(sword.groups['reaches'].variables['reach_id'][:])
rch_len = np.array(sword.groups['reaches'].variables['reach_length'][:])
rch_dist = np.array(sword.groups['reaches'].variables['dist_out'][:])
nodes = np.array(sword.groups['nodes'].variables['node_id'][:])
node_len = np.array(sword.groups['nodes'].variables['node_length'][:])
node_dist = np.array(sword.groups['nodes'].variables['dist_out'][:])
node_rch = np.array(sword.groups['nodes'].variables['reach_id'][:])
sword.close()

nlen_diff = np.zeros(len(reaches))
do_diff = np.zeros(len(reaches))
for ind in list(range(len(reaches))):
    test = np.where(node_rch == reaches[ind])[0]
    nlen_diff[ind] = np.abs(np.round(sum(node_len[test])-rch_len[ind]))
    do_diff[ind] = np.abs(np.round(max(node_dist[test])-rch_dist[ind]))

len_diff_perc = len(np.where(nlen_diff != 0)[0])/len(reaches)*100
do_diff_perc = len(np.where(do_diff != 0)[0])/len(reaches)*100

print('Percent Length Differences:', np.round(len_diff_perc, 2), ", Max Diff:", np.max(nlen_diff))
print('Percent DistOut Differences:', np.round(do_diff_perc,2), ", Max Diff:", np.max(do_diff))
print('DONE')

# rand = random.sample(range(0,len(reaches)), 1000)
# for ind in list(range(len(rand))):
#     test = np.where(node_rch == reaches[rand[ind]])[0]
#     print(reaches[rand[ind]], 
#           abs(np.round(sum(node_len[test])-rch_len[rand[ind]])), 
#           abs(np.round(max(node_dist[test])-rch_dist[rand[ind]])))
# print('DONE')

