import numpy as np
import netCDF4 as nc
import geopandas as gp
from geopy import distance
import pandas as pd
import random

region = 'NA'
version = 'v18'

nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'\
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

rand = random.sample(range(0,len(reaches)), 500)
for ind in list(range(len(rand))):
    test = np.where(node_rch == reaches[rand[ind]])[0]
    print(reaches[rand[ind]], 
          abs(np.round(sum(node_len[test])-rch_len[rand[ind]])), 
          abs(np.round(max(node_dist[test])-rch_dist[rand[ind]])))

print('DONE')

# np.median(rch_len)
# np.median(node_len)

# check = np.where(reaches == 71140700123)[0]
# test = np.where(node_rch == reaches[check])[0]
# print(reaches[check], 
#       abs(np.round(sum(node_len[test])-rch_len[check])), 
#       abs(np.round(max(node_dist[test])-rch_dist[check])))
