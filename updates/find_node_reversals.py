from __future__ import division
import numpy as np
import netCDF4 as nc
import pandas as pd

region = 'OC'
version = 'v16'
fn_nc = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'

sword = nc.Dataset(fn_nc)
nodes = sword.groups['nodes'].variables['node_id'][:]
node_wse= sword.groups['nodes'].variables['wse'][:]
node_rchs = sword.groups['nodes'].variables['reach_id'][:]

unq_rchs = np.unique(node_rchs)
node_reverse = []
for r in list(range(len(unq_rchs))):
    print(r, len(unq_rchs)-1)
    n = np.where(node_rchs == unq_rchs[r])
    min_wse = node_wse[n][np.where(nodes[n] == np.min(nodes[n]))]
    max_wse = node_wse[n][np.where(nodes[n] == np.max(nodes[n]))]
    if max_wse < min_wse:
        node_reverse.append(unq_rchs[r])

node_reverse = np.array(node_reverse)
df = pd.DataFrame(node_reverse)
df.to_csv('/Users/ealteanau/Documents/SWORD_Dev/update_requests/v16/'+region.lower()+'_'+version+'_node_reversals.csv')
print('DONE')