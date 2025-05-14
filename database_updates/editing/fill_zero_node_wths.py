import netCDF4 as nc
import pandas as pd
import numpy as np

region = 'AS'
version = 'v18'
sword_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+\
    '/netcdf/'+region.lower()+'_sword_'+version+'.nc'
out_dir = '/Users/ealtenau/Documents/SWORD_Dev/update_requests/'+version+'/'+region+'/'

sword = nc.Dataset(sword_dir,'r+')
nodes = np.array(sword.groups['nodes'].variables['node_id'][:])
node_rch = np.array(sword.groups['nodes'].variables['reach_id'][:])
node_wth = np.array(sword.groups['nodes'].variables['width'][:])

zero_nodes = np.where(node_wth <= 0)[0]
unq_rchs = np.unique(node_rch[zero_nodes])
for r in list(range(len(unq_rchs))):
    nind = np.where(node_rch == unq_rchs[r])[0]
    min_wth = np.median(node_wth[nind[np.where(node_wth[nind] > 0)[0]]])
    z = np.where(node_wth[nind] <= 0)[0]
    node_wth[nind[z]] = min_wth

#write csv of zero width nodes for reference. 
csv = pd.DataFrame({"node_id": nodes[zero_nodes]})
csv.to_csv(out_dir+region.lower()+'_'+version+'_nodes_zero_widths_filled.csv', index = False)

#update netcdf. 
if min(node_wth) > 0:
    print('Updating NetCDF')
    sword.groups['nodes'].variables['width'][:] = node_wth
    sword.close()