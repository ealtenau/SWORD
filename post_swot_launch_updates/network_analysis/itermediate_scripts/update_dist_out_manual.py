import numpy as np
import netCDF4 as nc
import pandas as pd

region = 'NA'
version = 'v17'

csv_fn = '/Users/ealtenau/Documents/SWORD_Dev/update_requests/v17/NA/hb75_dist_out_updates.csv'
nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'\
    +region.lower()+'_sword_'+version+'.nc'

csv = pd.read_csv(csv_fn)
csv_rchs = np.array(csv['reach_id'])
csv_dist = np.array(csv['dist_out'])

sword = nc.Dataset(nc_fn)
nc_rchs = np.array(sword.groups['reaches'].variables['reach_id'][:])
nc_dist = np.array(sword.groups['reaches'].variables['dist_out'][:])
nc_rch_len = np.array(sword.groups['reaches'].variables['reach_length'][:])
nc_node_rchs = np.array(sword.groups['nodes'].variables['reach_id'][:])
nc_node_dist = np.array(sword.groups['nodes'].variables['dist_out'][:])
nc_node_len = np.array(sword.groups['nodes'].variables['node_length'][:])

print('Updating Attributes from SHP File')
for r in list(range(len(csv_rchs))):
    # print(r)
    rch = np.where(nc_rchs == csv_rchs[r])[0]
    nds = np.where(nc_node_rchs == csv_rchs[r])[0]
    print(nc_node_rchs[nds])

    nc_dist[rch] = csv_dist[r]
    node_add_val = csv_dist[r] - nc_rch_len[rch]
    nc_node_dist[nds] = np.cumsum(nc_node_len[nds]) + node_add_val
    
sword.groups['reaches'].variables['dist_out'][:] = nc_dist
sword.groups['nodes'].variables['dist_out'][:] = nc_node_dist
sword.close()



nc_node_len[nds]
np.cumsum(nc_node_len[nds])



nc_node_dist[nds]
np.diff(nc_node_dist[nds])
np.cumsum(np.diff(nc_node_dist[nds]))
