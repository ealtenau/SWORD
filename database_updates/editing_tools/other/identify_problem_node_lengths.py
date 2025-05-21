import pandas as pd
import numpy as np
import netCDF4 as nc

region = 'NA'
version = 'v18'
nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'\
    +region.lower()+'_sword_'+version+'.nc'
out_dir = '/Users/ealtenau/Documents/SWORD_Dev/update_requests/'+version+'/'+region+'/'

sword = nc.Dataset(nc_fn)
ndist18 = np.array(sword.groups['nodes'].variables['dist_out'][:])
nrch18 = np.array(sword.groups['nodes'].variables['reach_id'][:])
rch18 = np.array(sword.groups['reaches'].variables['reach_id'][:])
dist18 = np.array(sword.groups['reaches'].variables['dist_out'][:])
len18 = np.array(sword.groups['reaches'].variables['reach_length'][:])
node18 = np.array(sword.groups['nodes'].variables['node_id'][:])
nlen18 = np.array(sword.groups['nodes'].variables['node_length'][:])
sword.close()

problem = []
length = []
for r in list(range(len(rch18))):
    nds = np.where(nrch18 == rch18[r])[0]
    sort_nodes = np.argsort(node18[nds])
    node_cs = np.cumsum(nlen18[nds[sort_nodes]])
    if np.round(max(node_cs)-len18[r]) != 0:
        problem.append(rch18[r])
        length.append(np.round(max(node_cs)-len18[r]))
        # print(rch18[r])
        # print(max(node_cs), len18[r])
        # print(max(ndist18[nds]), dist18[r])

df = {'reach_id': np.array(problem).astype('int64'), 'len_diff': np.array(length).astype('int64')}
df = pd.DataFrame(df)
df.to_csv(out_dir+region.lower()+'_problematic_node_len.csv', index=False)
print('Done. Problem Reaches:', len(problem))