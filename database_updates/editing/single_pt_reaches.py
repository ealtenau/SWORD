import numpy as np
import netCDF4 as nc
import pandas as pd

region = 'NA'
version = 'v18'
sword_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'
sword = nc.Dataset(sword_dir+region.lower()+'_sword_'+version+'.nc')
outpath = '/Users/ealtenau/Documents/SWORD_Dev/update_requests/'+version+'/'+region+'/'+region.lower()+'_'+version+'_single_pt_rchs.csv'

#centerline variables to update.
cl_ids = sword.groups['centerlines'].variables['cl_id'][:]
cl_rchs = sword.groups['centerlines'].variables['reach_id'][:,:]
cl_nodes = sword.groups['centerlines'].variables['node_id'][:,:]
#node variables to update.
nodes = sword.groups['nodes'].variables['node_id'][:]
node_rchs = sword.groups['nodes'].variables['reach_id'][:]
node_cl_ids = sword.groups['nodes'].variables['cl_ids'][:]
node_x = sword.groups['nodes'].variables['x'][:]
node_y = sword.groups['nodes'].variables['y'][:]
node_len = sword.groups['nodes'].variables['node_length'][:]
node_dist = sword.groups['nodes'].variables['dist_out'][:]
#reach variables to update.
rchs = sword.groups['reaches'].variables['reach_id'][:]
rch_cl_ids = sword.groups['reaches'].variables['cl_ids'][:]
rch_x = sword.groups['reaches'].variables['x'][:]
rch_xmin = sword.groups['reaches'].variables['x_min'][:]
rch_xmax = sword.groups['reaches'].variables['x_max'][:]
rch_y = sword.groups['reaches'].variables['y'][:]
rch_ymin = sword.groups['reaches'].variables['y_min'][:]
rch_ymax = sword.groups['reaches'].variables['y_max'][:]
rch_len = sword.groups['reaches'].variables['reach_length'][:]
rch_dist = sword.groups['reaches'].variables['dist_out'][:]
n_rch_up = sword.groups['reaches'].variables['n_rch_up'][:]
n_rch_down = sword.groups['reaches'].variables['n_rch_down'][:]
rch_id_up = sword.groups['reaches'].variables['rch_id_up'][:,:]
rch_id_dn = sword.groups['reaches'].variables['rch_id_dn'][:,:]

single_pt_rchs = []
for ind in list(range(len(rchs))):
    print(ind)
    pts = np.where(cl_rchs[0,:] == rchs[ind])[0]
    if len(pts) == 1:
        single_pt_rchs.append(ind)

#export reaches to delete.
rch_list = rchs[single_pt_rchs]
df = pd.DataFrame(rch_list)
df.to_csv(outpath)
print(len(rch_list)) 