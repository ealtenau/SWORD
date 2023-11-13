import numpy as np
import netCDF4 as nc
import pandas as pd

region = 'SA'
fn_sword_v16 = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16/netcdf/'+region.lower()+'_sword_v16.nc'
fn_sword_old = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16/netcdf_pre_node_update/'+region.lower()+'_sword_v16.nc'
fn_csv = '/Users/ealteanau/Documents/SWORD_Dev/update_requests/v16/completed/'+region.lower()+'_v16_node_reversals.csv'

sword = nc.Dataset(fn_sword_v16, 'r+')
sword2 = nc.Dataset(fn_sword_old)
csv = pd.read_csv(fn_csv)

reaches = sword.groups['reaches'].variables['reach_id'][:]
rch_cls = sword.groups['centerlines'].variables['reach_id'][0,:]
node_cls = sword2.groups['centerlines'].variables['node_id'][0,:]
new_cls = np.copy(node_cls)

unq_rch = np.array(csv['reach_id'])
for ind in list(range(len(unq_rch))):
    print(ind)
    rch = np.where(rch_cls == unq_rch[ind])[0]
    current_nodes = node_cls[rch]
    new_nodes = current_nodes[::-1]
    new_cls[rch] = new_nodes

sword.groups['centerlines'].variables['node_id'][:,:] = sword2.groups['centerlines'].variables['node_id'][:,:]
sword.groups['centerlines'].variables['node_id'][0,:] = new_cls
sword.close()
sword2.close()

print('Done')