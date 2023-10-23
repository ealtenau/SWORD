import numpy as np
import netCDF4 as nc

region = 'AS'
version = 'v16'
fn_nc = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'

sword = nc.Dataset(fn_nc)

# centerline dimensions
cl_unq = len(np.unique(sword.groups['centerlines'].variables['cl_id'][:]))
cl_dim = len(sword.groups['centerlines'].variables['cl_id'][:])

# node dimensions
node_unq = len(np.unique(sword.groups['nodes'].variables['node_id'][:]))
node_dim = len(sword.groups['nodes'].variables['node_id'][:])
cl_node_unq = len(np.unique(sword.groups['centerlines'].variables['node_id'][0,:]))

# reach dimensions
rch_unq = len(np.unique(sword.groups['reaches'].variables['reach_id'][:]))
node_rch_unq = len(np.unique(sword.groups['nodes'].variables['reach_id'][:]))
rch_dim = len(sword.groups['reaches'].variables['reach_id'][:])
cl_rch_unq = len(np.unique(sword.groups['centerlines'].variables['reach_id'][0,:]))

print('cl_dim: ' + str(cl_dim) + ' cl_unq: ' + str(cl_unq))
print('node_dim: ' + str(node_dim) + ' node_unq: ' + str(node_unq) + ' cl_node_unq: ' +  str(cl_node_unq))
print('rch_dim: ' + str(rch_dim) + ' rch_unq: ' + str(rch_unq) + ' node_rch_unq: ' + str(node_rch_unq) + ' cl_rch_unq: ' + str(cl_rch_unq))