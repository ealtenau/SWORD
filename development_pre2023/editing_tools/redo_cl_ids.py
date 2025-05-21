import netCDF4 as nc
import numpy as np

region = 'NA'
version = 'v16'
fn_nc = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'

sword = nc.Dataset(fn_nc, 'r+')
cl_rch = np.array(sword.groups['centerlines'].variables['reach_id'][:])
cl_node = np.array(sword.groups['centerlines'].variables['node_id'][:])
cl_id = np.array(sword.groups['centerlines'].variables['cl_id'][:])
Reaches = np.array(sword.groups['reaches'].variables['reach_id'][:])
Nodes = np.array(sword.groups['nodes'].variables['node_id'][:])
node_rch = np.array(sword.groups['nodes'].variables['reach_id'][:])
rch_ids = np.array(sword.groups['reaches'].variables['cl_ids'][:])
node_ids = np.array(sword.groups['nodes'].variables['cl_ids'][:])
unq_rch = np.unique(cl_rch[0,:])

new_cl_ids = np.zeros(len(cl_id))
new_rch_ids = np.zeros([2, len(rch_ids[0,:])])
new_node_ids = np.zeros([2, len(node_ids[0,:])])
cnt = 0
for r in list(range(len(unq_rch))):
    print(r, len(unq_rch)-1)
    pts = np.where(cl_rch[0,:] == unq_rch[r])[0]
    rch = np.where(Reaches == unq_rch[r])[0]
    new_vals = cl_id[pts]-np.min(cl_id[pts])+cnt
    new_cl_ids[pts] = new_vals
    new_rch_ids[0,rch] = np.min(new_vals)
    new_rch_ids[1,rch] = np.max(new_vals)
    ### nodes 
    unq_nodes = np.unique(cl_node[0,pts])
    for n in list(range(len(unq_nodes))):
        nds = np.where(Nodes == unq_nodes[n])[0]
        new_node_ids[0,nds] = np.min(new_vals[np.where(cl_node[0,pts] == unq_nodes[n])[0]])
        new_node_ids[1,nds] = np.max(new_vals[np.where(cl_node[0,pts] == unq_nodes[n])[0]])
    cnt = np.max(new_vals)+1

sword.groups['centerlines'].variables['cl_id'][:] = new_cl_ids
sword.groups['reaches'].variables['cl_ids'][:] = new_rch_ids
sword.groups['nodes'].variables['cl_ids'][:] = new_node_ids

print(len(np.unique(cl_id)), cl_id.shape)
print(len(np.unique(new_cl_ids)), new_cl_ids.shape)
sword.close()