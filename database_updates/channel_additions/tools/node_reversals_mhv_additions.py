import numpy as np
import netCDF4 as nc
import geopandas as gp
import pandas as pd
import argparse
import sys
import time
import matplotlib.pyplot as plt
from scipy import spatial as sp

start_all = time.time()

region = 'OC'
version = 'v18'
update_nc = 'False'

nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'\
    +version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
outpath = '/Users/ealtenau/Documents/SWORD_Dev/update_requests/'+version+'/'+region+'/'

sword = nc.Dataset(nc_fn,'r+')

reaches = np.array(sword.groups['reaches'].variables['reach_id'][:])
rch_id_up = np.array(sword.groups['reaches'].variables['rch_id_up'][:])
rch_id_dn = np.array(sword.groups['reaches'].variables['rch_id_dn'][:])
rch_dist = np.array(sword.groups['reaches'].variables['dist_out'][:])
n_rch_up = np.array(sword.groups['reaches'].variables['n_rch_up'][:])
n_rch_dn = np.array(sword.groups['reaches'].variables['n_rch_down'][:])
edit_flag = np.array(sword.groups['reaches'].variables['edit_flag'][:])
cl_ids = np.array(sword.groups['centerlines'].variables['cl_id'][:])
cl_rchs = np.array(sword.groups['centerlines'].variables['reach_id'][:])
cl_nodes = np.array(sword.groups['centerlines'].variables['node_id'][:])
cl_x = np.array(sword.groups['centerlines'].variables['x'][:])
cl_y = np.array(sword.groups['centerlines'].variables['y'][:])
nodes = np.array(sword.groups['nodes'].variables['node_id'][:])
node_rchs = np.array(sword.groups['nodes'].variables['reach_id'][:])
node_dist = np.array(sword.groups['nodes'].variables['dist_out'][:])
node_cl_ids = np.array(sword.groups['nodes'].variables['cl_ids'][:])
nx = np.array(sword.groups['nodes'].variables['x'][:])
ny = np.array(sword.groups['nodes'].variables['y'][:])

#copies of pre-edit centerline and node ids. 
old_cl_ids = np.copy(cl_ids)
old_cl_nodes = np.copy(cl_nodes)
old_nodes = np.copy(nodes)
old_node_cl_ids = np.copy(node_cl_ids)
old_cl_rchs = np.copy(cl_rchs)

start = time.time()
cl_pts = np.vstack((cl_x, cl_y)).T
kdt = sp.cKDTree(cl_pts)
pt_dist, pt_ind = kdt.query(cl_pts, k = 4)
print('Reversing Node IDs')   
node_rev = []
order_issues = []
for r in list(range(len(reaches))):
    #getting centerline and node dimension indexes for a reach. 
    cl_ind = np.where(cl_rchs[0,:] == reaches[r])[0]
    node_ind = np.where(node_rchs == reaches[r])[0]
    #finding the min and max centerline ids 
    min_id = min(cl_ids[cl_ind])
    max_id = max(cl_ids[cl_ind])
    min_ind = np.where(cl_ids[cl_ind] == min_id)[0] 
    max_ind = np.where(cl_ids[cl_ind] == max_id)[0]
    #seeing if the node id listed for each min and max is in reverse order. 
    if cl_nodes[0,cl_ind[min_ind]] > cl_nodes[0,cl_ind[max_ind]]:
        #finding if there is an existing index ordering issue and skipping. recorded in order_issues. 
        up_int = int(str(cl_nodes[0,cl_ind[min_ind]])[-4:-2])
        dn_int = int(str(cl_nodes[0,cl_ind[max_ind]])[-4:-2])
        if abs(dn_int-up_int) < len(node_ind)/2:
            order_issues.append(reaches[r])
            continue
        #if no existing issues, reverse the node ids. 
        node_rev.append(reaches[r])
        sort_inds2 = np.argsort(nodes[node_ind])
        nodes[node_ind[sort_inds2]] = nodes[node_ind[sort_inds2]][::-1] #reversing node dimension
        node_dist[node_ind[sort_inds2]] = node_dist[node_ind[sort_inds2]][::-1] #reversing node dist out
        #looping through the nodes and reversing the ids in the centerline dimension. 
        #and updating the cl_ids in the the node dimension. 
        subnodes = old_nodes[node_ind]
        for n in list(range(len(subnodes))):
            nind = np.where(old_cl_nodes[0,:] == subnodes[n])[0]
            cl_nodes[0,nind] = nodes[node_ind[n]]
end = time.time()
print(str(np.round((end-start)/60,2))+' mins')

start = time.time()
print('Updating Node Centerline ID Ranges') 
subreaches = np.array(node_rev)  
for r in list(range(len(node_rev))):
    node_ind = np.where(node_rchs == subreaches[r])[0]
    subnodes = nodes[node_ind]
    for n in list(range(len(subnodes))):
            nind = np.where(cl_nodes[0,:] == subnodes[n])[0]
            mn = min(cl_ids[nind]) 
            mx = max(cl_ids[nind]) 
            node_cl_ids[0,node_ind[n]] = mn
            node_cl_ids[1,node_ind[n]] = mx
end = time.time()
print(str(np.round((end-start)/60,2))+' mins')

print('Updating Centerline Node ID Neighbors')
id_arr = cl_nodes[0,pt_ind]
row_sum = np.sum(abs(np.diff(id_arr, axis=1)), axis = 1)
# (len(np.where(row_sum > 0)[0])/len(row_sum))*100
update = np.where(row_sum > 0)[0]
cl_nodes[1:4,update] = id_arr[update,1:4].T

#writing csv files for reversed nodes and centerline ids.             
print('Writing CSV Files')
rev_flag = edit_flag[np.where(np.in1d(reaches, node_rev)==True)[0]]
node_csv = pd.DataFrame({"reach_id": node_rev, "edit_flag": rev_flag})
node_csv.to_csv(outpath+'node_reversals.csv', index = False)
issue_csv = pd.DataFrame({"reach_id": order_issues})
issue_csv.to_csv(outpath+'order_problems.csv', index = False)

#updating the netcdf. 
if update_nc == 'True':
    print('Updating the NetCDF')
    sword.groups['nodes'].variables['node_id'][:] = nodes
    sword.groups['nodes'].variables['cl_ids'][:] = node_cl_ids
    sword.groups['nodes'].variables['dist_out'][:] = node_dist
    sword.groups['centerlines'].variables['node_id'][:] = cl_nodes

sword.close()
end_all = time.time()
print('*** '+region + ' Done in: '+str(np.round((end_all-start_all)/60,2))+' mins ***')
