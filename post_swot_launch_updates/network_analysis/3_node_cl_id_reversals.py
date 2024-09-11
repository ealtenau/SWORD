import numpy as np
import netCDF4 as nc
import geopandas as gp
import pandas as pd
import argparse
import sys
import time
import matplotlib.pyplot as plt

start_all = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

# region = 'NA'
# version = 'v17'

# nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/na_sword_v17_reversal_testing.nc'
nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'\
    +version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
outpath = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Topology/'+region+'/'

sword = nc.Dataset(nc_fn,'r+')

reaches = np.array(sword.groups['reaches'].variables['reach_id'][:])
rch_id_up = np.array(sword.groups['reaches'].variables['rch_id_up'][:])
rch_id_dn = np.array(sword.groups['reaches'].variables['rch_id_dn'][:])
n_rch_up = np.array(sword.groups['reaches'].variables['n_rch_up'][:])
n_rch_dn = np.array(sword.groups['reaches'].variables['n_rch_down'][:])
cl_ids = np.array(sword.groups['centerlines'].variables['cl_id'][:])
cl_rchs = np.array(sword.groups['centerlines'].variables['reach_id'][:])
cl_nodes = np.array(sword.groups['centerlines'].variables['node_id'][:])
cl_x = np.array(sword.groups['centerlines'].variables['x'][:])
cl_y = np.array(sword.groups['centerlines'].variables['y'][:])
nodes = np.array(sword.groups['nodes'].variables['node_id'][:])
node_rchs = np.array(sword.groups['nodes'].variables['reach_id'][:])
node_cl_ids = np.array(sword.groups['nodes'].variables['cl_ids'][:])
nx = np.array(sword.groups['nodes'].variables['x'][:])
ny = np.array(sword.groups['nodes'].variables['y'][:])

#copies of pre-edit centerline and node ids. 
old_cl_ids = np.copy(cl_ids)
old_cl_nodes = np.copy(cl_nodes)
old_nodes = np.copy(nodes)
old_node_cl_ids = np.copy(node_cl_ids)

start = time.time()
print('Reversing Centerline IDs')
cl_rev = []
for r in list(range(len(reaches))):
    cl_ind = np.where(cl_rchs[0,:] == reaches[r])[0]
    min_id = min(cl_ids[cl_ind])
    max_id = max(cl_ids[cl_ind])
    min_ind = np.where(cl_ids[cl_ind] == min_id)[0] #cl_ids[cl_ind[min_ind]]
    max_ind = np.where(cl_ids[cl_ind] == max_id)[0] #cl_ids[cl_ind[max_ind]]
    up_nghs = cl_rchs[1:,cl_ind[max_ind]]
    up_nghs = up_nghs[up_nghs>0]
    dn_nghs = cl_rchs[1:,cl_ind[min_ind]]
    dn_nghs = dn_nghs[dn_nghs>0]
    if len(up_nghs) == 0 and len(dn_nghs) > 0:
        if dn_nghs[0] in rch_id_up[:,r]:
            cl_rev.append(reaches[r])
            sort_inds = np.argsort(cl_ids[cl_ind])
            cl_ids[cl_ind[sort_inds]] = cl_ids[cl_ind[sort_inds]][::-1]       

    if len(dn_nghs) == 0 and len(up_nghs) > 0:
        if up_nghs[0] in rch_id_dn[:,r]:
            cl_rev.append(reaches[r])
            sort_inds = np.argsort(cl_ids[cl_ind])
            cl_ids[cl_ind[sort_inds]] = cl_ids[cl_ind[sort_inds]][::-1]

    if len(dn_nghs) > 0 and len(up_nghs) > 0:
        if up_nghs[0] in rch_id_dn[:,r] or dn_nghs[0] in rch_id_up[:,r]: #use to be and 
            cl_rev.append(reaches[r])
            sort_inds = np.argsort(cl_ids[cl_ind])
            cl_ids[cl_ind[sort_inds]] = cl_ids[cl_ind[sort_inds]][::-1]
end = time.time()
print(str(np.round((end-start)/60,2))+' mins')

start = time.time()
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
        # nodes_dist[node_ind[sort_inds2]] = nodes_dist[node_ind[sort_inds2]][::-1] #reversing node dist out
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

#writing csv files for reversed nodes and centerline ids.             
print('Writing CSV Files')
cl_csv = pd.DataFrame({"reach_id": cl_rev})
cl_csv.to_csv(outpath+'centerline_reversals.csv', index = False)
node_csv = pd.DataFrame({"reach_id": node_rev})
node_csv.to_csv(outpath+'node_reversals.csv', index = False)
issue_csv = pd.DataFrame({"reach_id": order_issues})
issue_csv.to_csv(outpath+'order_problems.csv', index = False)

#updating the netcdf. 
print('Updating the NetCDF')
sword.groups['nodes'].variables['node_id'][:] = nodes
sword.groups['nodes'].variables['cl_ids'][:] = node_cl_ids
sword.groups['centerlines'].variables['cl_id'][:] = cl_ids
sword.groups['centerlines'].variables['node_id'][:] = cl_nodes
sword.close()
end_all = time.time()
print('*** '+region + ' Done in: '+str(np.round((end_all-start_all)/60,2))+' mins ***')


# np.array([old_nodes[node_ind], nodes[node_ind]]).T
# np.array([old_cl_nodes[0,cl_ind], cl_nodes[0,cl_ind]]).T
# np.array([cl_ids[cl_ind], cl_nodes[0,cl_ind]]).T
# np.array([old_cl_ids[cl_ind], old_cl_nodes[0,cl_ind]]).T


# plt.scatter(cl_x[cl_ind[sort_cl_inds]], cl_y[cl_ind[sort_cl_inds]], c=old_cl_nodes[0,cl_ind[sort_cl_inds]], s = 5)
# plt.show()

# plt.scatter(cl_x[cl_ind[sort_cl_inds]], cl_y[cl_ind[sort_cl_inds]], c=cl_nodes[0,cl_ind[sort_cl_inds]], s = 5)
# plt.show()

# plt.scatter(nx[node_ind[sort_inds2]], ny[node_ind[sort_inds2]], c=old_nodes[node_ind[sort_inds2]], s = 5)
# plt.show()

# plt.scatter(nx[node_ind[sort_inds2]], ny[node_ind[sort_inds2]], c=nodes[node_ind[sort_inds2]], s = 5)
# plt.show()


# np.diff(old_cl_nodes[0,cl_ind[sort_cl_inds]])
# np.diff(cl_nodes[0,cl_ind[sort_cl_inds]])