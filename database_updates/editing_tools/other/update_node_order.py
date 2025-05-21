import numpy as np
import netCDF4 as nc
import geopandas as gp
import pandas as pd 
from scipy import spatial as sp
import matplotlib.pyplot as plt

region = 'NA'
version = 'v17b'
sword_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+\
    '/netcdf/'+region.lower()+'_sword_'+version+'.nc'
csv_dir = '/Users/ealtenau/Documents/SWORD_Dev/update_requests/'+version+'/'+region+\
    '/'+region.lower()+'_node_order_problems.csv'

sword = nc.Dataset(sword_dir,'r+')
updates = pd.read_csv(csv_dir)

cl_rchs = np.array(sword['/centerlines/reach_id/'][0,:])
cl_ids = np.array(sword['/centerlines/cl_id/'][:])
cl_nodes = np.array(sword['/centerlines/node_id/'][:])
cl_x = np.array(sword['/centerlines/x/'][:])
cl_y = np.array(sword['/centerlines/y/'][:])
nodes = np.array(sword['/nodes/node_id/'][:])
nx = np.array(sword['/nodes/x/'][:])
ny = np.array(sword['/nodes/y/'][:])
node_rchs = np.array(sword['/nodes/reach_id/'][:])
node_dist = np.array(sword['/nodes/dist_out/'][:])

#make a copy for updating. 
new_nodes = np.copy(nodes) 
new_cl_node_ids = np.copy(cl_nodes)
new_dist = np.copy(node_dist)

unq_rchs = np.array(updates['reach_id'])
for r in list(range(len(unq_rchs))):
    print(r, len(unq_rchs)-1)
    rch = np.where(cl_rchs == unq_rchs[r])[0]
    sort_ids = np.argsort(cl_ids[rch])
    orig_nodes = cl_nodes[0,rch[sort_ids]]
    node_nums = np.array([int(str(n)[-4:-1]) for n in orig_nodes])
    node_diff = abs(np.diff(node_nums))
    node_diff = np.insert(node_diff,0,0)
    breaks = np.where(node_diff > 0)[0]
    
    idx = np.unique(node_nums, return_index=True)[1]
    old_nums = np.array([node_nums[index] for index in sorted(idx)])
    new_nums = np.array(list(range(len(old_nums))))+1
    #print(np.array([old_nums, new_nums]).T[0:20,:])

    #creating new correctly ordered node numbers based on breaks. 
    new_cl_nums = np.zeros(len(rch),dtype=int)
    for old in list(range(len(old_nums))):
        pts = np.where(node_nums == old_nums[old])[0]
        new_cl_nums[pts] = new_nums[old]
    # print(np.array([node_nums, new_cl_nums]).T[0:20,:])
    
    #formatting the new node ids. 
    new_node_ids = np.zeros(len(rch),dtype=int)
    for n in list(range(len(new_cl_nums))):
        if len(str(new_cl_nums[n])) == 1:
            fill = '00'
            new_node_ids[n] = int(str(orig_nodes[n])[0:10]+fill+str(new_cl_nums[n])+str(orig_nodes[n])[-1])
        if len(str(new_cl_nums[n])) == 2:
            fill = '0'
            new_node_ids[n] = int(str(orig_nodes[n])[0:10]+fill+str(new_cl_nums[n])+str(orig_nodes[n])[-1])
        if len(str(new_cl_nums[n])) == 3:
            new_node_ids[n] = int(str(orig_nodes[n])[0:10]+str(new_cl_nums[n])+str(orig_nodes[n])[-1]) 
    #print(np.array([orig_nodes, new_node_ids]).T[0:20,:])
    new_cl_node_ids[0,rch[sort_ids]] = new_node_ids

    #updating the node dimension 
    indexes = np.unique(orig_nodes, return_index=True)[1]
    old_ids = np.array([orig_nodes[index] for index in sorted(indexes)])
    indexes2 = np.unique(new_node_ids, return_index=True)[1]
    new_ids = np.array([new_node_ids[index] for index in sorted(indexes2)])
    #print(np.array([old_ids, new_ids]).T)
    if len(old_ids) == len(new_ids):
        for ind in list(range(len(old_ids))):
            nind = np.where(nodes == old_ids[ind])[0]
            new_nodes[nind] = new_ids[ind]
    else:
        print('!!! Old Number of Nodes Differs from New Number of Nodes !!!')
        break
    
    #finally sort distance from outlet by new node number order
    nrch = np.where(node_rchs == unq_rchs[r])[0] 
    nsort = np.argsort(new_nodes[nrch]) 
    dist_sort = np.sort(node_dist[nrch])
    new_dist[nrch[nsort]] = dist_sort

#updating the other columns in the centerline node dimension. 
cl_pts = np.vstack((cl_x, cl_y)).T
kdt = sp.cKDTree(cl_pts)
pt_dist, pt_ind = kdt.query(cl_pts, k = 4)
id_arr = new_cl_node_ids[0,pt_ind]
row_sum = np.sum(abs(np.diff(id_arr, axis=1)), axis = 1)
update = np.where(row_sum > 0)[0]
new_cl_node_ids[1:4,update] = id_arr[update,1:4].T

#update netcdf
sword['/centerlines/node_id/'][:] = new_cl_node_ids
sword['/nodes/node_id/'][:] = new_nodes
sword['/nodes/dist_out/'][:] = new_dist
sword.close()

print('Done with', region)



# plt.scatter(cl_x[rch[sort_ids]], cl_y[rch[sort_ids]], c=cl_nodes[0,rch[sort_ids]], s = 5)
# plt.show()

# plt.scatter(cl_x[rch[sort_ids]], cl_y[rch[sort_ids]], c=new_cl_node_ids[0,rch[sort_ids]], s = 5)
# plt.show()

# nds = np.where(node_rchs == unq_rchs[r])[0]
# plt.scatter(nx[nds], ny[nds], c=nodes[nds], s = 5)
# plt.show()

# plt.scatter(nx[nds], ny[nds], c=new_nodes[nds], s = 5)
# plt.show()