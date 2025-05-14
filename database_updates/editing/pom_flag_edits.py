#checking and fixing where number of nodes is not correct per reach
#checking and fixing if node ids do not trend with distance from outlet (within the reach)
#fixing ghost reaches distance from outlet that were created, have same as neighboring reach. 

import numpy as np
import netCDF4 as nc
import argparse
import time
import matplotlib.pyplot as plt

# start_all = time.time()
# parser = argparse.ArgumentParser()
# parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
# parser.add_argument("version", help="version", type = str)
# args = parser.parse_args()

# region = args.region
# version = args.version

region = 'SA'
version = 'v18'

nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'\
    +version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
outpath = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Topology/'+version+'/'+region+'/'

sword = nc.Dataset(nc_fn,'r+')
# sword = nc.Dataset(nc_fn)

reaches = np.array(sword.groups['reaches'].variables['reach_id'][:])
rch_lon = np.array(sword.groups['reaches'].variables['x'][:])
rch_lat = np.array(sword.groups['reaches'].variables['y'][:])
rch_id_up = np.array(sword.groups['reaches'].variables['rch_id_up'][:])
rch_id_dn = np.array(sword.groups['reaches'].variables['rch_id_dn'][:])
n_rch_up = np.array(sword.groups['reaches'].variables['n_rch_up'][:])
n_rch_dn = np.array(sword.groups['reaches'].variables['n_rch_down'][:])
rch_dist = np.array(sword.groups['reaches'].variables['dist_out'][:])
rch_nnodes = np.array(sword.groups['reaches'].variables['n_nodes'][:])
rch_len = np.array(sword.groups['reaches'].variables['reach_length'][:])
nodes = np.array(sword.groups['nodes'].variables['node_id'][:])
node_rchs = np.array(sword.groups['nodes'].variables['reach_id'][:])
nx = np.array(sword.groups['nodes'].variables['x'][:])
ny = np.array(sword.groups['nodes'].variables['y'][:])
node_dist = np.array(sword.groups['nodes'].variables['dist_out'][:])
Type = np.array([int(str(r)[-1]) for r in reaches])

#correcting dist_out for created ghost reaches. 
ghost_rchs = reaches[np.where(Type == 6)[0]]
ghost_flag = []
for ind in list(range(len(ghost_rchs))):
    rch = np.where(reaches == ghost_rchs[ind])[0]
    #headwater
    if n_rch_up[rch] == 0:
        nghs = rch_id_dn[:,rch]; nghs = nghs[nghs>0]
        ngh_dist = np.array([rch_dist[np.where(reaches == n)][0] for n in nghs])
        diff = abs(rch_dist[rch]-ngh_dist)
        if min(diff) < 1:
            ghost_flag.append(ghost_rchs[ind])
            ngh_ind = np.where(reaches == nghs[np.where(diff < 1)[0]])[0]
            rch_dist[ngh_ind] = rch_dist[ngh_ind] - rch_len[rch]
    #outlet
    else:
        nghs = rch_id_up[:,rch]; nghs = nghs[nghs>0]
        ngh_dist = np.array([rch_dist[np.where(reaches == n)][0] for n in nghs])
        diff = abs(rch_dist[rch]-ngh_dist)
        if min(diff) < 1:
            ghost_flag.append(ghost_rchs[ind])
            rch_dist[rch] = rch_len[rch]


# correcting number of nodes in reach and node dist_out trend. 
nnode_flag = []
ndist_flag = []
for idx in list(range(len(reaches))):
    nind = np.where(node_rchs == reaches[idx])[0]
    #updating number of nodes in a reach if not correct. 
    if len(nind) != rch_nnodes[idx]:
        # print(idx)
        nnode_flag.append(reaches[idx])
        rch_nnodes[idx] = len(nind)
    #reversing distance from outlet in nodes if opposite trend from node ids. 
    mn = np.where(nodes[nind] == min(nodes[nind]))[0]
    mx = np.where(nodes[nind] == max(nodes[nind]))[0]
    if node_dist[nind[mn]] > node_dist[nind[mx]]:
        # print(idx)
        ndist_flag.append(reaches[idx])
        node_dist[nind] = node_dist[nind][::-1]


#updating the netcdf. 
# print('Updating the NetCDF')
# sword.groups['nodes'].variables['dist_out'][:] = node_dist
# sword.groups['reaches'].variables['dist_out'][:] = rch_dist
# sword.groups['reaches'].variables['n_nodes'][:] = rch_nnodes
# sword.close()

print('number of ghost reach distances updated:', len(ghost_flag))
print('number of nodes in reach updated:', len(nnode_flag))
print('number of node distances updated:', len(ndist_flag))


# plt.scatter(nx[nind],ny[nind], c=node_dist[nind], s=5)
# plt.show()

# plt.scatter(nx[nind],ny[nind], c=node_dist[nind][::-1], s=5)
# plt.show()