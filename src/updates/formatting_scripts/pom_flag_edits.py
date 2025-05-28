# -*- coding: utf-8 -*-

#checking and fixing where number of nodes is not correct per reach
#checking and fixing if node ids do not trend with distance from outlet (within the reach)
#fixing ghost reaches distance from outlet that were created, have same as neighboring reach. 

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import argparse
import time
import src.updates.sword_utils as swd
# import matplotlib.pyplot as plt

start_all = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

# region = 'OC'
# version = 'v18'

paths = swd.prepare_paths(main_dir, region, version)
sword_fn = paths['geom_dir']+paths['geom_fn']
outpath = paths['topo_dir']

#read data. 
centerlines, nodes, reaches = swd.read_nc(sword_fn)
Type = np.array([int(str(r)[-1]) for r in reaches.id])

#correcting dist_out for created ghost reaches. 
ghost_rchs = reaches[np.where(Type == 6)[0]]
ghost_flag = []
for ind in list(range(len(ghost_rchs))):
    rch = np.where(reaches.id == ghost_rchs[ind])[0]
    #headwater
    if reaches.n_rch_up[rch] == 0:
        nghs = reaches.rch_id_down[:,rch]; nghs = nghs[nghs>0]
        ngh_dist = np.array([reaches.dist_out[np.where(reaches.id == n)][0] for n in nghs])
        diff = abs(reaches.dist_out[rch]-ngh_dist)
        if min(diff) < 1:
            ghost_flag.append(ghost_rchs[ind])
            ngh_ind = np.where(reaches.id == nghs[np.where(diff < 1)[0]])[0]
            reaches.dist_out[ngh_ind] = reaches.dist_out[ngh_ind] - reaches.len[rch]
    #outlet
    else:
        nghs = reaches.rch_id_up[:,rch]; nghs = nghs[nghs>0]
        ngh_dist = np.array([reaches.dist_out[np.where(reaches.id == n)][0] for n in nghs])
        diff = abs(reaches.dist_out[rch]-ngh_dist)
        if min(diff) < 1:
            ghost_flag.append(ghost_rchs[ind])
            reaches.dist_out[rch] = reaches.len[rch]


# correcting number of nodes in reach and node dist_out trend. 
nnode_flag = []
ndist_flag = []
for idx in list(range(len(reaches.id))):
    nind = np.where(nodes.reach_id == reaches.id[idx])[0]
    #updating number of nodes in a reach if not correct. 
    if len(nind) != reaches.rch_n_nodes[idx]:
        # print(idx)
        nnode_flag.append(reaches[idx])
        reaches.rch_n_nodes[idx] = len(nind)
    #reversing distance from outlet in nodes if opposite trend from node ids. 
    mn = np.where(nodes[nind] == min(nodes[nind]))[0]
    mx = np.where(nodes[nind] == max(nodes[nind]))[0]
    if nodes.dist_out[nind[mn]] > nodes.dist_out[nind[mx]]:
        # print(idx)
        ndist_flag.append(reaches[idx])
        nodes.dist_out[nind] = nodes.dist_out[nind][::-1]


#updating the netcdf. 
print('Updating the NetCDF')
swd.discharge_attr_nc(reaches)
swd.write_nc(centerlines, nodes, reaches, sword_fn)

print('number of ghost reach distances updated:', len(ghost_flag))
print('number of nodes in reach updated:', len(nnode_flag))
print('number of node distances updated:', len(ndist_flag))


# plt.scatter(nx[nind],ny[nind], c=node_dist[nind], s=5)
# plt.show()

# plt.scatter(nx[nind],ny[nind], c=node_dist[nind][::-1], s=5)
# plt.show()