# -*- coding: utf-8 -*-
"""
Postformatting MHV Additions to SWORD
(additions_final_formatting.py)
===================================================

This script updates specific attributes in the added 
MHV reaches in SWORD.

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python path/to/additions_final_formatting.py NA v17

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import netCDF4 as nc
from datetime import datetime
import shutil
import argparse 

##################################################################

def find_character_in_array(arr, char):
    """
    Finds the indices of a character within an array of strings.

    Args:
        arr: A list of strings.
        char: The character to search for.

    Returns:
        A list of tuples, where each tuple contains:
            - The index of the string in the array.
            - The index of the character within that string.
        Returns an empty list if the character is not found.
    """
    results = []
    for i, string in enumerate(arr):
        for j, c in enumerate(string):
            if c == char:
                results.append(i)
    return results

##################################################################

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

nc_fn = main_dir+'/data/outputs/Reaches_Nodes/'+version+'/netcdf/'\
    +region.lower()+'_sword_'+version+'.nc'
#copy sword for version control.
current_datetime = datetime.now()
copy_fn = nc_fn[:-3]+'_copy_'+current_datetime.strftime("%Y-%m-%d_%H-%M-%S")+'.nc'
shutil.copy2(nc_fn, copy_fn)

#read sword data. 
sword = nc.Dataset(nc_fn,'r+')
edit_flag = np.array(sword['/reaches/edit_flag'][:])
reaches = np.array(sword['/reaches/reach_id'][:])
rch_id_up = np.array(sword['/reaches/rch_id_up'][:])
rch_id_dn = np.array(sword['/reaches/rch_id_dn'][:])
n_rch_up = np.array(sword['/reaches/n_rch_up'][:])
n_rch_dn = np.array(sword['/reaches/n_rch_down'][:])
rch_net = np.array(sword['/reaches/network'][:])
rch_ends = np.array(sword['/reaches/end_reach'][:])
rch_len = np.array(sword['/reaches/reach_length'][:])
rch_dist = np.array(sword['/reaches/dist_out'][:])
node_rchs = np.array(sword['/nodes/reach_id'][:])
node_net = np.array(sword['/nodes/network'][:])
node_ends = np.array(sword['/nodes/end_reach'][:])
node_len = np.array(sword['/nodes/node_length'][:])
node_dist = np.array(sword['/nodes/dist_out'][:])

l2_basins = np.array([int(str(ind)[0:2]) for ind in reaches])

subset = find_character_in_array(edit_flag, '7')
subset = np.array(subset)
add_rchs = reaches[subset]

print('Creating New Flag')
mhv_rch_flag = np.zeros(len(reaches))
mhv_node_flag = np.zeros(len(node_rchs))
nind = np.where(np.in1d(node_rchs, reaches[subset]) == True)[0]
mhv_rch_flag[subset] = 1
mhv_node_flag[nind] = 1

#for outlets find all neighbors and fill in the network number. 
print('Updating Network Variable')
outlets = np.where(n_rch_dn == 0)[0]
out_rchs = reaches[outlets]
for out in list(range(len(out_rchs))):
    # print(out, len(out_rchs)-1)
    rch = np.where(reaches == out_rchs[out])[0]
    net = rch_net[rch]
    up_rchs = np.unique(rch_id_up[:,rch]); up_rchs = up_rchs[up_rchs>0]
    while len(up_rchs) > 0:
        fill = np.where(np.in1d(reaches, up_rchs) == True)[0]
        if min(rch_net[fill]) == 0:
            update = np.where(rch_net[fill] == 0)[0]
            rch_net[fill[update]] = net
            #nodes 
            nfill = np.where(np.in1d(node_rchs, reaches[fill[update]]) == True)[0]
            node_net[nfill] = net
        up_rchs = np.unique(rch_id_up[:,fill]); up_rchs = up_rchs[up_rchs>0]
#creating unique network numbers for entire continent. 
unq_l2 = np.unique(l2_basins)
cnt = 0
net_tot = np.zeros(len(unq_l2))
for b in list(range(len(unq_l2))):
    basin = np.where(l2_basins == unq_l2[b])[0]
    unq_nets = np.unique(rch_net[basin])
    net_tot[b] = len(unq_nets)
    # print(unq_l2[b], min(unq_nets), len(unq_nets))
    rch_net[basin] = rch_net[basin] + cnt
    cnt = max(rch_net[basin])
#print(sum(net_tot))
#update nodes
unets = np.unique(rch_net)
for n in list(range(len(unets))):
    rnet = np.where(rch_net == unets[n])[0]
    nnet = np.where(np.in1d(node_rchs, reaches[rnet]) == True)[0]
    node_net[nnet] = unets[n]
print('new networks:', max(rch_net), max(node_net))

print('Updating End Reach Variable')
new_rch_ends = np.zeros(len(reaches))
new_node_ends = np.zeros(len(node_rchs))
hw = np.where(n_rch_up == 0)[0]
ot = np.where(n_rch_dn == 0)[0]
junc1 = np.where(n_rch_up > 1)[0]
junc2 = np.where(n_rch_dn > 1)[0]
node_hw = np.where(np.in1d(node_rchs, reaches[hw]) == True)[0]
node_ot = np.where(np.in1d(node_rchs, reaches[ot]) == True)[0]
node_junc1 = np.where(np.in1d(node_rchs, reaches[junc1]) == True)[0]
node_junc2 = np.where(np.in1d(node_rchs, reaches[junc2]) == True)[0]
new_rch_ends[junc1] = 3
new_rch_ends[junc1] = 3
new_rch_ends[hw] = 1
new_rch_ends[ot] = 2
new_node_ends[node_junc1] = 3
new_node_ends[node_junc1] = 3
new_node_ends[node_hw] = 1
new_node_ends[node_ot] = 2
# np.unique(rch_ends)
# np.unique(node_ends)

#think about updating path segments?
print('Updating Path Segment Variable')
b1 = np.where(n_rch_dn == 0)[0]
n2 = np.unique(rch_id_up[:,np.where(n_rch_up > 1)[0]]); n2 = n2[n2>0]
b2 = np.where(np.in1d(reaches, n2)==True)[0]
b3 = np.where(n_rch_dn > 1)[0]
start_pts = np.append(b1, b2)
start_pts = np.append(start_pts, b3)
unq_rchs = np.unique(reaches[start_pts])
path_segs = np.zeros(len(reaches))
node_path_segs = np.zeros(len(node_rchs))
cnt = 1
for r in list(range(len(unq_rchs))):
    # print(r, len(unq_rchs)-1, cnt)
    rch = np.where(reaches == unq_rchs[r])[0]
    path_segs[rch] = cnt
    node_path_segs[np.where(np.in1d(node_rchs, unq_rchs[r]))[0]] = cnt
    #get all neighbors 
    up_rchs = np.unique(rch_id_up[:,rch]); up_rchs = up_rchs[up_rchs>0]
    while len(up_rchs) == 1:
        fill = np.where(np.in1d(reaches, up_rchs) == True)[0]
        path_segs[fill] = cnt
        node_path_segs[np.where(np.in1d(node_rchs, up_rchs)==True)[0]] = cnt
        up_rchs = np.unique(rch_id_up[:,fill]); up_rchs = up_rchs[up_rchs>0]
    cnt = cnt+1 
          
#think about updating reaches with zero reach lengths?
print('Filling Zero Reach and Node Lengths')
zero_rchs = np.where(rch_len == 0)[0]
zero_nodes = np.where(node_len == 0)[0]
rch_len[zero_rchs] = 90
node_len[zero_nodes] = 90
rch_dist[zero_rchs] = rch_dist[zero_rchs]+90
node_dist[zero_nodes] = node_dist[zero_nodes]+90

print('Updating NetCDF')
sword['/reaches/network'][:] = rch_net
sword['/reaches/end_reach'][:] = new_rch_ends
sword['/nodes/network'][:] = node_net
sword['/nodes/end_reach'][:] = new_node_ends
sword['/reaches/path_segs'][:] = path_segs
sword['/nodes/path_segs'][:] = node_path_segs
sword['/reaches/reach_length'][:] = rch_len
sword['/nodes/node_length'][:] = node_len
sword['/reaches/dist_out'][:] = rch_dist
sword['/nodes/dist_out'][:] = node_dist
## new flag variable
sword.groups['reaches'].createVariable('add_flag', 'i4', ('num_reaches',), fill_value=-9999.)
sword.groups['nodes'].createVariable('add_flag', 'i4', ('num_nodes',), fill_value=-9999.)
sword['/reaches/add_flag'][:] = mhv_rch_flag
sword['/nodes/add_flag'][:] = mhv_node_flag
sword.close()