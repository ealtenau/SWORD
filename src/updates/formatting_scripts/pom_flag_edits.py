# -*- coding: utf-8 -*-
"""
Flagged Edits (pom_flag_edits.py)
===============================================================

This script checks and corrects promenient issues flagged by 
Pierre-Olivier Malaterre. Key problems addressed:
 - Checking and fixing where number of nodes is not correct per reach
 - Checking and fixing if node ids do not trend with distance 
    from outlet (within the reach)
 - Fixing ghost reaches distance from outlet that were created, 
    have same as neighboring reach. 


The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python path/to/pom_flag_edits.py NA v17

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import argparse
import time
from src.updates.sword import SWORD

start_all = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

#read data. 
sword = SWORD(main_dir, region, version)
sword.copy() #copies original file for version control.

#create type array from Reach and Node IDs. 
Type = np.array([int(str(r)[-1]) for r in sword.reaches.id])
#create outpath.
outpath = sword.paths['topo_dir']

#correcting dist_out for added ghost reaches. 
ghost_rchs = sword.reaches.id[np.where(Type == 6)[0]]
ghost_flag = []
for ind in list(range(len(ghost_rchs))):
    rch = np.where(sword.reaches.id == ghost_rchs[ind])[0]
    #headwater
    if sword.reaches.n_rch_up[rch] == 0:
        nghs = sword.reaches.rch_id_down[:,rch]; nghs = nghs[nghs>0]
        ngh_dist = np.array([sword.reaches.dist_out[np.where(sword.reaches.id == n)][0] for n in nghs])
        diff = abs(sword.reaches.dist_out[rch]-ngh_dist)
        if min(diff) < 1:
            ghost_flag.append(ghost_rchs[ind])
            ngh_ind = np.where(sword.reaches.id == nghs[np.where(diff < 1)[0]])[0]
            sword.reaches.dist_out[ngh_ind] = sword.reaches.dist_out[ngh_ind] - sword.reaches.len[rch]
    #outlet
    else:
        nghs = sword.reaches.rch_id_up[:,rch]; nghs = nghs[nghs>0]
        ngh_dist = np.array([sword.reaches.dist_out[np.where(sword.reaches.id == n)][0] for n in nghs])
        diff = abs(sword.reaches.dist_out[rch]-ngh_dist)
        if min(diff) < 1:
            ghost_flag.append(ghost_rchs[ind])
            sword.reaches.dist_out[rch] = sword.reaches.len[rch]


#correcting number of nodes in reach and node dist_out trend. 
nnode_flag = []
ndist_flag = []
for idx in list(range(len(sword.reaches.id))):
    nind = np.where(sword.nodes.reach_id == sword.reaches.id[idx])[0]
    #updating number of nodes in a reach if not correct. 
    if len(nind) != sword.reaches.rch_n_nodes[idx]:
        # print(idx)
        nnode_flag.append(sword.reaches.id[idx])
        sword.reaches.rch_n_nodes[idx] = len(nind)
    #reversing distance from outlet in nodes if opposite trend from node ids. 
    mn = np.where(sword.nodes.id[nind] == min(sword.nodes.id[nind]))[0]
    mx = np.where(sword.nodes.id[nind] == max(sword.nodes.id[nind]))[0]
    if sword.nodes.dist_out[nind[mn]] > sword.nodes.dist_out[nind[mx]]:
        # print(idx)
        ndist_flag.append(sword.reaches.id[idx])
        sword.nodes.dist_out[nind] = sword.nodes.dist_out[nind][::-1]


#updating the netcdf. 
print('Updating the NetCDF')
sword.save_nc()

print('number of ghost reach distances updated:', len(ghost_flag))
print('number of nodes in reach updated:', len(nnode_flag))
print('number of node distances updated:', len(ndist_flag))
