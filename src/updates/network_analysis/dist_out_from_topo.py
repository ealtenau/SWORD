"""
Calculating Distance from Outlet based on topology
(dist_out_from_topo.py).
===============================================================

This script calculates distance from outlet based on SWORD
topology. 

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter region 
identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python path/to/dist_out_from_topo.py NA v17

"""

import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import netCDF4 as nc
import geopandas as gp
import pandas as pd
import argparse
import sys
import time
from itertools import chain
import matplotlib.pyplot as plt
from src.updates.sword import SWORD

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

#read data. 
sword = SWORD(main_dir, region, version)
sword.copy() #copies original file for version control.

#calculate dist out from topology. 
print('Calculating DistOut from Topology')
dist_out = np.repeat(-9999, len(sword.reaches.id)).astype(np.float64) #filler array for new outlet distance. 
flag = np.zeros(len(sword.reaches.id))
outlets = sword.reaches.id[np.where(sword.reaches.n_rch_down == 0)[0]]
start_rchs = np.array([outlets[0]]) #start with any outlet first. 
loop = 1
### While loop 
while len(start_rchs) > 0:
    #for loop to go through all start_rchs, which are the upstream neighbors of 
    #the previously updated reaches. The first reach is any outlet. 
    # print('LOOP:',loop, start_rchs)
    up_ngh_list = []
    for r in list(range(len(start_rchs))):
        rch = np.where(sword.reaches.id == start_rchs[r])[0]
        rch_flag = np.max(flag[rch])
        if sword.reaches.n_rch_down[rch] == 0:
            dist_out[rch] = sword.reaches.len[rch]
            up_nghs = sword.reaches.rch_id_up[:,rch]; up_nghs = up_nghs[up_nghs>0]
            up_flag = np.array([np.max(flag[np.where(sword.reaches.id == n)[0]]) for n in up_nghs])
            up_nghs = up_nghs[up_flag == 0]
            up_ngh_list.append(up_nghs)
            # loop=loop+1
        else:
            dn_nghs = sword.reaches.rch_id_down[:,rch]; dn_nghs = dn_nghs[dn_nghs>0]
            dn_dist = np.array([dist_out[np.where(sword.reaches.id == n)[0]][0] for n in dn_nghs])
            if min(dn_dist) == -9999:
                if rch_flag == 1:
                    # print(loop)
                    add_val = max(dn_dist)
                    dist_out[rch] = sword.reaches.len[rch]+add_val
                    up_nghs = sword.reaches.rch_id_up[:,rch]; up_nghs = up_nghs[up_nghs>0]
                    up_flag = np.array([np.max(flag[np.where(sword.reaches.id == n)[0]]) for n in up_nghs])
                    up_nghs = up_nghs[up_flag == 0]
                    flag[np.where(np.in1d(sword.reaches.id, up_nghs)==True)[0]] = 1
                else:
                    #set condition to start at next outlet. multichannel cases. 
                    flag[rch] = 1
            else:
                add_val = max(dn_dist)
                dist_out[rch] = sword.reaches.len[rch]+add_val
                up_nghs = sword.reaches.rch_id_up[:,rch]; up_nghs = up_nghs[up_nghs>0]
                up_flag = np.array([np.max(flag[np.where(sword.reaches.id == n)[0]]) for n in up_nghs])
                up_nghs = up_nghs[up_flag == 0]
                up_ngh_list.append(up_nghs) 
    #formatting next start reach.         
    up_ngh_arr = np.array(list(chain.from_iterable(up_ngh_list)))
    start_rchs = np.unique(up_ngh_arr)
    #if no more upstream neighbors move to next outlet. 
    if len(start_rchs) == 0:
        outlets = sword.reaches.id[np.where((sword.reaches.n_rch_down == 0) & (dist_out == -9999))[0]]
        #a case where all downstream reaches have filled but not all upstream.
        if len(outlets) == 0 and min(dist_out) > -9999:
            start_rchs = np.array([])
        elif len(outlets) == 0 and min(dist_out) == -9999:
            #find reach with downstream distances filled but a value of -9999
            check_flag = np.where((flag == 1) & (dist_out == -9999))[0]#added 7/12/25
            if len(check_flag) > 0:#added 7/12/25
                start_rchs = np.array([sword.reaches.id[check_flag[0]]]) #added 7/12/25
            else:
                print('!!! PROBLEM !!! --> No more upstream sword.reaches.id, but still -9999 values in outlet distance')
                break
        else:
            start_rchs = np.array([outlets[0]])
    loop = loop+1
    if loop > 5*len(sword.reaches.id):
        print('!!! LOOP STUCK !!!')
        break

# update node levels based on new dist out. 
print('Updating Nodes')
nodes_out = np.copy(sword.nodes.dist_out)
for r in list(range(len(sword.reaches.id))):
    nds = np.where(sword.nodes.reach_id == sword.reaches.id[r])[0]
    sort_nodes = np.argsort(sword.nodes.id[nds])
    base_val = dist_out[r] - sword.reaches.len[r]
    node_cs = np.cumsum(sword.nodes.len[nds[sort_nodes]])
    nodes_out[nds[sort_nodes]] = node_cs+base_val

print('Updating NetCDF')
sword.reaches.dist_out = dist_out
sword.nodes.dist_out = nodes_out
sword.save_nc()
print('Done')
