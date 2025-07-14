# -*- coding: utf-8 -*-
"""
Changing Basins (change_reach_basin_post_paths.py)
==============================================

This script updates the Pfafstetter Level 6 basin codes
in the SWOT River Database (SWORD).

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA), SWORD version (i.e. v17), 
and a csv file containing Reach IDs that need new basin
codes.

Execution example (terminal):
    python change_reach_basin_post_paths.py NA v17 path/to/basin.csv

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import pandas as pd
import argparse
from src.updates.sword import SWORD

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("csv", help="csv file of reaches for basin id change", type = str)
args = parser.parse_args()

region = args.region
version = args.version

#read sword and csv data.
sword = SWORD(main_dir, region, version)
sword.copy() #copies original file for version control.

rch_list = args.csv
df = pd.read_csv(rch_list)

#loop to update basins for specified reaches. 
for r in list(range(len(df))):
    print(r, len(df)-1)
    rch = int(df['reach_id'][r])
    rch_ind = np.where(sword.reaches.id == rch)[0]
    nds_ind = np.where(sword.nodes.reach_id == rch)[0]
    cl_ind = np.where(sword.centerlines.reach_id[0,:] == rch)
    
    # get new id variables. 
    b = int(df['new_basin'][r])
    n = int(df['new_num'][r])
    t = int(df['category'][r])
    if len(str(n)) == 1:
        fill = '000'
        new_rch_id = int(str(b)+fill+str(n)+str(t))
    if len(str(n)) == 2:
        fill = '00'
        new_rch_id = int(str(b)+fill+str(n)+str(t))
    if len(str(n)) == 3:
        fill = '0'
        new_rch_id = int(str(b)+fill+str(n)+str(t))
    if len(str(n)) == 4:
        new_rch_id = int(str(b)+fill+str(n)+str(t))
    
    #get node numbers for new ids and loop through to update. 
    node_nums = np.array([int(str(ns)[11:13]) for ns in sword.nodes.id[nds_ind]])
    unq_nodes = sword.nodes.id[nds_ind]
    for nds in list(range(len(unq_nodes))):
        nd_ind = np.where(sword.nodes.id == unq_nodes[nds])[0]
        cl_nd_ind = np.where(sword.centerlines.node_id[0,:] == unq_nodes[nds])
        if len(str(node_nums[nds])) == 1:
            fill = '00'
            new_node_id = int(str(new_rch_id)[0:-1]+fill+str(node_nums[nds])+str(new_rch_id)[-1])
        if len(str(node_nums[nds])) == 2:
            fill = '0'
            new_node_id = int(str(new_rch_id)[0:-1]+fill+str(node_nums[nds])+str(new_rch_id)[-1])
        if len(str(node_nums[nds])) == 3:
            new_node_id = int(str(new_rch_id)[0:-1]+fill+str(node_nums[nds])+str(new_rch_id)[-1])
        #update node level variables. 
        sword.centerlines.node_id[0,cl_nd_ind] = new_node_id
        sword.nodes.id[nd_ind] = new_node_id
    
    #update reach level variables. 
    sword.reaches.id[rch_ind] = new_rch_id
    sword.nodes.reach_id[nds_ind] = new_rch_id
    sword.centerlines.reach_id[0,cl_ind] = new_rch_id

### Write data
sword.save_nc()

print('Cl Dimensions:', len(np.unique(sword.centerlines.cl_id)), len(sword.centerlines.cl_id))
print('Rch Dimensions:', len(np.unique(sword.centerlines.reach_id[0,:])), len(np.unique(sword.nodes.reach_id)), len(sword.reaches.id))
print('Node Dimensions:', len(np.unique(sword.centerlines.node_id[0,:])), len(np.unique(sword.nodes.id)), len(sword.nodes.id))

