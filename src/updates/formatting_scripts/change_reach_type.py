# -*- coding: utf-8 -*-
"""
Changing Reach Type (change_reach_type.py)
========================================

This script updates the Reach and Node ID type 
identifiers in the SWOT River Database (SWORD).

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA), SWORD version (i.e. v17), 
and a csv file containing the Reach IDs that need a 
type change.

Execution example (terminal):
    python path/to/change_reach_type.py NA v17 path/to/type_change.csv

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import pandas as pd
import numpy as np
import argparse
from src.updates.sword import SWORD

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("csv", help="csv file of reaches with type updates", type = str)
args = parser.parse_args()

region = args.region
version = args.version

#read data. 
sword = SWORD(main_dir, region, version)
csv_dir = args.csv #sword.paths['nc_dir']+region.lower()+'_incorrect_ghost_reaches.csv'
updates = pd.read_csv(csv_dir)

#loop to change ids with new type. 
for row in list(range(len(updates))):
    print(row, len(updates)-1)
    rch = np.where(sword.reaches.id == updates['reach_id'][row])[0]
    if len(rch) == 0:
        continue
    
    nodes_rch = np.where(sword.nodes.reach_id[:] == updates['reach_id'][row])[0]
    if sword.reaches.edit_flag[rch] == 'NaN':
        edit_val = '1'
    elif '1' not in str(sword.reaches.edit_flag[rch]).split(','):
        edit_val = str(sword.reaches.edit_flag[rch])[2:-2] + ',1'
    else:
        edit_val = str(sword.reaches.edit_flag[rch])[2:-2]
                    
    rch_up1 = np.where(sword.reaches.rch_id_up[0,:] == updates['reach_id'][row])[0]
    rch_up2 = np.where(sword.reaches.rch_id_up[1,:] == updates['reach_id'][row])[0]
    rch_up3 = np.where(sword.reaches.rch_id_up[2,:] == updates['reach_id'][row])[0]
    rch_up4 = np.where(sword.reaches.rch_id_up[3,:] == updates['reach_id'][row])[0]
    rch_dn1 = np.where(sword.reaches.rch_id_down[0,:] == updates['reach_id'][row])[0]
    rch_dn2 = np.where(sword.reaches.rch_id_down[1,:] == updates['reach_id'][row])[0]
    rch_dn3 = np.where(sword.reaches.rch_id_down[2,:] == updates['reach_id'][row])[0]
    rch_dn4 = np.where(sword.reaches.rch_id_down[3,:] == updates['reach_id'][row])[0]
    cl_rch1 = np.where(sword.centerlines.reach_id[0,:] == updates['reach_id'][row])[0]
    cl_rch2 = np.where(sword.centerlines.reach_id[1,:] == updates['reach_id'][row])[0]
    cl_rch3 = np.where(sword.centerlines.reach_id[2,:] == updates['reach_id'][row])[0]
    cl_rch4 = np.where(sword.centerlines.reach_id[3,:] == updates['reach_id'][row])[0]
    #create new ids with new type
    rch_id = str(int(sword.reaches.id[rch]))
    node_ids = [str(int(n)) for n in sword.nodes.id[nodes_rch]]
    rch_basin = rch_id[0:-1]
    node_basin = [n[0:-1] for n in node_ids]
    new_rch_id = int(rch_basin + str(updates['new_type'][row]))
    new_node_ids = [int(n + str(updates['new_type'][row])) for n in node_basin]
    #update reach id variables
    sword.reaches.id[rch] = new_rch_id
    sword.reaches.edit_flag[rch] = edit_val
    sword.nodes.reach_id[nodes_rch] = new_rch_id
    sword.nodes.edit_flag[nodes_rch] = np.repeat(edit_val, len(nodes_rch))
    if len(cl_rch1) > 0:
        sword.centerlines.reach_id[0,cl_rch1] = new_rch_id
    if len(rch_up1) > 0:
        sword.reaches.rch_id_up[0,rch_up1] = new_rch_id
    if len(rch_dn1) > 0:
        sword.reaches.rch_id_down[0,rch_dn1] = new_rch_id
    if len(cl_rch2) > 0:
        sword.centerlines.reach_id[1,cl_rch2] = new_rch_id
    if len(rch_up2) > 0:
        sword.reaches.rch_id_up[1,rch_up2] = new_rch_id
    if len(rch_dn2) > 0:
        sword.reaches.rch_id_down[1,rch_dn2] = new_rch_id
    if len(cl_rch3) > 0:
        sword.centerlines.reach_id[2,cl_rch3] = new_rch_id
    if len(rch_up3) > 0:
        sword.reaches.rch_id_up[2,rch_up3] = new_rch_id
    if len(rch_dn3) > 0:
        sword.reaches.rch_id_down[2,rch_dn3] = new_rch_id
    if len(cl_rch4) > 0:
        sword.centerlines.reach_id[3,cl_rch4] = new_rch_id
    if len(rch_up4) > 0:
        sword.reaches.rch_id_up[3,rch_up4] = new_rch_id
    if len(rch_dn4) > 0:
        sword.reaches.rch_id_down[3,rch_dn4] = new_rch_id
    #update node id variables
    for n in list(range(len(node_ids))):
        sword.nodes.id[nodes_rch[n]] = new_node_ids[n]
        cl_n1 = np.where(sword.centerlines.node_id[0,:] == int(node_ids[n]))[0]
        cl_n2 = np.where(sword.centerlines.node_id[1,:] == int(node_ids[n]))[0]
        cl_n3 = np.where(sword.centerlines.node_id[2,:] == int(node_ids[n]))[0]
        cl_n4 = np.where(sword.centerlines.node_id[3,:] == int(node_ids[n]))[0]
        #update centerline node variables.
        if len(cl_n1) > 0:
            sword.centerlines.node_id[0,cl_n1] = new_node_ids[n]
        if len(cl_n2) > 0:
            sword.centerlines.node_id[1,cl_n2] = new_node_ids[n]
        if len(cl_n3) > 0:
            sword.centerlines.node_id[2,cl_n3] = new_node_ids[n]
        if len(cl_n4) > 0:
            sword.centerlines.node_id[3,cl_n4] = new_node_ids[n]

### Write data
sword.save_nc()

print('Cl Dimensions:', len(np.unique(sword.centerlines.cl_id)), len(sword.centerlines.cl_id))
print('Node Dimensions:', len(np.unique(sword.centerlines.node_id[0,:])), len(sword.nodes.id), len(sword.nodes.id))
print('Rch Dimensions:', len(np.unique(sword.centerlines.reach_id[0,:])), len(np.unique(sword.nodes.reach_id)), len(np.unique(sword.reaches.id)),len(sword.reaches.id))
print('Edit Flag Values:', np.unique(sword.reaches.edit_flag))
print('UPDATES DONE')