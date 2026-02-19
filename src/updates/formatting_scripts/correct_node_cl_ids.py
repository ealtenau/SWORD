# -*- coding: utf-8 -*-
"""
Correcting Node cl_ids (correct_node_cl_ids.py)
===================================================

This script corrects min/max cl_ids in the SWORD 
node dimension if they don't match the centerline 
dimension.

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python path/to/correct_node_cl_ids.py NA v17

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import argparse
from src.updates.sword_duckdb import SWORD

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

#read sword data.
db_path = os.path.join(main_dir, f'data/duckdb/sword_{version}.duckdb')
sword = SWORD(db_path, region, version)
sword.copy() #copies original file for version control.

#look for incorrect node cl_ids in a reach. 
issues = []
for r in list(range(len(sword.reaches.id))): #r = 10 for testing. 
    print(r, len(sword.reaches.id)-1)
    #finding centerline points associated with a reach and sorting by index. 
    cind = np.where(sword.centerlines.reach_id[0,:] == sword.reaches.id[r])[0]
    sort_ids = cind[np.argsort(sword.centerlines.cl_id[cind])]
    #isolating the first node id along the sorted centerline points. 
    test_node = sword.centerlines.node_id[0,sort_ids][0]
    #finding what centerline ids are listed in the node dimension. 
    nind = np.where(sword.nodes.id[:] == test_node)[0]
    check_id = np.where(sword.centerlines.cl_id[sort_ids] == sword.nodes.cl_id[0,nind])[0]
    check_node = sword.centerlines.node_id[0,sort_ids[check_id]]
    #if the first centerline id in the node dimension does not match the centerline dimension -> fix. 
    if check_node != test_node:
        issues.append(sword.reaches.id[r])
        unq_nodes = np.unique(sword.centerlines.node_id[0,sort_ids])
        #re-calculating the min and max centerline ids for each node. 
        for n in list(range(len(unq_nodes))):
            cl_idx = np.where(sword.centerlines.node_id[0,sort_ids] == unq_nodes[n])[0]
            nd_idx = np.where(sword.nodes.id == unq_nodes[n])[0]
            mn = np.min(sword.centerlines.cl_id[sort_ids[cl_idx]])
            mx = np.max(sword.centerlines.cl_id[sort_ids[cl_idx]])
            sword.nodes.cl_id[0,nd_idx] = mn
            sword.nodes.cl_id[0,nd_idx] = mx

if len(issues) > 0:
    sword.save_nc()
    #print issues. 
    print('Issues Found:', str(len(issues))+',', 
        str(np.round(len(issues)/len(sword.reaches.id)*100,2))+'%')
else:
    print('No Issues Found.')
