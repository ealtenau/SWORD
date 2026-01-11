# -*- coding: utf-8 -*-
"""
Checking SWORD Dimensions (check_nc_dimensions.py)
=====================
Script for making sure all the identification numbers 
for centerlines, reaches, and nodes are unqiue and consistent
across all dimensions of the SWOT River Database (SWORD). 

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python path/to/check_nc_dimensions.py NA v17 

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

#reading data
db_path = os.path.join(main_dir, f'data/duckdb/sword_{version}.duckdb')
sword = SWORD(db_path, region, version)

#checking dimensions
print('Cl Dimensions:', len(np.unique(sword.centerlines.cl_id)), len(sword.centerlines.cl_id))
print('Node Dimensions:', len(np.unique(sword.centerlines.node_id[0,:])), len(np.unique(sword.nodes.id)), len(sword.nodes.id))
print('Rch Dimensions:', len(np.unique(sword.centerlines.reach_id[0,:])), len(np.unique(sword.nodes.reach_id)), len(np.unique(sword.reaches.id)), len(sword.reaches.id))
print('min node char len:', len(str(np.min(sword.nodes.id)))) #should be 14
print('max node char len:', len(str(np.max(sword.nodes.id)))) #should be 14
print('min reach char len:', len(str(np.min(sword.reaches.id)))) #should be 11
print('max reach char len:', len(str(np.max(sword.reaches.id)))) #should be 11
print('Edit flag values:', np.unique(sword.reaches.edit_flag))
