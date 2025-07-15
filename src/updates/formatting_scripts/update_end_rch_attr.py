"""
Updating End Reach Attribute after centerline additions.
(update_end_rch_attr.py)
===============================================================

This scripts updates the end reach attribute based on new
centerline additions in SWORD.  

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python path/to/update_end_rch_attr.py NA v17

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import argparse
from src.updates.sword import SWORD
import src.updates.geo_utils as geo

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

#read data.
sword = SWORD(main_dir, region, version)
sword.copy() #copies original file for version control.

print('Updating End Reach Variable')
sword.reaches.end_rch = np.zeros(len(sword.reaches.end_rch))
sword.nodes.end_rch = np.zeros(len(sword.nodes.end_rch))
hw = np.where(sword.reaches.n_rch_up == 0)[0]
ot = np.where(sword.reaches.n_rch_down == 0)[0]
junc1 = np.where(sword.reaches.n_rch_up > 1)[0]
junc2 = np.where(sword.reaches.n_rch_down > 1)[0]
node_hw = np.where(np.in1d(sword.nodes.reach_id, sword.reaches.id[hw]) == True)[0]
node_ot = np.where(np.in1d(sword.nodes.reach_id, sword.reaches.id[ot]) == True)[0]
node_junc1 = np.where(np.in1d(sword.nodes.reach_id, sword.reaches.id[junc1]) == True)[0]
node_junc2 = np.where(np.in1d(sword.nodes.reach_id, sword.reaches.id[junc2]) == True)[0]
sword.reaches.end_rch[junc1] = 3
sword.reaches.end_rch[junc1] = 3
sword.reaches.end_rch[hw] = 1
sword.reaches.end_rch[ot] = 2
sword.nodes.end_rch[node_junc1] = 3
sword.nodes.end_rch[node_junc1] = 3
sword.nodes.end_rch[node_hw] = 1
sword.nodes.end_rch[node_ot] = 2

#write data.
sword.save_nc()