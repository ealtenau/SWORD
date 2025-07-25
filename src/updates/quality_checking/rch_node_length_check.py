# -*- coding: utf-8 -*-
"""
Checking SWORD Topology (check_topo_consistency.py)
=====================
Script for checking the node and reach lengths
match in the SWOT River Database (SWORD).

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python path/to/rch_node_length_check.py NA v17 

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import argparse
from src.updates.sword import SWORD

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

#read data.
sword = SWORD(main_dir, region, version)

#checking node lengths against reach lengths. 
nlen_diff = np.zeros(len(sword.reaches.id))
do_diff = np.zeros(len(sword.reaches.id))
for ind in list(range(len(sword.reaches.id))):
    test = np.where(sword.nodes.reach_id == sword.reaches.id[ind])[0]
    nlen_diff[ind] = np.abs(np.round(sum(sword.nodes.len[test])-sword.reaches.len[ind]))
    do_diff[ind] = np.abs(np.round(max(sword.nodes.dist_out[test])-sword.reaches.dist_out[ind]))

len_diff_perc = len(np.where(nlen_diff != 0)[0])/len(sword.reaches.id)*100
do_diff_perc = len(np.where(do_diff != 0)[0])/len(sword.reaches.id)*100

print('Percent Length Differences:', np.round(len_diff_perc, 2), ", Max Diff:", np.max(nlen_diff))
print('Percent DistOut Differences:', np.round(do_diff_perc,2), ", Max Diff:", np.max(do_diff))
print('DONE')

### checks a random sample of 1000 sword reaches. 
# rand = random.sample(range(0,len(sword.reaches.id)), 1000)
# for ind in list(range(len(rand))):
#     test = np.where(sword.nodes.reach_id == sword.reaches.id[rand[ind]])[0]
#     print(reaches[rand[ind]], 
#           abs(np.round(sum(sword.nodes.len[test])-sword.reaches.len[rand[ind]])), 
#           abs(np.round(max(sword.nodes.dist_out[test])-sword.reaches.dist_out[rand[ind]])))
# print('DONE')

