# -*- coding: utf-8 -*-
"""
Finding Problematic Nodes (find_problem_node_order_length.py)
===============================================================

This script goes through and finds incorrect node 
ordering or large/small node lengths 
(i.e. node length = 0 or node length > 1000) in the 
SWOT River Database (SWORD) to be updated in 
"fix_problem_node_order_lengths.py".

Two csv files containing 1) nodes with ordering problems
and 2) nodes with length problems are written to 
sword.paths['update_dir'].

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python path/to/find_problem_node_order_length.py NA v17
"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import pandas as pd
import time
import argparse
from src.updates.sword import SWORD

start_all = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="continental region", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version
 
sword = SWORD(main_dir, region, version)
out_dir = sword.paths['update_dir']

#get node numbers. 
cl_node_num_int = np.array([int(str(ind)[10:13]) for ind in sword.centerlines.node_id[0,:]])

#finding reaches with problematic node ordering. 
unq_rchs = np.unique(sword.reaches.id)
fixed_rchs = []
for r in list(range(len(unq_rchs))):
    print(r, unq_rchs[r], len(unq_rchs)-1)
    cl_r = np.where(sword.centerlines.reach_id[0,:] == unq_rchs[r])[0]
    order_ids = np.argsort(sword.centerlines.cl_id[cl_r])
    nodes_rch =  cl_node_num_int[cl_r[order_ids]]
    nodes_diff = np.abs(np.diff(nodes_rch))
    node_issues = np.where(nodes_diff > 1)[0]
    if len(node_issues) > 0:
        fixed_rchs.append(unq_rchs[r])

# find long node lengths
long_nodes = np.unique(sword.nodes.reach_id[np.where(sword.nodes.len > 1000)[0]])
# find zero node lengths 
zero_len = np.unique(sword.nodes.reach_id[np.where(sword.nodes.len == 0)[0]])
# combine length problems. 
len_issues = np.unique(np.append(long_nodes, zero_len))

#formatting and writing data. 
order_problems = {'reach_id': np.array(fixed_rchs).astype('int64')}
order_problems = pd.DataFrame(order_problems)
length_problems = {'reach_id': np.array(len_issues).astype('int64')}
length_problems = pd.DataFrame(length_problems)

order_problems.to_csv(out_dir+region.lower()+'_node_order_problems.csv', index=False)
length_problems.to_csv(out_dir+region.lower()+'_node_length_probems.csv', index=False)
print("order problems:", len(order_problems), ", length problems:", len(length_problems))
print('DONE')