# -*- coding: utf-8 -*-
'''
This script goes through and finds incorrect node ordering or large/small node 
lengths (i.e. node length = 0 or node length > 1000) to be updated in 
"fix_problem_node_order_lengths.py".

(c) E. Altenau 4/22/2025.
'''

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import pandas as pd
import time
import argparse
from scipy import stats as st
import src.updates.sword_utils as swd 

start_all = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="continental region", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("csv", help="csv file of reaches to delete", type = str)
args = parser.parse_args()

region = args.region
version = args.version

# region = 'OC'
# version = 'v18'
 
paths = swd.prepare_paths(main_dir, region, version)
sword_fn = paths['nc_dir']+paths['nc_fn']
out_dir = paths['update_dir']

#read data. 
centerlines, nodes, reaches = swd.read_nc(sword_fn)
cl_node_num_int = np.array([int(str(ind)[10:13]) for ind in centerlines.node_id[0,:]])

unq_rchs = np.unique(reaches.id)
fixed_rchs = []
for r in list(range(len(unq_rchs))):
    print(r, unq_rchs[r], len(unq_rchs)-1)
    cl_r = np.where(centerlines.reach_id[0,:] == unq_rchs[r])[0]
    order_ids = np.argsort(centerlines.cl_id[cl_r])
    nodes_rch =  cl_node_num_int[cl_r[order_ids]]
    nodes_diff = np.abs(np.diff(nodes_rch))
    node_issues = np.where(nodes_diff > 1)[0]
    if len(node_issues) > 0:
        fixed_rchs.append(unq_rchs[r])

# find long node lengths
long_nodes = np.unique(nodes.reach_id[np.where(nodes.len > 1000)[0]])
# find zero node lengths 
zero_len = np.unique(nodes.reach_id[np.where(nodes.len == 0)[0]])
# combine length problems. 
len_issues = np.unique(np.append(long_nodes, zero_len))

order_problems = {'reach_id': np.array(fixed_rchs).astype('int64')}
order_problems = pd.DataFrame(order_problems)
length_problems = {'reach_id': np.array(len_issues).astype('int64')}
length_problems = pd.DataFrame(length_problems)

order_problems.to_csv(out_dir+region.lower()+'_node_order_problems.csv', index=False)
length_problems.to_csv(out_dir+region.lower()+'_node_length_probems.csv', index=False)
print("order problems:", len(order_problems), ", length problems:", len(length_problems))
print('DONE')