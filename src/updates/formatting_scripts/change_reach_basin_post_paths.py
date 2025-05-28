# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import pandas as pd
import argparse
import src.updates.sword_utils as swd 

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("csv", help="csv file of reaches for basin id change", type = str)
args = parser.parse_args()

region = args.region
version = args.version

### Line-by-line degugging.
# region = 'EU'
# version = 'v17'

paths = swd.prepare_paths(main_dir, region, version)
nc_fn = paths['nc_dir']+paths['nc_fn']
rch_list = args.csv

centerlines, nodes, reaches = swd.read_nc(nc_fn)
df = pd.read_csv(rch_list)

for r in list(range(len(df))):
    print(r, len(df)-1)
    rch = int(df['reach_id'][r])
    rch_ind = np.where(reaches.id == rch)[0]
    nds_ind = np.where(nodes.reach_id == rch)[0]
    cl_ind = np.where(centerlines.reach_id[0,:] == rch)
    
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
    node_nums = np.array([int(str(ns)[11:13]) for ns in nodes.id[nds_ind]])
    unq_nodes = nodes.id[nds_ind]
    for nds in list(range(len(unq_nodes))):
        nd_ind = np.where(nodes.id == unq_nodes[nds])[0]
        cl_nd_ind = np.where(centerlines.node_id[0,:] == unq_nodes[nds])
        if len(str(node_nums[nds])) == 1:
            fill = '00'
            new_node_id = int(str(new_rch_id)[0:-1]+fill+str(node_nums[nds])+str(new_rch_id)[-1])
        if len(str(node_nums[nds])) == 2:
            fill = '0'
            new_node_id = int(str(new_rch_id)[0:-1]+fill+str(node_nums[nds])+str(new_rch_id)[-1])
        if len(str(node_nums[nds])) == 3:
            new_node_id = int(str(new_rch_id)[0:-1]+fill+str(node_nums[nds])+str(new_rch_id)[-1])
        #update node level variables. 
        centerlines.node_id[0,cl_nd_ind] = new_node_id
        nodes.id[nd_ind] = new_node_id
    
    #update reach level variables. 
    reaches.id[rch_ind] = new_rch_id
    nodes.reach_id[nds_ind] = new_rch_id
    centerlines.reach_id[0,cl_ind] = new_rch_id

### Filler variables
swd.discharge_attr_nc(reaches)
### Write data
swd.write_nc(centerlines, reaches, nodes, region, nc_fn)

print('Cl Dimensions:', len(np.unique(centerlines.cl_id)), len(centerlines.cl_id))
print('Rch Dimensions:', len(np.unique(centerlines.reach_id[0,:])), len(np.unique(nodes.reach_id)), len(reaches.id))
print('Node Dimensions:', len(np.unique(centerlines.node_id[0,:])), len(np.unique(nodes.id)), len(nodes.id))

