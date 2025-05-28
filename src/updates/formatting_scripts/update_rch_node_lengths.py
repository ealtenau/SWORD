# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import argparse
import src.updates.sword_utils as swd
import src.updates.calc_utils as ct

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

# region = 'SA'
# version = 'v17b'

paths = swd.prepare_paths(main_dir, region, version)
sword_fn = paths['nc_dir']+paths['nc_fn']

#read data.
centerlines, nodes, reaches = swd.read_nc(sword_fn)

for r in list(range(len(reaches))):
    print(r, len(reaches)-1)
    rch = np.where(centerlines.reach_id[0,:] == reaches.id[r])[0] #if multiple choose first.
    sort_ind = rch[np.argsort(centerlines.cl_id[rch])] 
    x_coords = centerlines.x[sort_ind]
    y_coords = centerlines.y[sort_ind]
    diff = ct.get_distances(x_coords,y_coords)
    rch_dist = np.cumsum(diff)
    reaches.len[r] = max(rch_dist)
    #nodes     
    unq_nodes = np.unique(centerlines.node_id[0,sort_ind])
    for n in list(range(len(unq_nodes))):
        nds = np.where(centerlines.node_id[0,sort_ind] == unq_nodes[n])[0]
        nind = np.where(nodes.id == unq_nodes[n])[0]
        nodes.len[nind] = max(np.cumsum(diff[nds]))

#write data.
swd.discharge_attr_nc(reaches)
swd.write_nc(centerlines, nodes, reaches, sword_fn)

#check lengths for random reaches. 
import random
rand = random.sample(range(0,len(reaches.id)), 1000)
for ind in list(range(len(rand))):
    test = np.where(nodes.reach_id == reaches.id[rand[ind]])[0]
    print(reaches.id[rand[ind]], 
          abs(np.round(sum(nodes.len[test])-reaches.len[rand[ind]]))) 
        #   abs(np.round(max(node_dist[test])-rch_dist[rand[ind]])))

print('DONE')
