# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import argparse
from src.updates.sword import SWORD
import src.updates.aux_utils as aux

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

# region = 'SA'
# version = 'v17b'

#read data
sword = SWORD(main_dir, region, version)

for r in list(range(len(sword.reaches.id))):
    print(r, len(sword.reaches.id)-1)
    rch = np.where(sword.centerlines.reach_id[0,:] == sword.reaches.id[r])[0] #if multiple choose first.
    sort_ind = rch[np.argsort(sword.centerlines.cl_id[rch])] 
    x_coords = sword.centerlines.x[sort_ind]
    y_coords = sword.centerlines.y[sort_ind]
    diff = aux.get_distances(x_coords,y_coords)
    rch_dist = np.cumsum(diff)
    sword.reaches.len[r] = max(rch_dist)
    #nodes     
    unq_nodes = np.unique(sword.centerlines.node_id[0,sort_ind])
    for n in list(range(len(unq_nodes))):
        nds = np.where(sword.centerlines.node_id[0,sort_ind] == unq_nodes[n])[0]
        nind = np.where(sword.nodes.id == unq_nodes[n])[0]
        sword.nodes.len[nind] = max(np.cumsum(diff[nds]))

#write data.
sword.save_nc()

#check lengths for random sword.reaches. 
import random
rand = random.sample(range(0,len(sword.reaches.id)), 1000)
for ind in list(range(len(rand))):
    test = np.where(sword.nodes.reach_id == sword.reaches.id[rand[ind]])[0]
    print(sword.reaches.id[rand[ind]], 
          abs(np.round(sum(sword.nodes.len[test])-sword.reaches.len[rand[ind]]))) 
        #   abs(np.round(max(node_dist[test])-rch_dist[rand[ind]])))

print('DONE')
