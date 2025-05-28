# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import netCDF4 as nc
import geopandas as gp
from geopy import distance
import pandas as pd
import random
import argparse
import src.updates.sword_utils as swd

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

paths = swd.prepare_paths(main_dir, region, version)
sword_fn = paths['geom_dir']+paths['geom_fn']

#read data.
centerlines, nodes, reaches = swd.read_nc(sword_fn)

nlen_diff = np.zeros(len(reaches.id))
do_diff = np.zeros(len(reaches.id))
for ind in list(range(len(reaches.id))):
    test = np.where(nodes.reach_id == reaches.id[ind])[0]
    nlen_diff[ind] = np.abs(np.round(sum(nodes.len[test])-reaches.len[ind]))
    do_diff[ind] = np.abs(np.round(max(nodes.dist_out[test])-reaches.dist_out[ind]))

len_diff_perc = len(np.where(nlen_diff != 0)[0])/len(reaches.id)*100
do_diff_perc = len(np.where(do_diff != 0)[0])/len(reaches.id)*100

print('Percent Length Differences:', np.round(len_diff_perc, 2), ", Max Diff:", np.max(nlen_diff))
print('Percent DistOut Differences:', np.round(do_diff_perc,2), ", Max Diff:", np.max(do_diff))
print('DONE')

# rand = random.sample(range(0,len(reaches)), 1000)
# for ind in list(range(len(rand))):
#     test = np.where(node_rch == reaches[rand[ind]])[0]
#     print(reaches[rand[ind]], 
#           abs(np.round(sum(node_len[test])-rch_len[rand[ind]])), 
#           abs(np.round(max(node_dist[test])-rch_dist[rand[ind]])))
# print('DONE')

