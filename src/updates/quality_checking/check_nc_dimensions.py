# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import argparse
import src.updates.sword_utils as swd 

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

# region = 'OC'
# version = 'v17b'

#reading data
paths = swd.prepare_paths(main_dir, region, version)
sword_fn = paths['nc_dir']+paths['nc_fn']
centerlines, nodes, reaches = swd.read_nc(sword_fn)

#checking dimensions
print('Cl Dimensions:', len(np.unique(centerlines.cl_id)), len(centerlines.cl_id))
print('Node Dimensions:', len(np.unique(centerlines.node_id[0,:])), len(np.unique(nodes.id)), len(nodes.id))
print('Rch Dimensions:', len(np.unique(centerlines.reach_id[0,:])), len(np.unique(nodes.reach_id)), len(np.unique(reaches.id)), len(reaches.id))
print('min node char len:', len(str(np.min(nodes.id))))
print('max node char len:', len(str(np.max(nodes.id))))
print('min reach char len:', len(str(np.min(reaches.id))))
print('max reach char len:', len(str(np.max(reaches.id))))
print('Edit flag values:', np.unique(reaches.edit_flag))
