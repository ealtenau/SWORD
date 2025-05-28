# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import pandas as pd
import numpy as np
import argparse
import src.updates.sword_utils as swd 

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

centerlines, nodes, reaches = swd.read_nc(sword_fn)

reaches.type = np.array([int(str(rch)[-1]) for rch in reaches.id])
correct = np.where((reaches.n_rch_up > 0)&(reaches.n_rch_down > 0)&(reaches.type == 6))[0]
missing_ghost_headwater = np.where((reaches.n_rch_up == 0)&(reaches.type < 6))[0]
missing_ghost_outlet = np.where((reaches.n_rch_down == 0)&(reaches.type < 6))[0]
all_missing = np.append(missing_ghost_headwater,missing_ghost_outlet)

hw_end = np.repeat(1,len(missing_ghost_headwater))
out_end = np.repeat(2,len(missing_ghost_outlet))
all_ends = np.append(hw_end,out_end)

subreaches = reaches.id[correct]
new_type = []
for r in list(range(len(subreaches))):
    # print(r)
    rch = np.where(reaches.id == subreaches[r])[0]
    up_type = reaches.type[np.where(np.in1d(reaches.id, reaches.rch_id_up[:,rch])==True)[0]]
    dn_type = reaches.type[np.where(np.in1d(reaches.id, reaches.rch_id_down[:,rch])==True)[0]]
    all_types = np.append(up_type,dn_type)
    new_type.append(max(all_types[np.where(all_types<6)[0]]))

ghost = {'reach_id': np.array(reaches.id[correct]).astype('int64'), 'new_type': np.array(new_type).astype('int64')}
ghost = pd.DataFrame(ghost)
ends = {'reach_id': np.array(reaches.id[all_missing]).astype('int64'), 'hw_out': np.array(all_ends).astype('int64')}
ends = pd.DataFrame(ends)

ghost.to_csv(out_dir+region.lower()+'_incorrect_ghost_reaches.csv', index=False)
ends.to_csv(out_dir+region.lower()+'_missing_ghost_reaches.csv', index=False)
print("incorrect ghost:", len(ghost), ", missing ghost:", len(ends))
print('DONE')