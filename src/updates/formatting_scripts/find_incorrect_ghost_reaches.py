# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import pandas as pd
import numpy as np
import argparse
from src.updates.sword import SWORD

parser = argparse.ArgumentParser()
parser.add_argument("region", help="continental region", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("csv", help="csv file of reaches to delete", type = str)
args = parser.parse_args()

region = args.region
version = args.version

# region = 'OC'
# version = 'v18'

sword = SWORD(main_dir, region, version)
out_dir = sword.paths['update_dir']

sword.reaches.type = np.array([int(str(rch)[-1]) for rch in sword.reaches.id])
correct = np.where((sword.reaches.n_rch_up > 0)&(sword.reaches.n_rch_down > 0)&(sword.reaches.type == 6))[0]
missing_ghost_headwater = np.where((sword.reaches.n_rch_up == 0)&(sword.reaches.type < 6))[0]
missing_ghost_outlet = np.where((sword.reaches.n_rch_down == 0)&(sword.reaches.type < 6))[0]
all_missing = np.append(missing_ghost_headwater,missing_ghost_outlet)

hw_end = np.repeat(1,len(missing_ghost_headwater))
out_end = np.repeat(2,len(missing_ghost_outlet))
all_ends = np.append(hw_end,out_end)

subreaches = sword.reaches.id[correct]
new_type = []
for r in list(range(len(subreaches))):
    # print(r)
    rch = np.where(sword.reaches.id == subreaches[r])[0]
    up_type = sword.reaches.type[np.where(np.in1d(sword.reaches.id, sword.reaches.rch_id_up[:,rch])==True)[0]]
    dn_type = sword.reaches.type[np.where(np.in1d(sword.reaches.id, sword.reaches.rch_id_down[:,rch])==True)[0]]
    all_types = np.append(up_type,dn_type)
    new_type.append(max(all_types[np.where(all_types<6)[0]]))

ghost = {'reach_id': np.array(sword.reaches.id[correct]).astype('int64'), 'new_type': np.array(new_type).astype('int64')}
ghost = pd.DataFrame(ghost)
ends = {'reach_id': np.array(sword.reaches.id[all_missing]).astype('int64'), 'hw_out': np.array(all_ends).astype('int64')}
ends = pd.DataFrame(ends)

ghost.to_csv(out_dir+region.lower()+'_incorrect_ghost_sword.reaches.csv', index=False)
ends.to_csv(out_dir+region.lower()+'_missing_ghost_sword.reaches.csv', index=False)
print("incorrect ghost:", len(ghost), ", missing ghost:", len(ends))
print('DONE')