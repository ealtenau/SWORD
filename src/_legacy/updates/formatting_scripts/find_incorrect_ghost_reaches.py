# -*- coding: utf-8 -*-
"""
Finding Missing and Incorrectly Labeled 
Ghost Reaches (find_incorrect_ghost_reaches.py)
===================================================

This script identifies reaches in the SWOT River
Database (SWORD) that are mislabeled as ghost 
reaches or should be labeled a ghost reach.

Two csv files containing 1) reaches that should 
be a ghost reach and 2) mislabeled ghost 
reaches are written to sword.paths['update_dir'].

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python path/to/find_incorrect_ghost_reaches.py NA v17

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import pandas as pd
import numpy as np
import argparse
from src.updates.sword_duckdb import SWORD

parser = argparse.ArgumentParser()
parser.add_argument("region", help="continental region", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

#read data.
db_path = os.path.join(main_dir, f'data/duckdb/sword_{version}.duckdb')
sword = SWORD(db_path, region, version)
out_dir = sword.paths['update_dir']

#finding missing ghost reaches. 
sword.reaches.type = np.array([int(str(rch)[-1]) for rch in sword.reaches.id])
missing_ghost_headwater = np.where((sword.reaches.n_rch_up == 0)&(sword.reaches.type < 6))[0]
missing_ghost_outlet = np.where((sword.reaches.n_rch_down == 0)&(sword.reaches.type < 6))[0]
all_missing = np.append(missing_ghost_headwater,missing_ghost_outlet)

#creating a label for the csv output identifying whether 
#the missing ghost reach was a headwater or outlet reach. 
hw_end = np.repeat(1,len(missing_ghost_headwater))
out_end = np.repeat(2,len(missing_ghost_outlet))
all_ends = np.append(hw_end,out_end)

#finding incorrect ghost reaches. 
correct = np.where((sword.reaches.n_rch_up > 0)&(sword.reaches.n_rch_down > 0)&(sword.reaches.type == 6))[0]

#determining the proper reach type for the incorrect 
#ghost reaches. 
subreaches = sword.reaches.id[correct]
new_type = []
for r in list(range(len(subreaches))):
    # print(r)
    rch = np.where(sword.reaches.id == subreaches[r])[0]
    up_type = sword.reaches.type[np.where(np.in1d(sword.reaches.id, sword.reaches.rch_id_up[:,rch])==True)[0]]
    dn_type = sword.reaches.type[np.where(np.in1d(sword.reaches.id, sword.reaches.rch_id_down[:,rch])==True)[0]]
    all_types = np.append(up_type,dn_type)
    new_type.append(max(all_types[np.where(all_types<6)[0]]))

#formatting and writing data. 
ghost = {'reach_id': np.array(sword.reaches.id[correct]).astype('int64'), 'new_type': np.array(new_type).astype('int64')}
ghost = pd.DataFrame(ghost)
ends = {'reach_id': np.array(sword.reaches.id[all_missing]).astype('int64'), 'hw_out': np.array(all_ends).astype('int64')}
ends = pd.DataFrame(ends)

ghost.to_csv(out_dir+region.lower()+'_incorrect_ghost_reaches.csv', index=False)
ends.to_csv(out_dir+region.lower()+'_missing_ghost_reaches.csv', index=False)
print("incorrect ghost:", len(ghost), ", missing ghost:", len(ends))
print('DONE')