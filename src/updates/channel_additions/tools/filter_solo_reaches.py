# -*- coding: utf-8 -*-
"""
Identify Solo Reaches in SWORD
(filter_solo_reaches.py).
===================================================

Identifies and flags single reaches in SWORD with 
no upstream or downstream neighbors.

Flagged reaches are saved as a csv file located at:
'/data/update_requests/'+version+'/'+region+'/'

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python path/to/filter_solo_reaches.py NA v17

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import pandas as pd
import numpy as np
import netCDF4 as nc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

#data paths. 
nc_fn = main_dir+'/data/outputs/Reaches_Nodes/'+version+'/netcdf/'\
    +region.lower()+'_sword_'+version+'.nc'
outdir = main_dir+'/data/update_requests/'+version+'/'+region+'/'

#read sword. 
sword = nc.Dataset(nc_fn)
n_rch_up = np.array(sword['/reaches/n_rch_up'][:])
n_rch_down = np.array(sword['/reaches/n_rch_down'][:])
rch_len = np.array(sword['/reaches/reach_length'][:])
rch_id = np.array(sword['/reaches/reach_id'][:])

#find solo reaches. 
solo = np.where((n_rch_up == 0)&(n_rch_down == 0))[0]
rmv = rch_id[solo]

#save identified reaches. 
rch_csv = pd.DataFrame({"reach_id": rmv})
rch_csv.to_csv(outdir+'solo_rch_deletions.csv', index = False)
print('Done')
