# -*- coding: utf-8 -*-
"""
Checking for SWORD Network Breaks
(check_for_addition_breaks.py)
===================================================

This script checks for SWORD reaches that have no 
downstream neighbors but are not ghost reaches.
This indicates a possible topology break in the network. 

Flagged reaches are saved as a csv file located at:
'/data/update_requests/'+version+'/'+region+'/'

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python path/to/check_for_addition_breaks.py NA v17

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import netCDF4 as nc
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

#reading in sword data.
sword_fn = main_dir+'/data/outputs/Reaches_Nodes/'+version+'/netcdf/'\
    +region.lower()+'_sword_'+version+'.nc'
outdir = main_dir+'/data/update_requests/'+version+'/'+region+'/'
sword = nc.Dataset(sword_fn)

reaches = np.array(sword['/reaches/reach_id'][:])
n_rch_up = np.array(sword['/reaches/n_rch_up'][:])
n_rch_down = np.array(sword['/reaches/n_rch_down'][:])
rch_id_up = np.array(sword['/reaches/rch_id_up'][:])
rch_id_down = np.array(sword['/reaches/rch_id_dn'][:])

#flag reaches with zero downstream reaches but are not ghost reaches. 
category = np.array([int(str(ind)[-1]) for ind in reaches])
break_check = np.where((n_rch_down == 0) & (category != 6))[0]
rch_check = reaches[break_check]

rch_csv = pd.DataFrame({"reach_id": rch_check})
rch_csv.to_csv(outdir+'downstream_break_check.csv', index = False)


