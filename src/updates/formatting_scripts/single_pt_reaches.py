# -*- coding: utf-8 -*-
"""
Identifying Single Point Reaches (single_pt_reaches.py)
===============================================================

This script finds SWORD reaches that are made up of one 
centerline point and outputs the single point reaches to a 
csv file located at sword.paths['update_dir'].  

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python path/to/single_pt_reaches.py NA v17

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import pandas as pd
import argparse
from src.updates.sword import SWORD

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

#read data. 
sword = SWORD(main_dir, region, version)
outpath = sword.paths['update_dir']+region.lower()+'_'+version+'_single_pt_rchs.csv'

#find reaches that are only one centerline point in length. 
single_pt_rchs = []
for ind in list(range(len(sword.reaches.id))):
    print(ind, len(sword.reaches.id)-1)
    pts = np.where(sword.centerlines.reach_id[0,:] == sword.reaches.id[ind])[0]
    if len(pts) == 1:
        single_pt_rchs.append(ind)

#export reaches to csv file.
rch_list = sword.reaches.id[single_pt_rchs]
df = pd.DataFrame(rch_list)
df.to_csv(outpath)
print(len(rch_list)) 