# -*- coding: utf-8 -*-
"""
Filling Zero Node Width Values (fill_zero_wths.py)
===================================================

This script fills in zero value node widths in 
the SWOT River Database (SWORD) based on other 
node width values within a reach.

A csv file containing the updated nodes is written
to sword.paths['update_dir'].

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python path/to/fill_zero_wths.py NA v17

"""

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
args = parser.parse_args()

region = args.region
version = args.version

sword = SWORD(main_dir, region, version)
sword.copy() #copies original file for version control. 

out_dir = sword.paths['update_dir']

zero_nodes = np.where(sword.nodes.wth <= 0)[0]
unq_rchs = np.unique(sword.nodes.reach_id[zero_nodes])
for r in list(range(len(unq_rchs))):
    nind = np.where(sword.nodes.reach_id == unq_rchs[r])[0]
    min_wth = np.median(sword.nodes.wth[nind[np.where(sword.nodes.wth[nind] > 0)[0]]])
    z = np.where(sword.nodes.wth[nind] <= 0)[0]
    sword.nodes.wth[nind[z]] = min_wth

#write csv of zero width nodes for reference. 
csv = pd.DataFrame({"node_id": sword.nodes.id[zero_nodes]})
csv.to_csv(out_dir+region.lower()+'_'+version+'_nodes_zero_widths_filled.csv', index = False)

#update netcdf. 
if min(sword.nodes.wth) > 0:
    print('Writing NetCDF')
    sword.save_nc()