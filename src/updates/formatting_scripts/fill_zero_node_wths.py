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