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

zero_nodes = np.where(nodes.wth <= 0)[0]
unq_rchs = np.unique(nodes.reach_id[zero_nodes])
for r in list(range(len(unq_rchs))):
    nind = np.where(nodes.reach_id == unq_rchs[r])[0]
    min_wth = np.median(nodes.wth[nind[np.where(nodes.wth[nind] > 0)[0]]])
    z = np.where(nodes.wth[nind] <= 0)[0]
    nodes.wth[nind[z]] = min_wth

#write csv of zero width nodes for reference. 
csv = pd.DataFrame({"node_id": nodes[zero_nodes]})
csv.to_csv(out_dir+region.lower()+'_'+version+'_nodes_zero_widths_filled.csv', index = False)

#update netcdf. 
if min(nodes.wth) > 0:
    print('Writing NetCDF')
    swd.discharge_attr_nc(reaches)
    swd.write_nc(centerlines, nodes, reaches, sword_fn)