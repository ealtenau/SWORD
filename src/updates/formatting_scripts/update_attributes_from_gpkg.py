# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import geopandas as gp
import sys
import argparse
from src.updates.sword import SWORD

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("dist_update", help="True or False: Whether or not to update distance from outlet attribute", type = str)
args = parser.parse_args()

region = args.region
version = args.version
dist_update = args.dist_update

# region = 'NA'
# version = 'v17b'
# dist_update = 'True'

sword = SWORD(main_dir, region, version)
gpkg_fn = sword.paths['gpkg_dir']+sword.paths['gpkg_fn']
nc_fn = sword.paths['nc_dir']+sword.paths['nc_fn']

#read geopackage data.
gpkg = gp.read_file(gpkg_fn)

if len(sword.reaches.id) != len(gpkg):
    print('!!! Reaches in NetCDF not equal to GPKG !!!')
    sys.exit()

print('Updating Attributes from SHP File')
unq_rchs = np.array(gpkg['reach_id'])
for r in list(range(len(unq_rchs))):
    # print(r, len(unq_rchs)-1)
    rch = np.where(sword.reaches.id == unq_rchs[r])[0]
    nds = np.where(sword.nodes.reach_id == unq_rchs[r])[0] 
    sword.reaches.main_side[rch] = gpkg['main_side'][r]
    sword.nodes.main_side[nds] = gpkg['main_side'][r]
    sword.reaches.river_name[rch] = gpkg['river_name'][r]
    sword.nodes.river_name[nds] = gpkg['river_name'][r]
    sword.reaches.end_rch[rch] = gpkg['end_reach'][r]
    sword.nodes.end_rch[nds] = gpkg['end_reach'][r]
    sword.reaches.network[rch] = gpkg['network'][r]
    sword.nodes.network[nds] = gpkg['network'][r]
    sword.reaches.path_freq[rch] = gpkg['path_freq'][r]
    sword.nodes.path_freq[nds] = gpkg['path_freq'][r]
    sword.reaches.path_order[rch] = gpkg['path_order'][r]
    sword.nodes.path_order[nds] = gpkg['path_order'][r]
    sword.reaches.path_segs[rch] = gpkg['path_segs'][r]
    sword.nodes.path_segs[nds] = gpkg['path_segs'][r]
    sword.reaches.strm_order[rch] = gpkg['strm_order'][r]
    sword.nodes.strm_order[nds] = gpkg['strm_order'][r]
    if dist_update == 'True': 
        sword.reaches.dist_out[rch] = gpkg['dist_out'][r]
        sort_nodes = np.argsort(sword.nodes.id[nds])
        base_val = gpkg['dist_out'][r] - gpkg['reach_len'][r]
        node_cs = np.cumsum(sword.nodes.len[nds[sort_nodes]])
        sword.nodes.dist_out[nds[sort_nodes]] = node_cs+base_val 

print('Updating NetCDF')
sword.save_nc()
print('DONE')