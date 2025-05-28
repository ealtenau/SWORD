# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import netCDF4 as nc
import geopandas as gp
import sys
import argparse
import src.updates.sword_utils as swd

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

paths = swd.prepare_paths(main_dir, region, version)
gpkg_fn = paths['gpkg_dir']+paths['gpkg_fn']
nc_fn = paths['nc_dir']+paths['nc_fn']

#read data.
centerlines, nodes, reaches = swd.read_nc(nc_fn)
gpkg = gp.read_file(gpkg_fn)

if len(reaches.id) != len(gpkg):
    print('!!! Reaches in NetCDF not equal to GPKG !!!')
    sys.exit()

print('Updating Attributes from SHP File')
unq_rchs = np.array(gpkg['reach_id'])
for r in list(range(len(unq_rchs))):
    # print(r, len(unq_rchs)-1)
    rch = np.where(reaches.id == unq_rchs[r])[0]
    nds = np.where(nodes.reach_id == unq_rchs[r])[0] 
    reaches.main_side[rch] = gpkg['main_side'][r]
    nodes.main_side[nds] = gpkg['main_side'][r]
    reaches.river_name[rch] = gpkg['river_name'][r]
    nodes.river_name[nds] = gpkg['river_name'][r]
    reaches.end_rch[rch] = gpkg['end_reach'][r]
    nodes.end_rch[nds] = gpkg['end_reach'][r]
    reaches.network[rch] = gpkg['network'][r]
    nodes.network[nds] = gpkg['network'][r]
    reaches.path_freq[rch] = gpkg['path_freq'][r]
    nodes.path_freq[nds] = gpkg['path_freq'][r]
    reaches.path_order[rch] = gpkg['path_order'][r]
    nodes.path_order[nds] = gpkg['path_order'][r]
    reaches.path_segs[rch] = gpkg['path_segs'][r]
    nodes.path_segs[nds] = gpkg['path_segs'][r]
    reaches.strm_order[rch] = gpkg['strm_order'][r]
    nodes.strm_order[nds] = gpkg['strm_order'][r]
    if dist_update == 'True': 
        reaches.dist_out[rch] = gpkg['dist_out'][r]
        sort_nodes = np.argsort(nodes.id[nds])
        base_val = gpkg['dist_out'][r] - gpkg['reach_len'][r]
        node_cs = np.cumsum(nodes.len[nds[sort_nodes]])
        nodes.dist_out[nds[sort_nodes]] = node_cs+base_val 

print('Updating NetCDF')
swd.discharge_attr_nc(reaches)
swd.write_nc(centerlines, nodes, reaches, nc_fn)
print('DONE')