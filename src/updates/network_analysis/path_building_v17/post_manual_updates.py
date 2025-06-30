"""
Updating NetCDF values based on Geopackage values.
(post_manual_updates.py)
===============================================================

This scripts updates the SWORD netCDF file based on values in 
the SWORD geopackage files if they were manually edited. A 
refactored, universal version of this script is located at:
'src/updates/formatting_scripts/update_attributes_from_gpkg.py'.  

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA), SWORD version (i.e. v17), 
and a "dist_update" parameter indicating whether to 
update the distance from outlet variable or not (True/False).

Execution example (terminal):
    python post_manual_updates.py NA v17 False

""" 

import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import netCDF4 as nc
import geopandas as gp
import geopy.distance
from shapely.geometry import LineString, Point
from geopandas import GeoSeries
import pandas as pd
import time
import argparse
from scipy import spatial as sp
import matplotlib.pyplot as plt
from scipy import stats
from scipy import interpolate
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("dist_update", help="True or False: Whether or not to update distance from outlet attribute", type = str)
args = parser.parse_args()

region = args.region
version = args.version
dist_update = args.dist_update

#filepaths. 
gpkg_fn = main_dir+'/data/outputs/Reaches_Nodes/'+version+'/gpkg/'\
    +region.lower()+'_sword_reaches_'+version+'.gpkg'
nc_fn = main_dir+'/data/outputs/Reaches_Nodes/'+version+'/netcdf/'\
    +region.lower()+'_sword_'+version+'.nc'
con_fn = sword_dir = main_dir+'/data/outputs/Reaches_Nodes/'+version+\
    '/reach_geometry/'+region.lower()+'_sword_'+version+'_connectivity.nc'

#read data. 
sword = nc.Dataset(nc_fn, 'r+')
con = nc.Dataset(con_fn)
gpkg = gp.read_file(gpkg_fn)

#assign relevant data to arrays. 
cl_rchs = np.array(con.groups['centerlines'].variables['reach_id'][:])
cl_id = np.array(con.groups['centerlines'].variables['cl_id'][:])
cl_x = np.array(con.groups['centerlines'].variables['x'][:])
cl_y = np.array(con.groups['centerlines'].variables['y'][:])
cl_ends = np.array(con.groups['centerlines'].variables['end_reach'][:])
reaches = np.array(sword.groups['reaches'].variables['reach_id'][:])
reach_dist = np.array(sword.groups['reaches'].variables['dist_out'][:])
rch_len = np.array(sword.groups['reaches'].variables['reach_length'][:])
node_rchs = np.array(sword.groups['nodes'].variables['reach_id'][:])
rch_ms = np.array(sword.groups['reaches'].variables['main_side'][:])
node_ms = np.array(sword.groups['nodes'].variables['main_side'][:])
rch_rn = np.array(sword.groups['reaches'].variables['river_name'][:])
node_rn = np.array(sword.groups['nodes'].variables['river_name'][:])
path_freq = np.array(sword.groups['reaches'].variables['path_freq'][:])
node_path_freq = np.array(sword.groups['nodes'].variables['path_freq'][:])
path_order = np.array(sword.groups['reaches'].variables['path_order'][:])
node_path_order = np.array(sword.groups['nodes'].variables['path_order'][:])
rch_x = np.array(sword.groups['reaches'].variables['x'][:])
rch_y = np.array(sword.groups['reaches'].variables['y'][:])
rch_segs = np.array(sword.groups['reaches'].variables['path_segs'][:])
rch_ends = np.array(sword.groups['reaches'].variables['end_reach'][:])
node_ends = np.array(sword.groups['nodes'].variables['end_reach'][:])
nodes = np.array(sword.groups['nodes'].variables['node_id'][:])
node_dist = np.array(sword.groups['nodes'].variables['dist_out'][:])
node_len = np.array(sword.groups['nodes'].variables['node_length'][:])
rch_net = np.array(sword.groups['reaches'].variables['network'][:])
node_net = np.array(sword.groups['nodes'].variables['network'][:])
path_segs = np.array(sword.groups['reaches'].variables['path_segs'][:])
node_path_segs = np.array(sword.groups['nodes'].variables['path_segs'][:])

#update netCDF attribute values with associated geopackage attribute values. 
print('Updating Attributes from SHP File')
unq_rchs = np.array(gpkg['reach_id'])
for r in list(range(len(unq_rchs))):
    # print(r, len(unq_rchs)-1)
    rch = np.where(reaches == unq_rchs[r])[0]
    nds = np.where(node_rchs == unq_rchs[r])[0] 
    rch_ms[rch] = gpkg['main_side'][r]
    node_ms[nds] = gpkg['main_side'][r]
    rch_rn[rch] = gpkg['river_name'][r]
    node_rn[nds] = gpkg['river_name'][r]
    rch_ends[rch] = gpkg['end_reach'][r]
    node_ends[nds] = gpkg['end_reach'][r]
    rch_net[rch] = gpkg['network'][r]
    node_net[nds] = gpkg['network'][r]
    path_freq[rch] = gpkg['path_freq'][r]
    node_path_freq[nds] = gpkg['path_freq'][r]
    path_order[rch] = gpkg['path_order'][r]
    node_path_order[nds] = gpkg['path_order'][r]
    path_segs[rch] = gpkg['path_segs'][r]
    node_path_segs[nds] = gpkg['path_segs'][r]
    if dist_update == True: 
        reach_dist[rch] = gpkg['dist_out'][r]

#update stream order based on new attribute information. 
print('Updating Stream Order')
strm_order_all = np.zeros(len(reaches))
level2 = np.array([int(str(r)[0:2]) for r in reaches])
unq_l2 = np.unique(level2)
for ind in list(range(len(unq_l2))):
    # print(unq_l2[ind])
    l2 = np.where(level2 == unq_l2[ind])[0]
    
    strm_order = np.zeros(len(path_freq[l2]))
    normalize = np.where((rch_ms[l2] == 0)& (path_freq[l2]>0))[0] 
    strm_order[normalize] = (np.round(np.log(path_freq[l2[normalize]])))+1
    strm_order[np.where(rch_ms[l2] > 0)] = -9999
    strm_order[np.where(path_freq[l2] == -9999)] = -9999

    #filter stream order.
    subpaths = path_order[l2]
    subdist = reach_dist[l2] 
    deltas = rch_ms[l2]
    unq_pths = np.unique(subpaths[np.where(deltas == 0)[0]])
    unq_pths = unq_pths[::-1]
    unq_pths = unq_pths[unq_pths>0]
    for p in list(range(len(unq_pths))):
        pth = np.where(subpaths == unq_pths[p])[0]
        sort_inds = np.argsort(subdist[pth])
        diff = np.diff(strm_order[pth[sort_inds]])
        wrong = np.where(diff > 0)[0] 
        if len(wrong) > 0:
            # print(p, unq_pths[p])
            max_break = wrong[-1]+1
            
            if max_break == max(sort_inds):
                max_break = max_break-1
            
            new_val = strm_order[pth[sort_inds[max_break+1]]]
            strm_order[pth[sort_inds[0:max_break]]] = new_val
    
    # strm_order[np.where(deltas > 0)] = -9999
    strm_order[np.where(rch_ms[l2] > 0)] = -9999
    strm_order[np.where(path_freq[l2] == -9999)] = -9999
    strm_order[np.where(strm_order == 0)] = -9999
    strm_order_all[l2] = strm_order

#updating distance from outlet at the node scale. 
print('Updating Node Attributes')
nodes_strm_order_all = np.zeros(len(nodes))
for r2 in list(range(len(reaches))):
    # print(r2, len(reaches)-1)
    nds = np.where(node_rchs == reaches[r2])[0]
    nodes_strm_order_all[nds] = strm_order_all[r2]
    #updating dist_out for nodes.
    if dist_update == True: 
        base_val = reach_dist[r2] - rch_len[r2]
        node_cs = np.cumsum(node_len[nds])
        node_dist[nds] = node_cs+base_val 

# good = np.where(strm_order_all != -9999)[0]        
# plt.scatter(rch_x[good], rch_y[good], c=strm_order_all[good], s=5)
# plt.show()

#updating end reach value.
print('Updating End Reach Values')
node_l2 = np.array([int(str(n)[0:2]) for n in nodes])
ends = nodes[np.where((node_ends > 0)&(node_ends<3))[0]]
for idx in list(range(len(unq_l2))):
    # print('Basin:', unq_l2[idx])
    nl2 = np.where(node_l2 == unq_l2[idx])[0]
    subnodes = nodes[nl2]
    subends = node_ends[nl2]
    subndist = node_dist[nl2]
    subnpaths = node_path_order[nl2]
    subnode_rchs = node_rchs[nl2]
    for e in list(range(len(ends))):
        pt = np.where(subnodes == ends[e])[0]
        if len(pt) == 0:
            continue
        rch_pt = np.where(reaches == subnode_rchs[np.where(subnodes == ends[e])[0]])[0]
        end_val = subends[pt]
        pt_dist = subndist[pt]
        pth = subnpaths[pt]
        pth_mx_dist = max(subndist[np.where(subnpaths == pth)])
        pth_mn_dist = min(subndist[np.where(subnpaths == pth)])
        if end_val == 1 and pt_dist == pth_mn_dist:
            # print('headwater to outlet: ', ends[e])
            node_ends[nl2[pt]] = 2
            rch_ends[rch_pt] = 2
        if end_val == 2 and pt_dist == pth_mx_dist:
            # print('outlet to headwater: ', ends[e])
            node_ends[nl2[pt]] = 1
            rch_ends[rch_pt] = 1

#updating netCDF.
print('Updating NetCDF')
sword.groups['reaches'].variables['main_side'][:] = rch_ms
sword.groups['nodes'].variables['main_side'][:] = node_ms
sword.groups['reaches'].variables['river_name'][:] = rch_rn
sword.groups['nodes'].variables['river_name'][:] = node_rn
sword.groups['reaches'].variables['end_reach'][:] = rch_ends
sword.groups['nodes'].variables['end_reach'][:] = node_ends
sword.groups['reaches'].variables['stream_order'][:] = strm_order_all
sword.groups['nodes'].variables['stream_order'][:] = nodes_strm_order_all
sword.groups['reaches'].variables['network'][:] = rch_net
sword.groups['nodes'].variables['network'][:] = node_net
sword.groups['reaches'].variables['path_freq'][:] = path_freq
sword.groups['nodes'].variables['path_freq'][:] = node_path_freq
sword.groups['reaches'].variables['path_order'][:] = path_order
sword.groups['nodes'].variables['path_order'][:] = node_path_order
sword.groups['reaches'].variables['path_segs'][:] = path_segs
sword.groups['nodes'].variables['path_segs'][:] = node_path_segs
sword.groups['reaches'].variables['dist_out'][:] = reach_dist
sword.groups['nodes'].variables['dist_out'][:] = node_dist

rch_side = np.where(rch_ms == 1)[0]
node_side = np.where(node_ms == 1)[0]
sword.groups['reaches'].variables['path_freq'][rch_side] = -9999
sword.groups['reaches'].variables['path_order'][rch_side] = -9999
sword.groups['nodes'].variables['path_freq'][node_side] = -9999
sword.groups['nodes'].variables['path_order'][node_side] = -9999
sword.close()