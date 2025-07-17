# -*- coding: utf-8 -*-
"""
Format MHV Additions (2_format_mhv_additions.py)
===================================================

This script updates identified interior MHV additions 
for SWORD based on manual edits to the MHV addition
geopackage files.

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python path/to/2_format_mhv_additions.py NA v17

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import time
import netCDF4 as nc
import pandas as pd
from scipy import spatial as sp
from shapely.geometry import Point
import geopandas as gp
import glob
import argparse

### remove reaches/networks that and marked zero in gpkg from nc file. 
### correct single point reaches. 

start_all = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

sword_fn = main_dir+'/data/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
mhv_nc_dir = main_dir+'/data/inputs/MHV_SWORD/netcdf/' + region +'/'
mhv_gpkg_dir = main_dir+'/data/inputs/MHV_SWORD/gpkg/' + region +'/additions/'
mhv_nc_files = np.sort(glob.glob(os.path.join(mhv_nc_dir, '*.nc')))
mhv_gpkg_files = np.sort(glob.glob(os.path.join(mhv_gpkg_dir, '*.gpkg')))

### make sure file lists match up. 
gpkg_nums = [ind[-25:-23] for ind in mhv_gpkg_files]
nc_nums = [ind[-13:-11] for ind in mhv_nc_files]
fidx = np.where(np.in1d(nc_nums, gpkg_nums)==True)[0]
mhv_nc_files = mhv_nc_files[fidx]

### pre-formatting: find based on a few variables the indexes to read in the data. 
for f in list(range(len(mhv_nc_files))):
    print('Starting Basin:', mhv_gpkg_files[f][-25:-23])
    ### get key mhv variables.
    mhv_gpkg = gp.read_file(mhv_gpkg_files[f])
    gpkg_flag = np.array(mhv_gpkg['add_flag'])
    gpkg_segs = np.array(mhv_gpkg['new_segs'])
    gpkg_basins = np.array(mhv_gpkg['basin_code'])
    gpkg_x = np.array(mhv_gpkg['x'])
    gpkg_y = np.array(mhv_gpkg['y'])
    flag_idx = np.where(gpkg_flag>0)[0]
    unq_gpkg_segs = np.unique(gpkg_segs[flag_idx])
    # del mhv_gpkg, gpkg_flag, gpkg_segs

    ### loop through and remove and segments that have any zero add_flag values.
    rmv_segs = [] 
    rmv_idx = []
    for s in list(range(len(unq_gpkg_segs))):
        sind = np.where(gpkg_segs == unq_gpkg_segs[s])[0]
        check = np.where(gpkg_flag[sind] == 0)[0]
        if len(check) > 0:
            rmv_segs.append(unq_gpkg_segs[s])
            rmv_idx.append(s)
    rmv_segs = np.array(rmv_segs)
    rmv_idx = np.array(rmv_idx)
    if len(rmv_idx) > 0:
        unq_gpkg_segs = np.delete(unq_gpkg_segs, rmv_idx)

    ### associate with key mhv_nc variables. 
    mhv_nc = nc.Dataset(mhv_nc_files[f], 'r+')
    mhv_segs = np.array(mhv_nc['/centerlines/new_segs'][:])
    mhv_x = np.array(mhv_nc['/centerlines/x'][:])
    mhv_y = np.array(mhv_nc['/centerlines/y'][:])
    mhv_basins = np.array(mhv_nc['/centerlines/basin_code'][:])
    mhv_cl_id = np.array(mhv_nc['/centerlines/cl_id'][:])
    mhv_segs_ind = np.array(mhv_nc['/centerlines/new_segs_ind'][:])
    mhv_rchs = np.array(mhv_nc['/centerlines/reach_id'][:])
    mhv_nodes = np.array(mhv_nc['/centerlines/node_id'][:])
    mhv_node_num = np.array(mhv_nc['/centerlines/node_num'][:])
    mhv_rch_num = np.array(mhv_nc['/centerlines/rch_num'][:])
    mhv_rch_dist = np.array(mhv_nc['/centerlines/rch_dist'][:])
    mhv_rch_len = np.array(mhv_nc['/centerlines/rch_len'][:])
    mhv_rch_ind = np.array(mhv_nc['/centerlines/rch_ind'][:])
    mhv_node_len = np.array(mhv_nc['/centerlines/node_len'][:])
    mhv_elv = np.array(mhv_nc['/centerlines/p_height'][:])
    mhv_rch_len[np.where(mhv_rch_len <= 30)[0]] = 90
    mhv_node_len[np.where(mhv_node_len <= 30)[0]] = mhv_rch_len[np.where(mhv_node_len <= 30)[0]]
    mhv_network = np.array(mhv_nc['/centerlines/network'][:])
    mhv_add_flag_old = np.array(mhv_nc['/centerlines/add_flag/'][:])
    
    #update add flag 
    mhv_add_flag = np.zeros(len(mhv_add_flag_old))
    keep = np.where(np.in1d(mhv_segs, unq_gpkg_segs) == True)[0]
    mhv_add_flag[keep] = mhv_add_flag_old[keep]

    #updating basin codes
    gpkg_pts = np.vstack((gpkg_x, gpkg_y)).T
    nc_pts = np.vstack((mhv_x, mhv_y)).T
    kdt = sp.cKDTree(nc_pts)
    pt_dist, pt_ind = kdt.query(gpkg_pts, k = 1)
    mhv_basins[pt_ind] = gpkg_basins

    #see if there are joining points missing. 
    join_fix = []
    unq_nets = np.unique(mhv_network[keep])
    for n in list(range(len(unq_nets))):
        # print(n)
        net = np.where(mhv_network[keep] == unq_nets[n])[0]
        if max(mhv_add_flag[keep[net]]) < 3:
            join_fix.append(unq_nets[n])
            net_segs = np.unique(mhv_segs[keep[net]])
            if len(net_segs) > 1:
                mn_id = np.where(mhv_elv[keep[net]] == min(mhv_elv[keep[net]]))
                min_seg = np.unique(mhv_segs[keep[net[mn_id]]])
                join_seg = np.where(mhv_segs == min_seg)[0]
                join_add = np.where(mhv_add_flag[join_seg] > 0)[0]
                jpt = np.where(mhv_segs_ind[join_seg[join_add]] == min(mhv_segs_ind[join_seg[join_add]]))[0]
                mhv_add_flag[join_seg[join_add[jpt]]] = 3
            else:
                join_seg = np.where(mhv_segs == net_segs[0])[0]
                join_add = np.where(mhv_add_flag[join_seg] > 0)[0]
                jpt = np.where(mhv_segs_ind[join_seg[join_add]] == min(mhv_segs_ind[join_seg[join_add]]))[0]
                mhv_add_flag[join_seg[join_add[jpt]]] = 3
    print('joining points added:', len(join_fix))

    ### find and correct single point reaches:
    unq_rchs = np.unique(mhv_rchs[keep])
    for r in list(range(len(unq_rchs))):
        rch = np.where(mhv_rchs == unq_rchs[r])[0]
        if len(rch) == 1:
            # print(r)
            seg = np.where(mhv_segs == np.unique(mhv_segs[rch]))[0]
            seg_sort = seg[np.argsort(mhv_segs_ind[seg])]
            #update key variables based on lower point.. 
            pt = np.where(mhv_segs_ind[seg_sort] == mhv_segs_ind[rch])[0]
            if pt == 0:
                pt = pt + 1
            else:
                pt = pt - 1
            mhv_rchs[rch] = mhv_rchs[seg_sort[pt]]
            mhv_nodes[rch] = mhv_nodes[seg_sort[pt]]
            mhv_node_num[rch] = mhv_node_num[seg_sort[pt]]
            mhv_rch_num[rch] = mhv_rch_num[seg_sort[pt]]
            mhv_rch_dist[rch] = mhv_rch_dist[seg_sort[pt]] + mhv_rch_len[rch]
            mhv_rch_len[rch] = mhv_rch_len[seg_sort[pt]] + mhv_rch_len[rch]
            mhv_node_len[rch] = mhv_node_len[seg_sort[pt]] + mhv_node_len[rch]

    single = []
    unq_rchs = np.unique(mhv_rchs[keep])
    for r in list(range(len(unq_rchs))):
        rch = np.where(mhv_rchs == unq_rchs[r])[0]
        if len(rch) == 1:
            single.append(r)

    if len(single) > 0:
        print('!!Single Reaches!!')
        break
    else:
        ### update netcdf 
        mhv_nc['/centerlines/add_flag'][:] = mhv_add_flag
        mhv_nc['/centerlines/reach_id'][:] = mhv_rchs
        mhv_nc['/centerlines/node_id'][:] = mhv_nodes
        mhv_nc['/centerlines/node_num'][:] = mhv_node_num
        mhv_nc['/centerlines/rch_num'][:] = mhv_rch_num
        mhv_nc['/centerlines/rch_len'][:] = mhv_rch_len
        mhv_nc['/centerlines/node_len'][:] = mhv_node_len
        mhv_nc['/centerlines/rch_dist'][:] = mhv_rch_dist
        mhv_nc['/centerlines/basin_code'][:] = mhv_basins
        mhv_nc.close()

print('DONE')