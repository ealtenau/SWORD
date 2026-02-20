"""
Format MHV Additions (2_format_mhv_coastal_additions.py)
========================================================

This script updates identified coastal MHV additions 
for SWORD based on manual edits to the MHV addition
geopackage files.

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python path/to/2_format_mhv_coastal_additions.py NA v17

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import time
import netCDF4 as nc
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

mhv_nc_dir = main_dir+'/data/inputs/MHV_SWORD/netcdf/' + region +'/'
mhv_gpkg_dir = main_dir+'/data/inputs/MHV_SWORD/gpkg/' + region +'/coast_additions/'
mhv_nc_files = np.sort(glob.glob(os.path.join(mhv_nc_dir, '*.nc')))
mhv_gpkg_files = np.sort(glob.glob(os.path.join(mhv_gpkg_dir, '*.gpkg')))

### make sure file lists match up. 
gpkg_nums = [ind[-33:-31] for ind in mhv_gpkg_files]
nc_nums = [ind[-13:-11] for ind in mhv_nc_files]
fidx = np.where(np.in1d(nc_nums, gpkg_nums)==True)[0]
mhv_nc_files = mhv_nc_files[fidx]

### pre-formatting: find based on a few variables the indexes to read in the data. 
for f in list(range(len(mhv_nc_files))):
    print('Starting Basin:', mhv_gpkg_files[f][-33:-31])
    ### get key mhv variables.
    mhv_gpkg = gp.read_file(mhv_gpkg_files[f])
    gpkg_flag = np.array(mhv_gpkg['add_flag'])
    gpkg_segs = np.array(mhv_gpkg['new_segs'])
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
    mhv_segs_ind = np.array(mhv_nc['/centerlines/new_segs_ind'][:])
    mhv_basins = np.array(mhv_nc['/centerlines/basin_code'][:])
    mhv_rchs = np.array(mhv_nc['/centerlines/reach_id'][:])
    mhv_nodes = np.array(mhv_nc['/centerlines/node_id'][:])
    mhv_node_num = np.array(mhv_nc['/centerlines/node_num'][:])
    mhv_rch_num = np.array(mhv_nc['/centerlines/rch_num'][:])
    mhv_type = np.array(mhv_nc['/centerlines/type'][:])
    mhv_rch_dist = np.array(mhv_nc['/centerlines/rch_dist'][:])
    mhv_rch_len = np.array(mhv_nc['/centerlines/rch_len'][:])
    mhv_rch_ind = np.array(mhv_nc['/centerlines/rch_ind'][:])
    mhv_node_len = np.array(mhv_nc['/centerlines/node_len'][:])
    mhv_elv = np.array(mhv_nc['/centerlines/p_height'][:])
    mhv_rch_len[np.where(mhv_rch_len <= 30)[0]] = 90
    mhv_node_len[np.where(mhv_node_len <= 30)[0]] = mhv_rch_len[np.where(mhv_node_len <= 30)[0]]
    mhv_network = np.array(mhv_nc['/centerlines/network_all'][:])
    mhv_add_flag_old = np.array(mhv_nc['/centerlines/add_coast/'][:])
    
    #update add flag 
    mhv_add_flag = np.zeros(len(mhv_add_flag_old))
    keep = np.where(np.in1d(mhv_segs, unq_gpkg_segs) == True)[0]
    mhv_add_flag[keep] = mhv_add_flag_old[keep]

    ### fill in zero basin segments. 
    zero_segs = np.unique(mhv_segs[np.where(mhv_basins == 0)[0]])
    for z in list(range(len(zero_segs))):
        zpts = np.where(mhv_segs == zero_segs[z])[0]
        nonzero = np.where(mhv_basins[zpts] > 0)[0]
        if len(nonzero) == 0:
            continue 
        else:
            #fill zero points with basin values of farthest downstream point (based on index). 
            mn_pt = np.where(mhv_segs_ind[zpts[nonzero]] == min(mhv_segs_ind[zpts[nonzero]]))[0]
            fill_val = mhv_basins[zpts[nonzero[mn_pt]]]
            fill_pts = np.where(mhv_basins[zpts] == 0)[0]
            mhv_basins[zpts[fill_pts]] = fill_val

    ### update reach and node ids that had zero basin values. 
    rch_update = np.where((mhv_rchs < 10000000000) & (mhv_basins > 0))[0]
    for r in list(range(len(rch_update))):
        basin = str(mhv_basins[rch_update[r]])[0:6]
        reach = str(mhv_rch_num[rch_update[r]])
        node = str(mhv_node_num[rch_update[r]])
        rtype = str(mhv_type[rch_update[r]])
        # create string reach and node numbers. 
        if len(reach) == 1:
            rch_str = '000'+reach 
        if len(reach) == 2:
            rch_str = '00'+reach
        if len(reach) == 3:
            rch_str = '0'+reach
        if len(reach) == 4:
            rch_str = reach
        ### node ids
        if len(node) == 1:
            node_str = '00'+node
        if len(node) == 2:
            node_str = '0'+node
        if len(node) == 3:
            node_str = node
        new_rch_id = int(basin+rch_str+rtype)
        new_node_id = int(basin+rch_str+node_str+rtype)
        #update arrays. 
        mhv_rchs[rch_update[r]] = new_rch_id
        mhv_nodes[rch_update[r]] = new_rch_id
        
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
        mhv_nc['/centerlines/add_coast'][:] = mhv_add_flag
        mhv_nc['/centerlines/reach_id'][:] = mhv_rchs
        mhv_nc['/centerlines/node_id'][:] = mhv_nodes
        mhv_nc['/centerlines/node_num'][:] = mhv_node_num
        mhv_nc['/centerlines/rch_num'][:] = mhv_rch_num
        mhv_nc['/centerlines/rch_len'][:] = mhv_rch_len
        mhv_nc['/centerlines/node_len'][:] = mhv_node_len
        mhv_nc['/centerlines/rch_dist'][:] = mhv_rch_dist
        mhv_nc['/centerlines/basin_code'][:] = mhv_basins
        mhv_nc.close()
        print('single reaches corrected:', len(single))

print('DONE')