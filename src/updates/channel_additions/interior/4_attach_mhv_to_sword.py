"""
Adding Interior MHV Additions to SWORD 
(4_attach_mhv_to_sword.py)
==================================================

This script reads in identified MHV rivers and 
adds them to the SWOT River Database (SWORD).

Output is a new SWORD netCDF file with all MHV 
location and hydrologic attributes added.

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python path/to/4_attach_mhv_to_sword.py NA v17

"""
from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import time
import glob
import argparse
import src.updates.channel_additions.interior.mhv_to_sword_tools as mst
from src.updates.sword import SWORD

start_all = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

### mhv addition files. 
mhv_nc_dir = main_dir+'/data/inputs/MHV_SWORD/netcdf/' + region +'/'
mhv_nc_files = np.sort(glob.glob(os.path.join(mhv_nc_dir, '*.nc')))

### read in sword data. 
sword = SWORD(main_dir, region, version)
sword_rch_basins = np.array([int(str(ind)[0:6]) for ind in sword.reaches.id])
sword_rch_nums = np.array([int(str(ind)[6:10]) for ind in sword.reaches.id])
max_id = max(sword.centerlines.cl_id)

### loop through mhv level 2 basins. 
for f in list(range(len(mhv_nc_files))):
    print('Starting Basin:', mhv_nc_files[f][-13:-11])
    
    ### read in data. 
    subcls = mst.read_mhv_sword(mhv_nc_files[f])
    if len(subcls.cl_id) == 0:
        print("No Additions")
        continue
    
    ### find single point reaches.
    rmv = [] 
    unq_rchs = np.unique(subcls.reach_id)
    for r in list(range(len(unq_rchs))):
        rch = np.where(subcls.reach_id == unq_rchs[r])[0]
        if len(rch) == 1:
            rmv.extend(rch)
            if subcls.add_flag[rch] == 3:
                # print(r)
                rmv_seg = np.where(subcls.seg == subcls.seg[rch])[0]
                rmv_ind = subcls.ind[rch]
                update_pt = np.where(subcls.ind[rmv_seg] == rmv_ind+1)[0]
                subcls.add_flag[rmv_seg[update_pt]] = 3
                
    ### delete identified indexes.
    mst.delete_mhv_reaches(subcls, rmv)

    ### make sure centerline ids are unique. 
    print('Reformat Cl IDs')
    mst.renumber_cl_id(subcls, max_id)     
    max_id = max(subcls.new_cl_id) #len(np.unique(subcls.new_cl_id)), len(subcls.cl_id)

    ### renumber based on sword reaches. 
    print('Renumber MHV Reaches')
    subcls.basins = np.array([int(str(r)[0:6]) for r in subcls.reach_id])
    subcls.new_reach_id = np.zeros([4,len(subcls.basins)], dtype=int)
    subcls.new_node_id = np.zeros([4,len(subcls.basins)], dtype=int)
    subcls.new_reach_id[0,:], subcls.new_node_id[0,:] = mst.renumber_reaches(subcls, sword_rch_basins, sword_rch_nums)

    ### define mhv topology.
    print('MHV Topology')
    mst.fill_mhv_topology(subcls)

    print('Removing Ghost Reaches at Junctions')
    ### find and delete ghost reaches at junctions. 
    rmv_ghost = mst.remove_ghost_juncs(subcls)
    if len(rmv_ghost) > 0:
        rmv2 = np.where(np.in1d(subcls.new_reach_id[0,:], rmv_ghost)==True)[0] #subcls.new_reach_id[0,rmv2]
        mst.delete_mhv_reaches(subcls, rmv2) #np.where(subcls.new_reach_id == rmv_ghost[0])[1]
    print('~~~ ghost reaches removed:', len(rmv_ghost))

    print('Ensuring Join Points after Deletions')
    mst.fill_missing_join_pts(subcls)
    
    print('Checking Short Reach Topology')
    short_check = mst.check_sort_rch_topo(subcls)
    mst.correct_short_rchs(subcls, short_check)
    print('~~~ short reaches fixed:', len(short_check))

    ### find sword joining reaches and update topology. 
    print('SWORD Topology')
    mst.join_topology(subcls, sword)

    ### create reach and node dimensions for mhv. 
    print('Creating Node Attributes')
    # Defining node attributes.
    subnodes = mst.Object()
    subnodes.id, subnodes.x, subnodes.y, subnodes.len, subnodes.wse,\
        subnodes.wse_var, subnodes.wth, subnodes.wth_var, subnodes.facc, subnodes.nchan_max,\
        subnodes.nchan_mod, subnodes.reach_id,\
        subnodes.grod, subnodes.lakeflag, subnodes.grod_fid,\
        subnodes.hfalls_fid, subnodes.lake_id = mst.basin_node_attributes(subcls.new_node_id[0,:],subcls.node_len,subcls.elv, subcls.wth,
                                                                        subcls.facc, subcls.nchan,subcls.lon, subcls.lat,
                                                                        subcls.new_reach_id[0,:], subcls.grod,subcls.lake, subcls.grod_fid,
                                                                        subcls.hfalls_fid, subcls.lake_id)
    # Node filler variables. 
    subnodes.ext_dist_coef = np.repeat(3,len(subnodes.id))
    subnodes.wth_coef = np.repeat(0.5, len(subnodes.id))
    subnodes.meand_len = np.repeat(-9999, len(subnodes.id))
    subnodes.river_name = np.repeat('NODATA', len(subnodes.id))
    subnodes.manual_add = np.repeat(1, len(subnodes.id))
    subnodes.sinuosity = np.repeat(-9999, len(subnodes.id))
    subnodes.edit_flag = np.repeat('7', len(subnodes.id))
    subnodes.trib_flag = np.repeat(0, len(subnodes.id))
    subnodes.max_wth = subnodes.wth #filled with '-9999' in v17.   
    
    print('Creating Reach Attributes')
    # Defining reach attributes.
    subreaches = mst.Object()
    subreaches.id, subreaches.x, subreaches.y, subreaches.x_max,\
        subreaches.x_min, subreaches.y_max, subreaches.y_min, subreaches.len,\
        subreaches.wse, subreaches.wse_var, subreaches.wth, subreaches.wth_var,\
        subreaches.nchan_max, subreaches.nchan_mod, subreaches.rch_n_nodes,\
        subreaches.slope, subreaches.grod, subreaches.lakeflag,\
        subreaches.facc, subreaches.grod_fid,\
        subreaches.hfalls_fid, subreaches.lake_id = mst.reach_attributes(subcls)

    print('Calculating SWOT Coverage')
    # Calculating swot coverage.
    subreaches.coverage, subreaches.orbits, subreaches.max_obs,\
        subreaches.median_obs, subreaches.mean_obs = mst.swot_obs_percentage(subcls, subreaches)

    ### create subreaches topology variables. 
    print('Reach Topology Variables')
    mst.subreach_topo_variables(subreaches, subcls)

    print('Formatting Centerline IDs')
    subcls.id = np.copy(subcls.new_cl_id)
    subreaches.cl_id, subnodes.cl_id = mst.centerline_ids(subreaches, subnodes, subcls)
    
    print('Filler Variables')
    ### create filler variables.
    subreaches.iceflag = np.zeros([len(subreaches.id),366])
    subreaches.iceflag[:,:] = -9999
    subreaches.max_wth = subreaches.wth
    subreaches.river_name = np.repeat('NODATA', len(subreaches.id))
    subreaches.low_slope = np.repeat(0, len(subreaches.id))
    subreaches.edit_flag = np.repeat('7', len(subreaches.id))
    subreaches.trib_flag = np.repeat(0, len(subreaches.id))

    subreaches.dist_out = np.repeat(-9999, len(subreaches.id))
    subnodes.dist_out = np.repeat(-9999, len(subnodes.id))

    subnodes.path_freq = np.repeat(-9999, len(subnodes.id))
    subnodes.path_order = np.repeat(-9999, len(subnodes.id))
    subnodes.path_segs = np.repeat(-9999, len(subnodes.id))
    subnodes.main_side = np.repeat(0, len(subnodes.id))
    subnodes.strm_order = np.repeat(-9999, len(subnodes.id))
    subnodes.network = np.repeat(0, len(subnodes.id))
    subnodes.end_rch = np.repeat(0, len(subnodes.id))

    subreaches.path_freq = np.repeat(-9999, len(subreaches.id))
    subreaches.path_order = np.repeat(-9999, len(subreaches.id))
    subreaches.path_segs = np.repeat(-9999, len(subreaches.id))
    subreaches.main_side = np.repeat(0, len(subreaches.id))
    subreaches.strm_order = np.repeat(-9999, len(subreaches.id))
    subreaches.network = np.repeat(0, len(subreaches.id))
    subreaches.end_rch = np.repeat(0, len(subreaches.id))

    # Transpose variables.
    subreaches.cl_id = subreaches.cl_id.T
    subnodes.cl_id = subnodes.cl_id.T
    subreaches.orbits = subreaches.orbits.T
    subreaches.iceflag = subreaches.iceflag.T

    ### append data. 
    # Append new data to existing data. 
    sword.append_data(subcls, subnodes, subreaches)
    # print('Cl Dimensions:', len(np.unique(sword.centerlines.cl_id)), len(sword.centerlines.cl_id))
    # print('Node Dimensions:', len(np.unique(sword.centerlines.node_id[0,:])), len(np.unique(nodes.id)), len(nodes.id))
    # print('Rch Dimensions:', len(np.unique(sword.centerlines.reach_id[0,:])), len(np.unique(nodes.reach_id)), len(np.unique(reaches.id)),len(reaches.id))
    # print('Cl Dimensions:', len(np.unique(subcls.new_cl_id)), len(subcls.new_cl_id))
    # print('Node Dimensions:', len(np.unique(subcls.new_node_id[0,:])), len(np.unique(subnodes.id)), len(subnodes.id))
    # print('Rch Dimensions:', len(np.unique(subcls.new_reach_id[0,:])), len(np.unique(subnodes.reach_id)), len(np.unique(subreaches.id)),len(subreaches.id))

### write netcdf
sword.save_nc()
print('Cl Dimensions:', len(np.unique(sword.centerlines.cl_id)), len(sword.centerlines.cl_id))
print('Node Dimensions:', len(np.unique(sword.centerlines.node_id[0,:])), len(np.unique(sword.nodes.id)), len(sword.nodes.id))
print('Rch Dimensions:', len(np.unique(sword.centerlines.reach_id[0,:])), len(np.unique(sword.nodes.reach_id)), len(np.unique(sword.reaches.id)),len(sword.reaches.id))

