# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
                    Merging River & Water Body Datasets
-------------------------------------------------------------------------------
Copyright (c) 2018-2021 UNC Chapel Hill. All rights reserved.
Created by E. Altenau.

DESCRIPTON:
    This is the main script used to create the final reaches and nodes in the
    SWOT River Database (SWORD). Necessary accompanying scripts include
    "Reach_Definition_Tools.py", "Write_Database_Files.py", "Write_Node_shp.py",
    and "Write_Reach_shp.m".
    "Reach_Definition_Tools.py" includes essential functions used to define
    the reaches, "Write_Database_files.py" contains the functions used
    to output the final data in netcdf format, and "Write_Node_shp.py" &
    "Write_Reach_shp.m" are scripts to produce the node and reach shapefiles.

INPUTS:
    nc_file -- NetCDF file of the merged databases produced by the
        "Merged_Databases.py" script.

OUTPUTS:
    SWORD NetCDF -- NetCDF file containing attributes for the high-resolution
        centerline, node, and reach locations.
    SWOT_Coverage_NetCDF -- NetCDF file containing swot orbit coverage
        information for each reach.
-------------------------------------------------------------------------------
"""

from __future__ import division
import os
os.chdir('C:/Users/ealtenau/Documents/Research/SWAG/For_Server/scripts/reach_definition/')
import Reach_Definition_Tools_v11 as rdt
import Write_Database_Files as wf
import time
import numpy as np
#from scipy import spatial as sp
#import pandas as pd

###############################################################################
##################### Defining Reach and Node Locations #######################
###############################################################################
start_all = time.time()

region = 'EU'

# Input file.
nc_file = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/outputs/Merged_Data/'+region+'/'+region+'_Merge_v10.nc'
# Output files.
nc_outpath = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/outputs/Reaches_Nodes/netcdf/'+region.lower()+'_sword_v12.nc'
swot_outpath = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/outputs/SWOT_Coverage/'+region.lower()+'_swot_obs_v12.nc'

# Reading in data.
data = rdt.read_merge_netcdf(nc_file)

# Making sure flow accumulation minimum isn't zero.
data.facc[np.where(data.facc == 0)[0]] = 0.001

# Creating empty objects to fill with attributes.
centerlines = rdt.Object()
reaches = rdt.Object()
nodes = rdt.Object()

# Loop through each level 2 basin. Subsetting per level 2 basin speeds up the script.
level2_basins = np.array([np.int(np.str(ind)[0:2]) for ind in data.basins])
uniq_level2 = np.unique(level2_basins)
uniq_level2 = np.delete(uniq_level2, 0)
cnt = 0
start_id = 0
for ind in list(range(len(uniq_level2))):

    print('STARTING BASIN: ' + str(uniq_level2[ind]))

    start = time.time()

    # Define objects to assign attributes.
    subcls = rdt.Object()
    subreaches = rdt.Object()
    subnodes = rdt.Object()

    # Subset data.
    level2 = np.where(level2_basins == uniq_level2[ind])[0]
    subcls.id = data.id[level2]
    subcls.lon = data.lon[level2]
    subcls.lat = data.lat[level2]
    subcls.seg = data.seg[level2]
    subcls.ind = data.ind[level2]
    subcls.dist = data.segDist[level2]
    subcls.wth = data.wth[level2]
    subcls.elv = data.elv[level2]
    subcls.facc = data.facc[level2]
    subcls.lake = data.lake[level2]
    subcls.delta = data.delta[level2]
    subcls.nchan = data.nchan[level2]
    subcls.grand = data.grand[level2]
    subcls.grod = data.grod[level2]
    subcls.grod_fid = data.grod_fid[level2]
    subcls.hfalls_fid = data.hfalls_fid[level2]
    subcls.basins = data.basins[level2]
    subcls.manual = data.manual[level2]
    subcls.num_obs = data.num_obs[level2]
    subcls.orbits = data.orbits[level2]
    subcls.lake_id = data.lake_id[level2]
    subcls.lon[np.where(subcls.lon < -180)] = -180.0
    subcls.lon[np.where(subcls.lon > 180)] = 180.0
    subcls.x = np.copy(subcls.lon)
    subcls.y = np.copy(subcls.lat)

    #########################################################################
    #REACH DEFINITION

    print('Defining Intial Reaches')
    # Find all reach boundaries.
    dam_radius = 225
    subcls.lake_bnds, subcls.lake_dist, subcls.lake_num,\
     subcls.dam_bnds, subcls.dam_dist, subcls.dam_id, subcls.swot_bnds,\
     subcls.swot_dist, subcls.swot_id, subcls.all_bnds, \
     subcls.type0 = rdt.find_all_bounds(subcls, dam_radius)

    # Define inital reaches.
    subcls.rch_id0, subcls.rch_len0 = rdt.number_reaches(subcls)

    print('Cutting Long Reaches')
    subcls.rch_id1, subcls.rch_len1 = rdt.cut_reaches(subcls.rch_id0,
                                                      subcls.rch_len0, subcls.dist,
                                                      subcls.ind, 20000)

    # Make sure there is only one "type" value per reach.
    subcls.type1 = np.zeros(len(subcls.rch_id1))
    uniq_rch = np.unique(subcls.rch_id1)
    for idx in list(range(len(uniq_rch))):
        rch = np.where(subcls.rch_id1 == uniq_rch[idx])[0]
        if len(rch) <=2:
            subcls.type1[rch] = np.min(subcls.type0[rch])
        else:
            subcls.type1[rch] = max(set(list(subcls.type0[rch])),
                                    key=list(subcls.type0[rch]).count)

    print('Aggregating River Reaches')
    # Aggregating river reach types.
    river_min_dist = 10000
    subcls.rch_id2, subcls.rch_len2,\
     subcls.type2 = rdt.aggregate_rivers(subcls, river_min_dist)
    # Updating reach indexes.
    subcls.rch_ind2, __ = rdt.update_rch_indexes(subcls, subcls.rch_id2)

    print('Aggregating Lake Reaches')
    # Aggregating lake reach types.
    lake_min_dist = 10000
    subcls.rch_id3, subcls.rch_len3,\
     subcls.type3 = rdt.aggregate_lakes(subcls, lake_min_dist)
    # Updating reach indexes.
    subcls.rch_ind3, __ = rdt.update_rch_indexes(subcls, subcls.rch_id3)

    print('Aggregating Dam Reaches')
    # Aggregating dam reach types.
    dam_min_dist = 200
    subcls.rch_id4, subcls.rch_len4,\
     subcls.type4 = rdt.aggregate_dams(subcls, dam_min_dist)
    # Updating reache indexes.
    subcls.rch_ind4, __ = rdt.update_rch_indexes(subcls, subcls.rch_id4)
    # Updating reach flow distance.
    subcls.rch_dist4 = rdt.calc_segDist(subcls.lon, subcls.lat, subcls.rch_id4,
                                        subcls.facc, subcls.rch_ind4)

    print('Second Cutting of Long Reaches')
    subcls.rch_id5, subcls.rch_len5 = rdt.cut_reaches(subcls.rch_id4,
                                                      subcls.rch_len4, subcls.rch_dist4,
                                                      subcls.rch_ind4, 20000)
    subcls.type5 = np.copy(subcls.type4)
    # Updating reach indexes.
    subcls.rch_ind5, \
     subcls.rch_eps5 = rdt.update_rch_indexes(subcls, subcls.rch_id5)
    # Updating reach flow distance.
    subcls.rch_dist5 = rdt.calc_segDist(subcls.lon, subcls.lat, subcls.rch_id5,
                                        subcls.facc, subcls.rch_ind5)

    #########################################################################
    #TOPOLOGY

    print('Creating Reach ID')
    subcls.reach_id, subcls.rch_topo = rdt.reach_topology(subcls)

    # Creating Nodes.
    print('Creating Nodes')
    node_length = 200
    subcls.node_id, subcls.node_len,\
     subcls.node_num = rdt.node_reaches(subcls, node_length)

    print('Creating Ghost Reaches')
    # Defining ghost reaches and nodes.
    subcls.reach_id, subcls.node_id, subcls.rch_len6 = rdt.ghost_reaches(subcls)
    # Updating reach indexes and type.
    subcls.rch_ind6,\
     subcls.rch_eps6 = rdt.update_rch_indexes(subcls, subcls.reach_id)
    # Updating reach flow distance.
    subcls.rch_dist6 = rdt.calc_segDist(subcls.lon, subcls.lat, subcls.reach_id,
                                        subcls.facc, subcls.rch_ind6)
    # Updating type information.
    subcls.type6 = np.zeros(len(subcls.reach_id))
    for j in list(range(len(subcls.reach_id))):
        subcls.type6[j] = np.int(np.str(subcls.reach_id[j])[10:11])

    print('Creating Node Attributes')
    # Defining node attributes.
    subnodes.id, subnodes.x, subnodes.y, subnodes.len, subnodes.wse,\
     subnodes.wse_var, subnodes.wth, subnodes.wth_var, subnodes.facc, subnodes.nchan_max,\
     subnodes.nchan_mod, subnodes.reach_id,\
     subnodes.grod, subnodes.lakeflag, subnodes.grod_fid,\
     subnodes.hfalls_fid, subnodes.lake_id = rdt.node_attributes(subcls)

    # Creating width coeficient.
    subnodes.wth_coef = np.repeat(0.5, len(subnodes.id))

    print('Creating Reach Attributes')
    # Defining reach attributes.
    subreaches.id, subreaches.x, subreaches.y, subreaches.x_max,\
     subreaches.x_min, subreaches.y_max, subreaches.y_min, subreaches.len,\
     subreaches.wse, subreaches.wse_var, subreaches.wth, subreaches.wth_var,\
     subreaches.nchan_max, subreaches.nchan_mod, subreaches.rch_n_nodes,\
     subreaches.slope, subreaches.grod, subreaches.lakeflag,\
     subreaches.facc, subreaches.grod_fid,\
     subreaches.hfalls_fid, subreaches.lake_id = rdt.reach_attributes(subcls)

    print('Finding All Neighbors')
    subreaches.neighbors = rdt.neigboring_reaches(subcls, subreaches)

    print('Finding Basin Pathways')
    subcls.paths, subreaches.path, subreaches.path_order,\
     subreaches.path_dist = rdt.find_pathways(subcls, subreaches)

    print('Finding Distance From Outlet')
    subreaches.dist_out_prior = rdt.calc_basin_dist(subreaches)
    subreaches.dist_out = rdt.filter_basin_dist(subreaches)
    subnodes.dist_out = rdt.calc_node_dist_out(subreaches, subnodes)

    print('Defining Local Topology')
    # Defining intial topology.
    subreaches.n_rch_up, subreaches.n_rch_down, subreaches.rch_id_up,\
     subreaches.rch_id_down = rdt.local_topology(subcls, subreaches)
     # Filtering downstream neighbors
    subreaches.rch_id_up_filt, subreaches.n_rch_up_filt,\
     subreaches.rch_id_down_filt, subreaches.n_rch_down_filt = rdt.filter_neighbors(subreaches)

    print('Calculating SWOT Coverage')
    # Calculating swot coverage.
    subreaches.coverage, subreaches.orbits, subreaches.max_obs,\
	subreaches.median_obs, subreaches.mean_obs = rdt.swot_obs_percentage(subcls, subreaches, data)

    print('Formatting Centerline IDs')
    # Creating unique centerline ID.
    subcls.cl_id, subreaches.cl_id, \
     subnodes.cl_id = rdt.centerline_ids(subreaches, subnodes, subcls, start_id)
    start_id = np.max(subcls.cl_id)+1
    # Formating reach and node ids along high_resolution centerline.
    cl_nodes_id = rdt.format_cl_node_ids(subnodes, subcls)
    cl_rch_id = rdt.format_cl_rch_ids(subreaches, subcls)
    subcls.reach_id = np.insert(cl_rch_id, 0, subcls.reach_id, axis = 1)
    subcls.reach_id = subcls.reach_id[:,0:4]
    subcls.node_id =  np.insert(cl_nodes_id, 0, subcls.node_id, axis = 1)
    subcls.node_id = subcls.node_id[:,0:4]

    print('Aggregating Data')
    # Append current data to previous data.
    rdt.append_centerlines(centerlines, subcls, cnt)
    rdt.append_reaches(reaches, subreaches, cnt)
    rdt.append_nodes(nodes, subnodes, cnt)
    cnt = cnt+1

    end=time.time()
    print('Time to Create Reaches and Nodes: ' +
          str(np.round((end-start)/60, 2)) + ' min')

end_all=time.time()
print('Time to Finish All Reaches and Nodes: ' +
      str(np.round((end_all-start_all)/60, 2)) + ' min')

###############################################################################
################################ Writing Data #################################
###############################################################################

start = time.time()
reaches.rch_id_up = reaches.rch_id_up[:,0:4]
reaches.rch_id_down = reaches.rch_id_down[:,0:4]
reaches.n_rch_up[np.where(reaches.n_rch_up > 4)] = 4
reaches.n_rch_down[np.where(reaches.n_rch_down > 4)] = 4
reaches.nchan_mod[np.where(reaches.nchan_mod == 0)] = 1
reaches.nchan_max[np.where(reaches.nchan_max == 0)] = 1

# Creating filler variables for nodes.
nodes.wth_coef = np.repeat(0.5, len(nodes.id))
nodes.ext_dist_coef = np.repeat(20, len(nodes.id))
nodes.max_wth = np.zeros(len(nodes.id))
nodes.meand_len = np.zeros(len(nodes.id))
nodes.sinuosity = np.zeros(len(nodes.id))
nodes.river_name = np.repeat('NaN', len(nodes.id))
nodes.manual_add = np.zeros(len(nodes.id))
nodes.edit_flag = np.repeat('NaN', len(nodes.id))
nodes.manual_add[np.where(nodes.wth == 1)] = 1
nodes.nchan_mod[np.where(nodes.nchan_mod == 0)] = 1
nodes.nchan_max[np.where(nodes.nchan_max == 0)] = 1

# Creating filler variables for reaches.
reaches.iceflag = np.zeros((366, len(reaches.id)))
reaches.river_name = np.repeat('NaN', len(reaches.id))
reaches.max_wth = np.zeros(len(reaches.id))
reaches.low_slope = np.zeros(len(reaches.id))
reaches.edit_flag = np.repeat('NaN', len(reaches.id))
# discharge subgroup 1
reaches.h_break = np.full((4,len(reaches.id)), -9999.0)
reaches.w_break = np.full((4,len(reaches.id)), -9999.0)
reaches.hw_covariance = np.repeat(-9999., len(reaches.id))
reaches.h_err_stdev = np.repeat(-9999., len(reaches.id))
reaches.w_err_stdev = np.repeat(-9999., len(reaches.id))
reaches.h_w_nobs = np.repeat(-9999., len(reaches.id))
reaches.fit_coeffs = np.zeros((2, 3, len(reaches.id)))
reaches.fit_coeffs[np.where(reaches.fit_coeffs == 0)] = -9999.0
reaches.med_flow_area = np.repeat(-9999., len(reaches.id))
#MetroMan
reaches.metroman_ninf = np.repeat(-9999, len(reaches.id))
reaches.metroman_p = np.repeat(-9999, len(reaches.id))
reaches.metroman_abar = np.repeat(-9999, len(reaches.id))
reaches.metroman_abar_stdev = np.repeat(-9999, len(reaches.id))
reaches.metroman_ninf_stdev = np.repeat(-9999, len(reaches.id))
reaches.metroman_p_stdev = np.repeat(-9999, len(reaches.id))
reaches.metroman_ninf_p_cor = np.repeat(-9999, len(reaches.id))
reaches.metroman_ninf_abar_cor = np.repeat(-9999, len(reaches.id))
reaches.metroman_p_abar_cor = np.repeat(-9999, len(reaches.id))
reaches.metroman_sbQ_rel = np.repeat(-9999, len(reaches.id))
#HiDVI
reaches.hivdi_abar = np.repeat(-9999, len(reaches.id))
reaches.hivdi_alpha = np.repeat(-9999, len(reaches.id))
reaches.hivdi_beta = np.repeat(-9999, len(reaches.id))
reaches.hivdi_sbQ_rel = np.repeat(-9999, len(reaches.id))
#MOMMA
reaches.momma_b = np.repeat(-9999, len(reaches.id))
reaches.momma_h = np.repeat(-9999, len(reaches.id))
reaches.momma_save = np.repeat(-9999, len(reaches.id))
reaches.momma_sbQ_rel = np.repeat(-9999, len(reaches.id))
#SADS
reaches.sads_abar = np.repeat(-9999, len(reaches.id))
reaches.sads_n = np.repeat(-9999, len(reaches.id))
reaches.sads_sbQ_rel = np.repeat(-9999, len(reaches.id))
#BAM
reaches.bam_abar = np.repeat(-9999, len(reaches.id))
reaches.bam_n = np.repeat(-9999, len(reaches.id))
reaches.bam_sbQ_rel = np.repeat(-9999, len(reaches.id))
#SIC4DVar
reaches.sic4d_abar = np.repeat(-9999, len(reaches.id))
reaches.sic4d_n = np.repeat(-9999, len(reaches.id))
reaches.sic4d_sbQ_rel = np.repeat(-9999, len(reaches.id))

### TRANSPOSE CERTAIN VARIABLES:
centerlines.reach_id = centerlines.reach_id.T
centerlines.node_id = centerlines.node_id.T
nodes.cl_id = nodes.cl_id.T
reaches.cl_id = reaches.cl_id.T
reaches.rch_id_up = reaches.rch_id_up.T
reaches.rch_id_down = reaches.rch_id_down.T
reaches.orbits = reaches.orbits.T
reaches.coverage = reaches.coverage.T

# Writing netcdf.
wf.write_database_nc(centerlines, reaches, nodes, region, nc_outpath)
wf.write_swotobs_nc(centerlines, reaches, region, swot_outpath)

end = time.time()
print('Time to Write Reach and Node Output Files: ' +
      str(np.round((end-start)/60, 2)) + ' min')
