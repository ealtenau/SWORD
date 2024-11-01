from __future__ import division
import os
if os.path.exists('/Users/ealtenau/Documents/SWORD_Dev/src/SWORD/database_updates/channel_additions/'):
    os.chdir('/Users/ealtenau/Documents/SWORD_Dev/src/SWORD/database_updates/channel_additions/')
else:
    os.chdir('/afs/cas.unc.edu/users/e/a/ealtenau/SWORD/post_swot_launch_updates/sword_adjustments/')
import cl_additions_reach_def_tools as rdt
# import Write_Database_Files as wf
import time
import numpy as np
import geopandas as gp
from shapely.geometry import Point
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from statistics import mode
import netCDF4 as nc

###############################################################################
##################### Defining Reach and Node Locations #######################
###############################################################################

start_all = time.time()
region = 'NA'

# Input file(s).
nc_file = '/Users/ealtenau/Documents/SWORD_Dev/update_requests/v17/'+region+'/channel_additions/'\
    +region.lower()+'_channel_additions3.nc'
sword_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/'+region.lower()+'_sword_v17.nc'

# Reading in data.
centerlines, nodes, reaches = rdt.read_data(sword_fn)
subcls = rdt.read_merge_netcdf(nc_file)

# Saving some key existing SWORD variables for id creation. 
sword_basins = np.array([int(str(r)[0:6]) for r in reaches.id])
sword_rch_nums = np.array([int(str(r)[6:10]) for r in reaches.id])

# Creating empty objects to fill with attributes.
subnodes = rdt.Object()
subreaches = rdt.Object()

# Making sure flow accumulation minimum isn't zero.
subcls.facc[np.where(subcls.facc == 0)[0]] = 0.001
# Cutting basins to 6 digits.
subcls.basins = np.array([int(str(ind)[0:6]) for ind in subcls.basins])

# Defining initial reaches. 
subcls.rch_id0 = np.copy(subcls.seg)
subcls.rch_len0 = np.zeros(len(subcls.lon))
unq_rchs = np.unique(subcls.rch_id0)
for r in list(range(len(unq_rchs))):
    rch = np.where(subcls.rch_id0 == unq_rchs[r])[0]
    length = np.max(subcls.dist[rch])-np.min(subcls.dist[rch])
    subcls.rch_len0[rch] = length

# Cutting Long Reaches. 
subcls.rch_id1, subcls.rch_len1 = rdt.cut_reaches(subcls.rch_id0,subcls.rch_len0,subcls.dist,subcls.ind,20000)
subcls.rch_id5 = subcls.rch_id1-(min(subcls.rch_id1)-1)
subcls.rch_len5 = np.copy(subcls.rch_len1)
subcls.rch_dist5 = np.copy(subcls.dist)
subcls.rch_ind5 = np.copy(subcls.ind)

# Creating Official Reach IDs.
print('Creating Reach ID')
subcls.reach_id = rdt.simple_rch_ids(subcls, sword_basins, sword_rch_nums)

# Creating Nodes.
print('Creating Nodes')
node_length = 200
subcls.node_id, subcls.node_len, subcls.node_num = rdt.node_reaches(subcls, node_length)

print('Creating Node Attributes')
# Defining node attributes.
subnodes.id, subnodes.x, subnodes.y, subnodes.len, subnodes.wse,\
    subnodes.wse_var, subnodes.wth, subnodes.wth_var, subnodes.facc, subnodes.nchan_max,\
    subnodes.nchan_mod, subnodes.reach_id,\
    subnodes.grod, subnodes.lakeflag, subnodes.grod_fid,\
    subnodes.hfalls_fid, subnodes.lake_id = rdt.basin_node_attributes(subcls.node_id,subcls.node_len,subcls.elv, subcls.wth,
                                                                      subcls.facc, subcls.nchan,subcls.lon, subcls.lat,
                                                                      subcls.reach_id, subcls.grod,subcls.lake, subcls.grod_fid,
                                                                      subcls.hfalls_fid, subcls.lake_id)    

# Node filler variables. 
subnodes.ext_dist_coef = np.repeat(5,len(subnodes.id))
subnodes.wth_coef = np.repeat(0.5, len(subnodes.id))
subnodes.meand_len = np.repeat(-9999, len(subnodes.id))
subnodes.river_name = np.repeat('NODATA', len(subnodes.id))
subnodes.manual_add = np.repeat(1, len(subnodes.id))
subnodes.sinuosity = np.repeat(-9999, len(subnodes.id))
subnodes.edit_flag = np.repeat('7', len(subnodes.id))
subnodes.trib_flag = np.repeat(0, len(subnodes.id))
subnodes.max_wth = np.repeat(-9999, len(subnodes.id))

print('Creating Reach Attributes')
# Defining reach attributes.
subcls.rch_len6 = np.copy(subcls.rch_len5)
subcls.rch_dist6 = np.copy(subcls.rch_dist5)
subcls.rch_ind6 = np.copy(subcls.rch_ind5)
subreaches.id, subreaches.x, subreaches.y, subreaches.x_max,\
    subreaches.x_min, subreaches.y_max, subreaches.y_min, subreaches.len,\
    subreaches.wse, subreaches.wse_var, subreaches.wth, subreaches.wth_var,\
    subreaches.nchan_max, subreaches.nchan_mod, subreaches.rch_n_nodes,\
    subreaches.slope, subreaches.grod, subreaches.lakeflag,\
    subreaches.facc, subreaches.grod_fid,\
    subreaches.hfalls_fid, subreaches.lake_id = rdt.reach_attributes(subcls)

print('Calculating SWOT Coverage')
# Calculating swot coverage.
subreaches.coverage, subreaches.orbits, subreaches.max_obs,\
    subreaches.median_obs, subreaches.mean_obs = rdt.swot_obs_percentage(subcls, subreaches)

# Filling in topology variables with fill values. 
subreaches.dist_out = np.repeat(-9999, len(subreaches.id))
subnodes.dist_out = np.repeat(-9999, len(subnodes.id))
subreaches.n_rch_up = np.zeros(len(subreaches.id))
subreaches.n_rch_down = np.zeros(len(subreaches.id))
subreaches.rch_id_up = np.zeros([len(subreaches.id),4])
subreaches.rch_id_down = np.zeros([len(subreaches.id),4])
subreaches.iceflag = np.zeros([len(subreaches.id),366])
subreaches.iceflag[:,:] = -9999
subreaches.max_wth = np.repeat(-9999, len(subreaches.id))
subreaches.river_name = np.repeat('NODATA', len(subreaches.id))
subreaches.low_slope = np.repeat(0, len(subreaches.id))
subreaches.edit_flag = np.repeat('7', len(subreaches.id))
subreaches.trib_flag = np.repeat(0, len(subreaches.id))

print('Formatting Centerline IDs')
# Creating unique centerline ID.
start_id = np.max(centerlines.cl_id)+1
subcls.id = subcls.rch_ind6+start_id
subcls.cl_id = subcls.rch_ind6+start_id
subreaches.cl_id, subnodes.cl_id = rdt.centerline_ids(subreaches, subnodes, subcls)
# Formating reach and node ids along high_resolution centerline.
cl_nodes_id = rdt.format_cl_node_ids(subnodes, subcls)
cl_rch_id = rdt.format_cl_rch_ids(subreaches, subcls)
subcls.reach_id = np.insert(cl_rch_id, 0, subcls.reach_id, axis = 1)
subcls.reach_id = subcls.reach_id[:,0:4]
subcls.node_id =  np.insert(cl_nodes_id, 0, subcls.node_id, axis = 1)
subcls.node_id = subcls.node_id[:,0:4]
# Transpose variables.
subcls.reach_id = subcls.reach_id.T; subcls.node_id = subcls.node_id.T
subreaches.cl_id = subreaches.cl_id.T; subnodes.cl_id = subnodes.cl_id.T
subreaches.rch_id_up = subreaches.rch_id_up.T; subreaches.rch_id_down = subreaches.rch_id_down.T
subreaches.orbits = subreaches.orbits.T
subreaches.iceflag = subreaches.iceflag.T

#################################################################################
#################################################################################

subnodes.path_freq = np.repeat(-9999, len(subnodes.id))
subnodes.path_order = np.repeat(-9999, len(subnodes.id))
subnodes.path_segs = np.repeat(-9999, len(subnodes.id))
subnodes.main_side = np.repeat(1, len(subnodes.id))
subnodes.strm_order = np.repeat(-9999, len(subnodes.id))
subnodes.network = np.repeat(0, len(subnodes.id))
subnodes.end_rch = np.repeat(0, len(subnodes.id))

subreaches.path_freq = np.repeat(-9999, len(subreaches.id))
subreaches.path_order = np.repeat(-9999, len(subreaches.id))
subreaches.path_segs = np.repeat(-9999, len(subreaches.id))
subreaches.main_side = np.repeat(1, len(subreaches.id))
subreaches.strm_order = np.repeat(-9999, len(subreaches.id))
subreaches.network = np.repeat(0, len(subreaches.id))
subreaches.end_rch = np.repeat(0, len(subreaches.id))

#################################################################################
#################################################################################

# Append new data to existing data. 
rdt.append_data(centerlines, nodes, reaches, 
                subcls, subnodes, subreaches)

###############################################################################
###############################################################################
### Filler variables
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

rdt.write_database_nc(centerlines, reaches, nodes, region, sword_fn)

print('Cl Dimensions:', len(np.unique(centerlines.cl_id)), len(centerlines.cl_id))
print('Node Dimensions:', len(np.unique(centerlines.node_id[0,:])), len(np.unique(nodes.id)), len(nodes.id))
print('Rch Dimensions:', len(np.unique(centerlines.reach_id[0,:])), len(np.unique(nodes.reach_id)), len(np.unique(reaches.id)),len(reaches.id))

end_all=time.time()
print('Time to Finish All Reaches and Nodes: ' +
      str(np.round((end_all-start_all)/60, 2)) + ' min')



# z = np.where(subcls.basins == 0)[0]
# plt.scatter(subcls.lon, subcls.lat)
# plt.scatter(subcls.lon[z], subcls.lat[z], c='red')
# plt.show()