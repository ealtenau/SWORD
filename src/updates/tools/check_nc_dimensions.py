from __future__ import division
import os
main_dir = os.getcwd()
import numpy as np
import time
import netCDF4 as nc
from statistics import mode
import pandas as pd

###############################################################################
###############################################################################
###############################################################################

class Object(object):
    """
    FUNCTION:
        Creates class object to assign attributes to.
    """
    pass 

###############################################################################

def read_data(filename):

    centerlines = Object()
    nodes = Object()
    reaches = Object ()
    
    data = nc.Dataset(filename)
    
    centerlines.cl_id = data.groups['centerlines'].variables['cl_id'][:]
    centerlines.x = data.groups['centerlines'].variables['x'][:]
    centerlines.y = data.groups['centerlines'].variables['y'][:]
    centerlines.reach_id = data.groups['centerlines'].variables['reach_id'][:]
    centerlines.node_id = data.groups['centerlines'].variables['node_id'][:]
    
    nodes.id = data.groups['nodes'].variables['node_id'][:]
    nodes.cl_id = data.groups['nodes'].variables['cl_ids'][:]
    nodes.x = data.groups['nodes'].variables['x'][:]
    nodes.y = data.groups['nodes'].variables['y'][:]
    nodes.len = data.groups['nodes'].variables['node_length'][:]
    nodes.wse = data.groups['nodes'].variables['wse'][:]
    nodes.wse_var = data.groups['nodes'].variables['wse_var'][:]
    nodes.wth = data.groups['nodes'].variables['width'][:]
    nodes.wth_var = data.groups['nodes'].variables['width_var'][:]
    nodes.grod = data.groups['nodes'].variables['obstr_type'][:]
    nodes.grod_fid = data.groups['nodes'].variables['grod_id'][:]
    nodes.hfalls_fid = data.groups['nodes'].variables['hfalls_id'][:]
    nodes.nchan_max = data.groups['nodes'].variables['n_chan_max'][:]
    nodes.nchan_mod = data.groups['nodes'].variables['n_chan_mod'][:]
    nodes.dist_out = data.groups['nodes'].variables['dist_out'][:]
    nodes.reach_id = data.groups['nodes'].variables['reach_id'][:]
    nodes.facc = data.groups['nodes'].variables['facc'][:]
    nodes.lakeflag = data.groups['nodes'].variables['lakeflag'][:]
    nodes.wth_coef = data.groups['nodes'].variables['wth_coef'][:]
    nodes.ext_dist_coef = data.groups['nodes'].variables['ext_dist_coef'][:]
    nodes.max_wth = data.groups['nodes'].variables['max_width'][:]
    nodes.meand_len = data.groups['nodes'].variables['meander_length'][:]
    nodes.river_name = data.groups['nodes'].variables['river_name'][:]
    nodes.manual_add = data.groups['nodes'].variables['manual_add'][:]
    nodes.sinuosity = data.groups['nodes'].variables['sinuosity'][:]
    nodes.edit_flag = data.groups['nodes'].variables['edit_flag'][:]
    nodes.trib_flag = data.groups['nodes'].variables['trib_flag'][:]
    nodes.path_freq = data.groups['nodes'].variables['path_freq'][:]
    nodes.path_order = data.groups['nodes'].variables['path_order'][:]
    nodes.path_segs = data.groups['nodes'].variables['path_segs'][:]
    nodes.strm_order = data.groups['nodes'].variables['stream_order'][:]
    nodes.main_side = data.groups['nodes'].variables['main_side'][:]
    nodes.end_rch = data.groups['nodes'].variables['end_reach'][:]
    nodes.network = data.groups['nodes'].variables['network'][:]

    reaches.id = data.groups['reaches'].variables['reach_id'][:]
    reaches.cl_id = data.groups['reaches'].variables['cl_ids'][:]
    reaches.x = data.groups['reaches'].variables['x'][:]
    reaches.x_min = data.groups['reaches'].variables['x_min'][:]
    reaches.x_max = data.groups['reaches'].variables['x_max'][:]
    reaches.y = data.groups['reaches'].variables['y'][:]
    reaches.y_min = data.groups['reaches'].variables['y_min'][:]
    reaches.y_max = data.groups['reaches'].variables['y_max'][:]
    reaches.len = data.groups['reaches'].variables['reach_length'][:]
    reaches.wse = data.groups['reaches'].variables['wse'][:]
    reaches.wse_var = data.groups['reaches'].variables['wse_var'][:]
    reaches.wth = data.groups['reaches'].variables['width'][:]
    reaches.wth_var = data.groups['reaches'].variables['width_var'][:]
    reaches.slope = data.groups['reaches'].variables['slope'][:]
    reaches.rch_n_nodes = data.groups['reaches'].variables['n_nodes'][:]
    reaches.grod = data.groups['reaches'].variables['obstr_type'][:]
    reaches.grod_fid = data.groups['reaches'].variables['grod_id'][:]
    reaches.hfalls_fid = data.groups['reaches'].variables['hfalls_id'][:]
    reaches.lakeflag = data.groups['reaches'].variables['lakeflag'][:]
    reaches.nchan_max = data.groups['reaches'].variables['n_chan_max'][:]
    reaches.nchan_mod = data.groups['reaches'].variables['n_chan_mod'][:]
    reaches.dist_out = data.groups['reaches'].variables['dist_out'][:]
    reaches.n_rch_up = data.groups['reaches'].variables['n_rch_up'][:]
    reaches.n_rch_down = data.groups['reaches'].variables['n_rch_down'][:]
    reaches.rch_id_up = data.groups['reaches'].variables['rch_id_up'][:]
    reaches.rch_id_down = data.groups['reaches'].variables['rch_id_dn'][:]
    reaches.max_obs = data.groups['reaches'].variables['swot_obs'][:]
    reaches.orbits = data.groups['reaches'].variables['swot_orbits'][:]
    reaches.facc = data.groups['reaches'].variables['facc'][:]
    reaches.iceflag = data.groups['reaches'].variables['iceflag'][:]
    reaches.max_wth = data.groups['reaches'].variables['max_width'][:]
    reaches.river_name = data.groups['reaches'].variables['river_name'][:]
    reaches.low_slope = data.groups['reaches'].variables['low_slope_flag'][:]
    reaches.edit_flag= data.groups['reaches'].variables['edit_flag'][:]
    reaches.trib_flag = data.groups['reaches'].variables['trib_flag'][:]
    reaches.path_freq = data.groups['reaches'].variables['path_freq'][:]
    reaches.path_order = data.groups['reaches'].variables['path_order'][:]
    reaches.path_segs = data.groups['reaches'].variables['path_segs'][:]
    reaches.strm_order = data.groups['reaches'].variables['stream_order'][:]
    reaches.main_side = data.groups['reaches'].variables['main_side'][:]
    reaches.end_rch = data.groups['reaches'].variables['end_reach'][:]
    reaches.network = data.groups['reaches'].variables['network'][:]

    data.close()    

    return centerlines, nodes, reaches
    
###############################################################################    

def delete_rchs(reaches, rm_rch):
    for ind in list(range(len(rm_rch))):
        # print(ind, len(rm_rch))
        rch_ind = np.where(reaches.id == rm_rch[ind])[0]
        reaches.id = np.delete(reaches.id, rch_ind, axis = 0)
        reaches.cl_id = np.delete(reaches.cl_id, rch_ind, axis = 1)
        reaches.x = np.delete(reaches.x, rch_ind, axis = 0)
        reaches.x_min = np.delete(reaches.x_min, rch_ind, axis = 0)
        reaches.x_max = np.delete(reaches.x_max, rch_ind, axis = 0)
        reaches.y = np.delete(reaches.y, rch_ind, axis = 0)
        reaches.y_min = np.delete(reaches.y_min, rch_ind, axis = 0)
        reaches.y_max = np.delete(reaches.y_max, rch_ind, axis = 0)
        reaches.len = np.delete(reaches.len, rch_ind, axis = 0)
        reaches.wse = np.delete(reaches.wse, rch_ind, axis = 0)
        reaches.wse_var = np.delete(reaches.wse_var, rch_ind, axis = 0)
        reaches.wth = np.delete(reaches.wth, rch_ind, axis = 0)
        reaches.wth_var = np.delete(reaches.wth_var, rch_ind, axis = 0)
        reaches.slope = np.delete(reaches.slope, rch_ind, axis = 0)
        reaches.rch_n_nodes = np.delete(reaches.rch_n_nodes, rch_ind, axis = 0)
        reaches.grod = np.delete(reaches.grod, rch_ind, axis = 0)
        reaches.grod_fid = np.delete(reaches.grod_fid, rch_ind, axis = 0)
        reaches.hfalls_fid = np.delete(reaches.hfalls_fid, rch_ind, axis = 0)
        reaches.lakeflag = np.delete(reaches.lakeflag, rch_ind, axis = 0)
        reaches.nchan_max = np.delete(reaches.nchan_max, rch_ind, axis = 0)
        reaches.nchan_mod = np.delete(reaches.nchan_mod, rch_ind, axis = 0)
        reaches.dist_out = np.delete(reaches.dist_out, rch_ind, axis = 0)
        reaches.n_rch_up = np.delete(reaches.n_rch_up, rch_ind, axis = 0)
        reaches.n_rch_down = np.delete(reaches.n_rch_down, rch_ind, axis = 0)
        reaches.rch_id_up = np.delete(reaches.rch_id_up, rch_ind, axis = 1)
        reaches.rch_id_down = np.delete(reaches.rch_id_down, rch_ind, axis = 1)
        reaches.max_obs = np.delete(reaches.max_obs, rch_ind, axis = 0)
        reaches.orbits = np.delete(reaches.orbits, rch_ind, axis = 1)
        reaches.facc = np.delete(reaches.facc, rch_ind, axis = 0)
        reaches.iceflag = np.delete(reaches.iceflag, rch_ind, axis = 1)
        reaches.max_wth = np.delete(reaches.max_wth, rch_ind, axis = 0)
        reaches.river_name = np.delete(reaches.river_name, rch_ind, axis = 0)
        reaches.low_slope = np.delete(reaches.low_slope, rch_ind, axis = 0)
        reaches.edit_flag = np.delete(reaches.edit_flag, rch_ind, axis = 0)
        reaches.trib_flag = np.delete(reaches.trib_flag, rch_ind, axis = 0)
        reaches.path_freq = np.delete(reaches.path_freq, rch_ind, axis = 0)
        reaches.path_order = np.delete(reaches.path_order, rch_ind, axis = 0)
        reaches.path_segs = np.delete(reaches.path_segs, rch_ind, axis = 0)
        reaches.main_side = np.delete(reaches.main_side, rch_ind, axis = 0)
        reaches.strm_order = np.delete(reaches.strm_order, rch_ind, axis = 0)
        reaches.end_rch = np.delete(reaches.end_rch, rch_ind, axis = 0)
        reaches.network = np.delete(reaches.network, rch_ind, axis = 0)

###############################################################################
###############################################################################
###############################################################################

region = 'OC'
version = 'v18'
sword_dir = main_dir+'/data/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'

### read data. 
centerlines, nodes, reaches = read_data(sword_dir)

#checking dimensions
print('Cl Dimensions:', len(np.unique(centerlines.cl_id)), len(centerlines.cl_id))
print('Node Dimensions:', len(np.unique(centerlines.node_id[0,:])), len(np.unique(nodes.id)), len(nodes.id))
print('Rch Dimensions:', len(np.unique(centerlines.reach_id[0,:])), len(np.unique(nodes.reach_id)), len(np.unique(reaches.id)), len(reaches.id))
print('min node char len:', len(str(np.min(nodes.id))))
print('max node char len:', len(str(np.max(nodes.id))))
print('min reach char len:', len(str(np.min(reaches.id))))
print('max reach char len:', len(str(np.max(reaches.id))))
print('Edit flag values:', np.unique(reaches.edit_flag))
