"""

SWORD Utilities (sword_tools.py)
=======================================

Utilities for reading, writing, managing, and processing the SWORD 
database. SWORD formats include netCDF, shapefile, and geopackage. 

"""

from __future__ import division
import os
import numpy as np
import time
import netCDF4 as nc
from statistics import mode
import pandas as pd

###############################################################################

class Object(object):
    """
    FUNCTION:
        Creates an empty class object to assign SWORD attributes to.
    """
    pass 

###############################################################################

def read_nc(filename):

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

def write_nc(centerlines, reaches, nodes, region, outfile):

    """
    FUNCTION
    --------
        Outputs the SWOT River Database (SWORD) information in netcdf
        format. The file contains attributes for the high-resolution centerline,
        nodes, and reaches.

    INPUTS
    ------
        centerlines: Object containing lcation and attribute information
            along the high-resolution centerline.
        reaches: Object containing lcation and attribute information for
            each reach.
        nodes: Object containing lcation and attribute information for
            each node.
        outfile -- Path for netcdf to be written.

    OUTPUTS
    -------
        SWORD NetCDF: NetCDF file containing attributes for the high-resolution
            centerline, node, and reach locations.
    """

    start = time.time()

    # global attributes
    root_grp = nc.Dataset(outfile, 'w', format='NETCDF4')
    root_grp.x_min = np.min(centerlines.x)
    root_grp.x_max = np.max(centerlines.x)
    root_grp.y_min = np.min(centerlines.y)
    root_grp.y_max = np.max(centerlines.y)
    root_grp.Name = region
    root_grp.production_date = time.strftime("%d-%b-%Y %H:%M:%S", time.gmtime()) #utc time

    # groups
    cl_grp = root_grp.createGroup('centerlines')
    node_grp = root_grp.createGroup('nodes')
    rch_grp = root_grp.createGroup('reaches')
    # subgroups
    sub_grp1 = rch_grp.createGroup('area_fits')
    sub_grp2 = rch_grp.createGroup('discharge_models')
    # discharge subgroups
    qgrp1 = sub_grp2.createGroup('unconstrained')
    qgrp2 = sub_grp2.createGroup('constrained')
    # unconstrained discharge models
    ucmod1 = qgrp1.createGroup('MetroMan')
    ucmod2 = qgrp1.createGroup('BAM')
    ucmod3 = qgrp1.createGroup('HiVDI')
    ucmod4 = qgrp1.createGroup('MOMMA')
    ucmod5 = qgrp1.createGroup('SADS')
    ucmod6 = qgrp1.createGroup('SIC4DVar')
    # constrained discharge models
    cmod1 = qgrp2.createGroup('MetroMan')
    cmod2 = qgrp2.createGroup('BAM')
    cmod3 = qgrp2.createGroup('HiVDI')
    cmod4 = qgrp2.createGroup('MOMMA')
    cmod5 = qgrp2.createGroup('SADS')
    cmod6 = qgrp2.createGroup('SIC4DVar')

    # dimensions
    #root_grp.createDimension('d1', 2)
    cl_grp.createDimension('num_points', len(centerlines.cl_id))
    cl_grp.createDimension('num_domains', 4)

    node_grp.createDimension('num_nodes', len(nodes.id))
    node_grp.createDimension('num_ids', 2)

    rch_grp.createDimension('num_reaches', len(reaches.id))
    rch_grp.createDimension('num_ids', 2)
    rch_grp.createDimension('num_domains', 4)
    rch_grp.createDimension('julian_day', 366)
    rch_grp.createDimension('orbits', 75)
    sub_grp1.createDimension('nCoeffs', 2)
    sub_grp1.createDimension('nReg', 3)
    sub_grp1.createDimension('num_domains', 4)

    # centerline variables
    cl_id = cl_grp.createVariable(
        'cl_id', 'i8', ('num_points',), fill_value=-9999.)
    cl_x = cl_grp.createVariable(
        'x', 'f8', ('num_points',), fill_value=-9999.)
    cl_x.units = 'degrees east'
    cl_y = cl_grp.createVariable(
        'y', 'f8', ('num_points',), fill_value=-9999.)
    cl_y.units = 'degrees north'
    reach_id = cl_grp.createVariable(
        'reach_id', 'i8', ('num_domains','num_points'), fill_value=-9999.)
    reach_id.format = 'CBBBBBRRRRT'
    node_id = cl_grp.createVariable(
        'node_id', 'i8', ('num_domains','num_points'), fill_value=-9999.)
    node_id.format = 'CBBBBBRRRRNNNT'

    # node variables
    Node_ID = node_grp.createVariable(
        'node_id', 'i8', ('num_nodes',), fill_value=-9999.)
    Node_ID.format = 'CBBBBBRRRRNNNT'
    node_cl_id = node_grp.createVariable(
        'cl_ids', 'i8', ('num_ids','num_nodes'), fill_value=-9999.)
    node_x = node_grp.createVariable(
        'x', 'f8', ('num_nodes',), fill_value=-9999.)
    node_x.units = 'degrees east'
    node_y = node_grp.createVariable(
        'y', 'f8', ('num_nodes',), fill_value=-9999.)
    node_y.units = 'degrees north'
    node_len = node_grp.createVariable(
        'node_length', 'f8', ('num_nodes',), fill_value=-9999.)
    node_len.units = 'meters'
    node_rch_id = node_grp.createVariable(
        'reach_id', 'i8', ('num_nodes',), fill_value=-9999.)
    node_rch_id.format = 'CBBBBBRRRRT'
    node_wse = node_grp.createVariable(
        'wse', 'f8', ('num_nodes',), fill_value=-9999.)
    node_wse.units = 'meters'
    node_wse_var = node_grp.createVariable(
        'wse_var', 'f8', ('num_nodes',), fill_value=-9999.)
    node_wse_var.units = 'meters^2'
    node_wth = node_grp.createVariable(
        'width', 'f8', ('num_nodes',), fill_value=-9999.)
    node_wth.units = 'meters'
    node_wth_var = node_grp.createVariable(
        'width_var', 'f8', ('num_nodes',), fill_value=-9999.)
    node_wth_var.units = 'meters^2'
    node_chan_max = node_grp.createVariable(
        'n_chan_max', 'i4', ('num_nodes',), fill_value=-9999.)
    node_chan_mod = node_grp.createVariable(
        'n_chan_mod', 'i4', ('num_nodes',), fill_value=-9999.)
    node_grod_id = node_grp.createVariable(
        'obstr_type', 'i4', ('num_nodes',), fill_value=-9999.)
    node_grod_fid = node_grp.createVariable(
        'grod_id', 'i8', ('num_nodes',), fill_value=-9999.)
    node_hfalls_fid = node_grp.createVariable(
        'hfalls_id', 'i8', ('num_nodes',), fill_value=-9999.)
    node_dist_out = node_grp.createVariable(
        'dist_out', 'f8', ('num_nodes',), fill_value=-9999.)
    node_dist_out.units = 'meters'
    node_wth_coef = node_grp.createVariable(
        'wth_coef', 'f8', ('num_nodes',), fill_value=-9999.)
    node_ext_dist_coef = node_grp.createVariable(
        'ext_dist_coef', 'f8', ('num_nodes',), fill_value=-9999.)
    node_facc = node_grp.createVariable(
        'facc', 'f8', ('num_nodes',), fill_value=-9999.)
    node_facc.units = 'km^2'
    node_lakeflag = node_grp.createVariable(
        'lakeflag', 'i8', ('num_nodes',), fill_value=-9999.)
    #node_lake_id = node_grp.createVariable(
        #'lake_id', 'i8', ('num_nodes',), fill_value=-9999.)
    node_max_wth = node_grp.createVariable(
        'max_width', 'f8', ('num_nodes',), fill_value=-9999.)
    node_max_wth.units = 'meters'
    node_meand_len = node_grp.createVariable(
        'meander_length', 'f8', ('num_nodes',), fill_value=-9999.)
    node_sinuosity = node_grp.createVariable(
        'sinuosity', 'f8', ('num_nodes',), fill_value=-9999.)
    node_manual_add = node_grp.createVariable(
        'manual_add', 'i4', ('num_nodes',), fill_value=-9999.)
    node_river_name = node_grp.createVariable(
        'river_name', 'S50', ('num_nodes',))
    node_river_name._Encoding = 'ascii'
    node_edit_flag = node_grp.createVariable(
        'edit_flag',  'S50', ('num_nodes',))
    node_edit_flag._Encoding = 'ascii'
    node_trib_flag = node_grp.createVariable(
        'trib_flag', 'i4', ('num_nodes',), fill_value=-9999.)
    node_path_freq = node_grp.createVariable(
        'path_freq', 'i8', ('num_nodes',), fill_value=-9999.)
    node_path_order = node_grp.createVariable(
        'path_order', 'i8', ('num_nodes',), fill_value=-9999.)
    node_path_seg = node_grp.createVariable(
        'path_segs', 'i8', ('num_nodes',), fill_value=-9999.)
    node_strm_order = node_grp.createVariable(
        'stream_order', 'i4', ('num_nodes',), fill_value=-9999.)
    node_main_side = node_grp.createVariable(
        'main_side', 'i4', ('num_nodes',), fill_value=-9999.)
    node_end_rch = node_grp.createVariable(
        'end_reach', 'i4', ('num_nodes',), fill_value=-9999.)
    node_network = node_grp.createVariable(
        'network', 'i4', ('num_nodes',), fill_value=-9999.)

    # reach variables
    Reach_ID = rch_grp.createVariable(
        'reach_id', 'i8', ('num_reaches',), fill_value=-9999.)
    Reach_ID.format = 'CBBBBBRRRRT'
    rch_cl_id = rch_grp.createVariable(
        'cl_ids', 'i8', ('num_ids','num_reaches'), fill_value=-9999.)
    rch_x = rch_grp.createVariable(
        'x', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_x.units = 'degrees east'
    rch_x_min = rch_grp.createVariable(
        'x_min', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_x_min.units = 'degrees east'
    rch_x_max = rch_grp.createVariable(
        'x_max', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_x_max.units = 'degrees east'
    rch_y = rch_grp.createVariable(
        'y', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_y.units = 'degrees north'
    rch_y_min = rch_grp.createVariable(
        'y_min', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_y_min.units = 'degrees north'
    rch_y_max = rch_grp.createVariable(
        'y_max', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_y_max.units = 'degrees north'
    rch_len = rch_grp.createVariable(
        'reach_length', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_len.units = 'meters'
    num_nodes = rch_grp.createVariable(
        'n_nodes', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_wse = rch_grp.createVariable(
        'wse', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_wse.units = 'meters'
    rch_wse_var = rch_grp.createVariable(
        'wse_var', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_wse_var.units = 'meters^2'
    rch_wth = rch_grp.createVariable(
        'width', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_wth.units = 'meters'
    rch_wth_var = rch_grp.createVariable(
        'width_var', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_wth_var.units = 'meters^2'
    rch_facc = rch_grp.createVariable(
        'facc', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_facc.units = 'km^2'
    rch_chan_max = rch_grp.createVariable(
        'n_chan_max', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_chan_mod = rch_grp.createVariable(
        'n_chan_mod', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_grod_id = rch_grp.createVariable(
        'obstr_type', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_grod_fid = rch_grp.createVariable(
        'grod_id', 'i8', ('num_reaches',), fill_value=-9999.)
    rch_hfalls_fid = rch_grp.createVariable(
        'hfalls_id', 'i8', ('num_reaches',), fill_value=-9999.)
    rch_slope = rch_grp.createVariable(
        'slope', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_slope.units = 'meters/kilometers'
    rch_dist_out = rch_grp.createVariable(
        'dist_out', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_dist_out.units = 'meters'
    n_rch_up = rch_grp.createVariable(
        'n_rch_up', 'i4', ('num_reaches',), fill_value=-9999.)
    n_rch_down= rch_grp.createVariable(
        'n_rch_down', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_id_up = rch_grp.createVariable(
        'rch_id_up', 'i8', ('num_domains','num_reaches'), fill_value=-9999.)
    rch_id_down = rch_grp.createVariable(
        'rch_id_dn', 'i8', ('num_domains','num_reaches'), fill_value=-9999.)
    rch_lakeflag = rch_grp.createVariable(
        'lakeflag', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_iceflag = rch_grp.createVariable(
        'iceflag', 'i4', ('julian_day','num_reaches'), fill_value=-9999.)
    #rch_lake_id = rch_grp.createVariable(
        #'lake_id', 'i8', ('num_reaches',), fill_value=-9999.)
    rch_swot_obs = rch_grp.createVariable(
        'swot_obs', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_orbits = rch_grp.createVariable(
        'swot_orbits', 'i8', ('orbits','num_reaches'), fill_value=-9999.)
    rch_river_name = rch_grp.createVariable(
        'river_name', 'S50', ('num_reaches',))
    rch_river_name._Encoding = 'ascii'
    rch_max_wth = rch_grp.createVariable(
        'max_width', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_max_wth.units = 'meters'
    rch_low_slope = rch_grp.createVariable(
        'low_slope_flag', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_edit_flag = rch_grp.createVariable(
        'edit_flag', 'S50', ('num_reaches',), fill_value=-9999.)
    rch_edit_flag._Encoding = 'ascii'
    rch_trib_flag = rch_grp.createVariable(
        'trib_flag', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_path_freq = rch_grp.createVariable(
        'path_freq', 'i8', ('num_reaches',), fill_value=-9999.)
    rch_path_order = rch_grp.createVariable(
        'path_order', 'i8', ('num_reaches',), fill_value=-9999.)
    rch_path_seg = rch_grp.createVariable(
        'path_segs', 'i8', ('num_reaches',), fill_value=-9999.)
    rch_strm_order = rch_grp.createVariable(
        'stream_order', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_main_side = rch_grp.createVariable(
        'main_side', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_end_rch = rch_grp.createVariable(
        'end_reach', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_network = rch_grp.createVariable(
        'network', 'i4', ('num_reaches',), fill_value=-9999.)
    # subgroup 1 - 'area_fits'
    h_break = sub_grp1.createVariable(
        'h_break', 'f8', ('num_domains','num_reaches'), fill_value=-9999.)
    h_break.units = 'meters'
    w_break = sub_grp1.createVariable(
        'w_break', 'f8', ('num_domains','num_reaches'), fill_value=-9999.)
    w_break.units = 'meters'
    h_variance = sub_grp1.createVariable(
        'h_variance', 'f8', ('num_reaches',), fill_value=-9999.)
    h_variance.units = 'meters^2'
    w_variance = sub_grp1.createVariable(
        'w_variance', 'f8', ('num_reaches',), fill_value=-9999.)
    w_variance.units = 'meters^2'
    hw_covariance = sub_grp1.createVariable(
        'hw_covariance', 'f8', ('num_reaches',), fill_value=-9999.)
    hw_covariance.units = 'meters^2'
    h_err_stdev = sub_grp1.createVariable(
        'h_err_stdev', 'f8', ('num_reaches',), fill_value=-9999.)
    h_err_stdev.units = 'meters'
    w_err_stdev = sub_grp1.createVariable(
        'w_err_stdev', 'f8', ('num_reaches',), fill_value=-9999.)
    w_err_stdev.units = 'meters'
    h_w_nobs = sub_grp1.createVariable(
        'h_w_nobs', 'f8', ('num_reaches',), fill_value=-9999.)
    fit_coeffs = sub_grp1.createVariable(
        'fit_coeffs', 'f8', ('nCoeffs','nReg','num_reaches'), fill_value=-9999.)
    med_flow_area = sub_grp1.createVariable(
        'med_flow_area', 'f8', ('num_reaches',), fill_value=-9999.)
    # unconstrained discharge subgroups
    # MetroMan (ucmod1)
    uc_metroman_Abar = ucmod1.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_Abar.units = 'meters'
    uc_metroman_Abar_stdev = ucmod1.createVariable(
        'Abar_stdev', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_Abar_stdev.units = 'meters'
    uc_metroman_ninf = ucmod1.createVariable(
        'ninf', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_ninf_stdev = ucmod1.createVariable(
        'ninf_stdev', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_p = ucmod1.createVariable(
        'p', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_p_stdev = ucmod1.createVariable(
        'p_stdev', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_ninf_p_cor = ucmod1.createVariable(
        'ninf_p_cor', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_p_Abar_cor = ucmod1.createVariable(
        'p_Abar_cor', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_ninf_Abar_cor = ucmod1.createVariable(
        'ninf_Abar_cor', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_sbQ_rel = ucmod1.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # BAM (ucmod2)
    uc_bam_Abar = ucmod2.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_bam_Abar.units = 'meters'
    uc_bam_n = ucmod2.createVariable(
        'n', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_bam_sbQ_rel = ucmod2.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # HiVDI (ucmod3)
    uc_hivdi_Abar = ucmod3.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_hivdi_Abar.units = 'meters'
    uc_hivdi_alpha = ucmod3.createVariable(
        'alpha', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_hivdi_beta = ucmod3.createVariable(
        'beta', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_hivdi_sbQ_rel = ucmod3.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # MOMMA (ucmod4)
    uc_momma_B = ucmod4.createVariable(
        'B', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_momma_B.units = 'meters'
    uc_momma_H = ucmod4.createVariable(
        'H', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_momma_H.units = 'meters'
    uc_momma_Save = ucmod4.createVariable(
        'Save', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_momma_Save.units = 'meters/kilometers'
    uc_momma_sbQ_rel = ucmod4.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # SADS (ucmod5)
    uc_sads_Abar = ucmod5.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_sads_Abar.units = 'meters'
    uc_sads_n = ucmod5.createVariable(
        'n', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_sads_sbQ_rel = ucmod5.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # SIC4DVar (ucmod6)
    uc_sic4d_Abar = ucmod6.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_sic4d_Abar.units = 'meters'
    uc_sic4d_n = ucmod6.createVariable(
        'n', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_sic4d_sbQ_rel = ucmod6.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # constrained discharge subgroups
    # MetroMan (cmod1)
    c_metroman_Abar = cmod1.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_Abar.units = 'meters'
    c_metroman_Abar_stdev = cmod1.createVariable(
        'Abar_stdev', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_Abar_stdev.units = 'meters'
    c_metroman_ninf = cmod1.createVariable(
        'ninf', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_ninf_stdev = cmod1.createVariable(
        'ninf_stdev', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_p = cmod1.createVariable(
        'p', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_p_stdev = cmod1.createVariable(
        'p_stdev', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_ninf_p_cor = cmod1.createVariable(
        'ninf_p_cor', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_p_Abar_cor = cmod1.createVariable(
        'p_Abar_cor', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_ninf_Abar_cor = cmod1.createVariable(
        'ninf_Abar_cor', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_sbQ_rel = cmod1.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # BAM (cmod2)
    c_bam_Abar = cmod2.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    c_bam_Abar.units = 'meters'
    c_bam_n = cmod2.createVariable(
        'n', 'f8', ('num_reaches',), fill_value=-9999.)
    c_bam_sbQ_rel = cmod2.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # HiDVI (cmod3)
    c_hivdi_Abar = cmod3.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    c_hivdi_Abar.units = 'meters'
    c_hivdi_alpha = cmod3.createVariable(
        'alpha', 'f8', ('num_reaches',), fill_value=-9999.)
    c_hivdi_beta = cmod3.createVariable(
        'beta', 'f8', ('num_reaches',), fill_value=-9999.)
    c_hivdi_sbQ_rel = cmod3.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # MOMMA (cmod4)
    c_momma_B = cmod4.createVariable(
        'B', 'f8', ('num_reaches',), fill_value=-9999.)
    c_momma_B.units = 'meters'
    c_momma_H = cmod4.createVariable(
        'H', 'f8', ('num_reaches',), fill_value=-9999.)
    c_momma_H.units = 'meters'
    c_momma_Save = cmod4.createVariable(
        'Save', 'f8', ('num_reaches',), fill_value=-9999.)
    c_momma_Save.units = 'meters/kilometers'
    c_momma_sbQ_rel = cmod4.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # SADS (cmod5)
    c_sads_Abar = cmod5.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    c_sads_Abar.units = 'meters'
    c_sads_n = cmod5.createVariable(
        'n', 'f8', ('num_reaches',), fill_value=-9999.)
    c_sads_sbQ_rel = cmod5.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # SIC4DVar (cmod6)
    c_sic4d_Abar = cmod6.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    c_sic4d_Abar.units = 'meters'
    c_sic4d_n = cmod6.createVariable(
        'n', 'f8', ('num_reaches',), fill_value=-9999.)
    c_sic4d_sbQ_rel = cmod6.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)

    # saving data
    print("saving nc")
    cl_sort = np.argsort(centerlines.cl_id)
    node_sort = np.argsort(nodes.id)
    rch_sort = np.argsort(reaches.id)

    # centerline data
    cl_id[:] = centerlines.cl_id[cl_sort]
    cl_x[:] = centerlines.x[cl_sort]
    cl_y[:] = centerlines.y[cl_sort]
    reach_id[:,:] = centerlines.reach_id[:,cl_sort]
    node_id[:,:] = centerlines.node_id[:,cl_sort]

    # node data
    Node_ID[:] = nodes.id[node_sort]
    node_cl_id[:,:] = nodes.cl_id[:,node_sort]
    node_x[:] = nodes.x[node_sort]
    node_y[:] = nodes.y[node_sort]
    node_len[:] = nodes.len[node_sort]
    node_rch_id[:] = nodes.reach_id[node_sort]
    node_wse[:] = nodes.wse[node_sort]
    node_wse_var[:] = nodes.wse_var[node_sort]
    node_wth[:] = nodes.wth[node_sort]
    node_wth_var[:] = nodes.wth_var[node_sort]
    node_chan_max[:] = nodes.nchan_max[node_sort]
    node_chan_mod[:] = nodes.nchan_mod[node_sort]
    node_grod_id[:] = nodes.grod[node_sort]
    node_grod_fid[:] = nodes.grod_fid[node_sort]
    node_hfalls_fid[:] = nodes.hfalls_fid[node_sort]
    node_dist_out[:] = nodes.dist_out[node_sort]
    node_wth_coef[:] = nodes.wth_coef[node_sort]
    node_ext_dist_coef[:] = nodes.ext_dist_coef[node_sort]
    node_facc[:] = nodes.facc[node_sort]
    node_lakeflag[:] = nodes.lakeflag[node_sort]
    #node_lake_id[:] = nodes.lake_id[node_sort]
    node_max_wth[:] = nodes.max_wth[node_sort]
    node_meand_len[:] = nodes.meand_len[node_sort]
    node_sinuosity[:] = nodes.sinuosity[node_sort]
    node_river_name[:] = nodes.river_name[node_sort]
    node_manual_add[:] = nodes.manual_add[node_sort]
    node_edit_flag[:] = nodes.edit_flag[node_sort]
    node_trib_flag[:] = nodes.trib_flag[node_sort]
    node_path_freq[:] = nodes.path_freq[node_sort]
    node_path_order[:] = nodes.path_order[node_sort]
    node_path_seg[:] = nodes.path_segs[node_sort]
    node_strm_order[:] = nodes.strm_order[node_sort]
    node_main_side[:] = nodes.main_side[node_sort]
    node_end_rch[:] = nodes.end_rch[node_sort]
    node_network[:] = nodes.network[node_sort]

    # reach data
    Reach_ID[:] = reaches.id[rch_sort]
    rch_cl_id[:,:] = reaches.cl_id[:,rch_sort]
    rch_x[:] = reaches.x[rch_sort]
    rch_x_min[:] = reaches.x_min[rch_sort]
    rch_x_max[:] = reaches.x_max[rch_sort]
    rch_y[:] = reaches.y[rch_sort]
    rch_y_min[:] = reaches.y_min[rch_sort]
    rch_y_max[:] = reaches.y_max[rch_sort]
    rch_len[:] = reaches.len[rch_sort]
    num_nodes[:] = reaches.rch_n_nodes[rch_sort]
    rch_wse[:] = reaches.wse[rch_sort]
    rch_wse_var[:] = reaches.wse_var[rch_sort]
    rch_wth[:] = reaches.wth[rch_sort]
    rch_wth_var[:] = reaches.wth_var[rch_sort]
    rch_facc[:] = reaches.facc[rch_sort]
    rch_chan_max[:] = reaches.nchan_max[rch_sort]
    rch_chan_mod[:] = reaches.nchan_mod[rch_sort]
    rch_grod_id[:] = reaches.grod[rch_sort]
    rch_grod_fid[:] = reaches.grod_fid[rch_sort]
    rch_hfalls_fid[:] = reaches.hfalls_fid[rch_sort]
    rch_slope[:] = reaches.slope[rch_sort]
    rch_dist_out[:] = reaches.dist_out[rch_sort]
    n_rch_up[:] = reaches.n_rch_up[rch_sort]
    n_rch_down[:] = reaches.n_rch_down[rch_sort]
    rch_id_up[:,:] = reaches.rch_id_up[:,rch_sort]
    rch_id_down[:,:] = reaches.rch_id_down[:,rch_sort]
    rch_lakeflag[:] = reaches.lakeflag[rch_sort]
    rch_iceflag[:,:] = reaches.iceflag[:,rch_sort]
    #rch_lake_id[:] = reaches.lake_id[rch_sort]
    rch_swot_obs[:] = reaches.max_obs[rch_sort]
    rch_orbits[:,:] = reaches.orbits[:,rch_sort]
    rch_river_name[:] = reaches.river_name[rch_sort]
    rch_max_wth[:] = reaches.max_wth[rch_sort]
    rch_low_slope[:] = reaches.low_slope[rch_sort]
    rch_edit_flag[:] = reaches.edit_flag[rch_sort]
    rch_trib_flag[:] = reaches.trib_flag[rch_sort]
    rch_path_freq[:] = reaches.path_freq[rch_sort]
    rch_path_order[:] = reaches.path_order[rch_sort]
    rch_path_seg[:] = reaches.path_segs[rch_sort]
    rch_strm_order[:] = reaches.strm_order[rch_sort]
    rch_main_side[:] = reaches.main_side[rch_sort]
    rch_end_rch[:] = reaches.end_rch[rch_sort]
    rch_network[:] = reaches.network[rch_sort]
    # subgroup1 - area fits
    h_break[:,:] = reaches.h_break[:,rch_sort]
    w_break[:,:] = reaches.w_break[:,rch_sort]
    h_variance[:] = reaches.wse_var[rch_sort]
    w_variance[:] = reaches.wth_var[rch_sort]
    hw_covariance[:] = reaches.hw_covariance[rch_sort]
    h_err_stdev[:] = reaches.h_err_stdev[rch_sort]
    w_err_stdev[:] = reaches.w_err_stdev[rch_sort]
    h_w_nobs[:] = reaches.h_w_nobs[rch_sort]
    fit_coeffs[:,:,:] = reaches.fit_coeffs[:,:,rch_sort]
    med_flow_area[:] = reaches.med_flow_area[rch_sort]
    # ucmod1
    uc_metroman_Abar[:] = reaches.metroman_abar[rch_sort]
    uc_metroman_ninf[:] = reaches.metroman_ninf[rch_sort]
    uc_metroman_p[:] = reaches.metroman_p[rch_sort]
    uc_metroman_Abar_stdev[:] = reaches.metroman_abar_stdev[rch_sort]
    uc_metroman_ninf_stdev[:] = reaches.metroman_ninf_stdev[rch_sort]
    uc_metroman_p_stdev[:] = reaches.metroman_p_stdev[rch_sort]
    uc_metroman_ninf_p_cor[:] = reaches.metroman_ninf_p_cor[rch_sort]
    uc_metroman_ninf_Abar_cor[:] = reaches.metroman_ninf_abar_cor[rch_sort]
    uc_metroman_p_Abar_cor[:] = reaches.metroman_p_abar_cor[rch_sort]
    uc_metroman_sbQ_rel[:] = reaches.metroman_sbQ_rel[rch_sort]
    # ucmod2
    uc_bam_Abar[:] = reaches.bam_abar[rch_sort]
    uc_bam_n[:] = reaches.bam_n[rch_sort]
    uc_bam_sbQ_rel[:] = reaches.bam_sbQ_rel[rch_sort]
    # ucmod3
    uc_hivdi_Abar[:] = reaches.hivdi_abar[rch_sort]
    uc_hivdi_alpha[:] = reaches.hivdi_alpha[rch_sort]
    uc_hivdi_beta[:] = reaches.hivdi_beta[rch_sort]
    uc_hivdi_sbQ_rel[:] = reaches.hivdi_sbQ_rel[rch_sort]
    # ucmod4
    uc_momma_B[:] = reaches.momma_b[rch_sort]
    uc_momma_H[:] = reaches.momma_h[rch_sort]
    uc_momma_Save[:] = reaches.momma_save[rch_sort]
    uc_momma_sbQ_rel[:] = reaches.momma_sbQ_rel[rch_sort]
    # ucmod5
    uc_sads_Abar[:] = reaches.sads_abar[rch_sort]
    uc_sads_n[:] = reaches.sads_n[rch_sort]
    uc_sads_sbQ_rel[:] = reaches.sads_sbQ_rel[rch_sort]
    # ucmod6
    uc_sic4d_Abar[:] = reaches.sic4d_abar[rch_sort]
    uc_sic4d_n[:] = reaches.sic4d_n[rch_sort]
    uc_sic4d_sbQ_rel[:] = reaches.sic4d_sbQ_rel[rch_sort]
    # cmod1
    c_metroman_Abar[:] = reaches.metroman_abar[rch_sort]
    c_metroman_ninf[:] = reaches.metroman_ninf[rch_sort]
    c_metroman_p[:] = reaches.metroman_p[rch_sort]
    c_metroman_Abar_stdev[:] = reaches.metroman_abar_stdev[rch_sort]
    c_metroman_ninf_stdev[:] = reaches.metroman_ninf_stdev[rch_sort]
    c_metroman_p_stdev[:] = reaches.metroman_p_stdev[rch_sort]
    c_metroman_ninf_p_cor[:] = reaches.metroman_ninf_p_cor[rch_sort]
    c_metroman_ninf_Abar_cor[:] = reaches.metroman_ninf_abar_cor[rch_sort]
    c_metroman_p_Abar_cor[:] = reaches.metroman_p_abar_cor[rch_sort]
    c_metroman_sbQ_rel[:] = reaches.metroman_sbQ_rel[rch_sort]
    # cmod2
    c_bam_Abar[:] = reaches.bam_abar[rch_sort]
    c_bam_n[:] = reaches.bam_n[rch_sort]
    c_bam_sbQ_rel[:] = reaches.bam_sbQ_rel[rch_sort]
    # cmod3
    c_hivdi_Abar[:] = reaches.hivdi_abar[rch_sort]
    c_hivdi_alpha[:] = reaches.hivdi_alpha[rch_sort]
    c_hivdi_beta[:] = reaches.hivdi_beta[rch_sort]
    c_hivdi_sbQ_rel[:] = reaches.hivdi_sbQ_rel[rch_sort]
    # cmod4
    c_momma_B[:] = reaches.momma_b[rch_sort]
    c_momma_H[:] = reaches.momma_h[rch_sort]
    c_momma_Save[:] = reaches.momma_save[rch_sort]
    c_momma_sbQ_rel[:] = reaches.momma_sbQ_rel[rch_sort]
    # cmod5
    c_sads_Abar[:] = reaches.sads_abar[rch_sort]
    c_sads_n[:] = reaches.sads_n[rch_sort]
    c_sads_sbQ_rel[:] = reaches.sads_sbQ_rel[rch_sort]
    # cmod6
    c_sic4d_Abar[:] = reaches.sic4d_abar[rch_sort]
    c_sic4d_n[:] = reaches.sic4d_n[rch_sort]
    c_sic4d_sbQ_rel[:] = reaches.sic4d_sbQ_rel[rch_sort]

    root_grp.close()

    end = time.time()

    print("Ended Saving Main NetCDF in: ", str(np.round((end-start)/60, 2)), " min")

    return outfile

###############################################################################

def discharge_attr_nc(reaches):
    """
    FUNCTION
    --------
        Populates SWORD reaches object with necessary placeholder attributes needed for 
        writing the SWORD netCDF files. These attributes pertain to the SWOT discharge 
        algorithms and are calculated by the Dishcarge Algorithm Working Group (DAWG). 

    INPUTS
    ------
        reaches: Object containing lcation and attribute information for
            each reach.

    OUTPUTS
    -------
        None. Fill attributes are added to existing SWORD reaches object. 
    """
    
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

###############################################################################