from __future__ import division
import numpy as np
import time
import netCDF4 as nc
import pandas as pd
from scipy import spatial as sp
import utm 
import argparse
from pyproj import Proj
import geopandas as gp
import geopy.distance

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
    
    centerlines.cl_id = np.array(data.groups['centerlines'].variables['cl_id'][:])
    centerlines.x = np.array(data.groups['centerlines'].variables['x'][:])
    centerlines.y = np.array(data.groups['centerlines'].variables['y'][:])
    centerlines.reach_id = np.array(data.groups['centerlines'].variables['reach_id'][:])
    centerlines.node_id = np.array(data.groups['centerlines'].variables['node_id'][:])
    
    nodes.id = np.array(data.groups['nodes'].variables['node_id'][:])
    nodes.cl_id = np.array(data.groups['nodes'].variables['cl_ids'][:])
    nodes.x = np.array(data.groups['nodes'].variables['x'][:])
    nodes.y = np.array(data.groups['nodes'].variables['y'][:])
    nodes.len = np.array(data.groups['nodes'].variables['node_length'][:])
    nodes.wse = np.array(data.groups['nodes'].variables['wse'][:])
    nodes.wse_var = np.array(data.groups['nodes'].variables['wse_var'][:])
    nodes.wth = np.array(data.groups['nodes'].variables['width'][:])
    nodes.wth_var = np.array(data.groups['nodes'].variables['width_var'][:])
    nodes.grod = np.array(data.groups['nodes'].variables['obstr_type'][:])
    nodes.grod_fid = np.array(data.groups['nodes'].variables['grod_id'][:])
    nodes.hfalls_fid = np.array(data.groups['nodes'].variables['hfalls_id'][:])
    nodes.nchan_max = np.array(data.groups['nodes'].variables['n_chan_max'][:])
    nodes.nchan_mod = np.array(data.groups['nodes'].variables['n_chan_mod'][:])
    nodes.dist_out = np.array(data.groups['nodes'].variables['dist_out'][:])
    nodes.reach_id = np.array(data.groups['nodes'].variables['reach_id'][:])
    nodes.facc = np.array(data.groups['nodes'].variables['facc'][:])
    nodes.lakeflag = np.array(data.groups['nodes'].variables['lakeflag'][:])
    nodes.wth_coef = np.array(data.groups['nodes'].variables['wth_coef'][:])
    nodes.ext_dist_coef = np.array(data.groups['nodes'].variables['ext_dist_coef'][:])
    nodes.max_wth = np.array(data.groups['nodes'].variables['max_width'][:])
    nodes.meand_len = np.array(data.groups['nodes'].variables['meander_length'][:])
    nodes.river_name = np.array(data.groups['nodes'].variables['river_name'][:])
    nodes.manual_add = np.array(data.groups['nodes'].variables['manual_add'][:])
    nodes.sinuosity = np.array(data.groups['nodes'].variables['sinuosity'][:])
    nodes.edit_flag = np.array(data.groups['nodes'].variables['edit_flag'][:])
    nodes.trib_flag = np.array(data.groups['nodes'].variables['trib_flag'][:])

    reaches.id = np.array(data.groups['reaches'].variables['reach_id'][:])
    reaches.cl_id = np.array(data.groups['reaches'].variables['cl_ids'][:])
    reaches.x = np.array(data.groups['reaches'].variables['x'][:])
    reaches.x_min = np.array(data.groups['reaches'].variables['x_min'][:])
    reaches.x_max = np.array(data.groups['reaches'].variables['x_max'][:])
    reaches.y = np.array(data.groups['reaches'].variables['y'][:])
    reaches.y_min = np.array(data.groups['reaches'].variables['y_min'][:])
    reaches.y_max = np.array(data.groups['reaches'].variables['y_max'][:])
    reaches.len = np.array(data.groups['reaches'].variables['reach_length'][:])
    reaches.wse = np.array(data.groups['reaches'].variables['wse'][:])
    reaches.wse_var = np.array(data.groups['reaches'].variables['wse_var'][:])
    reaches.wth = np.array(data.groups['reaches'].variables['width'][:])
    reaches.wth_var = np.array(data.groups['reaches'].variables['width_var'][:])
    reaches.slope = np.array(data.groups['reaches'].variables['slope'][:])
    reaches.rch_n_nodes = np.array(data.groups['reaches'].variables['n_nodes'][:])
    reaches.grod = np.array(data.groups['reaches'].variables['obstr_type'][:])
    reaches.grod_fid = np.array(data.groups['reaches'].variables['grod_id'][:])
    reaches.hfalls_fid = np.array(data.groups['reaches'].variables['hfalls_id'][:])
    reaches.lakeflag = np.array(data.groups['reaches'].variables['lakeflag'][:])
    reaches.nchan_max = np.array(data.groups['reaches'].variables['n_chan_max'][:])
    reaches.nchan_mod = np.array(data.groups['reaches'].variables['n_chan_mod'][:])
    reaches.dist_out = np.array(data.groups['reaches'].variables['dist_out'][:])
    reaches.n_rch_up = np.array(data.groups['reaches'].variables['n_rch_up'][:])
    reaches.n_rch_down = np.array(data.groups['reaches'].variables['n_rch_down'][:])
    reaches.rch_id_up = np.array(data.groups['reaches'].variables['rch_id_up'][:])
    reaches.rch_id_down = np.array(data.groups['reaches'].variables['rch_id_dn'][:])
    reaches.max_obs = np.array(data.groups['reaches'].variables['swot_obs'][:])
    reaches.orbits = np.array(data.groups['reaches'].variables['swot_orbits'][:])
    reaches.facc = np.array(data.groups['reaches'].variables['facc'][:])
    reaches.iceflag = np.array(data.groups['reaches'].variables['iceflag'][:])
    reaches.max_wth = np.array(data.groups['reaches'].variables['max_width'][:])
    reaches.river_name = np.array(data.groups['reaches'].variables['river_name'][:])
    reaches.low_slope = np.array(data.groups['reaches'].variables['low_slope_flag'][:])
    reaches.edit_flag= np.array(data.groups['reaches'].variables['edit_flag'][:])
    reaches.trib_flag = np.array(data.groups['reaches'].variables['trib_flag'][:])

    data.close()    

    return centerlines, nodes, reaches
    
###############################################################################    

def write_database_nc(centerlines, reaches, nodes, region, outfile):

    """
    FUNCTION:
        Outputs the SWOT River Database (SWORD) information in netcdf
        format. The file contains attributes for the high-resolution centerline,
        nodes, and reaches.

    INPUTS
        centerlines -- Object containing lcation and attribute information
            along the high-resolution centerline.
        reaches -- Object containing lcation and attribute information for
            each reach.
        nodes -- Object containing lcation and attribute information for
            each node.
        outfile -- Path for netcdf to be written.

    OUTPUTS
        SWORD NetCDF -- NetCDF file containing attributes for the high-resolution
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
    #root_grp.history = 'Created ' + time.ctime(time.time())

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

    # root group data
    #cont_str = nc.stringtochar(np.array(['NA'], 'S2'))
    #Name[:] = cont_str

    # centerline data
    cl_id[:] = centerlines.cl_id
    cl_x[:] = centerlines.x
    cl_y[:] = centerlines.y
    reach_id[:,:] = centerlines.reach_id
    node_id[:,:] = centerlines.node_id

    # node data
    Node_ID[:] = nodes.id
    node_cl_id[:,:] = nodes.cl_id
    node_x[:] = nodes.x
    node_y[:] = nodes.y
    node_len[:] = nodes.len
    node_rch_id[:] = nodes.reach_id
    node_wse[:] = nodes.wse
    node_wse_var[:] = nodes.wse_var
    node_wth[:] = nodes.wth
    node_wth_var[:] = nodes.wth_var
    node_chan_max[:] = nodes.nchan_max
    node_chan_mod[:] = nodes.nchan_mod
    node_grod_id[:] = nodes.grod
    node_grod_fid[:] = nodes.grod_fid
    node_hfalls_fid[:] = nodes.hfalls_fid
    node_dist_out[:] = nodes.dist_out
    node_wth_coef[:] = nodes.wth_coef
    node_ext_dist_coef[:] = nodes.ext_dist_coef
    node_facc[:] = nodes.facc
    node_lakeflag[:] = nodes.lakeflag
    #node_lake_id[:] = nodes.lake_id
    node_max_wth[:] = nodes.max_wth
    node_meand_len[:] = nodes.meand_len
    node_sinuosity[:] = nodes.sinuosity
    node_river_name[:] = nodes.river_name
    node_manual_add[:] = nodes.manual_add
    node_edit_flag[:] = nodes.edit_flag
    node_trib_flag[:] = nodes.trib_flag

    # reach data
    Reach_ID[:] = reaches.id
    rch_cl_id[:,:] = reaches.cl_id
    rch_x[:] = reaches.x
    rch_x_min[:] = reaches.x_min
    rch_x_max[:] = reaches.x_max
    rch_y[:] = reaches.y
    rch_y_min[:] = reaches.y_min
    rch_y_max[:] = reaches.y_max
    rch_len[:] = reaches.len
    num_nodes[:] = reaches.rch_n_nodes
    rch_wse[:] = reaches.wse
    rch_wse_var[:] = reaches.wse_var
    rch_wth[:] = reaches.wth
    rch_wth_var[:] = reaches.wth_var
    rch_facc[:] = reaches.facc
    rch_chan_max[:] = reaches.nchan_max
    rch_chan_mod[:] = reaches.nchan_mod
    rch_grod_id[:] = reaches.grod
    rch_grod_fid[:] = reaches.grod_fid
    rch_hfalls_fid[:] = reaches.hfalls_fid
    rch_slope[:] = reaches.slope
    rch_dist_out[:] = reaches.dist_out
    n_rch_up[:] = reaches.n_rch_up
    n_rch_down[:] = reaches.n_rch_down
    rch_id_up[:,:] = reaches.rch_id_up
    rch_id_down[:,:] = reaches.rch_id_down
    rch_lakeflag[:] = reaches.lakeflag
    rch_iceflag[:,:] = reaches.iceflag
    #rch_lake_id[:] = reaches.lake_id
    rch_swot_obs[:] = reaches.max_obs
    rch_orbits[:,:] = reaches.orbits
    rch_river_name[:] = reaches.river_name
    rch_max_wth[:] = reaches.max_wth
    rch_low_slope[:] = reaches.low_slope
    rch_edit_flag[:] = reaches.edit_flag
    rch_trib_flag[:] = reaches.trib_flag
    # subgroup1 - area fits
    h_break[:,:] = reaches.h_break
    w_break[:,:] = reaches.w_break
    h_variance[:] = reaches.wse_var
    w_variance[:] = reaches.wth_var
    hw_covariance[:] = reaches.hw_covariance
    h_err_stdev[:] = reaches.h_err_stdev
    w_err_stdev[:] = reaches.w_err_stdev
    h_w_nobs[:] = reaches.h_w_nobs
    fit_coeffs[:,:,:] = reaches.fit_coeffs
    med_flow_area[:] = reaches.med_flow_area
    # ucmod1
    uc_metroman_Abar[:] = reaches.metroman_abar
    uc_metroman_ninf[:] = reaches.metroman_ninf
    uc_metroman_p[:] = reaches.metroman_p
    uc_metroman_Abar_stdev[:] = reaches.metroman_abar_stdev
    uc_metroman_ninf_stdev[:] = reaches.metroman_ninf_stdev
    uc_metroman_p_stdev[:] = reaches.metroman_p_stdev
    uc_metroman_ninf_p_cor[:] = reaches.metroman_ninf_p_cor
    uc_metroman_ninf_Abar_cor[:] = reaches.metroman_ninf_abar_cor
    uc_metroman_p_Abar_cor[:] = reaches.metroman_p_abar_cor
    uc_metroman_sbQ_rel[:] = reaches.metroman_sbQ_rel
    # ucmod2
    uc_bam_Abar[:] = reaches.bam_abar
    uc_bam_n[:] = reaches.bam_n
    uc_bam_sbQ_rel[:] = reaches.bam_sbQ_rel
    # ucmod3
    uc_hivdi_Abar[:] = reaches.hivdi_abar
    uc_hivdi_alpha[:] = reaches.hivdi_alpha
    uc_hivdi_beta[:] = reaches.hivdi_beta
    uc_hivdi_sbQ_rel[:] = reaches.hivdi_sbQ_rel
    # ucmod4
    uc_momma_B[:] = reaches.momma_b
    uc_momma_H[:] = reaches.momma_h
    uc_momma_Save[:] = reaches.momma_save
    uc_momma_sbQ_rel[:] = reaches.momma_sbQ_rel
    # ucmod5
    uc_sads_Abar[:] = reaches.sads_abar
    uc_sads_n[:] = reaches.sads_n
    uc_sads_sbQ_rel[:] = reaches.sads_sbQ_rel
    # ucmod6
    uc_sic4d_Abar[:] = reaches.sic4d_abar
    uc_sic4d_n[:] = reaches.sic4d_n
    uc_sic4d_sbQ_rel[:] = reaches.sic4d_sbQ_rel
    # cmod1
    c_metroman_Abar[:] = reaches.metroman_abar
    c_metroman_ninf[:] = reaches.metroman_ninf
    c_metroman_p[:] = reaches.metroman_p
    c_metroman_Abar_stdev[:] = reaches.metroman_abar_stdev
    c_metroman_ninf_stdev[:] = reaches.metroman_ninf_stdev
    c_metroman_p_stdev[:] = reaches.metroman_p_stdev
    c_metroman_ninf_p_cor[:] = reaches.metroman_ninf_p_cor
    c_metroman_ninf_Abar_cor[:] = reaches.metroman_ninf_abar_cor
    c_metroman_p_Abar_cor[:] = reaches.metroman_p_abar_cor
    c_metroman_sbQ_rel[:] = reaches.metroman_sbQ_rel
    # cmod2
    c_bam_Abar[:] = reaches.bam_abar
    c_bam_n[:] = reaches.bam_n
    c_bam_sbQ_rel[:] = reaches.bam_sbQ_rel
    # cmod3
    c_hivdi_Abar[:] = reaches.hivdi_abar
    c_hivdi_alpha[:] = reaches.hivdi_alpha
    c_hivdi_beta[:] = reaches.hivdi_beta
    c_hivdi_sbQ_rel[:] = reaches.hivdi_sbQ_rel
    # cmod4
    c_momma_B[:] = reaches.momma_b
    c_momma_H[:] = reaches.momma_h
    c_momma_Save[:] = reaches.momma_save
    c_momma_sbQ_rel[:] = reaches.momma_sbQ_rel
    # cmod5
    c_sads_Abar[:] = reaches.sads_abar
    c_sads_n[:] = reaches.sads_n
    c_sads_sbQ_rel[:] = reaches.sads_sbQ_rel
    # cmod6
    c_sic4d_Abar[:] = reaches.sic4d_abar
    c_sic4d_n[:] = reaches.sic4d_n
    c_sic4d_sbQ_rel[:] = reaches.sic4d_sbQ_rel

    root_grp.close()

    end = time.time()

    print("Ended Saving Main NetCDF in: ", str(np.round((end-start)/60, 2)), " min")

    return outfile

###############################################################################

def reproject_utm(latitude, longitude):

    """
    Modified from C. Lion's function by E. Altenau
    Copyright (c) 2018 UNC Chapel Hill. All rights reserved.

    FUNCTION:
        Projects all points in UTM.

    INPUTS
        latitude -- latitude in degrees (1-D array)
        longitude -- longitude in degrees (1-D array)

    OUTPUTS
        east -- easting in UTM (1-D array)
        north -- northing in UTM (1-D array)
        utm_num -- UTM zone number (1-D array of utm zone numbers for each point)
        utm_let -- UTM zone letter (1-D array of utm zone letters for each point)
    """

    east = np.zeros(len(latitude))
    north = np.zeros(len(latitude))
    east_int = np.zeros(len(latitude))
    north_int = np.zeros(len(latitude))
    zone_num = np.zeros(len(latitude))
    zone_let = []

	# Finds UTM letter and zone for each lat/lon pair.

    for ind in list(range(len(latitude))):
        (east_int[ind], north_int[ind],
	 zone_num[ind], zone_let_int) = utm.from_latlon(latitude[ind],
	                                                longitude[ind])
        zone_let.append(zone_let_int)

    # Finds the unique UTM zones and converts the lat/lon pairs to UTM.
    unq_zones = np.unique(zone_num)
    utm_let = np.unique(zone_let)[0]

    for idx in list(range(len(unq_zones))):
        pt_len = len(np.where(zone_num == unq_zones[idx])[0])

    idx = np.where(pt_len == np.max(pt_len))

    # Set the projection

    if np.sum(latitude) > 0:
        myproj = Proj(
		"+proj=utm +zone=" + str(int(unq_zones[idx])) + utm_let +
		" +ellips=WGS84 +datum=WGS84 +units=m")
    else:
        myproj = Proj(
		"+proj=utm +south +zone=" + str(int(unq_zones[idx])) + utm_let +
		" +ellips=WGS84 +datum=WGS84 +units=m")

    # Convert all the lon/lat to the main UTM zone
    (east, north) = myproj(longitude, latitude)

    return east, north, zone_num, zone_let

###############################################################################

def calc_segDist(subcls_lon, subcls_lat, subcls_rch_id, subcls_facc,
                 subcls_rch_ind):

    """
    FUNCTION:
        Creates a 1-D array of flow distances for each specified reach. Flow
        distance is build to start at 0 and increases in the upstream direction.

    INPUTS
        subcls -- Object containing reach and node attributes for the
            high-resolution centerline.
            [attributes used]:
                lon -- Longitude values along the high-resolution centerline.
                lat -- Latitude values along the high-resolution centerline.
                facc -- Flow accumulation along the high-resolution centerline.
                rch_id5 -- Reach numbers after aggregating short river, lake,
                    and dam reaches.

    OUTPUTS
        segDist -- Flow distance per reach (meters).
    """

    #loop through each reach and calculate flow distance.
    segDist = np.zeros(len(subcls_lon))
    uniq_segs = np.unique(subcls_rch_id)
    for ind in list(range(len(uniq_segs))):
        #for a single reach, reproject lat-lon coordinates to utm.
        #print(ind, uniq_segs[ind])
        seg = np.where(subcls_rch_id == uniq_segs[ind])[0]
        seg_lon = subcls_lon[seg]
        seg_lat = subcls_lat[seg]
        seg_x, seg_y, __, __ = reproject_utm(seg_lat, seg_lon)
        upa = subcls_facc[seg]

        #order the reach points based on index values, then calculate the
        #eculdean distance bewteen each ordered point.
        order_ids = np.argsort(subcls_rch_ind[seg])
        dist = np.zeros(len(seg))
        dist[order_ids[0]] = 0
        for idx in list(range(len(order_ids)-1)):
            d = np.sqrt((seg_x[order_ids[idx]]-seg_x[order_ids[idx+1]])**2 +
                        (seg_y[order_ids[idx]]-seg_y[order_ids[idx+1]])**2)
            dist[order_ids[idx+1]] = d + dist[order_ids[idx]]

        #format flow distance as array and determine flow direction by flowacc.
        dist = np.array(dist)
        start = upa[np.where(dist == np.min(dist))[0][0]]
        end = upa[np.where(dist == np.max(dist))[0][0]]

        if end > start:
            segDist[seg] = abs(dist-np.max(dist))

        else:
            segDist[seg] = dist

    return segDist

###############################################################################

def calc_rch_facc(nodes, centerlines, cl_rch):
    unq_nodes = np.unique(centerlines.node_id[0,cl_rch])
    facc = np.zeros(len(cl_rch))
    for n in list(range(len(unq_nodes))):
        cl_ind = np.where(centerlines.node_id[0,cl_rch] == unq_nodes[n])[0]
        nd_ind = np.where(nodes.id == unq_nodes[n])[0]
        facc[cl_ind] = nodes.facc[nd_ind]
    return facc

###############################################################################
###############################################################################
###############################################################################

start_all = time.time()

#read in netcdf data. 
parser = argparse.ArgumentParser()
parser.add_argument("region", help="continental region", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("local_processing", help="'True' for local machine, 'False' for server", type = str)
args = parser.parse_args()

region = args.region
version = args.version

if args.local_processing == 'True':
    outdir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'
else:
    outdir = '/afs/cas.unc.edu/depts/geological_sciences/pavelsky/students/ealtenau/SWORD_dev/outputs/Reaches_Nodes/'

outpath = outdir+version+'/'
fn = outpath+'netcdf/'+region.lower()+'_sword_'+version+'.nc'
centerlines, nodes, reaches = read_data(fn)

df = pd.DataFrame(np.array([centerlines.x, centerlines.y]).T)
dups = df.duplicated()
rmv = np.where(dups == True)[0]

rch_updates = np.unique(centerlines.reach_id[0,rmv]) #len(rch_updates)
node_updates = np.unique(centerlines.node_id[0,rmv]) #len(node_updates)

#remove duplicated centerline points
print('Removing Duplicated Centerline Points')
start = time.time()
centerlines.cl_id = np.delete(centerlines.cl_id, rmv, axis=0)
centerlines.x = np.delete(centerlines.x, rmv, axis=0)
centerlines.y = np.delete(centerlines.y, rmv, axis=0)
centerlines.reach_id = np.delete(centerlines.reach_id, rmv, axis=1)
centerlines.node_id = np.delete(centerlines.node_id, rmv, axis=1)
end = time.time()
print('Finished Removing Duplicated Points in: '+str(np.round((end-start),2))+' sec')

#loop through and update attributes for affected reaches and nodes. 
#reach attributes to update: 'cl_ids', 'x', 'x_min', 'x_max', 'y', 'y_min', 'y_max', 'reach_length', 'dist_out'
#node attributes to update: 'cl_ids', 'x', 'y', 'node_length', 'dist_out'
print('Starting Reach and Nodes Updates')
start = time.time()
for ind in list(range(len(rch_updates))):
    print(ind, len(rch_updates)-1)
    cl_rch = np.where(centerlines.reach_id[0,:] == rch_updates[ind])[0]
    rch_ind = np.where(reaches.id == rch_updates[ind])[0]

    if len(cl_rch) == 0:
        print('reach removed completely', rch_updates[ind])
        node_ind = np.where(nodes.reach_id == rch_updates[ind])[0]
        nodes.id = np.delete(nodes.id, node_ind, axis = 0)
        nodes.cl_id = np.delete(nodes.cl_id, node_ind, axis = 1)
        nodes.x = np.delete(nodes.x, node_ind, axis = 0)
        nodes.y = np.delete(nodes.y, node_ind, axis = 0)
        nodes.len = np.delete(nodes.len, node_ind, axis = 0)
        nodes.wse = np.delete(nodes.wse, node_ind, axis = 0)
        nodes.wse_var = np.delete(nodes.wse_var, node_ind, axis = 0)
        nodes.wth = np.delete(nodes.wth, node_ind, axis = 0)
        nodes.wth_var = np.delete(nodes.wth_var, node_ind, axis = 0)
        nodes.grod = np.delete(nodes.grod, node_ind, axis = 0)
        nodes.grod_fid = np.delete(nodes.grod_fid, node_ind, axis = 0)
        nodes.hfalls_fid = np.delete(nodes.hfalls_fid, node_ind, axis = 0)
        nodes.nchan_max = np.delete(nodes.nchan_max, node_ind, axis = 0)
        nodes.nchan_mod = np.delete(nodes.nchan_mod, node_ind, axis = 0)
        nodes.dist_out = np.delete(nodes.dist_out, node_ind, axis = 0)
        nodes.reach_id = np.delete(nodes.reach_id, node_ind, axis = 0)
        nodes.facc = np.delete(nodes.facc, node_ind, axis = 0)
        nodes.lakeflag = np.delete(nodes.lakeflag, node_ind, axis = 0)
        nodes.wth_coef = np.delete(nodes.wth_coef, node_ind, axis = 0)
        nodes.ext_dist_coef = np.delete(nodes.ext_dist_coef, node_ind, axis = 0)
        nodes.max_wth = np.delete(nodes.max_wth, node_ind, axis = 0)
        nodes.meand_len = np.delete(nodes.meand_len, node_ind, axis = 0)
        nodes.river_name = np.delete(nodes.river_name, node_ind, axis = 0)
        nodes.manual_add = np.delete(nodes.manual_add, node_ind, axis = 0)
        nodes.sinuosity = np.delete(nodes.sinuosity, node_ind, axis = 0)
        nodes.edit_flag = np.delete(nodes.edit_flag, node_ind, axis = 0)
        nodes.trib_flag = np.delete(nodes.trib_flag, node_ind, axis = 0)

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

        #removing residual neighbors with deleted reach id in centerline and reach groups. 
        cl_ind1 = np.where(centerlines.reach_id[0,:] == rch_updates[ind])[0]
        cl_ind2 = np.where(centerlines.reach_id[1,:] == rch_updates[ind])[0]
        cl_ind3 = np.where(centerlines.reach_id[2,:] == rch_updates[ind])[0]
        cl_ind4 = np.where(centerlines.reach_id[3,:] == rch_updates[ind])[0]
        if len(cl_ind1) > 0:
            centerlines.reach_id[0,cl_ind1] = 0
        if len(cl_ind2) > 0:
            centerlines.reach_id[1,cl_ind2] = 0
        if len(cl_ind3) > 0:
            centerlines.reach_id[2,cl_ind3] = 0
        if len(cl_ind4) > 0:
            centerlines.reach_id[3,cl_ind4] = 0

        rch_up_ind1 = np.where(reaches.rch_id_up[0,:] == rch_updates[ind])[0]
        rch_up_ind2 = np.where(reaches.rch_id_up[1,:] == rch_updates[ind])[0]
        rch_up_ind3 = np.where(reaches.rch_id_up[2,:] == rch_updates[ind])[0]
        rch_up_ind4 = np.where(reaches.rch_id_up[3,:] == rch_updates[ind])[0]
        if len(rch_up_ind1) > 0:
            reaches.rch_id_up[0,rch_up_ind1] = 0
        if len(rch_up_ind2) > 0:
            reaches.rch_id_up[1,rch_up_ind2] = 0
        if len(rch_up_ind3) > 0:
            reaches.rch_id_up[2,rch_up_ind3] = 0
        if len(rch_up_ind4) > 0:
            reaches.rch_id_up[3,rch_up_ind4] = 0

        rch_dn_ind1 = np.where(reaches.rch_id_down[0,:] == rch_updates[ind])[0]
        rch_dn_ind2 = np.where(reaches.rch_id_down[1,:] == rch_updates[ind])[0]
        rch_dn_ind3 = np.where(reaches.rch_id_down[2,:] == rch_updates[ind])[0]
        rch_dn_ind4 = np.where(reaches.rch_id_down[3,:] == rch_updates[ind])[0]
        if len(rch_dn_ind1) > 0:
            reaches.rch_id_down[0,rch_dn_ind1] = 0
        if len(rch_dn_ind2) > 0:
            reaches.rch_id_down[1,rch_dn_ind2] = 0
        if len(rch_dn_ind3) > 0:
            reaches.rch_id_down[2,rch_dn_ind3] = 0
        if len(rch_dn_ind4) > 0:
            reaches.rch_id_down[3,rch_dn_ind4] = 0
        # skip rest of the loop. 
        continue

    #calculate distance variables for centerline points. 
    facc = calc_rch_facc(nodes, centerlines, cl_rch)
    # dist = calc_segDist(centerlines.x[cl_rch], centerlines.y[cl_rch], 
    #                     centerlines.reach_id[0,cl_rch], 
    #                     facc, centerlines.cl_id[cl_rch])
    sort_ids = np.argsort(centerlines.cl_id[cl_rch])
    gdf = gp.GeoDataFrame(geometry=gp.points_from_xy(centerlines.x[cl_rch[sort_ids]], 
                                                     centerlines.y[cl_rch[sort_ids]]),crs="EPSG:4326").to_crs("EPSG:3857")
    diff = gdf.distance(gdf.shift(1)); diff[0] = 0
    dist = np.array(np.cumsum(diff))
    
    #update reach variables:
    diff_len = reaches.len[rch_ind] - np.max(dist)
    reaches.x[rch_ind] = np.median(centerlines.x[cl_rch])
    reaches.y[rch_ind] = np.median(centerlines.y[cl_rch])
    reaches.x_max[rch_ind] = np.max(centerlines.x[cl_rch])
    reaches.x_min[rch_ind] = np.min(centerlines.x[cl_rch])
    reaches.y_max[rch_ind] = np.max(centerlines.y[cl_rch])
    reaches.y_min[rch_ind] = np.min(centerlines.y[cl_rch])
    reaches.cl_id[0,rch_ind] = np.min(centerlines.cl_id[cl_rch])
    reaches.cl_id[1,rch_ind] = np.max(centerlines.cl_id[cl_rch])
    reaches.rch_n_nodes[rch_ind] = len(np.unique(centerlines.node_id[0,cl_rch]))
    reaches.len[rch_ind] = np.max(dist)
    reaches.dist_out[rch_ind] = reaches.dist_out[rch_ind]-diff_len

    nds = node_updates[np.in1d(node_updates, centerlines.node_id[0,cl_rch])]
    for idx in list(range(len(nds))):
        cl_nds = np.where(centerlines.node_id[0,cl_rch] == nds[idx])[0]
        ns = np.where(nodes.id == nds[idx])[0]

        #update reach variables:
        node_len = np.max(dist[cl_nds])-np.min(dist[cl_nds])
        node_diff_len = nodes.len[ns] - node_len
        nodes.x[ns] = np.median(centerlines.x[cl_rch[cl_nds]])
        nodes.y[ns] = np.median(centerlines.y[cl_rch[cl_nds]])
        nodes.cl_id[0,ns] = np.min(centerlines.cl_id[cl_rch[cl_nds]])
        nodes.cl_id[1,ns] = np.max(centerlines.cl_id[cl_rch[cl_nds]])
        nodes.len[ns] = node_len
        nodes.dist_out[ns] = nodes.dist_out[ns]-node_diff_len 

diff = np.in1d(nodes.id, np.unique(centerlines.node_id[0,:]))
rmv2 = np.where(diff == False)[0]
nodes.id = np.delete(nodes.id, rmv2, axis = 0)
nodes.cl_id = np.delete(nodes.cl_id, rmv2, axis = 1)
nodes.x = np.delete(nodes.x, rmv2, axis = 0)
nodes.y = np.delete(nodes.y, rmv2, axis = 0)
nodes.len = np.delete(nodes.len, rmv2, axis = 0)
nodes.wse = np.delete(nodes.wse, rmv2, axis = 0)
nodes.wse_var = np.delete(nodes.wse_var, rmv2, axis = 0)
nodes.wth = np.delete(nodes.wth, rmv2, axis = 0)
nodes.wth_var = np.delete(nodes.wth_var, rmv2, axis = 0)
nodes.grod = np.delete(nodes.grod, rmv2, axis = 0)
nodes.grod_fid = np.delete(nodes.grod_fid, rmv2, axis = 0)
nodes.hfalls_fid = np.delete(nodes.hfalls_fid, rmv2, axis = 0)
nodes.nchan_max = np.delete(nodes.nchan_max, rmv2, axis = 0)
nodes.nchan_mod = np.delete(nodes.nchan_mod, rmv2, axis = 0)
nodes.dist_out = np.delete(nodes.dist_out, rmv2, axis = 0)
nodes.reach_id = np.delete(nodes.reach_id, rmv2, axis = 0)
nodes.facc = np.delete(nodes.facc, rmv2, axis = 0)
nodes.lakeflag = np.delete(nodes.lakeflag, rmv2, axis = 0)
nodes.wth_coef = np.delete(nodes.wth_coef, rmv2, axis = 0)
nodes.ext_dist_coef = np.delete(nodes.ext_dist_coef, rmv2, axis = 0)
nodes.max_wth = np.delete(nodes.max_wth, rmv2, axis = 0)
nodes.meand_len = np.delete(nodes.meand_len, rmv2, axis = 0)
nodes.river_name = np.delete(nodes.river_name, rmv2, axis = 0)
nodes.manual_add = np.delete(nodes.manual_add, rmv2, axis = 0)
nodes.sinuosity = np.delete(nodes.sinuosity, rmv2, axis = 0)
nodes.edit_flag = np.delete(nodes.edit_flag, rmv2, axis = 0)
nodes.trib_flag = np.delete(nodes.trib_flag, rmv2, axis = 0)

end = time.time()
print('Finished Reach and Node Updates in: '+str(np.round((end-start)/60,2))+' min')
print('Cl Dimensions:', len(np.unique(centerlines.cl_id)), len(centerlines.cl_id))
print('Rch Dimensions:', len(np.unique(centerlines.reach_id[0,:])), len(np.unique(nodes.reach_id)), len(reaches.id))
print('Node Dimensions:', len(np.unique(centerlines.node_id[0,:])), len(np.unique(nodes.id)), len(nodes.id))

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

print('Writing NetCDF')
write_database_nc(centerlines, reaches, nodes, region, fn)

end_all = time.time()
print('Finished '+region+' Updates in: '+str(np.round((end_all-start_all)/3600,2))+' hrs')