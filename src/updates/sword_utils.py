# -*- coding: utf-8 -*-
"""

SWORD Utilities (sword_utils.py)
=======================================

Utilities for reading, writing, managing, and processing the SWORD 
database. SWORD formats include netCDF, shapefile, and geopackage. 

"""

from __future__ import division
import os
import numpy as np
import time
import netCDF4 as nc
import geopy.distance
from shapely.geometry import LineString, Point
from geopandas import GeoSeries
import geopandas as gp
import pandas as pd

###############################################################################

class Object(object):
    """
    FUNCTION:
        Creates an empty class object to assign SWORD attributes to.
    """
    pass 

###############################################################################

def prepare_paths(main_dir, region, version):
    
    # Create dictionary of directories
    paths = dict()

    # input/output shapefile directory.
    paths['shp_dir'] = main_dir+'/data/outputs/Reaches_Nodes/'+version+'/shp/'+region+'/'
    # input/output geopackage directory.
    paths['gpkg_dir'] = main_dir+'/data/outputs/Reaches_Nodes/'+version+'/gpkg/'
    # input/output netcdf directory.
    paths['nc_dir'] = main_dir+'/data/outputs/Reaches_Nodes/'+version+'/netcdf/'
    # input/output connectivity netcdf directory. 
    paths['geom_dir'] = main_dir+'/data/outputs/Reaches_Nodes/'+version+'/reach_geometry/'
    # updates directory. 
    paths['update_dir'] = main_dir+'/data/update_requests/'+version+'/'+region+'/'
    # topology directory. 
    paths['topo_dir'] = main_dir+'/data/outputs/Topology/'+version+'/'+region+'/'
    # version directory. 
    paths['version_dir'] = main_dir+'/data/outputs/Version_Differences/'+version+'/'
    # 30 m centerline points gpkg directory. 
    paths['pts_gpkg_dir'] = main_dir+'/data/outputs/Reaches_Nodes/'+version+'/gpkg_30m/'+region+'/'

    # input/output netcdf filename. 
    paths['nc_fn'] = region.lower()+'_sword_'+version+'.nc'
    # input/output reaches geopackage filename. 
    paths['gpkg_rch_fn'] = region.lower()+'_sword_reaches_'+version+'.gpkg'
    # input/output nodes geopackage filename. 
    paths['gpkg_node_fn'] = region.lower()+'_sword_nodes_'+version+'.gpkg'
    # input/output reaches shapefile filename. 
    paths['shp_rch_fn'] = region.lower()+'_sword_reaches_hbXX_'+version+'.shp'
    # input/output nodes shapefile filename. 
    paths['shp_node_fn'] = region.lower()+'_sword_nodes_hbXX_'+version+'.shp'
    # input/output connectivity netcdf filename. 
    paths['geom_fn'] = region.lower()+'_sword_'+version+'_connectivity.nc'
    
    # Create directories if they don't exist.
    if os.path.isdir(paths['shp_dir']) is False:
        os.makedirs(paths['shp_dir'])
    if os.path.isdir(paths['gpkg_dir']) is False:
        os.makedirs(paths['gpkg_dir'])
    if os.path.isdir(paths['nc_dir']) is False:
        os.makedirs(paths['nc_dir'])
    if os.path.isdir(paths['geom_dir']) is False:
        os.makedirs(paths['geom_dir'])
    if os.path.isdir(paths['update_dir']) is False:
        os.makedirs(paths['update_dir'])
    if os.path.isdir(paths['topo_dir']) is False:
        os.makedirs(paths['topo_dir'])
    if os.path.isdir(paths['version_dir']) is False:
        os.makedirs(paths['version_dir'])
    if os.path.isdir(paths['pts_gpkg_dir']) is False:
        os.makedirs(paths['pts_gpkg_dir'])

    return paths

###############################################################################

def read_nc(filename):

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
    nodes.path_freq = np.array(data.groups['nodes'].variables['path_freq'][:])
    nodes.path_order = np.array(data.groups['nodes'].variables['path_order'][:])
    nodes.path_segs = np.array(data.groups['nodes'].variables['path_segs'][:])
    nodes.strm_order = np.array(data.groups['nodes'].variables['stream_order'][:])
    nodes.main_side = np.array(data.groups['nodes'].variables['main_side'][:])
    nodes.end_rch = np.array(data.groups['nodes'].variables['end_reach'][:])
    nodes.network = np.array(data.groups['nodes'].variables['network'][:])
    nodes.add_flag = np.array(data.groups['nodes'].variables['add_flag'][:])

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
    reaches.edit_flag = np.array(data.groups['reaches'].variables['edit_flag'][:])
    reaches.trib_flag = np.array(data.groups['reaches'].variables['trib_flag'][:])
    reaches.path_freq = np.array(data.groups['reaches'].variables['path_freq'][:])
    reaches.path_order = np.array(data.groups['reaches'].variables['path_order'][:])
    reaches.path_segs = np.array(data.groups['reaches'].variables['path_segs'][:])
    reaches.strm_order = np.array(data.groups['reaches'].variables['stream_order'][:])
    reaches.main_side = np.array(data.groups['reaches'].variables['main_side'][:])
    reaches.end_rch = np.array(data.groups['reaches'].variables['end_reach'][:])
    reaches.network = np.array(data.groups['reaches'].variables['network'][:])
    reaches.add_flag = np.array(data.groups['reaches'].variables['add_flag'][:])

    data.close()    

    return centerlines, nodes, reaches
    
###############################################################################    

def find_common_points(centerlines, reaches):
    # function: find_common_points
    multi_pts = np.where(centerlines.multi_flag == 2)[0]
    common = np.zeros(len(centerlines.x), dtype=int)
    for ind in list(range(len(multi_pts))):
        # print(ind, len(multi_pts)-1)
        if common[multi_pts[ind]] == 1:
            continue
        
        #find all neighbors
        nghs = centerlines.neighbors[np.where(centerlines.neighbors[:,multi_pts[ind]] > 0)[0],multi_pts[ind]]
        nghs = nghs[np.in1d(nghs,np.unique(centerlines.neighbors[0,:]))] #added on 9/9/2024 to account for deleted reaches that were still in neighbors...

        #need to loop through and see if any neighbor pts are already common and continue.
        flag=[]
        for n in list(range(0,len(nghs))):
            # print(n)
            if n == 0:
                flag.append(common[multi_pts[ind]])
            else:
                r = np.where(centerlines.neighbors[0,:] == nghs[n])[0]
                mn = r[np.where(centerlines.cl_id[r] == np.min(centerlines.cl_id[r]))[0]]
                mx = r[np.where(centerlines.cl_id[r] == np.max(centerlines.cl_id[r]))[0]]

                coords_1 = (centerlines.y[multi_pts[ind]], centerlines.x[multi_pts[ind]])
                coords_2 = (centerlines.y[mn], centerlines.x[mn])
                coords_3 = (centerlines.y[mx], centerlines.x[mx])
                d1 = geopy.distance.geodesic(coords_1, coords_2).m
                d2 = geopy.distance.geodesic(coords_1, coords_3).m

                if d1 < d2:
                    flag.append(common[mn][0])
                else:
                    flag.append(common[mx][0])
        
        if np.max(flag) == 1:
            continue

        # if no neighbors are common attach topology variables.
        facc = np.zeros(len(nghs))
        wse = np.zeros(len(nghs))
        wth = np.zeros(len(nghs))
        for n in list(range(len(nghs))):
            r = np.where(reaches.id == nghs[n])
            facc[n] =  reaches.facc[r]
            wse[n] =  reaches.wse[r]
            wth[n] = reaches.wth[r]

        f = np.where(facc == np.max(facc))[0]
        h = np.where(wse == np.min(wse))[0]
        w = np.where(wth == np.max(wth))[0]

        if len(f) == 1:
            # print('cond.1')
            if f == 0:
                common[multi_pts[ind]] = 1
        elif len(h) == 1:
            # print('cond.2')
            if h == 0:
                common[multi_pts[ind]] = 1
        elif len(w) == 1:
            # print('cond.3')
            if w == 0:
                common[multi_pts[ind]] = 1
        else:
            # print('cond.4')
            common[multi_pts[ind]] = 1   

    return common

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

def delete_nodes(nodes, node_ind):
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
    nodes.path_freq = np.delete(nodes.path_freq, node_ind, axis = 0)
    nodes.path_order = np.delete(nodes.path_order, node_ind, axis = 0)
    nodes.path_segs = np.delete(nodes.path_segs, node_ind, axis = 0)
    nodes.main_side = np.delete(nodes.main_side, node_ind, axis = 0)
    nodes.strm_order = np.delete(nodes.strm_order, node_ind, axis = 0)
    nodes.end_rch = np.delete(nodes.end_rch, node_ind, axis = 0)
    nodes.network = np.delete(nodes.network, node_ind, axis = 0)
    nodes.add_flag = np.delete(nodes.add_flag, node_ind, axis = 0)

###############################################################################

def append_nodes(nodes, subnodes):

    nodes.id = np.append(nodes.id, subnodes.id)
    nodes.cl_id = np.append(nodes.cl_id, subnodes.cl_id, axis=1)
    nodes.x = np.append(nodes.x, subnodes.x)
    nodes.y = np.append(nodes.y, subnodes.y)
    nodes.len = np.append(nodes.len, subnodes.len)
    nodes.wse = np.append(nodes.wse, subnodes.wse)
    nodes.wse_var = np.append(nodes.wse_var, subnodes.wse_var)
    nodes.wth = np.append(nodes.wth, subnodes.wth)
    nodes.wth_var = np.append(nodes.wth_var, subnodes.wth_var)
    nodes.grod = np.append(nodes.grod, subnodes.grod)
    nodes.grod_fid = np.append(nodes.grod_fid, subnodes.grod_fid)
    nodes.hfalls_fid = np.append(nodes.hfalls_fid, subnodes.hfalls_fid)
    nodes.nchan_max = np.append(nodes.nchan_max, subnodes.nchan_max)
    nodes.nchan_mod = np.append(nodes.nchan_mod, subnodes.nchan_mod)
    nodes.dist_out = np.append(nodes.dist_out, subnodes.dist_out)
    nodes.reach_id = np.append(nodes.reach_id, subnodes.reach_id)
    nodes.facc = np.append(nodes.facc, subnodes.facc)
    nodes.lakeflag = np.append(nodes.lakeflag, subnodes.lakeflag)
    nodes.wth_coef = np.append(nodes.wth_coef, subnodes.wth_coef)
    nodes.ext_dist_coef = np.append(nodes.ext_dist_coef, subnodes.ext_dist_coef)
    nodes.max_wth = np.append(nodes.max_wth, subnodes.max_wth)
    nodes.meand_len = np.append(nodes.meand_len, subnodes.meand_len)
    nodes.river_name = np.append(nodes.river_name, subnodes.river_name)
    nodes.manual_add = np.append(nodes.manual_add, subnodes.manual_add)
    nodes.sinuosity = np.append(nodes.sinuosity, subnodes.sinuosity)
    nodes.edit_flag = np.append(nodes.edit_flag, subnodes.edit_flag)
    nodes.trib_flag = np.append(nodes.trib_flag, subnodes.trib_flag)
    nodes.path_freq = np.append(nodes.path_freq, subnodes.path_freq)
    nodes.path_order = np.append(nodes.path_order, subnodes.path_order)
    nodes.path_segs = np.append(nodes.path_segs, subnodes.path_segs)
    nodes.main_side = np.append(nodes.main_side, subnodes.main_side)
    nodes.strm_order = np.append(nodes.strm_order, subnodes.strm_order)
    nodes.end_rch = np.append(nodes.end_rch, subnodes.end_rch)
    nodes.network = np.append(nodes.network, subnodes.network)
    nodes.add_flag = np.append(nodes.add_flag, subnodes.add_flag)

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
        reaches.add_flag = np.delete(reaches.add_flag, rch_ind, axis = 0)

###############################################################################

def delete_data(centerlines, nodes, reaches, rm_rch):
    
    for ind in list(range(len(rm_rch))):
        print(ind, len(rm_rch)-1)
        rch_ind = np.where(reaches.id == rm_rch[ind])[0]
        node_ind = np.where(nodes.reach_id == rm_rch[ind])[0]
        cl_ind = np.where(centerlines.reach_id[0,:] == rm_rch[ind])[0]

        if len(rch_ind) == 0:
            print(rm_rch[ind], 'not in database')

        centerlines.cl_id = np.delete(centerlines.cl_id, cl_ind, axis=0)
        centerlines.x = np.delete(centerlines.x, cl_ind, axis=0)
        centerlines.y = np.delete(centerlines.y, cl_ind, axis=0)
        centerlines.reach_id = np.delete(centerlines.reach_id, cl_ind, axis=1)
        centerlines.node_id = np.delete(centerlines.node_id, cl_ind, axis=1)

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
        nodes.path_freq = np.delete(nodes.path_freq, node_ind, axis = 0)
        nodes.path_order = np.delete(nodes.path_order, node_ind, axis = 0)
        nodes.path_segs = np.delete(nodes.path_segs, node_ind, axis = 0)
        nodes.main_side = np.delete(nodes.main_side, node_ind, axis = 0)
        nodes.strm_order = np.delete(nodes.strm_order, node_ind, axis = 0)
        nodes.end_rch = np.delete(nodes.end_rch, node_ind, axis = 0)
        nodes.network = np.delete(nodes.network, node_ind, axis = 0)
        nodes.add_flag = np.delete(nodes.add_flag, node_ind, axis = 0)

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
        reaches.add_flag = np.delete(reaches.add_flag, rch_ind, axis = 0)

        #removing residual neighbors with deleted reach id in centerline and reach groups. 
        cl_ind1 = np.where(centerlines.reach_id[0,:] == rm_rch[ind])[0]
        cl_ind2 = np.where(centerlines.reach_id[1,:] == rm_rch[ind])[0]
        cl_ind3 = np.where(centerlines.reach_id[2,:] == rm_rch[ind])[0]
        cl_ind4 = np.where(centerlines.reach_id[3,:] == rm_rch[ind])[0]
        if len(cl_ind1) > 0:
            centerlines.reach_id[0,cl_ind1] = 0
        if len(cl_ind2) > 0:
            centerlines.reach_id[1,cl_ind2] = 0
        if len(cl_ind3) > 0:
            centerlines.reach_id[2,cl_ind3] = 0
        if len(cl_ind4) > 0:
            centerlines.reach_id[3,cl_ind4] = 0

        rch_up_ind1 = np.where(reaches.rch_id_up[0,:] == rm_rch[ind])[0]
        rch_up_ind2 = np.where(reaches.rch_id_up[1,:] == rm_rch[ind])[0]
        rch_up_ind3 = np.where(reaches.rch_id_up[2,:] == rm_rch[ind])[0]
        rch_up_ind4 = np.where(reaches.rch_id_up[3,:] == rm_rch[ind])[0]
        if len(rch_up_ind1) > 0:
            reaches.rch_id_up[0,rch_up_ind1] = 0
            reaches.rch_id_up[:,rch_up_ind1] = np.sort(reaches.rch_id_up[:,rch_up_ind1], axis = 0)[::-1]
            up1 = np.unique(reaches.rch_id_up[:,rch_up_ind1]); up1 = up1[up1>0]
            reaches.n_rch_up[rch_up_ind1] = len(up1)
        if len(rch_up_ind2) > 0:
            reaches.rch_id_up[1,rch_up_ind2] = 0
            reaches.rch_id_up[:,rch_up_ind2] = np.sort(reaches.rch_id_up[:,rch_up_ind2], axis = 0)[::-1]
            up2 = np.unique(reaches.rch_id_up[:,rch_up_ind2]); up2 = up2[up2>0]
            reaches.n_rch_up[rch_up_ind2] = len(up2)
        if len(rch_up_ind3) > 0:
            reaches.rch_id_up[2,rch_up_ind3] = 0
            reaches.rch_id_up[:,rch_up_ind3] = np.sort(reaches.rch_id_up[:,rch_up_ind3], axis = 0)[::-1]
            up3 = np.unique(reaches.rch_id_up[:,rch_up_ind3]); up3 = up3[up3>0]
            reaches.n_rch_up[rch_up_ind3] = len(up3)
        if len(rch_up_ind4) > 0:
            reaches.rch_id_up[3,rch_up_ind4] = 0
            reaches.rch_id_up[:,rch_up_ind4] = np.sort(reaches.rch_id_up[:,rch_up_ind4], axis = 0)[::-1]
            up4 = np.unique(reaches.rch_id_up[:,rch_up_ind4]); up4 = up4[up4>0]
            reaches.n_rch_up[rch_up_ind4] = len(up4)

        rch_dn_ind1 = np.where(reaches.rch_id_down[0,:] == rm_rch[ind])[0]
        rch_dn_ind2 = np.where(reaches.rch_id_down[1,:] == rm_rch[ind])[0]
        rch_dn_ind3 = np.where(reaches.rch_id_down[2,:] == rm_rch[ind])[0]
        rch_dn_ind4 = np.where(reaches.rch_id_down[3,:] == rm_rch[ind])[0]
        if len(rch_dn_ind1) > 0:
            reaches.rch_id_down[0,rch_dn_ind1] = 0
            reaches.rch_id_down[:,rch_dn_ind1] = np.sort(reaches.rch_id_down[:,rch_dn_ind1], axis = 0)[::-1]
            dn1 = np.unique(reaches.rch_id_down[:,rch_dn_ind1]); dn1 = dn1[dn1>0]
            reaches.n_rch_down[rch_dn_ind1] = len(dn1)
        if len(rch_dn_ind2) > 0:
            reaches.rch_id_down[1,rch_dn_ind2] = 0
            reaches.rch_id_down[:,rch_dn_ind2] = np.sort(reaches.rch_id_down[:,rch_dn_ind2], axis = 0)[::-1]
            dn2 = np.unique(reaches.rch_id_down[:,rch_dn_ind2]); dn2 = dn2[dn2>0]
            reaches.n_rch_down[rch_dn_ind2] = len(dn2)
        if len(rch_dn_ind3) > 0:
            reaches.rch_id_down[2,rch_dn_ind3] = 0
            reaches.rch_id_down[:,rch_dn_ind3] = np.sort(reaches.rch_id_down[:,rch_dn_ind3], axis = 0)[::-1]
            dn3 = np.unique(reaches.rch_id_down[:,rch_dn_ind3]); dn3 = dn3[dn3>0]
            reaches.n_rch_down[rch_dn_ind3] = len(dn3)
        if len(rch_dn_ind4) > 0:
            reaches.rch_id_down[3,rch_dn_ind4] = 0
            reaches.rch_id_down[:,rch_dn_ind4] = np.sort(reaches.rch_id_down[:,rch_dn_ind4], axis = 0)[::-1]
            dn4 = np.unique(reaches.rch_id_down[:,rch_dn_ind4]); dn4 = dn4[dn4>0]
            reaches.n_rch_down[rch_dn_ind4] = len(dn4)

###############################################################################

def append_data(centerlines, nodes, reaches, 
                subcls, subnodes, subreaches):

    centerlines.cl_id = np.append(centerlines.cl_id, subcls.new_cl_id)
    centerlines.x = np.append(centerlines.x, subcls.lon)
    centerlines.y = np.append(centerlines.y, subcls.lat)
    centerlines.reach_id = np.append(centerlines.reach_id, subcls.new_reach_id, axis=1)
    centerlines.node_id = np.append(centerlines.node_id, subcls.new_node_id, axis=1)
    
    nodes.id = np.append(nodes.id, subnodes.id)
    nodes.cl_id = np.append(nodes.cl_id, subnodes.cl_id, axis=1)
    nodes.x = np.append(nodes.x, subnodes.x)
    nodes.y = np.append(nodes.y, subnodes.y)
    nodes.len = np.append(nodes.len, subnodes.len)
    nodes.wse = np.append(nodes.wse, subnodes.wse)
    nodes.wse_var = np.append(nodes.wse_var, subnodes.wse_var)
    nodes.wth = np.append(nodes.wth, subnodes.wth)
    nodes.wth_var = np.append(nodes.wth_var, subnodes.wth_var)
    nodes.grod = np.append(nodes.grod, subnodes.grod)
    nodes.grod_fid = np.append(nodes.grod_fid, subnodes.grod_fid)
    nodes.hfalls_fid = np.append(nodes.hfalls_fid, subnodes.hfalls_fid)
    nodes.nchan_max = np.append(nodes.nchan_max, subnodes.nchan_max)
    nodes.nchan_mod = np.append(nodes.nchan_mod, subnodes.nchan_mod)
    nodes.dist_out = np.append(nodes.dist_out, subnodes.dist_out)
    nodes.reach_id = np.append(nodes.reach_id, subnodes.reach_id)
    nodes.facc = np.append(nodes.facc, subnodes.facc)
    nodes.lakeflag = np.append(nodes.lakeflag, subnodes.lakeflag)
    nodes.wth_coef = np.append(nodes.wth_coef, subnodes.wth_coef)
    nodes.ext_dist_coef = np.append(nodes.ext_dist_coef, subnodes.ext_dist_coef)
    nodes.max_wth = np.append(nodes.max_wth, subnodes.max_wth)
    nodes.meand_len = np.append(nodes.meand_len, subnodes.meand_len)
    nodes.river_name = np.append(nodes.river_name, subnodes.river_name)
    nodes.manual_add = np.append(nodes.manual_add, subnodes.manual_add)
    nodes.sinuosity = np.append(nodes.sinuosity, subnodes.sinuosity)
    nodes.edit_flag = np.append(nodes.edit_flag, subnodes.edit_flag)
    nodes.trib_flag = np.append(nodes.trib_flag, subnodes.trib_flag)
    nodes.path_freq = np.append(nodes.path_freq, subnodes.path_freq)
    nodes.path_order = np.append(nodes.path_order, subnodes.path_order)
    nodes.path_segs = np.append(nodes.path_segs, subnodes.path_segs)
    nodes.main_side = np.append(nodes.main_side, subnodes.main_side)
    nodes.strm_order = np.append(nodes.strm_order, subnodes.strm_order)
    nodes.end_rch = np.append(nodes.end_rch, subnodes.end_rch)
    nodes.network = np.append(nodes.network, subnodes.network)
    nodes.add_flag = np.append(nodes.add_flag, subnodes.add_flag)

    reaches.id = np.append(reaches.id, subreaches.id)
    reaches.cl_id = np.append(reaches.cl_id, subreaches.cl_id, axis=1)
    reaches.x = np.append(reaches.x, subreaches.x)
    reaches.x_min = np.append(reaches.x_min, subreaches.x_min)
    reaches.x_max = np.append(reaches.x_max, subreaches.x_max)
    reaches.y = np.append(reaches.y, subreaches.y)
    reaches.y_min = np.append(reaches.y_min, subreaches.y_min)
    reaches.y_max = np.append(reaches.y_max, subreaches.y_max)
    reaches.len = np.append(reaches.len, subreaches.len)
    reaches.wse = np.append(reaches.wse, subreaches.wse)
    reaches.wse_var = np.append(reaches.wse_var, subreaches.wse_var)
    reaches.wth = np.append(reaches.wth, subreaches.wth)
    reaches.wth_var = np.append(reaches.wth_var, subreaches.wth_var)
    reaches.slope = np.append(reaches.slope, subreaches.slope)
    reaches.rch_n_nodes = np.append(reaches.rch_n_nodes, subreaches.rch_n_nodes)
    reaches.grod = np.append(reaches.grod, subreaches.grod)
    reaches.grod_fid = np.append(reaches.grod_fid, subreaches.grod_fid)
    reaches.hfalls_fid = np.append(reaches.hfalls_fid, subreaches.hfalls_fid)
    reaches.lakeflag = np.append(reaches.lakeflag, subreaches.lakeflag)
    reaches.nchan_max = np.append(reaches.nchan_max, subreaches.nchan_max)
    reaches.nchan_mod = np.append(reaches.nchan_mod, subreaches.nchan_mod)
    reaches.dist_out = np.append(reaches.dist_out, subreaches.dist_out)
    reaches.n_rch_up = np.append(reaches.n_rch_up, subreaches.n_rch_up)
    reaches.n_rch_down = np.append(reaches.n_rch_down, subreaches.n_rch_down)
    reaches.rch_id_up = np.append(reaches.rch_id_up, subreaches.rch_id_up, axis=1)
    reaches.rch_id_down = np.append(reaches.rch_id_down, subreaches.rch_id_down, axis=1)
    reaches.max_obs = np.append(reaches.max_obs, subreaches.max_obs)
    reaches.orbits = np.append(reaches.orbits, subreaches.orbits, axis=1)
    reaches.facc = np.append(reaches.facc, subreaches.facc)
    reaches.iceflag = np.append(reaches.iceflag, subreaches.iceflag, axis=1)
    reaches.max_wth = np.append(reaches.max_wth, subreaches.max_wth)
    reaches.river_name = np.append(reaches.river_name, subreaches.river_name)
    reaches.low_slope = np.append(reaches.low_slope, subreaches.low_slope)
    reaches.edit_flag = np.append(reaches.edit_flag, subreaches.edit_flag)
    reaches.trib_flag = np.append(reaches.trib_flag, subreaches.trib_flag)
    reaches.path_freq = np.append(reaches.path_freq, subreaches.path_freq)
    reaches.path_order = np.append(reaches.path_order, subreaches.path_order)
    reaches.path_segs = np.append(reaches.path_segs, subreaches.path_segs)
    reaches.main_side = np.append(reaches.main_side, subreaches.main_side)
    reaches.strm_order = np.append(reaches.strm_order, subreaches.strm_order)
    reaches.end_rch = np.append(reaches.end_rch, subreaches.end_rch)
    reaches.network = np.append(reaches.network, subreaches.network)
    reaches.add_flag = np.append(reaches.add_flag, subreaches.add_flag)
    
###############################################################################

def define_geometry(unq_rch, reach_id, cl_x, cl_y, cl_id, common, max_dist, region):
    geom = []
    rm_ind = []
    connections = np.zeros([reach_id.shape[0], reach_id.shape[1]], dtype=int)
    for ind in list(range(len(unq_rch))):
        # print(ind, len(unq_rch)-1)
        in_rch = np.where(reach_id[0,:] == unq_rch[ind])[0]
        sort_ind = in_rch[np.argsort(cl_id[in_rch])]
        x_coords = cl_x[sort_ind]
        y_coords = cl_y[sort_ind]

        #ultimatley don't want to have to use this...
        # if len(in_rch) == 0:
        #     print(unq_rch[ind], 'no centerline points')
        #     continue
     
        #appending neighboring reach endpoints to coordinates
        in_rch_up_dn = []
        for ngh in list(range(1,4)):
            neighbors = np.where(reach_id[ngh,:]==unq_rch[ind])[0]
            keep = np.where(connections[ngh,neighbors] == 0)[0]
            in_rch_up_dn.append(neighbors[keep])
        #formating into single list.
        in_rch_up_dn = np.unique([j for sub in in_rch_up_dn for j in sub]) #reach_id[0,in_rch_up_dn]
            
        #loop through and find what ends each point belong to.
        if len(in_rch_up_dn) > 0:
            end1_dist = []; end2_dist = []
            end1_pt = []; end2_pt = []
            end1_x = []; end2_x = []
            end1_y = []; end2_y = []
            end1_flag = []; end2_flag = []
            for ct in list(range(len(in_rch_up_dn))):
                x_pt = cl_x[in_rch_up_dn[ct]]
                y_pt = cl_y[in_rch_up_dn[ct]]
                if region == 'AS' and x_pt < 0 and np.min([cl_x[sort_ind[0]], cl_x[sort_ind[-1]]]) > 0:
                    print(unq_rch[ind])
                    continue
                elif region == 'AS' and x_pt > 0 and np.min([cl_x[sort_ind[0]], cl_x[sort_ind[-1]]]) < 0:
                    print(unq_rch[ind])
                    continue
                else:
                    #distance to first and last point. 
                    coords_1 = (y_pt, x_pt)
                    coords_2 = (cl_y[sort_ind[0]], cl_x[sort_ind[0]])
                    coords_3 = (cl_y[sort_ind[-1]], cl_x[sort_ind[-1]])
                    d1 = geopy.distance.geodesic(coords_1, coords_2).m
                    d2 = geopy.distance.geodesic(coords_1, coords_3).m
                        
                    if d1 < d2:
                        end1_pt.append(in_rch_up_dn[ct])
                        end1_dist.append(d1)
                        end1_x.append(x_pt)
                        end1_y.append(y_pt)
                        end1_flag.append(common[in_rch_up_dn[ct]])
                    if d1 > d2:
                        end2_pt.append(in_rch_up_dn[ct])
                        end2_dist.append(d2)
                        end2_x.append(x_pt)
                        end2_y.append(y_pt)
                        end2_flag.append(common[in_rch_up_dn[ct]])

            #append coords to ends
            if len(end1_pt) > 0: #reach_id[:,end1_pt]
                #if point has two neighbors it is a common reach and should be skipped.
                if common[sort_ind[0]] == 1: # len(end1_pt) > 1
                    x_coords = x_coords
                    y_coords = y_coords
                
                else:
                    end1_pt = np.array(end1_pt)
                    end1_dist = np.array(end1_dist)
                    end1_x = np.array(end1_x)
                    end1_y = np.array(end1_y)
                    end1_flag = np.array(end1_flag)
                    sort_ind1 = np.argsort(end1_dist)
                    end1_pt = end1_pt[sort_ind1]
                    end1_dist = end1_dist[sort_ind1]
                    end1_x = end1_x[sort_ind1]
                    end1_y = end1_y[sort_ind1]
                    end1_flag = end1_flag[sort_ind1]
                    flag1 = np.where(end1_flag == 1)[0]
                    if len(flag1) > 0: 
                        idx1 = flag1[0] 
                    else: 
                        idx1 = 0
                        
                    if np.min(end1_dist) <= max_dist:
                        x_coords = np.insert(x_coords, 0, end1_x[idx1], axis=0)
                        y_coords = np.insert(y_coords, 0, end1_y[idx1], axis=0)
                    else:
                        ngh_x = cl_x[np.where(reach_id[0,:] == reach_id[0,end1_pt[idx1]])]
                        ngh_y = cl_y[np.where(reach_id[0,:] == reach_id[0,end1_pt[idx1]])]
                        d=[]
                        for c in list(range(len(ngh_x))):
                            temp_coords = (ngh_y[c], ngh_x[c])
                            d.append(geopy.distance.geodesic(coords_2, temp_coords).m)
                        if np.min(d) <= max_dist:
                            append_x = ngh_x[np.where(d == np.min(d))]
                            append_y = ngh_y[np.where(d == np.min(d))]
                            x_coords = np.insert(x_coords, 0, append_x[0], axis=0)
                            y_coords = np.insert(y_coords, 0, append_y[0], axis=0)
                    #flag current reach for neighbors. MAY NEED TO HAPPEN ANYWAY...
                    ngh1 = reach_id[0,end1_pt[idx1]]
                    col1 = np.where(reach_id[1,:]== ngh1)[0]
                    col2 = np.where(reach_id[2,:]== ngh1)[0]
                    col3 = np.where(reach_id[3,:]== ngh1)[0] 
                    if unq_rch[ind] in reach_id[0,col1]:
                        c = np.where(reach_id[0,col1] == unq_rch[ind])[0]
                        connections[1,col1[c]] = 1
                    if unq_rch[ind] in reach_id[0,col2]:
                        c = np.where(reach_id[0,col2] == unq_rch[ind])[0]
                        connections[2,col2[c]] = 1
                    if unq_rch[ind] in reach_id[0,col3]:
                        c = np.where(reach_id[0,col3] == unq_rch[ind])[0]
                        connections[3,col3[c]] = 1

            if len(end2_pt) > 0: #reach_id[:,end2_pt]
                #if point has two neighbors it is a common reach and should be skipped.
                if common[sort_ind[-1]] == 1: # len(end2_pt) > 1
                    x_coords = x_coords
                    y_coords = y_coords
                
                else:
                    end2_pt = np.array(end2_pt)
                    end2_dist = np.array(end2_dist)
                    end2_x = np.array(end2_x)
                    end2_y = np.array(end2_y)
                    end2_flag = np.array(end2_flag)
                    sort_ind2 = np.argsort(end2_dist)
                    end2_pt = end2_pt[sort_ind2]
                    end2_dist = end2_dist[sort_ind2]
                    end2_x = end2_x[sort_ind2]
                    end2_y = end2_y[sort_ind2]
                    end2_flag = end2_flag[sort_ind2]
                    flag2 = np.where(end2_flag == 1)[0]
                    if len(flag2) > 0: 
                        idx2 = flag2[0] 
                    else: 
                        idx2 = 0

                    if np.min(end2_dist) < max_dist:
                        x_coords = np.insert(x_coords, len(x_coords), end2_x[idx2], axis=0)
                        y_coords = np.insert(y_coords, len(y_coords), end2_y[idx2], axis=0)
                    else:
                        ngh_x = cl_x[np.where(reach_id[0,:] == reach_id[0,end2_pt[idx2]])]
                        ngh_y = cl_y[np.where(reach_id[0,:] == reach_id[0,end2_pt[idx2]])]
                        d=[]
                        for c in list(range(len(ngh_x))):
                            temp_coords = (ngh_y[c], ngh_x[c])
                            d.append(geopy.distance.geodesic(coords_3, temp_coords).m)
                        if np.min(d) <= max_dist:
                            append_x = ngh_x[np.where(d == np.min(d))]
                            append_y = ngh_y[np.where(d == np.min(d))]
                            x_coords = np.insert(x_coords, len(x_coords), append_x[0], axis=0)
                            y_coords = np.insert(y_coords, len(y_coords), append_y[0], axis=0)
                    #flag current reach for neighbors.
                    ngh2 = reach_id[0,end2_pt[idx2]]
                    col1 = np.where(reach_id[1,:]== ngh2)[0]
                    col2 = np.where(reach_id[2,:]== ngh2)[0]
                    col3 = np.where(reach_id[3,:]== ngh2)[0] 
                    if unq_rch[ind] in reach_id[0,col1]:
                        c = np.where(reach_id[0,col1] == unq_rch[ind])[0]
                        connections[1,col1[c]] = 1
                    if unq_rch[ind] in reach_id[0,col2]:
                        c = np.where(reach_id[0,col2] == unq_rch[ind])[0]
                        connections[2,col2[c]] = 1
                    if unq_rch[ind] in reach_id[0,col3]:
                        c = np.where(reach_id[0,col3] == unq_rch[ind])[0]
                        connections[3,col3[c]] = 1

        pts = GeoSeries(map(Point, zip(x_coords, y_coords)))
        if len(pts) <= 1:
            rm_ind.append(ind)
            continue
        else:
            line = LineString(pts.tolist())
            geom.append(line) 

    return geom, rm_ind

###############################################################################

def format_rch_attr(reaches):
    #reformat multi-dimensional variables
    rch_type = np.array([int(str(rch)[-1]) for rch in reaches.id])
    rch_id_up = []; rch_id_dn = []; swot_orbits = []
    for ind in list(range(len(rch_type))):
        rch_id_up.append(str(reaches.rch_id_up[ind,np.where(reaches.rch_id_up[ind,:] > 0)[0]])[1:-1])
        rch_id_dn.append(str(reaches.rch_id_down[ind,np.where(reaches.rch_id_down[ind,:] > 0)[0]])[1:-1])
        swot_orbits.append(str(reaches.orbits[ind,np.where(reaches.orbits[ind,:] > 0)[0]])[1:-1])

    return rch_type, rch_id_up, rch_id_dn, swot_orbits

###############################################################################

def write_rchs(reaches, geom, rm_ind, paths):
    start_all = time.time()

    #determine outpaths.
    outpath_gpkg = paths['gpkg_dir']
    outpath_shp = paths['shp_dir']  
    
    #format multidimensional attributes.
    rch_type, rch_id_up, rch_id_dn, swot_orbits = format_rch_attr(reaches)

    #create initial GeoDataFrame.
    rch_df = gp.GeoDataFrame([
        reaches.x,
        reaches.y,
        reaches.id,
        reaches.len,
        reaches.rch_n_nodes,
        reaches.wse,
        reaches.wse_var,
        reaches.wth,
        reaches.wth_var,
        reaches.facc,
        reaches.n_chan_max,
        reaches.n_chan_mod,
        reaches.grod,
        reaches.grod_id,
        reaches.hfalls_id,
        reaches.slope,
        reaches.dist_out,
        reaches.lakeflag,
        reaches.max_wth,
        reaches.n_rch_up,
        reaches.n_rch_down,
        rch_id_up,
        rch_id_dn,
        swot_orbits,
        reaches.max_obs,
        rch_type,
        reaches.river_name,
        reaches.edit_flag,
        reaches.trib_flag,
        reaches.path_freq,
        reaches.path_order,
        reaches.path_segs,
        reaches.main_side,
        reaches.strm_order,
        reaches.end_rch,
        reaches.network,
    ]).T

    #rename columns.
    rch_df.rename(
        columns={
            0:"x",
            1:"y",
            2:"reach_id",
            3:"reach_len",
            4:"n_nodes",
            5:"wse",
            6:"wse_var",
            7:"width",
            8:"width_var",
            9:"facc",
            10:"n_chan_max",
            11:"n_chan_mod",
            12:"obstr_type",
            13:"grod_id",
            14:"hfalls_id",
            15:"slope",
            16:"dist_out",
            17:"lakeflag",
            18:"max_width",
            19:"n_rch_up",
            20:"n_rch_dn",
            21:"rch_id_up",
            22:"rch_id_dn",
            23:"swot_orbit",
            24:"swot_obs",
            25:"type",
            26:"river_name",
            27:"edit_flag",
            28:"trib_flag",
            29:"path_freq",
            30:"path_order",
            31:"path_segs",
            32:"main_side",
            33:"strm_order",
            34:"end_reach",
            35:"network"
            },inplace=True)

    #removing rows where reach was only one point.
    rch_df.drop(rm_ind, inplace=True)
    #update data types
    rch_df = rch_df.apply(pd.to_numeric, errors='ignore') # rch_df.dtypes
    #add geometry column and define crs. 
    rch_df['geometry'] = geom
    rch_df = gp.GeoDataFrame(rch_df)
    rch_df.set_geometry(col='geometry') #removed "inplace=True" option on leopold. 
    rch_df = rch_df.set_crs(4326, allow_override=True)

    print('Writing GeoPackage File')
    # write geopackage (continental scale)
    start = time.time()
    outgpkg = outpath_gpkg+paths['gpkg_rch_fn']
    rch_df.to_file(outgpkg, driver='GPKG', layer='reaches')
    end = time.time()
    print('Finished GPKG in: '+str(np.round((end-start)/60,2))+' min')

    # write as shapefile per level2 basin.
    print('Writing Shapefiles')
    start = time.time()
    level2 = np.array([int(str(r)[0:2]) for r in rch_df['reach_id']])
    unq_l2 = np.unique(level2)
    rch_cp = rch_df.copy(); rch_cp['level2'] = level2
    outshp = outpath_shp + paths['shp_rch_fn']
    for lvl in list(range(len(unq_l2))):
        outshp = outshp.replace("XX",str(unq_l2[lvl]))
        subset = rch_cp[rch_cp['level2'] == unq_l2[lvl]]
        subset = subset.drop(columns=['level2'])
        subset.to_file(outshp)
        del(subset)
    end = time.time()
    end_all = time.time()
    print('Finished SHPs in: '+str(np.round((end-start)/60,2))+' min')
    print('Finished All in: '+str(np.round((end_all-start_all)/60,2))+' min')

###############################################################################
    
def write_nodes(nodes, paths):

    #determine outpaths.
    outpath_gpkg = paths['gpkg_dir']
    outpath_shp = paths['shp_dir']

    node_type = np.array([int(str(rch)[-1]) for rch in nodes.id])
    node_df = gp.GeoDataFrame([
        nodes.x,
        nodes.y,
        nodes.id,
        nodes.len,
        nodes.reach_id,
        nodes.wse,
        nodes.wse_var,
        nodes.wth,
        nodes.wth_var,
        nodes.facc,
        nodes.n_chan_max,
        nodes.n_chan_mod,
        nodes.grod,
        nodes.grod_id,
        nodes.hfalls_id,
        nodes.dist_out,
        nodes.lakeflag,
        nodes.max_wth,
        nodes.meand_len,
        nodes.sinuosity,
        node_type,
        nodes.river_name,
        nodes.edit_flag,
        nodes.trib_flag,
        nodes.path_freq,
        nodes.path_order,
        nodes.path_segs,
        nodes.main_side,
        nodes.strm_order,
        nodes.end_rch,
        nodes.network,
    ]).T

    #rename columns.
    node_df.rename(
        columns={
            0:"x",
            1:"y",
            2:"node_id",
            3:"node_len",
            4:"reach_id",
            5:"wse",
            6:"wse_var",
            7:"width",
            8:"width_var",
            9:"facc",
            10:"n_chan_max",
            11:"n_chan_mod",
            12:"obstr_type",
            13:"grod_id",
            14:"hfalls_id",
            15:"dist_out",
            16:"lakeflag",
            17:"max_width",
            18:"meand_len",
            19:"sinuosity",
            20:"type",
            21:"river_name",
            22:"edit_flag",
            23:"trib_flag",
            24:"path_freq",
            25:"path_order",
            26:"path_segs",
            27:"main_side",
            28:"strm_order",
            29:"end_reach",
            30:"network",
            },inplace=True)

    node_df = node_df.apply(pd.to_numeric, errors='ignore') # node_df.dtypes
    geom = gp.GeoSeries(map(Point, zip(nodes.x, nodes.y)))
    node_df['geometry'] = geom
    node_df = gp.GeoDataFrame(node_df)
    node_df.set_geometry(col='geometry')
    node_df = node_df.set_crs(4326, allow_override=True)

    print('Writing GeoPackage File')
    #write geopackage (continental scale)
    start = time.time()
    node_df.to_file(outpath_gpkg+paths['gpkg_node_fn'], driver='GPKG', layer='nodes')
    end = time.time()
    print('Finished GPKG in: '+str(np.round((end-start)/60,2))+' min')

    #write as shapefile per level2 basin.
    start = time.time()
    level2 = np.array([int(str(n)[0:2]) for n in node_df['node_id']])
    unq_l2 = np.unique(level2)
    nodes_cp = node_df.copy(); nodes_cp['level2'] = level2
    outshp = outpath_shp + paths['shp_node_fn']
    for lvl in list(range(len(unq_l2))):
        print(unq_l2[lvl])
        outshp = outshp.replace("XX",str(unq_l2[lvl]))
        subset = nodes_cp[nodes_cp['level2'] == unq_l2[lvl]]
        subset = subset.drop(columns=['level2'])
        subset.to_file(outshp)
        del(subset)
    end = time.time()
    print('Finished SHPs in: '+str(np.round((end-start)/60,2))+' min')
  
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
    node_add_flag = node_grp.createVariable(
        'add_flag', 'i4', ('num_nodes',), fill_value=-9999.)

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
    rch_add_flag = rch_grp.createVariable(
        'add_flag', 'i4', ('num_reaches',), fill_value=-9999.)
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
    node_add_flag[:] = nodes.add_flag[node_sort]

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
    rch_add_flag[:] = reaches.add_flag[rch_sort]
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

def write_con_nc(centerlines, outfile):

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

    # dimensions
    #root_grp.createDimension('d1', 2)
    cl_grp.createDimension('num_points', len(centerlines.cl_id))
    cl_grp.createDimension('num_domains', 4)

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
    common = cl_grp.createVariable(
        'common', 'i4', ('num_points',), fill_value=-9999.)

    # saving data
    print("saving nc")

    # centerline data
    cl_id[:] = centerlines.cl_id
    cl_x[:] = centerlines.x
    cl_y[:] = centerlines.y
    reach_id[:,:] = centerlines.neighbors
    common[:] = centerlines.common

    root_grp.close()

    end = time.time()

    print("Ended Saving NetCDF in: ", str(np.round((end-start)/60, 2)), " min")

    return outfile

###############################################################################