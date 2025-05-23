# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 12:44:06 2020
"""

from __future__ import division
import os
main_dir = os.getcwd()
import utm
from pyproj import Proj
import numpy as np
from osgeo import gdal
from scipy import spatial as sp
import netCDF4 as nc
#import matplotlib.pyplot as plt 
import pandas as pd
import time

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
    nodes.grod = data.groups['nodes'].variables['grod_id'][:]
    nodes.nchan_max = data.groups['nodes'].variables['n_chan_max'][:]
    nodes.nchan_mod = data.groups['nodes'].variables['n_chan_mod'][:]
    nodes.dist_out = data.groups['nodes'].variables['dist_out'][:]
    nodes.reach_id = data.groups['nodes'].variables['reach_id'][:]
    nodes.wth_coef = data.groups['nodes'].variables['wth_coef'][:]
    
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
    reaches.grod = data.groups['reaches'].variables['grod_id'][:]
    reaches.lakeflag = data.groups['reaches'].variables['lakeflag'][:]
    reaches.nchan_max = data.groups['reaches'].variables['n_chan_max'][:]
    reaches.nchan_mod = data.groups['reaches'].variables['n_chan_mod'][:]
    reaches.dist_out = data.groups['reaches'].variables['dist_out'][:]
    reaches.n_rch_up = data.groups['reaches'].variables['n_rch_up'][:]
    reaches.n_rch_down = data.groups['reaches'].variables['n_rch_down'][:]
    reaches.rch_id_up = data.groups['reaches'].variables['rch_id_up'][:]
    reaches.rch_id_down = data.groups['reaches'].variables['rch_id_dn'][:]
    reaches.iceflag = data.groups['reaches'].variables['iceflag'][:]    

    reaches.h_break = data.groups['reaches']['area_fits'].variables['h_break'][:]
    reaches.w_break = data.groups['reaches']['area_fits'].variables['w_break'][:]
    reaches.hw_covariance = data.groups['reaches']['area_fits'].variables['hw_covariance'][:]
    reaches.h_err_stdev = data.groups['reaches']['area_fits'].variables['h_err_stdev'][:]
    reaches.w_err_stdev = data.groups['reaches']['area_fits'].variables['w_err_stdev'][:]
    reaches.h_w_nobs = data.groups['reaches']['area_fits'].variables['h_w_nobs'][:]
    reaches.fit_coeffs = data.groups['reaches']['area_fits'].variables['fit_coeffs'][:]
    reaches.med_flow_area = data.groups['reaches']['area_fits'].variables['med_flow_area'][:]
    
    reaches.metroman_abar = data.groups['reaches']['discharge_models'].variables['MetroMan_Abar'][:]
    reaches.metroman_na = data.groups['reaches']['discharge_models'].variables['MetroMan_na'][:]
    reaches.metroman_nb = data.groups['reaches']['discharge_models'].variables['MetroMan_nb'][:]
    reaches.bam_abar = data.groups['reaches']['discharge_models'].variables['BAM_Abar'][:]
    reaches.bam_n = data.groups['reaches']['discharge_models'].variables['BAM_n'][:]    
    
    return centerlines, nodes, reaches
    
###############################################################################

def write_database_nc(centerlines, reaches, nodes, region, outfile):
    
    """
    FUNCTION:
        Outputs the SWOT a priori river database (SWORD) information in netcdf
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
        NetCDF file. 
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
        'grod_id', 'i4', ('num_nodes',), fill_value=-9999.)
    node_dist_out = node_grp.createVariable(
        'dist_out', 'f8', ('num_nodes',), fill_value=-9999.)
    node_dist_out.units = 'meters'
    node_wth_coef = node_grp.createVariable(
        'wth_coef', 'f8', ('num_nodes',), fill_value=-9999.)
    node_ext_dist_coef = node_grp.createVariable(
        'ext_dist_coef', 'f8', ('num_nodes',), fill_value=-9999.)
    
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
    rch_chan_max = rch_grp.createVariable(
        'n_chan_max', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_chan_mod = rch_grp.createVariable(
        'n_chan_mod', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_grod_id = rch_grp.createVariable(
        'grod_id', 'i4', ('num_reaches',), fill_value=-9999.)
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
    # subgroup 1
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
    # discharge subgroups
    # MetroMan (qgrp1)
    metroman_Abar = sub_grp2.createVariable(
        'MetroMan_Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    metroman_Abar.units = 'meters'
    metroman_na = sub_grp2.createVariable(
        'MetroMan_na', 'f8', ('num_reaches',), fill_value=-9999.)
    metroman_nb = sub_grp2.createVariable(
        'MetroMan_nb', 'f8', ('num_reaches',), fill_value=-9999.)
    # BAM (qgrp2)
    bam_Abar = sub_grp2.createVariable(
        'BAM_Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    bam_Abar.units = 'meters'
    bam_n = sub_grp2.createVariable(
        'BAM_n', 'f8', ('num_reaches',), fill_value=-9999.)
    
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
    node_dist_out[:] = nodes.dist_out
    node_wth_coef[:] = nodes.wth_coef
    node_ext_dist_coef[:] = nodes.ext_dist_coef
    
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
    rch_chan_max[:] = reaches.nchan_max 
    rch_chan_mod[:] = reaches.nchan_mod
    rch_grod_id[:] = reaches.grod
    rch_slope[:] = reaches.slope
    rch_dist_out[:] = reaches.dist_out
    n_rch_up[:] = reaches.n_rch_up
    n_rch_down[:] = reaches.n_rch_down
    rch_id_up[:,:] = reaches.rch_id_up
    rch_id_down[:,:] = reaches.rch_id_down
    rch_lakeflag[:] = reaches.lakeflag
    rch_iceflag[:,:] = reaches.iceflag
    # subgroup1
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
    # subgroup2
    metroman_Abar[:] = reaches.metroman_abar
    metroman_na[:] = reaches.metroman_na
    metroman_nb[:] = reaches.metroman_nb   
    bam_Abar[:] = reaches.bam_abar
    bam_n[:] = reaches.bam_n
    
    root_grp.close()

    end = time.time()

    print("Ended Saving NetCDF in: ", str(np.round((end-start)/60, 2)), " min")

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
        utm_num -- UTM zone number (single number)
        utm_let -- UTM zone letter (single letter)
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

def read_tiff(filename):
    """
    Opens a TIFF file and returns the 2D image, 1D values, and 1D x and y coordinates, and image information.
    """
    file_info = gdal.Open(filename)
    vals = file_info.ReadAsArray().flatten()
    xul,xres,_,yul,_,yres=file_info.GetGeoTransform() #gives metadata
    x=np.array([xul+xres*c+xres/2. for r in range(file_info.RasterYSize) for c in range(file_info.RasterXSize)])
    y=np.array([yul+yres*r+yres/2. for r in range(file_info.RasterYSize) for c in range(file_info.RasterXSize)])
    return (vals, x, y)

###############################################################################
###############################################################################
###############################################################################

# define paths. 
fn_lakes = main_dir+'/data/inputs/lakes_near_rivers/shp/2543_dist_thresh_test.tif'
fn_nodes = main_dir+'/data/inputs/lakes_near_rivers/databases/na_sword_scene2534_wc05_v06.nc'
outpath = main_dir+'/data/inputs/lakes_near_rivers/s2543_dist_thresh_testing.csv'
out_nc = main_dir+'/data/inputs/lakes_near_rivers/databases/na_sword_scene2534_wc05_extdvar_v06.nc'

# read originial data.
centerlines, nodes, reaches = read_data(fn_nodes)

# read in lakes.
lake_ids, lx, ly = read_tiff(fn_lakes)

# reproject node coordinates to utm.
nx, ny, __, __ = reproject_utm(nodes.y, nodes.x)

# create type variable. 
ntype = np.zeros(len(nodes.id))
for ind in list(range(len(nodes.id))):
    ntype[ind] = np.int(np.str(nodes.id[ind])[13:14]) 

# finding 10 closest nodes to each lake pixel.    
lake_locs = np.where(lake_ids > 0)[0] 
node_pts = np.vstack((nx, ny)).T
lake_pts = np.vstack((lx[lake_locs], ly[lake_locs])).T
kdt = sp.cKDTree(node_pts)
dist, index = kdt.query(lake_pts, k = 10)

# edit the extreme distance coeficient values. 
uniq_ids = np.unique(index)
remove = np.where(ntype[uniq_ids] == 3)[0]
final_ids = np.delete(uniq_ids, remove)
dist_thresh = np.repeat(20, len(nodes.id)) 
dist_thresh[final_ids] = 1
nodes.ext_dist_coef = dist_thresh

#write csv.
data = pd.DataFrame(np.array([nodes.x, nodes.y, nodes.id, ntype, dist_thresh])).T
data.columns = ['lon', 'lat', 'id', 'type', 'dist_thresh']
data.to_csv(outpath)

#write netcdf.
region = 'NA'
write_database_nc(centerlines, reaches, nodes, region, out_nc)

'''
plt.scatter(nx, ny, c=dist_thresh, s = 5, edgecolors=None)
plt.scatter(lx[lake_locs], ly[lake_locs], c='red', s = 5, edgecolors=None)
'''
# test = nc.Dataset(out_nc)
# np.unique(test.groups['nodes'].variables['ext_dist_coef'][:])