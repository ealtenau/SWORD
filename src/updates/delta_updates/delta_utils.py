# -*- coding: utf-8 -*-
"""
Delta Utilities (delta_utils.py)
=======================================

Utilities for attaching auxillary data to delta 
polyline and node shapefiles. These tools are 
specific to the delta shapefiles provided by 
Dr. Paola Passalacqua's lab at UT Austin 
(https://sites.google.com/site/passalacquagroup/home). 

"""

from __future__ import division
import os
main_dir = os.getcwd()
import time
import numpy as np
from scipy import spatial as sp
import glob
import geopandas as gp
import netCDF4 as nc
from shapely.geometry import Point
import pandas as pd
from osgeo import ogr, gdal
from statistics import mode
import src.updates.geo_utils as geo 
import warnings
warnings.filterwarnings("ignore") #if code stops working may need to comment out to check warnings. 

###############################################################################

def format_topo_attributes(rch_id_up, rch_id_dn, n_rch_up, n_rch_dn):
    """
    Reformats string topological attributes into array format 
    for writing netCDF files.

    Parameters
    ----------
    rch_id_up: numpy.array()
        1-D array of upstream reach IDs in string format.
    rch_id_dn: numpy.array()
        1-D array of downstream reach IDs in string format.
    n_rch_up: numpy.array()
        Number of upstream reaches.
    n_rch_dn: numpy.array()
        Number of downstream reaches.

    Returns
    -------
    id_up_arr: numpy.array() [4,number of reaches]
        Array containing upstream reach IDs in separate rows. 
        Up to 4 upstream reaches. 
    id_dn_arr: numpy.array() [4,number of reaches]
        Array containing downstream reach IDs in separate rows. 
        Up to 4 downstream reaches. 

    """

    id_up_arr = np.zeros((4,len(n_rch_up)), dtype=int)
    id_dn_arr = np.zeros((4,len(n_rch_up)), dtype=int)
    for pt in list(range(len(n_rch_up))):
        if n_rch_up[pt] > 0:
            up_ids = np.array(rch_id_up[pt].split())
            id_up_arr[0:len(up_ids),pt] = up_ids
        if n_rch_dn[pt] > 0:
            dn_ids = np.array(rch_id_dn[pt].split())  
            id_dn_arr[0:len(dn_ids),pt] = dn_ids

    return id_up_arr, id_dn_arr

###############################################################################

def reverse_indexes(reaches, index):
    """
    Reverses indexes from the default upstream-to-downstream 
    order to downstream-to-upstream order.

    Parameters
    ----------
    reaches: numpy.array()
        Reach IDs. 
    index: numpy.array() 
        Unique ID for each point/coordinate.

    Returns
    -------
    new_indexes: numpy.array() 
        Reversed IDs for each point/coordinate. 
       
    """

    new_index = np.zeros(len(reaches))
    unq_rchs = np.unique(reaches)
    for r in list(range(len(unq_rchs))):
        rch = np.where(reaches == unq_rchs[r])[0]
        new_index[rch] = index[rch][::-1]

    return new_index

###############################################################################

def write_data_nc(data_obj, outfile):
    """
    Writes delta attributes as a netCDF file.

    Parameters
    ----------
    data_obj: obj
        Object containing delta centerline data and attached 
        auxillary attributes. 
    outfile: str 
        Filepath to write netCDF file. 

    Returns
    -------
    None. 
       
    """
    
    # global attributes
    root_grp = nc.Dataset(outfile, 'w', format='NETCDF4')
    root_grp.delta_file = data_obj.file
    root_grp.x_min = np.min(data_obj.x)
    root_grp.x_max = np.max(data_obj.x)
    root_grp.y_min = np.min(data_obj.x)
    root_grp.y_max = np.max(data_obj.x)
    root_grp.production_date = time.strftime("%d-%b-%Y %H:%M:%S", time.gmtime()) #utc time

    # subgroups
    cl_grp = root_grp.createGroup('centerlines')

    # dimensions
    cl_grp.createDimension('num_points', len(data_obj.x))
    cl_grp.createDimension('orbit', 200)
    cl_grp.createDimension('nghs', 4)

    ### variables and units
    # centerline variables
    x = cl_grp.createVariable('x', 'f8', ('num_points',), fill_value=-9999.)
    x.units = 'degrees east'
    y = cl_grp.createVariable('y', 'f8', ('num_points',), fill_value=-9999.)
    y.units = 'degrees north'
    segID = cl_grp.createVariable('segID', 'i8', ('num_points',), fill_value=-9999.)
    segInd = cl_grp.createVariable('segInd', 'i8', ('num_points',), fill_value=-9999.)
    #global attributes
    lakeflag = cl_grp.createVariable('lakeflag', 'i4', ('num_points',), fill_value=-9999.)
    deltas = cl_grp.createVariable('deltaflag', 'i4', ('num_points',), fill_value=-9999.)
    grand = cl_grp.createVariable('grand_id', 'i4', ('num_points',), fill_value=-9999.)
    grod_type = cl_grp.createVariable('grod_id', 'i4', ('num_points',), fill_value=-9999.)
    grod_id = cl_grp.createVariable('grod_fid', 'i8', ('num_points',), fill_value=-9999.)
    hfalls_id = cl_grp.createVariable('hfalls_fid', 'i8', ('num_points',), fill_value=-9999.)
    basin = cl_grp.createVariable('basin_code', 'i8', ('num_points',), fill_value=-9999.)
    swot_obs = cl_grp.createVariable('number_obs', 'i4', ('num_points',), fill_value=-9999.)
    orbits = cl_grp.createVariable('orbits', 'i4', ('orbit','num_points'), fill_value=-9999.)
    lake_id = cl_grp.createVariable('lake_id', 'i8', ('num_points',), fill_value=-9999.)
    #regional attributes
    cl_id = cl_grp.createVariable('cl_id', 'i8', ('num_points',), fill_value=-9999.)
    east = cl_grp.createVariable('easting', 'f8', ('num_points',), fill_value=-9999.)
    north = cl_grp.createVariable('northing', 'f8', ('num_points',), fill_value=-9999.)
    dist = cl_grp.createVariable('new_segDist', 'f8', ('num_points',), fill_value=-9999.)
    wth = cl_grp.createVariable('p_width', 'f8', ('num_points',), fill_value=-9999.)
    wse = cl_grp.createVariable('p_height', 'f8', ('num_points',), fill_value=-9999.)
    facc = cl_grp.createVariable('flowacc', 'f8', ('num_points',), fill_value=-9999.)
    nchan = cl_grp.createVariable('nchan', 'i4', ('num_points',), fill_value=-9999.)
    manual = cl_grp.createVariable('manual_add', 'i4', ('num_points',), fill_value=-9999.)
    eps = cl_grp.createVariable('endpoints', 'i4', ('num_points',), fill_value=-9999.)
    tile = cl_grp.createVariable('mh_tile', 'S7', ('num_points',), fill_value=-9999.)
    tile._Encoding = 'ascii'
    #delta specific attributes
    wth_var = cl_grp.createVariable('wth_var', 'f8', ('num_points',), fill_value=-9999.)
    max_width = cl_grp.createVariable('max_width', 'f8', ('num_points',), fill_value=-9999.)
    rch_id_up = cl_grp.createVariable('rch_id_up', 'i8', ('nghs','num_points',), fill_value=-9999.)
    rch_id_dn = cl_grp.createVariable('rch_id_down', 'i8', ('nghs','num_points',), fill_value=-9999.)
    n_rch_up = cl_grp.createVariable('n_rch_up', 'i4', ('num_points',), fill_value=-9999.)
    n_rch_dn = cl_grp.createVariable('n_rch_down', 'i4', ('num_points',), fill_value=-9999.)
        
    # saving data
    # centerline data
    x[:] = data_obj.x
    y[:] = data_obj.y
    segID[:] = data_obj.reach_id_R
    segInd[:] = data_obj.index
    lakeflag[:] = data_obj.lakeflag
    deltas[:] = data_obj.deltaflag
    grand[:] = data_obj.grand
    grod_type[:] = data_obj.obstr_type
    grod_id[:] = data_obj.grod_id
    hfalls_id[:] = data_obj.hfalls_id
    basin[:] = data_obj.basins_filt
    swot_obs[:] = data_obj.swot_obs
    orbits[:,:] = data_obj.swot_orbit
    lake_id[:] = data_obj.lakeid
    cl_id[:] = data_obj.cl_id
    east[:] = data_obj.east
    north[:] = data_obj.north
    dist[:] = data_obj.dist
    wth[:] = data_obj.width
    wse[:] = data_obj.wse
    facc[:] = data_obj.facc
    nchan[:] = data_obj.nchan
    manual[:] = data_obj.manual_add
    eps[:] = data_obj.ends
    tile[:] = data_obj.mh_tile
    wth_var[:] = data_obj.wth_var
    max_width[:] = data_obj.max_width
    rch_id_up[:,:] = data_obj.fmt_rch_id_up
    rch_id_dn[:,:] = data_obj.fmt_rch_id_dn
    n_rch_up[:] = data_obj.n_rch_up
    n_rch_dn[:] = data_obj.n_rch_down
        
    root_grp.close()

###############################################################################