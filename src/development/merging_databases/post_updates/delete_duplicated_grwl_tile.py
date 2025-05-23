# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:05:27 2020
"""

from __future__ import division
import os
main_dir = os.getcwd()
import numpy as np
import netCDF4 as nc
import time 

###############################################################################

class Object(object):
    """
    FUNCTION:
        Creates class object to assign attributes to.
    """
    pass 

###############################################################################

def read_merge_netcdf(filename):
    
    """
    FUNCTION:
        Reads in attributes from the merged database and assigns them to an 
        object. 

    INPUTS
        filename -- Merged database netcdf file. 

    OUTPUTS
        data -- Object containing attributes from the merged database. 
    """
    
    data = Object()
    new = nc.Dataset(filename)
    data.lon = new.groups['centerlines'].variables['x'][:]
    data.lat = new.groups['centerlines'].variables['y'][:]
    data.x = new.groups['centerlines'].variables['easting'][:]
    data.y = new.groups['centerlines'].variables['northing'][:]
    data.seg = new.groups['centerlines'].variables['segID'][:]
    data.ind = new.groups['centerlines'].variables['segInd'][:]
    data.id = new.groups['centerlines'].variables['cl_id'][:]
    data.segDist = new.groups['centerlines'].variables['segDist'][:]
    data.wth = new.groups['centerlines'].variables['p_width'][:]
    data.elv = new.groups['centerlines'].variables['p_height'][:]
    data.facc = new.groups['centerlines'].variables['flowacc'][:]
    data.lake = new.groups['centerlines'].variables['lakeflag'][:] 
    data.delta = new.groups['centerlines'].variables['deltaflag'][:]
    data.nchan = new.groups['centerlines'].variables['nchan'][:]
    data.grand = new.groups['centerlines'].variables['grand_id'][:]
    data.grod = new.groups['centerlines'].variables['grod_id'][:]
    data.basins = new.groups['centerlines'].variables['basin_code'][:]
    data.manual = new.groups['centerlines'].variables['manual_add'][:]
    data.num_obs = new.groups['centerlines'].variables['number_obs'][:]
    data.orbits = new.groups['centerlines'].variables['orbits'][:]
    #data.lake_id = new.groups['centerlines'].variables['lake_id'][:]
    data.tile = new.groups['centerlines'].variables['grwl_tile'][:]
    data.eps = new.groups['centerlines'].variables['endpoints'][:]
    new.close()

    return data

###############################################################################

def save_merged_nc(merge, outfile):

    """
    FUNCTION:
        Writes filtered merged NetCDF. Datasets combined include: GRWL,
        MERIT Hydro, GROD, GRanD, HydroBASINS, Global Deltas, and SWOT Track
        information.

    INPUTS
        merged -- Object containing merged attributes for the GRWL centerline.
        outfile -- Outpath directory to write the NetCDF file.

    OUTPUTS
        Merged NetCDF file.
    """

    # global attributes
    root_grp = nc.Dataset(outfile, 'w', format='NETCDF4')
    root_grp.x_min = np.min(merge.lon)
    root_grp.x_max = np.max(merge.lon)
    root_grp.y_min = np.min(merge.lat)
    root_grp.y_max = np.max(merge.lat)
    root_grp.production_date = time.strftime("%d-%b-%Y %H:%M:%S", time.gmtime()) #utc time

    # subgroups
    cl_grp = root_grp.createGroup('centerlines')

    # dimensions
    maxview = 25
    root_grp.createDimension('ID', 2)
    cl_grp.createDimension('num_points', len(merge.id))
    cl_grp.createDimension('orbit', maxview)

    ### variables and units

    # root group variables
    Name = root_grp.createVariable('Name', 'S1', ('ID'))
    Name._Encoding = 'ascii'

    # centerline variables
    cl_id = cl_grp.createVariable(
        'cl_id', 'i8', ('num_points',), fill_value=-9999.)
    x = cl_grp.createVariable(
        'x', 'f8', ('num_points',), fill_value=-9999.)
    x.units = 'degrees east'
    y = cl_grp.createVariable(
        'y', 'f8', ('num_points',), fill_value=-9999.)
    y.units = 'degrees north'
    easting = cl_grp.createVariable(
        'easting', 'f8', ('num_points',), fill_value=-9999.)
    easting.units = 'm'
    northing = cl_grp.createVariable(
        'northing', 'f8', ('num_points',), fill_value=-9999.)
    northing.units = 'm'
    segID = cl_grp.createVariable(
        'segID', 'i4', ('num_points',), fill_value=-9999.)
    segInd = cl_grp.createVariable(
        'segInd', 'f8', ('num_points',), fill_value=-9999.)
    segDist= cl_grp.createVariable(
        'segDist', 'f8', ('num_points',), fill_value=-9999.)
    segDist.units = 'm'
    p_width = cl_grp.createVariable(
        'p_width', 'f8', ('num_points',), fill_value=-9999.)
    p_width.units = 'm'
    p_height = cl_grp.createVariable(
        'p_height', 'f8', ('num_points',), fill_value=-9999.)
    p_height.units = 'm'
    flowacc = cl_grp.createVariable(
        'flowacc', 'f8', ('num_points',), fill_value=-9999.)
    flowacc.units = 'km^2'
    lakeflag = cl_grp.createVariable(
        'lakeflag', 'i4', ('num_points',), fill_value=-9999.)
    deltaflag = cl_grp.createVariable(
        'deltaflag', 'i4', ('num_points',), fill_value=-9999.)
    nchan = cl_grp.createVariable(
        'nchan', 'i4', ('num_points',), fill_value=-9999.)
    grand_id = cl_grp.createVariable(
        'grand_id', 'i4', ('num_points',), fill_value=-9999.)
    grod_id = cl_grp.createVariable(
        'grod_id', 'i4', ('num_points',), fill_value=-9999.)
    basin_code = cl_grp.createVariable(
        'basin_code', 'i4', ('num_points',), fill_value=-9999.)
    manual_add = cl_grp.createVariable(
        'manual_add', 'i4', ('num_points',), fill_value=-9999.)
    number_obs = cl_grp.createVariable(
        'number_obs', 'i4', ('num_points',), fill_value=-9999.)
    orbits = cl_grp.createVariable(
        'orbits', 'i4', ('num_points','orbit'), fill_value=-9999.)
    grwl_tile = cl_grp.createVariable(
        'grwl_tile', 'S7', ('num_points',))
    grwl_tile._Encoding = 'ascii'
    endpoints = cl_grp.createVariable(
        'endpoints', 'i4', ('num_points',), fill_value=-9999.)
    #lake_id = cl_grp.createVariable(
        #'lake_id', 'i8', ('num_points',), fill_value=-9999.)
    #old_lakeflag = cl_grp.createVariable(
        #'old_lakeflag', 'i4', ('num_points',), fill_value=-9999.)

    # data
    print("saving nc")

    # root group data
    cont_str = nc.stringtochar(np.array(['NA'], 'S2'))
    Name[:] = cont_str

    # centerline data
    cl_id[:] = merge.id
    x[:] = merge.lon
    y[:] = merge.lat
    easting[:] = merge.x
    northing[:] = merge.y
    segInd[:] = merge.ind
    segID[:] = merge.seg
    segDist[:] = merge.segDist
    p_width[:] = merge.wth
    p_height[:] = merge.elv
    flowacc[:] = merge.facc
    lakeflag[:] = merge.lake
    deltaflag[:] = merge.delta
    nchan[:] = merge.nchan
    grand_id[:] = merge.grand
    grod_id[:] = merge.grod
    basin_code[:] = merge.basins
    manual_add[:] = merge.manual
    number_obs[:] = merge.num_obs
    orbits[:,:] = merge.orbits
    endpoints[:] = merge.eps
   
    #m_grwl_tile = np.array(merge.tile)
    grwl_tile[:] = merge.tile

    root_grp.close()
    
###############################################################################
###############################################################################
###############################################################################

fn = main_dir+'/data/outputs/Merged_Data/NA/NA_Merge_v06.nc'
outpath = main_dir+'/data/outputs/Merged_Data/NA/NA_Merge_v06_update.nc'
merge = read_merge_netcdf(fn)

remove = np.where(merge.tile == 'n16w102')[0]
#test = np.delete(merge.lon, remove, axis=0)

merge.lon = np.delete(merge.lon, remove, axis=0)
merge.lat = np.delete(merge.lat, remove, axis=0)
merge.x = np.delete(merge.x, remove, axis=0)
merge.y = np.delete(merge.y, remove, axis=0)
merge.seg = np.delete(merge.seg, remove, axis=0)
merge.ind = np.delete(merge.ind, remove, axis=0)
merge.id = np.delete(merge.id, remove, axis=0)
merge.segDist = np.delete(merge.segDist, remove, axis=0)
merge.wth = np.delete(merge.wth, remove, axis=0)
merge.elv = np.delete(merge.elv, remove, axis=0)
merge.facc = np.delete(merge.facc, remove, axis=0)
merge.lake = np.delete(merge.lake, remove, axis=0)
merge.delta = np.delete(merge.delta, remove, axis=0)
merge.nchan = np.delete(merge.nchan, remove, axis=0)
merge.grand = np.delete(merge.grand, remove, axis=0)
merge.grod = np.delete(merge.grod, remove, axis=0)
merge.basins = np.delete(merge.basins, remove, axis=0)
merge.manual = np.delete(merge.manual, remove, axis=0)
merge.num_obs = np.delete(merge.num_obs, remove, axis=0)
merge.orbits = np.delete(merge.orbits, remove, axis=0)
merge.tile = np.delete(merge.tile, remove, axis=0)
merge.eps = np.delete(merge.eps, remove, axis=0)

save_merged_nc(merge, outpath)






