# -*- coding: utf-8 -*-
"""
Created on Wed Jun 02 07:56:10 2021

@author: ealtenau
"""

from __future__ import division
#import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import time

###############################################################################

class Object(object):
    """
    FUNCTION:
        Creates class object to assign attributes to.
    """
    pass 

###############################################################################

def write_database_nc(sword, outfile):

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

    # groups
    cl_grp = root_grp.createGroup('centerlines')
    rch_grp = root_grp.createGroup('reaches')

    # dimensions
    #root_grp.createDimension('d1', 2)
    cl_grp.createDimension('num_points', len(sword.cl_id))
    cl_grp.createDimension('num_domains', 4)

    rch_grp.createDimension('num_reaches', len(sword.rch_id))
    rch_grp.createDimension('num_domains', 4)
     
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
   
    # reach variables
    Reach_ID = rch_grp.createVariable(
        'reach_id', 'i8', ('num_reaches',), fill_value=-9999.)
    Reach_ID.format = 'CBBBBBRRRRT'
    rch_swot_obs = rch_grp.createVariable(
        'swot_obs', 'i4', ('num_reaches',), fill_value=-9999.)
    n_rch_up = rch_grp.createVariable(
        'n_rch_up', 'i4', ('num_reaches',), fill_value=-9999.)
    n_rch_down= rch_grp.createVariable(
        'n_rch_down', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_id_up = rch_grp.createVariable(
        'rch_id_up', 'i8', ('num_domains','num_reaches'), fill_value=-9999.)
    rch_id_down = rch_grp.createVariable(
        'rch_id_dn', 'i8', ('num_domains','num_reaches'), fill_value=-9999.)
    
    # saving data
    print("saving nc")

    # root group data
    #cont_str = nc.stringtochar(np.array(['NA'], 'S2'))
    #Name[:] = cont_str

    # centerline data
    cl_id[:] = sword.cl_id
    cl_x[:] = sword.cl_x
    cl_y[:] = sword.cl_y
    reach_id[:,:] = sword.cl_rch_id

    # reach data
    Reach_ID[:] = sword.rch_id
    rch_swot_obs[:] = sword.ice_free_obs
    n_rch_up[:] = sword.n_rch_up
    n_rch_down[:] = sword.n_rch_dn
    rch_id_up[:,:] = sword.rch_id_up
    rch_id_down[:,:] = sword.rch_id_dn

    root_grp.close()

    end = time.time()

    print("Ended Saving Main NetCDF in: ", str(np.round((end-start)/60, 2)), " min")

    return outfile


###############################################################################
###############################################################################
###############################################################################

region = 'oc'
fn1 = 'E:/Users/Elizabeth Humphries/Documents/SWORD/For_Server/outputs/Reaches_Nodes_v10/netcdf/'+region+'_sword_v10.nc'
outpath = 'E:/Users/Elizabeth Humphries/Documents/SWORD/For_Server/outputs/SWOT_Coverage/Annual_IceFreeObs/netcdf/'+region+'_icefree_obs_v10.nc'
data1 = nc.Dataset(fn1)

rch_id = data1.groups['reaches'].variables['reach_id'][:]
rch_obs = data1.groups['reaches'].variables['swot_obs'][:]
rch_ice = data1.groups['reaches'].variables['iceflag'][:]
rch_n_rch_up = data1.groups['reaches'].variables['n_rch_up'][:]
rch_n_rch_dn = data1.groups['reaches'].variables['n_rch_down'][:]
rch_rch_id_up = data1.groups['reaches'].variables['rch_id_up'][:]
rch_rch_id_dn = data1.groups['reaches'].variables['rch_id_dn'][:]
cl_id = data1.groups['centerlines'].variables['cl_id'][:]
cl_rch_id = data1.groups['centerlines'].variables['reach_id'][:]
cl_x = data1.groups['centerlines'].variables['x'][:]
cl_y = data1.groups['centerlines'].variables['y'][:]
data1.close()

ice_free_obs = np.zeros(len(rch_id))
rchs = rch_ice.shape[1]
for ind in list(range(rchs)):
    ice_free_days = len(np.where(rch_ice[:,ind] == 0)[0])
    ice_free_per_cycle = ice_free_days/20.86
    ice_free_obs[ind] = int(rch_obs[ind]*ice_free_per_cycle)

    
sword = Object()
sword.cl_id = cl_id
sword.cl_x = cl_x
sword.cl_y = cl_y
sword.cl_rch_id = cl_rch_id
sword.rch_id = rch_id
sword.ice_free_obs = ice_free_obs
sword.n_rch_up = rch_n_rch_up
sword.n_rch_dn = rch_n_rch_dn 
sword.rch_id_up = rch_rch_id_up
sword.rch_id_dn = rch_rch_id_dn

write_database_nc(sword, outpath)

