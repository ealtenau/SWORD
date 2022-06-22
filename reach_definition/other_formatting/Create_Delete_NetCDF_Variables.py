# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 11:32:24 2021

@author: ealtenau
"""
import netCDF4 as nc
import numpy as np

region = 'EU'
#fn_sword = 'E:/Users/Elizabeth Humphries/Documents/SWORD/For_Server/outputs/Reaches_Nodes_v10/netcdf/'+region.lower()+'_sword_v10.nc'
fn_sword = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/outputs/Reaches_Nodes/netcdf/'+region.lower()+'_sword_v11.nc'

sword = nc.Dataset(fn_sword, 'r+')

n_lon = sword.groups['nodes'].variables['x'][:]
r_lon = sword.groups['reaches'].variables['x'][:]


### Create New Variable(s)

sword.groups['nodes'].createVariable('sinuosity', 'f8', ('num_nodes',), fill_value=-9999.)
sword.groups['nodes'].variables['sinuosity'][:] = np.zeros(len(n_lon))

sword.close()

'''
# NODES
node_manual_add = sword.groups['nodes'].createVariable(
    'manual_add', 'i4', ('num_nodes',), fill_value=-9999.)

node_meand_len = sword.groups['nodes'].createVariable(
    'meander_length', 'f8', ('num_nodes',), fill_value=-9999.)
node_meand_len.units = 'meters'

node_river_name = sword.groups['nodes'].createVariable(
    'river_name', 'S50', ('num_nodes',))
node_river_name._Encoding = 'ascii'
    
# REACHES
rch_river_name = sword.groups['reaches'].createVariable(
    'river_name', 'S50', ('num_reaches',))
rch_river_name._Encoding = 'ascii'

rch_max_wth = sword.groups['reaches'].createVariable(
    'max_width', 'f8', ('num_reaches',))
rch_max_wth.units = 'meters'

# Assign values to new variables.
n_meand_len = np.zeros(len(n_lon))
n_river_name = np.repeat('NaN', len(n_lon))
r_river_name = np.repeat('NaN', len(r_lon))
r_max_wth = np.zeros(len(r_lon))
n_manual_add = np.zeros(len(n_lon))

sword.groups['nodes'].variables['meander_length'][:] = n_meand_len 
sword.groups['nodes'].variables['river_name'][:] = n_river_name
sword.groups['nodes'].variables['manual_add'][:] = n_manual_add 

sword.groups['reaches'].variables['river_name'][:] = r_river_name
sword.groups['reaches'].variables['max_width'][:] = r_max_wth

###discharge parameters

sword.groups['reaches']['discharge_models']['constrained']['MetroMan'].createVariable('sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
sword.groups['reaches']['discharge_models']['constrained']['BAM'].createVariable('sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
sword.groups['reaches']['discharge_models']['constrained']['HiVDI'].createVariable('sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
sword.groups['reaches']['discharge_models']['constrained']['SADS'].createVariable('sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
sword.groups['reaches']['discharge_models']['constrained']['MOMMA'].createVariable('sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)

sword.groups['reaches']['discharge_models']['unconstrained']['MetroMan'].createVariable('sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
sword.groups['reaches']['discharge_models']['unconstrained']['BAM'].createVariable('sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
sword.groups['reaches']['discharge_models']['unconstrained']['HiVDI'].createVariable('sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
sword.groups['reaches']['discharge_models']['unconstrained']['SADS'].createVariable('sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
sword.groups['reaches']['discharge_models']['unconstrained']['MOMMA'].createVariable('sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)


sword.groups['reaches']['discharge_models']['constrained']['MetroMan'].variables['sbQ_rel'][:] = np.repeat(-9999, len(r_lon))
sword.groups['reaches']['discharge_models']['constrained']['BAM'].variables['sbQ_rel'][:] = np.repeat(-9999, len(r_lon))
sword.groups['reaches']['discharge_models']['constrained']['HiVDI'].variables['sbQ_rel'][:] = np.repeat(-9999, len(r_lon))
sword.groups['reaches']['discharge_models']['constrained']['SADS'].variables['sbQ_rel'][:] = np.repeat(-9999, len(r_lon))
sword.groups['reaches']['discharge_models']['constrained']['MOMMA'].variables['sbQ_rel'][:] = np.repeat(-9999, len(r_lon))

sword.groups['reaches']['discharge_models']['unconstrained']['MetroMan'].variables['sbQ_rel'][:] = np.repeat(-9999, len(r_lon))
sword.groups['reaches']['discharge_models']['unconstrained']['BAM'].variables['sbQ_rel'][:] = np.repeat(-9999, len(r_lon))
sword.groups['reaches']['discharge_models']['unconstrained']['HiVDI'].variables['sbQ_rel'][:] = np.repeat(-9999, len(r_lon))
sword.groups['reaches']['discharge_models']['unconstrained']['SADS'].variables['sbQ_rel'][:] = np.repeat(-9999, len(r_lon))
sword.groups['reaches']['discharge_models']['unconstrained']['MOMMA'].variables['sbQ_rel'][:] = np.repeat(-9999, len(r_lon))

'''

#sword.groups['nodes'].variables.keys()
#sword.groups['reaches'].variables.keys()

