# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 11:32:24 2021

@author: ealtenau
"""
import netCDF4 as nc
import numpy as np
import time

region = 'OC'
fn_sword = '/Users/ealteanau/Documents/SWORD_Dev/outputs/v13/netcdf/'+region.lower()+'_sword_v13.nc'

sword = nc.Dataset(fn_sword, 'r+')

# n_lon = sword.groups['nodes'].variables['x'][:]
r_lon = sword.groups['reaches'].variables['x'][:]

### Update date and time stamp
sword.production_date = time.strftime("%d-%b-%Y %H:%M:%S", time.gmtime())
# sword.updates = "Added SIC4D discharge algorithm group and associated parameters. \
#                     Added low_slopw_flag variable to reaches group. Changed a few Yukon \
#                     River reaches extreme distance coeficient from 1 to 20."
#sword.production_date
#sword.updates

### Create New Variable(s)

#create variable
sword.groups['reaches']['discharge_models']['constrained'].createGroup('SIC4DVar')
sword.groups['reaches']['discharge_models']['unconstrained'].createGroup('SIC4DVar')
#create variables
sword.groups['reaches']['discharge_models']['constrained']['SIC4DVar'].createVariable('sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
sword.groups['reaches']['discharge_models']['unconstrained']['SIC4DVar'].createVariable('sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
sword.groups['reaches']['discharge_models']['constrained']['SIC4DVar'].createVariable('Abar', 'f8', ('num_reaches',), fill_value=-9999.)
sword.groups['reaches']['discharge_models']['unconstrained']['SIC4DVar'].createVariable('Abar', 'f8', ('num_reaches',), fill_value=-9999.)
sword.groups['reaches']['discharge_models']['constrained']['SIC4DVar'].createVariable('n', 'f8', ('num_reaches',), fill_value=-9999.)
sword.groups['reaches']['discharge_models']['unconstrained']['SIC4DVar'].createVariable('n', 'f8', ('num_reaches',), fill_value=-9999.)
sword.groups['reaches'].createVariable('low_slope_flag', 'i4', ('num_reaches',), fill_value=-9999.)
#add fill values
sword.groups['reaches']['discharge_models']['constrained']['SIC4DVar'].variables['sbQ_rel'][:] = np.repeat(-9999, len(r_lon))
sword.groups['reaches']['discharge_models']['unconstrained']['SIC4DVar'].variables['sbQ_rel'][:] = np.repeat(-9999, len(r_lon))
sword.groups['reaches']['discharge_models']['constrained']['SIC4DVar'].variables['Abar'][:] = np.repeat(-9999, len(r_lon))
sword.groups['reaches']['discharge_models']['unconstrained']['SIC4DVar'].variables['Abar'][:] = np.repeat(-9999, len(r_lon))
sword.groups['reaches']['discharge_models']['constrained']['SIC4DVar'].variables['n'][:] = np.repeat(-9999, len(r_lon))
sword.groups['reaches']['discharge_models']['unconstrained']['SIC4DVar'].variables['n'][:] = np.repeat(-9999, len(r_lon))
sword.groups['reaches'].variables['low_slope_flag'][:] = np.repeat(0,len(r_lon))

sword.close()

#check new variables
#sword.groups['nodes'].variables.keys()
#sword.groups['reaches'].variables.keys()


### fix Yukon reaches
# rch1 = np.where(sword.groups['nodes'].variables['reach_id'][:] == 81250500011)[0]
# rch2 = np.where(sword.groups['nodes'].variables['reach_id'][:] == 81250500031)[0]
# rch3 = np.where(sword.groups['nodes'].variables['reach_id'][:] == 81250500041)[0]
# rch4 = np.where(sword.groups['nodes'].variables['reach_id'][:] == 81250700021)[0]

# sword.groups['nodes'].variables['ext_dist_coef'][rch1] = 20
# sword.groups['nodes'].variables['ext_dist_coef'][rch2] = 20
# sword.groups['nodes'].variables['ext_dist_coef'][rch3] = 20
# sword.groups['nodes'].variables['ext_dist_coef'][rch4] = 20



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



