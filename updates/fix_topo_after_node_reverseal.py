import numpy as np
import netCDF4 as nc

region = 'AS'
version = 'v16'
fn_nc1 = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
fn_nc2 = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf_pre_node_update/'+region.lower()+'_sword_'+version+'.nc'

new = nc.Dataset(fn_nc1, 'r+')
old = nc.Dataset(fn_nc2)

check = new.groups['reaches'].variables['reach_id'][:] - old.groups['reaches'].variables['reach_id'][:]
if max(check) == 0:
    new.groups['reaches'].variables['n_rch_up'][:] = old.groups['reaches'].variables['n_rch_up'][:]
    new.groups['reaches'].variables['n_rch_down'][:] = old.groups['reaches'].variables['n_rch_down'][:]
    new.groups['reaches'].variables['rch_id_up'][:,:] = old.groups['reaches'].variables['rch_id_up'][:,:]
    new.groups['reaches'].variables['rch_id_dn'][:,:] = old.groups['reaches'].variables['rch_id_dn'][:,:]
    new.close()
    old.close()
    print('Done')
else:
    print('ID indexes not identical')