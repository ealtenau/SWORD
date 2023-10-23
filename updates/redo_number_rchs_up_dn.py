import numpy as np
import netCDF4 as nc

region = 'AS'
version = 'v16'
fn_nc = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'

sword = nc.Dataset(fn_nc, 'r+')
rch_id_up = np.array(sword.groups['reaches'].variables['rch_id_up'][:])
rch_id_dn = np.array(sword.groups['reaches'].variables['rch_id_dn'][:])

n_up = np.zeros(len(rch_id_up[0,:]), dtype=int)
n_dn = np.zeros(len(rch_id_up[0,:]), dtype = int)
for r in list(range(len(rch_id_up[0,:]))):
    print(r, len(rch_id_up[0,:])-1)
    num_up = len(np.where(rch_id_up[:,r] > 0)[0])
    num_dn = len(np.where(rch_id_dn[:,r] > 0)[0])
    n_up[r] = num_up
    n_dn[r] = num_dn

sword.groups['reaches'].variables['n_rch_up'][:] = n_up
sword.groups['reaches'].variables['n_rch_down'][:] = n_dn
sword.close()