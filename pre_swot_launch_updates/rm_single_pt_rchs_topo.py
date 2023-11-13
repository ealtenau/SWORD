import netCDF4 as nc
import numpy as np

region = 'OC'
fn_sword = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16/netcdf_pre_cl_update_test/'+region.lower()+'_sword_v16.nc'
sword = nc.Dataset(fn_sword, 'r+')

reaches = sword.groups['reaches'].variables['reach_id'][:]
rchs_up = sword.groups['reaches'].variables['rch_id_up'][:]
rchs_dn = sword.groups['reaches'].variables['rch_id_dn'][:]

up0 = sword.groups['reaches'].variables['rch_id_up'][0,:]
up1 = sword.groups['reaches'].variables['rch_id_up'][1,:]
up2 = sword.groups['reaches'].variables['rch_id_up'][2,:]
up3 = sword.groups['reaches'].variables['rch_id_up'][3,:]

dn0 = sword.groups['reaches'].variables['rch_id_dn'][0,:]
dn1 = sword.groups['reaches'].variables['rch_id_dn'][1,:]
dn2 = sword.groups['reaches'].variables['rch_id_dn'][2,:]
dn3 = sword.groups['reaches'].variables['rch_id_dn'][3,:]

u0 = np.where(np.in1d(up0, np.array(list(set(up0) - set(reaches)))[1::]))[0]
u1 = np.where(np.in1d(up1, np.array(list(set(up1) - set(reaches)))[1::]))[0]
u2 = np.where(np.in1d(up2, np.array(list(set(up2) - set(reaches)))[1::]))[0]
u3 = np.where(np.in1d(up3, np.array(list(set(up3) - set(reaches)))[1::]))[0]

d0 = np.where(np.in1d(dn0, np.array(list(set(dn0) - set(reaches)))[1::]))[0]
d1 = np.where(np.in1d(dn1, np.array(list(set(dn1) - set(reaches)))[1::]))[0]
d2 = np.where(np.in1d(dn2, np.array(list(set(dn2) - set(reaches)))[1::]))[0]
d3 = np.where(np.in1d(dn3, np.array(list(set(dn3) - set(reaches)))[1::]))[0]

rchs_up[0,u0] = 0
rchs_up[1,u1] = 0
rchs_up[2,u2] = 0
rchs_up[3,u3] = 0

rchs_dn[0,d0] = 0
rchs_dn[1,d1] = 0
rchs_dn[2,d2] = 0
rchs_dn[3,d3] = 0

#loop through and reformat the arrays. 
for ind in list(range(rchs_up.shape[1])):
    print(ind, rchs_up.shape[1]-1)
    rchs_up[:,ind] = np.sort(rchs_up[:,ind])[::-1]
    rchs_dn[:,ind] = np.sort(rchs_dn[:,ind])[::-1]

## replace in actual netcdf and close. 
sword.groups['reaches'].variables['rch_id_up'][:,:] = rchs_up
sword.groups['reaches'].variables['rch_id_dn'][:,:] = rchs_dn
sword.close()
print('DONE')