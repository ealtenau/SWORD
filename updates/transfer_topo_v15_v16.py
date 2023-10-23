import numpy as np
import netCDF4 as nc

region = 'OC'
fn_v15 = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v15/netcdf/testing/'+region.lower()+'_sword_v15_raw.nc'
fn_v15_ref = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v15/netcdf/'+region.lower()+'_sword_v15.nc'
fn_v16 = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16/netcdf/'+region.lower()+'_sword_v16.nc'

s15_ref = nc.Dataset(fn_v15_ref)
s15 = nc.Dataset(fn_v15)
s16 = nc.Dataset(fn_v16, 'r+')

# np.unique(s16.groups['reaches'].variables['edit_flag'][:])
rchs_15 = s15.groups['reaches'].variables['reach_id'][:]
rchs_16 = s16.groups['reaches'].variables['reach_id'][:]
for r in list(range(len(rchs_15))):
    print(r, len(rchs_15)-1)
    pt = np.where(rchs_16 == rchs_15[r])[0]
    if len(pt) == 0:
        continue
    else:
        temp = np.where(s15_ref.groups['reaches'].variables['reach_id'][:] == rchs_15[r])[0]
        flag1 = s15_ref.groups['reaches'].variables['edit_flag'][temp]
        flag2 = s16.groups['reaches'].variables['edit_flag'][pt]
        if flag1 == '3' or flag2 == '3':
            continue
        else:
            s16.groups['reaches'].variables['n_rch_up'][pt] = s15.groups['reaches'].variables['n_rch_up'][r]
            s16.groups['reaches'].variables['n_rch_down'][pt] = s15.groups['reaches'].variables['n_rch_down'][r]
            s16.groups['reaches'].variables['rch_id_up'][:,pt] = s15.groups['reaches'].variables['rch_id_up'][:,r]
            s16.groups['reaches'].variables['rch_id_dn'][:,pt] = s15.groups['reaches'].variables['rch_id_dn'][:,r]

s15.close()
s16.close()