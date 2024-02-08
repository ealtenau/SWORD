import numpy as np
import netCDF4 as nc

region = 'NA'
fn_v15 = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v15/netcdf/'+region.lower()+'_sword_v15.nc'
fn_v17 = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/'+region.lower()+'_sword_v17.nc'

s15 = nc.Dataset(fn_v15)
s17 = nc.Dataset(fn_v17, 'r+')

# np.unique(s16.groups['reaches'].variables['edit_flag'][:])
rchs_15 = s15.groups['reaches'].variables['reach_id'][:]
rchs_17 = s17.groups['reaches'].variables['reach_id'][:]
for r in list(range(len(rchs_15))):
    print(r, len(rchs_15)-1)
    pt = np.where(rchs_17 == rchs_15[r])[0]
    if len(pt) == 0:
        continue
    else:
        flag1 = '3' in s15.groups['reaches'].variables['edit_flag'][r][0]
        flag2 = '3' in s17.groups['reaches'].variables['edit_flag'][pt][0]
        if flag1 == True and flag2 == False:
            if s17.groups['reaches'].variables['edit_flag'][pt] == 'NaN':
                edit_val = '3'
            else:
                edit_val = str(s17.groups['reaches'].variables['edit_flag'][pt][0]) + ',3'
                # print(r)
                # break
            s17.groups['reaches'].variables['edit_flag'][pt] = edit_val
        else:
            continue

s15.close()
s17.close()