import numpy as np
import netCDF4 as nc

region = 'OC'
version = 'v16'
fn_nc = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
# fn_nc = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16/netcdf/oc_sword_v16.nc'

#load sword.
sword = nc.Dataset(fn_nc)

reach_id = np.array(sword.groups['reaches'].variables['reach_id'][:])
n_rch_up = np.array(sword.groups['reaches'].variables['n_rch_up'][:])
n_rch_down = np.array(sword.groups['reaches'].variables['n_rch_down'][:])
rch_id_up = np.array(sword.groups['reaches'].variables['rch_id_up'][:])
rch_id_dn = np.array(sword.groups['reaches'].variables['rch_id_dn'][:])

#flag topological inconsistencies.
flag = np.zeros(len(reach_id))
for i in list(range(len(reach_id))):
    #check upstream
    for j in list(range(n_rch_up[i])):
        k = np.where(reach_id == rch_id_up[j,i])[0]
        if len(k) == 0:
            continue
        else:
            check = np.where(rch_id_dn[:,k] == reach_id[i])[0]
            if len(check) == 0:
                flag[i] = 1
    #check downstream 
    for j in list(range(n_rch_down[i])):
        k = np.where(reach_id == rch_id_dn[j,i])[0]
        if len(k) == 0:
            continue
        else:
            check = np.where(rch_id_up[:,k] == reach_id[i])[0]
            if len(check) == 0:
                flag[i] = 1

print('DONE. Percent flagged: ',(np.round((len(np.where(flag == 1)[0])/len(reach_id))*100,3)))