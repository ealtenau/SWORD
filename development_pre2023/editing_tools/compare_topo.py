import numpy as np
import netCDF4 as nc

def flag_topo(reach_id, n_rch_up, n_rch_down, rch_id_dn, rch_id_up):
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
    return flag

########################################################################################################

region = 'EU'
version = 'v16'
fn_nc = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
fn_nc2 = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16/v16_pre_rch_id_edits/'+region.lower()+'_sword_'+version+'.nc'

#load sword.
sword = nc.Dataset(fn_nc)
sword2 = nc.Dataset(fn_nc2)

reach_id = np.array(sword.groups['reaches'].variables['reach_id'][:])
n_rch_up = np.array(sword.groups['reaches'].variables['n_rch_up'][:])
n_rch_down = np.array(sword.groups['reaches'].variables['n_rch_down'][:])
rch_id_up = np.array(sword.groups['reaches'].variables['rch_id_up'][:])
rch_id_dn = np.array(sword.groups['reaches'].variables['rch_id_dn'][:])

reach_id2 = np.array(sword2.groups['reaches'].variables['reach_id'][:])
n_rch_up2 = np.array(sword2.groups['reaches'].variables['n_rch_up'][:])
n_rch_down2 = np.array(sword2.groups['reaches'].variables['n_rch_down'][:])
rch_id_up2 = np.array(sword2.groups['reaches'].variables['rch_id_up'][:])
rch_id_dn2 = np.array(sword2.groups['reaches'].variables['rch_id_dn'][:])

flag = flag_topo(reach_id, n_rch_up, n_rch_down, rch_id_dn, rch_id_up)
flag2 = flag_topo(reach_id2, n_rch_up2, n_rch_down2, rch_id_dn2, rch_id_up2)


diff = np.where((flag == 1) & (flag2 == 0))[0]

len(np.where(flag == 1)[0])
len(np.where(flag2 == 1)[0])
np.round((len(np.where(flag == 1)[0])/len(reach_id))*100,3)
np.round((len(np.where(flag2 == 1)[0])/len(reach_id2))*100,3)

reach_id[diff]-reach_id2[diff]

reach_id[diff[5]]
rch_id_up[:,diff[5]]
rch_id_dn[:,diff[5]]

reach_id2[diff[5]]
rch_id_up2[:,diff[5]]
rch_id_dn2[:,diff[5]]