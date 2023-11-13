import netCDF4 as nc
import numpy as np

region = 'OC'
version = 'v15'
sword_dir = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'
sword = nc.Dataset(sword_dir+region.lower()+'_sword_'+version+'.nc', 'r+')

up_nghs = sword.groups['reaches'].variables['rch_id_up'][:]
dn_nghs = sword.groups['reaches'].variables['rch_id_dn'][:]
facc = sword.groups['reaches'].variables['facc'][:]
wse = sword.groups['reaches'].variables['wse'][:]
rch_id = sword.groups['reaches'].variables['reach_id'][:]

up_new = np.copy(up_nghs)
dn_new = np.copy(dn_nghs)
for ind in list(range(len(rch_id))):
    up_temp = up_nghs[:,ind]
    dn_temp = dn_nghs[:,ind]
    up_cnt = up_nghs[np.where(up_nghs[:,ind] > 0)[0],ind]
    dn_cnt = dn_nghs[np.where(dn_nghs[:,ind] > 0)[0],ind]

    up_facc = []; up_wse = []
    for up in list(range(len(up_cnt))):
        up_loc = np.where(rch_id == up_cnt[up])[0]
        if len(up_loc) == 0:
            continue
        else:
            up_facc.append(facc[up_loc][0])
            up_wse.append(wse[up_loc][0])

    dn_facc = []; dn_wse = []
    for dn in list(range(len(dn_cnt))):
        dn_loc = np.where(rch_id == dn_cnt[dn])[0]
        if len(dn_loc) == 0:
            continue
        else:
            dn_facc.append(facc[dn_loc][0])
            dn_wse.append(wse[dn_loc][0])

    if len(up_wse) > 0 and len(dn_wse) > 0:
        if min(up_wse) < wse[ind] and min(dn_wse) > wse[ind]:
            up_new[:,ind] = dn_temp
            dn_new[:,ind] = up_temp
            print(ind, rch_id[ind], 'switched')
        elif max(up_facc) > facc[ind] and max(dn_facc) < facc[ind]:
            up_new[:,ind] = dn_temp
            dn_new[:,ind] = up_temp
            print(ind, rch_id[ind], 'switched')
        else:
            print(ind, rch_id[ind])
            continue
    
    elif len(up_wse) > 0 and len(dn_wse) == 0:
        if min(up_wse) < wse[ind]:
            up_new[:,ind] = dn_temp
            dn_new[:,ind] = up_temp
            print(ind, rch_id[ind], 'switched')
        elif max(up_facc) > facc[ind]:
            up_new[:,ind] = dn_temp
            dn_new[:,ind] = up_temp
            print(ind, rch_id[ind], 'switched')
        else:
            print(ind, rch_id[ind])
            continue

    elif len(dn_wse) > 0 and len(up_wse) == 0: 
        if min(dn_wse) > wse[ind]:
            up_new[:,ind] = dn_temp
            dn_new[:,ind] = up_temp
            print(ind, rch_id[ind], 'switched')
        elif max(dn_facc) < facc[ind]:
            up_new[:,ind] = dn_temp
            dn_new[:,ind] = up_temp
            print(ind, rch_id[ind], 'switched')
        else:
            print(ind, rch_id[ind])
            continue
    else:
        print(ind, rch_id[ind])
        continue

sword.groups['reaches'].variables['rch_id_up'][:] = up_new
sword.groups['reaches'].variables['rch_id_dn'][:] = dn_new
sword.close()