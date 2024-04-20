import numpy as np
import netCDF4 as nc
import geopandas as gp
import pandas as pd

region = 'NA'
version = 'v17'
basin = 'hb73'

rch_shp_fn = '/Users/ealtenau/Documents/SWORD_Dev/update_requests/'+version+'/'+region+'/topo_fixes/'\
    +region.lower()+'_sword_reaches_'+basin+'_'+version+'_FG1_LSFix_MS_TopoFix_VizAcc.shp'
nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'\
    +version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
csv_fn = '/Users/ealtenau/Documents/SWORD_Dev/update_requests/'+version+'/'+region+'/topo_fixes/'\
    +region.lower()+'_sword_reaches_'+basin+'_'+version+'_rev_LS.csv'

rch_shp = gp.read_file(rch_shp_fn)
netcdf = nc.Dataset(nc_fn,'r+')
csv = pd.read_csv(csv_fn)

rev_rchs = np.array(csv['reach_id'])

reaches = np.array(rch_shp['reach_id'])
rch_id_up = np.array(rch_shp['rch_id_up'])
rch_id_dn = np.array(rch_shp['rch_id_dn'])
n_rch_up = np.array(rch_shp['n_rch_up'])
n_rch_dn = np.array(rch_shp['n_rch_dn'])

nc_rchs = np.array(netcdf.groups['reaches'].variables['reach_id'][:])
nc_rch_id_up = np.array(netcdf.groups['reaches'].variables['rch_id_up'][:])
nc_rch_id_dn = np.array(netcdf.groups['reaches'].variables['rch_id_dn'][:])
nc_n_rch_up = np.array(netcdf.groups['reaches'].variables['n_rch_up'][:])
nc_n_rch_dn = np.array(netcdf.groups['reaches'].variables['n_rch_down'][:])
nc_cl_ids = np.array(netcdf.groups['centerlines'].variables['cl_id'][:])
nc_cl_rchs = np.array(netcdf.groups['centerlines'].variables['reach_id'][:])

print('reversing linestrings')
for rev in list(range(len(rev_rchs))):
    rch_main = np.where(nc_cl_rchs[0,:] == rev_rchs[rev])[0]
    sort_inds = np.argsort(nc_cl_ids[rch_main])
    nc_cl_ids[rch_main[sort_inds]] = nc_cl_ids[rch_main[sort_inds]][::-1]

print('updating topology')
for ind in list(range(len(reaches))):
    # print(ind, len(reaches)-1)
    nc_ind = np.where(nc_rchs == reaches[ind])[0]
    cl_ind = np.where(nc_cl_rchs[0,:] == reaches[ind])[0]
    cl_id_up = cl_ind[np.where(nc_cl_ids[cl_ind] == np.max(nc_cl_ids[cl_ind]))]
    cl_id_dn = cl_ind[np.where(nc_cl_ids[cl_ind] == np.min(nc_cl_ids[cl_ind]))]
    ###upstream
    if n_rch_up[ind] == 1:
        nc_rch_id_up[0,nc_ind] = int(rch_id_up[ind])
        nc_n_rch_up[nc_ind] = n_rch_up[ind]
        nc_cl_rchs[1,cl_id_up] = int(rch_id_up[ind])
    if n_rch_up[ind] > 1:
        rup = np.array(rch_id_up[ind].split(),dtype=int)
        rup = rup.reshape(len(rup),1)
        nc_rch_id_up[0:len(rup),nc_ind] = rup
        nc_n_rch_up[nc_ind] = n_rch_up[ind]
        if n_rch_up[ind] > 3:
            nc_cl_rchs[1:4,cl_id_up] = rup[0:3]
        else:
            nc_cl_rchs[1:len(rup)+1,cl_id_up] = rup #nc_cl_rchs[:,cl_id_up]
    ###downstream
    if n_rch_dn[ind] == 1:
        nc_rch_id_dn[0,nc_ind] = int(rch_id_dn[ind])
        nc_n_rch_dn[nc_ind] = n_rch_dn[ind]
        nc_cl_rchs[1,cl_id_dn] = int(rch_id_dn[ind])
    if n_rch_dn[ind] > 1:
        rdn = np.array(rch_id_dn[ind].split(),dtype=int)
        rdn = rdn.reshape(len(rdn),1)
        nc_rch_id_dn[0:len(rdn),nc_ind] = rdn
        nc_n_rch_dn[nc_ind] = n_rch_dn[ind]
        if n_rch_dn[ind] > 3:
            nc_cl_rchs[1:4,cl_id_dn] = rdn[0:3]
        else:
            nc_cl_rchs[1:len(rdn)+1,cl_id_dn] = rdn #nc_cl_rchs[:,cl_id_dn]
    
### update netcdf
netcdf.groups['reaches'].variables['rch_id_up'][:] = nc_rch_id_up
netcdf.groups['reaches'].variables['rch_id_dn'][:] = nc_rch_id_dn
netcdf.groups['reaches'].variables['n_rch_up'][:] = nc_n_rch_up
netcdf.groups['reaches'].variables['n_rch_down'][:] = nc_n_rch_dn
netcdf.groups['centerlines'].variables['cl_id'][:] = nc_cl_ids
netcdf.close()

print('DONE')