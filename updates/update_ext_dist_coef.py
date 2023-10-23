import numpy as np
import netCDF4 as nc
import geopandas as gp

region = 'NA'
sword_nc_fn = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16/netcdf/'+region.lower()+'_sword_v16.nc'
sword_gp_fn = '/Users/ealteanau/Documents/SWORD_Dev/testing_files/gis/'+region.lower()+'_extd_flag_v15.gpkg'

sword_nc = nc.Dataset(sword_nc_fn, 'r+')
sword_gp = gp.read_file(sword_gp_fn)

rchs_all = np.array(sword_gp['reach_id'])
change = np.where(sword_gp['extd_flag'] == 1)[0]
rchs = rchs_all[change]
for r in list(range(len(rchs))):
    print(r, len(rchs)-1)
    nodes = np.where(sword_nc.groups['nodes'].variables['reach_id'][:] == rchs[r])[0]
    sword_nc.groups['nodes'].variables['ext_dist_coef'][nodes] = 1
sword_nc.close()
