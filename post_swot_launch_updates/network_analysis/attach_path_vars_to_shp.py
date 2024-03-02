import geopandas as gp
import pandas as pd
import numpy as np
import netCDF4 as nc

nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17a/pathways/NA/hb77_path_vars.nc'
shp_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17a/shp/NA/na_sword_reaches_hb77_v17a.shp'

shp = gp.read_file(shp_fn)
netcdf = nc.Dataset(nc_fn)

rch_id = np.array(netcdf.groups['centerlines'].variables['reach_id'][0,:])
path = np.array(netcdf.groups['centerlines'].variables['path_order_by_length'][:])
path_freq = np.array(netcdf.groups['centerlines'].variables['path_travel_frequency'][:])
dist_out = np.array(netcdf.groups['centerlines'].variables['dist_out'][:])
dist_out2 = np.array(netcdf.groups['centerlines'].variables['dist_out_all'][:])
main_side = np.array(netcdf.groups['centerlines'].variables['main_side_chan'][:])
strm_order = np.array(netcdf.groups['centerlines'].variables['stream_order'][:])

shp_path = np.zeros(shp.shape[0])
shp_path_freq = np.zeros(shp.shape[0])
shp_dist_out = np.zeros(shp.shape[0])
shp_dist_out2 = np.zeros(shp.shape[0])
shp_main_side = np.zeros(shp.shape[0])
shp_strm_order = np.zeros(shp.shape[0])
unq_rch = np.unique([rch_id])
for ind in list(range(len(unq_rch))):
    rch = np.where(shp['reach_id'] == unq_rch[ind])[0]
    if len(rch) == 0:
        continue
    pts = np.where(rch_id == unq_rch[ind])[0]

    shp_path[rch] = np.max(path[pts])
    shp_path_freq[rch] = np.max(path_freq[pts])
    shp_dist_out[rch] = np.max(dist_out[pts])
    shp_dist_out2[rch] = np.max(dist_out2[pts])
    shp_main_side[rch] = np.max(main_side[pts])
    shp_strm_order[rch] = np.max(strm_order[pts])

shp['path_order'] = shp_path
shp['path_freq'] = shp_path_freq
shp['outdist_ns'] = shp_dist_out
shp['outletdist'] = shp_dist_out2
shp['main_side'] = shp_main_side
shp['strm_order'] = shp_strm_order

shp.to_file(shp_fn)