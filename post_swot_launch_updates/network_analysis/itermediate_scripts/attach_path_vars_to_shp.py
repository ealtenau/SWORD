import geopandas as gp
import pandas as pd
import numpy as np
import netCDF4 as nc

region = 'SA'
version = 'v17'
basin = 'hb62'

nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/network_building/pathway_netcdfs/'\
    +region+'/'+basin+'_path_vars.nc'
rch_shp_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/shp/'\
    +region+'/'+region.lower()+'_sword_reaches_'+basin+'_'+version+'.shp'
node_shp_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/shp/'\
    +region+'/'+region.lower()+'_sword_nodes_'+basin+'_'+version+'.shp'

rch_shp = gp.read_file(rch_shp_fn)
node_shp = gp.read_file(node_shp_fn)
netcdf = nc.Dataset(nc_fn)

rch_id = np.array(netcdf.groups['centerlines'].variables['reach_id'][0,:])
node_id = np.array(netcdf.groups['centerlines'].variables['node_id'][:])
path = np.array(netcdf.groups['centerlines'].variables['path_order_by_length'][:])
path_freq = np.array(netcdf.groups['centerlines'].variables['path_travel_frequency'][:])
dist_out = np.array(netcdf.groups['centerlines'].variables['dist_out'][:])
dist_out2 = np.array(netcdf.groups['centerlines'].variables['dist_out_all'][:])
main_side = np.array(netcdf.groups['centerlines'].variables['main_side_chan'][:])
strm_order = np.array(netcdf.groups['centerlines'].variables['stream_order'][:])

print('Starting Reaches')
rch_shp_path = np.zeros(rch_shp.shape[0])
rch_shp_path_freq = np.zeros(rch_shp.shape[0])
rch_shp_dist_out = np.zeros(rch_shp.shape[0])
rch_shp_dist_out2 = np.zeros(rch_shp.shape[0])
rch_shp_main_side = np.zeros(rch_shp.shape[0])
rch_shp_strm_order = np.zeros(rch_shp.shape[0])
unq_rch = np.unique([rch_id])
for ind in list(range(len(unq_rch))):
    rch = np.where(rch_shp['reach_id'] == unq_rch[ind])[0]
    if len(rch) == 0:
        continue
    pts = np.where(rch_id == unq_rch[ind])[0]
    rch_shp_path[rch] = np.max(path[pts])
    rch_shp_path_freq[rch] = np.max(path_freq[pts])
    rch_shp_dist_out[rch] = np.max(dist_out[pts])
    rch_shp_dist_out2[rch] = np.max(dist_out2[pts])
    rch_shp_main_side[rch] = np.max(main_side[pts])
    rch_shp_strm_order[rch] = np.max(strm_order[pts])

rch_shp['path_order'] = rch_shp_path
rch_shp['path_freq'] = rch_shp_path_freq
rch_shp['outdist_ns'] = rch_shp_dist_out
rch_shp['outletdist'] = rch_shp_dist_out2
rch_shp['main_side'] = rch_shp_main_side
rch_shp['strm_order'] = rch_shp_strm_order
rch_shp.to_file(rch_shp_fn)

print('Starting Nodes')
node_shp_path = np.zeros(node_shp.shape[0])
node_shp_path_freq = np.zeros(node_shp.shape[0])
node_shp_dist_out = np.zeros(node_shp.shape[0])
node_shp_dist_out2 = np.zeros(node_shp.shape[0])
node_shp_main_side = np.zeros(node_shp.shape[0])
node_shp_strm_order = np.zeros(node_shp.shape[0])
unq_nodes = np.unique([node_id])
for idx in list(range(len(unq_nodes))):
    nds = np.where(node_shp['node_id'] == unq_nodes[idx])[0]
    if len(nds) == 0:
        continue
    npts = np.where(node_id == unq_nodes[idx])[0]
    node_shp_path[nds] = np.max(path[npts])
    node_shp_path_freq[nds] = np.max(path_freq[npts])
    node_shp_dist_out[nds] = np.max(dist_out[npts])
    node_shp_dist_out2[nds] = np.max(dist_out2[npts])
    node_shp_main_side[nds] = np.max(main_side[npts])
    node_shp_strm_order[nds] = np.max(strm_order[npts])

node_shp['path_order'] = node_shp_path
node_shp['path_freq'] = node_shp_path_freq
node_shp['outdist_ns'] = node_shp_dist_out
node_shp['outletdist'] = node_shp_dist_out2
node_shp['main_side'] = node_shp_main_side
node_shp['strm_order'] = node_shp_strm_order
node_shp.to_file(node_shp_fn)