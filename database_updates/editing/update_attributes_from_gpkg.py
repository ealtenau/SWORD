import numpy as np
import netCDF4 as nc
import geopandas as gp
import sys

region = 'SA'
version = 'v18'
dist_update = 'True'

# gpkg_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/gpkg/'\
#     +region.lower()+'_sword_reaches_'+version+'.gpkg'
# gpkg_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/shp/SA/'\
#     +region.lower()+'_sword_reaches_hb67_'+version+'.shp'
gpkg_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Topology/'+version+'/'+region+\
        '/dist_out_updates/'+region.lower()+'_sword_reaches_'+version+'_distout_update.gpkg'
# gpkg_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/gpkg/sa_sword_reaches_v17.gpkg'
nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'\
    +region.lower()+'_sword_'+version+'.nc'

sword = nc.Dataset(nc_fn, 'r+')
gpkg = gp.read_file(gpkg_fn)

reaches = np.array(sword.groups['reaches'].variables['reach_id'][:])
reach_dist = np.array(sword.groups['reaches'].variables['dist_out'][:])
rch_len = np.array(sword.groups['reaches'].variables['reach_length'][:])
node_rchs = np.array(sword.groups['nodes'].variables['reach_id'][:])
rch_ms = np.array(sword.groups['reaches'].variables['main_side'][:])
node_ms = np.array(sword.groups['nodes'].variables['main_side'][:])
rch_rn = np.array(sword.groups['reaches'].variables['river_name'][:])
node_rn = np.array(sword.groups['nodes'].variables['river_name'][:])
path_freq = np.array(sword.groups['reaches'].variables['path_freq'][:])
node_path_freq = np.array(sword.groups['nodes'].variables['path_freq'][:])
path_order = np.array(sword.groups['reaches'].variables['path_order'][:])
node_path_order = np.array(sword.groups['nodes'].variables['path_order'][:])
rch_x = np.array(sword.groups['reaches'].variables['x'][:])
rch_y = np.array(sword.groups['reaches'].variables['y'][:])
rch_segs = np.array(sword.groups['reaches'].variables['path_segs'][:])
rch_ends = np.array(sword.groups['reaches'].variables['end_reach'][:])
rch_strm = np.array(sword.groups['reaches'].variables['stream_order'][:])
node_ends = np.array(sword.groups['nodes'].variables['end_reach'][:])
nodes = np.array(sword.groups['nodes'].variables['node_id'][:])
node_dist = np.array(sword.groups['nodes'].variables['dist_out'][:])
node_len = np.array(sword.groups['nodes'].variables['node_length'][:])
rch_net = np.array(sword.groups['reaches'].variables['network'][:])
node_net = np.array(sword.groups['nodes'].variables['network'][:])
path_segs = np.array(sword.groups['reaches'].variables['path_segs'][:])
node_path_segs = np.array(sword.groups['nodes'].variables['path_segs'][:])
node_strm = np.array(sword.groups['nodes'].variables['stream_order'][:])

if len(reaches) != len(gpkg):
    print('!!! Reaches in NetCDF not equal to GPKG !!!')
    sys.exit()

print('Updating Attributes from SHP File')
unq_rchs = np.array(gpkg['reach_id'])
for r in list(range(len(unq_rchs))):
    # print(r, len(unq_rchs)-1)
    rch = np.where(reaches == unq_rchs[r])[0]
    nds = np.where(node_rchs == unq_rchs[r])[0] 
    rch_ms[rch] = gpkg['main_side'][r]
    node_ms[nds] = gpkg['main_side'][r]
    rch_rn[rch] = gpkg['river_name'][r]
    node_rn[nds] = gpkg['river_name'][r]
    rch_ends[rch] = gpkg['end_reach'][r]
    node_ends[nds] = gpkg['end_reach'][r]
    rch_net[rch] = gpkg['network'][r]
    node_net[nds] = gpkg['network'][r]
    path_freq[rch] = gpkg['path_freq'][r]
    node_path_freq[nds] = gpkg['path_freq'][r]
    path_order[rch] = gpkg['path_order'][r]
    node_path_order[nds] = gpkg['path_order'][r]
    path_segs[rch] = gpkg['path_segs'][r]
    node_path_segs[nds] = gpkg['path_segs'][r]
    rch_strm[rch] = gpkg['strm_order'][r]
    node_strm[nds] = gpkg['strm_order'][r]
    if dist_update == 'True': 
        reach_dist[rch] = gpkg['dist_out2'][r]
        sort_nodes = np.argsort(nodes[nds])
        base_val = reach_dist[r] - rch_len[r]
        node_cs = np.cumsum(node_len[nds[sort_nodes]])
        node_dist[nds[sort_nodes]] = node_cs+base_val 
        #print(max(node_cs), rch_len[r])
        #print(max(node_cs)-min(node_cs), rch_len[r])
        #print(max(node_dist[nds])-min(node_dist[nds]), reach_dist[rch])
        #print(max(node_dist[nds]), reach_dist[rch])

print('Updating NetCDF')
sword.groups['reaches'].variables['main_side'][:] = rch_ms
sword.groups['nodes'].variables['main_side'][:] = node_ms
sword.groups['reaches'].variables['river_name'][:] = rch_rn
sword.groups['nodes'].variables['river_name'][:] = node_rn
sword.groups['reaches'].variables['end_reach'][:] = rch_ends
sword.groups['nodes'].variables['end_reach'][:] = node_ends
sword.groups['reaches'].variables['stream_order'][:] = rch_strm
sword.groups['nodes'].variables['stream_order'][:] = node_strm
sword.groups['reaches'].variables['network'][:] = rch_net
sword.groups['nodes'].variables['network'][:] = node_net
sword.groups['reaches'].variables['path_freq'][:] = path_freq
sword.groups['nodes'].variables['path_freq'][:] = node_path_freq
sword.groups['reaches'].variables['path_order'][:] = path_order
sword.groups['nodes'].variables['path_order'][:] = node_path_order
sword.groups['reaches'].variables['path_segs'][:] = path_segs
sword.groups['nodes'].variables['path_segs'][:] = node_path_segs
if dist_update == 'True':
    sword.groups['reaches'].variables['dist_out'][:] = reach_dist
    sword.groups['nodes'].variables['dist_out'][:] = node_dist
sword.close()
print('DONE')