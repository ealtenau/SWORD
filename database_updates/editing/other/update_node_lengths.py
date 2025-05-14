import numpy as np
import netCDF4 as nc
import geopandas as gp
from geopy import distance
import pandas as pd

################################################################################################

def get_distances(lon,lat):
    traces = len(lon) -1
    distances = np.zeros(traces)
    for i in range(traces):
        start = (lat[i], lon[i])
        finish = (lat[i+1], lon[i+1])
        distances[i] = distance.geodesic(start, finish).m
    distances = np.append(0,distances)
    return distances

################################################################################################

region = 'OC'
version = 'v18'
dist_update = 'True'

nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'\
    +region.lower()+'_sword_'+version+'.nc'
csv_fn = '/Users/ealtenau/Documents/SWORD_Dev/update_requests/'+version+'/'+region+'/'\
    +region.lower()+'_problematic_node_len.csv'

sword = nc.Dataset(nc_fn,'r+')
cl_lon = np.array(sword.groups['centerlines'].variables['x'][:])
cl_lat = np.array(sword.groups['centerlines'].variables['y'][:])
cl_id = np.array(sword.groups['centerlines'].variables['cl_id'][:])
cl_rchs = np.array(sword.groups['centerlines'].variables['reach_id'][0,:])
cl_nodes = np.array(sword.groups['centerlines'].variables['node_id'][0,:])

reaches = np.array(sword.groups['reaches'].variables['reach_id'][:])
rch_len = np.array(sword.groups['reaches'].variables['reach_length'][:])
rch_dist = np.array(sword.groups['reaches'].variables['dist_out'][:])
nodes = np.array(sword.groups['nodes'].variables['node_id'][:])
node_len = np.array(sword.groups['nodes'].variables['node_length'][:])
node_rch = np.array(sword.groups['nodes'].variables['reach_id'][:])
node_dist = np.array(sword.groups['nodes'].variables['dist_out'][:])

csv = pd.read_csv(csv_fn)
rch_arr = np.array(csv['reach_id']) 
for r in list(range(len(rch_arr))):
    print('reach:', r, len(rch_arr)-1)
    current = np.where(reaches == rch_arr[r])[0]
    rch = np.where(cl_rchs == rch_arr[r])[0] #if multiple choose first.
    sort_ind = rch[np.argsort(cl_id[rch])] 
    x_coords = cl_lon[sort_ind]
    y_coords = cl_lat[sort_ind]
    diff = get_distances(x_coords,y_coords)
    
    unq_nodes = np.unique(cl_nodes[rch])
    for n in list(range(len(unq_nodes))):
        nds = np.where(cl_nodes[rch] == unq_nodes[n])[0]
        nind = np.where(nodes == unq_nodes[n])[0]
        node_len[nind] = max(np.cumsum(diff[nds]))
        # print(node_len[nind])

    rch_nodes = np.where(node_rch == rch_arr[r])[0]
    if dist_update == 'True':
        sort_nodes = np.argsort(nodes[rch_nodes])
        base_val = rch_dist[current] - rch_len[current] 
        node_cs = np.cumsum(node_len[rch_nodes[sort_nodes]])
        node_dist[rch_nodes[sort_nodes]] = node_cs+base_val
    
    val1 = np.round(max(np.cumsum(node_len[rch_nodes]))-rch_len[current])[0]
    if val1 != 0:
        print('len diff:', val1)
    if dist_update == 'True':
        val2 = np.round(max(node_dist[rch_nodes])-rch_dist[current])[0]
        if val2 != 0:
            print('dist diff:', val2)

sword.groups['nodes'].variables['node_length'][:] = node_len
if dist_update == 'True':
    sword.groups['nodes'].variables['dist_out'][:] = node_dist

sword.close()
print('DONE')
