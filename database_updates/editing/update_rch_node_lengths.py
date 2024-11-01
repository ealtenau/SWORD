import numpy as np
import netCDF4 as nc
import geopandas as gp
from geopy import distance

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

region = 'AS'
version = 'v17'

nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'\
    +region.lower()+'_sword_'+version+'.nc'

sword = nc.Dataset(nc_fn,'r+')
cl_lon = np.array(sword.groups['centerlines'].variables['x'][:])
cl_lat = np.array(sword.groups['centerlines'].variables['y'][:])
cl_id = np.array(sword.groups['centerlines'].variables['cl_id'][:])
cl_rchs = np.array(sword.groups['centerlines'].variables['reach_id'][0,:])
cl_nodes = np.array(sword.groups['centerlines'].variables['node_id'][0,:])

reaches = np.array(sword.groups['reaches'].variables['reach_id'][:])
rch_len = np.array(sword.groups['reaches'].variables['reach_length'][:])
nodes = np.array(sword.groups['nodes'].variables['node_id'][:])
node_len = np.array(sword.groups['nodes'].variables['node_length'][:])
node_rch = np.array(sword.groups['nodes'].variables['reach_id'][:])

for r in list(range(len(reaches))):
    print(r, len(reaches)-1)
    rch = np.where(cl_rchs == reaches[r])[0] #if multiple choose first.
    sort_ind = rch[np.argsort(cl_id[rch])] 
    x_coords = cl_lon[sort_ind]
    y_coords = cl_lat[sort_ind]
    diff = get_distances(x_coords,y_coords)
    # gdf = gp.GeoDataFrame(geometry=gp.points_from_xy(x_coords, y_coords),crs="EPSG:4326").to_crs("EPSG:8857") #"EPSG:3857" (mercator), "ESRI:54009" (mollweide), "EPSG:9822" (albers), "ESRI:54030" (robinson), "ESRI:54042" (Winkle-Tripel). "EPSG:8857" (Equal Earth Greenwich)
    # diff = gdf.distance(gdf.shift(1));diff[0] = 0
    rch_dist = np.cumsum(diff)
    rch_len[r] = max(rch_dist)
    # np.median(diff) #should be closer to 30.... 
    
    unq_nodes = np.unique(cl_nodes[rch])
    for n in list(range(len(unq_nodes))):
        nds = np.where(cl_nodes[rch] == unq_nodes[n])[0]
        nind = np.where(nodes == unq_nodes[n])[0]
        node_len[nind] = max(np.cumsum(diff[nds]))

# sword.groups['reaches'].variables['reach_length'][:] = rch_len
# sword.groups['nodes'].variables['node_length'][:] = node_len
sword.close()

import random
rand = random.sample(range(0,len(reaches)), 100)
for ind in list(range(len(rand))):
    test = np.where(node_rch == reaches[rand[ind]])[0]
    print(ind, np.round(sum(node_len[test])-rch_len[rand[ind]]))

print('DONE')

# np.median(rch_len)
# np.median(node_len)