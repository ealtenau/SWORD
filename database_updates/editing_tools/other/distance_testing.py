import pandas as pd
from geopy import Point, distance
import geopandas as gp
import netCDF4 as nc
import numpy as np
import os 
import time

#########################################################################################
##########################################################################################

def get_distances(lon,lat):
    traces = len(lon) -1
    distances = np.zeros(traces)
    for i in range(traces):
        start = (lat[i], lon[i])
        finish = (lat[i+1], lon[i+1])
        distances[i] = distance.geodesic(start, finish).m
    return distances

##########################################################################################
##########################################################################################

region = 'NA'
version = 'v17'
basin = 'hb81'

print('Starting Basin: ', basin)
sword_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+\
    '/reach_geometry/'+region.lower()+'_sword_'+version+'_connectivity.nc'
path_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+\
    '/network_building/'+region+'/'+basin+'_paths/'
path_nc = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+\
    '/network_building/pathway_netcdfs/'+region+'/'+basin+'_path_vars.nc'
path_files = os.listdir(path_dir)
path_files = np.array([f for f in path_files if '.gpkg' in f])
path_files = np.array([f for f in path_files if 'path_' in f])

for ind in list(range(len(path_files))):
    print(ind, len(path_files)-1)
    path = gp.read_file(path_dir+path_files[ind])
    path_lens = [len(g.coords.xy[0]) for g in path['geometry']]
    sort = np.argsort(path_lens)[::-1]
    for p in list(range(len(sort))):
        # print(p, len(path)-1)
        lon = np.array(path['geometry'][sort[p]].coords.xy[0])
        lat = np.array(path['geometry'][sort[p]].coords.xy[1])  
        if len(lon) == 0:
                continue
        
        # dist = calc_path_dist(lon,lat) #takes longer...
        start = time.time()
        gdf = gp.GeoDataFrame(geometry=gp.points_from_xy(lon, lat),crs="EPSG:4326").to_crs("EPSG:8857") #Equal Area.
        diff = gdf.distance(gdf.shift(1)); diff[0] = 0
        dist = np.cumsum(diff)
        end = time.time()
        print(end-start)

        #method 2
        start = time.time()
        diff2 = get_distances(lon,lat)
        dist2 = np.cumsum(diff2)
        end = time.time()
        print(end-start)

        print(max(dist),max(dist2))

