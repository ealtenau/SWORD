import numpy as np
import netCDF4 as nc
import geopandas as gp
import geopy.distance
from shapely.geometry import LineString, Point
from geopandas import GeoSeries
import pandas as pd
import time
import argparse
import os
import matplotlib.pyplot as plt
import utm
from pyproj import Proj

###################################################w############################

def reproject_utm(latitude, longitude):

    """
    Modified from C. Lion's function by E. Altenau
    Copyright (c) 2018 UNC Chapel Hill. All rights reserved.

    FUNCTION:
        Projects all points in UTM.

    INPUTS
        latitude -- latitude in degrees (1-D array)
        longitude -- longitude in degrees (1-D array)

    OUTPUTS
        east -- easting in UTM (1-D array)
        north -- northing in UTM (1-D array)
        utm_num -- UTM zone number (1-D array of utm zone numbers for each point)
        utm_let -- UTM zone letter (1-D array of utm zone letters for each point)
    """

    east = np.zeros(len(latitude))
    north = np.zeros(len(latitude))
    east_int = np.zeros(len(latitude))
    north_int = np.zeros(len(latitude))
    zone_num = np.zeros(len(latitude))
    zone_let = []

	# Finds UTM letter and zone for each lat/lon pair.

    for ind in list(range(len(latitude))):
        (east_int[ind], north_int[ind],
	 zone_num[ind], zone_let_int) = utm.from_latlon(latitude[ind],
	                                                longitude[ind])
        zone_let.append(zone_let_int)

    # Finds the unique UTM zones and converts the lat/lon pairs to UTM.
    unq_zones = np.unique(zone_num)
    utm_let = np.unique(zone_let)[0]

    for idx in list(range(len(unq_zones))):
        pt_len = len(np.where(zone_num == unq_zones[idx])[0])

    idx = np.where(pt_len == np.max(pt_len))

    # Set the projection

    if np.sum(latitude) > 0:
        myproj = Proj(
		"+proj=utm +zone=" + str(int(unq_zones[idx])) + 
		" +ellips=WGS84 +datum=WGS84 +units=m")
    else:
        myproj = Proj(
		"+proj=utm +south +zone=" + str(int(unq_zones[idx])) + 
		" +ellips=WGS84 +datum=WGS84 +units=m")

    # Convert all the lon/lat to the main UTM zone
    (east, north) = myproj(longitude, latitude)

    return east, north, zone_num, zone_let

###############################################################################
###############################################################################
###############################################################################

gpkg_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17a/network_testing/Elyssa_basin_hb77/hb77_Colorado_SWORD_v17a1_GF.gpkg'
nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17a/netcdf/na_sword_v17a.nc'

gpkg = gp.read_file(gpkg_fn)
sword = nc.Dataset(nc_fn,'r+')

cl_rchs = sword.groups['centerlines'].variables['reach_id']
cl_nodes = sword.groups['centerlines'].variables['node_id']
cl_lon = sword.groups['centerlines'].variables['x']
cl_lat = sword.groups['centerlines'].variables['y']
cl_id = sword.groups['centerlines'].variables['cl_id']

### index issues [74262301234]
reach = 74100400015
rch = np.where(cl_rchs[0,:] == reach)[0]
cl_x, cl_y, __, __ = reproject_utm(cl_lat[rch], cl_lon[rch])
start_pt = np.where(cl_x==np.min(cl_x))[0]

plt.scatter(cl_lon[rch], cl_lat[rch], c=cl_id[rch], s=15)
plt.scatter(cl_lon[rch[start_pt]], cl_lat[rch[start_pt]], c='red', s=5)
plt.show()

#order indexes

index_order = np.sort(cl_id[rch])
idx = start_pt
new_order = np.zeros(len(rch))
new_order[start_pt] = index_order[0]
count = 1
while np.min(new_order) == 0:
    d = np.sqrt((cl_x[idx]-cl_x)**2 + (cl_y[idx] -cl_y)**2)
    dzero = np.where(new_order == 0)[0]
    next_pt = dzero[np.where(d[dzero] == np.min(d[dzero]))[0]][0]
    new_order[next_pt] = index_order[count]
    count = count+1
    idx = next_pt

# sort_ids = np.argsort(cl_id[rch])
sort_ids = np.argsort(new_order)
plt.plot(cl_lon[rch[sort_ids]], cl_lat[rch[sort_ids]], c='black')
plt.scatter(cl_lon[rch], cl_lat[rch], c=new_order, s=15)
plt.show()

#order the reach points based on index values, then calculate the
#eculdean distance bewteen each ordered point.
order_ids = sort_ids
dist = np.zeros(len(rch))
dist[order_ids[0]] = 0
for idx in list(range(len(order_ids)-1)):
    d = np.sqrt((cl_x[order_ids[idx]]-cl_x[order_ids[idx+1]])**2 +
                (cl_y[order_ids[idx]]-cl_y[order_ids[idx+1]])**2)
    dist[order_ids[idx+1]] = d + dist[order_ids[idx]]
dist = np.array(dist)

plt.scatter(cl_lon[rch], cl_lat[rch], c=dist, s=8)
plt.show()

plt.scatter(cl_lon[rch], cl_lat[rch], c=cl_nodes[0,rch], s=15)
plt.show()

unq_nodes = np.unique(cl_nodes[0,rch])
node_len = np.zeros(len(unq_nodes))
for n in list(range(len(unq_nodes))):
    pts = np.where(cl_nodes[0,rch] == unq_nodes[n])[0]
    node_len[n] = np.max(dist[pts])-np.min(dist[pts])
    if len(pts) == 1:
        node_len[n] = 30
'''
#update netcdf
sword.groups['centerlines'].variables['cl_id'][rch] = new_order

new_rch_len = np.max(dist)
nc_rch = np.where(sword.groups['reaches'].variables['reach_id'][:] == reach)[0]
sword.groups['reaches'].variables['reach_length'][nc_rch] = new_rch_len

for n2 in list(range(len(unq_nodes))):
    nc_node = np.where(sword.groups['nodes'].variables['node_id'][:] == unq_nodes[n2])[0]
    sword.groups['nodes'].variables['node_length'][:] = node_len[n2]

sword.close()
'''

###############################################################################
###############################################################################
###############################################################################
### reach geometry updates
gpkg_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17a/network_testing/Elyssa_basin_hb77/hb77_Colorado_SWORD_v17a1_GF.gpkg'
nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17a/netcdf/na_sword_v17a.nc'

gpkg = gp.read_file(gpkg_fn)
geom = [i for i in gpkg.geometry]

sword = nc.Dataset(nc_fn,'r+')
cl_rchs = sword.groups['centerlines'].variables['reach_id']
cl_nodes = sword.groups['centerlines'].variables['node_id']
cl_lon = sword.groups['centerlines'].variables['x']
cl_lat = sword.groups['centerlines'].variables['y']
cl_id = sword.groups['centerlines'].variables['cl_id']

### geometry issues [74100400015,74100400505,74262800301,74100900025]
### geometry issues [77250000381*, 77234100013, 77280400386]
reach = 77280400386

gp_rch = np.where(gpkg['reach_id'] == reach)[0][0]
lon = np.array(geom[gp_rch].coords.xy[0])
lat = np.array(geom[gp_rch].coords.xy[1])

plt.plot(lon, lat)
plt.show()

rch = np.where(cl_rchs[0,:] == reach)[0]
lon = lon[1:-1]
lat = lat[1:-1]

order_ids = np.argsort(cl_id[rch])
cl_lon[rch[order_ids]] = lon
cl_lat[rch[order_ids]] = lat

plt.plot(cl_lon[rch[order_ids]], cl_lat[rch[order_ids]])
plt.scatter(cl_lon[rch[order_ids[0]]], cl_lat[rch[order_ids[0]]], c='red')
plt.scatter(cl_lon[rch[order_ids[-1]]], cl_lat[rch[order_ids[-1]]],c='red')
plt.show()

#order the reach points based on index values, then calculate the
#eculdean distance bewteen each ordered point.
cl_x, cl_y, __, __ = reproject_utm(cl_lat[rch], cl_lon[rch])
dist = np.zeros(len(rch))
dist[order_ids[0]] = 0
for idx in list(range(len(order_ids)-1)):
    d = np.sqrt((cl_x[order_ids[idx]]-cl_x[order_ids[idx+1]])**2 +
                (cl_y[order_ids[idx]]-cl_y[order_ids[idx+1]])**2)
    dist[order_ids[idx+1]] = d + dist[order_ids[idx]]
dist = np.array(dist)

plt.scatter(cl_lon[rch], cl_lat[rch], c=dist, s=8)
plt.show()

plt.scatter(cl_lon[rch], cl_lat[rch], c=cl_nodes[0,rch], s=15)
plt.show()

unq_nodes = np.unique(cl_nodes[0,rch])
node_len = np.zeros(len(unq_nodes))
node_x = np.zeros(len(unq_nodes))
node_y = np.zeros(len(unq_nodes))
for n in list(range(len(unq_nodes))):
    pts = np.where(cl_nodes[0,rch] == unq_nodes[n])[0]
    node_x[n] = np.median(cl_lon[rch[pts]])
    node_y[n] = np.median(cl_lat[rch[pts]])
    node_len[n] = np.max(dist[pts])-np.min(dist[pts])
    if len(pts) == 1:
        node_len[n] = 30

plt.scatter(node_x, node_y, c=unq_nodes, s=15)
plt.show()

new_rch_len = np.max(dist)

'''
#update netcdf
sword.groups['centerlines'].variables['x'][rch] = cl_lon[rch]
sword.groups['centerlines'].variables['y'][rch] = cl_lat[rch]

nc_rch = np.where(sword.groups['reaches'].variables['reach_id'][:] == reach)[0]
sword.groups['reaches'].variables['reach_length'][nc_rch] = new_rch_len
sword.groups['reaches'].variables['x'][nc_rch] = np.median(cl_lon[rch])
sword.groups['reaches'].variables['y'][nc_rch] = np.median(cl_lat[rch])
sword.groups['reaches'].variables['x_min'][nc_rch] = np.min(cl_lon[rch])
sword.groups['reaches'].variables['y_min'][nc_rch] = np.min(cl_lat[rch])
sword.groups['reaches'].variables['x_max'][nc_rch] = np.max(cl_lon[rch])
sword.groups['reaches'].variables['y_max'][nc_rch] = np.max(cl_lat[rch])

for n2 in list(range(len(unq_nodes))):
    nc_node = np.where(sword.groups['nodes'].variables['node_id'][:] == unq_nodes[n2])[0]
    sword.groups['nodes'].variables['node_length'][:] = node_len[n2]
    sword.groups['nodes'].variables['x'][:] = node_x[n2]
    sword.groups['nodes'].variables['y'][:] = node_y[n2]

sword.close()
'''


