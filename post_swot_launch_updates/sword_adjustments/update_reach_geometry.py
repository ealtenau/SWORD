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

gpkg_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/shp/EU/eu_sword_reaches_hb26_v17.shp'
# gpkg_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/gpkg/eu_sword_reaches_v17.gpkg' #continental gpkg file. 
nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/eu_sword_v17.nc'

reach = [26170901135,26170901125,26200101521,26200100661,26200102961,26200101865,26200100361,
         26181000985,26190600461,26190600361,26170900605]

gpkg = gp.read_file(gpkg_fn)
geom = [i for i in gpkg.geometry]

sword = nc.Dataset(nc_fn,'r+')
cl_rchs = sword.groups['centerlines'].variables['reach_id']
cl_nodes = sword.groups['centerlines'].variables['node_id']
cl_lon = sword.groups['centerlines'].variables['x']
cl_lat = sword.groups['centerlines'].variables['y']
cl_id = sword.groups['centerlines'].variables['cl_id']

for r in list(range(len(reach))):
    print(r, len(reach)-1)
    gp_rch = np.where(gpkg['reach_id'] == reach[r])[0][0]
    lon = np.array(geom[gp_rch].coords.xy[0])
    lat = np.array(geom[gp_rch].coords.xy[1])

    # plt.plot(lon, lat)
    # plt.show()

    rch = np.where(cl_rchs[0,:] == reach[r])[0]
    if len(rch) != len(lon):
        if np.abs(len(rch)-len(lon)) == 1:
            mn = np.where(cl_id[rch] == np.min(cl_id[rch]))[0]
            mx = np.where(cl_id[rch] == np.max(cl_id[rch]))[0]
            coords_0 = (cl_lat[rch[mn]], cl_lon[rch[mn]])
            coords_1 = (cl_lat[rch[mx]], cl_lon[rch[mx]])
            coords_2 = (lat[0], lon[0])
            coords_3 = (lat[-1], lon[-1])
            end1 = np.min([geopy.distance.geodesic(coords_2, coords_0).m, 
                           geopy.distance.geodesic(coords_2, coords_1).m])
            end2 = np.min([geopy.distance.geodesic(coords_3, coords_0).m, 
                           geopy.distance.geodesic(coords_3, coords_1).m])
            if end1<end2:
                lon = lon[1::]
                lat = lat[1::]
            else:
                lon = lon[0:-1]
                lat = lat[0:-1]
        if np.abs(len(rch)-len(lon)) == 2:
            lon = lon[1:-1]
            lat = lat[1:-1]

    order_ids = np.argsort(cl_id[rch])
    cl_lon[rch[order_ids]] = lon
    cl_lat[rch[order_ids]] = lat

    # plt.plot(cl_lon[rch[order_ids]], cl_lat[rch[order_ids]])
    # plt.scatter(cl_lon[rch[order_ids[0]]], cl_lat[rch[order_ids[0]]], c='red')
    # plt.scatter(cl_lon[rch[order_ids[-1]]], cl_lat[rch[order_ids[-1]]],c='red')
    # plt.show()

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

    # plt.scatter(cl_lon[rch], cl_lat[rch], c=dist, s=8)
    # plt.show()

    # plt.scatter(cl_lon[rch], cl_lat[rch], c=cl_nodes[0,rch], s=15)
    # plt.show()

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

    # plt.scatter(node_x, node_y, c=unq_nodes, s=15)
    # plt.show()

    new_rch_len = np.max(dist)

    #update netcdf
    sword.groups['centerlines'].variables['x'][rch] = cl_lon[rch]
    sword.groups['centerlines'].variables['y'][rch] = cl_lat[rch]

    nc_rch = np.where(sword.groups['reaches'].variables['reach_id'][:] == reach[r])[0]
    sword.groups['reaches'].variables['reach_length'][nc_rch] = new_rch_len
    sword.groups['reaches'].variables['x'][nc_rch] = np.median(cl_lon[rch])
    sword.groups['reaches'].variables['y'][nc_rch] = np.median(cl_lat[rch])
    sword.groups['reaches'].variables['x_min'][nc_rch] = np.min(cl_lon[rch])
    sword.groups['reaches'].variables['y_min'][nc_rch] = np.min(cl_lat[rch])
    sword.groups['reaches'].variables['x_max'][nc_rch] = np.max(cl_lon[rch])
    sword.groups['reaches'].variables['y_max'][nc_rch] = np.max(cl_lat[rch])

    for n2 in list(range(len(unq_nodes))):
        nc_node = np.where(sword.groups['nodes'].variables['node_id'][:] == unq_nodes[n2])[0]
        sword.groups['nodes'].variables['node_length'][nc_node ] = node_len[n2]
        sword.groups['nodes'].variables['x'][nc_node ] = node_x[n2]
        sword.groups['nodes'].variables['y'][nc_node ] = node_y[n2]

sword.close()