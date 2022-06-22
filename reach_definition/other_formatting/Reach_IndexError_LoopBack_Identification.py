# -*- coding: utf-8 -*-
"""
Created on Fri Apr 09 18:54:45 2021

@author: ealtenau
"""
from __future__ import division
import netCDF4 as nc
import numpy as np
import pandas as pd
import time
from pyproj import Proj
import utm
import matplotlib.pyplot as plt 

###############################################################################

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
        utm_num -- UTM zone number (single number)
        utm_let -- UTM zone letter (single letter)
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
		"+proj=utm +zone=" + str(int(unq_zones[idx])) + utm_let +
		" +ellips=WGS84 +datum=WGS84 +units=m")
    else:
        myproj = Proj(
		"+proj=utm +south +zone=" + str(int(unq_zones[idx])) + utm_let +
		" +ellips=WGS84 +datum=WGS84 +units=m")

    # Convert all the lon/lat to the main UTM zone
    (east, north) = myproj(longitude, latitude)

    return east, north, zone_num, zone_let

###############################################################################

def calc_dist(subcls_lon, subcls_lat, subcls_rch_id, subcls_facc, 
                 subcls_rch_ind):
    
    """
    FUNCTION:
        Creates a 1-D array of flow distances for each specified reach. 

    INPUTS
        subcls -- Object containing reach and node attributes for the 
            high-resolution centerline.
            [attributes used]:
                lon -- Longitude values along the high-resolution centerline. 
                lat -- Latitude values along the high-resolution centerline.
                facc -- Flow accumulation along the high-resolution centerline.
                rch_id5 -- Reach numbers after aggregating short river, lake, 
                    and dam reaches. 
    
    OUTPUTS
        segDist -- Flow distance per reach (meters).
    """
    
    segDist = np.zeros(len(subcls_lon))
    uniq_segs = np.unique(subcls_rch_id)
    for ind in list(range(len(uniq_segs))):
        #print(ind, uniq_segs[ind])
        seg = np.where(subcls_rch_id == uniq_segs[ind])[0]
        seg_lon = subcls_lon[seg]
        seg_lat = subcls_lat[seg]
        seg_x, seg_y, __, __ = reproject_utm(seg_lat, seg_lon)
        upa = subcls_facc[seg]

        order_ids = np.argsort(subcls_rch_ind[seg])
        dist = np.zeros(len(seg))
        dist[order_ids[0]] = 0
        for idx in list(range(len(order_ids)-1)):
            d = np.sqrt((seg_x[order_ids[idx]]-seg_x[order_ids[idx+1]])**2 + 
                        (seg_y[order_ids[idx]]-seg_y[order_ids[idx+1]])**2)
            dist[order_ids[idx+1]] = d + dist[order_ids[idx]]

        dist = np.array(dist)
        start = upa[np.where(dist == np.min(dist))[0][0]]
        end = upa[np.where(dist == np.max(dist))[0][0]]

        if end > start:
            segDist[seg] = abs(dist-np.max(dist))

        else:
            segDist[seg] = dist

    return segDist

###############################################################################
###############################################################################
###############################################################################

start = time.time()

region = 'OC'
version = '_v11.csv'
fn_sword = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/outputs/Reaches_Nodes/netcdf/'+region.lower()+'_sword_v11.nc'

sword = nc.Dataset(fn_sword)
rch_id = sword.groups['centerlines'].variables['reach_id'][0,:]
lon = sword.groups['centerlines'].variables['x'][:]
lat = sword.groups['centerlines'].variables['y'][:]
cl_id = sword.groups['centerlines'].variables['cl_id'][:]
sword.close()

bad_reach = list()
uniq_rch = np.unique(rch_id)
#ind = np.where(uniq_rch == 72319000195)[0]
for ind in list(range(len(uniq_rch))):
    print(ind)
    pts = np.where(rch_id == uniq_rch[ind])[0]
    rch_lon = lon[pts]
    rch_lat = lat[pts]
    rch_cl_id = cl_id[pts]
    rch_x, rch_y, __, __ = reproject_utm(rch_lat, rch_lon)
    
    order_ids = np.argsort(rch_cl_id)
    dist = np.zeros(len(pts))
    dist[order_ids[0]] = 0
    for idx in list(range(len(order_ids)-1)):
        d = np.sqrt((rch_x[order_ids[idx]]-rch_x[order_ids[idx+1]])**2 + 
                    (rch_y[order_ids[idx]]-rch_y[order_ids[idx+1]])**2)
        dist[order_ids[idx+1]] = d + dist[order_ids[idx]]
    dist = np.array(dist)
    dist_diff = abs(np.diff(dist[order_ids]))
    m = np.where(dist_diff > 1000)[0]
    if len(m) > 0:
        bad_reach.append(uniq_rch[ind])

bad_reach = np.array(bad_reach) # 0.8%

output = pd.DataFrame(bad_reach)
output.to_csv('C:/Users/ealtenau/Documents/Research/SWAG/For_Server/outputs/test_reaches/reach_check/'+region+version)


###############################################################################
'''  
y = np.where(rch_cl_id == np.min(rch_cl_id))[0]
z = np.where(rch_cl_id == np.max(rch_cl_id))[0]
plt.scatter(rch_x[order_ids], rch_y[order_ids], c=rch_cl_id[order_ids], s=5, edgecolors = None)
plt.scatter(rch_x[order_ids], rch_y[order_ids], c = 'red', s=5, edgecolors = None)
#plt.scatter(rch_x[y], rch_y[y], c = 'red', s=5, edgecolors = None)

np.diff(rch_cl_id)

#plot reach as line
plt.plot(rch_x[order_ids], rch_y[order_ids], c='grey')
#plt.scatter(rch_x[order_ids], rch_y[order_ids], c=rch_cl_id[order_ids], s=10, edgecolors = None)
plt.title('Reach '+str(uniq_rch[ind]))
'''