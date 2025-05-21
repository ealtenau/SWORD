# -*- coding: utf-8 -*-
"""
Created on Thu Aug 06 15:49:11 2020

@author: ealtenau
"""
from __future__ import division
import netCDF4 as nc
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy import spatial as sp
import utm
from pyproj import Proj
import time
import matplotlib.pyplot as plt 
import os 

###############################################################################
'''
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
'''

def find_projection(latitude, longitude):

    """
    Modified by E.Altenau from C. Lion's function in "Tools.py" (c) 2016.

    FUNCTION:
        Projects all points in UTM.

    INPUTS
        latitude -- latitude in degrees VECTOR
        longitude -- longitude in degrees VECTOR

    OUTPUTS
        east -- easting in UTM
        north -- northing in UTM
        zone_num -- UTM zone number
        zone_let -- UTM zone letter
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
        pts = np.where(zone_num == unq_zones[idx])

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
        (east[pts], north[pts]) = myproj(longitude[pts], latitude[pts])

    return east, north, zone_num, zone_let


###############################################################################

def intersect_lakes(x, y, lakes):
    
    df = pd.DataFrame({'lon': x,'lat': y})
    points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs={'init': u'epsg:4326'})
         
    # Attaching basin codes
    intersect = gpd.sjoin(lakes, points, how="inner", op='intersects')
    intersect = pd.DataFrame(intersect)
    intersect = intersect.drop_duplicates(subset='index_right', keep='first')
    
    # Creating arrays.
    ids = np.array(intersect.index_right)
    lon = x[ids]
    lat = y[ids]

    return lon, lat
    
###############################################################################
###############################################################################
###############################################################################
region = 'NA'

grwl_dir = 'E:/Users/Elizabeth Humphries/Documents/SWORD/mask_buffer/'+region+'/'
fn_grwl = os.listdir(grwl_dir)
fn_lakes = 'E:/Users/Elizabeth Humphries/Documents/SWORD/For_Server/inputs/LakeDatabase/UCLA/FixedGeometries/NA_UCLA_Lakes2015_SWOT_FixGeom.shp'
fn_sword = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/outputs/Reaches_Nodes/netcdf/'+region.lower()+'_apriori_rivers_v10.nc'
outpath = 'E:/Users/Elizabeth Humphries/Documents/SWORD/lakes_near_rivers/'+region.lower()+'_extdist_nodes.csv'

# read in global data
lakes = gpd.GeoDataFrame.from_file(fn_lakes)
data = nc.Dataset(fn_sword, 'r+')

# make array of node locations.
nid = data.groups['nodes'].variables['node_id'][:]
nchan = data.groups['nodes'].variables['n_chan_max'][:]
nlongitude = data.groups['nodes'].variables['x'][:]
nlatitude = data.groups['nodes'].variables['y'][:]
node_locs = np.array([nlongitude,nlatitude]).T
    
# create type variable. 
ntype = np.zeros(len(nid))
for idx in list(range(len(nid))):
    ntype[idx] = np.int(np.str(nid[idx])[13:14]) 
    
# create new vector for extreme distance.     
extdist = np.repeat(20, len(nlongitude))

# LOOP through each grwl mask.     
for ind in list(range(len(fn_grwl))):
        
    start = time.time()    

    # read in and get dilated mask pixels locations. 
    fn = grwl_dir + fn_grwl[ind]
    pixels = pd.read_csv(fn)
    x = np.array(pixels.x)
    y = np.array(pixels.y) 
    
    # find all node locations within current mask pixel extents. 
    xmax = np.max(x)+0.1
    xmin = np.min(x)-0.1
    ymax = np.max(y)+0.1
    ymin = np.min(y)-0.1
    ll = np.array([xmin, ymin])  # lower-left
    ur = np.array([xmax, ymax])  # upper-right
    inbox_all = np.all(np.logical_and(ll <= node_locs, node_locs <= ur), axis=1)
    inbox = np.where(inbox_all == True)[0]
    
    if len(inbox) == 0:
        print(ind, fn_grwl[ind][0:7], 'no node locations')
        continue
    
    nlon = nlongitude[inbox]
    nlat = nlatitude[inbox]
    nt = ntype[inbox]
    num_chan = nchan[inbox] 

    # determine which mask pixels intersect the lakes. 
    plon, plat = intersect_lakes(x, y, lakes)
    
    if len(plon) == 0:
        print(ind, fn_grwl[ind][0:7], 'no lakes near rivers')
        continue
    
    #reproject node and pixel locations from latlon to utm. 
    nx, ny, __, __ = find_projection(nlat, nlon)
    px, py, __, __ = find_projection(plat, plon)
    
    # find 10 closest nodes to each lake pixel.    
    node_pts = np.vstack((nx, ny)).T
    mask_pts = np.vstack((px, py)).T
    kdt = sp.cKDTree(node_pts)
    dist, index = kdt.query(mask_pts, k = 10) 
    # remove duplicates.
    uniq_ids = np.unique(index)
    remove = np.where(nt[uniq_ids] == 3)[0]
    final_ids = np.delete(uniq_ids, remove)
    dist_thresh = np.repeat(20, len(nx)) 
    multi = np.where(num_chan[final_ids] > 1)[0]
    single = np.where(num_chan[final_ids] <= 1)[0]

    dist_thresh[final_ids[single]] = 1
    dist_thresh[final_ids[multi]] = 2

    # insert subset of extreme distance coef values into global locations. 
    extdist[inbox] = dist_thresh
    
    end = time.time()
    print(ind, fn_grwl[ind][0:7], 'Runtime: ' + str(np.round((end-start)/60, 2)) + ' min')

############################## outside loop!
    
# assign new coeficients to netcdf and output csv file.
data.groups['nodes'].variables['ext_dist_coef'][:] = extdist

data2 = pd.DataFrame(np.array([nlongitude, nlatitude, nid, ntype, extdist, nchan])).T
data2.columns = ['lon', 'lat', 'id', 'type', 'dist_thresh', 'num_channels']
data2.to_csv(outpath)

data.close()
    
