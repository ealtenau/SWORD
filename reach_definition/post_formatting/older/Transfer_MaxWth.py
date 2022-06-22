# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 09:14:17 2021

@author: ealtenau
"""

import netCDF4 as nc
import numpy as np
from scipy import spatial as sp
import pandas as pd
from pyproj import Proj
import os
import rasterio
#import matplotlib.pyplot as plt
import utm 
import time

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
###############################################################################
###############################################################################

start_all = time.time()
region = 'NA'

fn_sword = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/outputs/Reaches_Nodes/netcdf/'+region.lower()+'_sword_v11.nc'
#fn_sword = 'E:/Users/Elizabeth Humphries/Documents/SWORD/For_Server/outputs/Reaches_Nodes_v10/netcdf/'+region.lower()+'_sword_v10.nc'
csv_dir = 'E:/Users/Elizabeth Humphries/Documents/SWORD/GRWL/bank_widths/'+region+'/'
csv_paths = np.array([f for f in os.listdir(csv_dir) if '.csv' in f])
csv_names = np.array([name[0:7] for name in csv_paths])
outpath = 'E:/Users/Elizabeth Humphries/Documents/SWORD/GRWL/max_widths/'+region.lower()+'_max_width_test.csv'

raster_dir = 'C:/Users/ealtenau/Documents/Research/SWAG/GRWL/masks/GRWL_mask_V01.01_LatLonNames/'
raster_paths = np.array([f for f in os.listdir(raster_dir) if '.tif' in f])
raster_names = np.array([name[0:7] for name in raster_paths])
raster_len = np.array([len(str(name)) for name in raster_paths])
remove = np.where(raster_len > 11)[0]
raster_names = np.delete(raster_names, remove)
raster_paths = np.delete(raster_paths, remove)

# read in global data
sword = nc.Dataset(fn_sword, 'r+')

# make array of node locations.
cl_lon = sword.groups['centerlines'].variables['x'][:]
cl_lat = sword.groups['centerlines'].variables['y'][:]
cl_rch_id = sword.groups['centerlines'].variables['reach_id'][:]
cl_pts = np.array([cl_lon, cl_lat]).T
cl_node_id = sword.groups['centerlines'].variables['node_id'][:]

node_id = sword.groups['nodes'].variables['node_id'][:]
n_rch_id = sword.groups['nodes'].variables['reach_id'][:]
n_lon = sword.groups['nodes'].variables['x'][:]
n_lat = sword.groups['nodes'].variables['y'][:]
n_chan = sword.groups['nodes'].variables['n_chan_mod'][:]
n_wth = sword.groups['nodes'].variables['width'][:]

rch_id = sword.groups['reaches'].variables['reach_id'][:]
rch_wth = sword.groups['reaches'].variables['width'][:]
rch_n_chan = sword.groups['reaches'].variables['n_chan_mod'][:]
rch_id_up = sword.groups['reaches'].variables['rch_id_up'][:]
rch_id_dn = sword.groups['reaches'].variables['rch_id_dn'][:]

node_max_wth = np.zeros(len(node_id))
rch_max_wth = np.zeros(len(rch_id))

projection = []
for ras in list(range(len(raster_paths))):
    tif = rasterio.open(raster_dir+raster_paths[ras])
    projection.append(tif.crs)

for ind in list(range(len(csv_paths))):
    
    print(ind)
    
    #read in max_wth csv data.
    csv = pd.read_csv(csv_dir+csv_paths[ind])
    csv_x = np.array(csv.x[:])
    csv_y = np.array(csv.y[:])
    csv_max_wth = np.array(csv.bank_wth[:])
    
    #find assiciated raster to current tile, and find projection to calculate
    #lat-lon from utm info.
    raster = np.where(raster_names == csv_names[ind])[0]
    myProj = Proj(projection[int(raster)]) 
    csv_lon, csv_lat = myProj(csv_x, csv_y, inverse=True)
    
    #find sword points within tile extent. 
    ll = np.array([np.min(csv_lon), np.min(csv_lat)])  # lower-left
    ur = np.array([np.max(csv_lon), np.max(csv_lat)])  # upper-right
    
    inidx = np.all(np.logical_and(ll <= cl_pts, cl_pts <= ur), axis=1)
    inbox = cl_pts[inidx]
    
    if len(inbox) == 0:
        print(ind, csv_names[ind], ' no overlapping data')
        continue

    cl_lon_clip = inbox[:,0]
    cl_lat_clip = inbox[:,1]
    cl_rch_id_clip = cl_rch_id[0,inidx]
    cl_node_id_clip = cl_node_id[0,inidx]
    
    # find closest points.    
    csv_pts = np.vstack((csv_lon, csv_lat)).T
    cl_pts_clip = np.vstack((cl_lon_clip, cl_lat_clip)).T
    kdt = sp.cKDTree(csv_pts)
    eps_dist, eps_ind = kdt.query(cl_pts_clip, k = 50) 
    
    #assign max width of closest points to new vector
    cl_max_wth = np.max(csv_max_wth[eps_ind[:]], axis = 1)
    
    #assign max width per unique reach to node locations. 
    uniq_rch = np.unique(cl_rch_id_clip)
    for idx in list(range(len(uniq_rch))):
        rch = np.where(cl_rch_id_clip == uniq_rch[idx])[0]
        max_wth1 = np.max(cl_max_wth[rch])
        assign1 = np.where(rch_id == uniq_rch[idx])[0]
        rch_max_wth[assign1] = max_wth1
        assign2 = np.where(n_rch_id == uniq_rch[idx])[0]
        node_max_wth[assign2] = max_wth1
        
        
# filling in reach wth = 1 values. 
rch_one_wth = np.where(rch_wth == 1)[0]          
for ind2 in list(range(len(rch_one_wth))): 
    nghs = np.unique(np.array([rch_id_up[:,rch_one_wth[ind2]], rch_id_dn[:,rch_one_wth[ind2]]]))
    zero = np.where(nghs == 0)[0]
    nghs = np.delete(nghs, zero)
    if len(nghs) > 0:
        ngh_wths = np.zeros(len(nghs))
        for ind3 in list(range(len(nghs))):
            r = np.where(rch_id == nghs[ind3])[0]
            ngh_wths[ind3] = rch_wth[r]
        
        #find max width of neighbors
        ngh_max_wth = np.max(ngh_wths)
        
        if ngh_max_wth > 1:
            rch_wth[rch_one_wth[ind2]] = ngh_max_wth
        else:
            rch_wth[rch_one_wth[ind2]] = 1000
    
    else:
        # if the reach has no neighbors assign default width value = 1000. 
        rch_wth[rch_one_wth[ind2]] = 1000
    
           
# filling in node wth = 1 values. 
one_wth = np.where(n_wth == 1)[0] #32,566/164,2238 for NA

for idy in list(range(len(one_wth))):
    nrch = np.where(rch_id == n_rch_id[one_wth[idy]])[0]
    rwth = rch_wth[nrch]
    n_wth[one_wth[idy]] = rwth
    
           
#fill in nodes with no max width values with regular width values. 
no_max_wth = np.where(node_max_wth <= 1)[0]
node_max_wth[no_max_wth] = n_wth[no_max_wth]
#fill in reaches with no max width values with regular width values. 
no_max_wth2 = np.where(rch_max_wth <= 1)[0]
rch_max_wth[no_max_wth2] = rch_wth[no_max_wth2]

# replacing single channel max_wth values with regular widths.
single_channels = np.where(n_chan <= 1)[0]
node_max_wth[single_channels] = n_wth[single_channels]

# assign new coeficients.
sword.groups['nodes'].variables['width'][:] = n_wth
sword.groups['reaches'].variables['width'][:] = rch_wth
sword.groups['nodes'].variables['max_width'][:] = node_max_wth
sword.groups['reaches'].variables['max_width'][:] = rch_max_wth

sword.close()

end_all = time.time()
print('DONE, Total Runtime: ' + str((end_all-start_all)/60) + ' min')
