# -*- coding: utf-8 -*-
"""
Created on Thu Sep 03 09:03:42 2020

@author: ealtenau
"""
from __future__ import division
import utm
from pyproj import Proj
import numpy as np
from scipy import spatial as sp
import pandas as pd
import matplotlib.pyplot as plt


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
###############################################################################
###############################################################################

grod_fn = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/inputs/GROD/GROD_ALL.csv'
hf_fn = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/inputs/HydroFalls/hydrofalls.csv'
outpath = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/inputs/HydroFalls/hydrofalls_filt.csv'

grod_info = pd.read_csv(grod_fn)
grod_lat =  np.array(grod_info.lat)
grod_lon =  np.array(grod_info.lon)

hf_info = pd.read_csv(hf_fn)
hf_lat =  np.array(hf_info.LAT_HYSD)
hf_lon =  np.array(hf_info.LONG_HYSD)
hf_id = np.array(hf_info.FALLS_ID)
hf_cont = np.array(hf_info.CONTINENT)
hf_conf = np.array(hf_info.CONFIDENCE)


hf_x, hf_y, __, __ = find_projection(hf_lat, hf_lon)
grod_x, grod_y, __, __ = find_projection(grod_lat, grod_lon)

hf_pts = np.vstack((hf_x, hf_y)).T
grod_pts = np.vstack((grod_x, grod_y)).T

kdt = sp.cKDTree(hf_pts)
grod_dist, grod_idx = kdt.query(grod_pts, k = 1)

close_pts = np.unique(grod_idx[np.where(grod_dist <= 500)[0]])

hf_lat = np.delete(hf_lat, close_pts)
hf_lon =  np.delete(hf_lon, close_pts)
hf_id = np.delete(hf_id, close_pts)
hf_cont = np.delete(hf_cont, close_pts)
hf_conf = np.delete(hf_conf, close_pts)

df = pd.DataFrame(np.array([hf_id, hf_lat, hf_lon, hf_conf, hf_cont]).T)
df.columns = ['ID', 'lat', 'lon', 'confidence', 'continent']
df.to_csv(outpath)

plt.scatter(grod_lon, grod_lat, c='blue', s = 5, edgecolors = None)
plt.scatter(hf_lon, hf_lat, c='red', s = 5, edgecolors = None)
plt.scatter(hf_lon[close_pts], hf_lat[close_pts], c='gold', s = 5, edgecolors = None)
