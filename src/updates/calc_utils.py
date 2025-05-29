# -*- coding: utf-8 -*-
"""

Calculation Utilities (calc_utils.py)
=======================================

Utilities for general calculations across a variety of data. 
Many involve distance calculations and reprojection tools.  

"""

from __future__ import division
import os
import numpy as np
import netCDF4 as nc
import geopandas as gp
from shapely.geometry import LineString, Point
from geopandas import GeoSeries
import pandas as pd
import utm
from pyproj import Proj
from geopy import Point, distance

###############################################################################

class Object(object):
    """
    FUNCTION:
        Creates an empty class object to assign SWORD attributes to.
    """
    pass 

###############################################################################

def get_distances(lon,lat):
    traces = len(lon) -1
    distances = np.zeros(traces)
    for i in range(traces):
        start = (lat[i], lon[i])
        finish = (lat[i+1], lon[i+1])
        distances[i] = distance.geodesic(start, finish).m
    distances = np.append(0,distances)
    return distances

###############################################################################

def reproject_utm(latitude, longitude):

    """
    FUNCTION
    --------
        Projects all points in UTM.

    INPUTS
    ------
        latitude : latitude in degrees (1-D array)
        longitude : longitude in degrees (1-D array)

    OUTPUTS
    -------
        east : easting in UTM (1-D array)
        north : northing in UTM (1-D array)
        utm_num : UTM zone number (1-D array of utm zone numbers for each point)
        utm_let : UTM zone letter (1-D array of utm zone letters for each point)
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