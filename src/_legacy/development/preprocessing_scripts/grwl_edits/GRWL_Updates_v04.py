# -*- coding: utf-8 -*-

"""
-------------------------------------------------------------------------------
                    Adding GRWL Edits to Original GRWL
-------------------------------------------------------------------------------
Created on Sun Feb 03 12:58:11 2019

Copyright (c) 2018-2021 UNC Chapel Hill. All rights reserved.
Created by E. Altenau.

DESCRIPTON:
    This script takes GRWL (Global River Widths from Landsat,
    [Allen and Pavelsky, 2018]) edit requests and combines them with the
    original GRWL shapefiles.

INPUTS:
    grwl_dir -- Directory to all GRWL files (.shp)
    edits_dir -- Directory to all GRWL edit files (.csv)
    outpath -- Directory path to write files
    region -- Continent directory name (ex: "NA")
    use_updated_grwl -- A flag labeled "True" or "False" to indicate whether 
        the user wants to start from scratch with the original GRWL files, 
        or start with the current GRWL_2.0 files that have already have updates 
        implemented. 

OUTPUTS:
    Combined shapefiles -- Shapefile containing the original and edited GRWL
        data. The filename will contain the GRWL tile location and modifier
        "_edit" (ex: n60w150_edit.shp).
-------------------------------------------------------------------------------
"""

from __future__ import division
import os
main_dir = os.getcwd()
import time
import utm
from osgeo import ogr
from osgeo import osr
import numpy as np
import pandas as pd
from scipy import spatial as sp
import sys
import geopandas as gp
from pyproj import Proj
import argparse

###############################################################################
################################# Functions ###################################
###############################################################################

class Object(object):
    """
    FUNCTION:
        Creates class object to assign attributes to.
    """
    pass

###############################################################################

def getListOfFiles(dirName):

    """
    FUNCTION:
        For the given path, gets a recursive list of all files in the directory tree.

    INPUTS
        dirName -- Input directory

    OUTPUTS
        allFiles -- List of files under directory
    """

    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

###############################################################################

def find_projection(latitude, longitude):

    """
    Modified by E.Altenau from C. Lion's function in "Tools.py" (c) 2016.

    FUNCTION:
        Projects all points from lat/lon coordinates to UTM coordinates.

    INPUTS
        latitude -- latitude in degrees 1-D ARRAY
        longitude -- longitude in degrees 1-D ARRAY

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
        (east_int[ind], north_int[ind],zone_num[ind], zone_let_int) = utm.from_latlon(latitude[ind],longitude[ind])
        zone_let.append(zone_let_int)

    # Finds the unique UTM zones and converts the lat/lon pairs to UTM.
    unq_zones = np.unique(zone_num)
    utm_let = np.unique(zone_let)[0]

    for idx in list(range(len(unq_zones))):
        pts = np.where(zone_num == unq_zones[idx])

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
        (east[pts], north[pts]) = myproj(longitude[pts], latitude[pts])

    return east, north, zone_num, zone_let

###############################################################################

def smooth_grwl(grwl):

    """
    FUNCTION:
        Smooths original x-y locations in GRWL centerlines to remove
        "stair-step" effects of the original raster cells that created the
        GRWL centerlines.

    INPUTS
        grwl -- Object containing attributes for the GRWL centerline.

    OUTPUTS
        new_x -- Smoothed easting locations (1-D array).
        new_y -- Smoothed northing locations (1-D array).
    """

    # Calculating median x-y values of the closest 5 neighbors for each point.
    uniq_segs = np.unique(grwl.finalID)
    new_x = np.copy(grwl.x)
    new_y = np.copy(grwl.y)
    for ind in list(range(len(uniq_segs))):
        seg = np.where(grwl.finalID == uniq_segs[ind])[0]
        if len(seg) <= 10:
            continue

        # perform spatial query to find the five closes points to each segment point.
        pts = np.vstack((grwl.x[seg], grwl.y[seg])).T
        kdt = sp.cKDTree(pts)
        pt_dist, pt_ind = kdt.query(pts, k = 5)

        eps = np.where(grwl.finaleps[seg] > 0)
        if len(eps) < 2:
            keep = np.array([0, 1, len(seg)-2, len(seg)-1])
        else:
            keep = np.array([np.min(eps), np.min(eps)+1, np.max(eps)-1, np.max(eps)])

        # Averaging 5 closest points and preserving the endpoints from smoothing window.
        new_vals_x = np.mean(grwl.x[seg][pt_ind[:]], axis = 1)
        new_vals_y = np.mean(grwl.y[seg][pt_ind[:]], axis = 1)
        new_vals_x[keep] = grwl.x[seg][keep]
        new_vals_x[keep] = grwl.x[seg][keep]
        new_vals_y[keep] = grwl.y[seg][keep]
        new_vals_y[keep] = grwl.y[seg][keep]
        new_x[seg] = new_vals_x
        new_y[seg] = new_vals_y

    return new_x, new_y

###############################################################################

def read_grwl(filename):

    """
    FUNCTION:
        Opens GRWL shapefile and returns the fields inside a "grwl" object.
        Each field is stored inside the object in array format.

    INPUTS
        filename -- GRWL shapefile

    OUTPUTS
        grwl.lon -- Longitude (wgs84)
        grwl.lat -- Latitude (wgs84)
        grwl.wth -- Width (m)
        grwl.segID -- Segment ID
        grwl.nchan -- Number of Channels
        grwl.lake -- Lake Flag
        grwl.x -- Easting (m, utm)
        grwl.y -- Northing (m, utm)
        grwl.segInd -- Segment Point Index
        grwl.elv -- SRTM Elevation (m)
    """

    # Opening grwl shapefile and extracting layer attribute information.
    fn_grwl = filename
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shape = driver.Open(fn_grwl)
    layer = shape.GetLayer()
    numFeatures = layer.GetFeatureCount()

    attributes = []
    ldefn = layer.GetLayerDefn()
    for n in range(ldefn.GetFieldCount()):
        fdefn = ldefn.GetFieldDefn(n)
        attributes.append(fdefn.name)

    # Creating empty arrays to fill in with grwl attributes.
    grwl = Object()
    grwl.x = np.zeros(numFeatures)
    grwl.y = np.zeros(numFeatures)
    grwl.wth = np.zeros(numFeatures)
    grwl.nchan = np.zeros(numFeatures)
    grwl.segID = np.zeros(numFeatures)
    grwl.segInd = np.zeros(numFeatures)
    grwl.lake = np.zeros(numFeatures)
    grwl.lon = np.zeros(numFeatures)
    grwl.lat = np.zeros(numFeatures)
    grwl.elv = np.zeros(numFeatures)

    # Saving data.
    cnt = 0
    for feature in range(numFeatures):
        grwl.x[cnt] = layer.GetFeature(feature).GetField(attributes[0])
        grwl.y[cnt] = layer.GetFeature(feature).GetField(attributes[1])
        grwl.wth[cnt] = layer.GetFeature(feature).GetField(attributes[2])
        grwl.nchan[cnt] = layer.GetFeature(feature).GetField(attributes[3])
        grwl.segID[cnt] = layer.GetFeature(feature).GetField(attributes[4])
        grwl.segInd[cnt] = layer.GetFeature(feature).GetField(attributes[5])
        grwl.lake[cnt] = layer.GetFeature(feature).GetField(attributes[6])
        grwl.lon[cnt] = layer.GetFeature(feature).GetField(attributes[7])
        grwl.lat[cnt] = layer.GetFeature(feature).GetField(attributes[8])
        grwl.elv[cnt] = layer.GetFeature(feature).GetField(attributes[9])
        cnt += 1

    # Editing lake flag values.
    grwl.lake[np.where(grwl.lake == 255)[0]] = 0
    grwl.lake[np.where(grwl.lake == 250)[0]] = 0
    grwl.lake[np.where(grwl.lake == 180)[0]] = 1
    grwl.lake[np.where(grwl.lake == 181)[0]] = 1
    grwl.lake[np.where(grwl.lake == 163)[0]] = 1
    grwl.lake[np.where(grwl.lake == 126)[0]] = 3
    grwl.lake[np.where(grwl.lake == 125)[0]] = 3
    grwl.lake[np.where(grwl.lake == 86)[0]] = 2

    return grwl

###############################################################################

def read_grwl_updated(filename):

    """
    FUNCTION:
        Opens the updated GRWL shapefiles and returns the
        fields inside a "grwl" object. Each field is stored inside the object
        in array format. This function can replace the "read_grwl"
        function if manual edits have been made to the merged shapefiles
        requireing that geometry to be used.

    INPUTS
        filename -- Updated GRWL shapefile

    OUTPUTS
        grwl.lon -- Longitude (wgs84)
        grwl.lat -- Latitude (wgs84)
        grwl.wth -- Width (m)
        grwl.segID -- Segment ID
        grwl.nchan -- Number of Channels
        grwl.lake -- Lake Flag
        grwl.x -- Easting (m, utm)
        grwl.y -- Northing (m, utm)
        grwl.segInd -- Segment Point Index
    """

    # Opening grwl shapefile and extracting layer attribute information.
    fn_grwl = filename
    shp = gp.read_file(fn_grwl)
    geom = [i for i in shp.geometry]
    lon = np.zeros(len(geom))
    lat = np.zeros(len(geom))
    for ind in list(range(len(geom))):
        lon[ind] = np.array(geom[ind].coords.xy[0])
        lat[ind] = np.array(geom[ind].coords.xy[1])

    east, north, __, __ = find_projection(lat, lon)

    grwl = Object()
    grwl.lon = lon
    grwl.lat = lat
    grwl.x = east
    grwl.y = north
    grwl.wth = np.array(shp['width_m'])
    grwl.nchan = np.array(shp['nchannels'])
    grwl.segID = np.array(shp['segmentID'])
    grwl.segInd = np.array(shp['segmentInd'])
    grwl.lake = np.array(shp['lakeFlag'])
    grwl.eps = np.array(shp['endpoints'])
    grwl.manual = np.array(shp['manual_add'])

    # Editing lake flag values.
    grwl.lake[np.where(grwl.lake == 255)[0]] = 0
    grwl.lake[np.where(grwl.lake == 250)[0]] = 0
    grwl.lake[np.where(grwl.lake == 180)[0]] = 1
    grwl.lake[np.where(grwl.lake == 181)[0]] = 1
    grwl.lake[np.where(grwl.lake == 163)[0]] = 1
    grwl.lake[np.where(grwl.lake == 126)[0]] = 3
    grwl.lake[np.where(grwl.lake == 125)[0]] = 3
    grwl.lake[np.where(grwl.lake == 86)[0]] = 2

    return grwl

###############################################################################

def read_grwl_merge(filename):

    """
    FUNCTION:
        Opens the updated GRWL shapefiles and returns the
        fields inside a "grwl" object. Each field is stored inside the object
        in array format. This function can replace the "read_grwl"
        function if manual edits have been made to the merged shapefiles
        requireing that geometry to be used.

    INPUTS
        filename -- Updated GRWL shapefile

    OUTPUTS
        grwl.lon -- Longitude (wgs84)
        grwl.lat -- Latitude (wgs84)
        grwl.wth -- Width (m)
        grwl.segID -- Segment ID
        grwl.nchan -- Number of Channels
        grwl.lake -- Lake Flag
        grwl.x -- Easting (m, utm)
        grwl.y -- Northing (m, utm)
        grwl.segInd -- Segment Point Index
    """

    # Opening grwl shapefile and extracting layer attribute information.
    fn_grwl = filename
    shp = gp.read_file(fn_grwl)
    geom = [i for i in shp.geometry]
    lon = np.zeros(len(geom))
    lat = np.zeros(len(geom))
    for ind in list(range(len(geom))):
        lon[ind] = np.array(geom[ind].coords.xy[0])
        lat[ind] = np.array(geom[ind].coords.xy[1])

    east, north, __, __ = find_projection(lat, lon)

    grwl = Object()
    grwl.lon = lon
    grwl.lat = lat
    grwl.x = east
    grwl.y = north
    grwl.wth = np.array(shp['p_width'])
    grwl.nchan = np.array(shp['nchan'])
    grwl.segID = np.array(shp['segID'])
    grwl.segInd = np.array(shp['segInd'])
    grwl.lake = np.array(shp['lakeflag'])
    grwl.eps = np.array(shp['endpoints'])
    grwl.manual = np.array(shp['manual_add'])

    # Editing lake flag values.
    grwl.lake[np.where(grwl.lake == 255)[0]] = 0
    grwl.lake[np.where(grwl.lake == 250)[0]] = 0
    grwl.lake[np.where(grwl.lake == 180)[0]] = 1
    grwl.lake[np.where(grwl.lake == 181)[0]] = 1
    grwl.lake[np.where(grwl.lake == 163)[0]] = 1
    grwl.lake[np.where(grwl.lake == 126)[0]] = 3
    grwl.lake[np.where(grwl.lake == 125)[0]] = 3
    grwl.lake[np.where(grwl.lake == 86)[0]] = 2

    return grwl

###############################################################################

def read_edits_gpkg(filename, max_seg):

    """
    FUNCTION:
        Opens the updated GRWL shapefiles and returns the
        fields inside a "grwl" object. Each field is stored inside the object
        in array format. This function can replace the "read_grwl"
        function if manual edits have been made to the merged shapefiles
        requireing that geometry to be used.

    INPUTS
        filename -- Updated GRWL shapefile

    OUTPUTS
        grwl.lon -- Longitude (wgs84)
        grwl.lat -- Latitude (wgs84)
        grwl.wth -- Width (m)
        grwl.segID -- Segment ID
        grwl.nchan -- Number of Channels
        grwl.lake -- Lake Flag
        grwl.x -- Easting (m, utm)
        grwl.y -- Northing (m, utm)
        grwl.segInd -- Segment Point Index
    """

    # Opening grwl shapefile and extracting layer attribute information.
    fn_grwl = filename
    gpkg = gp.read_file(fn_grwl)
    geom = [i for i in gpkg.geometry]
    lon = np.zeros(len(geom))
    lat = np.zeros(len(geom))
    for ind in list(range(len(geom))):
        lon[ind] = np.array(geom[ind].coords.xy[0])
        lat[ind] = np.array(geom[ind].coords.xy[1])

    east, north, __, __ = find_projection(lat, lon)

    edits = Object()
    edits.lon = lon
    edits.lat = lat
    edits.x = east
    edits.y = north
    edits.seg = np.array(gpkg['seg'])
    edits.lake = np.repeat(1, len(edits.x))

    cnt=max_seg
    rch_segs = np.array([int(r) for r in edits.seg])
    unq_rch = np.unique(edits.seg)
    for ind in list(range(len(unq_rch))):
        vals = np.where(edits.seg == unq_rch[ind])[0]
        edits.seg[vals] = cnt=cnt+1

    return edits

###############################################################################   

def read_edits_csv(edit_files, max_seg):

    """
    FUNCTION:
        Opens edit file information to be added to GRWL shapefile and stores
        it in an "edit" object.

    INPUTS
        edit_files -- List of Edit files (.csv format)
        max_seg -- Max segment ID of orginial GRWL data.

    OUTPUTS
        edits.x = Easting (m, utm)
        edits.y = Northing (m, utm)
        edits.seg = Segment ID
        edits.lake = Lake Flag
    """

    # Read in tile edits:
    edits = Object()

    # Read in and save information for multiple edit files per GRWL tile.
    if len(edit_files) == 2:
        e1 = pd.read_csv(edit_files[0], sep=',', delimiter=None, header='infer')
        e1_x = np.array(e1.x)
        e1_y = np.array(e1.y)
        e1_seg = np.array(e1.seg)
        e1_lake = np.array(e1.lakeFlag)

        e2 = pd.read_csv(edit_files[1], sep=',', delimiter=None, header='infer')
        e2_x = np.array(e2.x)
        e2_y = np.array(e2.y)
        e2_seg = np.array(e2.seg)
        e2_lake = np.array(e2.lakeFlag)

        if np.max(e1_seg) > np.max(e2_seg):
            e2_seg = e2_seg+np.max(e1.seg)

        if np.max(e2_seg) > np.max(e1_seg):
            e1_seg = e1_seg+np.max(e2.seg)

        if np.max(e1_seg) == np.max(e2_seg):
            e2_seg = e2_seg+np.max(e1.seg)

        edits.x = np.insert(e1_x, len(e1_x), e2_x)
        edits.y = np.insert(e1_y, len(e1_y), e2_y)
        edits.seg = np.insert(e1_seg, len(e1_seg), e2_seg)
        edits.seg = edits.seg+max_seg
        edits.lake = np.insert(e1_lake, len(e1_lake), e2_lake)

    # Read in and save information for one edit file per GRWL tile.
    if len(edit_files) == 1:
        edit_fn = pd.read_csv(edit_files[0], sep=',', delimiter=None, header='infer')
        edits.x = np.array(edit_fn.x)
        edits.y = np.array(edit_fn.y)
        edits.seg = np.array(edit_fn.seg)
        edits.seg = edit_fn.seg+max_seg
        edits.lake = edit_fn.lakeFlag

    # Editing new lake flag values.
    edits.lake[np.where(edits.lake == 255)[0]] = 0
    edits.lake[np.where(edits.lake == 250)[0]] = 0
    edits.lake[np.where(edits.lake == 180)[0]] = 1
    edits.lake[np.where(edits.lake == 181)[0]] = 1
    edits.lake[np.where(edits.lake == 163)[0]] = 1
    edits.lake[np.where(edits.lake == 126)[0]] = 3
    edits.lake[np.where(edits.lake == 125)[0]] = 3
    edits.lake[np.where(edits.lake == 86)[0]] = 2
    edits.lake = np.asarray(edits.lake)

    return edits

###############################################################################

def update_segID(grwl):

    """
    FUNCTION:
        Renumbers GRWL segment ID's within a GRWL tile. Due to isolated errors,
        not every GRWL segment has a unique segment ID.

    INPUTS
        grwl -- Object containing attributes for the GRWL centerlines.

    OUTPUTS
        new_grwl_id -- New Segment IDs in 1-D array format.
    """

    # Create empty arrays to fill.
    new_grwl_id = np.zeros(len(grwl.segID))
    unq_grwl_ids = np.unique(grwl.segID)
    count = 0

    # Calculates distance between points for each GRWL segment. If there is a
    # distance greater than 100 m between points the segment is split in two
    # and re-numbered at that location.
    for num in range(len(unq_grwl_ids)):
        seg = np.where(grwl.segID == unq_grwl_ids[num])
        dist = np.sqrt((grwl.x[seg][0]-grwl.x[seg])**2 + (grwl.y[seg][0]-grwl.y[seg])**2)
        dist_diff = np.diff(dist)
        seg_divs = list(np.where(abs(dist_diff) > 100)[0]+1)

        if len(seg_divs) == 1:
            new_grwl_id[seg[0][0:int(seg_divs[0])]] = count
            new_grwl_id[seg[0][int(seg_divs[0]):len(seg[0])]] = count+1
            count+=2

        if len(seg_divs) > 1:

            for d in range(len(seg_divs)):

                if d == 0:
                    new_grwl_id[seg[0][0:int(seg_divs[d])]] = count
                    new_grwl_id[seg[0][int(seg_divs[d]):int(seg_divs[d+1])]] = count+1
                    count+=2
                if d == len(seg_divs)-1:
                    new_grwl_id[seg[0][int(seg_divs[d]):len(seg[0])]] = count
                    count+=1
                else:
                    new_grwl_id[seg[0][int(seg_divs[d]):int(seg_divs[d+1])]] = count
                    count+=1

        if len(seg_divs) == 0:
            new_grwl_id[seg] = count
            count+=1

    return new_grwl_id

###############################################################################

def find_edit_endpoints(edits):

    """
    FUNCTION:
        Creates a new 1-D array that contains the endpoints for each
        edited centerline segment. 0 = not an endpoint, 1 = first endpoint,
        2 = second endpoint.

    INPUTS
        edits -- Object containing attributes for the edited centerlines.

    OUTPUTS
        endpoints -- Endpoint locations for all edit segments.
    """

    # Loop through segments.
    endpoints = np.zeros(len(edits.seg))
    uniq_segs = np.unique(edits.seg)
    for ind in list(range(len(uniq_segs))):
        seg = np.where(edits.seg == uniq_segs[ind])[0]
        seg_x = edits.x[seg]
        seg_y = edits.y[seg]

        # Removing duplicate coordinates.
        coords_df = pd.DataFrame(np.array([seg_x, seg_y]).T)
        duplicates = np.where(pd.DataFrame.duplicated(coords_df))
        if len(duplicates) > 0:
            seg_x = np.delete(seg_x, duplicates)
            seg_y = np.delete(seg_y, duplicates)
            new_seg = np.delete(seg, duplicates)
        else:
            new_seg = np.copy(seg)

        # For each segment calculate distance between points and identify the
        # points with only two neighbors < 60 m away.
        count = 1
        for point in list(range(len(new_seg))):
            dist = np.sqrt((seg_x[point]-seg_x)**2 + (seg_y[point]-seg_y)**2)
            if len(np.where(dist < 60)[0]) < 3:
                endpoints[new_seg[point]] = count
                count = count+1

        # Default to the min and max indexes if no endpoints are found.
        eps = np.where(endpoints[seg] > 0)[0]
        if len(eps) < 2:
            mx = np.where(seg == np.max(seg))
            mn = np.where(seg == np.min(seg))
            endpoints[seg[eps]] = 0
            endpoints[seg[mn]] = 1
            endpoints[seg[mx]] = 2

    return endpoints

###############################################################################

def label_grwl_endpoints(grwl):

    """
    FUNCTION:
        Creates a new 1-D array that contains the endpoints for each
        GRWL segment. 0 = not an endpoint, 1 = first endpoint,
        2 = second endpoint.

    INPUTS
        grwl -- Object containing attributes for the GRWL centerlines.

    OUTPUTS
        endpoints -- Endpoint locations for all GRWL segments.
    """

    # Loop through segments and label the endpoints based on segment indexes.
    endpoints = np.zeros(len(grwl.segID))
    uniq_segs = np.unique(grwl.segID)
    for ind in list(range(len(uniq_segs))):
        seg = np.where(grwl.segID == uniq_segs[ind])[0]
        endpoints[seg[np.where(grwl.segInd[seg] == np.min(grwl.segInd[seg]))]] = 1
        endpoints[seg[np.where(grwl.segInd[seg] == np.max(grwl.segInd[seg]))]] = 2

    return endpoints

###############################################################################

def order_edits(edits):

    """
    FUNCTION:
        Creates a new 1-D array that contains ordered point indexes
        for each edited centerline segment.

    INPUTS
         edits -- Object containing attributes for the edited centerlines.

    OUTPUTS
        edits_segInd -- Ordered point index for each edited segment.
    """

    # Loop through edited segments.
    edits_segInd = np.zeros(len(edits.seg))
    edits_segDist = np.zeros(len(edits.seg))
    uniq_segs = np.unique(edits.seg)
    for ind in list(range(len(uniq_segs))):
        seg = np.where(edits.seg == uniq_segs[ind])[0]
        seg_x = edits.x[seg]
        seg_y = edits.y[seg]
        eps = np.where(edits.eps[seg] > 0)[0]

        # If no endpoints are found default to the first index value to start.
        if len(eps) == 0: # condition added on 9/19/19.
            eps = np.array([0])

        edits_segInd[seg[eps[0]]]=1
        edits_segDist[seg[eps[0]]]=0
        idx = eps[0]

        # Order points in a segment starting from the first endpoint.
        count = 2
        while np.min(edits_segInd[seg]) == 0:
            d = np.sqrt((seg_x[idx]-seg_x)**2 + (seg_y[idx]-seg_y)**2)
            dzero = np.where(edits_segInd[seg] == 0)[0]
            next_pt = dzero[np.where(d[dzero] == np.min(d[dzero]))[0]][0]
            edits_segInd[seg[next_pt]] = count
            edits_segDist[seg[next_pt]] = d[next_pt]
            count = count+1
            idx = next_pt

    return edits_segInd

###############################################################################

def find_tributary_junctions(grwl, edits):

    """
    FUNCTION:
        Creates a new 1-D that contains the locations of existing GRWL
        segments that insersect an edited segment and therefore need to
        be cut at the new tributary junction.

    INPUTS
        grwl --  Object containing attributes for the GRWL centerlines.
        edits -- Object containing attributes for the edited centerlines.

    OUTPUTS
        tribs -- Locations along a GRWL segment where the segment should
            be cut: 0 - no tributary, 1 - tributary.
    """

    # Loop through each edited segment and calculate closet GRWL points to the
    # edited segment endpoints.
    tribs = np.zeros(len(grwl.newID))
    grwl_pts = np.vstack((grwl.x, grwl.y)).T
    uniq_segs = np.unique(edits.seg)
    for ind in list(range(len(uniq_segs))):

        # Isolate endpoints for the edited segment.
        seg = np.where(edits.seg == uniq_segs[ind])[0]
        if len(seg) == 1:
            eps = np.array([0,0])
        else:
            pt1 = np.where(edits.ind[seg] == np.min(edits.ind[seg]))[0]
            pt2 = np.where(edits.ind[seg] == np.max(edits.ind[seg]))[0]
            eps = np.array([pt1,pt2]).T

        # Perform spatial query of closest GRWL points to the edited segment
        # endpoints.
        ep_pts = np.vstack((edits.x[seg[eps]], edits.y[seg[eps]])).T
        kdt = sp.cKDTree(grwl_pts)

        if len(seg) < 3:
            pt_dist, pt_ind = kdt.query(ep_pts, k = 4,
                                        distance_upper_bound = 45.0)
        elif 3 <= len(seg) and len(seg) <= 6:
            pt_dist, pt_ind = kdt.query(ep_pts, k = 10,
                                        distance_upper_bound = 100.0)
        elif len(seg) > 6:
            pt_dist, pt_ind = kdt.query(ep_pts, k = 10,
                                        distance_upper_bound = 200.0)

        ep1_ind = pt_ind[0,:]
        ep1_dist = pt_dist[0,:]
        na1 = np.where(ep1_ind == len(grwl_pts))
        ep1_dist = np.delete(ep1_dist, na1)
        ep1_ind = np.delete(ep1_ind, na1)

        ep2_ind = pt_ind[1,:]
        ep2_dist = pt_dist[1,:]
        na2 = np.where(ep2_ind == len(grwl_pts))
        ep2_dist = np.delete(ep2_dist, na2)
        ep2_ind = np.delete(ep2_ind, na2)

        ep1_segs = np.unique(grwl.newID[ep1_ind])
        ep2_segs = np.unique(grwl.newID[ep2_ind])

        # If there is only one neighboring GRWL segment, designate it as a
        # tributary junction if the edited segment endpoint falls in the middle
        # of the segment.
        if len(ep1_segs) == 1:
            ep1_min = np.min(grwl.segInd[np.where(grwl.newID == ep1_segs[0])[0]])
            ep1_max = np.max(grwl.segInd[np.where(grwl.newID == ep1_segs[0])[0]])
            if np.min(grwl.segInd[ep1_ind]) > ep1_min+5 and np.max(grwl.segInd[ep1_ind]) < ep1_max-5:
                tribs[ep1_ind[0]] = 1

        if len(ep2_segs) == 1:
            ep2_min = np.min(grwl.segInd[np.where(grwl.newID == ep2_segs[0])[0]])
            ep2_max = np.max(grwl.segInd[np.where(grwl.newID == ep2_segs[0])[0]])
            if np.min(grwl.segInd[ep2_ind]) > ep2_min+5 and np.max(grwl.segInd[ep2_ind]) < ep2_max-5:
                tribs[ep2_ind[0]] = 1

    return tribs

###############################################################################

def cut_segments(grwl, start_seg):

    """
    FUNCTION:
        Creates a new 1-D array that contains unique segment IDs for the GRWL
        segments that need to be cut at tributary junctions.

    INPUTS
        grwl --  Object containing attributes for the GRWL centerlines.
        start_seg -- Starting ID value to assign to the new cut segments. This
            is typically assigned the max_seg value + 1.

    OUTPUTS
        new_segs -- Updated Segment IDs.
    """

    new_segs = np.copy(grwl.newID)
    cut = np.where(grwl.tribs == 1)[0]
    cut_segs = np.unique(grwl.newID[cut])
    seg_id = start_seg

    # Loop through segments that contain tributary junctions and identify
    # the new boundaries of the segment to cut and re-number.
    for ind in list(range(len(cut_segs))):
        seg = np.where(grwl.newID == cut_segs[ind])[0]
        num_tribs = np.where(grwl.tribs[seg] == 1)[0]
        max_ind = np.where(grwl.segInd[seg] == np.max(grwl.segInd[seg]))[0]
        min_ind = np.where(grwl.segInd[seg] == np.min(grwl.segInd[seg]))[0]
        bounds = np.insert(num_tribs, 0, min_ind)
        bounds = np.insert(bounds, len(bounds), max_ind)
        for idx in list(range(len(bounds)-1)):
            id1 = bounds[idx]
            id2 = bounds[idx+1]
            new_segs[seg[id1:id2]] = seg_id
            seg_id = seg_id+1

    return new_segs

###############################################################################

def edit_short_segments(grwl, edits):

    """
    FUNCTION:
        Creates a new 1-D array that contains updated segment IDs for short
        edit segments. If an edited segment contains < 100 points, it is
        assigned to the closest GRWL segment.

    INPUTS
        grwl --  Object containing attributes for the GRWL centerlines.
        edits -- Object containing attributes for the edited centerlines.

    OUTPUTS
        new_segs -- Updated Segment IDs for the GRWL edits.
    """

    # Loop through edited segments.
    new_segs = np.copy(edits.seg)
    uniq_segs = np.unique(new_segs)
    for ind in list(range(len(uniq_segs))):
        seg = np.where(new_segs == uniq_segs[ind])[0]
        lakes = len(np.where(edits.lake[seg] > 0)[0])
        length = len(seg)

        # If the edited segment contains less than 100 points, find the closest
        # neighboring segments and assign the edited segment to the most suitable
        # GRWL or edited segment.
        if length < 100 and lakes == 0:

            # Create new point arrays that contain all GRWL and edited points
            # not included in the current segment.
            edits_x2 = np.delete(edits.x, seg)
            edits_y2 = np.delete(edits.y, seg)
            edits_width2 = np.delete(edits.wth, seg)
            edits_seg2 = np.delete(new_segs, seg)

            all_width = np.insert(grwl.wth, 0, edits_width2)
            all_x = np.insert(grwl.x, 0, edits_x2)
            all_y = np.insert(grwl.y, 0, edits_y2)
            all_segs = np.insert(grwl.finalID, 0, edits_seg2)

            all_pts = np.vstack((all_x, all_y)).T

            if len(seg) == 1:
                eps = np.array([0,0])
            else:
                eps = np.where(edits.eps[seg] > 0)[0]

            # Isolate endpoints for the short segment.
            ep_pts = np.vstack((edits.x[seg[eps]], edits.y[seg[eps]])).T
            kdt = sp.cKDTree(all_pts)

            # Perform spatial query of the closest remaining GRWL and edit points.
            if len(seg) < 4:
                pt_dist, pt_ind = kdt.query(ep_pts, k = 4, distance_upper_bound = 45.0)
            elif 4 <= len(seg) and len(seg) <= 6:
                pt_dist, pt_ind = kdt.query(ep_pts, k = 10, distance_upper_bound = 100.0)
            elif len(seg) > 6:
                pt_dist, pt_ind = kdt.query(ep_pts, k = 10, distance_upper_bound = 200.0)

            # Remove points that have "Inf" values in spatial query.
            ep1_ind = pt_ind[0,:]
            ep1_dist = pt_dist[0,:]
            na1 = np.where(ep1_ind == len(all_segs))
            ep1_dist = np.delete(ep1_dist, na1)
            ep1_ind = np.delete(ep1_ind, na1)

            ep2_ind = pt_ind[1,:]
            ep2_dist = pt_dist[1,:]
            na2 = np.where(ep2_ind == len(all_segs))
            ep2_dist = np.delete(ep2_dist, na2)
            ep2_ind = np.delete(ep2_ind, na2)

            # Final neighboring segments on each end.
            ep1_segs = np.unique(all_segs[ep1_ind])
            ep2_segs = np.unique(all_segs[ep2_ind])

            # Determine what segment the short segment should be assigned to
            # based on the following conditions:

            # No segments found
            if len(ep1_segs) == 0 and len(ep2_segs) == 0:
                continue

            # Multiple segments on one end only.
            elif len(ep1_segs) > 1 and len(ep2_segs) == 0:
                continue

            # Multiple segments on one end only.
            elif len(ep1_segs) == 0 and len(ep2_segs) > 1:
                continue

            # Multiple segments on both ends.
            elif len(ep1_segs) > 1 and len(ep2_segs) > 1:
                continue

            # Multiple segments on one end and one segment on the other end.
            elif len(ep1_segs) > 1 and len(ep2_segs) == 1:
                new_segs[seg] = ep2_segs

            # Multiple segments on one end and one segment on the other end.
            elif len(ep1_segs) == 1 and len(ep2_segs) > 1:
                new_segs[seg] = ep1_segs

            # One segment on one end only.
            elif len(ep1_segs) == 1 and len(ep2_segs) == 0:
                new_segs[seg] = ep1_segs

            # One segment on one end only.
            elif len(ep1_segs) == 0 and len(ep2_segs) == 1:
                new_segs[seg] = ep2_segs

            # One segment on BOTH ends.
            elif len(ep1_segs) == 1 and len(ep2_segs) == 1:
                if ep1_segs == ep2_segs:
                    new_segs[seg] = ep1_segs
                else:
                    ep1_wth = np.max(all_width[np.where(all_segs == ep1_segs)])
                    ep2_wth = np.max(all_width[np.where(all_segs == ep2_segs)])
                    if ep1_wth < ep2_wth:
                        new_segs[seg] = ep1_segs
                    elif ep1_wth > ep2_wth:
                        new_segs[seg] = ep2_segs
                    elif ep1_wth == ep2_wth:
                        new_segs[seg] = ep1_segs

        # If the segment is not short
        else:
            continue

    return new_segs

###############################################################################

def update_indexes(grwl, edits):

    """
    FUNCTION:
        Creates a new 1-D array that re-orders and updates segment point
        indexes and endponts to account for any changes in segment IDs.

    INPUTS
        grwl --  Object containing attributes for the GRWL centerlines.
        edits -- Object containing attributes for the edited centerlines.

    OUTPUTS
        new_rch_ind -- Updated segment point indexes for the final centerline
            segments.
        new_rch_eps -- Updated segment endpoint locations for the final
            centerline segments.
    """

    # Loop through segments and use manual edit flag to determine if a
    # segment has been altered. If a segment has been altered, update the
    # indexes and endpoint locations.
    uniq_rch = np.unique(grwl.finalID)
    new_rch_ind = np.zeros(len(grwl.segInd))
    new_rch_eps = np.zeros(len(grwl.segInd))
    for ind in list(range(len(uniq_rch))):
        # Current segment locations and coordinates.
        rch = np.where(grwl.finalID == uniq_rch[ind])[0]
        rch_x = grwl.x[rch]
        rch_y = grwl.y[rch]
        rch_pts = np.vstack((rch_x, rch_y)).T
        # Determining whether segments that have been edited were combined
        # with and existing GRWL segment.
        rch_segs = grwl.manual[rch]
        segs = np.unique(grwl.manual[rch])
        # Creating empty arrays.
        new_ind = np.zeros(len(rch))
        eps = np.zeros(len(rch))
        # condition that specifies the type of combined segments.
        # 0 = edited and non-edited; 1 = multiple edited segments.
        cond = 0

        # If length of segment is only one point.
        if len(rch) == 1:
            new_rch_ind[rch] = 1
            new_rch_eps[rch] = 1
            continue

        # If there are combined segments find endpoints of all original
        # segments.
        if len(segs) > 1 or np.min(rch_segs) == 1:

            # Combined edited segments.
            if np.min(rch_segs) == 1:
                sub_rch = np.where(edits.finalID == np.unique(grwl.finalID[rch]))[0]
                if len(np.unique(edits.seg[sub_rch])) > 1:
                    segs = np.unique(edits.seg[sub_rch])
                    cond = 1
                else:
                    new_rch_ind[rch] = grwl.segInd[rch]
                    ep1 = np.where(grwl.segInd[rch] == np.min(grwl.segInd[rch]))[0]
                    ep2 = np.where(grwl.segInd[rch] == np.max(grwl.segInd[rch]))[0]
                    new_rch_eps[rch[ep1]] = 1
                    new_rch_eps[rch[ep2]] = 1
                    continue

            # Combined edited and extisting GRWL segments.
            if cond == 0:
                for idx in list(range(len(segs))):
                    s = np.where(grwl.manual[rch] == segs[idx])[0]
                    # Condition for multiple combined edited segments.
                    if segs[idx] == 1:
                        sub_segs = np.unique(edits.seg[np.where(edits.finalID == np.unique(grwl.finalID[rch[s]]))[0]])
                        if len(sub_segs) > 1:
                            sub_rch = np.where(edits.finalID == np.unique(grwl.finalID[rch[s]]))[0]
                            for idy in list(range(len(sub_segs))):
                                s2 = np.where(edits.seg[sub_rch] == sub_segs[idy])[0]
                                mn = np.where(edits.ind[sub_rch[s2]] == np.min(edits.ind[sub_rch[s2]]))[0]
                                mx = np.where(edits.ind[sub_rch[s2]] == np.max(edits.ind[sub_rch[s2]]))[0]
                                eps[s[s2[mn]]] = 1
                                eps[s[s2[mx]]] = 1
                    # Condition for original GRWL segments.
                    else:
                        mn = np.where(grwl.segInd[rch[s]] == np.min(grwl.segInd[rch[s]]))[0]
                        mx = np.where(grwl.segInd[rch[s]] == np.max(grwl.segInd[rch[s]]))[0]
                        eps[s[mn]] = 1
                        eps[s[mx]] = 1

            # Multiple edited segments.
            else:
                for idx in list(range(len(segs))):
                    s = np.where(edits.seg[sub_rch] == segs[idx])[0]
                    mn = np.where(edits.ind[sub_rch[s]] == np.min(edits.ind[sub_rch[s]]))[0]
                    mx = np.where(edits.ind[sub_rch[s]] == np.max(edits.ind[sub_rch[s]]))[0]
                    eps[s[mn]] = 1
                    eps[s[mx]] = 1

            # Once endpoints are identified for all combined segments, perform
            # a spatial query between the endpoint locations and determine the
            # offical endpoints by the maximum distance.
            eps_ind = np.where(eps>0)[0]
            ep_pts = np.vstack((rch_x[eps_ind], rch_y[eps_ind])).T
            kdt = sp.cKDTree(rch_pts)
            if len(rch) <= 4:
                pt_dist, pt_ind = kdt.query(ep_pts, k = len(rch))
            else:
                pt_dist, pt_ind = kdt.query(ep_pts, k = 5)

            real_eps = []
            for idz in list(range(len(eps_ind))):
                neighbors = len(np.unique(rch_segs[pt_ind[idz,:]]))
                if neighbors == 1:
                    real_eps.append(eps_ind[idz])
            real_eps = np.array(real_eps)

            if len(real_eps) == 0:
                real_eps = eps_ind

            if len(real_eps) == 1 or len(real_eps) == 2:
                final_eps = real_eps

            else:
                kdt2 = sp.cKDTree(ep_pts)
                pt_dist2, pt_ind2 = kdt2.query(ep_pts, k = len(eps_ind))
                real_eps2 = np.unique(np.where(pt_dist2 == np.max(pt_dist2))[0])
                final_eps = eps_ind[real_eps2]

            # Order segment points based on new endpoint locations.
            new_ind[final_eps[0]]=1
            idz = final_eps[0]
            count = 2
            while np.min(new_ind) == 0:
                d = np.sqrt((rch_x[idz]-rch_x)**2 + (rch_y[idz]-rch_y)**2)
                dzero = np.where(new_ind == 0)[0]
                next_pt = dzero[np.where(d[dzero] == np.min(d[dzero]))[0]][0]
                new_ind[next_pt] = count
                count = count+1
                idz = next_pt

            # Save new values.
            new_rch_ind[rch] = new_ind
            ep1 = np.where(new_ind == np.min(new_ind))[0]
            ep2 = np.where(new_ind == np.max(new_ind))[0]
            new_rch_eps[rch[ep1]] = 1
            new_rch_eps[rch[ep2]] = 1

        # If segment has not been combined keep current index and endpoint
        # locations.
        else:
            new_rch_ind[rch] = grwl.segInd[rch]
            ep1 = np.where(grwl.segInd[rch] == np.min(grwl.segInd[rch]))[0]
            ep2 = np.where(grwl.segInd[rch] == np.max(grwl.segInd[rch]))[0]
            new_rch_eps[rch[ep1]] = 1
            new_rch_eps[rch[ep2]] = 1

    return new_rch_ind, new_rch_eps

###############################################################################

def save_grwl_edits(grwl, outfile):

    """
    FUNCTION:
        Writes updated GRWL shapefile.

    INPUTS
        grwl -- Object containing updated attributes for the GRWL centerline.
        outfile -- Outpath directory to write the shapefile.

    OUTPUTS
        Updated shapefile written to the specified outpath.
    """

    driver = ogr.GetDriverByName('ESRI Shapefile')

    fshpout = outfile
    if os.path.exists(fshpout):
        driver.DeleteDataSource(fshpout)
    try:
        dataout = driver.CreateDataSource(fshpout)
    except:
        print('Could not create file ' + fshpout)
        sys.exit(1)

    # Set spatial projection.
    proj = osr.SpatialReference()
    wkt_text = 'GEOGCS["WGS 84",DATUM["WGS_1984", \
                SPHEROID["WGS 84",6378137,298.257223563, AUTHORITY["EPSG",7030]],\
                TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG",6326]], PRIMEM["Greenwich",\
                0,AUTHORITY["EPSG",8901]],UNIT["DMSH",0.0174532925199433, \
                AUTHORITY["EPSG",9108]],AXIS["Lat",NORTH],AXIS["Long",EAST],\
                AUTHORITY["EPSG",4326]]'

    proj.ImportFromWkt(wkt_text)

    layerout = dataout.CreateLayer(outfile[:-4] + '_lay.shp',
                                   proj, geom_type=ogr.wkbPoint)

    # Define attributes.
    fieldDef1 = ogr.FieldDefn('lon', ogr.OFTReal)
    fieldDef2 = ogr.FieldDefn('lat', ogr.OFTReal)
    fieldDef3 = ogr.FieldDefn('utm_east', ogr.OFTReal)
    fieldDef4 = ogr.FieldDefn('utm_north', ogr.OFTReal)
    fieldDef5 = ogr.FieldDefn('segmentInd', ogr.OFTInteger)
    fieldDef6 = ogr.FieldDefn('segmentID', ogr.OFTInteger)
    fieldDef7 = ogr.FieldDefn('width_m', ogr.OFTReal)
    fieldDef8 = ogr.FieldDefn('lakeFlag', ogr.OFTInteger)
    fieldDef9 = ogr.FieldDefn('nchannels', ogr.OFTInteger)
    fieldDef10 = ogr.FieldDefn('manual_add', ogr.OFTInteger)
    fieldDef11 = ogr.FieldDefn('endpoints', ogr.OFTInteger)
    layerout.CreateField(fieldDef1)
    layerout.CreateField(fieldDef2)
    layerout.CreateField(fieldDef3)
    layerout.CreateField(fieldDef4)
    layerout.CreateField(fieldDef5)
    layerout.CreateField(fieldDef6)
    layerout.CreateField(fieldDef7)
    layerout.CreateField(fieldDef8)
    layerout.CreateField(fieldDef9)
    layerout.CreateField(fieldDef10)
    layerout.CreateField(fieldDef11)

    # Create feature (point) to store pixel
    floutDefn = layerout.GetLayerDefn()
    feature_out = ogr.Feature(floutDefn)

    for ipix in range(len(grwl.lat)):
        # Create Geometry Point with pixel coordinates
        pixel_point = ogr.Geometry(ogr.wkbPoint)
        pixel_point.AddPoint(grwl.lon[ipix], grwl.lat[ipix])
        # Add the geometry to the feature
        feature_out.SetGeometry(pixel_point)
        # Set feature attributes
        feature_out.SetField('lon', grwl.lon[ipix])
        feature_out.SetField('lat', grwl.lat[ipix])
        feature_out.SetField('utm_east', grwl.x[ipix])
        feature_out.SetField('utm_north', grwl.y[ipix])
        # int() is needed because facc.dtype=float64, needs to be saved with
        # all values whereas lat.dtype=float64, which raise an error.
        feature_out.SetField('segmentInd', int(grwl.finalInd[ipix]))
        feature_out.SetField('segmentID', int(grwl.finalID[ipix]))
        feature_out.SetField('width_m', int(grwl.wth[ipix]))
        feature_out.SetField('lakeFlag', int(grwl.lake[ipix]))
        feature_out.SetField('nchannels', int(grwl.nchan[ipix]))
        feature_out.SetField('manual_add', int(grwl.manual[ipix]))
        feature_out.SetField('endpoints', int(grwl.finaleps[ipix]))

        # Add the feature to the layer
        layerout.CreateFeature(feature_out)
        # Delete point geometry
        pixel_point.Destroy()

    # Close feature and shapefiles
    feature_out.Destroy()
    dataout.Destroy()

    return fshpout

###############################################################################
################################# Main Code ###################################
###############################################################################

'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!!! IMPORTANT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Note for "use_updated_grwl" flag (line 1130)

Condition to use either the original GRWL files from (Allen and Pavelsky, 2018)
or the currently updated files used in SWORD. To use the original files, 
set use_updated_grwl = False and assign the appropriate grwl_dir to original 
files. To use/edit the currently updated GRWL files set use_updated_grwl = True 
and set the appropriate grwl_dir to the updated files. 

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Region", type = str)
args = parser.parse_args()

# True = Use updated GRWL files; False = Use original GRWL files.
use_updated_grwl = True
use_edit_updated = True
 
region = args.region
grwl_dir = main_dir+'/data/outputs/Merged_Data/v11/tiles/' + region + '/'
edits_dir = main_dir+'/data/inputs/GRWL/EDITS/'
outdir = main_dir+'/data/inputs/GRWL/GRWL_Updates/' + region + '/'
grwl_paths = np.sort([file for file in getListOfFiles(grwl_dir) if '.shp' in file])
edit_paths = np.sort([file for file in getListOfFiles(edits_dir) if '.gpkg' in file])

start_all = time.time()
for ind in list(range(len(grwl_paths))):

    start = time.time()

    '''
    Identify files.
    '''
    pattern = grwl_paths[ind][-17:-10] #may need to be edited based on grwl path length. pattern ex: "n60w150"
    edit_files = [file for file in edit_paths if pattern in file]
    
    #Can choose a new path or overwrite existing files.
    if os.path.exists(outdir): 
        outpath = outdir + pattern + '_update.shp'
    else:
        os.makedirs(outdir)
        outpath = outdir + pattern + '_update.shp'
           
    '''
    Read in and format original GRWL data.
    '''  
    #Read in grwl files.
    if use_updated_grwl == True:
        grwl = read_grwl_merge(grwl_paths[ind])
    else:
        grwl = read_grwl(grwl_paths[ind])
        
    if len(grwl.lon) == 0:
        print(ind, pattern, 'NO GRWL DATA')
        continue

    #Adding Manual Flag.
    #If starting from original files, you need add manual add flag.
    if use_updated_grwl == False:
        grwl.manual = np.zeros(len(grwl.lon))

    #Correcting original segment errors.
    if use_updated_grwl == True:
        grwl.newID = np.copy(grwl.segID)
        grwl.eps = np.copy(grwl.eps)
    else:
        grwl.newID = update_segID(grwl)
        grwl.eps = label_grwl_endpoints(grwl)

    #Finding Tile UTM Zone Info.
    utm_zone = utm.from_latlon(grwl.lat[0], grwl.lon[0])[2]
    utm_let = utm.from_latlon(grwl.lat[0], grwl.lon[0])[3]

    #If there are not GRWL edits, and using original GRWL files, smooth the 
    #centerlines and save the file with updated attributes. If using already
    #updated GRWL files, copy or overwrite the file. 
    if len(edit_files) == 0:
        if use_updated_grwl == False:    
            # Copying exisiting attributes to names used to save file. 
            grwl.finaleps = np.copy(grwl.eps)
            grwl.finalInd = np.copy(grwl.segInd)
            grwl.finalID = np.copy(grwl.newID)
                
            # Finding UTM zone of current GRWL tile.
            __,__,zone,let = find_projection(grwl.lat, grwl.lon)
            zone = int(max(set(list(zone)), key=list(zone).count))
            let = max(set(list(let)), key=list(let).count)
            # Smoothing GRWL centerline locations.
            grwl.x, grwl.y = smooth_grwl(grwl)
            # Correcting odd values at equator boundary.
            grwl.y[np.where(grwl.y>10000000)[0]] = 10000000
            grwl.y[np.where(grwl.y<0)[0]] = 0
            # Creating new lat-lon values from smoothed utm coordinates.
            latlon = []
            for idz in list(range(len(grwl.x))):
                latlon.append(utm.to_latlon(grwl.x[idz], grwl.y[idz], zone, let))
            # Formatting lat-lon data.
            latlon = np.array(latlon)
            grwl.lat = latlon[:,0]
            grwl.lon = latlon[:,1]
                
            # Saving data
            save_grwl_edits(grwl, outpath)
            end = time.time()
            print(ind, pattern + ': No Edits - file written in: ' \
                  + str(np.round((end-start), 2)) + ' sec')
        if use_updated_grwl == True:
            # Copying exisiting attributes to names used to save file. 
            grwl.finaleps = np.copy(grwl.eps)
            grwl.finalInd = np.copy(grwl.segInd)
            grwl.finalID = np.copy(grwl.newID)
                
            # Saving data
            save_grwl_edits(grwl, outpath)
            end = time.time()
            print(ind, pattern + ': No Edits - file written in: ' \
                  + str(np.round((end-start), 2)) + ' sec')
        continue
 
    '''
    Read in and format GRWL edits.
    '''
    max_seg = np.max(grwl.newID)+1
    edits = read_edits_gpkg(edit_files[0], max_seg)   
    
    #formating coordinates. If the edits are in lat lon then convert to utm, 
    #or vice versa. 
    if use_edit_updated == False:
        if -90 < np.mean(edits.y) < 90:
            edits.lon = np.copy(edits.x)
            edits.lat = np.copy(edits.y)
            edits.x,edits.y,__,__ = find_projection(edits.lat, edits.lon)
        
        else:
            edits.lat = np.zeros(len(edits.finalID))
            edits.lon = np.zeros(len(edits.finalID))
            for idx in list(range(len(edits.finalID))):
                edits.lat[idx], edits.lon[idx] = utm.to_latlon(edits.x[idx], edits.y[idx], \
                                                                utm_zone, utm_let)

    edits.eps = find_edit_endpoints(edits)
    edits.ind = order_edits(edits)
    edits.wth = np.repeat(1,len(edits.seg))

    #Cutting grwl segments at new tributaries.
    grwl.tribs = find_tributary_junctions(grwl, edits)
    start_seg = np.max([np.max(grwl.newID), np.max(edits.seg)])+1
    grwl.finalID = cut_segments(grwl, start_seg)

    # edits.tribs = find_tributary_junctions_edits(grwl, edits)

    #Combining small edits with GRWL segments.
    edits.finalID = edit_short_segments(grwl, edits)

    #Creating filler values in edited data for other GRWL variables.
    edits.wth = np.array(np.repeat(1, len(edits.finalID)))
    edits.manual = np.array(np.repeat(1, len(edits.finalID)))
    edits.nchan = np.array(np.repeat(0, len(edits.finalID)))

    #Correcting odd values at equator boundary.
    if region == 'SA' or region == 'AF' or region == 'AS':
        edits.y[np.where(edits.y>10000000)[0]] = 10000000
        edits.y[np.where(edits.y<0)[0]] = 0

    #Add edits to existing grwl.
    grwl.x = np.insert(grwl.x, len(grwl.x), edits.x)
    grwl.y = np.insert(grwl.y, len(grwl.y), edits.y)
    grwl.wth = np.insert(grwl.wth, len(grwl.wth), edits.wth)
    grwl.nchan = np.insert(grwl.nchan, len(grwl.nchan), edits.nchan)
    grwl.finalID = np.insert(grwl.finalID, len(grwl.finalID), edits.finalID)
    grwl.segInd = np.insert(grwl.segInd, len(grwl.segInd), edits.ind)
    grwl.lake = np.insert(grwl.lake, len(grwl.lake), edits.lake)
    grwl.lon = np.insert(grwl.lon, len(grwl.lon), edits.lon)
    grwl.lat = np.insert(grwl.lat, len(grwl.lat), edits.lat)
    grwl.manual = np.insert(grwl.manual, len(grwl.manual), edits.manual)

    #Updating new segment indexes and endpoints.
    grwl.finalInd, grwl.finaleps = update_indexes(grwl, edits)

    #If starting from the orginial GRWL files from Allen and Pavelsky (2018)
    #the centerline locations need to be smoothed in order to reduce the 
    #"stair-step" effects resulting from the 30 m Landsat imagery.
    if use_updated_grwl == False:
        # Finding UTM zone of current GRWL tile.
        __,__,zone,let = find_projection(grwl.lat, grwl.lon)
        zone = int(max(set(list(zone)), key=list(zone).count))
        let = max(set(list(let)), key=list(let).count)
        # Smoothing GRWL centerline locations.
        grwl.x, grwl.y = smooth_grwl(grwl)
        # Correcting odd values at equator boundary.
        grwl.y[np.where(grwl.y>10000000)[0]] = 10000000
        grwl.y[np.where(grwl.y<0)[0]] = 0
        # Creating new lat-lon values from smoothed utm coordinates.
        latlon = []
        for idz in list(range(len(grwl.x))):
            latlon.append(utm.to_latlon(grwl.x[idz], grwl.y[idz], zone, let))
        # Formatting lat-lon data.
        latlon = np.array(latlon)
        grwl.lat = latlon[:,0]
        grwl.lon = latlon[:,1]
    
    """
    WRITING DATA
    """
    save_grwl_edits(grwl, outpath)
    end = time.time()
    print(ind, pattern + ': Edits Combined in: ' \
        + str(np.round((end-start), 2)) + ' sec (' \
            'tribs = '+str(len(np.where(grwl.tribs > 0)[0]))+')')

end_all = time.time()
print('All GRWL Edits Combined in: ' \
      + str(np.round((end_all-start_all)/60, 2)) + ' min')
