# -*- coding: utf-8 -*-
"""
Geospatial Utilities (geo_utils.py)
=======================================

Utilities for geospatial calculations across a 
variety of data. 

"""

from __future__ import division
import os
import numpy as np
import netCDF4 as nc
import geopandas as gp
from osgeo import gdal, ogr
from shapely.geometry import LineString, Point
from geopandas import GeoSeries
import pandas as pd
import utm
from pyproj import Proj
from geopy import Point, distance
import math

###############################################################################

class Object(object):
    """
    Creates an empty class object to assign attributes to. 
    """
    pass 

###############################################################################

def getListOfFiles(dirName):

    """
    Gets a recursive list of all files in the directory tree for a
    given path.

    Parameters
    ----------
    dirName: str
        Input directory.

    Returns
    -------
    allFiles: list
        List of files under directory.

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

def meters_to_degrees(meters, latitude):
    """
    Converts a specified distance value in meters to degrees
    based on latitude. 

    Parmeters
    ---------
    meters: int
        Meters. 
    latitude: float
        Average latitude value of data. 

    Returns
    -------
    deg: float
        Decimal degree value equivalent to the input meter
        value. 
    
    """
    
    deg = np.round(meters/(111.32 * 1000 * math.cos(latitude * (math.pi / 180))),5)
    return deg

###############################################################################

def lines_to_pts(db):
    """
    Function takes a geodataframe of polyline features and returns
    an object containing the geometric point locations and 
    associated attributes. 

    Parmeters
    ---------
    fn: geopandas dataframe
        Geodataframe of the polyline features.

    Returns
    -------
    points: obj 
        Object containing the polyline geometry and associated
        attributes as point coordinates.
    
    """

    db['points'] = db.apply(lambda x: [y for y in x['geometry'].coords], axis=1)
    attr = db.columns
    #initialize empty points object.
    points = Object()
    for ind in list(range(len(db))):
        #get line geometry as points. 
        x = np.array([line[0] for line in db['points'][ind]])
        y = np.array([line[1] for line in db['points'][ind]])
        #disregard connecting points. 
        # x=x[1:-1] #commented out becuase short connecting reaches were lost in some cases. 
        # y=y[1:-1]
        #create point index
        index = np.array(list(range(len(x))))+1
        #assign coordinates and index to points object. 
        if ind == 0:
            points.x = x
            points.y = y
            points.index = index
        else:
            points.x = np.append(points.x, x)
            points.y = np.append(points.y, y)
            points.index = np.append(points.index, index)

        #assign all other attributes to points object. 
        for a in list(range(len(attr))):
            if attr[a] == 'x' or attr[a] == 'y':
                continue
            name  = attr[a]
            value = np.repeat(db[attr[a]][ind], len(x))
            #adding attribute to points Object with same column name as geodataframe. 
            if ind == 0:
                setattr(points, name, value) #equivalent to: self.varname= 'something'
            else:
                temp_arr = np.array(getattr(points, name))
                all_vals = np.append(temp_arr, value)
                setattr(points, name, all_vals)

    return points      

###############################################################################

def vector_to_vector_intersect(df1, df2, attribute):
    """
    Performs a spatial intersection between two vector layers.
    Intersection is a direct overlap. 

    Parameters
    ----------
    df1: geopandas.dataframe
        Geodataframe of the vector data to attach the intersected 
        information to.
    df2: geopandas.dataframe
        Geodataframe of the vector data to intersect and extract
        information from.
    attribute: str
        Attribute name of the field to intersect.

    Returns
    -------
    attr: numpy.array()
        Numpy array of the attribute for the spatial intersection.
        
    """

    #getting spatial intersection ids. 
    intersect = gp.sjoin(df2, df1, how="inner")
    intersect = pd.DataFrame(intersect)
    intersect = intersect.drop_duplicates(subset='index_right', keep='first')

    #creating an array output for intersected attribute values.
    ids = np.array(intersect.index_right)
    attr = np.zeros(len(df1))
    if hasattr(intersect, attribute):
        attr[ids] = np.array(getattr(intersect, attribute))

    return attr

###############################################################################

def vector_to_vector_join_nearest(df1, df2, one2one, attribute):
    """
    Performs a nearest neighbor spatial join between two vector layers.

    Parameters
    ----------
    df1: geopandas.dataframe
        Geodataframe of the vector data to attach the queried 
        information to.
    df2: geopandas.dataframe
        Geodataframe of the vector data to query and extract
        information from.
    one2one: True / False
        Argument to indicate if a one-to-one match is desired. 
        Default is one-to-many. 
    attribute: str
        Attribute name of the field to intersect.

    Returns
    -------
    attr: numpy.array()
        Numpy array of the attribute for the spatial join.
        
    """
    #ensure a index column exists. 
    df1 = df1.reset_index(drop=False)
    
    #performing spatial join based on nearest feature. 
    intersect = gp.sjoin_nearest(df1, df2, how="left", distance_col='distance', max_distance=None)

    if one2one == True:
        # Drop duplicates to ensure 1:1 (if needed, e.g. due to equidistant points)
        intersect = intersect.sort_values(by='index').drop_duplicates(subset='index')

    #creating an array output for intersected attribute values.
    if hasattr(intersect, attribute):
        attr = np.array(getattr(intersect, attribute))

    return attr

###############################################################################

def pts_to_geodf(lon, lat):
    """
    Converts arrays or list of latitude and longitude values 
    into a geodataframe. 

    Parameters
    ----------
    lon: numpy.array() or list
        Longitude (WGS 84, EPSG:4326).
    lat: geopandas.dataframe
        Latitude (WGS 84, EPSG:4326).

    Returns
    -------
    df: geopandas dataframe
        Geodataframe of point features.
        
    """

    df = pd.DataFrame([lon, lat]).T
    gdf = gp.GeoDataFrame(df, 
                         geometry=gp.points_from_xy(lon, lat), 
                         crs="EPSG:4326")
    gdf.rename(columns={0:"x",1:"y"},inplace=True)
    return gdf

###############################################################################

def get_distances(lon,lat):
    """
    Calculates geodesic distance along a set of points given input 
    latitude and longitude values. 

    Parameters
    ----------
    lon: numpy.array() 
        Longitude (WGS 84, EPSG:4326).
    lat: numpy.array()
        Latitude (WGS 84, EPSG:4326).

    Returns
    -------
    distances: numpy.array()
        Numpy array of cumulative distances in meters along the 
        input coordinates. 
    
    """
     
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
    Projects given latitude and longitude arrays into in UTM
    coordinates.

    Parameters
    ----------
    latitude: numpy.array() or list
        Latitude (WGS 84, EPSG:4326).
    longitude: numpy.array() or list
        Longitude (WGS 84, EPSG:4326).

    Returns
    -------
    east: numpy.array()
        Easting in UTM.
    north: numpyr.array()
        Northing in UTM.
    utm_num: str
        UTM zone number.
    utm_let: str
        UTM zone letter.

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
         zone_num[ind], zone_let_int) = utm.from_latlon(latitude[ind],longitude[ind])
        zone_let.append(zone_let_int)

    # Finds the unique UTM zones and converts the lat/lon pairs to UTM.
    unq_zones = np.unique(zone_num)
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

def pt_raster_overlap(lon, lat, raster_paths):
    """
    Finds overlapping raster files for given point 
    latitude and longitude values.

    Parameters
    ----------
    lon: numpy.array() 
        Longitude (WGS 84, EPSG:4326).
    lat: numpy.array()
        Latitude (WGS 84, EPSG:4326).
    raster_paths: list
        List of raster paths. 

    Returns
    -------
    overlap_files: list
        List of raster paths that overlap the extent
        defined by given latitude and longitude values.
        
    """

    #define grwl extent as ogr geometry format.
    poly1 = ogr.Geometry(ogr.wkbLinearRing)
    poly1.AddPoint(min(lon), max(lat))
    poly1.AddPoint(min(lon), min(lat))
    poly1.AddPoint(max(lon), min(lat))
    poly1.AddPoint(max(lon), max(lat))
    poly1.AddPoint(min(lon), max(lat))
    mhvGeometry = ogr.Geometry(ogr.wkbPolygon)
    mhvGeometry.AddGeometry(poly1)
    poly_box = mhvGeometry.GetEnvelope()        

    #find overlapping SWOT tracks.
    overlap_files = []
    for fn in raster_paths:
        # Read raster extent
        # Open the raster file
        try:
            raster_ds = gdal.Open(fn)
            raster_geotransform = raster_ds.GetGeoTransform()
            raster_extent = (
                raster_geotransform[0],
                raster_geotransform[0] + raster_geotransform[1] * raster_ds.RasterXSize,
                raster_geotransform[3] + raster_geotransform[5] * raster_ds.RasterYSize,
                raster_geotransform[3]
            )

            # Check for overlap
            overlap = (
                poly_box[0] < raster_extent[1] and
                poly_box[1] > raster_extent[0] and
                poly_box[2] < raster_extent[3] and
                poly_box[3] > raster_extent[2]
            )

            if overlap == True:
                overlap_files.append(fn)

        except:
            print('!!Read Error!!', fn)
            continue
        
    overlap_files = np.unique(overlap_files)

    return(overlap_files)

###############################################################################

def pt_vector_overlap(lon, lat, vector_paths):
    """
    Finds overlapping vector files for given point 
    latitude and longitude values.

    Parameters
    ----------
    lon: numpy.array() 
        Longitude (WGS 84, EPSG:4326).
    lat: numpy.array()
        Latitude (WGS 84, EPSG:4326).
    vector_paths: list
        List of vector paths. 

    Returns
    -------
    overlap_files: list
        List of vector paths that overlap the extent
        defined by given latitude and longitude values.
        
    """

    #define grwl extent as ogr geometry format.
    poly1 = ogr.Geometry(ogr.wkbLinearRing)
    poly1.AddPoint(min(lon), max(lat))
    poly1.AddPoint(min(lon), min(lat))
    poly1.AddPoint(max(lon), min(lat))
    poly1.AddPoint(max(lon), max(lat))
    poly1.AddPoint(min(lon), max(lat))
    mhvGeometry = ogr.Geometry(ogr.wkbPolygon)
    mhvGeometry.AddGeometry(poly1)

    #find overlapping SWOT tracks.
    overlap_files = []
    for fn in vector_paths:
        driver = ogr.GetDriverByName('ESRI Shapefile')
        shape = driver.Open(fn)
        inLayer = shape.GetLayer()
        #inLayer is always of size one because polygon is a unique value.
        for feature in inLayer:
            track=feature.GetGeometryRef()
            answer = track.Intersects(mhvGeometry)
            if answer == True:
                overlap_files.append(fn)
    overlap_files = np.unique(overlap_files)

    return(overlap_files)

###############################################################################

