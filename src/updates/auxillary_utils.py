# -*- coding: utf-8 -*-
"""
Auxillary Data Utilities (auxillary_utils.py)
================================================

Utilities for attaching required auxillary dataset
attributes, needed to create the SWOT River Database
(SWORD), to provided river centerline locations. 

Auxiallry datasets include:
    - MERIT Hydro
    - Global Obstruction Database (GROD)
    - Global Reservoir and Dam Database (GRanD)
    - HydroFALLS
    - SWOT Orbit Passes
    - HydroBASINS
    - SWOT Prior Lake Database (PLD)

"""

from __future__ import division
import os
main_dir = os.getcwd()
import numpy as np
from scipy import spatial as sp
import geopandas as gp
import netCDF4 as nc
from shapely.geometry import Point
import pandas as pd
from osgeo import ogr, gdal
from statistics import mode
import src.updates.geo_utils as geo 
import warnings
warnings.filterwarnings("ignore") #if code stops working may need to comment out to check warnings. 

###############################################################################

def mh_vals(elv_fn, wth_fn, facc_fn):

    """
    Reads in and formats MERIT Hydro raster values as arrays.
    Only returns values associated with flow accumulation 
    values of 10 sq.km. or greater.

    Parameters
    ----------
        elv_fn: str
            MERIT Hydro elevation raster file.
        wth_fn: str
            MERIT Hydro width raster file.
        facc_fn: str
            MERIT Hydro flow accumulation raster file.

    Returns
    -------
        lon: numpy.array()
            Longitude (WGS 84, EPSG:4326).
        lat: numpy.array()
            Latitude (WGS 84, EPSG:4326).
        elv: numpy.array()
            Elevation (m).
        wth: numpy.array()
            Width (m).
        facc: numpy.array()
            Flow accumulation (sq.km)

    """

    # Mosaicking all MERIT Hydro rasters that overlap the GRWL shapefile.
    
    elv_raster = gdal.Open(elv_fn)
    wth_raster = gdal.Open(wth_fn)
    facc_raster = gdal.Open(facc_fn)

    # Pulls and flattens raster vlues.
    xul = elv_raster.GetGeoTransform()[0]
    xres = elv_raster.GetGeoTransform()[1]
    yul = elv_raster.GetGeoTransform()[3]
    yres = elv_raster.GetGeoTransform()[5]
    
    lon=np.array([xul+xres*c+xres/2. for r in range(elv_raster.RasterXSize) for c in range(elv_raster.RasterYSize)])
    lat=np.array([yul+yres*r+yres/2. for r in range(elv_raster.RasterXSize) for c in range(elv_raster.RasterYSize)])
    elv = np.array(elv_raster.GetRasterBand(1).ReadAsArray()).flatten()
    wth = np.array(wth_raster.GetRasterBand(1).ReadAsArray()).flatten()
    facc = np.array(facc_raster.GetRasterBand(1).ReadAsArray()).flatten()

    keep = np.where(facc >= 10)[0]
    elv = elv[keep]
    wth = wth[keep]
    facc = facc[keep]
    lon = lon[keep]
    lat = lat[keep]

    return lon, lat, elv, wth, facc

###############################################################################

def attach_mh(elv_paths, wth_paths, facc_paths, 
              overlap_paths, data_lon, data_lat):
    """
    Performs a spatial query between MERIT Hydro attributes and 
    provided latitude and longitude locations. Returns arrays of 
    associated MERIT Hydro attributes to provided coordinates.  
    
    Parameters
    ----------
        elv_paths: list
            List of MERIT Hydro elevation raster files.
        wth_paths: list
            List of MERIT Hydro width raster files.
        facc_paths: list
            List of MERIT Hydro flow accumulation raster files.
        overlap_paths: list
            List of MERIT Hydro rasters that overlap the provided 
            latitude/longitude extent. 
        data_lon: numpy.array()
            Longitude (WGS 84, EPSG:4326).
        data_lat: numpy.array()
            Latitude (WGS 84, EPSG:4326).

    Returns
    -------
        merge_elv: numpy.array()
            Elevation (m) associated with provided latitude/longitude 
            coordinates.
        merge_wth: numpy.array()
            Width (m) associated with provided latitude/longitude 
            coordinates.
        merge_facc: numpy.array()
            Flow accumulation (sq.km) associated with provided 
            latitude/longitude coordinates.
        merge_tile: numpy_array()
            MERIT Hydro tile identifier from which the attributes were 
            derived from. 

    """

    merge_elv = np.repeat(0, len(data_lon))
    merge_wth = np.repeat(0, len(data_lon))
    merge_facc = np.repeat(0, len(data_lon))
    merge_tile = np.repeat('NaNtile', len(data_lon))

    data_pts = np.array([(data_lon[i], data_lat[i]) for i in range(len(data_lon))])
    for ind in list(range(len(overlap_paths))):
        # print(ind, len(overlap_paths))
        ind2 = np.where(elv_paths == overlap_paths[ind])[0][0]
        tile = elv_paths[ind2][-15:-8]
        mh_lon, \
            mh_lat, \
                mh_elv, \
                    mh_wth, \
                        mh_facc = mh_vals(elv_paths[ind2],
                                          wth_paths[ind2], 
                                          facc_paths[ind2])
        
        if len(mh_lon) == 0:
            continue

        else:
            # create bounding box to clip mhv by.
            xmin = np.min(mh_lon)
            xmax = np.max(mh_lon)
            ymin = np.min(mh_lat)
            ymax = np.max(mh_lat)
            ll = np.array([xmin, ymin])  # lower-left
            ur = np.array([xmax, ymax])  # upper-right
            
            data_idx = np.all(np.logical_and(ll <= data_pts, data_pts <= ur), axis=1)
            if len(data_idx) == 0:
                continue
            else:
                data_lon_crop = data_lon[data_idx]
                data_lat_crop = data_lat[data_idx]

                mh_pts = np.vstack((mh_lon, mh_lat)).T
                data_pts_crop = np.vstack((data_lon_crop,data_lat_crop)).T
                kdt = sp.cKDTree(mh_pts)
                pt_dist, pt_ind = kdt.query(data_pts_crop, k = 5)

                elv = mh_elv[pt_ind[:,0]]
                wth = mh_wth[pt_ind[:,0]]
                facc = mh_facc[pt_ind[:,0]]

                merge_elv[data_idx] = elv
                merge_wth[data_idx] = wth
                merge_facc[data_idx] = facc
                merge_tile[data_idx] = tile

    return merge_elv, merge_wth, merge_facc, merge_tile

###############################################################################

def open_grand_shpfile(filename):
    """
    Opens the Global Reservoir and Dam Database (GRanD) 
    file and formats attributes as arrays.

    Parameters
    ----------
        filename: str
            GRanD filepath.
       
    Returns
    -------
        lon: numpy.array()
            Longitude (WGS 84, EPSG:4326).
        lat: numpy.array()
            Latitude (WGS 84, EPSG:4326).
        grand_id: numpy.array()
            Attribute ID.
        catch_skm: numpy.array()
            Upstream catchment drainage area (sq.km).

    """

    driver = ogr.GetDriverByName('ESRI Shapefile')

    shape = driver.Open(filename)
    layer = shape.GetLayer()
    numfeat = layer.GetFeatureCount()

    lon = np.zeros(numfeat)
    lat = np.zeros(numfeat)
    grand_id = np.zeros(numfeat)
    catch_skm = np.zeros(numfeat)

    cnt = 0
    for feature in layer:
        geom = feature.GetGeometryRef()
        lon[cnt] = geom.GetX()
        lat[cnt] = geom.GetY()
        grand_id[cnt] = feature.GetField('GRAND_ID')
        catch_skm[cnt] = feature.GetField('CATCH_SKM')
        cnt += 1
        feature.Destroy()

    del driver, shape, numfeat, layer

    return lat, lon, grand_id, catch_skm

###############################################################################

def add_dams(lon, lat, fn_grand, fn_grod):
    """
    Performs a spatial query between the GRanD, GROD, and HydroFALLS
    datasets and provided latitude and longitude locations. Returns 
    arrays of associated river obstruction attributes to provided 
    coordinates.

    Parameters
    ----------
        lon: numpy.array()
            Longitude (WGS 84, EPSG:4326).
        lat: numpy.array()
            Latitude (WGS 84, EPSG:4326).
        fn_grand: str
            GRanD filepath.
        fn_grod: str
            GROD filepath. Also, contains HydroFALLS data. 
       
    Returns
    -------
        grand_ID: numpy.array()
            Binary flag indicating if a GRanD feature is associated 
            with a provided latitude/longitude coordinate. 
        grod_ID: numpy.array()
            GROD obstruction type associated with a provided 
            latitude/longitude coordinate.
        grod_FID: numpy.array()
            GROD ID associated with a provided latitude/longitude 
            coordinate.
        hfalls_FID: numpy.array()
            HydroFALLS ID associated with a provided latitude/longitude 
            coordinate.

    """

    # Read in GRaND and GROD data.
    grand_lat, grand_lon, grand_id, grand_skm = open_grand_shpfile(fn_grand)
    grand_lat = abs(grand_lat)
    grod_info = pd.read_csv(fn_grod)
    grod_lat =  np.array(grod_info.lat)
    grod_lon =  np.array(grod_info.lon)
    grod_names = np.array(grod_info.name) #.flatten() #grod_names = np.array(grod_info[[0]]).flatten()
    grod_id = np.zeros(len(grod_names))
    grod_fid = np.array(grod_info.grod_id)#.flatten()

    # Remove NaN values from arrays.
    remove = np.isnan(grod_lat)
    delete = np.where(remove == True)[0]
    grod_lat = np.delete(grod_lat, delete)
    grod_lon = np.delete(grod_lon, delete)
    grod_names = np.delete(grod_names, delete)
    grod_fid = np.delete(grod_fid, delete)
    grod_id = np.delete(grod_id, delete)

    # Assign numbers to GROD dam types.
    grod_id[np.where(grod_names == 'Dam')] = 1
    grod_id[np.where(grod_names == 'Locks')] = 2
    grod_id[np.where(grod_names == 'Low_Permeable_Dams')] = 3
    grod_id[np.where(grod_names == 'Waterfall')] = 4
    grod_id[np.where(grod_names == 'Partial_Dams_gte50')] = 5
    grod_id[np.where(grod_names == 'Partial_Dams_lt50')] = 6
    grod_id[np.where(grod_names == 'Channel_Dams')] = 7 #was 2 before excluding.

    # narrowing down points in GRWL bounding box.
    pts = np.array([grod_lon, grod_lat]).T
    ll = np.array([np.min(lon), np.min(lat)])  # lower-left
    ur = np.array([np.max(lon), np.max(lat)])  # upper-right

    inidx = np.all(np.logical_and(ll <= pts, pts <= ur), axis=1)
    inbox = pts[inidx]
    grod_id_clip = grod_id[inidx]
    grod_fid_clip = grod_fid[inidx]
    grod_lon_clip = inbox[:,0]
    grod_lat_clip = inbox[:,1]

    # Attach dam locations to GRWL.
    data_pts = np.vstack((lon, lat)).T
    grand_pts = np.vstack((grand_lon, grand_lat)).T
    kdt = sp.cKDTree(data_pts)
    grand_dist, grand_idx = kdt.query(grand_pts, k = 1)
    # Narrowing dam locations by distance threshold.
    grand_dist_thresh = np.where(grand_dist <= 0.01)[0]
    grand_locs, grand_locs_idx = np.unique(grand_idx[grand_dist_thresh], return_index=True)
    # Creating arrays.
    grand_ID = np.zeros(len(lon))
    # Assigning values.
    grand_ID[grand_locs] = grand_id[grand_dist_thresh[grand_locs_idx]]


    if len(grod_lon_clip) == 0:
        # Creating arrays.
        grod_ID = np.zeros(len(lon))
        grod_FID = np.zeros(len(lon))
        hfalls_FID = np.zeros(len(lon))
    else:
        # Attach dam locations to GRWL.
        grod_pts = np.vstack((grod_lon_clip, grod_lat_clip)).T
        kdt = sp.cKDTree(data_pts)
        grod_dist, grod_idx = kdt.query(grod_pts, k = 1)

        # Narrowing dam locations by distance threshold.
        grod_dist_thresh = np.where(grod_dist <= 0.02)[0] #was 100m
        grod_locs, grod_locs_idx = np.unique(grod_idx[grod_dist_thresh], return_index=True)

        # Creating arrays.
        grod_ID = np.zeros(len(lon))
        grod_FID = np.zeros(len(lon))
        hfalls_FID = np.zeros(len(lon))
        # Assigning values.
        grod_ID[grod_locs] = grod_id_clip[grod_dist_thresh[grod_locs_idx]]
        grod_FID[grod_locs] = grod_fid_clip[grod_dist_thresh[grod_locs_idx]]
        hfalls_FID[grod_locs] = grod_fid_clip[grod_dist_thresh[grod_locs_idx]]
        grod_FID[np.where(grod_ID == 4)] = 0
        hfalls_FID[np.where(grod_ID != 4)] = 0

    return grand_ID, grod_ID, grod_FID, hfalls_FID

###############################################################################

def add_swot_tracks(df, swot_files):
    """
    Spatially intersects SWOT orbit track polygons with provided 
    geodataframe values and returns arrays of overlapping SWOT orbit 
    pass identifiers and number of SWOT passes for each geodataframe 
    feature.

    Parameters
    ----------
        df: geopandas dataframe.
            Geodataframe of river features.
       
    Returns
    -------
        num_obs: numpy.array()
            Number of SWOT passes. 
        orbit_array: numpy.array() [200, number of features]
            SWOT orbit pass IDs. 

    """

    # Reading in GRWL point information.
    points = df

    # Creating empty arrays to fill.
    orbit_array = np.zeros((200, len(points)), dtype=int)

    row = 0
    for ind in list(range(len(swot_files))):
        #print ind
        poly = gp.GeoDataFrame.from_file(swot_files[ind])
        intersect = gp.sjoin(poly, points, how="inner")
        intersect = pd.DataFrame(intersect)
        intersect = intersect.drop_duplicates(subset='index_right', keep='first')
        ids = np.array(intersect.index_right)
        if len(ids) == 0:
            continue 
        else:
            orbit_array[row, ids] = intersect.ID_PASS
            row = row+1

    # Finding number of SWOT observations.
    orbit_binary = np.copy(orbit_array)
    orbit_binary[np.where(orbit_binary > 0)] = 1
    num_obs = np.sum(orbit_binary, axis = 0)

    # Assigning SWOT track information to new object attributes.
    return num_obs, orbit_array

###############################################################################

def filter_basin_codes(reaches, basins):
    """
    Filters HydroBASINS IDs within a river or delta reach.

    Parameters
    ----------
        reaches: numpy.array().
            Reach or segment IDs. 
        basins: numpy.array()
            HydroBASIN IDs. 
       
    Returns
    -------
        new_basins: numpy.array()
            Filtered basin codes.

    """

    new_basins = np.copy(basins)
    unq_rchs = np.unique(reaches)
    for r in list(range(len(unq_rchs))):
        rch = np.where(reaches == unq_rchs[r])[0]
        md = mode(basins[rch])
        new_basins[rch] = md

    return new_basins

###############################################################################

def calc_geodesic_dist(lon, lat, reaches, index):
    """
    Calculates geodesic distance along a river reach
    based on latitude/longitude coordinates.

    Parameters
    ----------
        lon: numpy.array()
            Longitude (WGS 84, EPSG:4326).
        lat: numpy.array()
            Latitude (WGS 84, EPSG:4326).
        reaches: numpy.array().
            Reach or segment IDs. 
        index: numpy.array()
            Unique coordinate IDs per reach. 
       
    Returns
    -------
        dist: numpy.array()
            Cumulative geodesic distance (m).
        length: numpy.array()
            Length (m). The maximum cumulative distance 
            value along a reach/segment. 

    """

    unq_rchs = np.unique(reaches)
    dist = np.zeros(len(reaches))
    length = np.zeros(len(reaches))
    for r in list(range(len(unq_rchs))):
        rch = np.where(reaches == unq_rchs[r])[0] #if multiple choose first.
        sort_ind = rch[np.argsort(index[rch])]   
        x_coords = lon[sort_ind]
        y_coords = lat[sort_ind]
        diff = geo.get_distances(x_coords,y_coords)
        dist[sort_ind] = np.cumsum(diff)
        length[sort_ind] = np.max(np.cumsum(diff))
    return dist, length

###############################################################################

def unique_cl_id(reaches, index):
    """
    Creates unique coordinate IDs for every geometric 
    point in the provided river database.

    Parameters
    ----------
        reaches: numpy.array().
            Reach or segment IDs. 
        index: numpy.array()
            Unique coordinate IDs per reach. 
       
    Returns
    -------
        cl_id: numpy.array()
            Unique coordinate IDs for every point.

    """

    unq_rchs = np.unique(reaches)
    cl_id = np.zeros(len(reaches), dtype=int)
    max_id = 0 #value to add to indexes in another reach. 
    for r in list(range(len(unq_rchs))):
        rch = np.where(reaches == unq_rchs[r])[0]
        cl_id[rch] = index[rch]+max_id
        max_id = np.max(cl_id)
    return cl_id

###############################################################################

def assign_endpoints(reaches, index):
    """
    Creates endpoint array identifying endpoint coordinates
    for each reach feature.

    Parameters
    ----------
        reaches: numpy.array().
            Reach or segment IDs. 
        index: numpy.array()
            Unique coordinate IDs per reach. 
       
    Returns
    -------
        endpnts: numpy.array()
            Array of values indicating if a coordiate is a 
            reach endpoint.
                0 - not an endpoint. 
                1 - downstream endpoint. 
                2 - upstream endpoint. 

    """
    unq_rchs = np.unique(reaches)
    endpts = np.zeros(len(reaches), dtype=int)
    for r in list(range(len(unq_rchs))):
        rch = np.where(reaches == unq_rchs[r])[0]
        mn = np.where(index[rch] == np.min(index[rch]))
        mx = np.where(index[rch] == np.max(index[rch]))
        endpts[rch[mn]] = 1
        endpts[rch[mx]] = 2
    return endpts

###############################################################################