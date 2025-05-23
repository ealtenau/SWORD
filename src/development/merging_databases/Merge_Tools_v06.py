# -*- coding: utf-8 -*-

"""
-------------------------------------------------------------------------------
                                Merging Tools
-------------------------------------------------------------------------------
Copyright (c) 2018-2021 UNC Chapel Hill. All rights reserved.
Created by E. Altenau. Some functions were taken or modified from C. Lion's
"Tools.py" script (c) 2016.

DESCRIPTON:
    This script contains functions to aid in merging attributes from multiple
    datasets. The various datasets merged are as follows:

    GRWL (Global River Widths from Landsat): Contains river centerline
        locations, width, number of channels, and water body type information
        for rivers 30 m wide and greater [Allen and Pavelsky, 2018].
    MERIT Hydro: Provides hyrologically-corrected elevation values and flow
        accumulation for rivers derived from the MERIT DEM
        [Yamazaki et al., 2019].
    HydroBASINS: Contains basin boundary locations and Pfafstetter basin codes
        for each continent [Lehner and Grill, 2013].
    GRanD (Global Reservior and Dam Database): Provides locations of the worlds
        largest dams [Lehner et al., 2011].
    GROD (Global River Obstruction Database): Provides locations and structure
        type for all dams along the GRWL river network [Yang et al., 2021].
    Global Deltas: Provides the spatial extent of 48 of the world’s largest
        deltas [Tessler et al., 2015].
    SWOT Tracks: Provides SWOT orbit track numbers and days of overpass.
    Prior Lake Database: Provides lake extents derived from Landsat.
-------------------------------------------------------------------------------
"""

from __future__ import division
import os
import utm
import sys
from osgeo import ogr
from osgeo import osr
# from pyproj import Proj
from pyproj import Proj, transform
import numpy as np
from osgeo import gdal
#import shapefile
import rasterio
from rasterio.merge import merge
from scipy import spatial as sp
import geopandas as gp
import pandas as pd
import time
import netCDF4 as nc

###############################################################################
######################## Reading and Writing Functions ########################
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
        allFiles -- list of files under directory
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

def open_grwl(filename):

    """
    FUNCTION:
        Opens Edited GRWL file and returns the fields inside a "grwl" object.
        Each field is stored inside the object in array format.

    INPUTS
        filename -- GRWL shapefile

    OUTPUTS
        grwl -- Object containing point locations and attributes along the
                grwl centerlines. See attribute descriptions below:
        grwl.lon -- Longitude (wgs84)
        grwl.lat -- Latitude (wgs84)
        grwl.wth -- Width (m)
        grwl.seg -- Segment ID
        grwl.nchan -- Number of Channels
        grwl.lake -- Lake flag based off of prior lake database
        grwl.x -- Easting (utm)
        grwl.y -- Northing (utm)
        grwl.manual -- Manual Edit
        grwl.eps -- Segment Endpoints
        grwl.Ind -- Segment Point Index
        grwl.old_lakes -- original grwl flag
        grwl.lake_id -- Prior lake database id
    """

    fn_grwl = filename
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shape = driver.Open(fn_grwl)
    layer = shape.GetLayer()
    numFeatures = layer.GetFeatureCount()

    # Identifying grwl shapefile attributes.
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
    grwl.seg = np.zeros(numFeatures)
    grwl.Ind = np.zeros(numFeatures)
    grwl.lake = np.zeros(numFeatures)
    grwl.lon = np.zeros(numFeatures)
    grwl.lat = np.zeros(numFeatures)
    grwl.manual = np.zeros(numFeatures)
    grwl.eps = np.zeros(numFeatures)
    #grwl.old_lakes = np.zeros(numFeatures)
    #grwl.lake_id = np.zeros(numFeatures)

    # Saving data to arrays.
    if 'manual_add' in attributes:

        cnt = 0
        for feature in range(numFeatures):
            grwl.x[cnt] = layer.GetFeature(feature).GetField('utm_east')
            grwl.y[cnt] = layer.GetFeature(feature).GetField('utm_north')
            grwl.wth[cnt] = layer.GetFeature(feature).GetField('width_m')
            grwl.nchan[cnt] = layer.GetFeature(feature).GetField('nchannels')
            grwl.seg[cnt] = layer.GetFeature(feature).GetField('segmentID')
            grwl.Ind[cnt] = layer.GetFeature(feature).GetField('segmentInd')
            grwl.lake[cnt] = layer.GetFeature(feature).GetField('lakeFlag')
            grwl.lon[cnt] = layer.GetFeature(feature).GetField('lon')
            grwl.lat[cnt] = layer.GetFeature(feature).GetField('lat')
            grwl.manual[cnt] = layer.GetFeature(feature).GetField('manual_add')
            grwl.eps[cnt] = layer.GetFeature(feature).GetField('endpoints')
            #grwl.old_lakes[cnt] = layer.GetFeature(feature).GetField('old_lakes')
            #grwl.lake_id[cnt] = layer.GetFeature(feature).GetField('lake_id')
            cnt += 1

    else:

      cnt = 0
      for feature in range(numFeatures):
          grwl.x[cnt] = layer.GetFeature(feature).GetField('utm_east')
          grwl.y[cnt] = layer.GetFeature(feature).GetField('utm_north')
          grwl.wth[cnt] = layer.GetFeature(feature).GetField('width_m')
          grwl.nchan[cnt] = layer.GetFeature(feature).GetField('nchannels')
          grwl.seg[cnt] = layer.GetFeature(feature).GetField('segmentID')
          grwl.Ind[cnt] = layer.GetFeature(feature).GetField('segmentInd')
          grwl.lake[cnt] = layer.GetFeature(feature).GetField('lakeFlag')
          grwl.lon[cnt] = layer.GetFeature(feature).GetField('lon')
          grwl.lat[cnt] = layer.GetFeature(feature).GetField('lat')
          #grwl.old_lakes[cnt] = layer.GetFeature(feature).GetField('old_lakes')
          #grwl.lake_id[cnt] = layer.GetFeature(feature).GetField('lake_id')
          grwl.manual[cnt] = 0
          grwl.eps[cnt] = 0
          cnt += 1

    return grwl

###############################################################################

def open_merge_shp(filename):

    """
    FUNCTION:
        Opens existing merged file and returns the fields inside a "merged" object.
        Each field is stored inside the object in array format. This function
        is used in the case the merged netcdf needs updating based on existing
        merged shapefiles.

    INPUTS
        filename -- Merged shapefile path

    OUTPUTS
        merge -- Object containing point locations and attributes along the
                merged database centerlines. See attribute descriptions below:
        merge.lon = Longitude based on shapefile geometry (wgs84)
        merge.lat = Latitude based on shapefile geometry (wgs84)
        merge.x = Easting (m)
        merge.y = Northing (m)
        merge.orbits = SWOT Orbits that intersect a point location
        merge.old_lon = Longitude based on shapefile field (wgs84)
        merge.old_lat = Latitude based on shapefile field (wgs84)
        merge.wth = Width (m)
        merge.nchan = Number of Channels
        merge.seg = Segment ID
        merge.Ind = Segment Point Index
        merge.dist = Segment Distance (m)
        merge.elv = Elevation (m)
        merge.facc_filt = Flow Accumulation (km^2)
        merge.new_lakes = Lake Flag based on Prior Lake Database
        merge.grand = GRaND ID
        merge.grod =  Binary flag indicating GROD locations
        merge.grod_fid = GROD ID
        merge.hfalls_fid = HydroFALLS ID
        merge.basins = HydroBASINS Code
        merge.delta = Binary flag indicating delta extents
        merge.manual = Binary flag indicating if a point was added to
                       original GRWL centerlines
        merge.eps = Endpoint locations for each segment
        merge.num_obs = Number of SWOT passes per 21-day orbit cycle
        merge.tile = GRWL tile name
        merge.old_lakes = Original GRWL Lake Flag
        merge.lake_id = Prior Lake Database ID
        merge.old_deltas = Unflitered delta extents
    """

    # opening file and assigning geometry location value to arrays.
    fn_merge = filename
    shp = gp.read_file(fn_merge)
    geom = [i for i in shp.geometry]
    lon = np.zeros(len(geom))
    lat = np.zeros(len(geom))
    orbits = np.zeros([len(geom),75], dtype = int)
    merge = Object()
    for ind in list(range(len(geom))):
        lon[ind] = np.array(geom[ind].coords.xy[0])
        lat[ind] = np.array(geom[ind].coords.xy[1])
        vals = np.array(shp['orbits'][ind][1:-1].split(), dtype = int)
        orbits[ind,0:len(vals)] = vals

    if len(lon) == 0:
        merge.lon = []

    # Assigning other shapefile attributes to object arrays.
    else:

        lon[np.where(lon < -180)] = -179.9
        lon[np.where(lon > 180)] = 179.9
        east, north, __, __ = find_projection(lat, lon)
        #north[np.where(north>10000000)[0]] = 10000000
        #north[np.where(north<0)[0]] = 0

        merge.lon = lon
        merge.lat = lat
        merge.x = east
        merge.y = north
        merge.orbits = orbits
        merge.old_lon = np.array(shp['x'])
        merge.old_lat = np.array(shp['y'])
        merge.wth = np.array(shp['p_width'])
        merge.nchan = np.array(shp['nchan'])
        merge.seg = np.array(shp['segID'])
        merge.Ind = np.array(shp['segInd'])
        merge.dist = np.array(shp['segDist'])
        merge.elv = np.array(shp['p_height'])
        merge.facc_filt = np.array(shp['flowacc'])
        merge.new_lakes = np.array(shp['lakeflag'])
        merge.grand = np.array(shp['grand_id'])
        merge.grod = np.array(shp['grod_id'])
        merge.grod_fid = np.array(shp['grod_fid'])
        merge.hfalls_fid = np.array(shp['hfalls_fid'])
        merge.basins = np.array(shp['basin_code'])
        merge.delta = np.array(shp['deltaflag'])
        merge.manual = np.array(shp['manual_add'])
        merge.eps = np.array(shp['endpoints'])
        merge.num_obs = np.array(shp['number_obs'])
        merge.tile = np.array(shp['grwl_tile'])
        merge.tile = np.array([np.array(str(merge.tile[0])[0:7]) for i in merge.tile])
        merge.old_lakes = np.array(shp['old_lakef'])
        merge.lake_id = np.array(shp['lake_id'])
        merge.old_deltas = np.array(shp['old_deltas'])

    return merge

###############################################################################
def open_grand_shpfile(filename):

    """
    FUNCTION:
        Opens GRanD shapefile

    INPUTS:
        filename -- GRanD shapefile

    OUTPUTS:
        lat -- Latitude
        lon -- Longitude
        grand_id -- GRanD ID
        catch_skm -- catchment area of dam (catch_skm from GRanD)
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

def save_merge_shp(grwl, outfile):

    """
    FUNCTION:
        Writes merged shapefiles. Datasets combined include: GRWL,
        MERIT Hydro, GROD, GRanD, HydroBASINS, Global Deltas, SWOT Track
        information, and Prior Lake Database locations.

    INPUTS
        grwl -- Object containing attributes for the GRWL centerline.
        outfile -- Outpath directory to write the shapefile.

    OUTPUTS
        Combined shapefile written to the specified outpath.
    """

    # Create and open shapefile.
    driver = ogr.GetDriverByName('ESRI Shapefile')

    fshpout = outfile
    if os.path.exists(fshpout):
        driver.DeleteDataSource(fshpout)
    try:
        dataout = driver.CreateDataSource(fshpout)
    except:
        print('Could not create file ' + fshpout)
        sys.exit(1)

    # Define spatial projection.
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

    # Define pixel attributes
    fieldDef1 = ogr.FieldDefn('x', ogr.OFTReal)
    fieldDef2 = ogr.FieldDefn('y', ogr.OFTReal)
    fieldDef3 = ogr.FieldDefn('easting', ogr.OFTReal)
    fieldDef4 = ogr.FieldDefn('northing', ogr.OFTReal)
    fieldDef5 = ogr.FieldDefn('segInd', ogr.OFTInteger)
    fieldDef6 = ogr.FieldDefn('segID', ogr.OFTInteger)
    fieldDef7 = ogr.FieldDefn('segDist', ogr.OFTReal)
    fieldDef8 = ogr.FieldDefn('p_height', ogr.OFTReal)
    fieldDef9 = ogr.FieldDefn('flowacc', ogr.OFTReal)
    fieldDef10 = ogr.FieldDefn('p_width', ogr.OFTReal)
    fieldDef11 = ogr.FieldDefn('nchan', ogr.OFTInteger)
    fieldDef12 = ogr.FieldDefn('lakeflag', ogr.OFTInteger)
    fieldDef13 = ogr.FieldDefn('grand_id', ogr.OFTInteger)
    fieldDef14 = ogr.FieldDefn('grod_id', ogr.OFTInteger)
    fieldDef15 = ogr.FieldDefn('basin_code', ogr.OFTInteger)
    fieldDef16 = ogr.FieldDefn('manual_add', ogr.OFTInteger)
    fieldDef17 = ogr.FieldDefn('grwl_tile', ogr.OFTString)
    fieldDef18 = ogr.FieldDefn('number_obs', ogr.OFTInteger)
    fieldDef19 = ogr.FieldDefn('orbits', ogr.OFTString)
    fieldDef20 = ogr.FieldDefn('deltaflag', ogr.OFTInteger)
    fieldDef21 = ogr.FieldDefn('endpoints', ogr.OFTInteger)
    fieldDef22 = ogr.FieldDefn('grod_fid', ogr.OFTInteger)
    fieldDef23 = ogr.FieldDefn('hfalls_fid', ogr.OFTInteger)
    fieldDef24 = ogr.FieldDefn('lake_id', ogr.OFTString)
    fieldDef25 = ogr.FieldDefn('old_lakef', ogr.OFTInteger)
    fieldDef26 = ogr.FieldDefn('old_deltas', ogr.OFTInteger)
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
    layerout.CreateField(fieldDef12)
    layerout.CreateField(fieldDef13)
    layerout.CreateField(fieldDef14)
    layerout.CreateField(fieldDef15)
    layerout.CreateField(fieldDef16)
    layerout.CreateField(fieldDef17)
    layerout.CreateField(fieldDef18)
    layerout.CreateField(fieldDef19)
    layerout.CreateField(fieldDef20)
    layerout.CreateField(fieldDef21)
    layerout.CreateField(fieldDef22)
    layerout.CreateField(fieldDef23)
    layerout.CreateField(fieldDef24)
    layerout.CreateField(fieldDef25)
    layerout.CreateField(fieldDef26)

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
        feature_out.SetField('x', grwl.lon[ipix])
        feature_out.SetField('y', grwl.lat[ipix])
        feature_out.SetField('easting', grwl.x[ipix])
        feature_out.SetField('northing', grwl.y[ipix])
        # int() is needed because facc.dtype=float64, needs to be saved
        # with all values whereas lat.dtype=float64, which raise an error.
        feature_out.SetField('segInd', int(grwl.Ind[ipix]))
        feature_out.SetField('segID', int(grwl.seg[ipix]))
        feature_out.SetField('segDist', float(grwl.dist[ipix]))
        feature_out.SetField('p_height', float(grwl.elv[ipix]))
        feature_out.SetField('flowacc', float(grwl.facc_filt[ipix]))
        feature_out.SetField('p_width', float(grwl.wth[ipix]))
        feature_out.SetField('nchan', int(grwl.nchan[ipix]))
        feature_out.SetField('lakeflag', int(grwl.new_lakes[ipix]))
        feature_out.SetField('grand_id', int(grwl.grand[ipix]))
        feature_out.SetField('grod_id', int(grwl.grod[ipix]))
        feature_out.SetField('basin_code', int(grwl.basins[ipix]))
        feature_out.SetField('manual_add', int(grwl.manual[ipix]))
        feature_out.SetField('grwl_tile', str(grwl.tile[ipix]))
        feature_out.SetField('number_obs', int(grwl.num_obs[ipix]))
        feature_out.SetField('orbits', str(grwl.orbits[ipix][np.where(grwl.orbits[ipix] > 0)]))
        feature_out.SetField('deltaflag', int(grwl.delta[ipix]))
        feature_out.SetField('endpoints', int(grwl.eps[ipix]))
        feature_out.SetField('grod_fid', int(grwl.grod_fid[ipix]))
        feature_out.SetField('hfalls_fid', int(grwl.hfalls_fid[ipix]))
        feature_out.SetField('lake_id', str(int(grwl.lake_id[ipix])))
        feature_out.SetField('old_lakef', int(grwl.old_lakes[ipix]))
        feature_out.SetField('old_deltas', int(grwl.old_deltas[ipix]))

        # Add the feature to the layer
        layerout.CreateFeature(feature_out)
        # Delete point geometry
        pixel_point.Destroy()

    # Close feature and shapefiles
    feature_out.Destroy()
    dataout.Destroy()

    return fshpout

###############################################################################

def save_merged_nc(merged, outfile):

    """
    FUNCTION:
        Writes filtered merged NetCDF. Datasets combined include: GRWL,
        MERIT Hydro, GROD, GRanD, HydroBASINS, Global Deltas, SWOT Track
        information, and Prior Lake Database locations.

    INPUTS
        merged -- Object containing merged attributes for the GRWL centerline.
        outfile -- Outpath directory to write the NetCDF file.

    OUTPUTS
        Merged NetCDF file.
    """

    # global attributes
    root_grp = nc.Dataset(outfile, 'w', format='NETCDF4')
    root_grp.x_min = np.min(merged.lon)
    root_grp.x_max = np.max(merged.lon)
    root_grp.y_min = np.min(merged.lat)
    root_grp.y_max = np.max(merged.lat)
    root_grp.production_date = time.strftime("%d-%b-%Y %H:%M:%S", time.gmtime()) #utc time

    # subgroups
    cl_grp = root_grp.createGroup('centerlines')

    # dimensions
    maxview = 75
    root_grp.createDimension('ID', 2)
    cl_grp.createDimension('num_points', len(merged.ID))
    cl_grp.createDimension('orbit', maxview)

    ### variables and units

    # root group variables
    Name = root_grp.createVariable('Name', 'S1', ('ID'))
    Name._Encoding = 'ascii'

    # centerline variables
    cl_id = cl_grp.createVariable(
        'cl_id', 'i8', ('num_points',), fill_value=-9999.)
    x = cl_grp.createVariable(
        'x', 'f8', ('num_points',), fill_value=-9999.)
    x.units = 'degrees east'
    y = cl_grp.createVariable(
        'y', 'f8', ('num_points',), fill_value=-9999.)
    y.units = 'degrees north'
    easting = cl_grp.createVariable(
        'easting', 'f8', ('num_points',), fill_value=-9999.)
    easting.units = 'm'
    northing = cl_grp.createVariable(
        'northing', 'f8', ('num_points',), fill_value=-9999.)
    northing.units = 'm'
    segID = cl_grp.createVariable(
        'segID', 'i4', ('num_points',), fill_value=-9999.)
    segInd = cl_grp.createVariable(
        'segInd', 'f8', ('num_points',), fill_value=-9999.)
    segDist= cl_grp.createVariable(
        'segDist', 'f8', ('num_points',), fill_value=-9999.)
    segDist.units = 'm'
    p_width = cl_grp.createVariable(
        'p_width', 'f8', ('num_points',), fill_value=-9999.)
    p_width.units = 'm'
    p_height = cl_grp.createVariable(
        'p_height', 'f8', ('num_points',), fill_value=-9999.)
    p_height.units = 'm'
    flowacc = cl_grp.createVariable(
        'flowacc', 'f8', ('num_points',), fill_value=-9999.)
    flowacc.units = 'km^2'
    lakeflag = cl_grp.createVariable(
        'lakeflag', 'i4', ('num_points',), fill_value=-9999.)
    deltaflag = cl_grp.createVariable(
        'deltaflag', 'i4', ('num_points',), fill_value=-9999.)
    nchan = cl_grp.createVariable(
        'nchan', 'i4', ('num_points',), fill_value=-9999.)
    grand_id = cl_grp.createVariable(
        'grand_id', 'i4', ('num_points',), fill_value=-9999.)
    grod_id = cl_grp.createVariable(
        'grod_id', 'i4', ('num_points',), fill_value=-9999.)
    grod_fid = cl_grp.createVariable(
        'grod_fid', 'i8', ('num_points',), fill_value=-9999.)
    hfalls_fid = cl_grp.createVariable(
        'hfalls_fid', 'i8', ('num_points',), fill_value=-9999.)
    basin_code = cl_grp.createVariable(
        'basin_code', 'i4', ('num_points',), fill_value=-9999.)
    manual_add = cl_grp.createVariable(
        'manual_add', 'i4', ('num_points',), fill_value=-9999.)
    number_obs = cl_grp.createVariable(
        'number_obs', 'i4', ('num_points',), fill_value=-9999.)
    orbits = cl_grp.createVariable(
        'orbits', 'i4', ('num_points','orbit'), fill_value=-9999.)
    grwl_tile = cl_grp.createVariable(
        'grwl_tile', 'S7', ('num_points',))
    grwl_tile._Encoding = 'ascii'
    endpoints = cl_grp.createVariable(
        'endpoints', 'i4', ('num_points',), fill_value=-9999.)
    lake_id = cl_grp.createVariable(
        'lake_id', 'i8', ('num_points',), fill_value=-9999.)
    old_lakeflag = cl_grp.createVariable(
        'old_lakeflag', 'i4', ('num_points',), fill_value=-9999.)
    old_deltas = cl_grp.createVariable(
        'old_deltas', 'i4', ('num_points',), fill_value=-9999.)

    # data
    print("saving nc")

    # root group data
    cont_str = nc.stringtochar(np.array(['NA'], 'S2'))
    Name[:] = cont_str

    # centerline data
    cl_id[:] = merged.ID
    x[:] = merged.lon
    y[:] = merged.lat
    easting[:] = merged.x
    northing[:] = merged.y
    segInd[:] = merged.Ind
    segID[:] = merged.final_segs
    segDist[:] = merged.dist
    p_width[:] = merged.wth
    p_height[:] = merged.elv
    flowacc[:] = merged.facc
    lakeflag[:] = merged.new_lake
    deltaflag[:] = merged.delta
    nchan[:] = merged.nchan
    grand_id[:] = merged.grand
    grod_id[:] = merged.grod
    grod_fid[:] = merged.grod_fid
    hfalls_fid[:] = merged.hfalls_fid
    basin_code[:] = merged.new_basins
    manual_add[:] = merged.manual
    number_obs[:] = merged.num_obs
    orbits[:,:] = merged.orbits
    endpoints[:] = merged.new_eps
    lake_id[:] = merged.lake_id
    old_lakeflag[:] = merged.new_old_lakeflag
    old_deltas[:] = merged.old_deltas

    m_grwl_tile = np.array(merged.tile)
    grwl_tile[:] = m_grwl_tile

    root_grp.close()

###############################################################################

def save_filtered_shp(merged, outfile):

    """
    FUNCTION:
        Writes filtered merged shapefile. Datasets combined include: GRWL,
        MERIT Hydro, GROD, GRanD, HydroBASINS, Global Deltas, SWOT Track
        information, and Prior Lake Database locations.

    INPUTS
        merged -- Object containing merged attributes for the GRWL centerline.
        outfile -- Outpath directory to write the shapefile.

    OUTPUTS
        Combined shapefile written to the specified outpath.
    """

    # Create and open shapefile.
    driver = ogr.GetDriverByName('ESRI Shapefile')

    fshpout = outfile
    if os.path.exists(fshpout):
        driver.DeleteDataSource(fshpout)
    try:
        dataout = driver.CreateDataSource(fshpout)
    except:
        print('Could not create file ' + fshpout)
        sys.exit(1)

    # Define projection.
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

    # Define pixel attributes
    fieldDef1 = ogr.FieldDefn('x', ogr.OFTReal)
    fieldDef2 = ogr.FieldDefn('y', ogr.OFTReal)
    fieldDef3 = ogr.FieldDefn('easting', ogr.OFTReal)
    fieldDef4 = ogr.FieldDefn('northing', ogr.OFTReal)
    fieldDef5 = ogr.FieldDefn('cl_id', ogr.OFTInteger)
    fieldDef6 = ogr.FieldDefn('segID', ogr.OFTInteger)
    fieldDef7 = ogr.FieldDefn('segDist', ogr.OFTReal)
    fieldDef8 = ogr.FieldDefn('p_height', ogr.OFTReal)
    fieldDef9 = ogr.FieldDefn('flowacc', ogr.OFTReal)
    fieldDef10 = ogr.FieldDefn('p_width', ogr.OFTReal)
    fieldDef11 = ogr.FieldDefn('nchan', ogr.OFTInteger)
    fieldDef12 = ogr.FieldDefn('lakeflag', ogr.OFTInteger)
    fieldDef13 = ogr.FieldDefn('grand_id', ogr.OFTInteger)
    fieldDef14 = ogr.FieldDefn('grod_id', ogr.OFTInteger)
    fieldDef15 = ogr.FieldDefn('basin_code', ogr.OFTInteger)
    fieldDef16 = ogr.FieldDefn('manual_add', ogr.OFTInteger)
    fieldDef17 = ogr.FieldDefn('grwl_tile', ogr.OFTString)
    fieldDef18 = ogr.FieldDefn('number_obs', ogr.OFTInteger)
    fieldDef19 = ogr.FieldDefn('orbits', ogr.OFTString)
    fieldDef20 = ogr.FieldDefn('deltaflag', ogr.OFTInteger)
    fieldDef21 = ogr.FieldDefn('segInd', ogr.OFTInteger)
    fieldDef22 = ogr.FieldDefn('endpoints', ogr.OFTInteger)
    fieldDef23 = ogr.FieldDefn('grod_fid', ogr.OFTInteger)
    fieldDef24 = ogr.FieldDefn('hfalls_fid', ogr.OFTInteger)
    #fieldDef25 = ogr.FieldDefn('lake_id', ogr.OFTString)

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
    layerout.CreateField(fieldDef12)
    layerout.CreateField(fieldDef13)
    layerout.CreateField(fieldDef14)
    layerout.CreateField(fieldDef15)
    layerout.CreateField(fieldDef16)
    layerout.CreateField(fieldDef17)
    layerout.CreateField(fieldDef18)
    layerout.CreateField(fieldDef19)
    layerout.CreateField(fieldDef20)
    layerout.CreateField(fieldDef21)
    layerout.CreateField(fieldDef22)
    layerout.CreateField(fieldDef23)
    layerout.CreateField(fieldDef24)
    #layerout.CreateField(fieldDef23)

    # Create feature (point) to store pixel
    floutDefn = layerout.GetLayerDefn()
    feature_out = ogr.Feature(floutDefn)

    for ipix in range(len(merged.lon)):
        # Create Geometry Point with pixel coordinates
        pixel_point = ogr.Geometry(ogr.wkbPoint)
        pixel_point.AddPoint(merged.lon[ipix], merged.lat[ipix])
        # Add the geometry to the feature
        feature_out.SetGeometry(pixel_point)
        # Set feature attributes
        feature_out.SetField('x', merged.lon[ipix])
        feature_out.SetField('y', merged.lat[ipix])
        feature_out.SetField('easting', merged.x[ipix])
        feature_out.SetField('northing', merged.y[ipix])
        # int() is needed because facc.dtype=float64, needs to be saved
        # with all values whereas lat.dtype=float64, which raise an error.
        feature_out.SetField('cl_id', int(merged.ID[ipix]))
        feature_out.SetField('segID', int(merged.final_segs[ipix]))
        feature_out.SetField('segDist', float(merged.dist[ipix]))
        feature_out.SetField('p_height', float(merged.elv[ipix]))
        feature_out.SetField('flowacc', float(merged.facc[ipix]))
        feature_out.SetField('p_width', float(merged.wth[ipix]))
        feature_out.SetField('nchan', int(merged.nchan[ipix]))
        feature_out.SetField('lakeflag', int(merged.new_lake[ipix]))
        feature_out.SetField('grand_id', int(merged.grand[ipix]))
        feature_out.SetField('grod_id', int(merged.grod[ipix]))
        feature_out.SetField('basin_code', int(merged.new_basins[ipix]))
        feature_out.SetField('manual_add', int(merged.manual[ipix]))
        feature_out.SetField('grwl_tile', str(merged.tile[ipix]))
        feature_out.SetField('number_obs', int(merged.num_obs[ipix]))
        feature_out.SetField('orbits', str(merged.orbits[ipix][np.where(merged.orbits[ipix] > 0)]))
        feature_out.SetField('deltaflag', int(merged.delta[ipix]))
        feature_out.SetField('segInd', int(merged.Ind[ipix]))
        feature_out.SetField('endpoints', int(merged.new_eps[ipix]))
        feature_out.SetField('grod_fid', int(merged.grod_fid[ipix]))
        feature_out.SetField('hfalls_fid', int(merged.hfalls_fid[ipix]))
        feature_out.SetField('lake_id', int(merged.lake_id[ipix]))

        # Add the feature to the layer
        layerout.CreateFeature(feature_out)
        # Delete point geometry
        pixel_point.Destroy()

    # Close feature and shape files
    feature_out.Destroy()
    dataout.Destroy()

    return fshpout

###############################################################################
############################## Merging Tools ##################################
###############################################################################

def find_MH_tiles(grwl, paths):

    """
    FUNCTION:
        Determines which MERIT Hydro raster files overlap a GRWL shapefile.

    INPUTS
        grwl -- Object containing attributes for the GRWL centerline.
        paths -- List of MERIT Hydro raster file directories.

    OUTPUTS
        List of path indexes that overlap specified grwl tile from
        input path list.
    """

    #define grwl extent as ogr polygon format.
    poly1 = ogr.Geometry(ogr.wkbLinearRing)
    poly1.AddPoint(min(grwl.lon), max(grwl.lat))
    poly1.AddPoint(min(grwl.lon), min(grwl.lat))
    poly1.AddPoint(max(grwl.lon), min(grwl.lat))
    poly1.AddPoint(max(grwl.lon), max(grwl.lat))
    poly1.AddPoint(min(grwl.lon), max(grwl.lat))
    shapeGeometry = ogr.Geometry(ogr.wkbPolygon)
    shapeGeometry.AddGeometry(poly1)

    #define MERIT Hydro raster extents and see if they overlap GRWL shapefile.
    indexes = list()
    for ind in list(range(len(paths))):
        # Get raster geometry
        raster = gdal.Open(paths[ind])
        transform = raster.GetGeoTransform()
        pixelWidth = transform[1]
        pixelHeight = transform[5]
        cols = raster.RasterXSize
        rows = raster.RasterYSize

        xLeft = transform[0]
        yTop = transform[3]
        xRight = xLeft+cols*pixelWidth
        yBottom = yTop+rows*pixelHeight

        # Translate raster geometry into ogr polygon format.
        poly2 = ogr.Geometry(ogr.wkbLinearRing)
        poly2.AddPoint(xLeft, yTop)
        poly2.AddPoint(xLeft, yBottom)
        poly2.AddPoint(xRight, yBottom)
        poly2.AddPoint(xRight, yTop)
        poly2.AddPoint(xLeft, yTop)
        rasterGeometry = ogr.Geometry(ogr.wkbPolygon)
        rasterGeometry.AddGeometry(poly2)

        # Seeing if raster overlaps shapefile.
        answer = shapeGeometry.Intersect(rasterGeometry)
        if answer == True:
            indexes.append(ind)

    return indexes

###############################################################################

def MH_coords(file_paths, grwl_ext):

    """
    FUNCTION:
        Reads in and formats MERIT Hydro raster coordinates as arrays.

    INPUTS
        filepaths -- List of MERIT Hydro raster paths that overlap a GRWL shapefile.
        grwl_ext -- Latitude/Longitude extent of GRWL shapefile.

    OUTPUTS
        mhydro - Object containg 1-D latitude and longitude arrays for MERIT Hydro.
        Initial attributes: (mhydro.lat, mhydro.lon)
    """

    # Mosaicking all MERIT Hydro rasters that overlap the GRWL shapefile.
    src_files_to_mosaic = []
    for fp in file_paths:
       src = rasterio.open(fp)
       src_files_to_mosaic.append(src)
    mosaic, out_trans = merge(src_files_to_mosaic, tuple(grwl_ext))

    # Defining the lower left coordinates of the mosaicked raster.
    xul = out_trans[2]
    xres = out_trans[0]
    yul = out_trans[5]
    yres = out_trans[4]

    # Defining coordinates for every raster pixel.
    lon=np.array([xul+xres*c+xres/2. for r in range(mosaic.shape[1]) for c in range(mosaic.shape[2])])
    lat=np.array([yul+yres*r+yres/2. for r in range(mosaic.shape[1]) for c in range(mosaic.shape[2])])

    # Assiging lat/lon coordinates as attributes to "mhydro" object.
    mhydro = Object()
    mhydro.lon = lon
    mhydro.lat = lat

    return mhydro

###############################################################################

def MH_vals(file_paths, grwl_ext):

    """
    FUNCTION:
        Reads in and formats MERIT Hydro raster values as arrays.

    INPUTS
        filepaths -- List of MERIT Hydro raster paths that overlap a GRWL shapefile.
        grwl_ext -- Latitude/Longitude extent of GRWL shapefile.

    OUTPUTS
        vals -- raster values in 1-D array. These values will coincide with
        coordinate values returned from "MH_coords" function.
    """

    # Mosaicking all MERIT Hydro rasters that overlap the GRWL shapefile.
    src_files_to_mosaic = []
    for fp in file_paths:
       src = rasterio.open(fp)
       src_files_to_mosaic.append(src)
    mosaic, out_trans = merge(src_files_to_mosaic, tuple(grwl_ext))

    # Pulls and flattens raster vlues.
    vals = mosaic.flatten()

    return vals

###############################################################################

def find_projection(latitude, longitude):

    """
    Modified by E.Altenau from C. Lion's function in "Tools.py" (c) 2016.

    FUNCTION:
        Projects all points in UTM.

    INPUTS
        latitude -- latitude in degrees (ARRAY FORMAT)
        longitude -- longitude in degrees (ARRAY FORMAT)

    OUTPUTS
        east -- easting in UTM
        north -- northing in UTM
        zone_num -- UTM zone number
        zone_let -- UTM zone letter
    """

    # Creating initial arrays to fill.
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

def filter_facc(grwl):

    """
    FUNCTION:
        Filters flow accumulation values that are originally attached to the
        GRWL centerline from MERIT Hydro. Due to discrepancies in the two
        dataset centerline locations, post filtering must be done to eliminate
        outliers in flow accumulation values along the centerline, and ensure a
        downstream increase in flow accumulation.

    INPUTS
        grwl -- Object containing attributes for the GRWL centerline.

    OUTPUTS
        new_facc -- New filtered flow accumulation (1-D array).
    """

    # Identifying unique grwl segments.
    unq_grwl_ids = np.unique(grwl.seg)
    new_facc = np.zeros(len(grwl.facc))

    # Identifying outliers in flow accumulation in each grwl segment.
    for num in range(len(unq_grwl_ids)):
        # Calculating standard deviation and median values of flow accumulation.
        seg = np.where(grwl.seg == unq_grwl_ids[num])[0]
        facc_clip = grwl.facc[seg]
        facc_std = np.std(facc_clip)
        facc_mdn = np.median(facc_clip)

        # Deleting flow accumulation values greater than +/- 1 standard deviation
        # from median
        facc_clip[np.where(facc_clip > facc_mdn+facc_std)] = np.nan
        facc_clip[np.where(grwl.facc[seg] < facc_mdn-facc_std)] = np.nan

        # If the standard deviation is greater than the median, delete flow
        # accumulation values that are less than the median value - 1/2 the
        # median value.
        if facc_std > facc_mdn:
            facc_clip[np.where(grwl.facc[seg] < facc_mdn-(facc_mdn/2))] = np.nan

        # Calculating min/max values of remaining flow accumulation values
        # and linearly interpolating between the values.
        facc_max = np.nanmax(facc_clip)
        facc_min = np.nanmin(facc_clip)
        interp_vals = np.linspace(facc_min, facc_max, len(seg))

        # Determining the upstream and downstream direction using elevation values.
        elv_clip = grwl.elv[seg]
        elv_beg_id = elv_clip[0]
        elv_end_id = elv_clip[-1]

        # Assigning filtered flow accumulation values to new object attribute.
        if elv_end_id == elv_beg_id:
            new_facc[seg] = facc_max
        elif elv_end_id > elv_beg_id:
            new_facc[seg] = interp_vals[::-1]
        else:
            new_facc[seg] = interp_vals

    return new_facc

###############################################################################

def overlapping_tracks(grwl, track_list):

    """
    FUNCTION:
        Determines which SWOT Track files overlap a GRWL shapefile.

    INPUTS
        grwl -- Object containing attributes for the GRWL centerline.
        track_list -- List of SWOT Track polygon file directories.

    OUTPUTS
        List of path indexes that overlap specified grwl tile from
        input path list.
    """

    #define grwl extent as ogr geometry format.
    poly1 = ogr.Geometry(ogr.wkbLinearRing)
    poly1.AddPoint(min(grwl.lon), max(grwl.lat))
    poly1.AddPoint(min(grwl.lon), min(grwl.lat))
    poly1.AddPoint(max(grwl.lon), min(grwl.lat))
    poly1.AddPoint(max(grwl.lon), max(grwl.lat))
    poly1.AddPoint(min(grwl.lon), max(grwl.lat))
    grwlGeometry = ogr.Geometry(ogr.wkbPolygon)
    grwlGeometry.AddGeometry(poly1)

    #find overlapping SWOT tracks.
    track_files = []
    for fn in track_list:
        driver = ogr.GetDriverByName('ESRI Shapefile')
        shape = driver.Open(fn)
        inLayer = shape.GetLayer()
        #inLayer is always of size one because polygon is a unique value.
        for feature in inLayer:
            track=feature.GetGeometryRef()
            answer = track.Intersects(grwlGeometry)
            if answer == True:
                track_files.append(fn)
    track_files = np.unique(track_files)

    return(track_files)

###############################################################################

def add_tracks(grwl, shp_file, list_files):

    """
    FUNCTION:
        Identifies overlapping SWOT track (number/day) information along the
        GRWL centerlines.

    INPUTS
        grwl -- Object containing attributes for the GRWL centerline.
        shp_file -- GRWL shapefile directory.
        list_files -- List of SWOT track polygon directories that overlap
                      a GRWL shapefile.

    OUTPUTS
        num_obs -- Number of SWOT observations (1-D array).
        orbit_list -- List of overlapping SWOT tracks (List).
        day_list -- List of overpass dates for each overlapping SWOT track (List).
    """

    # Reading in GRWL point information.
    points = gp.GeoDataFrame.from_file(shp_file)

    # Creating empty arrays to fill.
    orbit_array = np.zeros((len(points), 75), dtype=int)
    #day_array = np.zeros((len(points), 50), dtype=int)
    '''
    # Looping through SWOT tracks and identifying intersecting points.
    if len(list_files) > 25:
        end = 25
    else:
        end = len(list_files)
    '''
    for ind in list(range(len(list_files))):
        #print ind
        poly = gp.GeoDataFrame.from_file(list_files[ind])
        intersect = gp.sjoin(poly, points, how="inner")
        intersect = pd.DataFrame(intersect)
        intersect = intersect.drop_duplicates(subset='index_right', keep='first')
        ids = np.array(intersect.index_right)
        orbit_array[ids, ind] = intersect.ID_PASS
        #days = np.array([int(str(intersect.START_TIME)[9:11]) for time in intersect.START_TIME])
        #day_array[ids, ind] = days #intersect.START_TIME

    '''
    # Turning SWOT information into lists.
    orbit_list = []
    day_list = []
    for ind in list(range(len(points))):
        orbit_vals = orbit_array[ind][np.where(orbit_array[ind,:] > 0)]
        day_vals = day_array[ind][np.where(day_array[ind] > 0)]
        if len(orbit_vals) == 0:
            orbit_list.append([0])
            day_list.append([0])
        else:
            orbit_list.append(str(orbit_vals))
            day_list.append(str(day_vals))
    '''
    # Finding number of SWOT observations.
    orbit_binary = np.copy(orbit_array)
    orbit_binary[np.where(orbit_binary > 0)] = 1
    num_obs = np.sum(orbit_binary, axis = 1)

    # Assigning SWOT track information to new object attributes.
    return num_obs, orbit_array

###############################################################################

def add_deltas(grwl, fn_grwl, delta_db):

    """
    FUNCTION:
        Creates a 1-D array designating delta exent along the GRWL centerlines.
        0 - no delta, 1 - delta.

    INPUTS
        grwl -- Object containing attributes for the GRWL centerline.
        fn_grwl -- GRWL shapefile directory.
        fn_deltas -- Delta shapefile directory.

    OUTPUTS
        coastal_flag -- Delta extent (binary flag).
    """

    # Finding where delta shapefiles intersect the GRWL shapefile.
    points = gp.GeoDataFrame.from_file(fn_grwl)
    poly = delta_db
    intersect = gp.sjoin(poly, points, how="inner")
    intersect = pd.DataFrame(intersect)
    intersect = intersect.drop_duplicates(subset='index_right', keep='first')

    # Identifying the delta ID.
    ids = np.array(intersect.index_right)
    delta_flag = np.zeros(len(grwl.seg))
    delta_flag[ids] = np.array(intersect.DeltaID)

    # Filtering GRWL lake flag and new delta flag information to create final
    # coastal flag.
    uniq_segs = np.unique(grwl.seg)
    coastal_flag = np.zeros(len(grwl.seg))
    for ind in list(range(len(uniq_segs))):
        seg = np.where(grwl.seg == uniq_segs[ind])[0]
        flag1 = (len(np.where(grwl.lake[seg] >= 3)[0])/len(seg))*100
        flag2 = (len(np.where(delta_flag[seg] > 0)[0])/len(seg))*100
        if flag1 > 25 or flag2 > 25:
            coastal_flag[seg] = 1

    return coastal_flag

###############################################################################

def calc_segDist(grwl):

    """
    FUNCTION:
        Creates a 1-D array of flow distances for each GRWL segment.

    INPUTS
        grwl -- Object containing attributes for the GRWL centerline.

    OUTPUTS
        seg_dist -- Flow distance per GRWL segment (m).
    """

    # Loop through each segment and calculate flow distance.
    seg_dist = np.zeros(len(grwl.x))
    uniq_segs = np.unique(grwl.seg)
    for ind in list(range(len(uniq_segs))):
        seg = np.where(grwl.seg == uniq_segs[ind])[0]
        seg_x = grwl.x[seg]
        seg_y = grwl.y[seg]
        facc = grwl.facc_filt[seg]

        # Calculating cumulative distance.
        order_ids = np.argsort(grwl.Ind[seg])
        dist = np.zeros(len(seg))
        dist[order_ids[0]] = 0
        for idx in list(range(len(order_ids)-1)):
            d = np.sqrt((seg_x[order_ids[idx]]-seg_x[order_ids[idx+1]])**2 +
                        (seg_y[order_ids[idx]]-seg_y[order_ids[idx+1]])**2)
            dist[order_ids[idx+1]] = d + dist[order_ids[idx]]

        # Identify flow direction using flow accumulation.
        dist = np.array(dist)
        start = facc[np.where(dist == np.min(dist))[0][0]]
        end = facc[np.where(dist == np.max(dist))[0][0]]

        # Assigning flow distance to object attribute.
        if end > start:
            seg_dist[seg] = abs(dist-np.max(dist))

        else:
            seg_dist[seg] = dist

    return seg_dist

###############################################################################

def add_dams(grwl, fn_grand, fn_grod):

    """
    FUNCTION:
        Creates 1-D arrays of GRanD and GROD dam locations along the  GRWL
        centerline.

    INPUTS
        grwl -- Object containing attributes for the GRWL centerline.
        fn_grand -- GRanD shapefile directory.
        fn_grod -- GROD shapefile directory.

    OUTPUTS
        grand_ID -- GRanD ID.
        grod_ID -- GROD ID.
    """

    # Read in GRaND and GROD data.
    grand_lat, grand_lon, grand_id, grand_skm = open_grand_shpfile(fn_grand)
    grand_lat = abs(grand_lat)
    grod_info = pd.read_csv(fn_grod)
    grod_lat =  np.array(grod_info.lat)
    grod_lon =  np.array(grod_info.lon)
    grod_names = np.array(grod_info.name) #.flatten() #grod_names = np.array(grod_info[[0]]).flatten()
    grod_id = np.zeros(len(grod_names))
    grod_fid = np.array(grod_info.fid)#.flatten()

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
    ll = np.array([np.min(grwl.lon), np.min(grwl.lat)])  # lower-left
    ur = np.array([np.max(grwl.lon), np.max(grwl.lat)])  # upper-right

    inidx = np.all(np.logical_and(ll <= pts, pts <= ur), axis=1)
    inbox = pts[inidx]
    grod_id_clip = grod_id[inidx]
    grod_fid_clip = grod_fid[inidx]
    grod_lon_clip = inbox[:,0]
    grod_lat_clip = inbox[:,1]

    # calculating x-y utm values.
    grand_x, grand_y, __, __ = find_projection(grand_lat, grand_lon)

    # Attach dam locations to GRWL.
    grwl_pts = np.vstack((grwl.x, grwl.y)).T
    grand_pts = np.vstack((grand_x, grand_y)).T
    kdt = sp.cKDTree(grwl_pts)
    grand_dist, grand_idx = kdt.query(grand_pts, k = 1)
    # Narrowing dam locations by distance threshold.
    grand_dist_thresh = np.where(grand_dist <= 1000)[0]
    grand_locs, grand_locs_idx = np.unique(grand_idx[grand_dist_thresh], return_index=True)
    # Creating arrays.
    grand_ID = np.zeros(len(grwl.x))
    # Assigning values.
    grand_ID[grand_locs] = grand_id[grand_dist_thresh[grand_locs_idx]]


    if len(grod_lon_clip) == 0:
        # Creating arrays.
        grod_ID = np.zeros(len(grwl.x))
        grod_FID = np.zeros(len(grwl.x))
        hfalls_FID = np.zeros(len(grwl.x))
    else:
        # calculating x-y utm values.
        grod_x, grod_y, __, __ = find_projection(grod_lat_clip, grod_lon_clip)

        # Attach dam locations to GRWL.
        grod_pts = np.vstack((grod_x, grod_y)).T
        kdt = sp.cKDTree(grwl_pts)
        grod_dist, grod_idx = kdt.query(grod_pts, k = 1)

        # Narrowing dam locations by distance threshold.
        grod_dist_thresh = np.where(grod_dist <= 2000)[0] #was 100m
        grod_locs, grod_locs_idx = np.unique(grod_idx[grod_dist_thresh], return_index=True)

        # Creating arrays.
        grod_ID = np.zeros(len(grwl.x))
        grod_FID = np.zeros(len(grwl.x))
        hfalls_FID = np.zeros(len(grwl.x))
        # Assigning values.
        grod_ID[grod_locs] = grod_id_clip[grod_dist_thresh[grod_locs_idx]]
        grod_FID[grod_locs] = grod_fid_clip[grod_dist_thresh[grod_locs_idx]]
        hfalls_FID[grod_locs] = grod_fid_clip[grod_dist_thresh[grod_locs_idx]]
        grod_FID[np.where(grod_ID == 4)] = 0
        hfalls_FID[np.where(grod_ID != 4)] = 0

    return grand_ID, grod_ID, grod_FID, hfalls_FID

###############################################################################

def add_basins(grwl, fn_grwl, fn_basins):

    """
    FUNCTION:
        Creates a 1-D array of intersecting HydroBASINS Pfafstetter IDs
        for each GRWL centerline point.

    INPUTS
        grwl -- Object containing attributes for the GRWL centerline.
        fn_grwl -- GRWL shapefile directory.
        fn_basins -- HydroBASINS shapefile directory.

    OUTPUTS
        basin_code -- Pfafstetter basin code.
    """

    # Attaching basin codes
    points = gp.GeoDataFrame.from_file(fn_grwl)
    poly = gp.GeoDataFrame.from_file(fn_basins)
    intersect = gp.sjoin(poly, points, how="inner")
    intersect = pd.DataFrame(intersect)
    intersect = intersect.drop_duplicates(subset='index_right', keep='first')

    # Creating arrays.
    ids = np.array(intersect.index_right)
    basin_code = np.zeros(len(grwl.seg)) #see if len(points) works...
    basin_code[ids] = np.array(intersect.PFAF_ID)

    # Assigning basin locations to GRWL object attributes.
    return basin_code

###############################################################################

def fill_zero_basins(grwl):

    """
    FUNCTION:
        Fills in basin codes along GRWL centerline locations that do not
        overlap the HydroBASINS polygons.

    INPUTS
        grwl -- Object containing attributes for the GRWL centerline.

    OUTPUTS
        grwl.basins -- Filled in basin values.
    """

    # Loop though each GRWL segment and fill in points where the basin code = 0.
    zero_pts = np.where(grwl.basins == 0)[0]
    uniq_segs = np.unique(grwl.seg[zero_pts])
    for ind in list(range(11,len(uniq_segs))):
        seg = np.where(grwl.seg == uniq_segs[ind])[0]
        zpts = np.where(grwl.basins[seg] == 0)[0]
        vpts = np.where(grwl.basins[seg] > 0)[0]

        if len(zpts) == 0:
            continue

        if len(vpts) == 0:
            # if there are no points within the segment greater than 0;
            # use all grwl tile points with basin values > 0.
            vpts = np.where(grwl.basins > 0)[0]

            if len(vpts) == 0:
                continue

            #find closest neighbor for all points.
            z_pts = np.vstack((grwl.x[seg[zpts]], grwl.y[seg[zpts]])).T
            v_pts = np.vstack((grwl.x[vpts], grwl.y[vpts])).T
            kdt = sp.cKDTree(v_pts)
            eps_dist, eps_ind = kdt.query(z_pts, k = 25)

            min_dist = np.min(eps_dist)
            if min_dist > 1000:
                continue

            #calculate mode of closest basin values.
            close_basins = grwl.basins[vpts[eps_ind]].flatten()
            basin_mode = max(set(list(close_basins)), key=list(close_basins).count)
            #assign zero basin values the mode value.
            grwl.basins[seg[zpts]] = basin_mode


        else:
            #find closest neighbor for all points.
            z_pts = np.vstack((grwl.x[seg[zpts]], grwl.y[seg[zpts]])).T
            v_pts = np.vstack((grwl.x[seg[vpts]], grwl.y[seg[vpts]])).T
            kdt = sp.cKDTree(v_pts)
            __, eps_ind = kdt.query(z_pts, k = 25)
            eps_ind = np.unique(eps_ind)
            rmv = np.where(eps_ind == len(vpts))[0] 
            eps_ind = np.delete(eps_ind, rmv)
            
            #calculate mode of closest basin values.
            if len(vpts) < len(zpts):
                close_basins = grwl.basins[seg[vpts]].flatten()
            else:
                close_basins = grwl.basins[seg[vpts[eps_ind]]].flatten()
            basin_mode = max(set(list(close_basins)), key=list(close_basins).count)
            #basin_mode = max(grwl.basins[seg])
            #assign zero basin values the mode value.
            grwl.basins[seg[zpts]] = basin_mode

        # print(ind, basin_mode)

###############################################################################

def add_lakedb(grwl, fn_grwl, lake_db):

    """
    FUNCTION:
        Creates a 1-D array of intersecting Prior Lake Database (PLD) IDs
        for each GRWL centerline point.

    INPUTS
        grwl -- Object containing attributes for the GRWL centerline.
        fn_grwl -- GRWL shapefile directory.
        fn_lakes -- Lake Database shapefile directory.

    OUTPUTS
        lb_code -- Prior Lake Database (PLD) IDs.
    """

    # Attaching PLD IDs
    points = gp.GeoDataFrame.from_file(fn_grwl)
    intersect = gp.sjoin(lake_db, points, how="inner")
    intersect = pd.DataFrame(intersect)
    intersect = intersect.drop_duplicates(subset='index_right', keep='first')

    # Creating arrays.
    ids = np.array(intersect.index_right)
    lb_code = np.zeros(len(grwl.seg)) #see if len(points) works...
    lb_code[ids] = np.array(intersect.lake_id)

    # Assigning basin locations to GRWL object attributes.
    return lb_code

###############################################################################

def renumber_grwl_segs(grwl_segs, grwl_tile):

    """
    FUNCTION:
        Creates a 1-D array of re-numbered GRWL segments in the continental-scale
        data to have individual IDs across all GRWL tiles.

    INPUTS
        grwl_segs -- GRWL segment ids.
        grwl_tile -- GRWL tile id containing the points.

    OUTPUTS
        new_segs -- Re-numbered GRWL segments.
    """

    uniq_tiles = np.unique(grwl_tile)
    new_segs = np.copy(grwl_segs)
    cnt = 1
    for ind in list(range(len(uniq_tiles))):
        tile = np.where(grwl_tile == uniq_tiles[ind])[0]
        segs = np.unique(grwl_segs[tile])
        for idx in list(range(len(np.unique(segs)))):
            seg2 = np.where(grwl_segs[tile] == segs[idx])[0]
            new_segs[tile[seg2]] = cnt
            cnt = cnt+1

    return new_segs

###############################################################################

# def edit_basins(basin_code, grwl_id):

#     """
#     FUNCTION:
#         Creates a 1-D array of updated basin code ids to be consistent along
#         a GRWL segment. This function Fixes locations where GRWL centerlines
#         and HydroBASINS boundaries don't line up perfectly.

#     INPUTS
#         basin_code -- Pfafstetter basin code along the GRWL centerline.
#         grwl_id -- GRWL segment id.

#     OUTPUTS
#         new_basins -- Filtered Pfafstetter basin codes.
#     """

#     # Assigns the basin code mode within a GRWL segment to the entire segment.
#     uniq_id = np.unique(grwl_id)
#     new_basins = np.copy(basin_code)
#     for ind in list(range(len(uniq_id))):
#         seg = np.where(grwl_id == uniq_id[ind])[0]
#         bcs = np.unique(basin_code[seg])
#         if len(bcs) == 1:
#             continue
#         if len(bcs) > 1:
#             mode = max(set(list(basin_code[seg])), key=list(basin_code[seg]).count)
#             new_basins[seg] = mode

#     return new_basins

###############################################################################

def edit_basins(basin_code, grwl_id, x, y):

    """
    FUNCTION:
        Creates a 1-D array of updated basin code ids to be consistent along
        a GRWL segment. This function Fixes locations where GRWL centerlines
        and HydroBASINS boundaries don't line up perfectly.

    INPUTS
        basin_code -- Pfafstetter basin code along the GRWL centerline.
        grwl_id -- GRWL segment id.

    OUTPUTS
        new_basins -- Filtered Pfafstetter basin codes.
    """

    # Assigns the basin code mode within a GRWL segment to the entire segment.
    uniq_id = np.unique(grwl_id)
    new_basins = np.copy(basin_code)
    for ind in list(range(len(uniq_id))):
        seg = np.where(grwl_id == uniq_id[ind])[0]
        seg_basins = basin_code[seg]
        seg_x = x[seg]
        seg_y = y[seg]
        bcs = np.unique(basin_code[seg])

        if len(bcs) == 1:
            continue
        if len(bcs) > 1:
            for b in list(range(len(bcs))):
                basin_pts = np.where(seg_basins == bcs[b])[0]
                if len(basin_pts) < 15:
                    other_pts = np.where(seg_basins != bcs[b])[0]
                    b_pts = np.vstack((seg_x[basin_pts], seg_y[basin_pts])).T
                    o_pts = np.vstack((seg_x[other_pts], seg_y[other_pts])).T
                    kdt = sp.cKDTree(o_pts)
                    __, eps_ind = kdt.query(b_pts, k = 10)
                    mode = max(set(list(seg_basins[eps_ind[:,0]])), key=list(seg_basins[eps_ind[:,0]]).count)
                    new_basins[seg[basin_pts]] = mode
                else:
                    continue

    return new_basins

###############################################################################

def cut_segments(grwl, start_seg):

    """
    FUNCTION:
        Creates a new 1-D that contains unique segment IDs for the GRWL
        segments that need to be cut at tributary junctions.

    INPUTS
        grwl --  Object containing attributes for the GRWL centerlines.
        start_seg -- Starting ID value to assign to the new cut segments.

    OUTPUTS
        new_segs -- Updated Segment IDs.
    """

    updated_segs = np.copy(grwl.new_segs)
    cut = np.where(grwl.tribs == 1)[0]
    cut_segs = np.unique(grwl.new_segs[cut])
    seg_id = start_seg

    # Loop through segments that contain tributary junctions and identify
    # the new boundaries of the segment to cut and re-number.
    for ind in list(range(len(cut_segs))):
        seg = np.where(grwl.new_segs == cut_segs[ind])[0]
        num_tribs = np.where(grwl.tribs[seg] == 1)[0]
        max_ind = np.where(grwl.Ind[seg] == np.max(grwl.Ind[seg]))[0]
        min_ind = np.where(grwl.Ind[seg] == np.min(grwl.Ind[seg]))[0]
        bounds = np.insert(num_tribs, 0, min_ind)
        bounds = np.insert(bounds, len(bounds), max_ind)
        for idx in list(range(len(bounds)-1)):
            id1 = bounds[idx]
            id2 = bounds[idx+1]
            new_vals = np.where((grwl.Ind[seg] >= grwl.Ind[seg[id1]]) & (grwl.Ind[seg] <= grwl.Ind[seg[id2]]))[0]
            updated_segs[seg[new_vals]] = seg_id
            seg_id = seg_id+1

    return updated_segs

###############################################################################

def cut_continental_tribs(merged):

    """
    FUNCTION:
        Cuts segments at tributary junctions that occur at the GRWL tile edges
        and assigns the cut segments new segment IDs. This is a subfunction
        for the "format_data" function.

    INPUTS
        merged -- Object containing attributes for the merged GRWL centerline.

    OUTPUTS
        final_segs -- Final merged segment IDs for an entire continent.
        new_eps = Updated endpoint locations for new segment definitions.
    """

    #find 20 closest neighbors for all points.
    all_pts = np.vstack((merged.lon, merged.lat)).T
    kdt = sp.cKDTree(all_pts)
    eps_dist, eps_ind = kdt.query(all_pts, k = 20)
    eps = np.where(merged.eps == 1)[0]
    merged.tribs = np.zeros(len(merged.new_segs))

    #for all the endpoint locations identify whether there is an unidentified tributary junction.
    for ind in list(range(len(eps))):
        close_pts = np.unique(merged.new_segs[eps_ind[eps[ind],:]])
        neighbors = close_pts[np.where(close_pts != merged.new_segs[eps[ind]])[0]]
        if len(neighbors) == 1:
            ep1_min = np.min(merged.Ind[np.where(merged.new_segs == neighbors[0])[0]])+5
            ep1_max = np.max(merged.Ind[np.where(merged.new_segs == neighbors[0])[0]])-5
            pt = eps_ind[eps[ind],np.min(np.where(merged.new_segs[eps_ind[eps[ind],:]] == neighbors[0])[0])]
            pt_id = merged.Ind[pt]
            if ep1_min < pt_id < ep1_max:
                #print(ind)
                merged.tribs[pt] = 1

    #cut segments at tributaries and update endpoints.
    start_seg = np.max(merged.new_segs)+1
    final_segs = cut_segments(merged, start_seg)
    new_eps = np.copy(merged.eps)
    keep = np.where(merged.tribs == 1)[0]
    new_eps[keep] = 1

    return final_segs, new_eps

###############################################################################

def format_data(merged):

    """
    FUNCTION:
        Post-filters merged data after individual merged tiles are mosaicked
        at the continental scale.

    INPUTS
        merged -- Object containing attributes for the merged GRWL centerline.

    OUTPUTS
        merged.ID -- Unique point id for all the centerline points.
        merged.new_segs = Unique segment id.
        merged.new_lake = Filtered lake flag.
        merged.new_basins = Filtered Pfafstetter basin code.
        merged.final_segs -- Final merged segment IDs for an entire continent.
        merged. new_eps = Updated endpoint locations for the final segment definitions.
    """

    merged.tile = np.array(merged.tile)
    merged.orbits = np.array(merged.orbits)
    #merged.days = np.array(merged.days)

    # Creating unique segment id and editing lake flags.
    cl_id = np.array(range(len(merged.Ind)))
    new_segs = renumber_grwl_segs(merged.seg, merged.tile)

    new_lakes = np.copy(merged.lake)
    new_lakes[np.where(merged.lake==250)] = 0
    new_lakes[np.where(merged.lake==163)] = 1
    new_lakes[np.where(merged.lake==181)] = 1
    new_lakes[np.where(merged.lake==125)] = 3
    new_lakes[np.where(merged.lake== 86)] = 2
    new_lakes[np.where(merged.lake== 87)] = 2

    old_new_lakes = np.copy(merged.old_lakeflag)
    old_new_lakes[np.where(merged.old_lakeflag==250)] = 0
    old_new_lakes[np.where(merged.old_lakeflag==163)] = 1
    old_new_lakes[np.where(merged.old_lakeflag==181)] = 1
    old_new_lakes[np.where(merged.old_lakeflag==125)] = 3
    old_new_lakes[np.where(merged.old_lakeflag==86)] = 2
    old_new_lakes[np.where(merged.old_lakeflag==87)] = 2

    # Basin junction filter
    basins = np.array([int(str(ind)[0:6]) for ind in merged.basins])
    level3 = np.array([int(str(ind)[0:3]) for ind in basins])
    new_basins = np.zeros(len(merged.Ind))

    uniq_basins = np.unique(level3)
    for ind in list(range(len(uniq_basins))):

        if uniq_basins[ind] == 0:
            continue

        pts = np.where(level3 == uniq_basins[ind])[0]
        new_id = new_segs[pts]
        subbasins = basins[pts]
        sub_x = merged.lon
        sub_y = merged.lat    
        basin_edits = edit_basins(subbasins, new_id, sub_x, sub_y)
        new_basins[pts] = basin_edits

    # Assigning filtered values to merged object attributes.
    merged.ID = cl_id
    merged.new_segs = new_segs
    merged.new_lake = new_lakes
    merged.new_basins = new_basins
    merged.new_old_lakeflag = old_new_lakes

    # Updating tributary junctions across basins.
    merged.final_segs, merged.new_eps = cut_continental_tribs(merged)

###############################################################################

def combine_vals(merged, grwl, cnt):

    """
    FUNCTION:
        Appends current GRWL tile information to previously merged data. This
        function creates an object containing all the merged data for a
        region or continent.

    INPUTS
        merged -- Empty object to be filled.
        grwl -- Object containing attributes for the merged GRWL centerline.
        cnt -- Count used to specify what GRWL tile is being appended to the
               merged object.

    OUTPUTS
       merge.lon = Longitude (wgs84)
       merge.lat = Latitude (wgs84)
       merge.x = Easting (m)
       merge.y = Northing (m)
       merge.orbits = SWOT Orbits that intersect a point location
       merge.wth = Width (m)
       merge.nchan = Number of Channels
       merge.seg = Segment ID
       merge.Ind = Segment Point Index
       merge.dist = Segment Distance (m)
       merge.elv = Elevation (m)
       merge.facc_filt = Flow Accumulation (km^2)
       merge.new_lakes = Lake Flag based on Prior Lake Database
       merge.grand = GRaND ID
       merge.grod =  Binary flag indicating GROD locations
       merge.grod_fid = GROD ID
       merge.hfalls_fid = HydroFALLS ID
       merge.basins = HydroBASINS Code
       merge.delta = Binary flag indicating delta extents
       merge.manual = Binary flag indicating if a point was added to
                      original GRWL centerlines
       merge.eps = Endpoint locations for each segment
       merge.num_obs = Number of SWOT passes per 21-day orbit cycle
       merge.tile = GRWL tile name
       merge.old_lakes = Original GRWL Lake Flag
       merge.lake_id = Prior Lake Database ID
       merge.old_deltas = Unflitered delta extents
    """

    if cnt == 0:
        merged.x = np.copy(grwl.x)
        merged.y = np.copy(grwl.y)
        merged.wth = np.copy(grwl.wth)
        merged.nchan = np.copy(grwl.nchan)
        merged.seg = np.copy(grwl.seg)
        merged.Ind = np.copy(grwl.Ind)
        merged.dist = np.copy(grwl.dist)
        merged.lake = np.copy(grwl.new_lakes)
        merged.delta = np.copy(grwl.delta)
        merged.lon = np.copy(grwl.lon)
        merged.lat = np.copy(grwl.lat)
        merged.manual = np.copy(grwl.manual)
        merged.elv = np.copy(grwl.elv)
        merged.facc = np.copy(grwl.facc_filt)
        merged.grand = np.copy(grwl.grand)
        merged.grod = np.copy(grwl.grod)
        merged.grod_fid = np.copy(grwl.grod_fid)
        merged.hfalls_fid = np.copy(grwl.hfalls_fid)
        merged.basins = np.copy(grwl.basins)
        merged.eps = np.copy(grwl.eps)
        merged.num_obs = np.copy(grwl.num_obs)
        merged.orbits = np.copy(grwl.orbits)
        merged.tile = np.copy(grwl.tile)
        merged.lake_id = np.copy(grwl.lake_id)
        merged.old_lakeflag = np.copy(grwl.old_lakes)
        merged.old_deltas = np.copy(grwl.old_deltas)

    else:
        merged.x = np.insert(merged.x, len(merged.x), np.copy(grwl.x))
        merged.y = np.insert(merged.y, len(merged.y), np.copy(grwl.y))
        merged.wth = np.insert(merged.wth, len(merged.wth), np.copy(grwl.wth))
        merged.nchan = np.insert(merged.nchan, len(merged.nchan), np.copy(grwl.nchan))
        merged.seg = np.insert(merged.seg, len(merged.seg), np.copy(grwl.seg))
        merged.Ind = np.insert(merged.Ind, len(merged.Ind), np.copy(grwl.Ind))
        merged.dist = np.insert(merged.dist, len(merged.dist), np.copy(grwl.dist))
        merged.lake = np.insert(merged.lake, len(merged.lake), np.copy(grwl.new_lakes))
        merged.delta = np.insert(merged.delta, len(merged.delta), np.copy(grwl.delta))
        merged.lon = np.insert(merged.lon, len(merged.lon), np.copy(grwl.lon))
        merged.lat = np.insert(merged.lat, len(merged.lat), np.copy(grwl.lat))
        merged.manual = np.insert(merged.manual, len(merged.manual), np.copy(grwl.manual))
        merged.elv = np.insert(merged.elv, len(merged.elv), np.copy(grwl.elv))
        merged.facc = np.insert(merged.facc, len(merged.facc), np.copy(grwl.facc_filt))
        merged.grand = np.insert(merged.grand, len(merged.grand), np.copy(grwl.grand))
        merged.grod = np.insert(merged.grod, len(merged.grod), np.copy(grwl.grod))
        merged.grod_fid = np.insert(merged.grod_fid, len(merged.grod_fid), np.copy(grwl.grod_fid))
        merged.hfalls_fid = np.insert(merged.hfalls_fid, len(merged.hfalls_fid), np.copy(grwl.hfalls_fid))
        merged.basins = np.insert(merged.basins, len(merged.basins), np.copy(grwl.basins))
        merged.eps = np.insert(merged.eps, len(merged.eps), np.copy(grwl.eps))
        merged.num_obs = np.insert(merged.num_obs, len(merged.num_obs), np.copy(grwl.num_obs))
        merged.orbits = np.insert(merged.orbits, len(merged.orbits), np.copy(grwl.orbits), axis = 0)
        merged.tile = np.insert(merged.tile, len(merged.tile), np.copy(grwl.tile))
        merged.lake_id = np.insert(merged.lake_id, len(merged.lake_id), np.copy(grwl.lake_id))
        merged.old_lakeflag = np.insert(merged.old_lakeflag, len(merged.old_lakeflag), np.copy(grwl.old_lakes))
        merged.old_deltas = np.insert(merged.old_deltas, len(merged.old_deltas), np.copy(grwl.old_deltas))
