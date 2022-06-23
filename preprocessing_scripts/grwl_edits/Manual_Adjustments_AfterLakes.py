# -*- coding: utf-8 -*-
"""
Created on Thu May 28 09:58:21 2020

@author: ealtenau
"""
from __future__ import division
import os
import time
import utm
from osgeo import ogr
from osgeo import osr
import numpy as np
import pandas as pd
from scipy import spatial as sp
import sys
import matplotlib.pyplot as plt


###############################################################################    

class Object(object):
    """
    FUNCTION:
        Creates class object to assign attributes to.
    """
    pass 

###############################################################################

def open_grwl(filename):
    
    """
    FUNCTION:
        Opens Edited GRWL file and returns the fields in array format.

    INPUTS
        filename -- GRWL shapefile

    OUTPUTS
        grwl.longitude -- Longitude (wgs84)
        grwl.latitude -- Latitude (wgs84)
        grwl.wth -- Width (m)
        grwl.seg -- Segment ID
        grwl.nchan -- Number of Channels
        grwl.lake -- Lake flag based off of prior lake database
        grwl.east -- Easting (utm)
        grwl.north -- Northing (utm)
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
    grwl.east = np.zeros(numFeatures)
    grwl.north = np.zeros(numFeatures)
    grwl.wth = np.zeros(numFeatures)
    grwl.nchan = np.zeros(numFeatures)
    grwl.seg = np.zeros(numFeatures)
    grwl.Ind = np.zeros(numFeatures)
    grwl.lake = np.zeros(numFeatures)
    grwl.longitude = np.zeros(numFeatures)
    grwl.latitude = np.zeros(numFeatures)
    grwl.manual = np.zeros(numFeatures)
    grwl.eps = np.zeros(numFeatures)
    grwl.old_lakes = np.zeros(numFeatures)
    grwl.lake_id = np.zeros(numFeatures)
    
    # Saving data to arrays.
    if 'manual_add' in attributes:

        cnt = 0
        for feature in range(numFeatures):
            grwl.east[cnt] = layer.GetFeature(feature).GetField('utm_east')
            grwl.north[cnt] = layer.GetFeature(feature).GetField('utm_north')
            grwl.wth[cnt] = layer.GetFeature(feature).GetField('width_m')
            grwl.nchan[cnt] = layer.GetFeature(feature).GetField('nchannels')
            grwl.seg[cnt] = layer.GetFeature(feature).GetField('segmentID')
            grwl.Ind[cnt] = layer.GetFeature(feature).GetField('segmentInd')
            grwl.lake[cnt] = layer.GetFeature(feature).GetField('lakeFlag')
            grwl.longitude[cnt] = layer.GetFeature(feature).GetField('lon')
            grwl.latitude[cnt] = layer.GetFeature(feature).GetField('lat')
            grwl.manual[cnt] = layer.GetFeature(feature).GetField('manual_add')
            grwl.eps[cnt] = layer.GetFeature(feature).GetField('endpoints')
            grwl.old_lakes[cnt] = layer.GetFeature(feature).GetField('old_lakes')
            grwl.lake_id[cnt] = layer.GetFeature(feature).GetField('lake_id')
            cnt += 1

    else:

      cnt = 0
      for feature in range(numFeatures):
          grwl.east[cnt] = layer.GetFeature(feature).GetField('utm_east')
          grwl.north[cnt] = layer.GetFeature(feature).GetField('utm_north')
          grwl.wth[cnt] = layer.GetFeature(feature).GetField('width_m')
          grwl.nchan[cnt] = layer.GetFeature(feature).GetField('nchannels')
          grwl.seg[cnt] = layer.GetFeature(feature).GetField('segmentID')
          grwl.Ind[cnt] = layer.GetFeature(feature).GetField('segmentInd')
          grwl.lake[cnt] = layer.GetFeature(feature).GetField('lakeFlag')
          grwl.longitude[cnt] = layer.GetFeature(feature).GetField('lon')
          grwl.latitude[cnt] = layer.GetFeature(feature).GetField('lat')
          grwl.old_lakes[cnt] = layer.GetFeature(feature).GetField('old_lakes')
          grwl.lake_id[cnt] = layer.GetFeature(feature).GetField('lake_id')
          grwl.manual[cnt] = 0
          grwl.eps[cnt] = 0
          cnt += 1

    return grwl
    
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
    fieldDef12 = ogr.FieldDefn('lake_id', ogr.OFTString)
    fieldDef13 = ogr.FieldDefn('old_lakes', ogr.OFTInteger)
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

    # Create feature (point) to store pixel
    floutDefn = layerout.GetLayerDefn()
    feature_out = ogr.Feature(floutDefn)

    for ipix in range(len(grwl.latitude)):
        # Create Geometry Point with pixel coordinates
        pixel_point = ogr.Geometry(ogr.wkbPoint)
        pixel_point.AddPoint(grwl.longitude[ipix], grwl.latitude[ipix])
        # Add the geometry to the feature
        feature_out.SetGeometry(pixel_point)
        # Set feature attributes
        feature_out.SetField('lon', grwl.longitude[ipix])
        feature_out.SetField('lat', grwl.latitude[ipix])
        feature_out.SetField('utm_east', grwl.east[ipix])
        feature_out.SetField('utm_north', grwl.north[ipix])
        # int() is needed because facc.dtype=float64, needs to be saved with 
        # all values whereas lat.dtype=float64, which raise an error.
        feature_out.SetField('segmentInd', int(grwl.Ind[ipix]))
        feature_out.SetField('segmentID', int(grwl.seg[ipix]))
        feature_out.SetField('width_m', int(grwl.wth[ipix]))
        feature_out.SetField('lakeFlag', int(grwl.lake[ipix]))
        feature_out.SetField('nchannels', int(grwl.nchan[ipix]))
        feature_out.SetField('manual_add', int(grwl.manual[ipix]))
        feature_out.SetField('endpoints', int(grwl.eps[ipix]))
        feature_out.SetField('lake_id', str(int(grwl.lake_id[ipix])))
        feature_out.SetField('old_lakes', int(grwl.old_lakes[ipix]))
        
        # Add the feature to the layer
        layerout.CreateFeature(feature_out)
        # Delete point geometry
        pixel_point.Destroy()

    # Close feature and shapefiles
    feature_out.Destroy()
    dataout.Destroy()

    return fshpout

###############################################################################
###############################################################################
###############################################################################

fn_grwl = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/inputs/GRWL/Edits_PLD/SA/s04w066_edit_lakes.shp'
grwl = open_grwl(fn_grwl)

s3 = np.where(grwl.seg == 228)[0]
org_ind3 = np.arange(1,510,1)
matches3 = np.array([int(np.where(grwl.Ind[s3] == org_ind3[i])[0]) for i in list(range(len(org_ind3)))])
vals3 = np.arange(52,561,1)
grwl.Ind[s3[matches3]] = vals3

s2 = np.where(grwl.seg == 311)[0]
org_ind2 = np.arange(963,1014,1)
matches2 = np.array([int(np.where(grwl.Ind[s2] == org_ind2[i])[0]) for i in list(range(len(org_ind2)))])
vals2 = np.arange(1,52,1)[::-1]
grwl.Ind[s2[matches2]] = vals2
grwl.seg[s2[matches2]] = 228

s4 = np.where(grwl.seg == 245)[0]
org_ind4 = np.arange(0,376,1)
matches4 = np.array([int(np.where(grwl.Ind[s4] == org_ind4[i])[0]) for i in list(range(len(org_ind4)))])
vals4 = np.arange(49,425,1)
grwl.Ind[s4[matches4]] = vals4

s5 = np.where(grwl.seg == 311)[0]
org_ind5 = np.arange(1014,1062,1)
matches5 = np.array([int(np.where(grwl.Ind[s5] == org_ind5[i])[0]) for i in list(range(len(org_ind5)))])
vals5 = np.arange(1,49,1)
grwl.Ind[s5[matches5]] = vals5
grwl.seg[s5[matches5]] = 245

s6 = np.where(grwl.seg == 314)[0]
org_ind6 = np.arange(10,47,1)
matches6 = np.array([int(np.where(grwl.Ind[s6] == org_ind6[i])[0]) for i in list(range(len(org_ind6)))])
vals6 = np.arange(425,462,1)[::-1]
grwl.Ind[s6[matches6]] = vals6
grwl.seg[s6[matches6]] = 245

s7 = np.where(grwl.seg == 314)[0]
org_ind7 = np.arange(1,10,1)
matches7 = np.array([int(np.where(grwl.Ind[s7] == org_ind7[i])[0]) for i in list(range(len(org_ind7)))])
vals7 = np.arange(307,316,1)[::-1]
grwl.Ind[s7[matches7]] = vals7

s8 = np.where(grwl.seg == 314)[0]
org_ind8 = np.arange(276,316,1)
matches8 = np.array([int(np.where(grwl.Ind[s8] == org_ind8[i])[0]) for i in list(range(len(org_ind8)))])
vals8 = np.arange(6,46,1)[::-1]
grwl.Ind[s8[matches8]] = vals8

s9 = np.where(grwl.seg == 311)[0]
org_ind9 = np.arange(1,17,1)
matches9 = np.array([int(np.where(grwl.Ind[s9] == org_ind9[i])[0]) for i in list(range(len(org_ind9)))])
vals9 = np.arange(632,648,1)
grwl.Ind[s9[matches9]] = vals9
grwl.seg[s9[matches9]] = 312

s10 = np.where(grwl.seg == 383)[0]
org_ind10 = np.arange(1,142,1)
matches10 = np.array([int(np.where(grwl.Ind[s10] == org_ind10[i])[0]) for i in list(range(len(org_ind10)))])
vals10 = np.arange(60,201,1)
grwl.Ind[s10[matches10]] = vals10

s11 = np.where(grwl.seg == 316)[0]
org_ind11 = np.arange(251,310,1)
matches11 = np.array([int(np.where(grwl.Ind[s11] == org_ind11[i])[0]) for i in list(range(len(org_ind11)))])
vals11 = np.arange(1,60,1)
grwl.Ind[s11[matches11]] = vals11
grwl.seg[s11[matches11]] = 383

s12 = np.where(grwl.seg == 310)[0]
org_ind12 = np.arange(1,28,1)
matches12 = np.array([int(np.where(grwl.Ind[s12] == org_ind12[i])[0]) for i in list(range(len(org_ind12)))])
vals12 = np.arange(963,990,1)
grwl.Ind[s12[matches12]] = vals12
grwl.seg[s12[matches12]] = 311

s13 = np.where(grwl.seg == 311)[0]
org_ind13 = np.arange(858,990,1)
matches13 = np.array([int(np.where(grwl.Ind[s13] == org_ind13[i])[0]) for i in list(range(len(org_ind13)))])
grwl.seg[s13[matches13]] = np.max(grwl.seg)+1

s14 = np.where(grwl.seg == 311)[0]
org_ind14 = np.arange(423,858,1)
matches14 = np.array([int(np.where(grwl.Ind[s14] == org_ind14[i])[0]) for i in list(range(len(org_ind14)))])
grwl.seg[s14[matches14]] = np.max(grwl.seg)+1

s15 = np.where(grwl.seg == 204)[0]
org_ind15 = np.arange(554,621,1)
matches15 = np.array([int(np.where(grwl.Ind[s15] == org_ind15[i])[0]) for i in list(range(len(org_ind15)))])
grwl.seg[s15[matches15]] = np.max(grwl.seg)+1




outfile = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/inputs/GRWL/Edits_PLD/test/s04w066_edit_lakes.shp'
save_grwl_edits(grwl, outfile)

'''
test = np.where(grwl.seg == 311)[0]
len(test)
len(np.unique(grwl.Ind[test]))
plt.scatter(grwl.east[test], grwl.north[test], c = grwl.Ind[test], s = 5, edgecolors = None)
'''