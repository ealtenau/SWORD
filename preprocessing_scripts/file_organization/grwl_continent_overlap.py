# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 08:53:34 2019

@author: ealtenau
"""
import os
from osgeo import ogr
import numpy as np
import pandas as pd

###############################################################################
###############################################################################

def getListOfFiles(dirName):
    '''
    For the given path, get the List of all files in the directory tree
    '''
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

def open_grwl(filename):

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
        
    # Createing empty arrays to fill in with grwl attributes
    lon = np.zeros(numFeatures)
    lat = np.zeros(numFeatures)

    # Saving
    cnt = 0
    for feature in range(numFeatures):
        lon[cnt] = layer.GetFeature(feature).GetField(attributes[7])
        lat[cnt] = layer.GetFeature(feature).GetField(attributes[8])
        #elev_m[cnt] = layer.GetFeature(feature).GetField(attributes[9])
        cnt += 1

    return lon, lat

###############################################################################
###############################################################################

grwl_paths = [file for file in getListOfFiles('C:/Users/ealtenau/Documents/Research/SWAG/GRWL/GRWL_vector_V01.01_LatLonNames/') if '.shp' in file]
cont_file = 'C:/Users/ealtenau/Documents/Research/SWAG/ArcFiles/continent.shp'

continents = []           
for ind in list(range(len(grwl_paths))):
    lon, lat = open_grwl(grwl_paths[ind])
    
    if len(lon) == 0:
        print('No GRWL Data: ' + grwl_paths[ind][-11:-4])
        continents.append('No GRWL Data')
        continue
        
    poly1 = ogr.Geometry(ogr.wkbLinearRing)
    poly1.AddPoint(min(lon), max(lat))
    poly1.AddPoint(min(lon), min(lat))
    poly1.AddPoint(max(lon), min(lat))
    poly1.AddPoint(max(lon), max(lat))
    poly1.AddPoint(min(lon), max(lat))
    grwlGeometry = ogr.Geometry(ogr.wkbPolygon)
    grwlGeometry.AddGeometry(poly1) 
    
    cont_temp = []
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shape = driver.Open(cont_file)
    inLayer = shape.GetLayer()
    for feature in inLayer: #inLayer is always of size one because polygon is a unique value
        cont=feature.GetGeometryRef()
        answer = cont.Intersects(grwlGeometry)
        #print(answer, feature.GetField('CONTINENT'))
        if answer == True:
            cont_temp.append(feature.GetField('CONTINENT'))
    
    continents.append(cont_temp)
    print(ind)
       
grwl_tiles = []
for ind in list(range(len(grwl_paths))):
    grwl_tiles.append(grwl_paths[ind][-11:-4])

grwl_tiles = np.asarray(grwl_tiles)
continents = np.asarray(continents, dtype=object) 
 
new_array = np.zeros([len(grwl_tiles),2], dtype='S60')    
new_array[:,0] = grwl_tiles
for idx in list(range(len(continents))):
    text = str(continents[idx])
    text = text[2:-2]
    new_array[idx,1] = text
        
header=['tile','continent']
df=pd.DataFrame(new_array, columns=header)
df.to_csv('C:/Users/ealtenau/Documents/Research/SWAG/GRWL/GRWL_Tiles_Continents.csv')         
         
         
