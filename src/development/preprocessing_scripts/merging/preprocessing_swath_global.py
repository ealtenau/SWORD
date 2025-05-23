# -*- coding: utf-8 -*-
"""
Created on Mon Apr 08 10:00:23 2019

Preprocee the tracks
Transform ESRI Shapefile track lines into ESRI Shapefiles Polygons

Created by C. Lion
Copyright (c) 2015 UNC Chapel Hill. All rights reserved.
"""

import os
main_dir = os.getcwd()
import sys
#import argparse
import glob
from osgeo import ogr
import numpy as np
#import tools_util as tu


def create_polygon(cliped_file, output_dir):
    '''
    Extract nodes from swath clipped shapefile
    and save it to polygon shapefile

	INPUTS
    cliped_file -- Track file cliped ESRI SHAPEFILE
    output_dir -- Directory to save polygons
    '''
    #log = open((output_dir+"poly_log.txt"), "w")
    #print(cliped_file)

    num_track = cliped_file.split("_")[-1][:-4]
    outputfile = os.path.join(output_dir, 'track_' + num_track + '.shp')

    # Opening -- Extracting the nodes

    #if not os.path.isfile(cliped_file):
        #tu.fatal_error('File %s does not exit' % cliped_file)

    driver = ogr.GetDriverByName('ESRI Shapefile')

    shape = driver.Open(cliped_file)
    inLayer = shape.GetLayer()
    feature = inLayer.GetFeature(0)
    numFeatures = inLayer.GetFeatureCount()
    if numFeatures == 0:
        print('**Track not over continent: ' + num_track)
        return
    
    geom = feature.GetGeometryRef()
    spatialRef = inLayer.GetSpatialRef()

    descrip = feature.GetField('descriptio')
    day_track = int(descrip.split(' ')[2][3:])
    time_track = descrip.split(' ')[3][:8]

    if geom.GetGeometryName() == 'LINESTRING':
        raise AttributeError

    if geom.GetGeometryCount() == 4:
        FR_L = geom.GetGeometryRef(0)
        FR_R = geom.GetGeometryRef(1)
        NR_L = geom.GetGeometryRef(2)
        NR_R = geom.GetGeometryRef(3)
        swath = 1
        
    elif geom.GetGeometryCount() == 2:
        FR_L = geom.GetGeometryRef(0)
        NR_L = geom.GetGeometryRef(1)
        FR_R = []
        NR_R = []
        swath = 1
  
    if geom.GetGeometryCount() > 4 and geom.GetGeometryCount() < 28:
        ind_save = []
        vals_save = []
        for ind in list(range(0, geom.GetGeometryCount())):
            line_temp = geom.GetGeometryRef(ind)
            wkt_line = line_temp.ExportToWkt()
            newgeom_line = ogr.CreateGeometryFromWkt(wkt_line)
            pt_line = newgeom_line.GetPoints()
            #print(len(pt_line))
            ind_save.append(ind)
            vals_save.append(len(pt_line))
            del line_temp, wkt_line, newgeom_line, pt_line
        ind_save = np.array(ind_save)
        vals_save = np.array(vals_save)    
        #print(len(ind_save))
        vals_sort = np.argsort(vals_save)    
        s1 = np.sort(vals_sort[-4::])
        
        FR_L = geom.GetGeometryRef(ind_save[s1[0]])
        FR_R = geom.GetGeometryRef(ind_save[s1[1]])
        NR_L = geom.GetGeometryRef(ind_save[s1[2]])
        NR_R = geom.GetGeometryRef(ind_save[s1[3]])
        swath = 1
        
    if geom.GetGeometryCount() == 28:
        FR_L = geom.GetGeometryRef(6)
        FR_R = geom.GetGeometryRef(7)
        NR_L = geom.GetGeometryRef(20)
        NR_R = geom.GetGeometryRef(21)
            
        FR_L2 = geom.GetGeometryRef(0)
        FR_R2 = geom.GetGeometryRef(13)
        NR_L2 = geom.GetGeometryRef(14)
        NR_R2 = geom.GetGeometryRef(27)
        swath = 2
        
    if geom.GetGeometryCount() == 1:
        raise AttributeError
        
    if geom.GetGeometryCount() == 0:
        print('**Track not over continent: ' + num_track)
        return
        
    ### NEEDS TO BE FIXED JUST DON'T KNOW HOW....
    optional_swaths = np.array([2,4,16,18,20,22,24,26,28])
    correct_swaths = np.where(optional_swaths == geom.GetGeometryCount())
        
    if len(correct_swaths[0]) == 0:
        print('**** Skipped for now due to ODD swaths: ' + num_track + ', cnt: ' + str(geom.GetGeometryCount()))
        return
    
    '''
    if geom.GetGeometryCount() > 4 and geom.GetGeometryCount() < 28:
        ind_save = []
        vals_save = []
        for ind in list(range(0, geom.GetGeometryCount())):
            line_temp = geom.GetGeometryRef(ind)
            wkt_line = line_temp.ExportToWkt()
            newgeom_line = ogr.CreateGeometryFromWkt(wkt_line)
            pt_line = newgeom_line.GetPoints()
            print(len(pt_line))
            if len(pt_line) > 4: #use to be > 2...
                ind_save.append(ind)
                vals_save.append(len(pt_line))
            del line_temp, wkt_line, newgeom_line, pt_line
        ind_save = np.array(ind_save)
        vals_save = np.array(vals_save)    
        #print(len(ind_save))
        vals_sort = np.argsort(vals_save)    
        s1 = np.sort(vals_sort[0:4])
        s2 = np.sort(vals_sort[4:9])  
        
        if len(ind_save) == 2:
            FR_L = geom.GetGeometryRef(ind_save[0])
            NR_L = geom.GetGeometryRef(ind_save[1])
            FR_R = []
            NR_R = []
            swath = 1
        if len(ind_save) == 4:
            FR_L = geom.GetGeometryRef(ind_save[0])
            FR_R = geom.GetGeometryRef(ind_save[1])
            NR_L = geom.GetGeometryRef(ind_save[2])
            NR_R = geom.GetGeometryRef(ind_save[3])
            swath = 1
            
        if len(ind_save) == 8:
            FR_L = geom.GetGeometryRef(ind_save[s1[0]])
            FR_R = geom.GetGeometryRef(ind_save[s1[1]])
            NR_L = geom.GetGeometryRef(ind_save[s1[2]])
            NR_R = geom.GetGeometryRef(ind_save[s1[3]])
            
            FR_L2 = geom.GetGeometryRef(ind_save[s2[0]])
            FR_R2 = geom.GetGeometryRef(ind_save[s2[1]])
            NR_L2 = geom.GetGeometryRef(ind_save[s2[2]])
            NR_R2 = geom.GetGeometryRef(ind_save[s2[3]])
            swath = 2
            
        if len(ind_save) == 1:
            raise AttributeError
        
        if len(ind_save) == 0:
            print('**Track not over continent: ' + num_track)
            return
        
        ### NEEDS TO BE FIXED JUST DON'T KNOW HOW....
        optional_swaths = np.array([2,4,8])
        correct_swaths = np.where(optional_swaths == len(ind_save))
        
        if len(correct_swaths[0]) == 0:
            print('**** Skipped for now due to ODD swaths: ' + num_track)
            return 
      '''  
        
    try:
        wkt_FR_L = FR_L.ExportToWkt()
        wkt_NR_L = NR_L.ExportToWkt()
        newgeom_FR_L = ogr.CreateGeometryFromWkt(wkt_FR_L)
        newgeom_NR_L = ogr.CreateGeometryFromWkt(wkt_NR_L)
        pt_FR_L = newgeom_FR_L.GetPoints()
        pt_NR_L = newgeom_NR_L.GetPoints()        
    except AttributeError:
        pt_FR_L = []
        pt_NR_L = []
    
    if swath == 2:    
        try:
            wkt_FR_L2 = FR_L2.ExportToWkt()
            wkt_NR_L2 = NR_L2.ExportToWkt()
            newgeom_FR_L2 = ogr.CreateGeometryFromWkt(wkt_FR_L2)
            newgeom_NR_L2 = ogr.CreateGeometryFromWkt(wkt_NR_L2)
            pt_FR_L2 = newgeom_FR_L2.GetPoints()
            pt_NR_L2 = newgeom_NR_L2.GetPoints()
        except AttributeError:
            pt_FR_L2 = []
            pt_NR_L2 = []
    
    try:
        wkt_FR_R = FR_R.ExportToWkt()
        wkt_NR_R = NR_R.ExportToWkt()
        newgeom_FR_R = ogr.CreateGeometryFromWkt(wkt_FR_R)
        newgeom_NR_R = ogr.CreateGeometryFromWkt(wkt_NR_R)
        pt_FR_R = newgeom_FR_R.GetPoints()
        pt_NR_R = newgeom_NR_R.GetPoints()
    except AttributeError:
        pt_FR_R = []
        pt_NR_R = []
    
    if swath == 2:    
        try:
            wkt_FR_R2 = FR_R2.ExportToWkt()
            wkt_NR_R2 = NR_R2.ExportToWkt()
            newgeom_FR_R2 = ogr.CreateGeometryFromWkt(wkt_FR_R2)
            newgeom_NR_R2 = ogr.CreateGeometryFromWkt(wkt_NR_R2)
            pt_FR_R2 = newgeom_FR_R2.GetPoints()
            pt_NR_R2 = newgeom_NR_R2.GetPoints()
        except AttributeError:
            pt_FR_R2 = []
            pt_NR_R2 = []

    if len(pt_FR_L) == 2 and len(pt_NR_L) == 2 and \
       len(pt_FR_R) == 2 and len(pt_NR_R) == 2:
        raise AttributeError

    if len(pt_FR_L) == 0 and \
       len(pt_FR_R) == 2 and len(pt_NR_R) == 2:
        raise AttributeError

    if len(pt_FR_R) == 0 and \
       len(pt_FR_L) == 2 and len(pt_NR_L) == 2:
        raise AttributeError

    #print("Create Polygon")
    #print(outputfile)
    
    #######################
    # Saving into polygons
    #######################
    
    if os.path.exists(outputfile):
        driver.DeleteDataSource(outputfile)

    try:
        outDS = driver.CreateDataSource(outputfile)
    except:
        print('Could not create file ' + outputfile)
        sys.exit(1)

    outLayer = outDS.CreateLayer(outputfile[:-4],
                                 spatialRef, geom_type=ogr.wkbMultiPolygon)

    # Define pixel attributes
    fieldDef0 = ogr.FieldDefn('Day', ogr.OFTInteger)
    fieldDef1 = ogr.FieldDefn('Time', ogr.OFTString)
    fieldDef2 = ogr.FieldDefn('Pass', ogr.OFTInteger)
    fieldDef3 = ogr.FieldDefn('Side', ogr.OFTString)
    outLayer.CreateField(fieldDef0)
    outLayer.CreateField(fieldDef1)
    outLayer.CreateField(fieldDef2)
    outLayer.CreateField(fieldDef3)

    # Get the FeatureDefn for the output layer
    featureDefn = outLayer.GetLayerDefn()

    #  Create a new feature
    outFeature = ogr.Feature(featureDefn)
    
    #######################
    # Create left swath
    #######################
    if len(pt_FR_L) > 0 or len(pt_NR_L) > 0 :
        left_swath = ogr.Geometry(ogr.wkbLinearRing)

        for ipx in range(len(pt_FR_L)):
            left_swath.AddPoint(pt_FR_L[ipx][0], pt_FR_L[ipx][1])

        for ipx in range(len(pt_NR_L)):
            indpx = -1 * ipx - 1
            left_swath.AddPoint(pt_NR_L[indpx][0], pt_NR_L[indpx][1])

        left_swath.CloseRings()

        # Create left polygon
        left_poly = ogr.Geometry(ogr.wkbPolygon)
        left_poly.AddGeometry(left_swath)
        outFeature.SetGeometry(left_poly)

        # Adding feature
        outFeature.SetField('Day', day_track)
        outFeature.SetField('Time', time_track)
        outFeature.SetField('Pass', int(num_track))
        outFeature.SetField('Side', 'left')

        outLayer.CreateFeature(outFeature)
        left_swath.Destroy()
    
    if swath == 2:    
        if len(pt_FR_L2) > 0 or len(pt_NR_L2) > 0 :
            left_swath2 = ogr.Geometry(ogr.wkbLinearRing)
        
            for ipx in range(len(pt_FR_L2)):
                left_swath2.AddPoint(pt_FR_L2[ipx][0], pt_FR_L2[ipx][1])
        
            for ipx in range(len(pt_NR_L2)):
                indpx = -1 * ipx - 1
                left_swath2.AddPoint(pt_NR_L2[indpx][0], pt_NR_L2[indpx][1])
        
            left_swath2.CloseRings()
        
            # Create left polygon
            left_poly2 = ogr.Geometry(ogr.wkbPolygon)
            left_poly2.AddGeometry(left_swath2)
            outFeature.SetGeometry(left_poly2)
        
            # Adding feature
            outFeature.SetField('Day', day_track)
            outFeature.SetField('Time', time_track)
            outFeature.SetField('Pass', int(num_track))
            outFeature.SetField('Side', 'left')
        
            outLayer.CreateFeature(outFeature)
            left_swath2.Destroy()
    
    #######################    
    ## Create right swath
    #######################
    if len(pt_FR_R) > 0 or len(pt_NR_R) > 0 :
        right_swath = ogr.Geometry(ogr.wkbLinearRing)
        for ipx in range(len(pt_FR_R)):
            right_swath.AddPoint(pt_FR_R[ipx][0], pt_FR_R[ipx][1])

        for ipx in range(len(pt_NR_R)):
            indpx = -1 * ipx - 1
            right_swath.AddPoint(pt_NR_R[indpx][0], pt_NR_R[indpx][1])

        right_swath.CloseRings()

        # Create right polygon
        right_poly = ogr.Geometry(ogr.wkbPolygon)
        right_poly.AddGeometry(right_swath)

        # Adding feature
        outFeature.SetGeometry(right_poly)
        outFeature.SetField('Day', day_track)
        outFeature.SetField('Time', time_track)
        outFeature.SetField('Pass', int(num_track))
        outFeature.SetField('Side', 'right')

        outLayer.CreateFeature(outFeature)
        right_swath.Destroy()
    
    if swath == 2:    
        if len(pt_FR_R2) > 0 or len(pt_NR_R2) > 0 :
            right_swath2 = ogr.Geometry(ogr.wkbLinearRing)
            for ipx in range(len(pt_FR_R2)):
                right_swath2.AddPoint(pt_FR_R2[ipx][0], pt_FR_R2[ipx][1])
        
            for ipx in range(len(pt_NR_R2)):
                indpx = -1 * ipx - 1
                right_swath2.AddPoint(pt_NR_R2[indpx][0], pt_NR_R2[indpx][1])
        
            right_swath2.CloseRings()
        
            # Create right polygon
            right_poly2 = ogr.Geometry(ogr.wkbPolygon)
            right_poly2.AddGeometry(right_swath2)
        
            # Adding feature
            outFeature.SetGeometry(right_poly2)
            outFeature.SetField('Day', day_track)
            outFeature.SetField('Time', time_track)
            outFeature.SetField('Pass', int(num_track))
            outFeature.SetField('Side', 'right')
        
            outLayer.CreateFeature(outFeature)
            right_swath2.Destroy()        
    
    #######################        
    # Close feature and shapefiles
    #######################
    outFeature.Destroy()
    shape.Destroy()
    outDS.Destroy()

    print("Polygon created: " + outputfile[94::])


def clip_track(filename, poly_shp, output_dir):
    '''
    Clips track that passes over continent
    Returns new file name

	INPUTS
    filename -- full track shapefile
    poly_shp -- polygon representing continent
    output_dir -- directory to save clipped file

	OUTPUT
    cliped_file -- clipped shapefile name
    '''

    cliped_file = os.path.join(output_dir,
                               'clip_pass_' + filename.split("_")[-1])

    os.system('ogr2ogr -fid 1 -clipsrc ' + poly_shp + ' ' +
              cliped_file + ' ' + filename)
    
    numFiles = len(glob.glob(os.path.join(output_dir, '*clip*')))
    
    if numFiles < 4:
        list_files = glob.glob(cliped_file[:-3] + '*')
        for file2rem in list_files:
            os.remove(file2rem)
        cliped_file = 'not_over_continent'
    
    '''
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(cliped_file)
    layer = ds.GetLayer()
    numFeatures = layer.GetFeatureCount()

    if numFeatures == 0:
        list_files = glob.glob(cliped_file[:-3] + '*')
        for file2rem in list_files:
            os.remove(file2rem)
        cliped_file = 'not_over_continent'
    '''
    
    return cliped_file


def main():
    '''
    Main program
    '''
    
    track_dir = main_dir+'/data/inputs/SWOT_Tracks/lines/'
    poly_shp = main_dir+'/data/inputs/SWOT_Tracks/swot_science_hr_2.0s_4.0s_Aug2020-v4_nadir.shp'
    output_dir = main_dir+'/data/inputs/SWOT_Tracks/2020_orbits/'
    
    list_files = glob.glob(os.path.join(track_dir, '*.shp'))
    if not list_files:
        print('Cannot find tracks files in folder %s' % track_dir)
        sys.exit(1)

    list_files.sort()

    for filename in list_files:
        cliped_file = clip_track(filename, poly_shp, output_dir)

        if cliped_file != 'not_over_continent':
            try:
                create_polygon(cliped_file, output_dir)
            except AttributeError:
                print('no polygon')
    
    del_files = glob.glob(os.path.join(output_dir, '*clip*'))
    for d in del_files:
        os.remove(d)
        
    print("End program")

if __name__ == '__main__':

    main()

    
    
#len(glob.glob(os.path.join(output_dir, '*clip*')))
