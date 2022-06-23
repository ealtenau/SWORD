# -*- coding: utf-8 -*-
"""
Created on Sun Feb 03 12:58:11 2019

@author: ealtenau
"""
###############################################################################
################################# Packages ####################################
###############################################################################    

from __future__ import division
import os
import shutil
import time
import utm
from osgeo import ogr
from osgeo import osr
import numpy as np
import geopandas as gp
#from shapely.geometry import Point 
import pandas as pd
#import gdal
#import glob
import matplotlib.pyplot as plt
from scipy import spatial as sp
#import math
#import random
import sys

###############################################################################
################################# Functions ###################################
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

###############################################################################
    
def attach_basins(fn_grwl, fn_basins):
    ### attaching basin codes
    points = gp.GeoDataFrame.from_file(fn_grwl) 
    poly = gp.GeoDataFrame.from_file(fn_basins)
    intersect = gp.sjoin(poly, points, how="inner", op='intersects')
    intersect = pd.DataFrame(intersect)
    intersect = intersect.drop_duplicates(subset='index_right', keep='first')
        
    ids = np.array(intersect.index_right)
    basin_code = np.zeros(len(points))
    basin_code[ids] = np.array(intersect.PFAF_ID)
    
    return basin_code

###############################################################################
    
def read_grwl(filename):
    fn_grwl = filename 
    driver = ogr.GetDriverByName('ESRI Shapefile')   
    shape = driver.Open(fn_grwl)
    layer = shape.GetLayer()
    numFeatures = layer.GetFeatureCount()
    #spatialRef = layer.GetSpatialRef()
        
    attributes = []
    ldefn = layer.GetLayerDefn()
    for n in range(ldefn.GetFieldCount()):
        fdefn = ldefn.GetFieldDefn(n)
        attributes.append(fdefn.name)
    #print attributes
        
    # Createing empty arrays to fill in with grwl attributes
    utm_east = np.zeros(numFeatures)
    utm_north = np.zeros(numFeatures)
    width_m = np.zeros(numFeatures)
    nchannels = np.zeros(numFeatures)
    segmentID = np.zeros(numFeatures)
    segmentInd = np.zeros(numFeatures)
    lakeFlag = np.zeros(numFeatures)
    lon = np.zeros(numFeatures)
    lat = np.zeros(numFeatures)
    elev_m = np.zeros(numFeatures)
    
    # Saving 
    cnt = 0
    for feature in range(numFeatures):
        utm_east[cnt] = layer.GetFeature(feature).GetField(attributes[0])
        utm_north[cnt] = layer.GetFeature(feature).GetField(attributes[1])
        width_m[cnt] = layer.GetFeature(feature).GetField(attributes[2])
        nchannels[cnt] = layer.GetFeature(feature).GetField(attributes[3])
        segmentID[cnt] = layer.GetFeature(feature).GetField(attributes[4])
        segmentInd[cnt] = layer.GetFeature(feature).GetField(attributes[5])
        lakeFlag[cnt] = layer.GetFeature(feature).GetField(attributes[6])
        lon[cnt] = layer.GetFeature(feature).GetField(attributes[7])
        lat[cnt] = layer.GetFeature(feature).GetField(attributes[8])
        elev_m[cnt] = layer.GetFeature(feature).GetField(attributes[9])
        cnt += 1
        
    return utm_east, utm_north, width_m, nchannels, segmentID, segmentInd, lakeFlag, lon, lat

###############################################################################

def changing_segID(grwl_id, grwl_x, grwl_y):
    
    """
    Renumbers GRWL segment ID's within a GFDM tile. Not every GRWL segment has a 
    unique segment ID within a GFDM tile extent. This step is needed to properly 
    interpolate the flow accumulation along a single river segment (See XXX function
    in SWAG_tools.py.
    """     
    new_grwl_id = np.zeros(len(grwl_id))
    unq_grwl_ids = np.unique(grwl_id)
    count = 0
    
    for num in range(len(unq_grwl_ids)):
        seg = np.where(grwl_id == unq_grwl_ids[num])  ### with grwl edits may need to change this to the minimum value....
        dist = np.sqrt((grwl_x[seg][0]-grwl_x[seg])**2 + (grwl_y[seg][0]-grwl_y[seg])**2)
        dist_diff = np.diff(dist)
        seg_divs = list(np.where(abs(dist_diff) > 100)[0]+1)
        
        if len(seg_divs) == 1:
            new_grwl_id[seg[0][0:int(seg_divs[0])]] = count
            new_grwl_id[seg[0][int(seg_divs[0]):len(seg[0])]] = count+1
            count+=2
        
        # THIS LOOP WORKS ALONE
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

def find_endpoints(edits_seg, edits_x, edits_y, r):
    #identifying endpoints of new segments...
    endpoints = np.zeros(len(edits_seg))
    uniq_segs = np.unique(edits_seg)
    for ind in list(range(len(uniq_segs))):
        seg = np.where(edits_seg == uniq_segs[ind])[0]
        seg_x = edits_x[seg]
        seg_y = edits_y[seg]
        
        count = 1
        for idx in list(range(len(seg))):
            d = np.sqrt((seg_x[idx]-seg_x)**2 + (seg_y[idx]-seg_y)**2)
            if len(np.where(d < r)[0]) < 3:
                endpoints[seg[idx]] = count
                count = count+1
    
    return endpoints

###############################################################################

def find_endpoints2(edits_seg, edits_segInd):
    eps = np.zeros(len(edits_seg))
    uniq_segs = np.unique(edits_seg)    
    for ind in list(range(len(uniq_segs))):
        seg = np.where(edits_seg == uniq_segs[ind])[0]
        eps[seg[np.where(edits_segInd[seg] == np.min(edits_segInd[seg]))]] = 1
        eps[seg[np.where(edits_segInd[seg] == np.max(edits_segInd[seg]))]] = 2
    return eps        

###############################################################################    

def order_segID(edits_seg, endpoints, edits_x, edits_y):
    #creating a segment id in order of flow distance...
    edits_segInd = np.zeros(len(edits_seg))
    edits_segDist = np.zeros(len(edits_seg))
    uniq_segs = np.unique(edits_seg)
    for ind in list(range(len(uniq_segs))):  
        #print(ind, uniq_segs[ind])
        seg = np.where(edits_seg == uniq_segs[ind])[0]
        seg_x = edits_x[seg]
        seg_y = edits_y[seg]
        eps = np.where(endpoints[seg] > 0)[0]
        edits_segInd[seg[eps[0]]]=1
        edits_segDist[seg[eps[0]]]=0
        idx = eps[0] 
        count = 2   
        while np.min(edits_segInd[seg]) == 0:
            d = np.sqrt((seg_x[idx]-seg_x)**2 + (seg_y[idx]-seg_y)**2)
            dzero = np.where(edits_segInd[seg] == 0)[0]
            #vals = np.where(edits_segInd[dzero] eq 0)
            next_pt = dzero[np.where(d[dzero] == np.min(d[dzero]))[0]]
            edits_segInd[seg[next_pt]] = count
            edits_segDist[seg[next_pt]] = d[next_pt]
            count = count+1
            idx = next_pt
            #print(next_pt)
            
    return edits_segInd

###############################################################################

def edit_short_segments(endpoints, edits_seg, edits_segIND, edits_x, edits_y, 
                        edits_lake, utm_east, utm_north, segmentID, 
                        segmentInd, width_m):
    # Attaching short edit segments to existing GRWL segments.    
    uniq_segs = np.unique(edits_seg)
    edits_segInd = np.copy(edits_segIND)
    
    for ind in list(range(len(uniq_segs))):
        #print(ind)
        seg = np.where(edits_seg == uniq_segs[ind])[0]
        lakes = len(np.where(edits_lake[seg] > 0)[0])       
        length = len(seg)
        #print(length)
        
        if length < 100 and lakes == 0:
            
            ### new as of 5/8/19
            edits_width = np.array(np.repeat(1, len(edits_seg)))
            edits_x2 = np.delete(edits_x, seg)
            edits_y2 = np.delete(edits_y, seg)
            edits_width2 = np.delete(edits_width, seg)
            edits_seg2 = np.delete(edits_seg, seg)
            edits_segInd2 = np.delete(edits_segInd, seg)
            
            ###
            
            grwl_pts = np.vstack((utm_east, utm_north)).T
            edit_pts = np.vstack((edits_x2, edits_y2)).T
            if len(seg) == 1:
                eps = np.array([0,0])
            else:
                eps = np.where(endpoints[seg] > 0)[0]
            ep_pts = np.vstack((edits_x[seg[eps]], edits_y[seg[eps]])).T
            kdt = sp.cKDTree(grwl_pts)
            
            if len(edit_pts) > 0:
                kdt2 = sp.cKDTree(edit_pts)
            
            if len(seg) == 1:
                pt_dist, pt_ind = kdt.query(ep_pts, k = 1) #use to be k = 2 before I put condition in for eps...
                remv = np.where(pt_dist > 500)[0]
                if len(remv) > 0:
                    pt_dist = np.delete(pt_dist, remv)
                    pt_ind = np.delete(pt_ind, remv)
                if len(edit_pts) > 0:
                    pt_dist2, pt_ind2 = kdt2.query(ep_pts, k = 1) #use to be k = 2 before I put condition in for eps...
                    remv2 = np.where(pt_dist2 > 500)[0]
                    if len(remv2) > 0:
                        pt_dist2 = np.delete(pt_dist2, remv2)
                        pt_ind2 = np.delete(pt_ind2, remv2)
                    dist_all = np.insert(pt_dist, len(pt_dist), pt_dist2)
            else:
                pt_dist, pt_ind = kdt.query(ep_pts, k = 1)
                remv = np.where(pt_dist > 500)[0]
                if len(remv) > 0:
                    pt_dist = np.delete(pt_dist, remv)
                    pt_ind = np.delete(pt_ind, remv)
                if len(edit_pts) > 0:
                    pt_dist2, pt_ind2 = kdt2.query(ep_pts, k = 1)
                    remv2 = np.where(pt_dist2 > 500)[0]
                    if len(remv2) > 0:
                        pt_dist2 = np.delete(pt_dist2, remv2)
                        pt_ind2 = np.delete(pt_ind2, remv2)
                    
                    if len(pt_ind2) > 0:
                        dist_all = np.insert(pt_dist, len(pt_dist), pt_dist2)
                        edt_segs = np.unique(edits_seg2[pt_ind2])
                    
                    if len(pt_ind2) == 0: ## indented these lines were on same indent as grwl_segs. 
                        dist_all = pt_dist
                        edt_segs = []
                    
                    if len(pt_ind2) > 0 and np.min(pt_dist2) > 100:
                        dist_all = pt_dist
                        edt_segs = []
            
            if len(edit_pts) == 0:
                dist_all = pt_dist
                edt_segs = []

            grwl_segs = segmentID[pt_ind]
            if len(np.unique(grwl_segs)) == 1:
                grwl_segs = np.array([segmentID[pt_ind[0]]])
            
            ### No segments found 
            if len(grwl_segs) == 0 and len(edt_segs) == 0:
                #print(str(uniq_segs[ind]) + ': Short edit segment with no bounding segments.')
                continue
            
            ### Segments are too far away
            if len(dist_all) == 0: #use to be: np.min(dist_all) > 500
                #print(str(uniq_segs[ind]) + ': Short edit segment with no bounding GRWL segments.')
                continue
            
            ### Multiple grwl segments, but no edit_segs
            if len(np.unique(grwl_segs)) > 1 and len(edt_segs) == 0:
                wth = []
                for idx in list(range(len(grwl_segs))):
                    wth.append(np.max(width_m[np.where(segmentID == grwl_segs[idx])]))
                min_wth = np.where(wth == np.min(wth))[0][0]            
                edits_seg[seg] = grwl_segs[min_wth]
                seg_start_id = np.min(segmentInd[np.where(segmentID == grwl_segs[min_wth])])
                seg_end_id = np.max(segmentInd[np.where(segmentID == grwl_segs[min_wth])])
                seg_ends = np.array([seg_start_id, seg_end_id])
                seg_diff = abs(segmentInd[pt_ind[min_wth]] - seg_ends)
                seg_match = np.where(seg_diff == np.min(seg_diff))[0]
                if seg_match == 1 and edits_segInd[seg[eps[min_wth]]] == 1:
                    start_id = seg_end_id
                    edits_segInd[seg] = edits_segInd[seg]+start_id                
                elif seg_match == 1 and edits_segInd[seg[eps[min_wth]]] > 1:
                    start_id = seg_end_id+1
                    edits_segInd[seg] = (abs(edits_segInd[seg]-np.max(edits_segInd[seg]))+1) + start_id  
                if seg_match == 0 and edits_segInd[seg[eps[min_wth]]] == 1:   
                    start_id = seg_start_id-edits_segInd[seg]
                    edits_segInd[seg] = start_id                
                elif seg_match == 0 and edits_segInd[seg[eps[min_wth]]] > 1:
                    start_id = seg_start_id
                    edits_segInd[seg] = start_id-(abs(edits_segInd[seg]-np.max(edits_segInd[seg]))+1)
                    
            ### Only one grwl segment and no edit segments
            if len(np.unique(grwl_segs)) == 1 and len(edt_segs) == 0:
                edits_seg[seg] = np.unique(grwl_segs)
                seg_min_id = np.min(segmentInd[pt_ind])
                seg_max_id = np.max(segmentInd[pt_ind])
                seg_max_all = np.max(segmentInd[np.where(segmentID == np.unique(grwl_segs))])
                seg_min_all = np.min(segmentInd[np.where(segmentID == np.unique(grwl_segs))])
                alter_vals = np.where(segmentInd[np.where(segmentID == np.unique(grwl_segs))] >= seg_max_id)
                
                ''' 
                # If there is only one point.         
                if len(seg) == 1:
                    edits_segInd[seg] = edits_segInd[seg]+seg_max_id
                    segmentInd[alter_vals] = segmentInd[alter_vals]+len(seg)+1
                '''
                
                # if the edit segment is on the end.
                if len(np.where(dist_all < 200)[0]) == 1:
                    closest_id = segmentInd[pt_ind[np.where(dist_all < 200)[0]]]
                    if closest_id > 10 and edits_segInd[seg[eps[0]]] == 1:
                        edits_segInd[seg] = edits_segInd[seg]+seg_max_all
                    elif closest_id > 10 and edits_segInd[seg[eps[0]]] > 1:
                        edits_segInd[seg] = (abs(edits_segInd[seg]-np.max(edits_segInd[seg]))+1)+seg_max_all
                    if closest_id < 10 and edits_segInd[seg[eps[0]]] == 1:
                        edits_segInd[seg] = seg_min_all-edits_segInd[seg]
                    elif closest_id < 10 and edits_segInd[seg[eps[0]]] > 1:
                        edits_segInd[seg] = seg_min_all-(abs(edits_segInd[seg]-np.max(edits_segInd[seg]))+1)
                
                #if the segment is in between grwl segments. 
                else:
                    if segmentInd[pt_ind[0]] < segmentInd[pt_ind[1]] and endpoints[eps[0]] < endpoints[eps[1]]:
                        edits_segInd[seg] = edits_segInd[seg]+seg_min_id
                        segmentInd[alter_vals] = segmentInd[alter_vals]+len(seg)
                    elif segmentInd[pt_ind[0]] < segmentInd[pt_ind[1]] and endpoints[eps[0]] > endpoints[eps[1]]:
                        edits_segInd[seg] = (abs(edits_segInd[seg]-np.max(edits_segInd[seg]))+1)+(seg_min_id)
                        segmentInd[alter_vals] = segmentInd[alter_vals]+len(seg)
                    if segmentInd[pt_ind[0]] > segmentInd[pt_ind[1]] and endpoints[eps[0]] < endpoints[eps[1]]:
                        edits_segInd[seg] = (abs(edits_segInd[seg]-np.max(edits_segInd[seg]))+1)+(seg_min_id)
                        segmentInd[alter_vals] = segmentInd[alter_vals]+len(seg)  
                    elif segmentInd[pt_ind[0]] > segmentInd[pt_ind[1]] and endpoints[eps[0]] > endpoints[eps[1]]:
                        edits_segInd[seg] = edits_segInd[seg]+seg_min_id
                        segmentInd[alter_vals] = segmentInd[alter_vals]+len(seg)                   

            ### grwl and edit segments            
            if len(grwl_segs) > 0 and len(edt_segs) > 0:
                wth = np.zeros(len(grwl_segs))
                wth2 = np.zeros(len(edt_segs))
                for idx in list(range(len(grwl_segs))):
                    wth[idx] = np.max(width_m[np.where(segmentID == grwl_segs[idx])])
                for idz in list(range(len(edt_segs))):    
                    wth2[idz] = np.max(edits_width2[np.where(edits_seg2 == edt_segs[idz])])
                
                min_wth = np.where(wth == np.min(wth))[0][0]
                min_wth2 = np.where(wth2 == np.min(wth2))[0][0]
                
                if wth[min_wth] < wth2[min_wth2]:

                    edits_seg[seg] = grwl_segs[min_wth]
                    seg_start_id = np.min(segmentInd[np.where(segmentID == grwl_segs[min_wth])])
                    seg_end_id = np.max(segmentInd[np.where(segmentID == grwl_segs[min_wth])])
                    seg_ends = np.array([seg_start_id, seg_end_id])
                    seg_diff = abs(segmentInd[pt_ind[min_wth]] - seg_ends)
                    seg_match = np.where(seg_diff == np.min(seg_diff))[0]
                    if seg_match == 1 and edits_segInd[seg[eps[min_wth]]] == 1:
                        start_id = seg_end_id+1
                        edits_segInd[seg] = edits_segInd[seg]+start_id
                    elif seg_match == 1 and edits_segInd[seg[eps[min_wth]]] > 1:
                        start_id = seg_end_id+1
                        edits_segInd[seg] = (abs(edits_segInd[seg]-np.max(edits_segInd[seg]))+1) + start_id 
                    if seg_match == 0 and edits_segInd[seg[eps[min_wth]]] == 1:
                        start_id = seg_start_id-edits_segInd[seg]
                        edits_segInd[seg] = start_id
                    elif seg_match == 0 and edits_segInd[seg[eps[min_wth]]] > 1:
                        start_id = seg_start_id
                        edits_segInd[seg] = start_id-(abs(edits_segInd[seg]-np.max(edits_segInd[seg]))+1)                   
            
                if wth[min_wth] > wth2[min_wth2]:
                    edits_seg[seg] = edt_segs[min_wth2]
                    seg_start_id = np.min(edits_segInd2[np.where(edits_seg2 == edt_segs[min_wth2])])
                    seg_end_id = np.max(edits_segInd2[np.where(edits_seg2 == edt_segs[min_wth2])])
                    seg_ends = np.array([seg_start_id, seg_end_id])
                    seg_diff = abs(edits_segInd2[pt_ind2[min_wth2]] - seg_ends)
                    seg_match = np.where(seg_diff == np.min(seg_diff))[0]
                    if len(seg_match) > 1: ### newly added 7/15/29
                        seg_match = 1
                    
                    if seg_match == 1 and edits_segInd[seg[eps[min_wth2]]] == 1:
                        start_id = seg_end_id+1
                        edits_segInd[seg] = edits_segInd[seg]+start_id
                    elif seg_match == 1 and edits_segInd[seg[eps[min_wth2]]] > 1:
                        start_id = seg_end_id+1
                        edits_segInd[seg] = (abs(edits_segInd[seg]-np.max(edits_segInd[seg]))+1) + start_id   
                    if seg_match == 0 and edits_segInd[seg[eps[min_wth2]]] == 1:
                        start_id = seg_start_id-edits_segInd[seg]
                        edits_segInd[seg] = start_id
                    elif seg_match == 0 and edits_segInd[seg[eps[min_wth2]]] > 1:
                        start_id = seg_start_id
                        edits_segInd[seg] = start_id-(abs(edits_segInd[seg]-np.max(edits_segInd[seg]))+1)

            ### One or Multiple edit segments, but no edit_segs
            if len(edt_segs) >= 1 and len(grwl_segs) == 0:
                min_wth2 = 0            
                edits_seg[seg] = edt_segs[min_wth2]
                seg_start_id = np.min(edits_segInd2[np.where(edits_seg2 == edt_segs[min_wth2])])
                seg_end_id = np.max(edits_segInd2[np.where(edits_seg2 == edt_segs[min_wth2])])
                seg_ends = np.array([seg_start_id, seg_end_id])
                seg_diff = abs(edits_segInd2[pt_ind2[min_wth2]] - seg_ends)
                seg_match = np.where(seg_diff == np.min(seg_diff))[0]
                if seg_match == 1 and edits_segInd[seg[eps[min_wth2]]] == 1:
                    start_id = seg_end_id+1
                    edits_segInd[seg] = edits_segInd[seg]+start_id
                elif seg_match == 1 and edits_segInd[seg[eps[min_wth2]]] > 1:
                    start_id = seg_end_id+1
                    edits_segInd[seg] = (abs(edits_segInd[seg]-np.max(edits_segInd[seg]))+1) + start_id   
                if seg_match == 0 and edits_segInd[seg[eps[min_wth2]]] == 1:
                    start_id = seg_start_id-edits_segInd[seg]
                    edits_segInd[seg] = start_id
                elif seg_match == 0 and edits_segInd[seg[eps[min_wth2]]] > 1:
                    start_id = seg_start_id
                    edits_segInd[seg] = start_id-(abs(edits_segInd[seg]-np.max(edits_segInd[seg]))+1)
        
        # If the segment is not short 
        else:
            continue
    
    return edits_segInd, segmentInd

###############################################################################    
    
def write_grwl_edits(grwl_lon, grwl_lat, grwl_x, grwl_y, grwl_Ind, grwl_ID, 
                     grwl_wth, grwl_lake, grwl_chan, grwl_manual, grwl_eps,
                     outfile):
    """
    Convert the result of merging into shapefile
    INPUT
    ncfile -- netCDF file of merged dataset
    OUTPUT
    fshpout -- name of output file
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
    
    proj = osr.SpatialReference()
    #proj.ImportFromEPSG(4326) # 4326 = EPSG code for lon/lat WGS84 projection
    wkt_text = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,\
                                                          AUTHORITY["EPSG",7030]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG",6326]],\
                                                          PRIMEM["Greenwich",0,AUTHORITY["EPSG",8901]],UNIT["DMSH",0.0174532925199433,\
                                                          AUTHORITY["EPSG",9108]],AXIS["Lat",NORTH],AXIS["Long",EAST],AUTHORITY["EPSG",4326]]'
    proj.ImportFromWkt(wkt_text)

    layerout = dataout.CreateLayer(outfile[:-4] + '_lay.shp',
                                   proj, geom_type=ogr.wkbPoint)

    # Define pixel attributes
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
    
    floutDefn = layerout.GetLayerDefn()
    # Create feature (point) to store pixel
    feature_out = ogr.Feature(floutDefn)

    for ipix in range(len(grwl_lat)):
        # Create Geometry Point with pixel coordinates
        pixel_point = ogr.Geometry(ogr.wkbPoint)
        pixel_point.AddPoint(grwl_lon[ipix], grwl_lat[ipix])
        # Add the geometry to the feature
        feature_out.SetGeometry(pixel_point)
        # Set feature attributes
        feature_out.SetField('lon', grwl_lon[ipix])
        feature_out.SetField('lat', grwl_lat[ipix])
        feature_out.SetField('utm_east', grwl_x[ipix])
        feature_out.SetField('utm_north', grwl_y[ipix])
        # int() is needed because facc.dtype=float64, needs to be saved with all values
        # whereas lat.dtype=float64, which raise an error
        feature_out.SetField('segmentInd', int(grwl_Ind[ipix]))
        feature_out.SetField('segmentID', int(grwl_ID[ipix]))
        feature_out.SetField('width_m', int(grwl_wth[ipix]))
        feature_out.SetField('lakeFlag', int(grwl_lake[ipix]))
        feature_out.SetField('nchannels', int(grwl_chan[ipix]))
        feature_out.SetField('manual_add', int(grwl_manual[ipix]))
        feature_out.SetField('endpoints', int(grwl_eps[ipix]))
        
        # Add the feature to the layer
        layerout.CreateFeature(feature_out)
        # Delete point geometry
        pixel_point.Destroy()

    # Close feature and shape files
    feature_out.Destroy()
    dataout.Destroy()

    return fshpout

###############################################################################
################################# Main Code ###################################
###############################################################################    

region = 'NorthAmerica'
fn_basins = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/inputs/HydroBASINS/NorthAmerica/NA_HydroBASINS.shp'
grwl_paths = [file for file in getListOfFiles('C:/Users/ealtenau/Documents/Research/SWAG/For_Server/inputs/GRWL/' + region + '/') if '.shp' in file]
edit_paths = [file for file in getListOfFiles('C:/Users/ealtenau/Documents/Research/SWAG/GRWL/EDITS/csv/NA/') if '.csv' in file]

start_all = time.time()
for ind in list(range(151,len(grwl_paths))):
    
    #start = time.time()
    
    '''
    Identify files.
    '''
    
    pattern = grwl_paths[ind][-11:-4]
    edit_files = [file for file in edit_paths if pattern in file]
    outpath = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/inputs/GRWL/Edits/' + region + '/' + pattern + '_edit.shp'
    
    if len(edit_files) == 0:
        copy_files = [file for file in getListOfFiles('C:/Users/ealtenau/Documents/Research/SWAG/For_Server/inputs/GRWL/' + region + '/') if pattern in file]
        for f in copy_files:
            name = f[-11:-4]
            ext = f[-4:]
            out_dir = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/inputs/GRWL/Edits/' + region + '/' + name + '_edit' + ext
            shutil.copy(f, out_dir)
        print(pattern + ': No Edits Required')
        continue
    
    '''
    Read in and format original GRWL data.
    '''
    start = time.time()
    
    #Read in grwl file:
    utm_east, utm_north, width_m, nchannels,\
        segmentID, segmentInd, lakeFlag, lon, lat = read_grwl(grwl_paths[ind])    
    
    #Creating Manual Flag.    
    manual_flag = np.zeros(len(lon))     
        
    #Editing Original Lake Flag.    
    lakeFlag[np.where(lakeFlag == 255)[0]] = 0
    lakeFlag[np.where(lakeFlag == 250)[0]] = 0           
    lakeFlag[np.where(lakeFlag == 180)[0]] = 1
    lakeFlag[np.where(lakeFlag == 181)[0]] = 1
    lakeFlag[np.where(lakeFlag == 163)[0]] = 1          
    lakeFlag[np.where(lakeFlag == 126)[0]] = 2
    lakeFlag[np.where(lakeFlag == 125)[0]] = 2
    lakeFlag[np.where(lakeFlag == 86)[0]] = 3    
    
    #Correcting original segment errors.         
    new_segID = changing_segID(segmentID, utm_east, utm_north) ###added recently   
    
    #Finding Tile UTM Zone Info.
    max_seg = np.max(new_segID)
    utm_zone = utm.from_latlon(lat[0], lon[0])[2]
    utm_let = utm.from_latlon(lat[0], lon[0])[3] 
    
    #Attach basin code to original grwl:
    #basins = attach_basins(grwl_paths[ind], fn_basins)
    
    '''
    Read in and format GRWL edits.
    '''
    
    #Read in tile edits:        
    if len(edit_files) == 2:
        e1 = pd.read_csv(edit_files[0], sep=',', delimiter=None, header='infer')
        e1_x = np.array(e1.x)
        e1_y = np.array(e1.y)
        e1_seg = np.array(e1.seg)
        e1_lake = np.array(e1.lakeFlag)
        #e1_basins = attach_basins(edit_files[0], fn_basins)
        
        e2 = pd.read_csv(edit_files[1], sep=',', delimiter=None, header='infer')
        e2_x = np.array(e2.x)
        e2_y = np.array(e2.y)
        e2_seg = np.array(e2.seg)
        e2_lake = np.array(e2.lakeFlag)
        #e2_basins = attach_basins(edit_files[1], fn_basins)
        
        if np.max(e1_seg) > np.max(e2_seg):
            e2_seg = e2_seg+np.max(e1.seg)
        
        if np.max(e2_seg) > np.max(e1_seg):
            e1_seg = e1_seg+np.max(e2.seg)
        
        if np.max(e1_seg) == np.max(e2_seg):
            e2_seg = e2_seg+np.max(e1.seg)
            
        edits_x = np.insert(e1_x, len(e1_x), e2_x)
        edits_y = np.insert(e1_y, len(e1_y), e2_y)
        edits_seg = np.insert(e1_seg, len(e1_seg), e2_seg)
        edits_seg = edits_seg+max_seg
        edits_lake = np.insert(e1_lake, len(e1_lake), e2_lake)
        #edits_basins = np.insert(e1_basins, len(e1_basins), e2_basins)
    
    if len(edit_files) == 1:
        edits = pd.read_csv(edit_files[0], sep=',', delimiter=None, header='infer')
        edits_x = np.array(edits.x)
        edits_y = np.array(edits.y)
        edits_seg = np.array(edits.seg)
        edits_seg = edits_seg+max_seg
        edits_lake = edits.lakeFlag
        #edits_basins = attach_basins(edit_files[0], fn_basins)
    
    #Editing New Lake Flag
    edits_lake[np.where(edits_lake == 255)[0]] = 0
    edits_lake[np.where(edits_lake == 250)[0]] = 0           
    edits_lake[np.where(edits_lake == 180)[0]] = 1
    edits_lake[np.where(edits_lake == 181)[0]] = 1
    edits_lake[np.where(edits_lake == 163)[0]] = 1          
    edits_lake[np.where(edits_lake == 126)[0]] = 2
    edits_lake[np.where(edits_lake == 125)[0]] = 2
    edits_lake[np.where(edits_lake == 86)[0]] = 3
    edits_lake = np.asarray(edits_lake)
    
    #Creating filler values in edited data for other GRWL variables.  
    edits_width = np.array(np.repeat(1, len(edits_seg)))
    edits_manual = np.array(np.repeat(1, len(edits_seg)))
    edits_chan = np.array(np.repeat(0, len(edits_seg)))
    edits_lat = np.zeros(len(edits_seg))
    edits_lon = np.zeros(len(edits_seg))
    for idx in list(range(len(edits_seg))): 
        edits_lat[idx], edits_lon[idx] = utm.to_latlon(edits_x[idx], edits_y[idx], utm_zone, utm_let)
        
        
    #Pre-formating edited data
    endpoints = find_endpoints(edits_seg, edits_x, edits_y, 60)            
    edits_segIND = order_segID(edits_seg, endpoints, edits_x, edits_y)    
    edits_segInd, segmentInd = edit_short_segments(endpoints, edits_seg, 
                                                   edits_segIND, edits_x, edits_y, 
                                                   edits_lake, utm_east, utm_north, 
                                                   new_segID, segmentInd, width_m)      
    
    # Add edits to existing grwl.
    grwl_x = np.insert(utm_east, len(utm_east), edits_x)
    grwl_y = np.insert(utm_north, len(utm_north), edits_y)
    grwl_wth = np.insert(width_m, len(width_m), edits_width)
    grwl_chan = np.insert(nchannels, len(nchannels), edits_chan)
    grwl_ID = np.insert(new_segID, len(new_segID), edits_seg)
    grwl_Ind = np.insert(segmentInd, len(segmentInd), edits_segInd)
    grwl_lake = np.insert(lakeFlag, len(lakeFlag), edits_lake)
    grwl_lon = np.insert(lon, len(lon), edits_lon)
    grwl_lat = np.insert(lat, len(lat), edits_lat)
    grwl_manual = np.insert(manual_flag, len(manual_flag), edits_manual)
    grwl_eps = find_endpoints2(grwl_ID, grwl_Ind)
     
    
    """
    WRITING DATA
    """
    #### creating and exporting shapefile 
    #start = time.time()
    write_grwl_edits(grwl_lon, grwl_lat, grwl_x, grwl_y, grwl_Ind, grwl_ID,\
                     grwl_wth, grwl_lake, grwl_chan, grwl_manual, grwl_eps,\
                     outpath) 
    #end = time.time()
    #print('Time to Write the Data, ' + pattern + ': ' +str(np.round((end-start)/60, 3)) + ' min')
    
    end = time.time()
    print(pattern + ': Edits Combined in ' + str(end-start) + 'sec')
    
end_all = time.time()
print('All GRWL Edits Combined in ' + str((end_all-start_all)/60) + 'min')  

'''
"""
FIGURES
"""
plt.figure(1, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Locations', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(grwl_x, grwl_y, c='red', edgecolors='none', s = 5)
plt.scatter(edits_x[seg], edits_y[seg], c='blue', edgecolors='none', s = 5)
plt.scatter(grwl_x[pt_ind], grwl_y[pt_ind], c='green', edgecolors='none', s = 5)
plt.scatter(edits_x2[pt_ind2], edits_y2[pt_ind2], c='orange', edgecolors='none', s = 5)

plt.figure(2, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Manual Locations', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(grwl_x, grwl_y, c=grwl_manual, edgecolors='none', s = 5)

plt.figure(3, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('New GRWL Ind', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(grwl_lon, grwl_lat, c=np.log(grwl_Ind), edgecolors='none', s = 5)

plt.figure(4, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Locations', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(grwl_lon, grwl_lat, c=grwl_ID, edgecolors='none', s = 5)

plt.figure(10, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Lake IDs', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(grwl_x, grwl_y, c=grwl_lake, edgecolors='none', s = 5)

ep = np.where(grwl_eps>0)[0]
plt.figure(5, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('End Points', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(grwl_x, grwl_y, c='blue', edgecolors='none', s = 3)
plt.scatter(grwl_x[ep], grwl_y[ep], c='red', edgecolors='none', s = 5)

plt.figure(6, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Segments', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(edits_x, edits_y, c=edits_seg, edgecolors='none', s = 5)

plt.figure(7, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Segment IDs', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(edits_x, edits_y, c=np.log(edits_segInd), edgecolors='none', s = 5)

plt.figure(8, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Lake IDs', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(edits_x, edits_y, c=edits_lake, edgecolors='none', s = 5)

plt.figure(9, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Data', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(utm_east, utm_north, c='red', edgecolors='none', s = 5)
plt.scatter(edits_x, edits_y, c='blue', edgecolors='none', s = 5)
#plt.scatter(edits_x[neighbors_id], edits_y[neighbors_id], c='yellow', edgecolors='none', s = 3)

###############################################################################
### Plotting GRWL IDs    
unq_id = np.unique(grwl_ID)
number_of_colors2 = len(unq_id)+5
color2 = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors2)]

plt.figure(7, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('GRWL segments',  fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
for i in list(range(len(unq_id))):
    seg = np.where(grwl_ID == unq_id[i])
    plt.scatter(grwl_x[seg], grwl_y[seg], c=color2[i], s = 3, edgecolors = 'None')


'''



### EXTRA Function
'''
def edit_short_segments(endpoints, edits_seg, edits_segInd, edits_x, edits_y, 
                        edits_lake, edits_basins, utm_east, utm_north, segmentID, 
                        segmentInd, basins):
    # Attaching short edit segments to existing GRWL segments.    
    uniq_segs = np.unique(edits_seg)
    for ind in list(range(len(uniq_segs))):
        
        seg = np.where(edits_seg == uniq_segs[ind])[0]
        lakes = len(np.where(edits_lake[seg] > 0)[0])       
        length = len(seg)
        #print(length)
        
        if length < 100 and lakes == 0:
            
            ### new as of 5/8/19
            edits_x2 = np.delete(edits_x, seg)
            edits_y2 = np.delete(edits_y, seg)
            edits_basins2 = np.delete(edits_basins, seg)
            edits_seg2 = np.delete(edits_seg, seg)
            edits_segInd2 = np.delete(edits_segInd, seg)
            
            ###
            
            grwl_pts = np.vstack((utm_east, utm_north)).T
            edit_pts = np.vstack((edits_x2, edits_y2)).T
            eps = np.where(endpoints[seg] > 0)[0]        
            ep_pts = np.vstack((edits_x[seg[eps]], edits_y[seg[eps]])).T
            kdt = sp.cKDTree(grwl_pts)
            
            if len(edit_pts) > 0:
                kdt2 = sp.cKDTree(edit_pts)
            
            if len(seg) == 1:
                pt_dist, pt_ind = kdt.query(ep_pts, k = 2)
                remv = np.where(pt_dist > 500)[0]
                if len(remv) > 0:
                    pt_dist = np.delete(pt_dist, remv)
                    pt_ind = np.delete(pt_ind, remv)
                if len(edit_pts) > 0:
                    pt_dist2, pt_ind2 = kdt2.query(ep_pts, k = 2)
                    remv2 = np.where(pt_dist2 > 500)[0]
                    if len(remv2) > 0:
                        pt_dist2 = np.delete(pt_dist2, remv2)
                        pt_ind2 = np.delete(pt_ind2, remv2)
                    dist_all = np.insert(pt_dist, len(pt_dist), pt_dist2)
            else:
                pt_dist, pt_ind = kdt.query(ep_pts, k = 1)
                remv = np.where(pt_dist > 500)[0]
                if len(remv) > 0:
                    pt_dist = np.delete(pt_dist, remv)
                    pt_ind = np.delete(pt_ind, remv)
                if len(edit_pts) > 0:
                    pt_dist2, pt_ind2 = kdt2.query(ep_pts, k = 1)
                    remv2 = np.where(pt_dist2 > 500)[0]
                    if len(remv2) > 0:
                        pt_dist2 = np.delete(pt_dist2, remv2)
                        pt_ind2 = np.delete(pt_ind2, remv2)
                    dist_all = np.insert(pt_dist, len(pt_dist), pt_dist2)
            
            if len(pt_ind2) == 0:
                dist_all = pt_dist                  
                    
            grwl_segs = segmentID[pt_ind]            
            
            if len(pt_ind2) > 0:
                edt_segs = np.unique(edits_seg2[pt_ind2])
            else:
                edt_segs = []
            
            if len(pt_ind2) > 0 and np.min(pt_dist2) > 100:
                edt_segs = []
            
            ### No segments found 
            if len(grwl_segs) == 0 and len(edt_segs) == 0:
                print(str(uniq_segs[ind]) + ': Short edit segment with no bounding segments.')
                continue
            
            ### Segments are too far away
            if len(dist_all) == 0: #use to be: np.min(dist_all) > 500
                print(str(uniq_segs[ind]) + ': Short edit segment with no bounding GRWL segments.')
                continue
            
            ### Multiple grwl segments, but no edit_segs
            if len(grwl_segs) > 1 and len(edt_segs) == 0:
                wth = []
                for idx in list(range(len(grwl_segs))):
                    wth.append(np.min(basins[np.where(segmentID == grwl_segs[idx])]))
                min_wth = np.where(wth == np.max(wth))[0][0]            
                edits_seg[seg] = grwl_segs[min_wth]
                seg_start_id = np.min(segmentInd[np.where(segmentID == grwl_segs[min_wth])])
                if segmentInd[pt_ind[min_wth]] > seg_start_id and edits_segInd[seg[eps[min_wth]]] == 1:
                    start_id = segmentInd[pt_ind[min_wth]]+1
                    edits_segInd[seg] = edits_segInd[seg]+start_id
                if segmentInd[pt_ind[min_wth]] == seg_start_id and edits_segInd[seg[eps[min_wth]]] == 1:
                    start_id = seg_start_id-edits_segInd[seg]
                    edits_segInd[seg] = start_id
                if segmentInd[pt_ind[min_wth]] > seg_start_id and edits_segInd[seg[eps[min_wth]]] > 1:
                    start_id = np.max(segmentInd[np.where(segmentID == grwl_segs[min_wth])])+1
                    edits_segInd[seg] = abs(edits_segInd[seg]-np.max(edits_segInd[seg])) + start_id  
                if segmentInd[pt_ind[min_wth]] == seg_start_id and edits_segInd[seg[eps[min_wth]]] > 1:
                    start_id = np.max(edits_segInd[seg])+1
                    segmentInd[np.where(segmentID== grwl_segs[min_wth])] = segmentInd[np.where(segmentID== grwl_segs[min_wth])]+start_id
            
            ### Only one grwl segment and no edit segments
            if len(np.unique(grwl_segs)) == 1 & len(edt_segs) == 0:
                edits_seg[seg] = np.unique(grwl_segs)
                seg_min_id = np.min(segmentInd[pt_ind])
                seg_max_id = np.max(segmentInd[pt_ind])
                alter_vals = np.where(segmentInd[np.where(segmentID == np.unique(grwl_segs))] >= seg_max_id)
                if len(seg) == 1:
                    edits_segInd[seg] = edits_segInd[seg]+seg_max_id
                    segmentInd[alter_vals] = segmentInd[alter_vals]+len(seg)+1
                else:
                    if segmentInd[pt_ind[0]] < segmentInd[pt_ind[1]] and endpoints[eps[0]] < endpoints[eps[1]]:
                        edits_segInd[seg] = edits_segInd[seg]+seg_min_id
                        segmentInd[alter_vals] = segmentInd[alter_vals]+len(seg)
                    if segmentInd[pt_ind[0]] < segmentInd[pt_ind[1]] and endpoints[eps[0]] > endpoints[eps[1]]:
                        edits_segInd[seg] = abs(edits_segInd[seg]-np.max(edits_segInd[seg]))+(seg_min_id+1)
                        segmentInd[alter_vals] = segmentInd[alter_vals]+len(seg)
                    if segmentInd[pt_ind[0]] > segmentInd[pt_ind[1]] and endpoints[eps[0]] < endpoints[eps[1]]:
                        edits_segInd[seg] = abs(edits_segInd[seg]-np.max(edits_segInd[seg]))+(seg_min_id+1)
                        segmentInd[alter_vals] = segmentInd[alter_vals]+len(seg)  
                    if segmentInd[pt_ind[0]] > segmentInd[pt_ind[1]] and endpoints[eps[0]] > endpoints[eps[1]]:
                        edits_segInd[seg] = edits_segInd[seg]+seg_min_id
                        segmentInd[alter_vals] = segmentInd[alter_vals]+len(seg)                   
            
            ### grwl and edit segments            
            if len(grwl_segs) > 0 and len(edt_segs) > 0:
                wth = np.zeros(len(grwl_segs))
                wth2 = np.zeros(len(edt_segs))
                for idx in list(range(len(grwl_segs))):
                    wth[idx] = np.max(basins[np.where(segmentID == grwl_segs[idx])])
                for idz in list(range(len(edt_segs))):    
                    wth2[idz] = np.max(edits_basins2[np.where(edits_seg2 == edt_segs[idz])])
                
                min_wth = np.where(wth == np.max(wth))[0]
                min_wth2 = np.where(wth2 == np.max(wth2))[0][0]
                
                if wth[min_wth] < wth2[min_wth2]:

                    edits_seg[seg] = grwl_segs[min_wth]
                    seg_start_id = np.min(segmentInd[np.where(segmentID == grwl_segs[min_wth])])
                    if segmentInd[pt_ind[min_wth]] > seg_start_id and edits_segInd[seg[eps[min_wth]]] == 1:
                        start_id = segmentInd[pt_ind[min_wth]]+1
                        edits_segInd[seg] = edits_segInd[seg]+start_id
                    if segmentInd[pt_ind[min_wth]] == seg_start_id and edits_segInd[seg[eps[min_wth]]] == 1:
                        start_id = seg_start_id-edits_segInd[seg]
                        edits_segInd[seg] = start_id
                    if segmentInd[pt_ind[min_wth]] > seg_start_id and edits_segInd[seg[eps[min_wth]]] > 1:
                        start_id = np.max(segmentInd[np.where(segmentID == grwl_segs[min_wth])])+1
                        edits_segInd[seg] = abs(edits_segInd[seg]-np.max(edits_segInd[seg])) + start_id  
                    if segmentInd[pt_ind[min_wth]] == seg_start_id and edits_segInd[seg[eps[min_wth]]] > 1:
                        start_id = np.max(edits_segInd[seg])+1
                        segmentInd[np.where(segmentID== grwl_segs[min_wth])] = segmentInd[np.where(segmentID== grwl_segs[min_wth])]+start_id                   
            
                if wth[min_wth] > wth2[min_wth2]:
                    edits_seg[seg] = edt_segs[min_wth2]
                    seg_start_id = np.min(edits_segInd2[np.where(edits_seg2 == edt_segs[min_wth2])])
                    if edits_segInd2[pt_ind2[min_wth2]] > seg_start_id and edits_segInd[seg[eps[min_wth2]]] == 1:
                        start_id = edits_segInd2[pt_ind2[min_wth2]]+1
                        edits_segInd[seg] = edits_segInd[seg]+start_id
                    if edits_segInd2[pt_ind2[min_wth2]] == seg_start_id and edits_segInd[seg[eps[min_wth2]]] == 1:
                        start_id = seg_start_id-edits_segInd[seg]
                        edits_segInd[seg] = start_id
                    if edits_segInd2[pt_ind2[min_wth2]] > seg_start_id and edits_segInd[seg[eps[min_wth2]]] > 1:
                        start_id = np.max(edits_segInd2[np.where(edits_seg2 == edt_segs[min_wth2])])+1
                        edits_segInd[seg] = abs(edits_segInd[seg]-np.max(edits_segInd[seg])) + start_id  
                    if edits_segInd2[pt_ind2[min_wth2]] == seg_start_id and edits_segInd[seg[eps[min_wth2]]] > 1:
                        start_id = np.max(edits_segInd[seg])+1
                        edits_segInd[np.where(edits_seg == edt_segs[min_wth2])] = edits_segInd[np.where(edits_seg == edt_segs[min_wth2])]+start_id
                                     
            ### One or Multiple edit segments, but no grwl segments
            if len(edt_segs) >= 1 and len(grwl_segs) == 0:
                wth2 = []
                for idx in list(range(len(edt_segs))):
                    wth2[idx] = np.max(edits_basins2[np.where(edits_seg2 == edt_segs[idx])])
                min_wth2 = np.where(wth2 == np.max(wth2))[0][0]
                
                #min_wth2 = 0            
                edits_seg[seg] = edt_segs[min_wth2]
                seg_start_id = np.min(edits_segInd2[np.where(edits_seg2 == edt_segs[min_wth2])])
                if edits_segInd2[pt_ind2[min_wth2]] > seg_start_id and edits_segInd[seg[eps[min_wth2]]] == 1:
                    start_id = edits_segInd2[pt_ind2[min_wth2]]+1
                    edits_segInd[seg] = edits_segInd[seg]+start_id
                if edits_segInd2[pt_ind2[min_wth2]] == seg_start_id and edits_segInd[seg[eps[min_wth2]]] == 1:
                    start_id = seg_start_id-edits_segInd[seg]
                    edits_segInd[seg] = start_id
                if edits_segInd2[pt_ind2[min_wth2]] > seg_start_id and edits_segInd[seg[eps[min_wth2]]] > 1:
                    start_id = np.max(edits_segInd2[np.where(edits_seg2 == edt_segs[min_wth2])])+1
                    edits_segInd[seg] = abs(edits_segInd[seg]-np.max(edits_segInd[seg])) + start_id  
                if edits_segInd2[pt_ind2[min_wth2]] == seg_start_id and edits_segInd[seg[eps[min_wth2]]] > 1:
                    start_id = np.max(edits_segInd[seg])+1
                    edits_segInd[np.where(edits_seg == edt_segs[min_wth2])] = edits_segInd[np.where(edits_seg == edt_segs[min_wth2])]+start_id
            
        # If the segment is not short 
        else:
            continue
        
    return edits_segInd, segmentInd

'''