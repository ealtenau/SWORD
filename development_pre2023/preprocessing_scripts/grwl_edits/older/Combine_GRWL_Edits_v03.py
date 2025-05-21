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
#import shutil
import time
import utm
from osgeo import ogr
from osgeo import osr
import numpy as np
#import geopandas as gp
#from shapely.geometry import Point 
import pandas as pd
#import gdal
#import glob
#import matplotlib.pyplot as plt
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

def read_edits(edit_files, max_seg):

    #Read in tile edits:        
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
            
        edits_x = np.insert(e1_x, len(e1_x), e2_x)
        edits_y = np.insert(e1_y, len(e1_y), e2_y)
        edits_seg = np.insert(e1_seg, len(e1_seg), e2_seg)
        edits_seg = edits_seg+max_seg
        edits_lake = np.insert(e1_lake, len(e1_lake), e2_lake)
    
    if len(edit_files) == 1:
        edits = pd.read_csv(edit_files[0], sep=',', delimiter=None, header='infer')
        edits_x = np.array(edits.x)
        edits_y = np.array(edits.y)
        edits_seg = np.array(edits.seg)
        edits_seg = edits_seg+max_seg
        edits_lake = edits.lakeFlag
        
    return edits_x, edits_y, edits_seg, edits_lake
    
###############################################################################

def update_segID(grwl_id, grwl_x, grwl_y):
    
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

def label_endpoints(edits_seg, edits_segInd):
    eps = np.zeros(len(edits_seg))
    uniq_segs = np.unique(edits_seg)    
    for ind in list(range(len(uniq_segs))):
        seg = np.where(edits_seg == uniq_segs[ind])[0]
        eps[seg[np.where(edits_segInd[seg] == np.min(edits_segInd[seg]))]] = 1
        eps[seg[np.where(edits_segInd[seg] == np.max(edits_segInd[seg]))]] = 2
    return eps        

###############################################################################    

def order_edits(edits_seg, endpoints, edits_x, edits_y):
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

def find_tributary_junctions(grwl_x, grwl_y, grwl_ID, grwl_ind, 
                             edits_x, edits_y, edits_ind, edits_ID):
    
    tribs = np.zeros(len(grwl_ID))
    grwl_pts = np.vstack((grwl_x, grwl_y)).T
    uniq_segs = np.unique(edits_ID)
    
    for ind in list(range(len(uniq_segs))):
        seg = np.where(edits_ID == uniq_segs[ind])[0]
        
        if len(seg) == 1:
            eps = np.array([0,0])
        else:
            pt1 = np.where(edits_ind[seg] == np.min(edits_ind[seg]))[0] ### eventually need to solve weird index issues.
            pt2 = np.where(edits_ind[seg] == np.max(edits_ind[seg]))[0]
            eps = np.array([pt1,pt2]).T
    
        ep_pts = np.vstack((edits_x[seg[eps]], edits_y[seg[eps]])).T
        kdt = sp.cKDTree(grwl_pts)
    
        if len(seg) < 3:
            pt_dist, pt_ind = kdt.query(ep_pts, k = 4, distance_upper_bound = 45.0)
        elif 3 <= len(seg) and len(seg) <= 6:
            pt_dist, pt_ind = kdt.query(ep_pts, k = 10, distance_upper_bound = 100.0)
        elif len(seg) > 6:
            pt_dist, pt_ind = kdt.query(ep_pts, k = 10, distance_upper_bound = 200.0)
    
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
    
        ep1_segs = np.unique(grwl_ID[ep1_ind])
        ep2_segs = np.unique(grwl_ID[ep2_ind])
        
        if len(ep1_segs) == 1:
            ep1_min = np.min(grwl_ind[np.where(grwl_ID == ep1_segs[0])[0]])
            ep1_max = np.max(grwl_ind[np.where(grwl_ID == ep1_segs[0])[0]])
            if np.min(grwl_ind[ep1_ind]) > ep1_min+5 and np.max(grwl_ind[ep1_ind]) < ep1_max-5:
                tribs[ep1_ind[0]] = 1
           
        if len(ep2_segs) == 1:
            ep2_min = np.min(grwl_ind[np.where(grwl_ID == ep2_segs[0])[0]])
            ep2_max = np.max(grwl_ind[np.where(grwl_ID == ep2_segs[0])[0]])
            if np.min(grwl_ind[ep2_ind]) > ep2_min+5 and np.max(grwl_ind[ep2_ind]) < ep2_max-5:
                tribs[ep2_ind[0]] = 1

    return tribs

###############################################################################
    
def cut_segments(tribs, grwl_ID, grwl_ind, start_seg):
    new_segs = np.copy(grwl_ID)
    cut = np.where(tribs == 1)[0]
    cut_segs = np.unique(grwl_ID[cut])
    seg_id = start_seg
    for ind in list(range(len(cut_segs))):
        seg = np.where(grwl_ID == cut_segs[ind])[0]
        num_tribs = np.where(tribs[seg] == 1)[0]
        max_ind = np.where(grwl_ind[seg] == np.max(grwl_ind[seg]))[0]
        min_ind = np.where(grwl_ind[seg] == np.min(grwl_ind[seg]))[0]
        bounds = np.insert(num_tribs, 0, min_ind)
        bounds = np.insert(bounds, len(bounds), max_ind)
        for idx in list(range(len(bounds)-1)):
            id1 = bounds[idx]
            id2 = bounds[idx+1]
            new_segs[seg[id1:id2]] = seg_id
            seg_id = seg_id+1
            
    return new_segs
        
###############################################################################
    
def update_indexes(segInd, segID, x, y, manual_add, edits_seg, edits_segInd):
    uniq_rch = np.unique(segID)
    new_rch_ind = np.zeros(len(segInd))
    new_rch_eps = np.zeros(len(segInd))
    for ind in list(range(len(uniq_rch))):
        rch = np.where(segID == uniq_rch[ind])[0]
        rch_x = x[rch]
        rch_y = y[rch]
        rch_pts = np.vstack((rch_x, rch_y)).T
        rch_segs = manual_add[rch]
        segs = np.unique(manual_add[rch])
        new_ind = np.zeros(len(rch))
        eps = np.zeros(len(rch))
        cond = 0
        
        if len(rch) == 1:
            new_rch_ind[rch] = 1
            new_rch_eps[rch] = 1
            continue
        
        if len(segs) > 1 or np.min(rch_segs) == 1:
            #print(ind, np.unique(segID[rch]), len(segs))
            
            if np.min(rch_segs) == 1:
                sub_rch = np.where(edits_segID == np.unique(segID[rch]))[0]
                if len(np.unique(edits_seg[sub_rch])) > 1:
                    segs = np.unique(edits_seg[sub_rch])
                    cond = 1
                else:
                    new_rch_ind[rch] = segInd[rch]
                    ep1 = np.where(segInd[rch] == np.min(segInd[rch]))[0]
                    ep2 = np.where(segInd[rch] == np.max(segInd[rch]))[0]
                    new_rch_eps[rch[ep1]] = 1
                    new_rch_eps[rch[ep2]] = 1
                    continue
                      
            if cond == 0:
                for idx in list(range(len(segs))):
                    s = np.where(manual_add[rch] == segs[idx])[0]
                    if segs[idx] == 1:
                        sub_segs = np.unique(edits_seg[np.where(edits_segID == np.unique(segID[rch[s]]))[0]])
                        if len(sub_segs) > 1:
                            sub_rch = np.where(edits_segID == np.unique(segID[rch[s]]))[0]
                            for idy in list(range(len(sub_segs))):
                                s2 = np.where(edits_seg[sub_rch] == sub_segs[idy])[0]
                                mn = np.where(edits_segInd[sub_rch[s2]] == np.min(edits_segInd[sub_rch[s2]]))[0]
                                mx = np.where(edits_segInd[sub_rch[s2]] == np.max(edits_segInd[sub_rch[s2]]))[0]
                                eps[s[s2[mn]]] = 1
                                eps[s[s2[mx]]] = 1
                                
                    else:
                        #s = np.where(manual_add[rch] == segs[idx])[0]
                        mn = np.where(segInd[rch[s]] == np.min(segInd[rch[s]]))[0]
                        mx = np.where(segInd[rch[s]] == np.max(segInd[rch[s]]))[0]
                        eps[s[mn]] = 1
                        eps[s[mx]] = 1
            else:
                for idx in list(range(len(segs))):
                    s = np.where(edits_seg[sub_rch] == segs[idx])[0]
                    mn = np.where(edits_segInd[sub_rch[s]] == np.min(edits_segInd[sub_rch[s]]))[0]
                    mx = np.where(edits_segInd[sub_rch[s]] == np.max(edits_segInd[sub_rch[s]]))[0]
                    eps[s[mn]] = 1
                    eps[s[mx]] = 1

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

            new_ind[final_eps[0]]=1
            idz = final_eps[0]
            count = 2
            while np.min(new_ind) == 0:
                d = np.sqrt((rch_x[idz]-rch_x)**2 + (rch_y[idz]-rch_y)**2)
                dzero = np.where(new_ind == 0)[0]
                #vals = np.where(edits_segInd[dzero] eq 0)
                next_pt = dzero[np.where(d[dzero] == np.min(d[dzero]))[0]][0]
                new_ind[next_pt] = count
                count = count+1
                idz = next_pt

            new_rch_ind[rch] = new_ind
            ep1 = np.where(new_ind == np.min(new_ind))[0]
            ep2 = np.where(new_ind == np.max(new_ind))[0]
            new_rch_eps[rch[ep1]] = 1
            new_rch_eps[rch[ep2]] = 1

        else:
            new_rch_ind[rch] = segInd[rch]
            ep1 = np.where(segInd[rch] == np.min(segInd[rch]))[0]
            ep2 = np.where(segInd[rch] == np.max(segInd[rch]))[0]
            new_rch_eps[rch[ep1]] = 1
            new_rch_eps[rch[ep2]] = 1

    return new_rch_ind, new_rch_eps
    
###############################################################################

def edit_short_segments(endpoints, edits_seg, edits_x, edits_y, 
                        edits_lake, utm_east, utm_north, segmentID, 
                        width_m):
    # Attaching short edit segments to existing GRWL segments.    
    new_segs = np.copy(edits_seg)
    uniq_segs = np.unique(new_segs)
    
    for ind in list(range(len(uniq_segs))):
        #print(ind)
        seg = np.where(new_segs == uniq_segs[ind])[0]
        lakes = len(np.where(edits_lake[seg] > 0)[0])       
        length = len(seg)
        #print(length)
        
        if length < 100 and lakes == 0:
            
            ### new as of 5/8/19
            edits_width = np.array(np.repeat(1, len(new_segs)))
            edits_x2 = np.delete(edits_x, seg)
            edits_y2 = np.delete(edits_y, seg)
            edits_width2 = np.delete(edits_width, seg)
            edits_seg2 = np.delete(new_segs, seg)
            
            all_width = np.insert(width_m, 0, edits_width2)
            all_x = np.insert(utm_east, 0, edits_x2)
            all_y = np.insert(utm_north, 0, edits_y2)
            all_segs = np.insert(segmentID, 0, edits_seg2)
            ###
            
            all_pts = np.vstack((all_x, all_y)).T
            
            if len(seg) == 1:
                eps = np.array([0,0])
            else:
                eps = np.where(endpoints[seg] > 0)[0]
            
            ep_pts = np.vstack((edits_x[seg[eps]], edits_y[seg[eps]])).T
            kdt = sp.cKDTree(all_pts)
            
            if len(seg) < 4:
                pt_dist, pt_ind = kdt.query(ep_pts, k = 4, distance_upper_bound = 45.0)
            elif 4 <= len(seg) and len(seg) <= 6:
                pt_dist, pt_ind = kdt.query(ep_pts, k = 10, distance_upper_bound = 100.0)
            elif len(seg) > 6:
                pt_dist, pt_ind = kdt.query(ep_pts, k = 10, distance_upper_bound = 200.0)
        
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
            
            ep1_segs = np.unique(all_segs[ep1_ind])
            ep2_segs = np.unique(all_segs[ep2_ind])
            
            ### No segments found 
            if len(ep1_segs) == 0 and len(ep2_segs) == 0:
                #print(str(uniq_segs[ind]) + ': Short edit segment with no bounding segments.')
                continue
            
            ### Multiple segments on one end only.
            elif len(ep1_segs) > 1 and len(ep2_segs) == 0:
                continue
            
            ### Multiple segments on one end only.
            elif len(ep1_segs) == 0 and len(ep2_segs) > 1:
                continue
            
            ### Multiple segments on both ends.
            elif len(ep1_segs) > 1 and len(ep2_segs) > 1:
                continue
            
            ### Multiple segments on one end and one segment on the other end.
            elif len(ep1_segs) > 1 and len(ep2_segs) == 1:
                new_segs[seg] = ep2_segs
            
            ### Multiple segments on one end and one segment on the other end.
            elif len(ep1_segs) == 1 and len(ep2_segs) > 1:
                new_segs[seg] = ep1_segs

            ### One segment on one end only.
            elif len(ep1_segs) == 1 and len(ep2_segs) == 0:
                new_segs[seg] = ep1_segs
            
            ### One segment on one end only.
            elif len(ep1_segs) == 0 and len(ep2_segs) == 1:
                new_segs[seg] = ep2_segs
                
            ### One segment on BOTH ends.
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

region = 'SouthAmerica'
grwl_paths = [file for file in getListOfFiles('C:/Users/ealtenau/Documents/Research/SWAG/For_Server/inputs/GRWL/Originals/' + region + '/') if '.shp' in file]
edit_paths = [file for file in getListOfFiles('C:/Users/ealtenau/Documents/Research/SWAG/GRWL/EDITS/csv/SA/') if '.csv' in file]

start_all = time.time()
for ind in list(range(45,len(grwl_paths))):
    
    '''
    Identify files.
    '''
    pattern = grwl_paths[ind][-11:-4]
    edit_files = [file for file in edit_paths if pattern in file]
    outpath = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/inputs/GRWL/Edits/' + region + '/' + pattern + '_edit.shp'
    
    '''
    Read in and format original GRWL data.
    '''
    start = time.time()
    
    #Read in grwl file:
    utm_east, utm_north, width_m, nchannels,\
        segmentID, segmentInd, lakeFlag, lon, lat = read_grwl(grwl_paths[ind])    
        
    if len(lon) == 0:
        print(ind, pattern, 'NO GRWL DATA')
        continue
    
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
    updated_segID = update_segID(segmentID, utm_east, utm_north) ###added recently   
    orig_eps = label_endpoints(updated_segID, segmentInd)    

    #Finding Tile UTM Zone Info.
    max_seg = np.max(updated_segID)
    utm_zone = utm.from_latlon(lat[0], lon[0])[2]
    utm_let = utm.from_latlon(lat[0], lon[0])[3] 

    if len(edit_files) == 0:
        write_grwl_edits(lon, lat, utm_east, utm_north, segmentInd, updated_segID,
                         width_m, lakeFlag, nchannels, manual_flag, orig_eps,
                         outpath)
        end = time.time()
        print(ind, pattern + ': No Edits - file written in: ' + str(np.round((end-start), 2)) + ' sec')
        continue
        
    '''
    Read in and format GRWL edits.
    '''
    edits_x, edits_y, edits_seg, edits_lake = read_edits(edit_files, max_seg)
    endpoints = find_endpoints(edits_seg, edits_x, edits_y, 60)            
    edits_segInd = order_edits(edits_seg, endpoints, edits_x, edits_y)
    
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
    
    #Cutting grwl segments at new tributaries.
    tribs = find_tributary_junctions(utm_east, utm_north, updated_segID, 
                                     segmentInd, edits_x, edits_y, 
                                     edits_segInd, edits_seg)
    
    start_seg = np.max([np.max(updated_segID), np.max(edits_seg)])+1
    new_segID = cut_segments(tribs, updated_segID, segmentInd, start_seg)
    
    #Combining small edits with GRWL segments. 
    edits_segID = edit_short_segments(endpoints, edits_seg, edits_x, edits_y, 
                                      edits_lake, utm_east, utm_north, 
                                      new_segID, width_m)
 
    #Creating filler values in edited data for other GRWL variables.  
    edits_width = np.array(np.repeat(1, len(edits_seg)))
    edits_manual = np.array(np.repeat(1, len(edits_seg)))
    edits_chan = np.array(np.repeat(0, len(edits_seg)))
    edits_lat = np.zeros(len(edits_seg))
    edits_lon = np.zeros(len(edits_seg))
    for idx in list(range(len(edits_seg))): 
        edits_lat[idx], edits_lon[idx] = utm.to_latlon(edits_x[idx], edits_y[idx], utm_zone, utm_let)
    
    # Add edits to existing grwl.
    grwl_x = np.insert(utm_east, len(utm_east), edits_x)
    grwl_y = np.insert(utm_north, len(utm_north), edits_y)
    grwl_wth = np.insert(width_m, len(width_m), edits_width)
    grwl_chan = np.insert(nchannels, len(nchannels), edits_chan)
    grwl_ID = np.insert(new_segID, len(new_segID), edits_segID)
    grwl_Ind = np.insert(segmentInd, len(segmentInd), edits_segInd)
    grwl_lake = np.insert(lakeFlag, len(lakeFlag), edits_lake)
    grwl_lon = np.insert(lon, len(lon), edits_lon)
    grwl_lat = np.insert(lat, len(lat), edits_lat)
    grwl_manual = np.insert(manual_flag, len(manual_flag), edits_manual)        
    
    #Updating new segments indexes.
    new_segInd, grwl_eps = update_indexes(grwl_Ind, grwl_ID, grwl_x, grwl_y, 
                                          grwl_manual, edits_seg, edits_segInd)
    
    """
    WRITING DATA
    """
    #### creating and exporting shapefile 
    write_grwl_edits(grwl_lon, grwl_lat, grwl_x, grwl_y, new_segInd, grwl_ID,\
                     grwl_wth, grwl_lake, grwl_chan, grwl_manual, grwl_eps,\
                     outpath) 
    
    end = time.time()
    print(ind, pattern + ': Edits Combined in: ' + str(np.round((end-start), 2)) + ' sec')
    
end_all = time.time()
print('All GRWL Edits Combined in: ' + str(np.round((end_all-start_all)/60, 2)) + ' min')  
