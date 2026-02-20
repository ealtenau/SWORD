# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 13:48:15 2020
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
import netCDF4 as nc

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

def renumber_grwl_segs(grwl_segs, grwl_tile):

    """
    FUNCTION:
        Creates a 1-D array of re-numbered GRWL segments in the concatenated
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
    tribs = np.zeros(len(grwl.ID))
    grwl_pts = np.vstack((grwl.x, grwl.y)).T
    uniq_segs = np.unique(grwl.ID)
    for ind in list(range(len(uniq_segs))):
        
        # Isolate endpoints for the edited segment.
        seg = np.where(grwl.ID == uniq_segs[ind])[0]
        if len(seg) == 1:
            eps = np.array([0,0])
        else:
            pt1 = np.where(grwl.segInd[seg] == np.min(grwl.segInd[seg]))[0] 
            pt2 = np.where(grwl.segInd[seg] == np.max(grwl.segInd[seg]))[0]
            eps = np.array([pt1,pt2]).T
        
        # Perform spatial query of closest GRWL points to the edited segment 
        # endpoints.
        ep_pts = np.vstack((grwl.x[seg[eps]], grwl.y[seg[eps]])).T
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
        na1 = np.where(ep1_ind == len(grwl_pts))[0]
        #na11 = np.where(grwl.ID[ep1_ind] == grwl.ID[seg])[0]
        ep1_dist = np.delete(ep1_dist, na1)
        ep1_ind = np.delete(ep1_ind, na1)
    
        ep2_ind = pt_ind[1,:]
        ep2_dist = pt_dist[1,:]
        na2 = np.where(ep2_ind == len(grwl_pts))
        ep2_dist = np.delete(ep2_dist, na2)
        ep2_ind = np.delete(ep2_ind, na2)
    
        ep1_segs = np.unique(grwl.ID[ep1_ind])
        ep2_segs = np.unique(grwl.ID[ep2_ind])
        
        # If there is only one neighboring GRWL segment, designate it as a 
        # tributary junction if the edited segment endpoint falls in the middle
        # of the segment. 
        if len(ep1_segs) == 1:
            ep1_min = np.min(grwl.segInd[np.where(grwl.ID == ep1_segs[0])[0]])
            ep1_max = np.max(grwl.segInd[np.where(grwl.ID == ep1_segs[0])[0]])
            if np.min(grwl.segInd[ep1_ind]) > ep1_min+5 and np.max(grwl.segInd[ep1_ind]) < ep1_max-5:
                tribs[ep1_ind[0]] = 1
           
        if len(ep2_segs) == 1:
            ep2_min = np.min(grwl.segInd[np.where(grwl.ID == ep2_segs[0])[0]])
            ep2_max = np.max(grwl.segInd[np.where(grwl.ID == ep2_segs[0])[0]])
            if np.min(grwl.segInd[ep2_ind]) > ep2_min+5 and np.max(grwl.segInd[ep2_ind]) < ep2_max-5:
                tribs[ep2_ind[0]] = 1
    
    return tribs

###############################################################################
    
def cut_segments(grwl, start_seg):
    
    """    
    FUNCTION:
        Creates a new 1-D that contains unique segment IDs for the GRWL 
        segments that need to be cut at tributary junctions.

    INPUTS
        grwl --  Object containing attributes for the GRWL centerlines.
        start_seg -- Starting ID value to assign to the new cut segments. Each 
            new segment will be ...
         
    OUTPUTS
        new_segs -- Updated Segment IDs. 
    """
    
    new_segs = np.copy(grwl.ID)
    cut = np.where(grwl.tribs == 1)[0]
    cut_segs = np.unique(grwl.ID[cut])
    seg_id = start_seg
    
    # Loop through segments that contain tributary junctions and identify 
    # the new boundaries of the segment to cut and re-number. 
    for ind in list(range(len(cut_segs))):
        seg = np.where(grwl.ID == cut_segs[ind])[0]
        num_tribs = np.where(grwl.tribs[seg] == 1)[0]
        max_ind = np.where(grwl.segInd[seg] == np.max(grwl.segInd[seg]))[0]
        min_ind = np.where(grwl.segInd[seg] == np.min(grwl.segInd[seg]))[0]
        bounds = np.insert(num_tribs, 0, min_ind)
        bounds = np.insert(bounds, len(bounds), max_ind)
        for idx in list(range(len(bounds)-1)):
            id1 = bounds[idx]
            id2 = bounds[idx+1]
            new_vals = np.where((grwl.segInd[seg] >= grwl.segInd[seg[id1]]) & (grwl.segInd[seg] <= grwl.segInd[seg[id2]]))[0]
            new_segs[seg[new_vals]] = seg_id
            seg_id = seg_id+1
    
    return new_segs  
           
###############################################################################
###############################################################################
###############################################################################

fn = main_dir+'/data/outputs/Merged_Data/OC/OC_Merge_v08.nc'
merge = nc.Dataset(fn, 'r+') 

#read in relevant data.
grwl = Object()
grwl.ID = np.copy(merge.groups['centerlines'].variables['segID'][:])
grwl.segInd = np.copy(merge.groups['centerlines'].variables['segInd'][:])
grwl.x = np.copy(merge.groups['centerlines'].variables['easting'][:])
grwl.y = np.copy(merge.groups['centerlines'].variables['northing'][:])
grwl.lon = np.copy(merge.groups['centerlines'].variables['x'][:])
grwl.lat = np.copy(merge.groups['centerlines'].variables['y'][:])
grwl.eps = np.copy(merge.groups['centerlines'].variables['endpoints'][:])
grwl.cl_id = np.copy(merge.groups['centerlines'].variables['cl_id'][:])
grwl.basins = np.copy(merge.groups['centerlines'].variables['basin_code'][:])

def cut_continental_tribs(grwl):

    #find 20 closest neighbors for all points.
    all_pts = np.vstack((grwl.lon, grwl.lat)).T
    kdt = sp.cKDTree(all_pts)
    eps_dist, eps_ind = kdt.query(all_pts, k = 20)
    eps = np.where(grwl.eps == 1)[0]
    grwl.tribs = np.zeros(len(grwl.ID))
    
    #for all the endpoint locations identify whether there is an unidentified tributary junction.
    for ind in list(range(len(eps))):
        close_pts = np.unique(grwl.ID[eps_ind[eps[ind],:]])
        neighbors = close_pts[np.where(close_pts != grwl.ID[eps[ind]])[0]]
        if len(neighbors) == 1:
            ep1_min = np.min(grwl.segInd[np.where(grwl.ID == neighbors[0])[0]])+5
            ep1_max = np.max(grwl.segInd[np.where(grwl.ID == neighbors[0])[0]])-5
            pt = eps_ind[eps[ind],np.min(np.where(grwl.ID[eps_ind[eps[ind],:]] == neighbors[0])[0])]
            pt_id = grwl.segInd[pt]
            if ep1_min < pt_id < ep1_max:
                #print(ind)
                grwl.tribs[pt] = 1
    
    #cut segments at tributaries and update endpoints.
    start_seg = np.max(grwl.ID)+1
    grwl.newID = cut_segments(grwl, start_seg)
    grwl.new_eps = np.copy(grwl.eps)
    keep = np.where(grwl.tribs == 1)[0]
    grwl.new_eps[keep] = 1

#replace current values. 
merge.groups['centerlines'].variables['segID'][:] = grwl.newID
merge.groups['centerlines'].variables['endpoints'][:] = grwl.eps
merge.close()


'''
#len(np.where(grwl.tribs == 1)[0])/len(eps)           

merge.delncattr('segID')


keep = np.where(grwl.tribs == 1)[0]
df = pd.DataFrame(np.array([grwl.lon[keep], grwl.lat[keep]]).T)
df.columns = ['lon', 'lat'] 
#df.to_csv(main_dir+'/data/outputs/test_merging/new_tribs.csv')

level2_basins = np.array([np.int(np.str(ind)[0:2]) for ind in grwl.basins])
uniq_level2 = np.unique(level2_basins)
uniq_level2 = np.delete(uniq_level2, 0)
l2 = np.where(level2_basins == 81)[0]

df2 = pd.DataFrame(np.array([grwl.lon[l2], grwl.lat[l2], grwl.ID[l2], grwl.cl_id[l2]]).T)
df2.columns = ['lon', 'lat', 'id', 'ind'] 
#df2.to_csv(main_dir+'/data/outputs/test_merging/old_id_hb81.csv')

df3 = pd.DataFrame(np.array([grwl.lon[l2], grwl.lat[l2], grwl.newID[l2], grwl.cl_id[l2]]).T)
df3.columns = ['lon', 'lat', 'id', 'ind'] 
#df3.to_csv(main_dir+'/data/outputs/test_merging/new_id_hb81.csv')

'''
    