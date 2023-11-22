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
import geopandas as gp
from pyproj import Proj
import argparse
import netCDF4 as nc

###############################################################################

def find_tributary_junctions(subcls):

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
    tribs = np.zeros(len(subcls.seg))
    mhv_pts = np.vstack((subcls.x, subcls.y)).T
    uniq_segs = np.unique(subcls.seg)
    for ind in list(range(len(uniq_segs))):

        # Isolate endpoints for the edited segment.
        seg = np.where(subcls.seg == uniq_segs[ind])[0]
        if len(seg) == 1:
            eps = np.array([0,0])
        else:
            pt1 = np.where(subcls.ind[seg] == np.min(subcls.ind[seg]))[0]
            pt2 = np.where(subcls.ind[seg] == np.max(subcls.ind[seg]))[0]
            eps = np.array([pt1,pt2]).T

        # Perform spatial query of closest GRWL points to the edited segment
        # endpoints.
        ep_pts = np.vstack((subcls.x[seg[eps]], subcls.y[seg[eps]])).T
        kdt = sp.cKDTree(mhv_pts)

        if len(seg) < 3:
            pt_dist, pt_ind = kdt.query(ep_pts, k = 4,
                                        distance_upper_bound = 200.0)
        elif 3 <= len(seg) and len(seg) <= 6:
            pt_dist, pt_ind = kdt.query(ep_pts, k = 10,
                                        distance_upper_bound = 500.0)
        elif len(seg) > 6:
            pt_dist, pt_ind = kdt.query(ep_pts, k = 10,
                                        distance_upper_bound = 1000.0)

        ep1_ind = pt_ind[1,:]
        ep1_dist = pt_dist[1,:]
        na1 = np.where(ep1_ind == len(mhv_pts))
        ep1_dist = np.delete(ep1_dist, na1)
        ep1_ind = np.delete(ep1_ind, na1)

        ep2_ind = pt_ind[1,:]
        ep2_dist = pt_dist[1,:]
        na2 = np.where(ep2_ind == len(mhv_pts))
        ep2_dist = np.delete(ep2_dist, na2)
        ep2_ind = np.delete(ep2_ind, na2)

        ep1_segs = np.unique(subcls.seg[ep1_ind])
        ep2_segs = np.unique(subcls.seg[ep2_ind])

        # If there is only one neighboring GRWL segment, designate it as a
        # tributary junction if the edited segment endpoint falls in the middle
        # of the segment.
        if len(ep1_segs) > 1:
            ep1_min = np.min(subcls.ind[np.where(subcls.seg == ep1_segs[0])[0]])
            ep1_max = np.max(subcls.ind[np.where(subcls.seg == ep1_segs[0])[0]])
            if np.min(subcls.ind[ep1_ind]) > ep1_min+5 and np.max(subcls.ind[ep1_ind]) < ep1_max-5:
                tribs[ep1_ind[0]] = 1

        if len(ep2_segs) > 1:
            ep2_min = np.min(subcls.ind[np.where(subcls.seg == ep2_segs[0])[0]])
            ep2_max = np.max(subcls.ind[np.where(subcls.seg == ep2_segs[0])[0]])
            if np.min(subcls.ind[ep2_ind]) > ep2_min+5 and np.max(subcls.ind[ep2_ind]) < ep2_max-5:
                tribs[ep2_ind[0]] = 1

    return tribs

###############################################################################

def update_rch_indexes(subcls, new_rch_id):
    # Set variables and find unique reaches.
    uniq_rch = np.unique(new_rch_id)
    new_rch_ind = np.zeros(len(subcls.ind))
    new_rch_eps = np.zeros(len(subcls.ind))

    # Loop through each reach and re-order indexes.
    for ind in list(range(len(uniq_rch))):
        rch = np.where(new_rch_id == uniq_rch[ind])[0]
        rch_lon = subcls.lon[rch]
        rch_lat = subcls.lat[rch]
        rch_x, rch_y, __, __ = reproject_utm(rch_lat, rch_lon)
        rch_pts = np.vstack((rch_x, rch_y)).T
        rch_segs = np.unique(subcls.seg[rch])
        rch_eps_all = np.zeros(len(rch))
        if len(rch_segs) == 1:
            new_rch_ind[rch] = subcls.ind[rch]
            ep1 = np.where(subcls.ind[rch] == np.min(subcls.ind[rch]))[0]
            ep2 = np.where(subcls.ind[rch] == np.max(subcls.ind[rch]))[0]
            new_rch_eps[rch[ep1]] = 1
            new_rch_eps[rch[ep2]] = 1
            #reverse index order to have indexes increasing in the upstream direction.
            if subcls.facc[rch[ep1]] < subcls.facc[rch[ep2]]:
                new_rch_ind[rch] = abs(new_rch_ind[rch] - np.max(new_rch_ind[rch]))
            
        else:
            for r in list(range(len(rch_segs))):
                seg = np.where(subcls.seg[rch] == rch_segs[r])[0]
                mn = np.where(subcls.ind[rch[seg]] == np.min(subcls.ind[rch[seg]]))[0]
                mx = np.where(subcls.ind[rch[seg]] == np.max(subcls.ind[rch[seg]]))[0]
                rch_eps_all[seg[mn]] = 1
                rch_eps_all[seg[mx]] = 1

            eps_ind = np.where(rch_eps_all>0)[0]
            ep_pts = np.vstack((rch_x[eps_ind], rch_y[eps_ind])).T
            kdt = sp.cKDTree(rch_pts)
            if len(rch) < 4: #use to be 5.
                pt_dist, pt_ind = kdt.query(ep_pts, k = len(rch_segs)) 
            else:
                pt_dist, pt_ind = kdt.query(ep_pts, k = 4, distance_upper_bound=300)
                pt_ind[np.where(pt_ind[:] == 112)] = 111
            row_sums = np.sum(rch_eps_all[pt_ind], axis = 1)
            final_eps = np.where(row_sums == 1)[0]
            if len(final_eps) == 0:
                print(ind, uniq_rch[ind], len(rch), 'index issue - short reach')
                # final_eps = np.where(rch_eps_all == 1)[0]
                final_eps = np.array([0,len(rch)-1])

            elif len(final_eps) > 2:
                print(ind, uniq_rch[ind], len(rch), 'index issue - possible tributary')
                # break

            # Re-ordering points based on updated endpoints.
            new_ind = np.zeros(len(rch))
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
            #reverse index order to have indexes increasing in the upstream direction.
            if subcls.facc[rch[ep1]] < subcls.facc[rch[ep2]]:
                new_rch_ind[rch] = abs(new_ind - np.max(new_ind))
            


