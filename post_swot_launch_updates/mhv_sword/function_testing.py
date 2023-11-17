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

tribs = find_tributary_junctions(subcls)