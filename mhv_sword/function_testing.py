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

def check_rchs(rch_id, dist):
    issues = []
    unq_rch = np.unique(rch_id)
    for ind in list(range(len(unq_rch))):
        rch = np.where(rch_id == unq_rch[ind])[0]
        if len(rch) == 1:
            continue
        diff = np.abs(np.diff(dist[rch]))
        if np.max(diff) > 1000:
            issues.append(unq_rch[ind])
    return issues

###############################################################################

def correct_rchs(subcls, issues):
    start_id = np.max(subcls.rch_id5) + 1
    for ind in list(range(len(issues))):
        rch = np.where(subcls.rch_id5 == issues[ind])[0]
        unq_segs = np.unique(subcls.seg[rch])
        for s in list(range(len(unq_segs))):
            seg = np.where(subcls.seg[rch] == unq_segs[s])[0]
            subcls.rch_id5[rch[seg]] = start_id
            min_dist = np.min(subcls.rch_dist5[rch[seg]])
            subcls.rch_dist5[rch[seg]] = subcls.rch_dist5[rch[seg]]-min_dist
            max_dist = np.max(subcls.rch_dist5[rch[seg]])
            subcls.rch_len5[rch[seg]] = max_dist-min_dist
            start_id = start_id+1

    update = np.where(subcls.rch_len5 == 0)[0]
    subcls.rch_len5[update] = 90
            
###############################################################################