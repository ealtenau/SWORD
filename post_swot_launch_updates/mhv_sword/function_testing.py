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
        rch_x_all, rch_y_all, __, __ = reproject_utm(rch_lat, rch_lon)
        rch_pts_all = np.vstack((rch_x_all, rch_y_all)).T
        
        rch_pts, index, count = np.unique(rch_pts_all, axis=0, return_counts=True, return_index = True)
        rch_x = rch_x_all[index]
        rch_y = rch_y_all[index]
        rch_ind = subcls.ind[rch[index]]
        rch_segs = subcls.seg[rch[index]]
        unq_segs = np.unique(subcls.seg[rch[index]]) #subcls.seg[rch[index]]
        
        rch_eps_all = np.zeros(len(index))
        if len(rch_segs) == 1 or len(rch) < 3:
            new_rch_ind[rch] = subcls.ind[rch]
            ep1 = np.where(subcls.ind[rch] == np.min(subcls.ind[rch]))[0]
            ep2 = np.where(subcls.ind[rch] == np.max(subcls.ind[rch]))[0]
            new_rch_eps[rch[ep1]] = 1
            new_rch_eps[rch[ep2]] = 1
            #reverse index order to have indexes increasing in the upstream direction.
            if subcls.facc[rch[ep1]] < subcls.facc[rch[ep2]]:
                new_rch_ind[rch] = abs(new_rch_ind[rch] - np.max(new_rch_ind[rch]))
            
        else:
            for r in list(range(len(unq_segs))):
                seg = np.where(rch_segs == unq_segs[r])[0]
                mn = np.where(rch_ind[seg] == np.min(rch_ind[seg]))[0]
                mx = np.where(rch_ind[seg] == np.max(rch_ind[seg]))[0]  
                rch_eps_all[seg[mn]] = 1
                rch_eps_all[seg[mx]] = 1
                eps_ind = np.where(rch_eps_all[seg]>0)[0]
                ep_pts = np.vstack((rch_x[seg[eps_ind]], rch_y[seg[eps_ind]])).T
                kdt = sp.cKDTree(rch_pts)
                if len(seg) < 3:
                    rch_eps_all[seg] = 0
                else:
                    pt_dist, pt_ind = kdt.query(ep_pts, k = 3)
                    nghs = abs(rch_segs[pt_ind] - unq_segs[r])
                    row_sums = np.sum(nghs, axis = 1)
                    rmv = np.where(row_sums > 0)[0]
                    rch_eps_all[seg[rmv]] = 0

            final_eps = np.where(rch_eps_all == 1)[0]
            if len(final_eps) == 0:
                print(ind, uniq_rch[ind], len(rch), 'index issue - no endpoints')
                # final_eps = np.where(rch_eps_all == 1)[0]
                final_eps = np.array([0,len(rch)-1])

            elif len(final_eps) > 2:
                print(ind, uniq_rch[ind], len(rch), 'index issue - more than two endpoints')
                # break

            # Re-ordering points based on updated endpoints.
            new_ind = np.zeros(len(rch))
            new_ind[index[final_eps[0]]]=1
            idz = index[final_eps[0]]
            # new_ind[eps_ind[final_eps[0]]]=1
            # idz = eps_ind[final_eps[0]]
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

    return new_rch_ind, new_rch_eps