from __future__ import division
import os
import utm
import sys
import math
from osgeo import ogr
from osgeo import osr
# from pyproj import Proj
from pyproj import Proj, transform
import numpy as np
from osgeo import gdal
#import shapefile
import rasterio
from scipy import spatial as sp
import geopandas as gp
import pandas as pd
import time
import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, BSpline, splev
import itertools
from geopandas import GeoSeries
from shapely.geometry import LineString, Point



###############################################################################

def meters_to_degrees(meters, latitude):
    #getting degree lengths from meters. 
    deg = np.round(meters/(111.32 * 1000 * math.cos(latitude * (math.pi / 180))),5)
    return deg

###############################################################################

def create_geom(nid, pt_ind, nx, ny, ortho):
    ### creating orthogonal geometry. 
    geom = []
    for ind in list(range(len(nid))):
        print(ind, len(nid)-1)
        line = meters_to_degrees(ortho[ind], ny[ind])
        x1 = nx[pt_ind[ind,1]] #was 1 & 2
        x2 = nx[pt_ind[ind,2]]
        y1 = ny[pt_ind[ind,1]]
        y2 = ny[pt_ind[ind,2]]
        dx = x1-x2
        dy = y1-y2
        dist = np.sqrt(dx*dx+dy*dy)
        dx /= dist 
        dy /= dist 
        x_coords = np.array([nx[ind]+(line/2)*dy])
        y_coords = np.array([ny[ind]-(line/2)*dx])
        x_coords = np.insert(x_coords, len(x_coords), nx[ind]-(line/2)*dy, axis=0)
        y_coords = np.insert(y_coords, len(y_coords), ny[ind]+(line/2)*dx, axis=0)
        pts = GeoSeries(map(Point, zip(x_coords, y_coords)))
        draw = LineString(pts.tolist())
        geom.append(draw)
        ### Testing Plot
        # plt.plot(x_coords, y_coords, c='cyan')
        # plt.scatter(nx, ny, c='black', s=15)
        # plt.show()  
    return geom

###############################################################################
### Reading in data. 

### netcdf input large amount of data.
# sword_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16_glows/netcdf/na_sword_v16_glows_5constant.nc'
# sword = nc.Dataset(sword_fn,'r+')
# nx = np.array(sword['/nodes/x/'][:])
# ny = np.array(sword['/nodes/y/'][:])
# nid = np.array(sword['/nodes/node_id/'][:])
# nchan_max = np.array(sword['/nodes/n_chan_max/'][:])
# nchan_mod = np.array(sword['/nodes/n_chan_mod/'][:])
# wth = np.array(sword['/nodes/width/'][:])
# max_wth = np.array(sword['/nodes/max_width/'][:])
# glows_sig = np.array(sword['/nodes/glows_wth_1sig/'][:])
# rid = np.array(sword['/nodes/reach_id/'][:])
# ext_dist = np.array(sword['/nodes/ext_dist_coef/'][:])

modifier = 'wil_orthos0_base.gpkg'
sword_fn = '/Users/ealtenau/Documents/SWORD_Dev/testing_files/willamette_extdist_test_subset_v16_glows.gpkg'
sword = gp.read_file(sword_fn)
nx = np.array(sword['x'])
ny = np.array(sword['y'])
nid = np.array(sword['node_id'])
nchan_max = np.array(sword['n_chan_max'])
nchan_mod = np.array(sword['n_chan_mod'])
wth = np.array(sword['width'])
max_wth = np.array(sword['max_width'][:])
glows_sig = np.array(sword['gs_wth_sig'][:])
rid = np.array(sword['reach_id'][:])
ext_dist = np.array(sword['extdist'][:])

#Performing spaital query for node neighbors. Needed for orthogonals. 
sword_pts = np.vstack((nx,ny)).T
kdt = sp.cKDTree(sword_pts)
pt_dist, pt_ind = kdt.query(sword_pts, k = 5) 

ext_dist = np.repeat(5,len(nid)) #setting everything a value of 3 to start. 

# single channel = 5 multichannel = 10. 
# ext_dist = np.repeat(5,len(nid))
# unq_rchs = np.unique(rid)
# for r in list(range(len(unq_rchs))):
#     npts = np.where(rid == unq_rchs[r])[0]
#     chan_med = np.median(nchan_max[npts]) #starting with nchan_max
#     if chan_med > 1:
#         ext_dist[npts] = 10


# adjusting multichannel to have higher values and single channels to have lower values. 
# ext_dist = np.repeat(5,len(nid))
# unq_rchs = np.unique(rid)
# for r in list(range(len(unq_rchs))):
#     npts = np.where(rid == unq_rchs[r])[0]
#     chan_med = np.median(nchan_max[npts]) #starting with nchan_max
#     if chan_med > 1:
#         ext_dist[npts] = chan_med

#trying cassie's ratio with glow-s data. 
ext_dist = np.repeat(5,len(nid))
unq_rchs = np.unique(rid)
for r in list(range(len(unq_rchs))):
    npts = np.where(rid == unq_rchs[r])[0]
    max_sig = np.max(glows_sig[npts])
    chan_med = np.median(nchan_max[npts])
    if max_sig > 0:
        ext_dist[npts] = (wth[npts]+(3*max_sig))/wth[npts]
        if chan_med > 1:
            update = np.where((ext_dist[npts] < 5) & (ext_dist[npts] > 10))[0]
            ext_dist[npts[update]] = 5
ext_dist[np.where(ext_dist < 0)] = 5
# ext_dist[np.where(ext_dist > 10)] = 5

# ortho = (max_wth/2) * ext_dist #creating orthogonal distance to display.
# ortho = ((max_wth/3)*3/2) * ext_dist
# ortho = (wth/2) * ext_dist 
ortho = ((wth/3)*3/2) * ext_dist

geom = create_geom(nid, pt_ind, nx, ny, ortho)

sword['ext_dist'] = ext_dist
sword['geometry'] = geom
outgpkg = '/Users/ealtenau/Documents/SWORD_Dev/testing_files/'+modifier
sword.to_file(outgpkg, driver='GPKG', layer='reaches')
print('Done')

#updating netcdf
# sword['/nodes/ext_dist_coef/'][:] = ext_dist
# sword.close()

print(np.unique(ext_dist))
# print(np.unique(sword['/nodes/ext_dist_coef/'][:]))



































##########################################################################
### adjusting max widths for outlier nodes. 

# def calc_outliers(vals):
#     q3, q1 = np.percentile(vals, [90, 10]) #typical - 75,25
#     iqr = q3 - q1
#     outliers = [_ for _ in vals if _ > q3 + (iqr * 1.5) or _ < q1 - (iqr * 1.5)]
#     outliers = np.unique(np.array(outliers))
#     if len(outliers) > 0:
#         out_ind = np.where(np.in1d(vals, outliers) == True)[0]
#     print(outliers)
#     # return outliers, out_ind

# unq_rchs = np.unique(rid)
# for r in list(range(len(unq_rchs))):
#     print(r)
#     rch = np.where(rid == unq_rchs[r])[0]
#     rw = wth[rch]
#     rmw = max_wth[rch]
#     calc_outliers(rw)
#     calc_outliers(rmw)
