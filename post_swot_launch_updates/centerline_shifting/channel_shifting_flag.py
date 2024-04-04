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

###############################################################################
######################## Reading and Writing Functions ########################
###############################################################################

class Object(object):
    """
    FUNCTION:
        Creates class object to assign attributes to.
    """
    pass
###############################################################################

def meters_to_degrees(meters, latitude):
    deg = np.round(meters/(111.32 * 1000 * math.cos(latitude * (math.pi / 180))),5)
    return deg

###############################################################################

def read_jrc(jrc_fn):
    
    #Getting vals
    jrc_ras = gdal.Open(jrc_fn)
    vals = np.array(jrc_ras.GetRasterBand(1).ReadAsArray()).flatten()

    # Getting Coordinates
    jrc = rasterio.open(jrc_fn)
    height = jrc.shape[0]
    width = jrc.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(jrc.transform, rows, cols)
    lon= np.array(xs).flatten()
    lat = np.array(ys).flatten()

    # Assiging lat/lon coordinates as attributes to "mhydro" object.
    keep = np.where(vals > 0)[0] 
    jrc = Object()
    jrc.lon = lon[keep]
    jrc.lat = lat[keep]
    jrc.vals = vals[keep]

    return jrc


###############################################################################

def clip_sword(sword, jrc):
    xmin = np.min(jrc.lon)
    xmax = np.max(jrc.lon)
    ymin = np.min(jrc.lat)
    ymax = np.max(jrc.lat)
    ll = np.array([xmin, ymin])  # lower-left
    ur = np.array([xmax, ymax])  # upper-right
    
    lon = np.array(sword.groups['centerlines'].variables['x'][:])
    lat = np.array(sword.groups['centerlines'].variables['y'][:])
    rch_id = np.array(sword.groups['centerlines'].variables['reach_id'][0,:])
    pts = np.array([(lon[i], lat[i]) for i in range(len(lon))])

    idx = np.all(np.logical_and(ll <= pts, pts <= ur), axis=1)
    sword_lon = lon[idx]
    sword_lat = lat[idx]
    sword_rch_id = rch_id[idx]

    return sword_lon, sword_lat, sword_rch_id

###############################################################################

def read_sword(sword_fn, jrc):
    
    sword = nc.Dataset(sword_fn)
    data = Object()
    data.lon, data.lat, data.reach_id = clip_sword(sword, jrc)

    unq_rchs = np.unique(data.reach_id)
    data.rch_len = np.zeros(len(data.reach_id))
    data.wth = np.zeros(len(data.reach_id))
    data.max_wth = np.zeros(len(data.reach_id))
    data.chan = np.zeros(len(data.reach_id))
    data.max_chan = np.zeros(len(data.reach_id))
    for r in list(range(len(unq_rchs))):
        cl_rch = np.where(data.reach_id == unq_rchs[r])[0]
        rch = np.where(sword.groups['reaches'].variables['reach_id'][:] == unq_rchs[r])[0]
        data.rch_len[cl_rch] = sword.groups['reaches'].variables['reach_length'][rch]
        data.wth[cl_rch] = sword.groups['reaches'].variables['width'][rch]
        data.max_wth[cl_rch] = sword.groups['reaches'].variables['max_width'][rch]
        data.chan[cl_rch] = sword.groups['reaches'].variables['n_chan_mod'][rch]
        data.max_chan[cl_rch] = sword.groups['reaches'].variables['n_chan_max'][rch]    
    sword.close()

    return data

###############################################################################
###############################################################################
###############################################################################

sword_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16/netcdf/na_sword_v16.nc'
jrc_fn = '/Users/ealtenau/Documents/SWORD_Dev/inputs/JRC_Water_Occurance/occurrence_110W_50Nv1_4_2021.tif'

jrc = read_jrc(jrc_fn)
sword = read_sword(sword_fn, jrc)

jrc_pts = np.vstack((jrc.lon, jrc.lat)).T
sword_pts = np.vstack((sword.lon, sword.lat)).T
kdt = sp.cKDTree(jrc_pts)
pt_dist, pt_ind = kdt.query(sword_pts, k = 10, distance_upper_bound=0.005)
dist = np.median(pt_dist, axis=1)
zero_out = np.where(dist == np.inf)[0]
dist[zero_out] = 0
ind_max = np.max(pt_ind, axis=1)
keep = np.where(ind_max != jrc_pts.shape[0])[0]
vals = np.zeros(len(dist))
vals[keep]  = np.median(jrc.vals[pt_ind[keep,:]],axis=1)

#filter the flag.
sword.shift_flag = np.zeros(len(sword.reach_id))
unq_rchs = np.unique(sword.reach_id)
for ind in list(range(len(unq_rchs))):
    rch = np.where(sword.reach_id == unq_rchs[ind])[0]
    chan = np.max(sword.chan[rch])

    # define radius for point flagging - based on reach width. 
    width = np.max(sword.wth[rch])
    max_width =  np.max(sword.max_wth[rch])
    # default radius/threshold is 200 for reaches with no width information.
    if width <= 30 and max_width <= 30:
        radius = 200
    # if width value for reach is bad, use 2x max_width value if it's good. 
    elif width <= 30 and max_width > 30:
        radius = max_width*2
    # use 2x reach width value for reach with width information. 
    else:
        radius = width*2

    threshold = meters_to_degrees(radius, np.median(sword.lat[rch]))
    # flag = np.where(dist[rch]>threshold)[0] #original flag did not consider no data areas. 
    flag = np.where((dist[rch]>threshold) & (vals[rch] >= 5))[0] #was 5.
    if len(flag) == 0:
        continue 
    else:
        perc = (len(flag)/len(rch))*100

    if perc > 50 and chan == 1:
        sword.shift_flag[rch] = 1



f = np.where(sword.shift_flag==1)[0]
plt.scatter(sword.lon, sword.lat, c='black')
plt.scatter(sword.lon[f], sword.lat[f], c='red')
plt.show()



shift = np.where(sword.shift_flag == 1)[0]
pt_dist2, pt_ind2 = kdt.query(sword_pts, k = 20)
shift_lon = np.mean(jrc.lon[pt_ind2[shift,:]], axis=1)
shift_lat = np.mean(jrc.lat[pt_ind2[shift,:]], axis=1)

new_lon = np.copy(sword.lon)
new_lat = np.copy(sword.lat)
new_lon[shift] = shift_lon
new_lat[shift] = shift_lat



unq_shift_rchs = np.unique(sword.reach_id[shift])
# idx=np.where(unq_shift_rchs==74298900291)[0]
new_lon_smooth = np.copy(new_lon)
new_lat_smooth = np.copy(new_lat)
for idx in list(range(len(unq_shift_rchs))):
    r = np.where(sword.reach_id == unq_shift_rchs[idx])[0]
    x_diff = np.max(new_lon[r]) - np.min(new_lon[r])
    y_diff = np.max(new_lat[r]) - np.min(new_lat[r])
    if y_diff < x_diff:
        s = np.var(new_lat[r])/3
    else:
        s = np.var(new_lon[r])/3


    okay = np.where(np.abs(np.diff(new_lon[r])) + np.abs(np.diff(new_lat[r])) > 0)[0]
    if len(okay) < 5:
        continue
    pts = np.vstack((new_lon[r[okay]], new_lat[r[okay]]))
    # Find the B-spline representation of an N-dimensional curve
    tck, u = splprep(pts, s=0.00000005) #0.0001, 0.000075
    # n_new = np.linspace(u.min(), u.max(), len(node_pts))
    cl_new = np.linspace(u.min(), u.max(), len(r))
    # Evaluate a B-spline
    # node_x_smooth, node_y_smooth = splev(n_new, tck)
    cl_x_smooth, cl_y_smooth = splev(cl_new, tck)
    new_lon_smooth[r] = cl_x_smooth
    new_lat_smooth[r] = cl_y_smooth






# df = pd.DataFrame(np.array([sword.lon, sword.lat, sword.shift_flag, sword.reach_id]).T)
# df.to_csv('/Users/ealtenau/Desktop/sword_shift_flag_test3.csv', index=False)


# meters_to_degrees(1000, np.median(sword.lat[rch]))


plt.scatter(sword.lon, sword.lat, c='blue', s = 3)
plt.scatter(new_lon, new_lat, c='magenta', s = 3)
plt.scatter(new_lon_smooth, new_lat_smooth, c='cyan', s = 3)
plt.show()

# df2 = pd.DataFrame(np.array([new_lon_smooth, new_lat_smooth, sword.shift_flag]).T)
# df2.to_csv('/Users/ealtenau/Desktop/sword_shift_test2.csv', index=False)