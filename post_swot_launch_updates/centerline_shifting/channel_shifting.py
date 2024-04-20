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
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

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
###############################################################################
###############################################################################

flag_fn = '/Users/ealtenau/Desktop/sword_shift_flag_test3_subset.csv'
jrc_fn = '/Users/ealtenau/Documents/SWORD_Dev/inputs/JRC_Water_Occurance/occurrence_110W_50Nv1_4_2021.tif'

start = time.time()
jrc = read_jrc(jrc_fn)
pts = gp.GeoDataFrame([
    np.array(jrc.lon),
    np.array(jrc.lat),
    ]).T
pts.rename(
    columns={
        0:"x",
        1:"y",
        },inplace=True)
geom = gp.GeoSeries(map(Point, zip(jrc.lon, jrc.lat)))
pts['geometry'] = geom
pts = gp.GeoDataFrame(pts)
pts.set_geometry(col='geometry')
pts = pts.set_crs(4326, allow_override=True)
end = time.time()
print('Time to load and format JRC tile: '+str(np.round((end-start)/60,2))+' mins')

df = pd.read_csv(flag_fn)
x = np.array(df['0'])
y = np.array(df['1'])
flag = np.array(df['2'])
rchs = np.array(df['3'])
index = np.array(df['4'])

#subset JRC
box = Polygon([(-108.6901,44.221), (-108.0658,44.221), (-108.0658,42.617), (-108.6901,42.617)])
jrc_mask = pts.within(box)
jrc_clip = pts.loc[jrc_mask]

sword_pts = np.vstack((x,y)).T
kdt = sp.cKDTree(sword_pts)
pt_dist, pt_ind = kdt.query(sword_pts, k = 15)

start = time.time()
shift = np.where(flag == 1)[0]
new_x = np.copy(x)
new_y = np.copy(y)
for ind in list(range(743,len(shift))):
    print(ind, len(shift)-1)
    p = shift[ind] #cl_id (4873082) is a tricky case....
    x1 = x[pt_ind[p,1]] #was 1 & 2
    x2 = x[pt_ind[p,2]]
    y1 = y[pt_ind[p,1]]
    y2 = y[pt_ind[p,2]]
    dx = x1-x2
    dy = y1-y2
    dist = np.sqrt(dx*dx+dy*dy)
    dx /= dist 
    dy /= dist 
    x3 = x1+(0.01/2)*dy
    y3 = y1-(0.01/2)*dx
    x4 = x1-(0.01/2)*dy
    y4 = y1+(0.01/2)*dx
    x5 = x2+(0.01/2)*dy
    y5 = y2-(0.01/2)*dx
    x6 = x2-(0.01/2)*dy
    y6 = y2+(0.01/2)*dx
    poly = Polygon([(x3, y3), (x4, y4), (x6, y6), (x5, y5)])
    end = time.time()
    # print('Time to Create Bounding Box: '+str(end-start))
    start = time.time()
    pt_mask = jrc_clip.within(poly)
    jrc_pts = jrc_clip.loc[pt_mask]
    end = time.time()
    # print('Method 1: '+str(end-start))
    new_x[shift[ind]] = np.median(jrc_pts['x'])
    new_y[shift[ind]] = np.median(jrc_pts['y'])

    # plt.scatter(jrc_clip['x'], jrc_clip['y'], c='lightgrey',s=2)
    # plt.scatter(x, y, c='black', s=3)
    # plt.scatter(x1, y1, c='cyan', s=5)
    # plt.scatter(x2, y2, c='cyan', s=5)
    # plt.scatter(jrc_pts['x'], jrc_pts['y'], c='gold', s=5)
    # plt.show()  


unq_shift_rchs = np.unique(rchs[shift])
new_x_smooth = np.copy(new_x)
new_y_smooth = np.copy(new_y)
for idx in list(range(len(unq_shift_rchs))):
    r = np.where(rchs == unq_shift_rchs[idx])[0]
    x_diff = np.max(new_x[r]) - np.min(new_x[r])
    y_diff = np.max(new_y[r]) - np.min(new_y[r])
    if y_diff < x_diff:
        s = np.var(new_y[r])/3
    else:
        s = np.var(new_x[r])/3

    okay = np.where(np.abs(np.diff(new_x[r])) + np.abs(np.diff(new_y[r])) > 0)[0]
    if len(okay) < 5:
        continue
    pts = np.vstack((new_x[r[okay]], new_y[r[okay]]))
    # Find the B-spline representation of an N-dimensional curve
    tck, u = splprep(pts, s=0.000001) #0.0001, 0.000075
    # n_new = np.linspace(u.min(), u.max(), len(node_pts))
    cl_new = np.linspace(u.min(), u.max(), len(r))
    # Evaluate a B-spline
    # node_x_smooth, node_y_smooth = splev(n_new, tck)
    cl_x_smooth, cl_y_smooth = splev(cl_new, tck)
    new_x_smooth[r] = cl_x_smooth
    new_y_smooth[r] = cl_y_smooth



plt.scatter(jrc_clip['x'], jrc_clip['y'], c='lightgrey',s=2)
plt.scatter(new_x, new_y, c='magenta', s=2)
plt.scatter(new_x_smooth, new_y_smooth, c='blue', s=2)
plt.scatter(x, y, c='black', s=3)
plt.show()


# df2 = pd.DataFrame(np.array([new_x_smooth, new_y_smooth, flag, rchs]).T)
# df2.to_csv('/Users/ealtenau/Desktop/sword_shift_test4.csv', index=False)

#######################################################################################
#######################################################################################
#######################################################################################

unq_rchs = np.unique(rchs[np.where(flag == 1)])
r = np.where(np.in1d(rchs,unq_rchs)==True)[0]

test_x = x[r]-0.001
test_y = y[r]+0.002

plt.scatter(jrc_clip['x'], jrc_clip['y'], c='lightgrey',s=2)
plt.scatter(test_x, test_y, c='cyan', s=3)
plt.scatter(x, y, c='black', s=3)
plt.show()