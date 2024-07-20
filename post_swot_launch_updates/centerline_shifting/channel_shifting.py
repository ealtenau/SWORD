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
import itertools

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
    jrc_arr = Object()
    jrc_arr.lon = lon[keep]
    jrc_arr.lat = lat[keep]
    jrc_arr.vals = vals[keep]

    return jrc_arr

###############################################################################
###############################################################################
###############################################################################

flag_fn = '/Users/ealtenau/Documents/SWORD_Dev/inputs/JRC_Water_Occurance/NA/160W_70_sword_v17_shift_flag.gpkg'
jrc_fn = '/Users/ealtenau/Documents/SWORD_Dev/inputs/JRC_Water_Occurance/occurrence_160W_70Nv1_4_2021.tif'

print('Reading JRC Data')
start = time.time()
jrc = read_jrc(jrc_fn)
jrc_pts = np.vstack((jrc.lon,jrc.lat)).T
kdt = sp.cKDTree(jrc_pts)
end = time.time()
print(str(np.round((end-start)/60,2))+' mins')

df = gp.read_file(flag_fn)
x = np.array(df['x'])
y = np.array(df['y'])
flag = np.array(df['shift_flag'])
rchs = np.array(df['reach_id'])
index = np.array(df['cl_ind'])

x_steps = list(np.round(np.arange(-0.005,0.0051,0.0001),10))
y_steps = list(np.round(np.arange(-0.005,0.0051,0.0001),10))
pairs = list(itertools.product(x_steps, y_steps))
x_pair = np.array([p[0] for p in pairs])
y_pair = np.array([p[1] for p in pairs])

shift = np.where(flag == 1)[0]
## need to loop through reaches with shift flags and do original shifting. 
unq_shift_rchs = np.unique(rchs[shift])
new_x = np.copy(x)
new_y = np.copy(y)
new_x_smooth = np.copy(new_x)
new_y_smooth = np.copy(new_y)
# ind = np.where(unq_shift_rchs == 81140600031)[0]
for ind in list(range(len(unq_shift_rchs))):
    print(ind, len(unq_shift_rchs)-1)
    r = np.where(rchs == unq_shift_rchs[ind])[0]
    sword_pts = np.vstack((x[r],y[r])).T
    pt_dist, pt_ind = kdt.query(sword_pts, k = 500) 
    med_dist_x = np.median(jrc.lon[pt_ind], axis = 1)
    med_dist_y = np.median(jrc.lat[pt_ind], axis = 1)

    x_diff = np.zeros(len(x_pair))
    y_diff = np.zeros(len(y_pair))
    for step in list(range(len(x_pair))):
       x_diff[step] =  np.median(med_dist_x - (x[r]+x_pair[step]))
       y_diff[step] = np.median(med_dist_y - (y[r]+y_pair[step]))

    add = abs(x_diff)+abs(y_diff) 
    min_ind = np.where(add == min(add))[0]

    # add[min_ind]
    # x_diff[min_ind]
    # y_diff[min_ind]
    x_offset = x_pair[min_ind][0]
    y_offset = y_pair[min_ind][0]
    
    new_x_smooth[r] = x[r] +  x_offset
    new_y_smooth[r] = y[r] + y_offset

    # avg_x_diff = np.median(med_dist_x - x[r])
    # avg_y_diff = np.median(med_dist_y - y[r])
    # if np.median(x) < 0:
    #     x_perc = 10
    # if np.median(x) >= 0:
    #     x_perc = 90
    # if np.median(y) < 0:
    #     y_perc = 10
    # if np.median(y) >= 0:
    #     y_perc = 90
    # avg_x_diff = np.percentile(med_dist_x - x[r], 50)
    # avg_y_diff = np.percentile(med_dist_y - y[r], 50)
    # new_x_smooth[r] = x[r] -  avg_x_diff #0.0005 (rch=81140600121)
    # new_y_smooth[r] = y[r] + avg_y_diff #0.0004 (rch=81140600121)

    # plt.scatter(jrc.lon[pt_ind], jrc.lat[pt_ind], c='lightgrey',s=2)
    # plt.scatter(x[r], y[r], c='black', s=3)
    # plt.scatter(new_x_smooth[r], new_y_smooth[r], c='magenta', s=3)
    # plt.show()


'''
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
'''

#######################################################################################
#######################################################################################
#######################################################################################

unq_rchs = np.unique(rchs[np.where(flag == 1)])
ridx = np.where(np.in1d(rchs,unq_rchs)==True)[0]

# ridx = np.where(rchs == 81185000121)[0]
# test_x = x[ridx]-0.003
# test_y = y[ridx]+0.001

plt.scatter(jrc.lon, jrc.lat, c='lightgrey',s=2)
plt.scatter(x[ridx], y[ridx], c='black', s=3)
# plt.scatter(test_x, test_y, c='cyan', s=3)
plt.scatter(new_x_smooth[ridx], new_y_smooth[ridx], c='cyan', s=3)
plt.show()

df = pd.DataFrame(np.array([new_x_smooth[ridx], new_y_smooth[ridx], shift[ridx], rch[ridx], index[ridx]]).T)
df.rename(
    columns={
        0:"x",
        1:"y",
        2:"shift_flag",
        3:"reach_id",
        4:"cl_ind"
        },inplace=True)

df_geom = gp.GeoSeries(map(Point, zip(new_x_smooth[ridx], new_y_smooth[ridx])))
df['geometry'] = df_geom
df = gp.GeoDataFrame(df)
df.set_geometry(col='geometry')
df = df.set_crs(4326, allow_override=True)
outgpkg='/Users/ealtenau/Documents/SWORD_Dev/testing_files/shifting_tests/sword_shift_test_160W_70N.gpkg'
df.to_file(outgpkg, driver='GPKG', layer='points')








### code section that defined a bounding box based on orthogonals. 
# start = time.time()
# shift = np.where(flag == 1)[0]
# new_x = np.copy(x)
# new_y = np.copy(y)
# for ind in list(range(743,len(shift))):
#     print(ind, len(shift)-1)
#     p = shift[ind] #cl_id (4873082) is a tricky case....
#     x1 = x[pt_ind[p,1]] #was 1 & 2
#     x2 = x[pt_ind[p,2]]
#     y1 = y[pt_ind[p,1]]
#     y2 = y[pt_ind[p,2]]
#     dx = x1-x2
#     dy = y1-y2
#     dist = np.sqrt(dx*dx+dy*dy)
#     dx /= dist 
#     dy /= dist 
#     x3 = x1+(0.01/2)*dy
#     y3 = y1-(0.01/2)*dx
#     x4 = x1-(0.01/2)*dy
#     y4 = y1+(0.01/2)*dx
#     x5 = x2+(0.01/2)*dy
#     y5 = y2-(0.01/2)*dx
#     x6 = x2-(0.01/2)*dy
#     y6 = y2+(0.01/2)*dx
#     poly = Polygon([(x3, y3), (x4, y4), (x6, y6), (x5, y5)])
#     end = time.time()
#     # print('Time to Create Bounding Box: '+str(end-start))
#     start = time.time()
#     pt_mask = jrc_clip.within(poly)
#     jrc_pts = jrc_clip.loc[pt_mask]
#     end = time.time()
#     # print('Method 1: '+str(end-start))
#     new_x[shift[ind]] = np.median(jrc_pts['x'])
#     new_y[shift[ind]] = np.median(jrc_pts['y'])

    # plt.scatter(jrc_clip['x'], jrc_clip['y'], c='lightgrey',s=2)
    # plt.scatter(x, y, c='black', s=3)
    # plt.scatter(x1, y1, c='cyan', s=5)
    # plt.scatter(x2, y2, c='cyan', s=5)
    # plt.scatter(jrc_pts['x'], jrc_pts['y'], c='gold', s=5)
    # plt.show()  

