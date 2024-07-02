from __future__ import division
import os
import math
from shapely.geometry import Point
import numpy as np
from osgeo import gdal
import rasterio
from scipy import spatial as sp
import geopandas as gp
import pandas as pd
import time
import netCDF4 as nc
import argparse
import matplotlib.pyplot as plt

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

def read_jrc(jrc, jrc_fn):
    
    #Getting vals
    jrc_ras = gdal.Open(jrc_fn)
    vals = np.array(jrc_ras.GetRasterBand(1).ReadAsArray()).flatten()

    # Getting Coordinates
    # jrc = rasterio.open(jrc_fn)
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

def clip_sword(sword, jrc):
    xmin = np.min(jrc.bounds[0])
    xmax = np.max(jrc.bounds[2])
    ymin = np.min(jrc.bounds[1])
    ymax = np.max(jrc.bounds[3])
    ll = np.array([xmin, ymin])  # lower-left
    ur = np.array([xmax, ymax])  # upper-right
    
    lon = np.array(sword.groups['centerlines'].variables['x'][:])
    lat = np.array(sword.groups['centerlines'].variables['y'][:])
    rch_id = np.array(sword.groups['centerlines'].variables['reach_id'][0,:])
    ind = np.array(sword.groups['centerlines'].variables['cl_id'][:])
    pts = np.array([(lon[i], lat[i]) for i in range(len(lon))])

    idx = np.all(np.logical_and(ll <= pts, pts <= ur), axis=1)
    sword_lon = lon[idx]
    sword_lat = lat[idx]
    sword_rch_id = rch_id[idx]
    sword_cl_id = ind[idx]

    return sword_lon, sword_lat, sword_rch_id, sword_cl_id

###############################################################################

def read_sword(sword_fn, jrc):
    
    sword = nc.Dataset(sword_fn)
    data = Object()
    data.lon, data.lat, data.reach_id, data.ind = clip_sword(sword, jrc)

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

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Region", type = str)
parser.add_argument("version", help="<Required> Version", type = str)
parser.add_argument("local_processing", help="'True' for local machine, 'False' for server", type = str)
args = parser.parse_args()

start_all = time.time()
region = args.region
version = args.version

# Input file(s).
if args.local_processing == 'True':
    main_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'
    tiff_dir = '/Users/ealtenau/Documents/SWORD_Dev/inputs/JRC_Water_Occurance/'
else:
    main_dir = '/afs/cas.unc.edu/depts/geological_sciences/pavelsky/students/ealtenau/SWORD_dev/outputs/Reaches_Nodes/'
    tiff_dir = '/afs/cas.unc.edu/depts/geological_sciences/pavelsky/students/ealtenau/SWORD_dev/inputs/JRC_Water_Occurance/'

sword_fn = main_dir+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
jrc_dir = tiff_dir
jrc_files = os.listdir(jrc_dir)
jrc_files = np.array([f for f in jrc_files if '.tif' in f])

#### loop through jrc tiles. 
for f in list(range(len(jrc_files))):
    print(f, len(jrc_files)-1)
    jrc_tile = jrc_files[f][11:18]
    jrc_fn = jrc_dir+jrc_files[f]
    jrc_arr = rasterio.open(jrc_fn)

    # print('Reading SWORD Data')
    start = time.time()
    sword = read_sword(sword_fn, jrc_arr)
    end = time.time()
    # print(str(np.round((end-start),2))+' sec')

    if len(sword.lon) == 0:
        continue

    print('Reading JRC Data')
    start = time.time()
    jrc = read_jrc(jrc_arr, jrc_fn)
    end = time.time()
    print(str(np.round((end-start)/60,2))+' mins')

    print('Starting Spatial Join')
    start = time.time()
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
    end = time.time()
    print(str(np.round((end-start)/60,2))+' min')

    print('Starting SWORD Filter')
    start = time.time()
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
            radius = 100
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

        if perc > 25 and chan == 1: #was perc > 50.
            sword.shift_flag[rch] = 1

    end = time.time()
    print(str(np.round((end-start),2))+' sec')

    ### create geopackage file of node locations to save flag by level 2 basin. 
    print('Saving Data')
    df = pd.DataFrame(np.array([sword.lon, sword.lat, sword.shift_flag, sword.reach_id, sword.ind]).T)
    df.rename(
        columns={
            0:"x",
            1:"y",
            2:"shift_flag",
            3:"reach_id",
            4:"cl_ind"
            },inplace=True)

    df_geom = gp.GeoSeries(map(Point, zip(sword.lon, sword.lat)))
    df['geometry'] = df_geom
    df = gp.GeoDataFrame(df)
    df.set_geometry(col='geometry')
    df = df.set_crs(4326, allow_override=True)
    outgpkg=jrc_dir+region+'/'+jrc_tile+'_sword_v17_shift_flag.gpkg'
    df.to_file(outgpkg, driver='GPKG', layer='headwaters')

    end = time.time()
    print(str(np.round((end-start),2))+' sec')

end_all = time.time()
print('FINISHED ALL TILES IN: '+str(np.round((end-start)/3600,2))+' hrs')

# f = np.where(sword.shift_flag==1)[0]
# plt.scatter(jrc.lon, jrc.lat, c='lightgrey',s=2)
# plt.scatter(sword.lon, sword.lat, c='black')
# plt.scatter(sword.lon[f], sword.lat[f], c='red')
# plt.show()

# meters_to_degrees(1000, np.median(sword.lat[rch]))
# plt.scatter(sword.lon, sword.lat, c='blue', s = 3)
# plt.scatter(new_lon, new_lat, c='magenta', s = 3)
# plt.scatter(new_lon_smooth, new_lat_smooth, c='cyan', s = 3)
# plt.show()
