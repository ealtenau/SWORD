# -*- coding: utf-8 -*-
"""
Flagging Offset SWORD reaches using JRC Water Occurance
(channel_shifting_flag_jrc.py)
=========================================================

This script uses SWORD's proximity to JRC water occurance
data to flag SWORD reaches that have a consistent offset
from the river center location. It is limited to single
channel rivers. 

Outputs are geopackage files of SWORD centerline points
with a shifting flag attribute for each Pfafstetter level 2
basin. 

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python path/to/channel_shifting_flag_jrc.py NA v17

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
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
import src.updates.geo_utils as geo 

###############################################################################
######################## Reading and Writing Functions ########################
###############################################################################

def read_jrc(jrc, jrc_fn):
    """
    Reads in the Joint Research Center's (JRC) water occurance 
    tiff files.

    Parmeters
    ---------
    jrc: rasterio.io.DatasetReader
        Raster of JRC water occurance data. 
    jrc_fn: str
        JRC file path. 

    Returns
    -------
    jrc_arr: obj
        Object containing JRC location and attribute
        data. 
    
    """
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
    jrc_arr = geo.Object()
    jrc_arr.lon = lon[keep]
    jrc_arr.lat = lat[keep]
    jrc_arr.vals = vals[keep]

    return jrc_arr

###############################################################################

def clip_sword(sword, jrc):
    """
    Subsets SWORD data to the extent of the JRC
    raster tile. 

    Parmeters
    ---------
    sword: netCDF4.Dataset()
        SWORD database in netCDF format. 
    jrc: obj
        Object containing JRC location and attribute
        data. 

    Returns
    -------
    sword_lon: numpy.array()
        SWORD centerline dimension longitude (WGS 84, EPSG:4326). 
    sword_lat: numpy.array()
        SWORD centerline dimension latitude (WGS 84, EPSG:4326).
    sword_rch_id: numpy.array()
        SWORD reach IDs in the centerline dimension.
    sword_cl_id: numpy.array()
        SWORD centerline IDs ().

    """

    #bounding box of JRC data. 
    xmin = np.min(jrc.bounds[0])
    xmax = np.max(jrc.bounds[2])
    ymin = np.min(jrc.bounds[1])
    ymax = np.max(jrc.bounds[3])
    ll = np.array([xmin, ymin])  # lower-left
    ur = np.array([xmax, ymax])  # upper-right
    
    #sword coordinates and attributes. 
    lon = np.array(sword.groups['centerlines'].variables['x'][:])
    lat = np.array(sword.groups['centerlines'].variables['y'][:])
    rch_id = np.array(sword.groups['centerlines'].variables['reach_id'][0,:])
    ind = np.array(sword.groups['centerlines'].variables['cl_id'][:])
    pts = np.array([(lon[i], lat[i]) for i in range(len(lon))])

    #finding sword points within jrc bounding box. 
    idx = np.all(np.logical_and(ll <= pts, pts <= ur), axis=1)
    sword_lon = lon[idx]
    sword_lat = lat[idx]
    sword_rch_id = rch_id[idx]
    sword_cl_id = ind[idx]

    return sword_lon, sword_lat, sword_rch_id, sword_cl_id

###############################################################################

def read_sword(sword_fn, jrc):
    """
    Description

    Parmeters
    ---------
    sword_fn: str
        SWORD file path. 
    jrc: obj
        Object containing JRC location and attribute
        data.

    Returns
    -------
    data: obj
        Object containing SWORD location and attribute
        data within JRC bounding box.
    
    """

    #reading SWORD and finding which points are within JRC tile extent. 
    sword = nc.Dataset(sword_fn)
    data = geo.Object()
    data.lon, data.lat, data.reach_id, data.ind = clip_sword(sword, jrc)

    #adding relevant SWORD attributes within clipped window. 
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
args = parser.parse_args()

start_all = time.time()
region = args.region
version = args.version

# Input file(s).
swd_dir = main_dir+'/data/outputs/Reaches_Nodes/'
tiff_dir = main_dir+'/data/inputs/JRC_Water_Occurance/'

sword_fn = swd_dir+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
jrc_dir = tiff_dir
jrc_files = os.listdir(jrc_dir)
jrc_files = np.array([f for f in jrc_files if '.tif' in f])

#loop through jrc tiles. 
for f in list(range(len(jrc_files))):

    jrc_tile = jrc_files[f][11:18]
    print(f, len(jrc_files)-1, jrc_tile)
    jrc_fn = jrc_dir+jrc_files[f]
    jrc_arr = rasterio.open(jrc_fn)

    print('Reading SWORD Data')
    start = time.time()
    sword = read_sword(sword_fn, jrc_arr)
    end = time.time()
    print(str(np.round((end-start),2))+' sec')

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

        threshold = geo.meters_to_degrees(radius, np.median(sword.lat[rch]))
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
    outgpkg=jrc_dir+region+'/'+jrc_tile+'_sword_'+version+'_shift_flag.gpkg'
    df.to_file(outgpkg, driver='GPKG', layer='headwaters')

    del jrc; del jrc_arr; del df; del df_geom; del jrc_pts; del kdt; del pt_ind; del pt_dist

    end = time.time()
    print(str(np.round((end-start),2))+' sec')

end_all = time.time()
print('FINISHED ALL TILES IN: '+str(np.round((end-start)/3600,2))+' hrs')

#### PLOTS
# import matplotlib.pyplot as plt
# f = np.where(sword.shift_flag==1)[0]
# plt.scatter(jrc.lon, jrc.lat, c='lightgrey',s=2)
# plt.scatter(sword.lon, sword.lat, c='black')
# plt.scatter(sword.lon[f], sword.lat[f], c='red')
# plt.show()

# geo.meters_to_degrees(1000, np.median(sword.lat[rch]))
# plt.scatter(sword.lon, sword.lat, c='blue', s = 3)
# plt.scatter(new_lon, new_lat, c='magenta', s = 3)
# plt.scatter(new_lon_smooth, new_lat_smooth, c='cyan', s = 3)
# plt.show()
