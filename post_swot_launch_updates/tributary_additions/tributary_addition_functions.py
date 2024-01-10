import requests
import json
import geopandas as gp
import glob
from pathlib import Path
import pandas as pd
import os
import zipfile
from urllib.request import urlretrieve
from json import dumps
import earthaccess
from earthaccess import Auth, DataCollections, DataGranules, Store
import numpy as np
import numpy.ma as ma
import netCDF4 as nc 

#############################################################################################

def download_data(granule, start_date, end_date, folder):
    results = earthaccess.search_data(short_name = 'SWOT_L2_HR_PIXC_1.1' ,
                                            temporal = (start_date, end_date),
                                            granule_name = granule)
    if len(results) > 0:
        downloads = []
        for g in results:
            for l in earthaccess.results.DataGranule.data_links(g):
                if 'archive.swot.podaac.earthdata.nasa.gov/podaac-swot-ops-cumulus-protected/' in l:
                    downloads.append(l)
        if len(downloads) > 4:
            downloads = downloads[0:4]            
        earthaccess.download(downloads, folder)

#############################################################################################

def read_pixc_data(folder):
    files = os.listdir(folder)
    for f in list(range(len(files))):
        pixc = nc.Dataset(folder+files[f])
        pixc_lat = ma.getdata(pixc.groups['pixel_cloud'].variables['latitude'][:])
        pixc_lon = ma.getdata(pixc.groups['pixel_cloud'].variables['longitude'][:])
        pixc_class = ma.getdata(pixc.groups['pixel_cloud'].variables['classification'][:])
        pixc_lat = pixc_lat[np.where(pixc_class == 4)]
        pixc_lon = pixc_lon[np.where(pixc_class == 4)]
        # pixc_lat = pixc_lat[np.where((pixc_class > 2) & (pixc_class < 5))]
        # pixc_lon = pixc_lon[np.where((pixc_class > 2) & (pixc_class < 5))]
        # print(np.min(pixc.groups['pixel_cloud'].variables['latitude'][:]), np.max(pixc.groups['pixel_cloud'].variables['latitude'][:]))
        # print(np.min(pixc.groups['pixel_cloud'].variables['longitude'][:]), np.max(pixc.groups['pixel_cloud'].variables['longitude'][:]))
        pixc.close()
        if f == 0:
            pixc_x = np.array(pixc_lon)
            pixc_y = np.array(pixc_lat)
        else:
            pixc_x = np.append(pixc_x, pixc_lon)
            pixc_y = np.append(pixc_y, pixc_lat)

    return pixc_x, pixc_y

#############################################################################################

def subset_sword(sword, pixc_lon, pixc_lat):
    xmin = np.min(pixc_lon)
    xmax = np.max(pixc_lon)
    ymin = np.min(pixc_lat)
    ymax = np.max(pixc_lat)
    ll = np.array([xmin, ymin])  # lower-left
    ur = np.array([xmax, ymax])
    
    sword_lon_all = sword.groups['nodes'].variables['x'][:]
    sword_lat_all =sword.groups['nodes'].variables['y'][:]
    sword_tribs_all =sword.groups['nodes'].variables['trib_flag'][:]
    sword_points = [(sword_lon_all[i], sword_lat_all[i]) for i in range(len(sword_lon_all))]
    sword_pts = np.array(sword_points)
    sword_idx = np.all(np.logical_and(ll <= sword_pts, sword_pts <= ur), axis=1)
    sword_lon = sword_lon_all[sword_idx]
    sword_lat = sword_lat_all[sword_idx]
    sword_tribs = sword_tribs_all[sword_idx]
    
    return sword_lon, sword_lat, sword_tribs

#############################################################################################

def subset_mhv(mhv, pixc_lon, pixc_lat):
    xmin = np.min(pixc_lon)
    xmax = np.max(pixc_lon)
    ymin = np.min(pixc_lat)
    ymax = np.max(pixc_lat)
    ll = np.array([xmin, ymin])  # lower-left
    ur = np.array([xmax, ymax])

    mhv_lon_all = mhv.groups['centerlines'].variables['x'][:]
    mhv_lat_all = mhv.groups['centerlines'].variables['y'][:]
    mhv_flag_all =  mhv.groups['centerlines'].variables['swordflag'][:]
    mhv_seg_all =  mhv.groups['centerlines'].variables['segID'][:]
    mhv_dist_all = mhv.groups['centerlines'].variables['segDist'][:]
    mhv_points = [(mhv_lon_all[i], mhv_lat_all[i]) for i in range(len(mhv_lon_all))]
    mhv_pts = np.array(mhv_points)
    mhv_idx = np.all(np.logical_and(ll <= mhv_pts, mhv_pts <= ur), axis=1)
    mhv_lon = mhv_lon_all[mhv_idx]
    mhv_lat = mhv_lat_all[mhv_idx]
    mhv_flag = mhv_flag_all[mhv_idx]
    mhv_seg = mhv_seg_all[mhv_idx]
    mhv_dist = mhv_dist_all[mhv_idx]

    return mhv_lon, mhv_lat, mhv_flag, mhv_seg, mhv_dist, mhv_idx

#############################################################################################