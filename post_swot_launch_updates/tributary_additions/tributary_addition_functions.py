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

def download_data_verbose(granule, start_date, end_date, folder):
    results = earthaccess.search_data(short_name = 'SWOT_L2_HR_PIXC_1.1' ,
                                            temporal = (start_date, end_date),
                                            granule_name = granule)
    if len(results) > 0:
        downloads_all = []
        for g in results:
            for l in earthaccess.results.DataGranule.data_links(g):
                if 'archive.swot.podaac.earthdata.nasa.gov/podaac-swot-ops-cumulus-protected/' in l:
                    downloads_all.append(l)
        if len(downloads_all) > 4:
            downloads = downloads_all[0:4]            
        earthaccess.download(downloads, folder)
    
    return(len(results))

#############################################################################################

def download_data(auth, granule, start_date, end_date, folder):
    Query = DataGranules(auth).short_name("SWOT_L2_HR_PIXC_1.1").temporal(start_date, end_date).granule_name(granule)
    num_hits = Query.hits()
    results = Query.get()
    if num_hits > 0:
        downloads_all = []
        for g in list(range(num_hits)):
            for l in results[g].data_links():
                if 'archive.swot.podaac.earthdata.nasa.gov/podaac-swot-ops-cumulus-protected/' in l:
                    downloads_all.append(l)
        if len(downloads_all) > 4:
            downloads = downloads_all[0:4]            
        earthaccess.download(downloads, folder)
    return(num_hits)

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

def update_mhv(mhv, index, add_points):
    if 'sword_add' in mhv.groups['centerlines'].variables:
        mhv.groups['centerlines'].variables['sword_add'][index[add_points]] = 1
    else:
        mhv.groups['centerlines'].createVariable('sword_add', 'i8', ('num_points',))
        mhv.groups['centerlines'].variables['sword_add'][index[add_points]] = 1

#############################################################################################

def filter_l2_flags(mhv, basin, outfile):
    
    segs = mhv.groups['centerlines'].variables['new_segs'][:]
    add = mhv.groups['centerlines'].variables['sword_add'][:]
    flag = mhv.groups['centerlines'].variables['swordflag_filt'][:]

    mhv_l2 = np.array([int(str(ind)[0:2]) for ind in 
                   mhv.groups['centerlines'].variables['basin_code'][:]])
    l2_pts = np.where(mhv_l2 == basin)[0]
    l2_segs = segs[l2_pts]
    l2_add = add[l2_pts]
    l2_flag = flag[l2_pts]

    new_flag = np.zeros(len(l2_segs))
    unq_segs = np.unique(l2_segs[np.where(l2_add > 0)[0]])
    for ind in list(range(len(unq_segs))):
        pts = np.where(l2_segs == unq_segs[ind])[0]
        valid = np.where(l2_flag[pts] == 0)[0]
        flagged = np.where(l2_add[pts] == 1)[0]
        perc = (len(flagged)/len(valid))*100
        num_pts = len(flagged)
        if len(pts) < 100: #less than ~10 km
            if perc > 90:
                new_flag[pts] = 1
        elif 100 <= len(pts) <= 1000: #between ~10 km and ~100 km 
            if perc > 70:
                new_flag[pts] = 1
        else: # greater than ~100 km
            if perc > 50:
                new_flag[pts] = 1

    #update MHV netcdf. 
    if 'sword_add_filt' in mhv.groups['centerlines'].variables:
        mhv.groups['centerlines'].variables['sword_add_filt'][l2_pts] = new_flag
    else:
        mhv.groups['centerlines'].createVariable('sword_add_filt', 'i8', ('num_points',))
        mhv.groups['centerlines'].variables['sword_add_filt'][l2_pts] = new_flag

    #save data as geopackage file.
    gpkg = gp.GeoDataFrame([
        mhv.groups['centerlines'].variables['x'][l2_pts],
        mhv.groups['centerlines'].variables['y'][l2_pts],
        mhv.groups['centerlines'].variables['cl_id'][l2_pts],
        mhv.groups['centerlines'].variables['segID'][l2_pts],
        mhv.groups['centerlines'].variables['strmorder'][l2_pts],
        mhv.groups['centerlines'].variables['segInd'][l2_pts],
        mhv.groups['centerlines'].variables['segDist'][l2_pts],
        mhv.groups['centerlines'].variables['p_width'][l2_pts],
        mhv.groups['centerlines'].variables['p_height'][l2_pts],
        mhv.groups['centerlines'].variables['flowacc'][l2_pts],
        mhv.groups['centerlines'].variables['swordflag_filt'][l2_pts],
        mhv.groups['centerlines'].variables['new_segs'][l2_pts],
        mhv.groups['centerlines'].variables['new_segs_ind'][l2_pts],
        mhv.groups['centerlines'].variables['new_segs_dist'][l2_pts],
        mhv.groups['centerlines'].variables['new_segs_eps'][l2_pts],
        mhv.groups['centerlines'].variables['reach_id'][l2_pts],
        mhv.groups['centerlines'].variables['rch_len'][l2_pts],
        mhv.groups['centerlines'].variables['node_id'][l2_pts],
        mhv.groups['centerlines'].variables['node_len'][l2_pts],
        mhv.groups['centerlines'].variables['sword_add'][l2_pts],
        mhv.groups['centerlines'].variables['sword_add_filt'][l2_pts]
    ]).T 

    gpkg.rename(
        columns={
            0:"x",
            1:"y",
            2:"cl_id",
            3:"segID",
            4:"strmorder",
            5:"segInd",
            6:"segDist",
            7:"p_width",
            8:"p_height",
            9:"flowacc",
            10:"swordflag_filt",
            11:"new_segs",
            12:"new_segs_ind",
            13:"new_segs_dist",
            14:"new_segs_eps",
            15:"reach_id",
            16:"rch_len",
            17:"node_id",
            18:"node_len",
            19:"sword_add",
            20:"sword_add_filt",
            },inplace=True)

    gpkg = gpkg.apply(pd.to_numeric, errors='ignore') 
    geom = gp.GeoSeries(map(Point, zip(mhv.groups['centerlines'].variables['x'][l2_pts], 
                                       mhv.groups['centerlines'].variables['y'][l2_pts])))
    gpkg['geometry'] = geom
    gpkg = gp.GeoDataFrame(gpkg)
    gpkg.set_geometry(col='geometry')
    gpkg = gpkg.set_crs(4326, allow_override=True)
    #save data.
    outgpkg = outfile
    gpkg.to_file(outgpkg, driver='GPKG', layer='points')           

#############################################################################################