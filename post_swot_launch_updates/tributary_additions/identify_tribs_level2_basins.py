from __future__ import division
import os
if os.path.exists('/Users/ealteanau/Documents/SWORD_Dev/src/SWORD/post_swot_launch_updates/tributary_additions/'):
    os.chdir('/Users/ealteanau/Documents/SWORD_Dev/src/SWORD/post_swot_launch_updates/tributary_additions/')
else:
    os.chdir('/afs/cas.unc.edu/users/e/a/ealtenau/SWORD/post_swot_launch_updates/tributary_additions/')
import tributary_addition_functions as tfs
import pandas as pd
import numpy as np
import netCDF4 as nc
import time
import matplotlib.pyplot as plt
from scipy import spatial as sp
import argparse
import earthaccess
# from earthaccess import Auth, DataCollections, DataGranules, Store
# import requests
# import json
# import geopandas as gp
# import glob
# from pathlib import Path
# import zipfile
# from urllib.request import urlretrieve
# from json import dumps

parser = argparse.ArgumentParser()
parser.add_argument("basin", help="<Required> HYDROBASINS Level 2 Basin", type = int)
parser.add_argument("local_processing", help="'True' for local machine, 'False' for server", type = str)
args = parser.parse_args()

start_all = time.time()

# Input file(s).
if args.local_processing == 'True':
    main_dir = '/Users/ealteanau/Documents/SWORD_Dev/'
else:
    main_dir = '/afs/cas.unc.edu/depts/geological_sciences/pavelsky/students/ealtenau/SWORD_dev/'

sourceFile = open(main_dir+'swot_data/temp_data/log_file.txt', 'w')
tile_fn = main_dir+'swot_data/SWORD_v16_PassTile/nominal/csv/level2_basins_pass_tile_nominal.csv'
mhv_dir = main_dir+'inputs/MHV_SWORD/'
sword_dir = main_dir+'outputs/Reaches_Nodes/v16/netcdf/'
folder = main_dir+'swot_data/temp_data/tiles/'
if not os.path.exists(folder):
    os.makedirs(folder)

# Log into EarthData.
# auth = earthaccess.login(strategy="interactive", persist=True) #if you do not have a netrc created, this line will do so with your credentials
auth = earthaccess.login(strategy="netrc")  #if you have created a netrc prior with your NASA Earthdata credentials, use strategy="netrc" to login

# Read in tile information.
basin = args.basin
tiles_all = pd.read_csv(tile_fn)
tiles_subset = tiles_all.loc[np.where(tiles_all.PFAF_ID == basin)[0]]
region = np.unique(tiles_subset.continent)[0]
l2_basins = np.array(tiles_subset.PFAF_ID)
tiles = np.array(tiles_subset.pass_tile)
    
# Read in SWORD and MHV
sword = nc.Dataset(sword_dir+region+'_sword_v16.nc') #will be able to remove sword from script once trouble shooting is complete. 
mhv = nc.Dataset(mhv_dir+region+'_mhv_sword.nc', 'r+')
    
# Define date range to pull SWOT tiles. 
start_date = "2023-01-01 12:00:00"
end_date = "2023-12-30 19:43:00"

# Loop tiles within a level 2 basin.
print('Starting Basin ' + str(basin))
print('Starting Basin ' + str(basin) + ': ' + time.strftime("%d-%b-%Y %H:%M:%S"), file = sourceFile) 
for ind in list(range(len(tiles))):
    start = time.time()
    print(ind, len(tiles)-1, tiles[ind])

    # Search and download tiles. 
    granule = '*'+str(tiles[ind])+'*' 
    num_files = tfs.download_data(granule, start_date, end_date, folder)
        
    if len(os.listdir(folder)) == 0:
        print(str(ind) + ', ' + str(tiles[ind]) + 
                ' - no SWOT data for tile', file = sourceFile)
        continue
        
    else:
        #read in / aggregate data
        print(str(ind) + ', ' + str(tiles[ind]) + 
                ' - ' + str(num_files) + ' tiles', file = sourceFile)
        pixc_lon, pixc_lat = tfs.read_pixc_data(folder)
        for f in os.listdir(folder):
            os.remove(folder+f) 

        #subset sword to tile extent.
        sword_lon, sword_lat, sword_tribs = tfs.subset_sword(sword, pixc_lon, pixc_lat) 

        #subset merrit hydro vector to tile extent.
        mhv_lon, mhv_lat, \
            mhv_flag, mhv_seg, \
                mhv_dist, indexes = tfs.subset_mhv(mhv, pixc_lon, pixc_lat) 
        index = np.where(indexes == True)[0]

        #spatial query between pixc and mhv
        mhv_pts = np.vstack((mhv_lon, mhv_lat)).T
        pixc_pts = np.vstack((pixc_lon,pixc_lat)).T
        kdt = sp.cKDTree(pixc_pts)
        pt_dist, pt_ind = kdt.query(mhv_pts, k = 5)

        keep = np.where((pt_dist[:,0] < 0.002) & (mhv_flag == 0))[0]
        keep_array = np.zeros(len(mhv_seg))
        keep_array[keep] = 1
        add_channels = np.unique(mhv_seg[keep])

        ### length or % segment threshold. 
        keep_channels = []
        for idx in list(range(len(add_channels))):
            seg = np.where(mhv_seg == add_channels[idx])[0]
            flagged = np.where(keep_array[seg] == 1)[0]
            perc = (len(flagged)/len(seg))*100
            if len(flagged) > 55 or perc > 90:
                keep_channels.append(add_channels[idx])
                
        ### will need to set length or # of points criteria for adding.
        add_points = np.where(np.in1d(mhv_seg, keep_channels))[0]
        # update_mhv(mhv, index, add_points)

        if len(add_points) > 0:
            plt.figure(figsize=(7,7))
            plt.scatter(pixc_lon, pixc_lat, s = 5, c='lightgrey')
            plt.scatter(mhv_lon, mhv_lat, s = 1, c='black')
            plt.scatter(sword_lon, sword_lat, s = 3, c='deepskyblue')
            plt.scatter(mhv_lon[add_points], mhv_lat[add_points], s = 3, c='crimson')
            plt.title(tiles[ind])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig('/Users/ealteanau/Documents/SWORD_Dev/testing_files/test_trib_add_plots/'+tiles[ind]+'.png')
            plt.close()

    end = time.time()
    print('Time Finish Tile: ' + str(np.round((end-start)/60, 2)))

'''
### need to filter at level two basin scale. 
mhv_l2 = np.array([int(str(ind)[0:2]) for ind in 
                   mhv.groups['centerlines'].variables['basin_code'][:]])
l2_pts = np.where(mhv_l2 == basin)[0]
l2_lon = mhv.groups['centerlines'].variables['basin_code'][l2_pts]
l2_lat = mhv.groups['centerlines'].variables['basin_code'][l2_pts]
#create filter... 
'''

#close mhv and log file. 
mhv.close()
sword.close()
sourceFile.close()
end_all = time.time()
print('Time Finish Basin '+str(basin)+
      ': '+str(np.round((end_all-start_all)/3600, 2))+' hrs')

'''
# test_channels = [47953]
# test_pts = np.where(np.in1d(mhv_seg, test_channels))[0]

plt.figure(figsize=(7,7))
plt.scatter(pixc_lon, pixc_lat, s = 5, c='lightgrey')
plt.scatter(mhv_lon, mhv_lat, s = 1, c='black')
plt.scatter(sword_lon, sword_lat, s = 3, c='deepskyblue')
plt.scatter(mhv_lon[add_points], mhv_lat[add_points], s = 3, c='crimson')
# plt.scatter(mhv_lon[test_pts], mhv_lat[test_pts], s = 3, c='gold')
plt.title(tiles[subset[ind]])
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('test_plots/'+tiles[subset[ind]]+'.png')
plt.close()
# plt.show()

'''