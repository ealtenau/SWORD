from __future__ import division
import os
if os.path.exists('/Users/ealteanau/Documents/SWORD_Dev/src/SWORD/post_swot_launch_updates/tributary_additions/'):
    os.chdir('/Users/ealteanau/Documents/SWORD_Dev/src/SWORD/post_swot_launch_updates/tributary_additions/')
else:
    os.chdir('/afs/cas.unc.edu/users/e/a/ealtenau/SWORD/post_swot_launch_updates/tributary_additions/')
import tributary_addition_functions as tfs
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

sourceFile = open('/Users/ealteanau/Documents/SWORD_Dev/swot_data/temp_data/out_messages.txt', 'w')
print('Hello, Python!', file = sourceFile)

folder = '/Users/ealteanau/Documents/SWORD_Dev/swot_data/temp_data/tiles/'
if not os.path.exists(folder):
    os.makedirs(folder)

# auth = earthaccess.login(strategy="interactive", persist=True) #if you do not have a netrc created, this line will do so with your credentials
auth = earthaccess.login(strategy="netrc")  #if you have created a netrc prior with your NASA Earthdata credentials, use strategy="netrc" to login

tile_fn = '/Users/ealteanau/Documents/SWORD_Dev/swot_data/SWORD_v16_PassTile/nominal/csv/level2_basins_pass_tile_nominal.csv'
tiles = pd.read_csv(tile_fn)

start_date = "2023-01-01 12:00:00"
end_date = "2023-12-30 19:43:00"

subset = np.where(tiles.PFAF_ID == 12)[0]

for ind in list(range(len(subset))):
    print(ind, len(subset)-1)
    granule = '*'+str(tiles.pass_tile[subset[ind]])+'*' 
    tfs.download_data(granule, start_date, end_date, folder)

    if len(os.listdir(folder)) == 0:
        print(str(tiles.pass_tile[subset[ind]]) + ' - no SWOT data for tile', file = sourceFile)
        continue
    
    else:
        #read in / aggregate data
        print(str(tiles.pass_tile[subset[ind]]) + ' - ' + str(len(os.listdir(folder))) + 'tiles', file = sourceFile)
        pixc_lon, pixc_lat = tfs.read_pixc_data(folder)
        # for f in os.listdir(folder):
        #     os.remove(folder+f) 

        #read in format sword
        sword_lon, sword_lat, sword_tribs = tfs.read_sword(sword_dir, pixc_lon, pixc_lat, cont)

        #read in and format merrit hydro vector
        mhv_lon, mhv_lat, mhv_flag, mhv_seg, mhv_dist = tfs.read_mhv(mhv_dir, pixc_lon, pixc_lat, cont)

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
        for ind in list(range(len(add_channels))):
            seg = np.where(mhv_seg == add_channels[ind])[0]
            flagged = np.where(keep_array[seg] == 1)[0]
            perc = (len(flagged)/len(seg))*100
            if len(flagged) > 55 or perc > 90:
                keep_channels.append(add_channels[ind])
            
        ### will need to set length or # of points criteria for adding.
        add_points = np.where(np.in1d(mhv_seg, keep_channels))[0]

        # test_channels = [47953]
        # test_pts = np.where(np.in1d(mhv_seg, test_channels))[0]

        plt.figure(figsize=(7,7))
        plt.scatter(pixc_lon, pixc_lat, s = 5, c='lightgrey')
        plt.scatter(mhv_lon, mhv_lat, s = 1, c='black')
        plt.scatter(sword_lon, sword_lat, s = 3, c='deepskyblue')
        plt.scatter(mhv_lon[add_points], mhv_lat[add_points], s = 3, c='crimson')
        # plt.scatter(mhv_lon[test_pts], mhv_lat[test_pts], s = 3, c='gold')
        plt.title(str(pixc_fn[-55:-43]))
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.savefig('test_plots/'+str(pixc_fn[-55:-43])+'.png')
        # plt.close()
        plt.show()


sourceFile.close()