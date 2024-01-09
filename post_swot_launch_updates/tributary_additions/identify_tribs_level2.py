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

folder = '/Users/ealteanau/Documents/SWORD_Dev/swot_data/temp_data/'
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
    
    #read in / aggregate data

    #read in format sword

    #spatial query