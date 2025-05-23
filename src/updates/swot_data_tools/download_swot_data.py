import os
main_dir = os.getcwd()
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

folder = main_dir+'/data/swot_data/temp_data/'
if not os.path.exists(folder):
    os.makedirs(folder)


### METHOD 1 (print statements):

# auth = earthaccess.login(strategy="interactive", persist=True) #if you do not have a netrc created, this line will do so with your credentials
auth = earthaccess.login(strategy="netrc")  #if you have created a netrc prior with your NASA Earthdata credentials, use strategy="netrc" to login

tile_fn = main_dir+'/data/swot_data/SWORD_v16_PassTile/nominal/csv/level2_basins_pass_tile_nominal.csv'
tiles = pd.read_csv(tile_fn)

start_date = "2023-01-01 12:00:00"
end_date = "2023-12-30 19:43:00"
granule = '*007_153R*'

results = earthaccess.search_data(short_name = 'SWOT_L2_HR_PIXC_1.1' ,
                                  temporal = (start_date, end_date),
                                  granule_name = granule)
downloads = []
for g in results:
    for l in earthaccess.results.DataGranule.data_links(g):
        if 'archive.swot.podaac.earthdata.nasa.gov/podaac-swot-ops-cumulus-protected/' in l:
            downloads.append(l)            
print(len(downloads))

earthaccess.download(downloads, folder)


#### METHOD 2 (no print statements):

auth = Auth()
auth.login(strategy="netrc")
# auth.authenticated

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


### testing searching for collections:
Query = DataCollections(auth).keyword('SWOT Pixel Cloud')
print(f'Collections found: {Query.hits()}')
collections = Query.fields(['ShortName','Version']).get()

#version 1.0
Query = DataGranules(auth).short_name("SWOT_L2_HR_PIXC_1.0")
Query = DataGranules(auth).concept_id("C2296989353-POCLOUD") #does not work...
Query.hits()

