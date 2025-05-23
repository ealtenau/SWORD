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

folder = main_dir+'/data/swot_data/Yukon/pixc/'
if not os.path.exists(folder):
    os.makedirs(folder)


### METHOD 1 (print statements):

# auth = earthaccess.login(strategy="interactive", persist=True) #if you do not have a netrc created, this line will do so with your credentials
auth = earthaccess.login(strategy="netrc")  #if you have created a netrc prior with your NASA Earthdata credentials, use strategy="netrc" to login

# tile_fn = 'main_dir+'/data/swot_data/Yukon/yukon_tiles.csv'
# tiles = pd.read_csv(tile_fn)
# tile_names = np.array(tiles['pass_tile'])
tile_names = np.array(['197_175L','016_134L','016_134R'])

start_date = "2024-01-01 12:00:00"
end_date = "2024-2-29 19:43:00"


for t in list(range(2,len(tile_names))):
    print(t, len(tile_names)-1, tile_names[t])
    granule = '*'+str(tile_names[t])+'*'
    results = earthaccess.search_data(short_name = 'SWOT_L2_HR_PIXC_2.0' ,
                                    temporal = (start_date, end_date),
                                    granule_name = granule)
    downloads = []
    for g in results:
        for l in earthaccess.results.DataGranule.data_links(g):
            if 'archive.swot.podaac.earthdata.nasa.gov/podaac-swot-ops-cumulus-protected/' in l:
                downloads.append(l)            
    # print(len(downloads))
    earthaccess.download(downloads[0], folder) #only download latest tile. 

### testing searching for collections:
# Query = DataCollections(auth).keyword('SWOT Pixel Cloud')
# print(f'Collections found: {Query.hits()}')
# collections = Query.fields(['ShortName','Version']).get()


