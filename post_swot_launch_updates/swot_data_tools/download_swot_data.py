import requests
import json
import geopandas as gpd
import glob
from pathlib import Path
import pandas as pd
import os
import zipfile
from urllib.request import urlretrieve
from json import dumps
import earthaccess
from earthaccess import Auth, DataCollections, DataGranules, Store

# auth = earthaccess.login(strategy="interactive", persist=True) #if you do not have a netrc created, this line will do so with your credentials
auth = earthaccess.login(strategy="netrc")  #if you have created a netrc prior with your NASA Earthdata credentials, use strategy="netrc" to login


### Search via bounding box.
# Data search.
results = earthaccess.search_data(concept_id="C2263384307-POCLOUD", 
                                  bounding_box=(-124.848974,24.396308,-66.885444,49.384358))

# subset data by reach or node. 
downloads = []
for g in results:
    for l in earthaccess.results.DataGranule.data_links(g):
        if 'https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/' in l:
            if 'Reach' in l:
                downloads.append(l)            
print(len(downloads))


#Create folder to house downloaded data 
folder = Path("SWOT_sample_files")
#newpath = r'SWOT_sample_files' 
if not os.path.exists(folder):
    os.makedirs(folder)

#download data
earthaccess.download(downloads, "./SWOT_sample_files")

#Retrieves granule from the day we want, in this case by passing to `earthdata.search_data` function the data collection shortname, temporal bounds, and for restricted data one must specify the search count
river_results = earthaccess.search_data(short_name = 'SWOT_L2_HR_PIXC_004_564_116L', 
                                        temporal = ('2023-10-01 00:00:00', '2023-10-30 23:59:59'),
                                        count=2000) #for restricted datasets, need to specify count number (1-2000)

print(len(river_results))



'''
# Works:

river_results = earthaccess.search_data(short_name = 'SWOT_L2_HR_RIVERSP_1.1', 
                                        temporal = ('2023-04-08 00:00:00', '2023-04-25 23:59:59'),
                                        granule_name = '*Reach*_013_NA*', # here we filter by Reach files (not node), pass #13 and continent code=NA
                                        count=2000) 

'''

