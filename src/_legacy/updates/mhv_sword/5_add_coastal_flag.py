"""
Adding Coastal Flag to MHV-SWORD dataset
(5_add_coastal_flag.py).
===================================================

This script attaches a binary flag indicating if 
points in the MHV-SWORD dataset are within 10 km 
of the coastline. 

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA).

Execution example (terminal):
    python path/to/5_add_coastal_flag.py NA

"""

from __future__ import division
import os
main_dir = os.getcwd()
import time
import numpy as np
from scipy import spatial as sp
import glob
import geopandas as gp
import netCDF4 as nc
from shapely.geometry import Point
import pandas as pd
from osgeo import ogr
import argparse
import src.updates.geo_utils as geo 
import warnings
warnings.filterwarnings("ignore") #if code stops working may need to comment out to check warnings. 

start_all = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Region", type = str)
args = parser.parse_args()

region = args.region

mhv_dir = main_dir+'/data/inputs/MHV_SWORD/netcdf/' + region + '/' 
mhv_files = glob.glob(os.path.join(mhv_dir, '*.nc'))
fn_buffer = main_dir+'/data/gis_files/continent_buffer_0.1deg.gpkg'
coast_db = gp.GeoDataFrame.from_file(fn_buffer)

# Merging each level two basin file. 
for ind in list(range(len(mhv_files))):
    # print('Basin', mhv_files[ind][-13:-11])
    start = time.time()
    
    # Reading in data.
    mhv = nc.Dataset(mhv_files[ind], 'r+') 
    mhv_lon = mhv.groups['centerlines'].variables['x'][:]
    mhv_lat = mhv.groups['centerlines'].variables['y'][:]

    # Creating geodataframe for spatial joins. 
    mhv_df = gp.GeoDataFrame([mhv_lon, mhv_lat]).T
    mhv_df.rename(columns={0:"x",1:"y"},inplace=True)
    mhv_df = mhv_df.apply(pd.to_numeric, errors='ignore')
    geom = gp.GeoSeries(map(Point, zip(mhv_lon, mhv_lat)))
    mhv_df['geometry'] = geom
    mhv_df = gp.GeoDataFrame(mhv_df)
    mhv_df.set_geometry(col='geometry')
    mhv_df = mhv_df.set_crs(4326, allow_override=True)

    # Intersecting the coastal buffer with the mhv points. 
    mhv_coast = geo.vector_to_vector_intersect(mhv_df, coast_db, 'fid')
    mhv_coast[np.where(mhv_coast > 0)[0]] = 1

    # Add attributes to NetCDF
    mhv.groups['centerlines'].createVariable('coastflag', 'i4', ('num_points',))
    mhv.groups['centerlines'].variables['coastflag'][:] = mhv_coast
    mhv.close()

    end = time.time()
    print('Finished Basin ' + str(mhv_files[ind][-13:-11]) + ' in: ' + str(np.round((end-start)/60, 2)) + ' min')

end_all = time.time()
print('Finished '+region+' in: ' + str(np.round((end_all-start_all)/60, 2)) + ' min')
