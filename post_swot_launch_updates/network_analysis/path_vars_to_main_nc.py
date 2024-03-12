import numpy as np
import netCDF4 as nc
import geopandas as gp
import geopy.distance
from shapely.geometry import LineString, Point
from geopandas import GeoSeries
import pandas as pd
import time
import argparse
import os
from scipy import spatial as sp
import matplotlib.pyplot as plt

start_all = time.time()
region = 'NA'
version = 'v17a'
basin = 'hb82'

print('Starting Basin: ', basin)
sword_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
con_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/reach_geometry/'+region.lower()+'_sword_'+version+'_connectivity.nc'
path_nc = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/pathways/'+region+'/'+basin+'_path_vars.nc'

#update end_reaches variable (i.e.) correct headwaters and outlets 
#change type of ghost reaches that are not an 'end reach'
#re-number reaches and nodes. 
#add new variables to netcdf. 
#reformat cl_ids per reach. 

