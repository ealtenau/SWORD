from __future__ import division
import numpy as np
import time
import netCDF4 as nc
import pandas as pd
from scipy import spatial as sp
from shapely.geometry import Point
import geopandas as gp
import glob
import os

region = 'OC'
version = 'v18'

#reading in sword data.
sword_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
outdir = '/Users/ealtenau/Documents/SWORD_Dev/update_requests/'+version+'/'+region+'/'
sword = nc.Dataset(sword_fn)

reaches = np.array(sword['/reaches/reach_id'][:])
n_rch_up = np.array(sword['/reaches/n_rch_up'][:])
n_rch_down = np.array(sword['/reaches/n_rch_down'][:])
rch_id_up = np.array(sword['/reaches/rch_id_up'][:])
rch_id_down = np.array(sword['/reaches/rch_id_dn'][:])

category = np.array([int(str(ind)[-1]) for ind in reaches])
break_check = np.where((n_rch_down == 0) & (category != 6))[0]
rch_check = reaches[break_check]

rch_csv = pd.DataFrame({"reach_id": rch_check})
rch_csv.to_csv(outdir+'downstream_break_check.csv', index = False)


