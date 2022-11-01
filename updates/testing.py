from __future__ import division
import numpy as np
from scipy import spatial as sp
import netCDF4 as nc
from pyproj import Proj
import time
import geopy.distance
import pandas as pd



fn = "/afs/cas.unc.edu/depts/geological_sciences/pavelsky/students/ealtenau/SWORD_dev/outputs/Reaches_Nodes/v14/netcdf/test.nc"
data = nc.Dataset(fn, 'r+')

data.groups['reaches'].variables['facc'][0] = 1
data.groups['reaches'].variables['facc'][1] = 2
data.close()

print('Done')


