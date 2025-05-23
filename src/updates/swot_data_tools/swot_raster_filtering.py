import os
main_dir = os.getcwd()
import numpy as np
import netCDF4 as nc
import pandas as pd
import geopandas as gp
from shapely.geometry import Point
import time

start=time.time()
fn_dir = main_dir+'/data/swot_data/Greenland/raster/'
f1 = 'SWOT_L2_HR_Raster_100m_UTM26W_N_x_x_x_021_117_142F_20240914T165018_20240914T165039_PIC0_01_filt.nc'
f2 = 'SWOT_L2_HR_Raster_100m_UTM26W_N_x_x_x_022_117_142F_20241005T133527_20241005T133548_PIC0_01_filt.nc'
f3 = 'SWOT_L2_HR_Raster_100m_UTM26W_N_x_x_x_023_117_142F_20241026T102030_20241026T102051_PIC2_01_filt.nc'
fn1 = fn_dir+f1
fn2 = fn_dir+f2
fn3 = fn_dir+f3

nc1 = nc.Dataset(fn1, 'r+')
nc2 = nc.Dataset(fn2, 'r+')
nc3 = nc.Dataset(fn3, 'r+')

sig0_1 = np.array(nc1['sig0'][:])
sig0_2 = np.array(nc2['sig0'][:])
sig0_3 = np.array(nc3['sig0'][:])

wse1 = np.array(nc1['wse'][:])
wse2 = np.array(nc2['wse'][:])
wse3 = np.array(nc3['wse'][:])

na_val = np.max(wse1)
mask = np.where(sig0_3 < 1)

# max(sig0_1[mask])
# min(wse1[mask])

wse1[mask] = na_val
wse2[mask] = na_val
wse3[mask] = na_val

nc1['wse'][:] = wse1
nc2['wse'][:] = wse2
nc3['wse'][:] = wse3

nc1.close()
nc2.close()
nc3.close()