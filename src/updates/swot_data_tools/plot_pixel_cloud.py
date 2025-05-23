import os 
main_dir = os.getcwd()
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

fn = main_dir+'/data/swot_data/Amazon/007_151R/503/'\
    'SWOT_L2_HR_PIXC_503_007_151R_20230426T230153_20230426T230204_PIA1_01.nc'

pixc = nc.Dataset(fn)
range = pixc.groups['pixel_cloud'].variables['range_index'][:]
azimuth = pixc.groups['pixel_cloud'].variables['azimuth_index'][:]
power = pixc.groups['pixel_cloud'].variables['coherent_power'][:]

plt.scatter(range, azimuth, power)
plt.show()