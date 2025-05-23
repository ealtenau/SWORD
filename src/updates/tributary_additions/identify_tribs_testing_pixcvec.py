import os
main_dir = os.getcwd()
import numpy as np
import numpy.ma as ma
import netCDF4 as nc 
import matplotlib.pyplot as plt
import geopandas as gp
from scipy import spatial as sp

# pixc_dir = '/u/franz-r0/stuurman/data/20230320/'
# mhv_dir = '/u/franz-r0/altenau/trib_testing/inputs/mhv/'
# sword_dir = '/u/franz-r0/altenau/trib_testing/inputs/sword/'

# os.listdir(pixc_dir)
# need to automatically be able to find files based on pixcvec tile. 
pixc_fn = main_dir+'/data/swot_data/shifting_tests/016_077L/pixcvec_river/SWOT_L2_HR_PIXCVecRiver_519_016_077L_20230513T035930_20230513T035941_PGA1_018.nc'
mhv_fn = main_dir+'/data/inputs/MHV_SWORD/eu_mhv_sword.nc'
sword_fn = main_dir+'/data/outputs/Reaches_Nodes/v15b/netcdf/eu_sword_v15b.nc'

pixc = nc.Dataset(pixc_fn)
pixc_lat = ma.getdata(pixc.variables['latitude_vectorproc'][:])
pixc_lon = ma.getdata(pixc.variables['longitude_vectorproc'][:])
pixc_lat = pixc_lat[np.where((pixc_lat != 0.0) & (pixc_lat < 1000))]
pixc_lon = pixc_lon[np.where((pixc_lon != 0.0) & (pixc_lon < 1000))]
pixc.close()

xmin = np.min(pixc_lon)
xmax = np.max(pixc_lon)
ymin = np.min(pixc_lat)
ymax = np.max(pixc_lat)
ll = np.array([xmin, ymin])  # lower-left
ur = np.array([xmax, ymax])  # upper-right

mhv = nc.Dataset(mhv_fn)
mhv_lon_all = mhv.groups['centerlines'].variables['x'][:]
mhv_lat_all = mhv.groups['centerlines'].variables['y'][:]
mhv_flag_all =  mhv.groups['centerlines'].variables['swordflag'][:]
mhv_seg_all =  mhv.groups['centerlines'].variables['segID'][:]
mhv_points = [(mhv_lon_all[i], mhv_lat_all[i]) for i in range(len(mhv_lon_all))]
mhv_pts = np.array(mhv_points)
mhv_idx = np.all(np.logical_and(ll <= mhv_pts, mhv_pts <= ur), axis=1)
mhv_lon = mhv_lon_all[mhv_idx]
mhv_lat = mhv_lat_all[mhv_idx]
mhv_flag = mhv_flag_all[mhv_idx]
mhv_seg = mhv_seg_all[mhv_idx]

sword = nc.Dataset(sword_fn)
sword_lon_all = sword.groups['nodes'].variables['x'][:]
sword_lat_all =sword.groups['nodes'].variables['y'][:]
sword_tribs_all =sword.groups['nodes'].variables['trib_flag'][:]
sword_points = [(sword_lon_all[i], sword_lat_all[i]) for i in range(len(sword_lon_all))]
sword_pts = np.array(sword_points)
sword_idx = np.all(np.logical_and(ll <= sword_pts, sword_pts <= ur), axis=1)
sword_lon = sword_lon_all[sword_idx]
sword_lat = sword_lat_all[sword_idx]
sword_tribs = sword_tribs_all[sword_idx]

#spatial query between pixc and mhv
mhv_pts = np.vstack((mhv_lon, mhv_lat)).T
pixc_pts = np.vstack((pixc_lon,pixc_lat)).T
kdt = sp.cKDTree(pixc_pts)
pt_dist, pt_ind = kdt.query(mhv_pts, k = 5)

keep = np.where((pt_dist[:,0] < 0.005) & (mhv_flag == 0))[0]
add_channels = np.unique(mhv_seg[keep])

### will need to set length or # of points criteria for adding.
add_points = np.where(np.in1d(mhv_seg, add_channels))[0]

'''
plt.scatter(pixc_lon, pixc_lat, s = 5, c='grey')
plt.scatter(mhv_lon, mhv_lat, s = 1, c='black')
plt.scatter(sword_lon, sword_lat, s = 3, c='deepskyblue')
plt.scatter(mhv_lon[keep], mhv_lat[keep], s = 3, c='red')
plt.title(str(pixc_fn[62:70]))
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('test_plots/'+str(pixc_fn[62:70])+'.png')
plt.close()
# plt.show()

plt.scatter(pixc_lon, pixc_lat, s = 5, c='grey')
plt.scatter(mhv_lon, mhv_lat, s = 1, c='black')
plt.scatter(sword_lon, sword_lat, s = 3, c='deepskyblue')
plt.scatter(mhv_lon[add_points], mhv_lat[add_points], s = 3, c='red')
plt.title(str(pixc_fn[62:70]))
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('test_plots/'+str(pixc_fn[62:70])+'.png')
plt.close()
# plt.show()

'''