# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:29:00 2020

@author: ealtenau
"""
import netCDF4 as nc
import numpy as np
from scipy import spatial as sp
import pandas as pd

region = 'OC'

fn_sword = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/outputs/Reaches_Nodes/netcdf/'+region.lower()+'_sword_v11.nc'
fn_csv = 'E:/Users/Elizabeth Humphries/Documents/SWORD/lakes_near_rivers/extdist_csv/'+region.lower()+'_dist_thresh_testing.csv'

# read in global data
new = nc.Dataset(fn_sword, 'r+')

# make array of node locations.
edcf = new.groups['nodes'].variables['ext_dist_coef'][:]
nlon = new.groups['nodes'].variables['x'][:]
nlat = new.groups['nodes'].variables['y'][:]
nchan = new.groups['nodes'].variables['n_chan_max'][:]
lake_id = new.groups['nodes'].variables['lake_id'][:]

csv = pd.read_csv(fn_csv)
olon = np.array(csv.lon[:])
olat = np.array(csv.lat[:])
odist = np.array(csv.dist_thresh[:])

# find closest points.     
csv_pts = np.vstack((olon, olat)).T
node_pts = np.vstack((nlon, nlat)).T
kdt = sp.cKDTree(csv_pts)
eps_dist, eps_ind = kdt.query(node_pts, k = 2) 

new_indexes = eps_ind[:,0]
new_ext_dist = odist[new_indexes]
#new_ext_dist[np.where(lake_id > 0)] = 20

# reduce coeficient to 1 everywhere for max width testing. 
update = np.where(new_ext_dist == 2)[0]
new_ext_dist[update] = 1
new_ext_dist[np.where(new_ext_dist > 20)] = 20
             
# assign new coeficients.
new.groups['nodes'].variables['ext_dist_coef'][:] = new_ext_dist
new.close()

print('DONE with ' + region)

'''
# output reach differences in csv format. 
data2 = pd.DataFrame(np.array([nlon, nlat, new_ext_dist])).T
data2.columns = ['lon', 'lat', 'ext_dist']
data2.to_csv(outpath)

np.unique(new_ext_dist)
'''



