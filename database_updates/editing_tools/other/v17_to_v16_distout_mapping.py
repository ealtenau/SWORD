# -*- coding: utf-8 -*-
"""
Created on Wed Nov 03 09:06:09 2021

@author: ealtenau
"""

import netCDF4 as nc
import numpy as np
from scipy import spatial as sp
import pandas as pd

region = 'NA'

fn_sword_new = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/'+region.lower()+'_sword_v17.nc'
fn_sword_old = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16/netcdf/'+region.lower()+'_sword_v16.nc'
# outpath = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Version_Differences/'+region+'_ReachID_v16_vs_v15b.csv'

# read in global data
new = nc.Dataset(fn_sword_new)
old = nc.Dataset(fn_sword_old)

# make array of node locations.
nlon = new.groups['nodes'].variables['x'][:]
nlat = new.groups['nodes'].variables['y'][:]
nid = new.groups['nodes'].variables['node_id'][:]
nrch_id = new.groups['nodes'].variables['reach_id'][:]
ndist_out = new.groups['nodes'].variables['dist_out'][:]
new.close()

olon = old.groups['nodes'].variables['x'][:]
olat = old.groups['nodes'].variables['y'][:]
oid = old.groups['nodes'].variables['node_id'][:]
orch_id = old.groups['nodes'].variables['reach_id'][:]
old.close()

# find closest points.     
old_pts = np.vstack((olon, olat)).T
new_pts = np.vstack((nlon, nlat)).T
kdt = sp.cKDTree(old_pts)
eps_dist, eps_ind = kdt.query(new_pts, k = 2) 

indexes = eps_ind[:,0]
dist = eps_dist[:,0]
old_node_ids = oid[indexes]
old_reach_ids = orch_id[indexes]

# Flag any nodes with a previous version node greater than 500 m away (i.e. new centerline channel).
new_cls = np.where(dist > 0.003)[0]
old_node_ids[new_cls] = -9999
old_reach_ids[new_cls] = -9999

# output reach differences in csv format. 
# data2 = pd.DataFrame(np.array([nlon, nlat, nid, nrch_id, old_node_ids, old_reach_ids])).T
# data2.columns = ['lon', 'lat', 'v16_node_id', 'v16_reach_id', 'v15b_node_id', 'v15b_reach_id']
# data2.to_csv(outpath)

print('DONE')

'''
import matplotlib.pyplot as plt


plt.scatter(nlon, nlat, c='blue', s=3)
plt.scatter(nlon[new_cls], nlat[new_cls], c='red', s=3)
plt.show()
'''
