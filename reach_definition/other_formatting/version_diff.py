# -*- coding: utf-8 -*-
"""
Created on Wed Nov 03 09:06:09 2021

@author: ealtenau
"""

import netCDF4 as nc
import numpy as np
from scipy import spatial as sp
import pandas as pd

region = 'OC'

fn_sword_new = 'F:/SWORD/Development_Files/outputs/Reaches_Nodes_v12/netcdf/'+region.lower()+'_sword_v12.nc'
fn_sword_old = 'F:/SWORD/Development_Files/outputs/Reaches_Nodes_v09/netcdf/'+region.lower()+'_apriori_rivers_v09.nc'
outpath = 'F:/SWORD/Development_Files/outputs/Version_Differences/'+region+'_ReachID_v12_vs_v09.csv'

# read in global data
new = nc.Dataset(fn_sword_new)
old = nc.Dataset(fn_sword_old)

# make array of node locations.
nlon = new.groups['nodes'].variables['x'][:]
nlat = new.groups['nodes'].variables['y'][:]
nid = new.groups['nodes'].variables['node_id'][:]
nrch_id = new.groups['nodes'].variables['reach_id'][:]
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
old_node_ids = oid[indexes]
old_reach_ids = orch_id[indexes]

# output reach differences in csv format. 
data2 = pd.DataFrame(np.array([nlon, nlat, nid, nrch_id, old_node_ids, old_reach_ids])).T
data2.columns = ['lon', 'lat', 'v12_node_id', 'v12_reach_id', 'v09_node_id', 'v09_reach_id']
data2.to_csv(outpath)

print('DONE')




