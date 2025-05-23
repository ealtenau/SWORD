# -*- coding: utf-8 -*-
"""
Created on Wed Nov 03 09:06:09 2021
"""

import os
main_dir = os.getcwd()
import netCDF4 as nc
import numpy as np
from scipy import spatial as sp
import pandas as pd

region = 'OC'
new_v = 'v17b'
old_v = 'v16'

fn_sword_new = main_dir+'/data/outputs/Reaches_Nodes/'+new_v+'/netcdf/'+region.lower()+'_sword_'+new_v+'.nc'
fn_sword_old = main_dir+'/data/outputs/Reaches_Nodes/'+old_v+'/netcdf/'+region.lower()+'_sword_'+old_v+'.nc'
outpath = main_dir+'/data/outputs/Version_Differences/'+new_v+'/'+region+'_ReachIDs_'+new_v+'_vs_'+old_v+'.csv'
outpath2 = main_dir+'/data/outputs/Version_Differences/'+new_v+'/'+region+'_NodeIDs_'+new_v+'_vs_'+old_v+'.csv'

# read in global data
new = nc.Dataset(fn_sword_new)
old = nc.Dataset(fn_sword_old)

# make array of node locations.
nlon = new.groups['nodes'].variables['x'][:]
nlat = new.groups['nodes'].variables['y'][:]
nid = new.groups['nodes'].variables['node_id'][:]
nrch_id = new.groups['nodes'].variables['reach_id'][:]
rlon = new.groups['reaches'].variables['x'][:]
rlat = new.groups['reaches'].variables['y'][:]
rid = new.groups['reaches'].variables['reach_id'][:]
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

# test = np.where(dist < 0.0005)
# np.median(dist[test])*111000

# Flag any nodes with a previous version node greater than 500 m away (i.e. new centerline channel).
new_cls = np.where(dist > 0.005)[0]
old_node_ids[new_cls] = 0
old_reach_ids[new_cls] = 0
shift_flag = np.zeros(len(old_reach_ids))
shift_flag[np.where(old_node_ids == 0)[0]] = 1
#len(np.where(shift_flag == 1)[0])/len(shift_flag)*100 #1.5% of nodes shifted or were added in NA from v17 to v16 

print('calculating node dimension')
boundary_flag = np.zeros(len(old_reach_ids),dtype=int)
boundary_perc = np.zeros(len(old_reach_ids),dtype=int)
dominant_rch = np.zeros(len(old_reach_ids),dtype=int)
number_of_rchs = np.zeros(len(old_reach_ids),dtype=int)
for r in list(range(len(rid))):
    pts = np.where(nrch_id == rid[r])[0]
    unique_elements, counts = np.unique(old_reach_ids[pts], return_counts=True)
    keep = np.where(unique_elements>0)[0]
    unq_elements = unique_elements[keep]
    counts = counts[keep]
    if len(unique_elements) > 1:
        boundary_flag[pts] = 1
        number_of_rchs[pts] = len(unq_elements)
        max_cnt = np.where(counts == max(counts))[0]
        if len(max_cnt) > 1:
            max_cnt = max_cnt[0]
        boundary_perc[pts] = counts[max_cnt]/len(pts)*100
        dominant_rch[pts] = unq_elements[max_cnt]

print('calculating reach dimension')
rch_bnd_flag = np.zeros(len(rid),dtype=int)
rch_bnd_perc = np.zeros(len(rid),dtype=int)
rch_dom = np.zeros(len(rid),dtype=int)
num_rch = np.zeros(len(rid),dtype=int)
old_rch_id = np.zeros(len(rid),dtype=int)
for ind in list(range(len(rid))):
    # print(ind)
    pts2 = np.where(nrch_id == rid[ind])[0]
    rch_bnd_flag[ind] = np.unique(boundary_flag[pts2])[0]
    rch_bnd_perc[ind] = np.unique(boundary_perc[pts2])[0]
    rch_dom[ind] = np.unique(dominant_rch[pts2])[0]
    num_rch[ind] = np.unique(number_of_rchs[pts2])[0]
    if np.unique(boundary_flag[pts2]) == 1:
        old_rch_id[ind] = np.unique(dominant_rch[pts2])[0]
    else:
        old_rch_id[ind] = np.unique(old_reach_ids[pts2])[0]
    
# output reach differences in csv format. 
data = pd.DataFrame(np.array([rlon, rlat, rid, old_rch_id, rch_bnd_flag, rch_bnd_perc, rch_dom, num_rch])).T
data.columns = ['lon', 'lat', 'v17_reach_id', 'v16_reach_id', 'boundary_flag', 'boundary_percent', 'dominant reach', 'v16 number of reaches']
for i in data.columns[2::]:
    try:
        data[[i]] = data[[i]].astype(float).astype(int)
    except:
        pass
data.to_csv(outpath,index=False)

data2 = pd.DataFrame(np.array([nlon, nlat, nid, nrch_id, old_node_ids, old_reach_ids, shift_flag, boundary_flag, boundary_perc, dominant_rch, number_of_rchs])).T
data2.columns = ['lon', 'lat', 'v17_node_id', 'v17_reach_id', 'v16_node_id', 'v16_reach_id', 'shift_flag', 'boundary_flag', 'boundary_percent', 'dominant reach', 'v16 number of reaches']
for j in data2.columns[2::]:
    try:
        data2[[j]] = data2[[j]].astype(float).astype(int)
    except:
        pass
data2.to_csv(outpath2, index=False)

print('DONE')
print('percent of nodes shifted:',np.round(len(np.where(shift_flag == 1)[0])/len(shift_flag)*100,2))
print('percent of reaches with different boundaries:',np.round(len(np.where(rch_bnd_flag == 1)[0])/len(rch_bnd_flag)*100,2))


'''
import matplotlib.pyplot as plt

plt.scatter(nlon, nlat, c='blue', s=3)
plt.scatter(nlon[new_cls], nlat[new_cls], c='red', s=3)
plt.show()
'''
