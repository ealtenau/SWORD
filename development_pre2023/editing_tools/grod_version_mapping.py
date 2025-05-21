import numpy as np
import pandas as pd
from scipy import spatial as sp
import os
import netCDF4 as nc

grodv1_fn = '/Users/ealteanau/Documents/SWORD_Dev/inputs/GROD/GROD_HydroFALLS_ALL.csv'
grodv2_fn = '/Users/ealteanau/Documents/SWORD_Dev/inputs/GROD/v1.1/GROD_v1.1.csv'
sword_fn = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16_original/netcdf/'
sword_files = [f for f in os.listdir(sword_fn) if '.nc' in f]

grod_v1 = pd.read_csv(grodv1_fn)
grod_v2 = pd.read_csv(grodv2_fn)

v1_pts = np.vstack((grod_v1.lon, grod_v1.lat)).T
v2_pts = np.vstack((grod_v2.lon, grod_v2.lat)).T
kdt = sp.cKDTree(v2_pts)
eps_dist, eps_ind = kdt.query(v1_pts, k = 1)

keep = np.where(eps_dist < 0.0000003)[0]
v1 = np.array(grod_v1.fid[keep])
v2 = np.array(grod_v2.grod_id[eps_ind[keep]])

for s in list(range(len(sword_files))):
    sword = nc.Dataset(sword_fn+sword_files[s], 'r+')
    rch_grod = sword.groups['reaches'].variables['grod_id'][:]
    node_grod = sword.groups['nodes'].variables['grod_id'][:]
    for g in list(range(len(v1))):
        rind = np.where(rch_grod == v1[g])[0]
        nind = np.where(node_grod == v1[g])[0]
        sword.groups['reaches'].variables['grod_id'][rind] = v2[g]
        sword.groups['nodes'].variables['grod_id'][nind] = v2[g]
    sword.close()
    print(sword_files[s], 'Done')

