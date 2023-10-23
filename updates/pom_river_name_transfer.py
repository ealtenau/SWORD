import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy import spatial as sp

pom = pd.read_csv('/Users/ealteanau/Desktop/user_reports_pom.csv')

reaches = np.array(pom.reach_id)
names = np.array(pom.data2)
rgn = np.array(pom.region)
rgn[581:603] = 'NA'
regions = np.unique(rgn)

for r in list(range(len(regions))):
    print(regions[r])
    sword14 = nc.Dataset('/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v14/netcdf/'+regions[r].lower()+'_sword_v14.nc')
    sword15 = nc.Dataset('/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v15/netcdf/'+regions[r].lower()+'_sword_v15.nc', 'r+')
    x14 = sword14.groups['centerlines'].variables['x'][:]
    y14 = sword14.groups['centerlines'].variables['y'][:]
    r14 = sword14.groups['centerlines'].variables['reach_id'][0,:]
    x15 = sword15.groups['centerlines'].variables['x'][:]
    y15 = sword15.groups['centerlines'].variables['y'][:]
    r15 = sword15.groups['centerlines'].variables['reach_id'][0,:]
    rch15 = sword15.groups['reaches'].variables['reach_id'][:]
    nodes15 = sword15.groups['nodes'].variables['reach_id'][:]

    s14_pts = np.vstack((x14, y14)).T
    s15_pts = np.vstack((x15, y15)).T
    kdt = sp.cKDTree(s15_pts)
    eps_dist, eps_ind = kdt.query(s14_pts, k = 2)

    nms = names[np.where(rgn == regions[r])[0]]
    rchs = reaches[np.where(rgn == regions[r])[0]]
    for ind in list(range(len(rchs))):
        pts = np.where(r14 == rchs[ind])[0]
        new_rchs = np.array(np.unique(r15[eps_ind[pts,0]]))
        if len(new_rchs) == 0:
            continue
        else: 
            for idx in list(range(len(new_rchs))):
                ind_nodes = np.where(nodes15 == new_rchs[idx])[0]
                ind_rchs = np.where(rch15 == new_rchs[idx])[0]
                sword15.groups['nodes'].variables['river_name'][ind_nodes] = np.repeat(nms[ind], len(ind_nodes))
                sword15.groups['reaches'].variables['river_name'][ind_rchs] = nms[ind]

sword14.close()
sword15.close()
print('DONE')