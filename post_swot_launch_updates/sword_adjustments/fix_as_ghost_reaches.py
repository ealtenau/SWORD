import numpy as np
import netCDF4 as nc
import geopandas as gp
import geopy.distance
from shapely.geometry import LineString, Point
from geopandas import GeoSeries
import pandas as pd
import time
import argparse
import os
from scipy import spatial as sp
import matplotlib.pyplot as plt
from scipy import stats
from scipy import interpolate
import matplotlib.pyplot as plt


region = 'AS'
version = 'v17'

nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'\
    +region.lower()+'_sword_'+version+'.nc'
outpath = '/Users/ealtenau/Documents/SWORD_Dev/update_requests/v17/AS/as_ghost_deletions.csv'

sword = nc.Dataset(nc_fn)
cl_rchs = np.array(sword.groups['centerlines'].variables['reach_id'][:])
cl_id = np.array(sword.groups['centerlines'].variables['cl_id'][:])
cl_x = np.array(sword.groups['centerlines'].variables['x'][:])
cl_y = np.array(sword.groups['centerlines'].variables['y'][:])
cl_type = np.array([str(rch)[-1] for rch in cl_rchs[0,:]])
reaches =  np.array(sword.groups['reaches'].variables['reach_id'][:])
rch_x = np.array(sword.groups['reaches'].variables['x'][:])
rch_y = np.array(sword.groups['reaches'].variables['y'][:])
sword.close()

ghost = np.where(cl_type == '6')[0]

#spatial query with all centerline points...
sword_pts = np.vstack((cl_x, cl_y)).T
kdt = sp.cKDTree(sword_pts)
pt_dist, pt_ind = kdt.query(sword_pts, k = 5)

unq_rchs = np.unique(cl_rchs[0,ghost])
remove = list()
rmv_x = list()
rmv_y = list()
#43610500036 (good), 45100300026 (bad), 43504600936 (bad)
# ind = np.where(unq_rchs == 43610500036)[0] 
for ind in list(range(len(unq_rchs))):
    print(ind, len(unq_rchs)-1)
    rch = np.where(cl_rchs[0,:] == unq_rchs[ind])[0]
    r = np.where(reaches == unq_rchs[ind])[0]

    mn = np.where(cl_id[rch] == min(cl_id[rch]))[0]
    mx = np.where(cl_id[rch] == max(cl_id[rch]))[0]

    if len(rch) <= 5:
        keep = 3
    else:
        keep = 5

    val1 = len(np.unique(cl_rchs[0,pt_ind[rch[mn],0:keep]]))
    val2 = len(np.unique(cl_rchs[0,pt_ind[rch[mx],0:keep]]))
    if val1 > 1 and val2 > 1:
        remove.append(unq_rchs[ind])
        rmv_x.append(rch_x[r][0])
        rmv_y.append(rch_y[r][0])

# len(remove)/len(unq_rchs)*100

df = pd.DataFrame(np.array([remove, rmv_x, rmv_y]).T)
df.rename(columns={0:"reach_id",1:"x",2:"y",},inplace=True)
df.to_csv(outpath)