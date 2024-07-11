from __future__ import division
import numpy as np
from scipy import spatial as sp
import netCDF4 as nc
from pyproj import Proj
import time
import geopy.distance
import pandas as pd
import argparse
import re
import os 
import geopandas as gp
from shapely.geometry import Point
import matplotlib.pyplot as plt


sword_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/sa_sword_v17.nc'
sword = nc.Dataset(sword_dir,'r+')

reaches = np.array(sword.groups['reaches'].variables['reach_id'][:])
node_rchs = np.array(sword.groups['nodes'].variables['reach_id'][:])

# rchs1 = np.array([22320700693,22320700681,22320700663,22320700651,22320700644,22320700631,22320700623,22320700611,22320700603,22320700591,22320700583,22320700571,22320700563,22320700551,22320700541,22320700534,22320700523,22320700511,22320700503])
# rchs2 = np.array([24434001793,24434001801,24434001811,24434001821,24434001831,24434001841,24434001853,24434001861,24434001871,24434001883])
# val1 = 608042.31378
# val2 = 671642.5202143731

# rch_ind1 = np.where(np.in1d(reaches, rchs1)==True)[0]
# rch_ind2 = np.where(np.in1d(reaches, rchs2)==True)[0]
# node_ind1 = np.where(np.in1d(node_rchs, rchs1)==True)[0]
# node_ind2 = np.where(np.in1d(node_rchs, rchs2)==True)[0]

# sword.groups['reaches'].variables['dist_out'][rch_ind1] = sword.groups['reaches'].variables['dist_out'][rch_ind1]+val1
# sword.groups['nodes'].variables['dist_out'][node_ind1] = sword.groups['nodes'].variables['dist_out'][node_ind1]+val1
# sword.groups['reaches'].variables['dist_out'][rch_ind2] = sword.groups['reaches'].variables['dist_out'][rch_ind2]+val2
# sword.groups['nodes'].variables['dist_out'][node_ind2] = sword.groups['nodes'].variables['dist_out'][node_ind2]+val2

csv = pd.read_csv('/Users/ealtenau/Documents/SWORD_Dev/update_requests/v17/EU/volga_delta_changes.csv')
rchs = np.array(csv['reach_id'])
main_side = np.array(csv['main_side'])
paths = np.array(csv['path_order'])

keep = np.where((paths > 1)&(main_side==0))[0]
rchs_change = rchs[keep]

update = np.where(np.in1d(reaches, rchs_change)==True)[0]
update2 = np.where(np.in1d(node_rchs, rchs_change)==True)[0]

sword.groups['reaches'].variables['main_side'][update] = 2
sword.groups['nodes'].variables['main_side'][update2] = 2
sword.groups['reaches'].variables['stream_order'][update] = -9999
sword.groups['nodes'].variables['stream_order'][update2] = -9999

sword.close()


#################################################################################################
#################################################################################################
#################################################################################################

sword_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/eu_sword_v17.nc'
sword = nc.Dataset(sword_dir,'r+')

rch_nan = np.where(np.array(sword.groups['reaches'].variables['main_side'][:]) == 1)[0]
node_nan = np.where(np.array(sword.groups['nodes'].variables['main_side'][:]) == 1)[0]
sword.groups['reaches'].variables['path_freq'][rch_nan] = -9999
sword.groups['nodes'].variables['path_freq'][node_nan] = -9999
sword.groups['reaches'].variables['path_order'][rch_nan] = -9999
sword.groups['nodes'].variables['path_order'][node_nan] = -9999
sword.close()

np.min(np.array(sword.groups['reaches'].variables['path_freq'][:]))
np.min(np.array(sword.groups['reaches'].variables['path_order'][:]))

#################################################################################################
#################################################################################################
#################################################################################################

sword_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/sa_sword_v17.nc'
sword = nc.Dataset(sword_dir,'r+')

rch = 61538000271
cl_ids = np.array(sword.groups['centerlines'].variables['cl_id'][:])
cl_rchs = np.array(sword.groups['centerlines'].variables['reach_id'][:])
x = np.array(sword.groups['centerlines'].variables['x'][:])
y = np.array(sword.groups['centerlines'].variables['y'][:])

r = np.where(cl_rchs[0,:] == rch)[0]
sort_ids = np.argsort(cl_ids[r])
plt.plot(x[r[sort_ids]], y[r[sort_ids]])
plt.show()


#################################################################################################
#################################################################################################
#################################################################################################

sword_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/sa_sword_v17.nc'
sword = nc.Dataset(sword_dir)

cl_nodes = np.array(sword.groups['centerlines'].variables['node_id'][0,:])
cl_id = np.array(sword.groups['centerlines'].variables['cl_id'][:])
cl_x = np.array(sword.groups['centerlines'].variables['x'][:])
cl_y = np.array(sword.groups['centerlines'].variables['y'][:])
nodes = np.array(sword.groups['nodes'].variables['node_id'][:])
nodes_cl_id = np.array(sword.groups['nodes'].variables['cl_ids'][:])
nx = np.array(sword.groups['nodes'].variables['x'][:])
ny = np.array(sword.groups['nodes'].variables['y'][:])

cln = np.unique(cl_nodes)
nds = np.unique(nodes)
missing = np.where(np.in1d(nds, cln) == False)[0]
missed_nodes = nds[missing]

for ind in list(range(len(missed_nodes))):
    id1 = nodes_cl_id[0,np.where(nds == missed_nodes[ind])[0]]
    id2 = nodes_cl_id[1,np.where(nds == missed_nodes[ind])[0]]
    indexes = list(range(id1[0],id2[0]+1))
    cl_inds = np.where(np.in1d(cl_id, indexes) == True)[0]
    cl_nodes[cl_inds] = missed_nodes[ind]


pt = np.where(nds == nodes[0])[0]
test = np.where(cl_nodes == nodes[0])[0]
plt.scatter(cl_x[test], cl_y[test])
plt.scatter(nx[0], ny[0],c = 'red')
plt.show()

#################################################################################################
#################################################################################################
#################################################################################################

dir1 = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/shp/NA/'
dir2 = '/Users/ealtenau/Desktop/NA_v17_04092024/'
files1 = os.listdir(dir1)
files2 = os.listdir(dir2)
files1 = np.array([f for f in files1 if '.shp' in f])
files2 = np.array([f for f in files2 if '.shp' in f])
files1 = np.sort(np.array([f for f in files1 if 'reaches' in f]))
files2 = np.sort(np.array([f for f in files2 if 'reaches' in f]))

for ind in list(range(len(files1))):
    f1 = gp.read_file(dir1+files1[ind])
    f2 = gp.read_file(dir2+files2[ind])
    print(files1[ind], np.unique(f1['reach_id']-f2['reach_id']))

#################################################################################################
#################################################################################################
#################################################################################################

import netCDF4 as nc
sword_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/na_sword_v17.nc'
sword = nc.Dataset(sword_dir, 'r+')


rch = np.where(sword.groups['reaches'].variables['reach_id'][:] == 81190800236)[0]
sword.groups['reaches'].variables['rch_id_up'][:,rch] = 0
sword.groups['reaches'].variables['n_rch_up'][rch] = 0

rch = np.where(sword.groups['reaches'].variables['reach_id'][:] == 81153200073)[0]
sword.groups['reaches'].variables['rch_id_up'][:,rch] = 0
sword.groups['reaches'].variables['rch_id_up'][0,rch] =  81153200081
sword.groups['reaches'].variables['n_rch_up'][rch] = 1

rch = np.where(sword.groups['reaches'].variables['reach_id'][:] == 81153200081)[0]
sword.groups['reaches'].variables['n_rch_down'][rch] = 1

rch = np.where(sword.groups['reaches'].variables['reach_id'][:] == 81340500051)[0]
sword.groups['reaches'].variables['n_rch_up'][rch] = 1

sword.close()

#################################################################################################
#################################################################################################
#################################################################################################
import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt

start_all = time.time()
sword_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/na_sword_v17_glows.nc'

print('Reading in SWORD Data')
start = time.time()
sword = nc.Dataset(sword_dir,'r+')
nlon = np.array(sword.groups['nodes'].variables['x'][:])
nlat = np.array(sword.groups['nodes'].variables['y'][:])
rlon = np.array(sword.groups['reaches'].variables['x'][:])
rlat = np.array(sword.groups['reaches'].variables['y'][:])
nid = np.array(sword.groups['nodes'].variables['node_id'][:])
rid = np.array(sword.groups['reaches'].variables['reach_id'][:])
nrid = np.array(sword.groups['nodes'].variables['reach_id'][:])
rwth = np.array(sword.groups['reaches'].variables['width'][:])
nwth = np.array(sword.groups['nodes'].variables['width'][:])

rwth_max = np.array(sword.groups['reaches'].variables['max_width'][:])
nwth_max = np.array(sword.groups['nodes'].variables['max_width'][:])

wth_median = sword.groups['nodes'].variables['glows_wth_med'][:] 
wth_min = sword.groups['nodes'].variables['glows_wth_min'][:] 
wth_max = sword.groups['nodes'].variables['glows_wth_max'][:] 
wth_1sigma = sword.groups['nodes'].variables['glows_wth_1sig'][:] 
end = time.time()
print(str(np.round((end-start),2))+' sec')

print('Starting Reaches')
start = time.time()
rch_wth_min = np.repeat(-9999, len(rid))
rch_wth_max = np.repeat(-9999, len(rid))
rch_wth_median = np.repeat(-9999, len(rid))
rch_wth_1sigma = np.repeat(-9999, len(rid))

data = np.where(wth_median > -9999)[0]
unq_rchs = np.unique(nrid[data])
for r in list(range(len(unq_rchs))):
    print(r, len(unq_rchs)-1)
    nind = np.where(nrid == unq_rchs[r])[0]
    rind = np.where(rid == unq_rchs[r])[0]
    good_data = np.where(wth_median[nind]>-9999)[0]
    prc = len(good_data)/len(wth_median[nind])*100
    if prc >= 50:
        rch_wth_min[rind] = np.min(wth_min[nind[good_data]])
        rch_wth_max[rind] = np.max(wth_max[nind[good_data]])
        rch_wth_median[rind] = np.median(wth_median[nind[good_data]])
        rch_wth_1sigma[rind] = np.mean(wth_1sigma[nind[good_data]])
end = time.time()
print(str(np.round((end-start),2))+' sec')



node_basins = np.array([int(str(ind)[0:4]) for ind in nid])
rch_basins = np.array([int(str(ind)[0:4]) for ind in rid])
node_zero = np.where(node_basins != 7426)[0]
rch_zero = np.where(rch_basins != 7426)[0]

wth_median[node_zero] = -9999
wth_min[node_zero] = -9999
wth_max[node_zero] = -9999
wth_1sigma[node_zero] = -9999
rch_wth_median[rch_zero] = -9999
rch_wth_min[rch_zero] = -9999
rch_wth_max[rch_zero] = -9999
rch_wth_1sigma[rch_zero] = -9999

# #nodes
# sword.groups['nodes'].variables['glows_wth_med'][:] = wth_median
# sword.groups['nodes'].variables['glows_wth_min'][:] = wth_min
# sword.groups['nodes'].variables['glows_wth_max'][:] = wth_max
# sword.groups['nodes'].variables['glows_wth_1sig'][:] = wth_1sigma
# #reaches
# sword.groups['reaches'].variables['glows_wth_med'][:] = rch_wth_median
# sword.groups['reaches'].variables['glows_wth_min'][:] = rch_wth_min
# sword.groups['reaches'].variables['glows_wth_max'][:] = rch_wth_max
# sword.groups['reaches'].variables['glows_wth_1sig'][:] = rch_wth_1sigma
# sword.close()

basin = np.where(node_basins == 7426)[0]
data = np.where(wth_median[basin]>-9999)[0]
plt.scatter(nlon[basin], nlat[basin], c='blue', s=5)
plt.scatter(nlon[basin[data]], nlat[basin[data]], c='red', s=5)
plt.show()

plt.scatter(nlon[basin], nlat[basin], c='black', s=5)
plt.scatter(nlon[basin[data]], nlat[basin[data]], c=wth_median[basin[data]], s=5, cmap='rainbow')
plt.show()

plt.scatter(nlon[basin], nlat[basin], c='black', s=5)
plt.scatter(nlon[basin[data]], nlat[basin[data]], c=np.log(wth_median[basin[data]]), s=5, cmap='rainbow')
plt.show()

rbasin = np.where(rch_basins == 7426)[0]
rdata = np.where(rch_wth_median[rbasin]>-9999)[0]

plt.scatter(rlon[rbasin], rlat[rbasin], c='black', s=5)
plt.scatter(rlon[rbasin[rdata]], rlat[rbasin[rdata]], c=np.log(rch_wth_median[rbasin[rdata]]), s=5, cmap='rainbow')
plt.show()

print(len(data)/len(nlon[basin])*100) #89%
print(len(rdata)/len(nlon[rbasin])*100) #92% 

np.median(abs(nwth[basin[data]]-wth_median[basin[data]])) #22 m 
np.mean(abs(nwth[basin[data]]-wth_median[basin[data]])) #45 m 

np.median(abs(nwth_max[basin[data]]-wth_max[basin[data]])) #39.5 m 
np.mean(abs(nwth_max[basin[data]]-wth_max[basin[data]])) #118.4 m 

np.median(abs(rwth[rbasin[rdata]]-rch_wth_median[rbasin[rdata]])) #22 m 
np.mean(abs(rwth[rbasin[rdata]]-rch_wth_median[rbasin[rdata]])) #41 m 

np.median(abs(rwth_max[rbasin[rdata]]-rch_wth_max[rbasin[rdata]])) #126 m 
np.mean(abs(rwth_max[rbasin[rdata]]-rch_wth_max[rbasin[rdata]])) #276 m 

