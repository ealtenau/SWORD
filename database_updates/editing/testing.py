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

sword_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16_glows/netcdf/na_sword_v16_glows.nc'

print('Reading in SWORD Data')
start = time.time()
sword = nc.Dataset(sword_dir)
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

wth_median = np.array(sword.groups['nodes'].variables['glows_wth_med'][:]) 
wth_min = np.array(sword.groups['nodes'].variables['glows_wth_min'][:]) 
wth_max = np.array(sword.groups['nodes'].variables['glows_wth_max'][:]) 
wth_1sigma = np.array(sword.groups['nodes'].variables['glows_wth_1sig'][:]) 
rch_wth_median = np.array(sword.groups['reaches'].variables['glows_wth_med'][:]) 
rch_wth_min = np.array(sword.groups['reaches'].variables['glows_wth_min'][:]) 
rch_wth_max = np.array(sword.groups['reaches'].variables['glows_wth_max'][:]) 
rch_wth_1sigma = np.array(sword.groups['reaches'].variables['glows_wth_1sig'][:]) 
wth_id = np.array(sword.groups['nodes'].variables['glows_river_id'][:]) 
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

data = np.where(wth_median>-9999)[0]
plt.scatter(nlon, nlat, c='blue', s=5)
plt.scatter(nlon[data], nlat[data], c='red', s=5)
plt.show()

plt.scatter(nlon, nlat, c='black', s=5)
plt.scatter(nlon[data], nlat[data], c=wth_median[data], s=5, cmap='rainbow')
plt.show()

plt.scatter(nlon, nlat, c='black', s=5)
plt.scatter(nlon[data], nlat[data], c=np.log(wth_median[data]), s=5, cmap='rainbow')
plt.show()

max_wth_diff = nwth_max - wth_max #positive is sword is larger...
plt.scatter(nlon, nlat, c='black', s=5)
plt.scatter(nlon[data], nlat[data], c=np.log(max_wth_diff[data]), s=5, cmap='rainbow')
plt.show()

rdata = np.where(rch_wth_median>-9999)[0]
plt.scatter(rlon, rlat, c='black', s=5)
plt.scatter(rlon[rdata], rlat[rdata], c=np.log(rch_wth_median[rdata]), s=5, cmap='rainbow')
plt.show()

id_check = np.where(wth_id != 'NaN')[0]
plt.scatter(nlon, nlat, c='black', s=5)
plt.scatter(nlon[id_check], nlat[id_check], c='cyan', s=5)
plt.show()


# print(len(data)/len(nlon)*100) #89%
# print(len(rdata)/len(nlon)*100) #92% 

np.median(abs(nwth[data]-wth_median[data])) #22 m 
np.mean(abs(nwth[data]-wth_median[data])) #45 m 

np.median(abs(nwth_max[data]-wth_max[data])) #39.5 m 
np.mean(abs(nwth_max[data]-wth_max[data])) #118.4 m 

np.median(abs(rwth[rdata]-rch_wth_median[rdata])) #22 m 
np.mean(abs(rwth[rdata]-rch_wth_median[rdata])) #41 m 

np.median(abs(rwth_max[rdata]-rch_wth_max[rdata])) #126 m 
np.mean(abs(rwth_max[rdata]-rch_wth_max[rdata])) #276 m 

### fixing ids...

id_check = np.where(wth_id != 'NaN')[0]
plt.scatter(nlon, nlat, c='black', s=5)
plt.scatter(nlon[id_check], nlat[id_check], c='cyan', s=5)
plt.show()

# sword_glows_id[id_check] = glows_ids[id_check]

# id_check2 = np.where(sword_glows_id != 'NaN')[0]
# plt.scatter(nlon, nlat, c='black', s=5)
# plt.scatter(nlon[id_check2], nlat[id_check2], c='cyan', s=5)
# plt.show()

# sword.groups['nodes'].variables['glows_river_id'][:] = sword_glows_id[:]


# test = np.where(wth_id == 'R81008984XS0060552')[0]

#################################################################################################
#################################################################################################

import itertools
from itertools import permutations, combinations
import numpy as np

x_steps = list(np.round(np.arange(-0.005,0.0051,0.0001),10))
y_steps = list(np.round(np.arange(-0.005,0.0051,0.0001),10))

pairs = list(itertools.product(x_steps, y_steps))
x_pair = np.array([p[0] for p in pairs])
y_pair = np.array([p[1] for p in pairs])



################################################################################

import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt

gaps = pd.read_csv('/Users/ealtenau/Desktop/gap_rchs_copy.csv')
sword = nc.Dataset('/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/as_sword_v17.nc')

end_rch = np.array(sword['/reaches/end_reach'][:])
rchs = np.array(sword['/reaches/reach_id'][:])

ends = rchs[np.where((end_rch > 0) & (end_rch < 3))[0]]

rmv = np.where(np.in1d(gaps['reach_id'], ends)== True)[0]

gaps = gaps.drop(rmv)

gaps.to_csv('/Users/ealtenau/Desktop/gap_rchs_copy.csv',index=False)

####################################################################################
####################################################################################

import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt

edits = pd.read_csv('/Users/ealtenau/Desktop/europe_hb23_node_edits.csv')
sword = nc.Dataset('/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/eu_sword_v17.nc', 'r+')

old_id = np.array(edits['node_id'])
cl_id = np.array(edits['cl_id'])
new_id = np.array(edits['new_node_id2'])
x = np.array(edits['x'])
y = np.array(edits['y'])

nc_cl_nodes = np.array(sword['/centerlines/node_id'][:])
nc_cl_id = np.array(sword['/centerlines/cl_id'][:])
nc_cl_x = np.array(sword['/centerlines/x'][:])
nc_cl_y = np.array(sword['/centerlines/y'][:])
nc_node_x = np.array(sword['/nodes/x'][:])
nc_node_y = np.array(sword['/nodes/y'][:])
nc_nodes = np.array(sword['/nodes/node_id'][:])

for ind in list(range(len(cl_id))):
    pt = np.where(nc_cl_id == cl_id[ind])[0]
    nc_cl_nodes[0,pt] = new_id[ind]

unq_nodes = np.unique(new_id)

for idx in list(range(len(unq_nodes))):
    pts = np.where(new_id == unq_nodes[idx])[0]
    new_x = np.median(x[pts])
    new_y = np.median(y[pts])
    nind = np.where(nc_nodes == unq_nodes[idx])[0]
    nc_node_x[nind] = new_x
    nc_node_y[nind] = new_y


sword['/nodes/x'][:] = nc_node_x
sword['/nodes/y'][:] = nc_node_y
sword['/centerlines/node_id'][:] = nc_cl_nodes
sword.close()


####################################################################################
####################################################################################

import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt


shp = gp.read_file('/Users/ealtenau/Documents/SWORD_Dev/outputs/Topology/AS/b35/as_sword_reaches_hb35_v17_FG1_pts.shp')
# shp = gp.read_file('/Users/ealtenau/Documents/SWORD_Dev/outputs/Topology/AS/b35/as_sword_reaches_hb35_v17_pts.shp')

pts = np.array(shp['geom1_rch_'])
unq_pts = np.unique(pts)
issues = []
for ind in list(range(len(unq_pts))):
    print(ind, len(unq_pts)-1)
    p = np.where(pts == unq_pts[ind])[0]
    if len(p) > 10:
        issues.append(unq_pts[ind])

print(len(issues))

#########################################################################################
#########################################################################################
#########################################################################################

import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt

sword = nc.Dataset('/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/as_sword_v17.nc','r+')

n = 81390900170051
check = np.where(sword['/nodes/node_id/'][:] == n)[0]
sword['/nodes/cl_ids/'][:,check]

cl_check = np.where(sword['/centerlines/node_id/'][0,:] == n)[0]
sword['/centerlines/node_id/'][0,cl_check]
sword['/centerlines/cl_id/'][cl_check]

r = 34217600636
rch = np.where(sword['/reaches/reach_id/'][:] == r)[0]

sword['/reaches/reach_id/'][rch]
sword['/reaches/n_rch_up/'][rch] = 0
sword['/reaches/n_rch_down/'][rch] = 1
sword['/reaches/rch_id_up/'][:,rch] = 0
sword['/reaches/rch_id_dn/'][:,rch] = 0
sword.close()

#########################################################################################
#########################################################################################
#########################################################################################


import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt

sword = nc.Dataset('/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/oc_sword_v17.nc','r+')

rch_neg = np.where(sword['/reaches/dist_out/'][:] < 0)[0]
node_neg = np.where(sword['/nodes/dist_out/'][:]  < 0)[0]
print(rch_neg);print(node_neg)

sword['/reaches/dist_out/'][rch_neg] = abs(sword['/reaches/dist_out/'][rch_neg])
sword['/nodes/dist_out/'][node_neg] = abs(sword['/nodes/dist_out/'][node_neg])


rch = np.where(sword['/reaches/reach_id/'][:] == 56259100225)[0]
sword['/reaches/reach_length/'][rch] = 10697.08948218
sword.close()

#########################################################################################
#########################################################################################
#########################################################################################

import numpy as np
import netCDF4 as nc
import geopandas as gp

region = 'OC'
version = 'v17'

gpkg_fn1 = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/gpkg/'\
    +region.lower()+'_sword_reaches_'+version+'.gpkg'
gpkg_fn2 = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Topology/'+version+'/'+region+\
        '/dist_out_updates/'+region.lower()+'_sword_reaches_'+version+'_distout_update.gpkg'

gpkg1 = gp.read_file(gpkg_fn1)
gpkg2 = gp.read_file(gpkg_fn2)

np.unique(gpkg1['dist_out']-gpkg2['dist_out']) 
np.unique(gpkg1['dist_out']-gpkg2['dist_out2']) #should be zero


#########################################################################################
#########################################################################################
#########################################################################################

import numpy as np
import netCDF4 as nc
import geopandas as gp
import pandas as pd 

region = 'NA'
version = 'v17'
sword_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
out_dir = '/Users/ealtenau/Documents/SWORD_Dev/update_requests/'+version+'/'+region+'/'

sword = nc.Dataset(sword_dir)
unq_rchs = np.unique(sword['/centerlines/reach_id/'][0,:])
flag = []
for r in list(range(len(unq_rchs))):
    print(r, len(unq_rchs)-1)
    rch = np.where(sword['/centerlines/reach_id/'][0,:] == unq_rchs[r])[0] #16220100033
    sort_ids = np.argsort(sword['/centerlines/cl_id/'][rch])
    node_nums = np.array([int(str(n)[-4:-1]) for n in sword['/centerlines/node_id/'][0,rch[sort_ids]]])
    idx = np.unique(node_nums, return_index=True)[1]
    old_nums = np.array([node_nums[index] for index in sorted(idx)])
    node_diff = abs(np.diff(old_nums))
    if len(node_diff) > 0:
        if max(node_diff) > 1:
            flag.append(unq_rchs[r])

df = {'reach_id': np.array(flag).astype('int64')}
df = pd.DataFrame(df)
# df.to_csv(out_dir+region.lower()+'_node_order_problems.csv', index=False)
print(len(flag)/len(unq_rchs)*100)
sword.close()

####################################################################################
####################################################################################
####################################################################################

import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt

reaches = pd.read_csv('/Users/ealtenau/Documents/SWORD_Dev/swot_data/glows_sword_calval_reaches/S2_Validation_reaches.csv')
sword = nc.Dataset('/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16_glows/netcdf/eu_sword_v16_glows.nc')

nc_node_rchs = sword['/nodes/reach_id/'][:]
nc_node_ids = sword['/nodes/node_id/'][:]
nc_glows_ids = sword['/nodes/glows_river_id/'][:]
unq_rchs = np.unique(reaches['reach_id'])

for r in list(range(len(unq_rchs))):
    print(r, len(unq_rchs)-1)
    pts = np.where(nc_node_rchs == unq_rchs[r])[0]
    if len(pts) == 0:
        continue
    else:
        if 'node_ids' in locals():
            node_ids = np.append(node_ids, nc_node_ids[pts])
            glows_ids = np.append(glows_ids, nc_glows_ids[pts])
        else:
            
            node_ids = nc_node_ids[pts]
            glows_ids = nc_glows_ids[pts]


df = pd.DataFrame(np.array([node_ids, glows_ids]).T)
df.rename(columns={0:'reach_id', 1:'glows_id'},inplace=True)
df.to_csv('/Users/ealtenau/Documents/SWORD_Dev/swot_data/glows_sword_calval_reaches/reach_csv_glows_s2validation/sword_glows_s2_validation_regions2.csv', index = False)
sword.close()

####################################################################################
####################################################################################
####################################################################################

import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt

calval = pd.read_csv('/Users/ealtenau/Documents/SWORD_Dev/swot_data/glows_sword_calval_reaches/reach_ids_4_dongmei.csv')
calval2 = pd.read_csv('/Users/ealtenau/Documents/SWORD_Dev/swot_data/glows_sword_calval_reaches/dswx_reach_ids.csv')
sword = pd.read_csv('/Users/ealtenau/Documents/SWORD_Dev/outputs/Version_Differences/v17/OC_ReachIDs_v17_vs_v16.csv')

rchs1 = np.array(calval['reach_id'])
rchs2 = np.array(calval2['reach_id'])
all_rchs = np.append(rchs1, rchs2)
all_rchs = np.unique(all_rchs)
check_rchs = all_rchs

v16_rchs = np.array(sword['v16_reach_id'])
v17_rchs = np.array(sword['v17_reach_id'])
flag = np.array(sword['boundary_flag'])


keep = np.where(np.in1d(v16_rchs, check_rchs)==True)[0]
sword_sub = sword.iloc[keep]

calval_flag = np.zeros(len(sword_sub))
for r in list(range(len(check_rchs))):
    rch = np.where(sword_sub['v16_reach_id'] == check_rchs[r])[0]
    if len(rch) == 0:
        continue
    if len(rch) > 1 or flag[rch] == 1:
        calval_flag[rch] = 1
sword_sub['calval_flag'] = calval_flag

len(np.where(calval_flag > 0)[0])
# check = np.where(calval_flag > 0)[0]
# sword_sub['v16_reach_id'].iloc[check]

na_sub = sword_sub.copy()
as_sub = sword_sub.copy()
sa_sub = sword_sub.copy()
eu_sub = sword_sub.copy()
af_sub = sword_sub.copy()
oc_sub = sword_sub.copy()
all_calval = pd.concat([na_sub,sa_sub,eu_sub,af_sub,oc_sub])
#print(len(np.unique(all_calval['v16_reach_id'])), len(check_rchs))

missing = check_rchs[np.where(np.in1d(check_rchs,np.unique(all_calval['v16_reach_id'])) == False)[0]]
d = {'reach_id': np.array(missing).astype('int64')}
df = pd.DataFrame(d)
df.to_csv('/Users/ealtenau/Documents/SWORD_Dev/outputs/Version_Differences/v17/CalVal_Reaches_v17_vs_v16_Deletions.csv', index=False)

all_calval.to_csv('/Users/ealtenau/Documents/SWORD_Dev/outputs/Version_Differences/v17/CalVal_Reaches_v17_vs_v16.csv', index = False)

len(np.unique(all_calval['v16_reach_id'].iloc[np.where(all_calval['calval_flag'] == 1)[0]]))

#########################################################################################
#########################################################################################
#########################################################################################


import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt

sword = nc.Dataset('/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/oc_sword_v17.nc','r+')
# np.unique(sword['/nodes/ext_dist_coef/'][:])

update = np.where((sword['/nodes/ext_dist_coef/'][:] > 3)&(sword['/nodes/n_chan_mod/'][:] == 1))[0]
# np.unique(sword['/nodes/ext_dist_coef/'][update])
# np.unique(sword['/nodes/n_chan_mod/'][update])
# len(update)/len(sword['/nodes/ext_dist_coef/'][:])*100

sword['/nodes/ext_dist_coef/'][update] = 3
sword.close()

#########################################################################################
#########################################################################################
#########################################################################################

import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt

sword = nc.Dataset('/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16_glows/netcdf/oc_sword_v16_glows.nc','r+')
gpkg = gp.read_file('/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16_glows/gpkg/oc_sword_nodes_v16_glows_comid.gpkg')
sword_comid = np.array(gpkg['COMID'])
nid = np.array(sword.groups['nodes'].variables['node_id'][:])
print(np.unique(gpkg['node_id']-nid))

sword.groups['nodes'].createVariable('comid', 'i8', ('num_nodes',), fill_value=-9999.)
sword.groups['nodes'].variables['comid'][:] = sword_comid 
# sword.groups['nodes']
sword.close()
del(gpkg)

#########################################################################################
#########################################################################################
#########################################################################################

import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt

sword = nc.Dataset('/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17_glows/netcdf/sa_sword_v17_glows.nc')
nlon = np.array(sword.groups['nodes'].variables['x'][:])
nlat = np.array(sword.groups['nodes'].variables['y'][:])
wth_med = np.array(sword.groups['nodes'].variables['glows_wth_med'][:])
sword.close()

data = np.where(wth_med>-9999)[0]
print(len(data)/len(wth_med)*100, '% Node Coverage') #89% node coverage in Ohio Basin. 


plt.scatter(nlon, nlat, c='black', s=5)
plt.scatter(nlon, nlat, c=np.log(wth_med), s=5, cmap='rainbow')
plt.show()


#########################################################################################
#########################################################################################
#########################################################################################

import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt

sword17 = nc.Dataset('/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/sa_sword_v17.nc')
sword18 = nc.Dataset('/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v18/netcdf/sa_sword_v18_pacora_only.nc')

rch17 = np.array(sword17.groups['reaches'].variables['reach_id'][:])
rch18 = np.array(sword18.groups['reaches'].variables['reach_id'][:])
dist17 = np.array(sword17.groups['reaches'].variables['dist_out'][:])
dist18 = np.array(sword18.groups['reaches'].variables['dist_out'][:])
wse17 = np.array(sword17.groups['reaches'].variables['wse'][:])
wse18 = np.array(sword18.groups['reaches'].variables['wse'][:])
facc17 = np.array(sword17.groups['reaches'].variables['facc'][:])
facc18 = np.array(sword18.groups['reaches'].variables['facc'][:])
strm17 = np.array(sword17.groups['reaches'].variables['stream_order'][:])
strm18 = np.array(sword18.groups['reaches'].variables['stream_order'][:])
wth17 = np.array(sword17.groups['reaches'].variables['width'][:])
wth18 = np.array(sword18.groups['reaches'].variables['width'][:])
###
ms17 = np.array(sword17.groups['reaches'].variables['main_side'][:])
ms18 = np.array(sword18.groups['reaches'].variables['main_side'][:])
name17 = np.array(sword17.groups['reaches'].variables['river_name'][:])
name18 = np.array(sword18.groups['reaches'].variables['river_name'][:])
end17 = np.array(sword17.groups['reaches'].variables['end_reach'][:])
end18 = np.array(sword18.groups['reaches'].variables['end_reach'][:])
net17 = np.array(sword17.groups['reaches'].variables['network'][:])
net18 = np.array(sword18.groups['reaches'].variables['network'][:])
freq17 = np.array(sword17.groups['reaches'].variables['path_freq'][:])
freq18 = np.array(sword18.groups['reaches'].variables['path_freq'][:])
order17 = np.array(sword17.groups['reaches'].variables['path_order'][:])
order18 = np.array(sword18.groups['reaches'].variables['path_order'][:])
segs17 = np.array(sword17.groups['reaches'].variables['path_segs'][:])
segs18 = np.array(sword18.groups['reaches'].variables['path_segs'][:])

r = np.where(np.in1d(rch18, rch17)==True)[0]
np.unique(dist17-dist18[r])
np.unique(wse17-wse18[r])
np.unique(facc17-facc18[r])
np.unique(wth17-wth18[r])
np.unique(strm17-strm18[r])
np.unique(ms17-ms18[r])
# np.unique(name17-name18[r])
np.unique(net17-net18[r])
np.unique(end17-end18[r])
np.unique(freq17-freq18[r])
np.unique(order17-order18[r])
np.unique(segs17-segs18[r])

node17 = np.array(sword17.groups['nodes'].variables['node_id'][:])
nrch17 = np.array(sword17.groups['nodes'].variables['reach_id'][:])
node18 = np.array(sword18.groups['nodes'].variables['node_id'][:])
ndist17 = np.array(sword17.groups['nodes'].variables['dist_out'][:])
ndist18 = np.array(sword18.groups['nodes'].variables['dist_out'][:])
nwse17 = np.array(sword17.groups['nodes'].variables['wse'][:])
nwse18 = np.array(sword18.groups['nodes'].variables['wse'][:])
nfacc17 = np.array(sword17.groups['nodes'].variables['facc'][:])
nfacc18 = np.array(sword18.groups['nodes'].variables['facc'][:])
nwth17 = np.array(sword17.groups['nodes'].variables['width'][:])
nwth18 = np.array(sword18.groups['nodes'].variables['width'][:])
nstrm17 = np.array(sword17.groups['nodes'].variables['stream_order'][:])
nstrm18 = np.array(sword18.groups['nodes'].variables['stream_order'][:])
###
nms17 = np.array(sword17.groups['nodes'].variables['main_side'][:])
nms18 = np.array(sword18.groups['nodes'].variables['main_side'][:])
nname17 = np.array(sword17.groups['nodes'].variables['river_name'][:])
nname18 = np.array(sword18.groups['nodes'].variables['river_name'][:])
nend17 = np.array(sword17.groups['nodes'].variables['end_reach'][:])
nend18 = np.array(sword18.groups['nodes'].variables['end_reach'][:])
nnet17 = np.array(sword17.groups['nodes'].variables['network'][:])
nnet18 = np.array(sword18.groups['nodes'].variables['network'][:])
nfreq17 = np.array(sword17.groups['nodes'].variables['path_freq'][:])
nfreq18 = np.array(sword18.groups['nodes'].variables['path_freq'][:])
norder17 = np.array(sword17.groups['nodes'].variables['path_order'][:])
norder18 = np.array(sword18.groups['nodes'].variables['path_order'][:])
nsegs17 = np.array(sword17.groups['nodes'].variables['path_segs'][:])
nsegs18 = np.array(sword18.groups['nodes'].variables['path_segs'][:])

n = np.where(np.in1d(node18, node17)==True)[0]
np.unique(node17-node18[n])
np.unique(ndist17-ndist18[n])
np.unique(nwse17-nwse18[n])
np.unique(nfacc17-nfacc18[n])
np.unique(nwth17-nwth18[n])
np.unique(nstrm17-nstrm18[n])
np.unique(nms17-nms18[n])
# np.unique(nname17-nname18[n])
np.unique(nnet17-nnet18[n])
np.unique(nend17-nend18[n])
np.unique(nfreq17-nfreq18[n])
np.unique(norder17-norder18[n])
np.unique(nsegs17-nsegs18[n])

cl17 = np.array(sword17.groups['centerlines'].variables['cl_id'][:])
cl18 = np.array(sword18.groups['centerlines'].variables['cl_id'][:])
clx17 = np.array(sword17.groups['centerlines'].variables['x'][:])
clx18 = np.array(sword18.groups['centerlines'].variables['x'][:])
cly17 = np.array(sword17.groups['centerlines'].variables['y'][:])
cly18 = np.array(sword18.groups['centerlines'].variables['y'][:])
clrch17 = np.array(sword17.groups['centerlines'].variables['reach_id'][:])
clrch18 = np.array(sword18.groups['centerlines'].variables['reach_id'][:])
clnode17 = np.array(sword17.groups['centerlines'].variables['node_id'][:])
clnode18 = np.array(sword18.groups['centerlines'].variables['node_id'][:])

c = np.where(np.in1d(cl18, cl17)==True)[0]
np.unique(cl17-cl18[c])
np.unique(clx17-clx18[c])
np.unique(cly17-cly18[c])
np.unique(clrch17[:]-clrch18[:,c])
np.unique(clnode17[:]-clnode18[:,c])

# test = np.where(ndist17-ndist18[n] != 0)[0]
# test2 = np.where(nfacc17-nfacc18[n] != 0)[0]
# len(np.unique(nrch17[test]))
# len(np.unique(nrch17[test2]))
# np.cumsum(ndist17[test[0]])

# sword18.groups['nodes'].variables['facc'][n] = nfacc17
# sword18.groups['nodes'].variables['dist_out'][n] = ndist17

sword17.close()
sword18.close()

##############################################################################################

import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt

sword18 = nc.Dataset('/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v18/netcdf/sa_sword_v18.nc')
ndist18 = np.array(sword18.groups['nodes'].variables['dist_out'][:])
nrch18 = np.array(sword18.groups['nodes'].variables['reach_id'][:])
rch18 = np.array(sword18.groups['reaches'].variables['reach_id'][:])
dist18 = np.array(sword18.groups['reaches'].variables['dist_out'][:])
len18 = np.array(sword18.groups['reaches'].variables['reach_length'][:])
node18 = np.array(sword18.groups['nodes'].variables['node_id'][:])
nlen18 = np.array(sword18.groups['nodes'].variables['node_length'][:])

problem = []
for r in list(range(len(rch18))):
    nds = np.where(nrch18 == rch18[r])[0]
    sort_nodes = np.argsort(node18[nds])
    node_cs = np.cumsum(nlen18[nds[sort_nodes]])
    if np.round(max(node_cs)-len18[r]) != 0:
        problem.append(rch18[r])
        print(rch18[r])
        print(max(node_cs), len18[r])
        print(max(ndist18[nds]), dist18[r])

len(problem)

sword18.close()

############################################################################################
############################################################################################
############################################################################################

import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt

region = 'SA'
version = 'v18'
nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'

sword = nc.Dataset(nc_fn)
cl_id = np.array(sword.groups['centerlines'].variables['cl_id'][:])
cl_rchs = np.array(sword.groups['centerlines'].variables['reach_id'][0,:])
clx = np.array(sword.groups['centerlines'].variables['x'][:])
cly = np.array(sword.groups['centerlines'].variables['y'][:])

r = 67209900201
rch = np.where(cl_rchs == r)[0]
sort_inds = np.argsort(cl_id[rch])

plt.plot(clx[rch[sort_inds]], cly[rch[sort_inds]])
plt.show()


############################################################################################
############################################################################################
############################################################################################

import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt

region = 'SA'
version = 'v18'
nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'_pacora_only.nc'

sword = nc.Dataset(nc_fn, 'r+')
cl_rchs = np.array(sword.groups['centerlines'].variables['reach_id'][:])

rch = 67209900151
r = np.where(cl_rchs[0,:] == rch)[0]
cl_rchs[1,r]
cl_rchs[2,r]

rch = 67209900091
r = np.where(cl_rchs[0,:] == rch)[0]
cl_rchs[1,r[-1]] = 67209900101

rch = 67209900111
r = np.where(cl_rchs[0,:] == rch)[0]
cl_rchs[1,r[-1]] = 67209900121

rch = 67209900131
r = np.where(cl_rchs[0,:] == rch)[0]
cl_rchs[2,r[-1]] = 67209900141

rch = 67209900151
r = np.where(cl_rchs[0,:] == rch)[0]
cl_rchs[2,r[-1]] = 67209900161

sword.groups['centerlines'].variables['reach_id'][:] = cl_rchs
sword.close()


############################################################################################
############################################################################################
############################################################################################

import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt

region = 'SA'
version = 'v18'
sword_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'\
    +region.lower()+'_sword_'+version+'.nc'
nc_file = '/Users/ealtenau/Documents/SWORD_Dev/update_requests/v18/'+region+'/channel_additions/'\
    +region.lower()+'_channel_additions.nc'

sword = nc.Dataset(sword_fn, 'r+')
additions = nc.Dataset(nc_file)

clx = np.array(sword['/centerlines/x'][:])
cly = np.array(sword['/centerlines/y'][:])
cl_nodes = np.array(sword['/centerlines/node_id'][0,:])
nodes = np.array(sword['/nodes/node_id'][:])
facc = np.array(sword['/nodes/facc'][:])
ax = np.array(additions['/centerlines/x'][:])
ay = np.array(additions['/centerlines/y'][:])
afacc = np.array(additions['/centerlines/flowacc'][:])

node_l6 = np.array([int(str(ind)[0:6]) for ind in nodes])
cl_l6 = np.array([int(str(ind)[0:6]) for ind in cl_nodes])
nind = np.where(node_l6 == 672099)[0]
cind = np.where(cl_l6 == 672099)[0]

add_pts = np.vstack((ax, ay)).T
cl_pts = np.vstack((clx[cind], cly[cind])).T
kdt = sp.cKDTree(add_pts)
pt_dist, pt_ind = kdt.query(cl_pts, k = 5)

cl_facc = np.median(afacc[pt_ind], axis = 1)
for n in list(range(len(nodes[nind]))):
    nix = np.where(cl_nodes[cind] == nodes[nind[n]])
    facc[nind[n]] = np.max(cl_facc[nix])

update = np.where(facc[nind] > 1000)[0]
facc[nind[update]] = 259.5

sword['/nodes/facc'][nind] = facc[nind]
sword.close()
additions.close()

#########################################################################################
#########################################################################################
#########################################################################################

import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt

region = 'OC'
sword16 = nc.Dataset('/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16/netcdf/'+region.lower()+'_sword_v16.nc')
sword17 = nc.Dataset('/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/'+region.lower()+'_sword_v17.nc')
rch_attr = list(sword16.groups['reaches'].variables.keys())
node_attr = list(sword16.groups['nodes'].variables.keys())

## reaches 
for r in list(range(len(rch_attr))):
    min16 = np.min(np.array(sword16.groups['reaches'].variables[rch_attr[r]][:]))
    min17 = np.min(np.array(sword17.groups['reaches'].variables[rch_attr[r]][:]))
    print(rch_attr[r], 'v16:', min16, 'v17:', min17)

## nodes
for n in list(range(len(node_attr))):
    min16 = np.min(np.array(sword16.groups['nodes'].variables[node_attr[n]][:]))
    min17 = np.min(np.array(sword17.groups['nodes'].variables[node_attr[n]][:]))
    print(node_attr[n], 'v16:', min16, 'v17:', min17)

sword16.close(); sword17.close()

#########################################################################################
#########################################################################################
#########################################################################################

import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt
import glob
import os

def aggregate_files(trib_files):
    
    for f in list(range(len(trib_files))):
        gdf = gp.read_file(trib_files[f])
        if f == 0:
            gdf_all = gdf.copy()
        else:
            gdf_all = pd.concat([gdf_all, gdf], ignore_index=True)

    return gdf_all
    
region = 'NA'
version='v18'
# sword = gp.read_file('/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v18/shp/NA/na_sword_reaches_hb74_v18.shp')
# mhv = gp.read_file('/Users/ealtenau/Documents/SWORD_Dev/inputs/MHV_SWORD/gpkg/NA/additions/mhv_sword_hb74_pts_v18_additions.gpkg')
nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
trib_fn = '/Users/ealtenau/Documents/SWORD_Dev/inputs/MHV_SWORD/gpkg/NA/additions/'
trib_files = glob.glob(os.path.join(trib_fn, '*.gpkg'))
mhv = aggregate_files(trib_files)

sword = nc.Dataset(nc_fn)
sword_len = np.sum(np.array(sword['/reaches/reach_length/'][:]))

keep = np.where(np.array(mhv['add_flag'])>0)[0]
unq_rchs = np.unique(np.array(mhv['reach_id'])[keep])
mhv_rch_len = np.zeros(len(unq_rchs))
mhv_rch_strm = np.zeros(len(unq_rchs))
mhv_rch_add = np.zeros(len(unq_rchs))
for r in list(range(len(unq_rchs))):
    rch = np.where(mhv['reach_id'] == unq_rchs[r])[0]
    mhv_rch_len[r] = max(np.unique(mhv['rch_len'][rch]))
    mhv_rch_strm[r] = max(np.unique(mhv['strmorder'][rch]))
    mhv_rch_add[r] = max(mhv['add_flag'][rch])

# add = np.where(mhv_rch_add>0)[0]
# add2 = np.where((mhv_rch_add>0)&(mhv_rch_strm>=4))[0]

mhv_len = np.sum(mhv_rch_len) #down to stream order 3
# mhv_len2 = np.sum(mhv_rch_len[add2]) #down to stream order 4

((mhv_len+sword_len)-sword_len)/sword_len*100 #188% increase in database length
# ((mhv_len2+sword_len)-sword_len)/sword_len*100 #59% increase in database length

mhv_len/(mhv_len+sword_len)*100 #additions would consitute 65% of new database length.
# mhv_len2/(mhv_len2+sword_len)*100 #additions would consitute 37% of new database length.

#overall that is a lot of data for one person to validate and for something to go wrong with non-consistent data bases. 


###################################################################

def aggregate_segs(seg_pts, seg_hwout_pts, seg_junc_pts, up_pts, down_pts):
    outlets = np.unique(seg_pts[np.where(seg_hwout_pts == 2)[0]])
    network = np.zeros(len(seg_pts))
    flag = np.zeros(len(seg_pts))
    start_seg = np.array([outlets[0]])
    cnt = 1
    loop = 1
    while min(flag) == 0:
        print(loop)
        pts = np.where(seg_pts == start_seg)[0]
        #upstream segment 
        up_nodes = np.unique(up_pts[pts]) 
        up_segs = np.unique(seg_pts[np.where(down_pts == up_nodes)[0]])

        if len(up_segs) == 0: #headwater
            network[pts] = cnt
            flag[pts] = 1
            start_pts = np.unique(seg_pts[np.where((seg_hwout_pts == 2)&(flag == 0))[0]]) #outlets 
            if len(start_pts) > 0:
                start_seg = np.array([start_pts[0]])
                cnt = cnt+1
                loop = loop+1
            elif len(start_pts) == 0:
                start_pts = np.unique(seg_pts[np.where((seg_junc_pts == 1)&(flag == 0))[0]]) #junctions
                if len(start_pts) > 0:
                    start_seg = np.array([start_pts[0]])
                    cnt = cnt+1
                    loop = loop+1
                else:
                    start_pts = np.unique(seg_pts[np.where(flag == 0)[0]]) #junctions
                    if len(start_pts) > 0:
                        start_seg = np.array([start_pts[0]])
                        cnt = cnt+1
                        loop = loop+1
                    else:
                        loop = loop+1
                        continue
        elif len(up_segs) == 1: #normal
            network[pts] = cnt
            flag[pts] = 1
            start_seg = up_segs
            loop = loop+1
        else: #junction
            network[pts] = cnt
            flag[pts] = 1
            start_pts = np.unique(seg_pts[np.where(np.where((seg_junc_pts == 1)&(flag == 0))[0])]) #junctions 
            if len(start_pts) > 0:
                start_seg = np.array([start_pts[0]])
                cnt = cnt+1
                loop = loop+1
            elif len(start_pts) == 0:
                start_pts = np.unique(seg_pts[np.where((seg_hwout_pts == 2)&(flag == 0))[0]]) #junctions
                if len(start_pts) > 0:
                    start_seg = np.array([start_pts[0]])
                    cnt = cnt+1
                    loop = loop+1
                else:
                    start_pts = np.unique(seg_pts[np.where(flag == 0)[0]]) #junctions
                    if len(start_pts) > 0:
                        start_seg = np.array([start_pts[0]])
                        cnt = cnt+1
                        loop = loop+1
                    else:
                        loop = loop+1
                        continue

        if loop > len(np.unique(seg_pts))*2:
            print('LOOP STUCK')
            break

z = np.where(network == 0)[0]
plt.scatter(x_pts, y_pts, c=network, cmap = 'rainbow', s = 5)
plt.scatter(x_pts[z], y_pts[z], c='grey', s = 5)
plt.show()

###################################################################

def filter_sword_flag(seg_pts, seg_ind_pts,flag_pts, x_pts, y_pts):
    cnt=[]
    flag = np.copy(flag_pts)
    check = np.unique(seg_pts[np.where(flag == 0)[0]])
    for s in list(range(len(check))):
        # print(s, len(check)-1)
        line = np.where(seg_pts == check[s])[0]
        # seg_x = x[line]
        # seg_y = y[line]
        seg_lon = x_pts[line]
        seg_lat = y_pts[line]
        seg_ind = seg_ind_pts[line]
        end1, end2 = find_neighbors(seg_pts, flag, x_pts, y_pts, seg_lon, 
                                    seg_lat, seg_ind, check[s], line)
        if len(end1) == 0:
            continue
        elif len(end2) == 0:
            continue
        else:
            # Cond. 1: end 1 has SWORD flag, but end 2 does not. 
            if np.max(end1[:,1]) == 1 and np.max(end2[:,1]) == 0:
                for n in list(range(len(end2))):
                    line2 = np.where(seg_pts == end2[0,0])[0]
                    seg_lon2 = x_pts[line2]
                    seg_lat2 = y_pts[line2]
                    seg_ind2 = seg_ind_pts[line2]
                    ngh_end1, ngh_end2 = find_neighbors(seg_pts, flag, x_pts, y_pts, seg_lon2, 
                                        seg_lat2, seg_ind2, check[s], line2)
                    if n == 0:
                        ngh_end1_all = np.copy(ngh_end1)
                        ngh_end2_all = np.copy(ngh_end2)
                    else:
                        ngh_end1_all = np.concatenate((ngh_end1_all, ngh_end1), axis = 0)
                        ngh_end2_all = np.concatenate((ngh_end2_all, ngh_end2), axis = 0)
                if np.max(ngh_end1_all[:,1]) == 1 or np.max(ngh_end2_all[:,1]) == 1:
                    # print(s, check[s], 'cond.1')
                    flag[line] = 1
                    # flag[line] = 1
                    cnt.append(check[s])
                else:
                    continue
            # Cond. 2: end 2 has SWORD flag, but end 1 does not.
            elif np.max(end1[:,1]) == 0 and np.max(end2[:,1]) == 1:
                for n in list(range(len(end1))):
                    line2 = np.where(seg_pts == end1[0,0])[0]
                    seg_lon2 = x_pts[line2]
                    seg_lat2 = y_pts[line2]
                    seg_ind2 = seg_ind_pts[line2]
                    ngh_end1, ngh_end2 = find_neighbors(seg_pts, flag, x_pts, y_pts, seg_lon2, 
                                        seg_lat2, seg_ind2, check[s], line2)
                    if n == 0:
                        ngh_end1_all = np.copy(ngh_end1)
                        ngh_end2_all = np.copy(ngh_end2)
                    else:
                        ngh_end1_all = np.concatenate((ngh_end1_all, ngh_end1), axis = 0)
                        ngh_end2_all = np.concatenate((ngh_end2_all, ngh_end2), axis = 0)
                if np.max(ngh_end1_all[:,1]) == 1 or np.max(ngh_end2_all[:,1]) == 1:
                    # print(s, check[s], 'cond.2')
                    # flag_all[subset[line]] = 1
                    flag[line] = 1
                    cnt.append(check[s])
                else:
                    continue
            # Cond. 3: Both ends have SWORD flag. 
            elif np.max(end1[:,1]) == 1 and np.max(end2[:,1]) == 1:
                # print(s, check[s], 'cond.3')
                # flag_all[subset[line]] = 1
                flag[line] = 1
                cnt.append(check[s])

            else:
                continue

    return flag, cnt

print('Filtering SWORD Flag')
start = time.time()
flag_filt, count = filter_sword_flag(seg_pts, seg_ind_pts,flag_pts, x_pts, y_pts)
end = time.time()
print(str((end-start)/60) + ' min, Segments corrected: ' + str(len(cnt)))

###############################################################################

def overlapping_files(mhv_lon, mhv_lat, elv_paths):

    #define grwl extent as ogr geometry format.
    poly1 = ogr.Geometry(ogr.wkbLinearRing)
    poly1.AddPoint(min(mhv_lon), max(mhv_lat))
    poly1.AddPoint(min(mhv_lon), min(mhv_lat))
    poly1.AddPoint(max(mhv_lon), min(mhv_lat))
    poly1.AddPoint(max(mhv_lon), max(mhv_lat))
    poly1.AddPoint(min(mhv_lon), max(mhv_lat))
    mhvGeometry = ogr.Geometry(ogr.wkbPolygon)
    mhvGeometry.AddGeometry(poly1)
    poly_box = mhvGeometry.GetEnvelope()        

    #find overlapping SWOT tracks.
    track_files = []
    for fn in elv_paths:
        # Read raster extent
        # Open the raster file
        raster_ds = gdal.Open(fn)
        raster_geotransform = raster_ds.GetGeoTransform()
        raster_extent = (
            raster_geotransform[0],
            raster_geotransform[0] + raster_geotransform[1] * raster_ds.RasterXSize,
            raster_geotransform[3] + raster_geotransform[5] * raster_ds.RasterYSize,
            raster_geotransform[3]
        )

        # Check for overlap
        overlap = (
            poly_box[0] < raster_extent[1] and
            poly_box[1] > raster_extent[0] and
            poly_box[2] < raster_extent[3] and
            poly_box[3] > raster_extent[2]
        )

        if overlap == True:
            track_files.append(fn)
    
    track_files = np.unique(track_files)

    return(track_files)

###############################################################################


import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt

mhv = nc.Dataset('/Users/ealtenau/Documents/SWORD_Dev/inputs/MHV_SWORD/netcdf/NA/mhv_sword_hb82_pts_v18.nc')
mhv.groups['centerlines']

np.unique(mhv.groups['centerlines'].variables['lakeflag'][:]) 
np.unique(mhv.groups['centerlines'].variables['deltaflag'][:]) 
np.unique(mhv.groups['centerlines'].variables['grand_id'][:]) 
np.unique(mhv.groups['centerlines'].variables['grod_id'][:]) 
np.unique(mhv.groups['centerlines'].variables['grod_fid'][:]) 
np.unique(mhv.groups['centerlines'].variables['hfalls_fid'][:]) 
np.unique(mhv.groups['centerlines'].variables['basin_code'][:]) 
np.unique(mhv.groups['centerlines'].variables['number_obs'][:]) 
np.unique(mhv.groups['centerlines'].variables['orbits'][:,:]) 
np.unique(mhv.groups['centerlines'].variables['lake_id'][:]) 

mhv.close()


###############################################################################
###############################################################################
###############################################################################

#see if any reaches cross segments
np.mean(subcls.rch_len1)
np.mean(subcls.rch_len2)
np.mean(subcls.rch_len3)
np.mean(subcls.rch_len4)
np.mean(subcls.rch_len5)
np.mean(subcls.rch_len6);np.median(subcls.rch_len6)

np.min(subcls.rch_len1)
np.min(subcls.rch_len2)
np.min(subcls.rch_len3)
np.min(subcls.rch_len4)

cp_rchs = np.copy(subcls.reach_id)
unq_rchs = np.unique(cp_rchs)
for r in list(range(len(unq_rchs))):
    pts = np.where(cp_rchs == unq_rchs[r])
    nsegs = len(np.unique(subcls.seg[pts]))
    if nsegs > 1:
        print(r, unq_rchs[r], nsegs)

###############################################################################
###############################################################################
###############################################################################

import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt

mhv_gpkg = gp.read_file('/Users/ealtenau/Documents/SWORD_Dev/inputs/MHV_SWORD/gpkg/NA/additions/mhv_sword_hb91_pts_v18_additions.gpkg')
mhv_nc = nc.Dataset('/Users/ealtenau/Documents/SWORD_Dev/inputs/MHV_SWORD/netcdf copy/NA/mhv_sword_hb91_pts_v18.nc','r+')

add_flag = np.array(mhv_nc['/centerlines/add_flag/'][:])
add_flag[:] = 0
seg_id = np.array(mhv_nc['/centerlines/new_segs/'][:])
cl_id = np.array(mhv_nc['/centerlines/cl_id/'][:])

mhv_segs = np.array(mhv_gpkg['new_segs'][:])
mhv_add = np.array(mhv_gpkg['add_flag'][:], dtype=int)
mhv_id = np.array(mhv_gpkg['cl_id'][:])
unq_segs = np.unique(mhv_segs)
for s in list(range(len(unq_segs))):
    print(s, len(unq_segs)-1)
    swd_pts = np.where(seg_id == unq_segs[s])[0]
    mhv_pts = np.where(mhv_segs == unq_segs[s])[0]
    keep = np.where(mhv_add[mhv_pts]>0)[0]
    swd_ids = cl_id[swd_pts]
    mhv_ids = mhv_id[mhv_pts[keep]]
    fill = np.where(np.in1d(swd_ids, mhv_ids)==True)[0]
    add_flag[swd_pts[fill]] = mhv_add[mhv_pts[keep]]

np.unique(add_flag); add_flag.shape

mhv_nc['/centerlines/add_flag/'][:] = add_flag
mhv_nc.close()

###############################################################################
###############################################################################
###############################################################################

import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt

region = 'NA'
version='v18'
nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
sword = nc.Dataset(nc_fn)
np.where(sword.groups['centerlines'].variables['reach_id'][0,:] == 71181301243)[0]



mhv = nc.Dataset('/Users/ealtenau/Documents/SWORD_Dev/inputs/MHV_SWORD/netcdf/NA/mhv_sword_hb83_pts_v18.nc')


mhv_add_flag = np.array(mhv_nc['/centerlines/add_flag/'][:])
ml2 = np.array([int(str(ind)[0:2]) for ind in mhv_rchs])
rmv = np.where(ml2 == 83)[0]
mhv_add_flag[rmv] = 0


r = 71280000566 #something odd happening in centerline dimension topology in mhv only... 
rch = np.where(subcls.new_reach_id[0,:] == r)[0]
# rch = np.where(subcls.reach_id == 71140900013)[0]
subcls.new_reach_id[0,rch]
subcls.new_reach_id[1,rch]
subcls.new_reach_id[2,rch]
subcls.new_reach_id[3,rch]


subcls.new_reach_id[0,rch] # (new) 71140700123 -> (old) 71140700103; # (new) 71140700103 -> (old) 71140700083
subcls.seg[rch]
subcls.reach_id[rch]
## i think 10 is skipping the reach before 12 for upstream. check that 12's downstream reach is correct...
# 71140900013

# see what is happening in mhv reach 71140700103 should have  71140900013 upstream before 71140700123.

r0 = np.where(subcls.new_reach_id[0,:] == 71140700103)[0]
r1 = np.where(subcls.new_reach_id[0,:] == 71140700123)[0]
r2 = np.where(subcls.new_reach_id[0,:] == 71140900013)[0]
r0_sort = r0[np.argsort(subcls.new_cl_id[r0])]
r1_sort = r1[np.argsort(subcls.new_cl_id[r1])]
r2_sort = r2[np.argsort(subcls.new_cl_id[r2])]

tx = subcls.lon[pt_ind[seg_sort[-1],vals[np.where(idx == ngh_seg_min)[0]]]]
ty = subcls.lat[pt_ind[seg_sort[-1],vals[np.where(idx == ngh_seg_min)[0]]]]
subcls.new_reach_id[0,pt_ind[seg_sort[-1],:]]


plt.plot(subcls.lon[r0_sort], subcls.lat[r0_sort], c = 'black')
plt.plot(subcls.lon[r1_sort], subcls.lat[r1_sort], c = 'gold')
plt.plot(subcls.lon[r2_sort], subcls.lat[r2_sort], c = 'red')
plt.scatter(tx, ty, c = 'cyan')
plt.show()

problems = np.where(np.diff(subcls.rch_dist6[r1_sort]) > 500)[0]



71181300773
71181301293 1763

issues = []
for s in list(range(len(unq_segs))):
    seg = np.where(subcls.seg == unq_segs[s])[0]
    seg_sort = seg[np.argsort(subcls.new_cl_id[seg])]
    unq_rchs = np.unique(subcls.new_reach_id[0,seg_sort])
    for r in list(range(len(unq_rchs))):
        pts = np.where(subcls.new_reach_id[0,seg_sort] == unq_rchs[r])[0]
        d = np.diff(subcls.rch_dist6[seg_sort[pts]])
        prob = np.where(d > 1000)[0]
        if len(prob) > 0:
            issues.append(unq_rchs[r])


rch = np.where(centerlines.reach_id[0,:] == r)[0]
centerlines.reach_id[0,rch]
centerlines.reach_id[1,rch]
centerlines.reach_id[2,rch]
centerlines.reach_id[3,rch]

###############################################################################
###############################################################################
###############################################################################

### redo reach and node lengths with median in added mhv reaches... 

import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt

region = 'NA'
version='v18'
nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
sword = nc.Dataset(nc_fn, 'r+')

cl_rchs = np.array(sword['/centerlines/reach_id'][0,:])
cl_nodes = np.array(sword['/centerlines/node_id'][0,:])
cx = np.array(sword['/centerlines/x'][:])
cy = np.array(sword['/centerlines/y'][:])
reaches = np.array(sword['/reaches/reach_id'][:])
rx = np.array(sword['/reaches/x'][:])
ry = np.array(sword['/reaches/y'][:])
nodes = np.array(sword['/nodes/node_id'][:])
nx = np.array(sword['/nodes/x'][:])
ny = np.array(sword['/nodes/y'][:])
edit_flag = np.array(sword['/reaches/edit_flag'][:])

update = reaches[np.where(edit_flag == '7')[0]]
for r in list(range(len(update))):
    print(r, len(update)-1)
    cind = np.where(cl_rchs == update[r])[0]
    rind = np.where(reaches == update[r])[0]
    rx[rind] = np.median(cx[cind])
    ry[rind] = np.median(cy[cind])
    unq_nodes = np.unique(cl_nodes[cind])
    for n in list(range(len(unq_nodes))):
        cnind = np.where(cl_nodes == unq_nodes[n])[0]
        nind = np.where(nodes == unq_nodes[n])[0]
        nx[nind] = np.median(cx[cnind])
        ny[nind] = np.median(cy[cnind])

sword['/reaches/x'][:] = rx
sword['/reaches/y'][:] = ry
sword['/nodes/x'][:] = nx
sword['/nodes/y'][:] = ny
sword.close()

###############################################################################
###############################################################################
###############################################################################

### redo negative reach and node lengths 

import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt
from geopy import distance

def get_distances(lon,lat):
    traces = len(lon) -1
    distances = np.zeros(traces)
    for i in range(traces):
        start = (lat[i], lon[i])
        finish = (lat[i+1], lon[i+1])
        distances[i] = distance.geodesic(start, finish).m
    distances = np.append(0,distances)
    return distances

region = 'NA'
version='v18'
nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
sword = nc.Dataset(nc_fn, 'r+')

cl_rchs = np.array(sword['/centerlines/reach_id'][0,:])
cl_nodes = np.array(sword['/centerlines/node_id'][0,:])
cl_id = np.array(sword['/centerlines/cl_id'][:])
cx = np.array(sword['/centerlines/x'][:])
cy = np.array(sword['/centerlines/y'][:])
reaches = np.array(sword['/reaches/reach_id'][:])
rch_len = np.array(sword['/reaches/reach_length'][:])
nodes = np.array(sword['/nodes/node_id'][:])
node_len = np.array(sword['/nodes/node_length'][:])

rch_update = reaches[np.where(rch_len < 30)[0]]
node_update = nodes[np.where(node_len < 30)[0]]

for r in list(range(len(rch_update))):
    cind = np.where(cl_rchs == rch_update[r])[0]
    sort_ids = cind[np.argsort(cl_id[cind])]
    rind = np.where(reaches == rch_update[r])[0]
    x_coords = cx[sort_ids]
    y_coords = cy[sort_ids]
    diff = get_distances(x_coords,y_coords)
    rlen = max(np.cumsum(diff))
    # print(r, rch_len[rind], rlen)
    rch_len[rind] = rlen

for n in list(range(len(node_update))):
    cind = np.where(cl_nodes == node_update[n])[0]
    sort_ids = cind[np.argsort(cl_id[cind])]
    nind = np.where(nodes == node_update[n])[0]
    x_coords = cx[sort_ids]
    y_coords = cy[sort_ids]
    diff = get_distances(x_coords,y_coords)
    nlen = max(np.cumsum(diff))
    # print(n, node_len[nind], nlen)
    node_len[nind] = nlen

sword['/reaches/reach_length'][:] = rch_len
sword['/nodes/node_length'][:] = node_len
sword.close()

###############################################################################
###############################################################################
###############################################################################

import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt
from geopy import distance

region = 'NA'
version='v18'
nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
sword = nc.Dataset(nc_fn,'r+')

cl_rchs = np.array(sword['/centerlines/reach_id'][:])
cl_id = np.array(sword['/centerlines/cl_id'][:])

rch = np.where(cl_rchs[0,:] == 82247900161)[0]
mn = np.where(cl_id[rch] == min(cl_id[rch]))[0]
mx = np.where(cl_id[rch] == max(cl_id[rch]))[0]
# cl_rchs[0,rch]
cl_rchs[1,rch]
cl_rchs[1,rch[mx]] = 82247900186


cl_rchs[2,rch]
cl_rchs[3,rch]

sword['/centerlines/reach_id'][:] = cl_rchs
sword.close()

###############################################################################
###############################################################################
###############################################################################
import matplotlib.pyplot as plt

def find_boundaries(subcls):

    # Combine lake and delta flags into one vector.
    lake_coast_flag = np.copy(subcls.lake)
    lake_coast_flag[np.where(subcls.delta > 0)] = 3

    # Set variables.
    reach_nums = np.zeros(len(subcls.ind))
    Type = np.repeat(1,len(subcls.ind))
    Len = np.zeros(len(subcls.ind))
    cnt = 1

    # Loop through each basin and identify SWOT orbit, lake, and dam boundaries.
    uniq_basins = np.unique(subcls.basins)
    for ind in list(range(len(uniq_basins))):
        basin = np.where(subcls.basins == uniq_basins[ind])[0]
        uniq_segs = np.unique(subcls.seg[basin])
        for idx in list(range(len(uniq_segs))):
            seg = np.where(subcls.seg[basin] == uniq_segs[idx])[0]
            sort_ids = np.argsort(subcls.ind[basin[seg]])
            ID = subcls.ind[basin[seg[sort_ids]]]
            dist = subcls.dist[basin[seg[sort_ids]]]
            lakes = lake_coast_flag[basin[seg[sort_ids]]]
            grod = subcls.grod[basin[seg[sort_ids]]]
            dams = np.where((grod > 0) & (grod <= 4))[0]

            # Find lake and dam boundaries.
            bounds = []
            if max(lakes) > 0:
                # print(idx)
                bounds.extend(np.where(np.diff(lakes) != 0)[0])

            if len(dams) > 0:
                # print(idx)
                for d in list(range(len(dams))):
                    if dams[d] < 3:
                        bounds.extend(np.array([5])) #first 5ish points are a dam
                    if dams[d] > len(ID)-4:
                        bounds.extend(np.array([len(ID) - 6])) #last 5ish points are a dam 
                    else:
                        b1 = np.array([dams[d] - 2])
                        b2 = np.array([dams[d] + 2])
                        bounds.extend(b1)
                        bounds.extend(b2)
            
            # Account for odd basin boundaries
            basin_breaks = np.where(np.diff(dist) > 250)[0]
            if len(basin_breaks) > 0:
                bounds.extend(basin_breaks+1)

            if len(bounds) > 0:
                bounds.extend(np.where(ID == np.min(ID))[0])
                bounds.extend(np.where(ID == np.max(ID))[0])
                bounds = np.sort(bounds)
                ### number between boundaries
                for b in list(range(len(bounds)-1)):
                    reach_nums[basin[seg[sort_ids[bounds[b]:bounds[b+1]+1]]]] = cnt 
                    cnt = cnt+1
                    # reach_nums[basin[seg[sort_ids]]]
            else:
                reach_nums[basin[seg[sort_ids]]] = cnt 
                cnt = cnt+1

            # Create reach "type" flag based on boundaries.
            unq_rchs = np.unique(reach_nums[basin[seg[sort_ids]]])
            for r in list(range(len(unq_rchs))):
                rind = np.where(reach_nums[basin[seg[sort_ids]]] == unq_rchs[r])[0]
                Len[basin[seg[sort_ids[rind]]]] = max(dist[rind]) - min(dist[rind])
                if max(lakes[rind]) > 0:
                    Type[basin[seg[sort_ids[rind]]]] = 3
                if max(lakes[rind]) == 3:
                    Type[basin[seg[sort_ids[rind]]]] = 5
                if max(grod[rind]) > 0:
                    Type[basin[seg[sort_ids[rind]]]] = 4

    return(reach_nums, Type, Len)




subcls_rch_id0 = np.copy(subcls.rch_id0)
subcls_rch_len0 = np.copy(subcls.rch_len0)
subcls_dist = np.copy(subcls.dist)
subcls_ind = np.copy(subcls.ind)
max_dist = 20000
def cut_reaches(subcls_rch_id0, subcls_rch_len0, subcls_dist,
                subcls_ind, max_dist):

    # Setting variables.
    cnt = np.max(subcls_rch_id0)+1
    new_rch_id = np.copy(subcls_rch_id0)
    new_rch_dist = np.copy(subcls_rch_len0)
    uniq_rch = np.unique(subcls_rch_id0[np.where(subcls_rch_len0 >= max_dist)])

    # Loop through each reach that is greater than the maximum distance and
    # divide it into smaller reaches.
    for ind in list(range(len(uniq_rch))):

        # Finding current reach id and length.
        rch = np.where(subcls_rch_id0 == uniq_rch[ind])[0]
        sort_ids = np.argsort(subcls_ind[rch])
        distance = subcls_dist[rch[sort_ids]]
        ID = subcls_ind[rch[sort_ids]]
        # Setting temporary variables.
        temp_rch_id = np.zeros(len(rch))
        temp_rch_dist = np.zeros(len(rch))
        # Determining the number of divisions to cut the reach.
        d = np.unique(subcls_rch_len0[rch])
        divs = np.around(d/10000)
        divs_dist = d/divs

        # Determining new boundaries to cut the reaches.
        break_index = np.zeros(int(divs-1))
        for idx in range(int(divs)-1):
            dist = divs_dist*(range(int(divs)-1)[idx]+1)+np.min(distance)
            cut = np.where(abs(distance - dist) == np.min(abs(distance - dist)))[0][0]
            break_index[idx] = cut
        div_ends = np.array([np.where(ID == np.min(ID))[0][0],np.where(ID == np.max(ID))[0][0]])
        borders = np.insert(div_ends, 0, break_index)
        borders = np.sort(borders)

        # Numbering the new cut reaches.
        for idy in list(range(len(borders)-1)):
            index1 = borders[idy]
            index2 = borders[idy+1]

            ID1 = ID[index1]
            ID2 = ID[index2]

            if ID1 > ID2:
                vals = np.where((ID2 <= ID) &  (ID <= ID1))[0]
            else:
                vals = np.where((ID1 <= ID) &  (ID <= ID2))[0]

            avg_dist = abs(np.max(distance[vals])-np.min(distance[vals]))
            if avg_dist == 0:
                temp_rch_dist[vals] = 90.0
            else:
                temp_rch_dist[vals] = avg_dist

            temp_rch_id[vals] = cnt
            cnt=cnt+1

        new_rch_id[rch[sort_ids]] = temp_rch_id
        new_rch_dist[rch[sort_ids]] = temp_rch_dist
        #if np.max(new_rch_dist[rch])>max_dist:
            #print(ind, 'max distance too long - likely an index problem')

    return new_rch_id, new_rch_dist



#### reach def checks 

plt.plot(subcls.x[basin[seg[sort_ids]]], subcls.y[basin[seg[sort_ids]]], c='red')
plt.scatter(subcls.x[basin[seg[sort_ids]]], subcls.y[basin[seg[sort_ids]]], c=reach_nums[basin[seg[sort_ids]]])
plt.show()

plt.scatter(subcls.x[basin[seg[sort_ids]]], subcls.y[basin[seg[sort_ids]]], c=Type[basin[seg[sort_ids]]])
plt.show()

plt.scatter(subcls.x[basin[seg[sort_ids]]], subcls.y[basin[seg[sort_ids]]], c=Len[basin[seg[sort_ids]]])
plt.show()

seg_check = np.where(subcls.seg == 22)[0]
plt.scatter(subcls.x[seg_check], subcls.y[seg_check], c=reach_nums[seg_check])
plt.show()


#### cut reach checks 

plt.plot(subcls.x[rch[sort_ids]], subcls.y[rch[sort_ids]], c='red')
plt.scatter(subcls.x[rch[sort_ids]], subcls.y[rch[sort_ids]], c=subcls.rch_id0[rch[sort_ids]])
plt.show()

plt.plot(subcls.x[rch[sort_ids]], subcls.y[rch[sort_ids]], c='red')
plt.scatter(subcls.x[rch[sort_ids]], subcls.y[rch[sort_ids]], c=new_rch_id[rch[sort_ids]])
plt.show()

plt.plot(subcls.x[rch[sort_ids]], subcls.y[rch[sort_ids]], c='red')
plt.scatter(subcls.x[rch[sort_ids]], subcls.y[rch[sort_ids]], c=new_rch_dist[rch[sort_ids]])
plt.show()

#Check and correct reaches with odd index problems. 
# issues = rdt.check_rchs(subcls.rch_id1, subcls.rch_dist1, subcls.rch_ind1)


seg_check = np.where(subcls.seg == 22)[0]

reaches = np.copy(subcls.rch_id6)
unq_rchs = np.unique(reaches)
# len(unq_rchs)
for r in list(range(len(unq_rchs))):
    rch = np.where(reaches == unq_rchs[r])[0]
    diff = subcls.ind[rch]-subcls.rch_ind6[rch]
    if max(diff) > 0:
        print(r)

#######################################################################################
#######################################################################################
#######################################################################################

### reach def checks

len(np.unique(subcls.rch_id0))
len(np.unique(subcls.rch_id1))
len(np.unique(subcls.rch_id2))
len(np.unique(subcls.rch_id3))
len(np.unique(subcls.rch_id4))
len(np.unique(subcls.rch_id5))
len(np.unique(subcls.reach_id))

max(subcls.rch_len0)
max(subcls.rch_len1)
max(subcls.rch_len2)
max(subcls.rch_len3)
max(subcls.rch_len4)
max(subcls.rch_len5)
max(subcls.rch_len6)

min(subcls.node_id)
max(subcls.node_len)

###############################################################################
###############################################################################
###############################################################################

### redo negative reach and node lengths 

import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt

region = 'SA'
version='v18'
nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
sword = nc.Dataset(nc_fn, 'r+')

edit_flag = sword['/reaches/edit_flag'][:]
reach_id = sword['/reaches/reach_id'][:]
node_rchs = sword['/nodes/reach_id'][:]
l6 = np.array([int(str(rch)[0:6]) for rch in reach_id])

np.unique(sword['/reaches/edit_flag'][:])
np.unique(sword['/nodes/edit_flag'][:])
sword['/reaches/edit_flag'][:] = np.repeat('NaN', len(sword['/reaches/edit_flag'][:]))
sword['/nodes/edit_flag'][:] = np.repeat('NaN', len(sword['/nodes/edit_flag'][:]))
np.unique(sword['/reaches/edit_flag'][:])
np.unique(sword['/nodes/edit_flag'][:])

### SA only
# keep_rchs = np.where(l6 == 672099)[0]
# sword['/reaches/edit_flag'][keep_rchs] = edit_flag[keep_rchs]
# keep_nodes = np.where(np.in1d(node_rchs, reach_id[keep_rchs]) == True)[0]
# sword['/nodes/edit_flag'][keep_nodes] = np.repeat('7', len(keep_nodes))
# np.unique(sword['/reaches/edit_flag'][:])
# np.unique(sword['/nodes/edit_flag'][:])

sword.close()

###############################################################################
###############################################################################
###############################################################################

### redo negative reach and node lengths 

import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt

nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/inputs/MHV_SWORD/netcdf/NA/mhv_sword_hb91_pts_v18.nc'
mhv = nc.Dataset(nc_fn, 'r+')

segs = np.array(mhv['/centerlines/new_segs/'][:])
index = np.array(mhv['/centerlines/new_segs_ind/'][:])
flag = np.array(mhv['/centerlines/add_flag/'][:])

s = np.where(segs == 1908)[0]
sub = np.where((index[s] < 31) & (index[s] > 7))[0]
index[s[sub]]
flag[s[sub]]


s = np.where(segs == 575)[0]
index[s[433]]

mhv['/centerlines/add_flag/'][s[433]] = 3
mhv.close()


###############################################################################
###############################################################################
###############################################################################
def remove_ghost_juncs(subcls):
    sub_type = np.array([int(str(r)[-1]) for r in subcls.new_reach_id[0,:]])
    ghost_ind = np.where(sub_type == 6)[0]
    ghost = np.unique(subcls.new_reach_id[0,ghost_ind])
    rmv_ghost = []
    for g in list(range(len(ghost))):
        pts = np.where(subcls.new_reach_id[0,:] == ghost[g])[0]
        mn_pt = np.where(subcls.new_cl_id[pts] == min(subcls.new_cl_id[pts]))[0]
        dn_rchs = np.unique(subcls.new_reach_id[:,pts[0]])
        dn_rchs = dn_rchs[dn_rchs != ghost[g]]
        dn_rchs = dn_rchs[dn_rchs>0]
        if len(dn_rchs) > 0:
            dn_pts = np.where(subcls.new_reach_id[1,:] == dn_rchs)[0]
            dn_nghs = np.unique(subcls.new_reach_id[0,dn_pts])
            dn_nghs = dn_nghs[dn_nghs != ghost[g]]
            if len(dn_nghs) > 0:
                for dn in list(range(len(dn_nghs))):
                    pts2 = np.where(subcls.new_reach_id[0,:] == dn_nghs[dn])[0]
                    mn_pt2 = np.where(subcls.new_cl_id[pts2] == min(subcls.new_cl_id[pts2]))[0]
                    if dn_rchs in subcls.new_reach_id[1:3,pts2[mn_pt2]]:
                        ### update mhv topology and record to remove. 
                        rmv_ghost.append(ghost[g])
                        r1 = np.where(subcls.new_reach_id[1,:] == ghost[g])[0]
                        r2 = np.where(subcls.new_reach_id[2,:] == ghost[g])[0]
                        r3 = np.where(subcls.new_reach_id[3,:] == ghost[g])[0]
                        if len(r1) > 0:
                            for r in list(range(len(r1))):
                                subcls.new_reach_id[1,r1] = 0
                                if max(subcls.new_reach_id[1:3,r1]) > 0:
                                    subcls.new_reach_id[1:3,r1] = np.sort(subcls.new_reach_id[1:3,r1])[::-1]
                        if len(r2) > 0:
                            for r in list(range(len(r2))):
                                subcls.new_reach_id[1,r2] = 0
                                if max(subcls.new_reach_id[1:3,r2]) > 0:
                                    subcls.new_reach_id[1:3,r2] = np.sort(subcls.new_reach_id[1:3,r2])[::-1]
                        if len(r3) > 0:
                            for r in list(range(len(r3))):
                                subcls.new_reach_id[1,r3] = 0
                                if max(subcls.new_reach_id[1:3,r3]) > 0:
                                    subcls.new_reach_id[1:3,r3] = np.sort(subcls.new_reach_id[1:3,r3])[::-1]


# test = np.array([1,2,3])
# test[0] = 0
# test[:] = np.sort(test)[::-1]