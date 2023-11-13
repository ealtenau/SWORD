# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:51:24 2019

@author: ealtenau
"""
from __future__ import division
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import random

###############################################################################

class Object(object):
    """
    FUNCTION:
        Creates class object to assign attributes to.
    """
    pass 

###############################################################################
###############################################################################
###############################################################################
region = 'NA'

#fn1 = 'E:/Users/Elizabeth Humphries/Documents/SWORD/For_Server/outputs/Reaches_Nodes_v10/netcdf/'+region.lower()+'_sword_v10.nc'
fn1 = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v15/netcdf/'+region.lower()+'_sword_v15.nc'
#fn2 = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/outputs/'+region+'_swot_coverage.csv'
data1 = nc.Dataset(fn1)
#data2 = pd.read_csv(fn2)
reaches_id = data1.groups['reaches'].variables['reach_id'][:]
reaches_len = data1.groups['reaches'].variables['reach_length'][:]
reaches_slope = data1.groups['reaches'].variables['slope'][:]
reaches_x = data1.groups['reaches'].variables['x'][:]
reaches_y =  data1.groups['reaches'].variables['y'][:]
nodes_id = data1.groups['nodes'].variables['node_id'][:]
data1.close()

##########################################    
nt = np.zeros(len(nodes_id))
for ind in list(range(len(nodes_id))):
    nt[ind] = np.int(np.str(nodes_id[ind])[13:14]) 
    
#len(np.where(nt != 6)[0])
###########################################

Type = np.zeros(len(reaches_id))
for ind in list(range(len(reaches_id))):
    Type[ind] = np.int(np.str(reaches_id[ind])[10:11]) 
    
All = np.where(Type != 6)[0]

ghost = np.where(Type == 6)[0]
dams = np.where(Type == 4)[0]
lakes = np.where(Type == 3)[0]
coast = np.where(Type == 5)[0]
rivers = np.where(Type == 1)[0]

len(rivers)
(len(rivers)/len(All))*100 # 0%

#min_vals = np.array(data2['min'])
#max_vals = np.array(data2['max'])
#min_vals_rivers = min_vals[rivers]
#max_vals_rivers = max_vals[rivers]

###############################################################################

np.max(reaches_len[ghost]/1000)
np.min(reaches_len[ghost]/1000)
np.median(reaches_len[ghost]/1000)

np.max(reaches_len[dams]/1000)
np.min(reaches_len[dams]/1000)
np.median(reaches_len[dams]/1000)

np.max(reaches_len[lakes]/1000)
np.min(reaches_len[lakes]/1000)
np.median(reaches_len[lakes]/1000)

np.max(reaches_len[coast])/1000
np.min(reaches_len[coast]/1000)
np.median(reaches_len[coast]/1000)

np.max(reaches_len[rivers])/1000
np.min(reaches_len[rivers]/1000)
np.median(reaches_len[rivers]/1000)


# (len(np.where(reaches_len[All] == 0)[0])/len(reaches_len[All]))*100 # 0%
#(len(np.where(reaches_len[All] < 8000)[0])/len(reaches_len[All]))*100 # 16.6%
(len(np.where(reaches_len[All] < 5000)[0])/len(reaches_len[All]))*100 # 16.6%
(len(np.where((reaches_len[All] >= 5000) & (reaches_len[All] < 10000))[0])/len(reaches_len[All]))*100 # 34.2%
(len(np.where((reaches_len[All] >= 10000) & (reaches_len[All] <= 20000))[0])/len(reaches_len[All]))*100 # 40.9% 
#(len(np.where((reaches_len[All] > 15000) & (reaches_len[All] <= 20000))[0])/len(reaches_len[All]))*100 # 7.2% 
#(len(np.where(reaches_len[All] < 20000)[0])/len(reaches_len[All]))*100 # 98.9%
(len(np.where(reaches_len[All] > 20000)[0])/len(reaches_len[All]))*100 # 0.03%

# (len(np.where(reaches_len[rivers] == 0)[0])/len(reaches_len[rivers]))*100 # 0%
(len(np.where(reaches_len[rivers] < 5000)[0])/len(reaches_len[rivers]))*100 # 7.7%
(len(np.where((reaches_len[rivers] >= 5000) & (reaches_len[rivers] < 10000))[0])/len(reaches_len[rivers]))*100 # 46.1%
(len(np.where((reaches_len[rivers] >= 10000) & (reaches_len[rivers] <= 20000))[0])/len(reaches_len[rivers]))*100 # 7.8%
#(len(np.where(reaches_len[rivers] < 20000)[0])/len(reaches_len[rivers]))*100 # 99.3%
(len(np.where(reaches_len[rivers] > 20000)[0])/len(reaches_len[rivers]))*100 # 0.7%

np.mean(reaches_len[All])
np.median(reaches_len[All])
np.mean(reaches_len[rivers])
np.median(reaches_len[rivers])

#small = np.where(rd_vals <= 8000)[0]
#(len(np.where(rt_vals[small]==1)[0])/len(small))*100 # 34%
#(len(np.where(rt_vals[small]==3)[0])/len(small))*100 # 40%
#(len(np.where(rt_vals[small]==4)[0])/len(small))*100 # 20%
#(len(np.where(rt_vals[small]==5)[0])/len(small))*100 # 5%



##############################################################################

(len(np.where(min_vals == 0)[0])/len(min_vals))*100 #  
(len(np.where((min_vals > 0) & (min_vals <= 25))[0])/len(min_vals))*100 # 
(len(np.where((min_vals > 25) & (min_vals <= 50))[0])/len(min_vals))*100 # 
(len(np.where((min_vals > 50) & (min_vals <= 75))[0])/len(min_vals))*100 # 
(len(np.where(min_vals > 75)[0])/len(min_vals))*100 # 

(len(np.where(max_vals == 0)[0])/len(max_vals))*100 #  
(len(np.where((max_vals > 0) & (max_vals <= 25))[0])/len(max_vals))*100 # 
(len(np.where((max_vals > 25) & (max_vals <= 50))[0])/len(max_vals))*100 # 
(len(np.where((max_vals > 50) & (max_vals <= 75))[0])/len(max_vals))*100 # 
(len(np.where(max_vals > 75)[0])/len(max_vals))*100 # 

# rivers only
(len(np.where(min_vals[rivers] == 0)[0])/len(min_vals[rivers]))*100 #  
(len(np.where((min_vals[rivers] > 0) & (min_vals[rivers] <= 25))[0])/len(min_vals[rivers]))*100 # 
(len(np.where((min_vals[rivers] > 25) & (min_vals[rivers] <= 50))[0])/len(min_vals[rivers]))*100 # 
(len(np.where((min_vals[rivers] > 50) & (min_vals[rivers] <= 75))[0])/len(min_vals[rivers]))*100 # 
(len(np.where(min_vals[rivers] > 75)[0])/len(min_vals[rivers]))*100 # 

(len(np.where(max_vals[rivers] == 0)[0])/len(max_vals[rivers]))*100 #  
(len(np.where((max_vals[rivers] > 0) & (max_vals[rivers] <= 25))[0])/len(max_vals[rivers]))*100 # 
(len(np.where((max_vals[rivers] > 25) & (max_vals[rivers] <= 50))[0])/len(max_vals[rivers]))*100 # 
(len(np.where((max_vals[rivers] > 50) & (max_vals[rivers] <= 75))[0])/len(max_vals[rivers]))*100 # 
(len(np.where(max_vals[rivers] > 75)[0])/len(max_vals[rivers]))*100 # 



###############################################################################
############################## FIGURES ########################################
###############################################################################

rivers = np.where(Type == 1)[0]
lakes = np.where(Type == 3)[0]
dams = np.where(Type == 4)[0]
coast = np.where(Type == 5)[0]
small = np.where(reaches_len < 8000)[0]

plt.figure(1, figsize=(11,8))
plt.title('Reach Type', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.scatter(reaches_x, reaches_y, s = 5, c = 'blue', edgecolors = None)
plt.scatter(reaches_x[lakes], reaches_y[lakes], s = 5, c = 'cyan', edgecolors = None)
plt.scatter(reaches_x[dams], reaches_y[dams], s = 5, c = 'gold', edgecolors = None)
plt.scatter(reaches_x[coast], reaches_y[coast], s = 5, c = 'magenta', edgecolors = None)
#plt.scatter(reaches.x[small], reaches.y[small], s = 5, c = 'gold', edgecolors = None)


vals = reaches_len/1000
vals2 = reaches_len[rivers]/1000
plt.figure(12, figsize=(11,8))   
plt.title('Reach Lengths', fontsize=16)
plt.xlabel('length (km)', fontsize=14)
plt.ylabel('count', fontsize=14)
plt.xlim(0,30)
plt.hist(vals, bins=100, color = 'orange')
plt.hist(vals2, bins = 100)


vals3 = rl_vals[np.where(rd_vals < 8000)[0]]
vals4 = rl_vals[rivers[np.where(rd_vals[rivers] < 8000)[0]]]
plt.figure(12, figsize=(11,8))   
plt.title('Short Reach Latitudes', fontsize=16)
plt.xlabel('latitude', fontsize=14)
plt.ylabel('count', fontsize=14)
plt.hist(vals3, bins=100, color = 'blueviolet')
plt.hist(vals4, bins = 100, color = 'limegreen')


unq_basins2 = np.unique(centerlines.reach_id[:,0])
number_of_colors = len(unq_basins2)
color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]

plt.figure(11, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('North America Reaches',  fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
for i in list(range(len(unq_basins2))):
    seg = np.where(centerlines.reach_id[:,0] == unq_basins2[i])
    plt.scatter(centerlines.x[seg], centerlines.y[seg], c=color[i], s = 5, edgecolors = 'None')


plt.figure(1, figsize=(11,8))   
plt.title('Reach WSE (m)', fontsize=16)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.scatter(reaches.x, reaches.y, c=reaches.slope, cmap = 'terrain', s=5, edgecolors = None)
plt.colorbar()

plt.figure(2, figsize=(11,8))   
plt.title('Reach Width (m)', fontsize=16)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.scatter(reaches.x, reaches.y, c=np.log(reaches.wth), cmap = 'YlGnBu', s=5, edgecolors = None)
plt.colorbar()

plt.figure(3, figsize=(11,8))   
plt.title('Reach Width Variation (m)', fontsize=16)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.scatter(reaches.x, reaches.y, c=np.log(reaches.wth_var), cmap ='YlGnBu', s=5, edgecolors = None)
plt.colorbar()


z = np.where(reaches.slope > 0.0)[0]
w = np.where(reaches.slope[z] <= 2)[0]

#vals = RCH_SLOPE[z]
vals = reaches.slope[z[w]]
plt.figure(4, figsize=(11,8))   
plt.title('Reach Slope', fontsize=16)
plt.xlabel('Slope (m/km)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.hist(vals, bins=100)

plt.figure(5, figsize=(11,8))   
plt.title('Reach Slopes', fontsize=16)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.scatter(reaches.x[z[w]], reaches.y[z[w]], c=reaches.slope[z[w]], cmap = 'plasma', s = 5, edgecolors = None)
plt.colorbar()

plt.figure(6, figsize=(11,8))   
plt.title('Reach Distance From Outlet', fontsize=16)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.scatter(reaches.x, reaches.y, c=reaches.dist_out, cmap = 'gnuplot', s = 5, edgecolors = None)
plt.colorbar()


z = np.where(N_RCH_UP == 9)[0]
plt.figure(7, figsize=(11,8))   
plt.title('Reach Distance From Outlet', fontsize=16)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.scatter(RCH_X, RCH_Y, c=RCH_DIST_OUT, cmap = 'gnuplot', s = 5, edgecolors = None)
plt.scatter(RCH_X[z], RCH_Y[z], c='red', s = 10, edgecolors = None)

plt.scatter(reaches.wth[rivers], reaches.slope[rivers])


'''
z = np.where(n_rch_down == 0)[0]
w = np.where(Reach_ID ==  7426410011)[0]
plt.figure(6, figsize=(11,8))   
plt.title('Reach Distance Out (km)', fontsize=16)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.scatter(rch_x, rch_y, c=rch_dist_out, s = 5, edgecolors = None)
plt.scatter(rch_x[z], rch_y[z], c='red', s = 5, edgecolors = None)
plt.scatter(rch_x[w], rch_y[w], c='pink', s = 10, edgecolors = None)
'''

#v = np.where(subcls.ghost == 1)[0]
plt.scatter(x, y, c='blue', s = 5, edgecolors='None')
#plt.scatter(x[v], y[v], c='red', s = 5, edgecolors='None')
plt.scatter(x[ghost_ids], y[ghost_ids], c='red', s = 5, edgecolors='None')
plt.scatter(x[rmv_ids2], y[rmv_ids2], c='cyan', s = 15, edgecolors='None')
plt.scatter(x[rmv_ids], y[rmv_ids], c='yellow', s = 15, edgecolors='None')


plt.scatter(subcls.x, subcls.y, c=np.log(subcls.rch_topo), s = 5, edgecolors='None')


b = np.where(subcls.basins == 742628)[0]
plt.figure(9, figsize=(11,8))   
plt.title('Reach Num', fontsize=16)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.scatter(subcls.x, subcls.y, c=np.log(subcls.rch_topo), s=5, edgecolors = None)

plt.figure(10, figsize=(11,8))   
plt.title('Reach Dist Out', fontsize=16)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.scatter(subreaches.x, subreaches.y, c=subreaches.dist_out, s=5, edgecolors = None)

plt.figure(11, figsize=(11,8))   
plt.title('Node Dist Out', fontsize=16)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.scatter(subnodes.x, subnodes.y, c=subnodes.dist_out, s=5, edgecolors = None)

z = np.where(subreaches.n_rch_down_filt == 0)[0]
v = np.where(subreaches.n_rch_down_filt == 2)[0]
w = np.where(subreaches.n_rch_down_filt == 3)[0]
q = np.where(subreaches.n_rch_up_filt == 0)[0]

plt.figure(12, figsize=(11,8))   
plt.title('Neighbors', fontsize=16)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.scatter(subreaches.x, subreaches.y, c='blue', s=5, edgecolors = None)
plt.scatter(subreaches.x[z], subreaches.y[z], c='red', s=15, edgecolors = None)
plt.scatter(subreaches.x[v], subreaches.y[v], c='gold', s=15, edgecolors = None)
plt.scatter(subreaches.x[w], subreaches.y[w], c='cyan', s=15, edgecolors = None)
plt.scatter(subreaches.x[q], subreaches.y[q], c='pink', s=15, edgecolors = None)










b = np.where(subcls.rch_eps2 == 1)[0]
plt.scatter(subcls.x, subcls.y, c = 'blue', s = 5, edgecolors= None)
plt.scatter(subcls.x[b], subcls.y[b], c = 'red', s = 5, edgecolors= None)
plt.show()

plt.scatter(subcls.x, subcls.y, c = np.log(subcls.rch_ind6), s = 5, edgecolors= None)
plt.scatter(subcls.x, subcls.y, c = np.log(subcls.rch_dist6), s = 5, edgecolors= None)
plt.scatter(subcls.x, subcls.y, c = np.log(subcls.facc), s = 5, edgecolors= None)


z = np.where(subcls.type6 == 3)[0]
plt.scatter(subcls.x, subcls.y, c = 'blue', edgecolors = None)
plt.scatter(subcls.x[z], subcls.y[z], c = 'cyan', edgecolors = None)



### sub-basin reaches and endpoints

reaches = subcls.rch_id2
x = subcls.x
y = subcls.y
eps = np.where(subcls.rch_eps2 == 1)


unq_rchs = np.unique(reaches)
number_of_colors = len(unq_rchs)
color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]

plt.figure(11, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('North America Reaches',  fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
for i in list(range(len(unq_rchs))):
    seg = np.where(reaches == unq_rchs[i])
    plt.scatter(x[seg], y[seg], c=color[i], s = 5, edgecolors = 'None')
plt.scatter(x[eps], y[eps], c='black', s = 10, edgecolors = 'None')