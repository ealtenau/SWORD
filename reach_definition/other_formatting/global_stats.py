 # -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:54:41 2020

@author: ealtenau
"""
from __future__ import division
import os
import utm
import numpy as np
from scipy import spatial as sp
import netCDF4 as nc
from pyproj import Proj
import glob
import matplotlib.pyplot as plt
import random

################################################################################

class Object(object):
    """
    FUNCTION:
        Creates class object to assign attributes to.
    """
    pass

###############################################################################

def read_netcdf(filename1):

    """
    FUNCTION:
        Reads in attributes from the merged database and assigns them to an
        object.

    INPUTS
        filename -- Merged database netcdf file.

    OUTPUTS
        data -- Object containing attributes from the merged database.
    """

    data = Object()
    new = nc.Dataset(filename1)
    # new2 = nc.Dataset(filename2)
    data.x = new.groups['reaches'].variables['x'][:]
    data.y = new.groups['reaches'].variables['y'][:]
    data.id = new.groups['reaches'].variables['reach_id'][:]
    data.node_dist = new.groups['nodes'].variables['dist_out'][:]
    data.node_rch_id = new.groups['nodes'].variables['reach_id'][:]
    data.node_wse = new.groups['nodes'].variables['wse'][:]
    data.len = new.groups['reaches'].variables['reach_length'][:]
    data.slope = new.groups['reaches'].variables['slope'][:]
    data.wse = new.groups['reaches'].variables['wse'][:]
    data.wth = new.groups['reaches'].variables['width'][:]
    data.nchan_mod = new.groups['reaches'].variables['n_chan_mod'][:]
    data.swot_obs = new.groups['reaches'].variables['swot_obs'][:]
    # data.swot_perc = new2.groups['reaches'].variables['perc_coverage'][:].T
    new.close()
    # new2.close()

    return data

###############################################################################

def append_data(reaches, subreaches, cnt):

    """
    FUNCTION:
        Appends sub-attributes within a loop to an object containing the final
        SWORD reach attributes for an entire specified region
        (in most cases a continent).

    INPUTS
        reaches -- Object to be appended with sub-attribute data.
        subreaches -- Object containing current attribute information for a
            single level 2 basin at the reach loctions.
        cnt -- Specifies the current loop iteration.
    """

    # Copy the very first sub-attributes.
    if cnt == 0:
        reaches.x = np.copy(subreaches.x)
        reaches.y = np.copy(subreaches.y)
        reaches.id = np.copy(subreaches.id)
        reaches.len = np.copy(subreaches.len)
        reaches.wse = np.copy(subreaches.wse)
        reaches.wth = np.copy(subreaches.wth)
        reaches.slope = np.copy(subreaches.slope)
        reaches.nchan_mod = np.copy(subreaches.nchan_mod)
        reaches.swot_obs = np.copy(subreaches.swot_obs)
        # reaches.swot_perc = np.copy(subreaches.swot_perc)
        reaches.node_dist = np.copy(subreaches.node_dist)
        reaches.node_rch_id = np.copy(subreaches.node_rch_id)
        reaches.node_wse = np.copy(subreaches.node_wse)
    # Otherwise, append the sub-attributes.
    else:
        reaches.id = np.insert(reaches.id, len(reaches.id), np.copy(subreaches.id))
        reaches.x = np.insert(reaches.x, len(reaches.x), np.copy(subreaches.x))
        reaches.y = np.insert(reaches.y, len(reaches.y), np.copy(subreaches.y))
        reaches.len = np.insert(reaches.len, len(reaches.len), np.copy(subreaches.len))
        reaches.wse = np.insert(reaches.wse, len(reaches.wse), np.copy(subreaches.wse))
        reaches.wth = np.insert(reaches.wth, len(reaches.wth), np.copy(subreaches.wth))
        reaches.slope = np.insert(reaches.slope, len(reaches.slope), np.copy(subreaches.slope))
        reaches.nchan_mod = np.insert(reaches.nchan_mod, len(reaches.nchan_mod), np.copy(subreaches.nchan_mod))
        reaches.swot_obs = np.insert(reaches.swot_obs, len(reaches.swot_obs), np.copy(subreaches.swot_obs))
        # reaches.swot_perc = np.insert(reaches.swot_perc, len(reaches.swot_perc), np.copy(subreaches.swot_perc), axis = 0)
        reaches.node_dist = np.insert(reaches.node_dist, len(reaches.node_dist), np.copy(subreaches.node_dist))
        reaches.node_rch_id = np.insert(reaches.node_rch_id, len(reaches.node_rch_id), np.copy(subreaches.node_rch_id))
        reaches.node_wse = np.insert(reaches.node_wse, len(reaches.node_wse), np.copy(subreaches.node_wse))

###############################################################################
#sword_rivers = np.sort(glob.glob('E:/Users/Elizabeth Humphries/Documents/SWORD/For_Server/outputs/Reaches_Nodes_v10/netcdf/*v10*'))
sword_rivers = np.sort(glob.glob('/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/*v17*'))
# sword_obs = np.sort(glob.glob('/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/SWOT_Coverage_Ice/v14/netcdf/*v14*'))
sword = Object()
cnt = 0
for ind in list(range(len(sword_rivers))):
    fn1 = sword_rivers[ind]
    # fn2 = sword_obs[ind]
    data = read_netcdf(fn1)
    append_data(sword, data, cnt)
    cnt = cnt+1


sword.type = np.zeros(len(sword.id))
for ind in list(range(len(sword.id))):
    sword.type[ind] = int(str(sword.id[ind])[10:11])
All = np.where(sword.type != 6)[0]

ghost = np.where(sword.type == 6)[0]
dams = np.where(sword.type == 4)[0]
lakes = np.where(sword.type == 3)[0]
coast = np.where(sword.type == 5)[0]
rivers = np.where(sword.type == 1)[0]

(len(rivers)/len(All))*100 #72.8 (v10), 73.0 (v11)
(len(dams)/len(All))*100 #10.5 (v10), 10.5 (v11)
(len(lakes)/len(All))*100 #10.1 (v10), 10.1 (v11)
(len(coast)/len(All))*100 #6.6 (v10), 6.4 (v11)

#type pie chart
p = np.array([(len(rivers)/len(All))*100, (len(dams)/len(All))*100 , 
              (len(lakes)/len(All))*100, (len(coast)/len(All))*100])
mylabels = ["Rivers", "Dams", "Lakes on Rivers", "Unreliable Topology"]
mycolors = ["royalblue", "crimson", "deepskyblue", "gold"]
plt.pie(p,labels = mylabels, colors = mycolors)
plt.show() 

## reach length percentages
np.sum(sword.len[dams])/np.sum(sword.len[All]) #0.5%
np.sum(sword.len[rivers])/np.sum(sword.len[All]) #86%

'''
# reach lengths globally.
(len(np.where(reaches_len[All] == 0)[0])/len(reaches_len[All]))*100 # 0%
(len(np.where(reaches_len[All] <= 8000)[0])/len(reaches_len[All]))*100 # 16.6%
(len(np.where((reaches_len[All] > 8000) & (reaches_len[All] <= 10000))[0])/len(reaches_len[All]))*100 # 34.2%
(len(np.where((reaches_len[All] > 10000) & (reaches_len[All] <= 15000))[0])/len(reaches_len[All]))*100 # 40.9%
(len(np.where((reaches_len[All] > 15000) & (reaches_len[All] <= 20000))[0])/len(reaches_len[All]))*100 # 7.2%
(len(np.where(reaches_len[All] < 20000)[0])/len(reaches_len[All]))*100 # 98.9%
(len(np.where(reaches_len[All] > 20000)[0])/len(reaches_len[All]))*100 # 1.1%
# reach lengths globally - rivers only.
(len(np.where(reaches_len[rivers] == 0)[0])/len(reaches_len[rivers]))*100 # 0%
(len(np.where(reaches_len[rivers] <= 8000)[0])/len(reaches_len[rivers]))*100 # 7.7%
(len(np.where((reaches_len[rivers] > 8000) & (reaches_len[rivers] <= 10000))[0])/len(reaches_len[rivers]))*100 # 37.7%
(len(np.where((reaches_len[rivers] > 10000) & (reaches_len[rivers] <= 15000))[0])/len(reaches_len[rivers]))*100 # 46.1%
(len(np.where((reaches_len[rivers] > 15000) & (reaches_len[rivers] <= 20000))[0])/len(reaches_len[rivers]))*100 # 7.8%
(len(np.where(reaches_len[rivers] < 20000)[0])/len(reaches_len[rivers]))*100 # 99.3%
(len(np.where(reaches_len[rivers] > 20000)[0])/len(reaches_len[rivers]))*100 # 0.7%
'''

# average reach lengths globally.
print(np.mean(sword.len[All])) #9.9 km 
print(np.median(sword.len[All])) #10.4 km
print(np.mean(sword.len[rivers])) #11.6 km 
print(np.median(sword.len[rivers])) #10.9 km 

np.max(sword.swot_obs[All]) #31
np.median(sword.swot_obs[All]) #2

(len(np.where(sword.len[All] < 5000)[0])/len(sword.len[All]))*100 # 23.0% 
(len(np.where(sword.len[rivers] < 5000)[0])/len(sword.len[rivers]))*100 # 10.9% 

# (len(np.where(sword.len[All] < 10000)[0])/len(sword.len[All]))*100 # 37.2% 
# (len(np.where(sword.len[rivers] < 10000)[0])/len(sword.len[rivers]))*100 # 23.9% 


(len(np.where((sword.len[All] >= 5000) & (sword.len[All] < 10000))[0])/len(sword.len[All]))*100 # 14.2%
(len(np.where((sword.len[rivers] >= 5000) & (sword.len[rivers] < 10000))[0])/len(sword.len[rivers]))*100 # 13.0%

(len(np.where((sword.len[All] >= 10000) & (sword.len[All] <= 20000))[0])/len(sword.len[All]))*100 # 62.8%
(len(np.where((sword.len[rivers] >= 10000) & (sword.len[rivers] <= 20000))[0])/len(sword.len[rivers]))*100 # 76.1%

(len(np.where(sword.len[All] > 20000)[0])/len(sword.len[All]))*100 # 1.0% (10 km)
(len(np.where(sword.len[rivers] > 20000)[0])/len(sword.len[rivers]))*100 # 0.9% (10 km)

# looking at the type percentage break down of the short reaches.
short_reaches = All[np.where(sword.len[All] < 10000)[0]]
len(np.where(sword.type[short_reaches] == 1)[0])/len(short_reaches) # 44% # 28% (< 5km)
len(np.where(sword.type[short_reaches] == 3)[0])/len(short_reaches) # 13% # 11% (< 5km)
len(np.where(sword.type[short_reaches] == 4)[0])/len(short_reaches) # 34% # 57% (< 5km)
len(np.where(sword.type[short_reaches] == 5)[0])/len(short_reaches) # 9% # 5% (< 5km)


### slopes
(len(np.where((sword.slope[All] > 0.01) & (sword.slope[All] <= 5))[0])/len(sword.slope[All]))*100 #69% of slopes are between 0.01 m/km and 5 m/km
dam_subset = np.where(sword.slope[dams] > 0.01)[0]
river_subset = np.where(sword.slope[rivers] > 0.01)[0]

# for reaches with enough data for slopes:
print(np.median(sword.slope[dams[dam_subset]])) # 3.6 m/km
print(np.mean(sword.slope[dams[dam_subset]])) # 12.0 m/km
print(np.median(sword.slope[rivers[river_subset]])) # 0.61 m/km
print(np.mean(sword.slope[rivers[river_subset]])) # 2.0 m/km

good_slopes = np.where(sword.slope[All] > 0.000000001)[0]
print((len(np.where(sword.slope[All[good_slopes]] < 1)[0])/len(sword.slope[All[good_slopes]]))*100) #61%
print((len(np.where(sword.slope[All[good_slopes]] < 3)[0])/len(sword.slope[All[good_slopes]]))*100) #82%
print((len(np.where(sword.slope[All[good_slopes]] < 5)[0])/len(sword.slope[All[good_slopes]]))*100) #89%

np.min(sword.slope[All[good_slopes]])
np.median(sword.slope[All[good_slopes]]) #0.61 m/km
np.mean(sword.slope[All[good_slopes]]) #2.66 m/km
(len(good_slopes)/len(All))*100 #78% slopes > 1 cm/km

good_river_slopes = np.where(sword.slope[rivers] > 0.01)[0]
np.median(sword.slope[rivers[good_river_slopes]]) #0.61 m/km
np.mean(sword.slope[rivers[good_river_slopes]]) #0.61 m/km

(len(good_river_slopes)/len(rivers))*100 #85%
(len(good_slopes)/len(All))*100 #78%

#30.9 cm/km median slope for all reaches.


'''
# excluding reaches where length < 90 m and slopes > 3 m/km.
All_wth = All[np.where(sword.wth[All] > 90)]
All_wth_slope = All_wth[np.where(sword.slope[All_wth] <= 3)]
All_wth_slope2 = All_wth_slope[np.where(sword.slope[All_wth_slope] >= 0.01)]
remv = All[np.where((sword.type[All] < 5) & (sword.slope[All] < 0.01))]
All2 = np.delete(All, remv)
'''
All2 = All[np.where(sword.type[All] != 4)[0]]

np.quantile(sword.slope[All],0.25) # 2.7 cm/km
np.quantile(sword.slope[All],0.5) # 31.6 cm/km
np.quantile(sword.slope[All],0.75) # 1.5 m/km
                        
len(np.where(sword.slope[All] < 5)[0])/len(All) #91%

### SWOT observations for 10 km reaches.
ten_km = np.where(sword.len[All] >= 10000)[0]
ten_km_rivers = np.where(sword.len[rivers] >= 10000)[0]

fifty_perc = np.zeros(len(sword.id))
for idy in list(range(len(sword.id))):
    vals = np.where(sword.swot_perc[idy,:] >= 50)[0]
    if len(vals) > 0:
        fifty_perc[idy] = 1

perc_75 = np.zeros(len(sword.id))
for idz in list(range(len(sword.id))):
    vals = np.where(sword.swot_perc[idz,:] >= 75)[0]
    if len(vals) > 0:
        perc_75[idz] = 1

print((len(np.where(fifty_perc[All] == 1)[0])/len(sword.id[All]))*100) # 95%
print((len(np.where(perc_75[All] == 1)[0])/len(sword.id[All]))*100) # 94%

print((len(np.where(fifty_perc[All[ten_km]] == 1)[0])/len(sword.id[All[ten_km]]))*100) # 95%
print((len(np.where(perc_75[All[ten_km]] == 1)[0])/len(sword.id[All[ten_km]]))*100) # 94%

print((len(np.where(fifty_perc[rivers[ten_km_rivers]] == 1)[0])/len(sword.id[rivers[ten_km_rivers]]))*100) # 95%
print((len(np.where(perc_75[rivers[ten_km_rivers]] == 1)[0])/len(sword.id[rivers[ten_km_rivers]]))*100) # 94%




plt.figure(1, figsize=(11,8))
plt.title('Global Slopes', fontsize=16)
plt.xlabel('Slope', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.hist(sword.slope[All], bins = 150, color = 'indigo')
plt.xlim(-5,5)
#plt.show()

plt.figure(2, figsize=(11,8))
plt.title('Number of Channels', fontsize=16)
plt.xlabel('Number of Channels', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.hist(sword.nchan_mod[All],bins = 25, color = 'royalblue')
plt.xlim(0,5)
#plt.show()

good_wse = np.where(sword.wse[All] > 0.000000001)[0]
plt.figure(3, figsize=(11,8))
plt.title('Elevation', fontsize=16)
plt.xlabel('Elevation', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.hist(sword.wse[All],bins = 150, color = 'mediumvioletred')
#plt.xlim(-10,2000)
#plt.show()
#np.median(sword.wse[All_wth])

z = np.where(sword.wth[All] > 1)[0]
plt.figure(4, figsize=(11,8))
plt.title('Width', fontsize=16)
plt.xlabel('Width', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.hist(sword.wth[All],bins = 50, color = 'mediumseagreen') #alternative color - mediumseagreen
#plt.xlim(0,2000)
#plt.show()
#np.median(sword.wth[All_wth])


plt.figure(5, figsize=(11,8))
plt.title('Reach Length', fontsize=16)
plt.xlabel('Reach Length', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.hist(sword.len[All],bins = 250, color = 'royalblue')
plt.xlim(0,20000)
#plt.show()


plt.figure(6, figsize=(11,8))
plt.title('nchan vs wth', fontsize=16)
plt.xlabel('width', fontsize=14)
plt.ylabel('nchan', fontsize=14)
plt.scatter(sword.wth[All], sword.nchan_mod[All], c = 'royalblue', s = 5, edgecolors=None)

#### log scale

bins1 = np.logspace(np.log10(1E-3),np.log10(300000), 75)
plt.figure(1, figsize=(11,8))
plt.title('Slope', fontsize=16)
plt.xlabel('log(slope)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.hist(sword.slope[All], bins=bins1, label='ERate', color = 'indigo')
plt.gca().set_xscale("log")

bins2 = np.logspace(np.log10(1E-3),np.log10(300000), 75)
plt.figure(2, figsize=(11,8))
plt.title('Elevation', fontsize=16)
plt.xlabel('log(elevation)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.hist(sword.wse[All], bins=bins2, label='ERate', color = 'mediumvioletred')
plt.gca().set_xscale("log")
plt.xlim(-0.5,100000)


z = np.where(sword.wth[All] > 1)[0]
bins3 = np.logspace(np.log10(1E-3),np.log10(300000), 75)
plt.figure(3, figsize=(11,8))
plt.title('Width', fontsize=16)
plt.xlabel('log(width)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.hist(sword.wth[All[z]], bins=bins3, label='ERate', color = 'mediumseagreen')
plt.gca().set_xscale("log")
plt.xlim(10,100000)


###############################################################################
###############################################################################
###############################################################################

lta3 = np.where(sword.slope[All] < 3)[0]
lt3 = np.where(sword.slope[All[good_slopes]] < 3)[0]
ltar3 = np.where(sword.slope[rivers] < 3)[0]
ltr3 = np.where(sword.slope[rivers[good_river_slopes]] < 3)[0]

gt5 = np.where(sword.len[All[good_slopes]] >= 5000)[0]
gtr5 = np.where(sword.len[rivers[good_river_slopes]] >= 5000)[0]
gta5 = np.where(sword.len[All] >= 5000)[0]
gtar5 = np.where(sword.len[rivers] >= 5000)[0]

### Including Zeros
np.quantile(sword.slope[All], 0.25)
np.quantile(sword.slope[All], 0.5)
np.quantile(sword.slope[All], 0.75)

np.quantile(sword.slope[All][gta5], 0.25)
np.quantile(sword.slope[All][gta5], 0.5)
np.quantile(sword.slope[All][gta5], 0.75)

#only slopes < 3m/km
np.quantile(sword.slope[All][lta3], 0.25)
np.quantile(sword.slope[All][lta3], 0.5)
np.quantile(sword.slope[All][lta3], 0.75)

np.quantile(sword.slope[rivers], 0.25)
np.quantile(sword.slope[rivers], 0.5)
np.quantile(sword.slope[rivers], 0.75)

np.quantile(sword.slope[rivers][gtar5], 0.25)
np.quantile(sword.slope[rivers][gtar5], 0.5)
np.quantile(sword.slope[rivers][gtar5], 0.75)

np.quantile(sword.slope[rivers][ltar3], 0.25)
np.quantile(sword.slope[rivers][ltar3], 0.5)
np.quantile(sword.slope[rivers][ltar3], 0.75)


#Excluding Zeros
np.quantile(sword.slope[All[good_slopes]], 0.25)
np.quantile(sword.slope[All[good_slopes]], 0.5)
np.quantile(sword.slope[All[good_slopes]], 0.75)
    
np.quantile(sword.slope[All[good_slopes]][gt5], 0.25)
np.quantile(sword.slope[All[good_slopes]][gt5], 0.5)
np.quantile(sword.slope[All[good_slopes]][gt5], 0.75)

#only slopes < 3m/km
np.quantile(sword.slope[All[good_slopes]][lt3], 0.25)
np.quantile(sword.slope[All[good_slopes]][lt3], 0.5)
np.quantile(sword.slope[All[good_slopes]][lt3], 0.75)
     
np.quantile(sword.slope[rivers[good_river_slopes]], 0.25)
np.quantile(sword.slope[rivers[good_river_slopes]], 0.5)
np.quantile(sword.slope[rivers[good_river_slopes]], 0.75)
        
np.quantile(sword.slope[rivers[good_river_slopes]][gtr5], 0.25)
np.quantile(sword.slope[rivers[good_river_slopes]][gtr5], 0.5)
np.quantile(sword.slope[rivers[good_river_slopes]][gtr5], 0.75)
    
np.quantile(sword.slope[rivers[good_river_slopes]][ltr3], 0.25)
np.quantile(sword.slope[rivers[good_river_slopes]][ltr3], 0.5)
np.quantile(sword.slope[rivers[good_river_slopes]][ltr3], 0.75)



indexes = random.sample(range(10, 150000), 12)
for ind in list(range(len(indexes))):
    rch=indexes[ind]
    outpath = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/outputs/test_reaches/slope_check_v09/rch_'+str(sword.id[rivers[good_river_slopes]][gtr5][rch])+'.jpg'
    print(sword.slope[rivers[good_river_slopes]][gtr5][rch])
    nodes = np.where(sword.node_rch_id == sword.id[rivers[good_river_slopes]][gtr5][rch])[0]
    plt.scatter(sword.node_dist[nodes]/1000, sword.node_wse[nodes], c = 'blue', s=5, edgecolors=None)
    plt.title('Reach ID = ' + str(sword.id[rivers[good_river_slopes]][gtr5][rch]) + ', Slope = ' + str(np.round(sword.slope[rivers[good_river_slopes]][gtr5][rch],3)))
    plt.xlabel('Node Dist (km)')
    plt.ylabel('WSE (m)')
    plt.savefig(outpath)
    plt.close()

    
plt.scatter(sword.x, sword.y, c='blue', s=5, edgecolors = None)
plt.scatter(sword.x[remv], sword.y[remv], c='red', s=5, edgecolors = None)
