# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 09:08:37 2020

@author: ealtenau
"""
from __future__ import division
import os 
os.chdir('C:/Users/ealtenau/Documents/Research/SWAG/For_Server/scripts/reach_definition/')
import Reach_Definition_Tools_v08 as rdt
import time
import numpy as np

###############################################################################
def swot_obs_percentage(subcls):
    
    """
    FUNCTION:
        Calculating the SWOT coverage for each overpass along a reach.

    INPUTS
        subcls -- Object containing attributes for the high-resolution centerline.
            [attributes used]:
                reach_id -- Reach IDs for along the high-resolution centerline.
                orbits -- SWOT orbit locations along the high-resolution 
                    centerline. 

    OUTPUTS
        swot_coverage -- 2-D array of the minimum and maximum swot coverage 
            per reach (%). Dimension is [number of reaches, 2] with the first 
            column representing the minimum swot coverage for a particular 
            reach and the second column representing the maximum swot coverage 
            for a particular reach.
    """
    
    # Set variables. 
    uniq_rch = np.unique(subcls.rch_id5)
    swot_coverage = np.zeros((len(uniq_rch), 3))
    
    # Loop through each reach and calculate the coverage for each swot overpass.
    for ind in list(range(len(uniq_rch))):
        rch = np.where(subcls.rch_id5 == uniq_rch[ind])[0]
        orbs = subcls.orbits[rch]
        Type = np.max(subcls.type5[rch])
        uniq_orbits = np.unique(orbs)
        uniq_orbits = uniq_orbits[np.where(uniq_orbits>0)[0]]
        
        if len(uniq_orbits) == 0:
            continue
        
        percent = []
        for idz in list(range(len(uniq_orbits))):
            rows = np.where(orbs == uniq_orbits[idz])[0]
            percent.append((len(rows)/len(rch))*100)
        
        percent = np.array(percent)
        swot_coverage[ind, 0] = np.min(np.round(percent, 2))
        swot_coverage[ind, 1] = np.max(np.round(percent, 2))
        swot_coverage[ind, 2] = Type

    return swot_coverage

###############################################################################
##################### Defining Reach and Node Locations #######################
###############################################################################

start_all = time.time()

region = 'OC'

# Input file.
nc_file = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/outputs/Merged_Data/'+region+'/'+region+'_Merge_v04.nc'

# Reading in data.
data = rdt.read_merge_netcdf(nc_file)

# Correcting mississippi basin type problem.
#data.lake[np.where((data.basins == 742540) & (data.lake == 3))[0]] = 0

# Making sure flow accumulation minimum isn't zero.             
data.facc[np.where(data.facc == 0)[0]] = 0.001

# Creating empty objects to fill with attributes. 
centerlines = rdt.Object()
reaches = rdt.Object()
nodes = rdt.Object()

# Loop through each level 2 basin. 
level2_basins = np.array([np.int(np.str(ind)[0:2]) for ind in data.basins])
uniq_level2 = np.unique(level2_basins)
uniq_level2 = np.delete(uniq_level2, 0)
cnt = 0
start_id = 0 
for ind in list(range(len(uniq_level2))):

    print('STARTING BASIN: ' + str(uniq_level2[ind]))
    
    start = time.time()
    
    # Define objects to assign attributes.
    subcls = rdt.Object()
    subreaches = rdt.Object()
    subnodes = rdt.Object()
    
    # Subset data.
    level2 = np.where(level2_basins == uniq_level2[ind])[0]
    subcls.id = data.id[level2]
    subcls.lon = data.lon[level2]
    subcls.lat = data.lat[level2]
    subcls.seg = data.seg[level2]
    subcls.ind = data.ind[level2]
    subcls.dist = data.segDist[level2] 
    subcls.wth = data.wth[level2]
    subcls.elv = data.elv[level2]
    subcls.facc = data.facc[level2]
    subcls.lake = data.lake[level2]
    subcls.delta = data.delta[level2]
    subcls.nchan = data.nchan[level2]
    subcls.grand = data.grand[level2]
    subcls.grod = data.grod[level2]
    subcls.basins = data.basins[level2]
    subcls.manual = data.manual[level2]
    subcls.num_obs = data.num_obs[level2]
    subcls.orbits = data.orbits[level2]
    subcls.lakeid = data.lake_id[level2]
    subcls.lon[np.where(subcls.lon < -180)] = -180.0
    subcls.lon[np.where(subcls.lon > 180)] = 180.0
    subcls.x = np.copy(subcls.lon)
    subcls.y = np.copy(subcls.lat)
    
    #########################################################################
    #REACH DEFINITION
           
    print('Defining Intial Reaches')
    # Find all reach boundaries. 
    dam_radius = 225
    subcls.lake_bnds, subcls.lake_dist, subcls.lake_id,\
     subcls.dam_bnds, subcls.dam_dist, subcls.dam_id, subcls.swot_bnds,\
     subcls.swot_dist, subcls.swot_id, subcls.all_bnds, \
     subcls.type0 = rdt.find_all_bounds(subcls, dam_radius)
    
    # Define inital reaches.
    subcls.rch_id0, subcls.rch_len0 = rdt.number_reaches(subcls)
    
    print('Cutting Long Reaches')
    subcls.rch_id1, subcls.rch_len1 = rdt.cut_reaches(subcls.rch_id0, 
                                                      subcls.rch_len0, subcls.dist, 
                                                      subcls.ind, 16000)     
    
    # Make sure there is only one "type" value per reach. 
    subcls.type1 = np.zeros(len(subcls.rch_id1))
    uniq_rch = np.unique(subcls.rch_id1)
    for idx in list(range(len(uniq_rch))):
        rch = np.where(subcls.rch_id1 == uniq_rch[idx])[0]
        if len(rch) <=2:
            subcls.type1[rch] = np.min(subcls.type0[rch])
        else:
            subcls.type1[rch] = max(set(list(subcls.type0[rch])), 
                                    key=list(subcls.type0[rch]).count)
    
    print('Aggregating River Reaches') 
    # Aggregating river reach types.
    river_min_dist = 8000
    subcls.rch_id2, subcls.rch_len2,\
     subcls.type2 = rdt.aggregate_rivers(subcls, river_min_dist)
    # Updating reach indexes.
    subcls.rch_ind2, __ = rdt.update_rch_indexes(subcls, subcls.rch_id2) 
    
    print('Aggregating Lake Reaches')
    # Aggregating lake reach types.
    lake_min_dist = 8000
    subcls.rch_id3, subcls.rch_len3,\
     subcls.type3 = rdt.aggregate_lakes(subcls, lake_min_dist)
    # Updating reach indexes.   
    subcls.rch_ind3, __ = rdt.update_rch_indexes(subcls, subcls.rch_id3) 
    
    print('Aggregating Dam Reaches')
    # Aggregating dam reach types.
    dam_min_dist = 200
    subcls.rch_id4, subcls.rch_len4,\
     subcls.type4 = rdt.aggregate_dams(subcls, dam_min_dist) 
    # Updating reache indexes.
    subcls.rch_ind4, __ = rdt.update_rch_indexes(subcls, subcls.rch_id4) 
    # Updating reach flow distance.
    subcls.rch_dist4 = rdt.calc_segDist(subcls.lon, subcls.lat, subcls.rch_id4, 
                                        subcls.facc, subcls.rch_ind4)
    
    print('Second Cutting of Long Reaches')
    subcls.rch_id5, subcls.rch_len5 = rdt.cut_reaches(subcls.rch_id4, 
                                                      subcls.rch_len4, subcls.rch_dist4, 
                                                      subcls.rch_ind4, 20000)
    subcls.type5 = np.copy(subcls.type4)
    # Updating reach indexes.
    subcls.rch_ind5, \
     subcls.rch_eps5 = rdt.update_rch_indexes(subcls, subcls.rch_id5) 
    # Updating reach flow distance.
    subcls.rch_dist5 = rdt.calc_segDist(subcls.lon, subcls.lat, subcls.rch_id5, 
                                        subcls.facc, subcls.rch_ind5)
    
    #########################################################################
    
    print('Calculating SWOT Coverage')
    # Calculating swot coverage. 
    subreaches.swot_perc = swot_obs_percentage(subcls)       
   
    if cnt == 0:
        reaches.swot_perc = np.copy(subreaches.swot_perc)
    else:
        reaches.swot_perc = np.insert(reaches.swot_perc, len(reaches.swot_perc), np.copy(subreaches.swot_perc), axis = 0)

    print('Aggregating SWOT Coverage')
    Type = subreaches.swot_perc[:,2]
    rivers = np.where(Type == 1)[0]

    if region == 'NA' and ind == 0:
        min_vals_all = subreaches.swot_perc[:,0]
        max_vals_all = subreaches.swot_perc[:,1]
        min_vals_rivers_all = subreaches.swot_perc[rivers,0]
        max_vals_rivers_all = subreaches.swot_perc[rivers,1]
    
    else:
        min_vals_all = np.insert(min_vals_all, len(min_vals_all), np.copy(subreaches.swot_perc[:,0]))
        max_vals_all = np.insert(max_vals_all, len(max_vals_all), np.copy(subreaches.swot_perc[:,1]))
        min_vals_rivers_all = np.insert(min_vals_rivers_all, len(min_vals_rivers_all), np.copy(subreaches.swot_perc[rivers,0]))
        max_vals_rivers_all = np.insert(max_vals_rivers_all, len(max_vals_rivers_all), np.copy(subreaches.swot_perc[rivers,1]))

    cnt = cnt+1
    
    
###############################################################################    
###############################################################################    
###############################################################################
    
(len(np.where(min_vals_all == 0)[0])/len(min_vals_all))*100 #  
(len(np.where((min_vals_all > 0) & (min_vals_all <= 25))[0])/len(min_vals_all))*100 # 
(len(np.where((min_vals_all > 25) & (min_vals_all <= 50))[0])/len(min_vals_all))*100 # 
(len(np.where((min_vals_all > 50) & (min_vals_all <= 75))[0])/len(min_vals_all))*100 # 
(len(np.where(min_vals_all > 75)[0])/len(min_vals_all))*100 # 

(len(np.where(max_vals_all == 0)[0])/len(max_vals_all))*100 #  
(len(np.where((max_vals_all > 0) & (max_vals_all <= 25))[0])/len(max_vals_all))*100 # 
(len(np.where((max_vals_all > 25) & (max_vals_all <= 50))[0])/len(max_vals_all))*100 # 
(len(np.where((max_vals_all > 50) & (max_vals_all <= 75))[0])/len(max_vals_all))*100 # 
(len(np.where(max_vals_all > 75)[0])/len(max_vals_all))*100 # 


(len(np.where(min_vals_rivers_all == 0)[0])/len(min_vals_rivers_all))*100 #  
(len(np.where((min_vals_rivers_all > 0) & (min_vals_rivers_all <= 25))[0])/len(min_vals_rivers_all))*100 # 
(len(np.where((min_vals_rivers_all > 25) & (min_vals_rivers_all <= 50))[0])/len(min_vals_rivers_all))*100 # 
(len(np.where((min_vals_rivers_all > 50) & (min_vals_rivers_all <= 75))[0])/len(min_vals_rivers_all))*100 # 
(len(np.where(min_vals_rivers_all > 75)[0])/len(min_vals_rivers_all))*100 # 

(len(np.where(max_vals_rivers_all == 0)[0])/len(max_vals_rivers_all))*100 #  
(len(np.where((max_vals_rivers_all > 0) & (max_vals_rivers_all <= 25))[0])/len(max_vals_rivers_all))*100 # 
(len(np.where((max_vals_rivers_all > 25) & (max_vals_rivers_all <= 50))[0])/len(max_vals_rivers_all))*100 # 
(len(np.where((max_vals_rivers_all > 50) & (max_vals_rivers_all <= 75))[0])/len(max_vals_rivers_all))*100 # 
(len(np.where(max_vals_rivers_all > 75)[0])/len(max_vals_rivers_all))*100 # 

(len(np.where(min_vals_all >= 50)[0])/len(min_vals_all))*100 # 
(len(np.where(min_vals_all >= 75)[0])/len(min_vals_all))*100 # 

(len(np.where(max_vals_all >= 50)[0])/len(max_vals_all))*100 
(len(np.where(max_vals_all >= 75)[0])/len(max_vals_all))*100 
