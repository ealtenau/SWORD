# -*- coding: utf-8 -*-
"""
Created on Thu Jun 03 14:41:11 2021

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
    data.id = new.groups['reaches'].variables['reach_id'][:]
    data.swot_obs = new.groups['reaches'].variables['swot_obs'][:]
    new.close()

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
        reaches.id = np.copy(subreaches.id)
        reaches.swot_obs = np.copy(subreaches.swot_obs)
        
    # Otherwise, append the sub-attributes.
    else:
        reaches.id = np.insert(reaches.id, len(reaches.id), np.copy(subreaches.id))
        reaches.swot_obs = np.insert(reaches.swot_obs, len(reaches.swot_obs), np.copy(subreaches.swot_obs))
        
###############################################################################
###############################################################################
###############################################################################
        
sword_rivers = np.sort(glob.glob('E:/Users/Elizabeth Humphries/Documents/SWORD/For_Server/outputs/SWOT_Coverage/Annual_IceFreeObs/netcdf/*v10*'))
sword = Object()
cnt = 0
for ind in list(range(len(sword_rivers))):
    fn = sword_rivers[ind]
    data = read_netcdf(fn)
    append_data(sword, data, cnt)
    cnt = cnt+1

    
# getting data of the histogram
count, bins_count = np.histogram(sword.swot_obs, bins=100)
  
# finding the PDF of the histogram using count values
pdf = count / sum(count)
  
# using numpy np.cumsum to calculate the CDF
# We can also find using the PDF values by looping and adding
cdf = np.cumsum(pdf)
  
# plotting PDF and CDF
plt.plot(bins_count[1:], pdf, color="darkorange", linewidth=1)
plt.plot(bins_count[1:], cdf, color = "indigo", linewidth=1)
plt.axvline(17, 0, 1.5, color = "grey", linestyle = "--", linewidth=3)
plt.axvline(35, 0, 1.5, color = "grey", linestyle = "--", linewidth=3)
plt.axvline(52, 0, 1.5, color = "grey", linestyle = "--", linewidth=3)
plt.plot(bins_count[1:], pdf, color="darkorange", linewidth=5, label="PDF")
plt.plot(bins_count[1:], cdf, color = "indigo", linewidth=5, label="CDF")
plt.legend()
plt.xlabel('SWOT Passes')
plt.ylabel('Probability') 
plt.title('Annual Number of Ice-Free SWOT Passes')