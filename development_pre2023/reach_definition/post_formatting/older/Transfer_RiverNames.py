# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 09:14:17 2021

@author: ealtenau
"""

import netCDF4 as nc
import numpy as np
from scipy import spatial as sp
import os
from osgeo import ogr
import time

###############################################################################

class Object(object):
    """
    FUNCTION:
        Creates class object to assign attributes to.
    """
    pass

###############################################################################

def open_shp(filename):

    fn_grwl = filename
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shape = driver.Open(fn_grwl)
    layer = shape.GetLayer()
    numFeatures = layer.GetFeatureCount()

    # Creating empty arrays to fill in with grwl attributes.
    shp = Object()
    shp.lon = np.zeros(numFeatures)
    shp.lat = np.zeros(numFeatures)
    shp.name = np.zeros(numFeatures)
    shp.name = shp.name.astype(str)
    
    # Saving data to arrays.
    cnt = 0
    for feature in range(numFeatures):
        shp.lon[cnt] = layer.GetFeature(feature).GetField('x')
        shp.lat[cnt] = layer.GetFeature(feature).GetField('y')
        shp.name[cnt] = layer.GetFeature(feature).GetField('name')
        cnt += 1

    return shp

###############################################################################
###############################################################################
###############################################################################

start = time.time()

region = 'OC'

fn_sword = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/outputs/Reaches_Nodes/netcdf/'+region.lower()+'_sword_v11.nc'
#fn_sword = 'E:/Users/Elizabeth Humphries/Documents/SWORD/For_Server/outputs/Reaches_Nodes_v10/netcdf/'+region.lower()+'_sword_v10.nc'

names_dir = 'E:/Users/Elizabeth Humphries/Documents/SWORD/For_Server/inputs/RiverNames/SWORD_Node_Names/'+region+'/'
fn_names = np.array([f for f in os.listdir(names_dir) if '.shp' in f])
names_basin = np.array([name[-6:-4] for name in fn_names])
names_basin = names_basin.astype(int)

# read in global data
sword = nc.Dataset(fn_sword, 'r+')

nid = sword.groups['nodes'].variables['node_id'][:]
nrch_id = sword.groups['nodes'].variables['reach_id'][:]
nlon = sword.groups['nodes'].variables['x'][:]
nlat = sword.groups['nodes'].variables['y'][:]
node_names = sword.groups['nodes'].variables['river_name'][:]

rch_id = sword.groups['reaches'].variables['reach_id'][:]
rch_names = sword.groups['reaches'].variables['river_name'][:]

level2 = np.array([np.int(np.str(ind)[0:2]) for ind in nid])

for ind in list(range(len(names_basin))):
    
    shp = open_shp(names_dir+fn_names[ind])
    
    sword_subset = np.where(level2 == names_basin[ind])[0]
    # find closest points.     
    name_pts = np.vstack((shp.lon, shp.lat)).T
    node_pts = np.vstack((nlon[sword_subset], nlat[sword_subset])).T
    kdt = sp.cKDTree(name_pts)
    eps_dist, eps_ind = kdt.query(node_pts, k = 2) 
    
    new_indexes = eps_ind[:,0]
    node_names[sword_subset] = shp.name[new_indexes]
    
    #assigning reach names based on nodes
    unq_rch = np.unique(nrch_id[sword_subset])
    for idx in list(range(len(unq_rch))):
        rch = np.where(rch_id == unq_rch[idx])[0]
        unq_names = np.unique(node_names[np.where(nrch_id == unq_rch[idx])[0]])
        if len(unq_names) > 1:
            rmv = np.where(unq_names == 'NODATA')[0]
            legit_names = np.delete(unq_names, rmv)
            
            if len(legit_names) > 1:
                for idy in list(range(len(legit_names))):
                    if idy == 0:
                        combined_names = legit_names[idy] + '; '
                    elif idy == len(legit_names)-1:
                        combined_names = combined_names + legit_names[idy]
                    else:
                        combined_names = combined_names + legit_names[idy] + '; '
                rch_names[rch] = np.array(combined_names)
                #print(ind, idx, rch_id[rch], legit_names, combined_names)
                
            if len(legit_names) == 0:
                rch_names[rch] = 'NODATA'
                
            if len(legit_names) == 1:
                rch_names[rch] = legit_names
        
        else:
            rch_names[rch] = unq_names
               

sword.groups['nodes'].variables['river_name'][:] = node_names       
sword.groups['reaches'].variables['river_name'][:] = rch_names
sword.close()

end = time.time()
print('DONE with ' + region + ': in ' + str(np.round((end-start)/60, 2)) + ' min')
