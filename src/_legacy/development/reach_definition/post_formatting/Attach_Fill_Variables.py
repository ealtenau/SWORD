# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
                    Attaching Fill Variables
-------------------------------------------------------------------------------
Copyright (c) 2018-2021 UNC Chapel Hill. All rights reserved.
Created by E. Altenau. Some functions were taken or modified from C. Lion's
"Tools.py" script (c) 2016.

DESCRIPTON:
    This script fills in values for several SWORD variables that are originally
    written as fill values. The different functions read in separate files 
    containing the variable values at the node or reach locations and attaches the 
    the values to the official SWORD database. Variables that are filled in are 
    the "extreme distance coeficient", "river names", "maximum width", 
    "ice flag", and "meander length". The "ice flag" variable will need to be 
    re-computed (by Xiao Yang) with each new SWORD version. 

INPUTS:
    region -- Two letter regional identifier for the desired SWORD continent 
        to be run. (i.e.: 'NA')
    fn_sword -- Directory to the SWORD database NetCDF files.
    fn_iceflag -- Directory to the ice flag files (.csv).
    fn_ext_dist -- Directory to the extreme distance coeficient values (.csv).
    names_dir -- Directory to the river names files (.shp)
    max_wth_dir -- Directory to the maximum width files (.csv).
    raster_dir -- Directory to the GRWL river masks (.tif)

OUTPUTS:
    SWORD database with filled in variable values.
"""

import os
main_dir = os.getcwd()
import netCDF4 as nc
import numpy as np
import pandas as pd
import time
from scipy import spatial as sp
from osgeo import ogr
from pyproj import Proj
import rasterio
#import matplotlib.pyplot as plt

###############################################################################
################################# Functions ###################################
###############################################################################

class Object(object):
    """
    FUNCTION:
        Creates class object to assign attributes to.
    """
    pass

###############################################################################

def open_shp(filename):
    
    """
    FUNCTION:
        Attaches river names to SWORD reaches and nodes based on loaction 
        information.
    
    INPUTS:
        filename -- Directory to shapefiles containing river names.
        
    OUTPUTS:
        shp -- Object containing river names and location information in array
            format.
    """
    
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

def attach_ice_flag(fn_iceflag, fn_sword):
    
    """
    FUNCTION:
        Attaches ice flag values to SWORD reaches based on Reach ID.
    
    INPUTS:
        filename -- Directory to ice flag file (.csv).
        
    OUTPUTS:
        SWORD NetCDF file conatining filled ice flag values.
    """
    
    start = time.time()
    
    # read in global data
    sword = nc.Dataset(fn_sword, 'r+')
    iceflag = pd.read_csv(fn_iceflag)
    iceflag = np.array(iceflag)
    
    # assigning variables to arrays.
    sword_rch_id = sword.groups['reaches'].variables['reach_id'][:]
    sword_ice_flag = sword.groups['reaches'].variables['iceflag'][:]
    iceflag_rch_id = iceflag[:,0]
    iceflag_vals = iceflag[:,1:367]
    
    # looping though each reach and attaching ice flag values.
    for ind in list(range(len(sword_rch_id))):
        ice = np.where(iceflag_rch_id == sword_rch_id[ind])[0]
        if len(ice) == 0:
            #print(ind, sword_rch_id[ind], "no match for ice flag")
            continue
        else:
            sword_ice_flag[:,ind] = iceflag_vals[ice,:]
    
    sword.groups['reaches'].variables['iceflag'][:] = sword_ice_flag
    sword.close()
    
    end = time.time()
    print('Finished Attaching IceFlag in: ' + str(np.round((end-start)/60, 2)) + ' min')

###############################################################################
    
def attach_ext_dist(fn_ext_dist, fn_sword):
    
    """
    FUNCTION:
        Attaches extreme distance coeficient values to SWORD nodes based on 
        loaction information.
    
    INPUTS:
        filename -- Directory to the extreme distance coeficient files (.csv). 
        
    OUTPUTS:
        SWORD NetCDF file conatining filled extreme distance coeficient values.
    """
    
    start = time.time()
    
    # read in global data
    new = nc.Dataset(fn_sword, 'r+')
    
    # make array of node locations.
    nlon = new.groups['nodes'].variables['x'][:]
    nlat = new.groups['nodes'].variables['y'][:]
    wth = new.groups['nodes'].variables['width'][:]
    max_wth = new.groups['nodes'].variables['max_width'][:]
    lakeflag = new.groups['nodes'].variables['lakeflag'][:]
    nchan = new.groups['nodes'].variables['n_chan_mod'][:]
    
    csv = pd.read_csv(fn_ext_dist)
    olon = np.array(csv.lon[:])
    olat = np.array(csv.lat[:])
    odist = np.array(csv.dist_thresh[:])
    
    # find closest points.     
    csv_pts = np.vstack((olon, olat)).T
    node_pts = np.vstack((nlon, nlat)).T
    kdt = sp.cKDTree(csv_pts)
    eps_dist, eps_ind = kdt.query(node_pts, k = 2) 
    
    new_indexes = eps_ind[:,0]
    new_ext_dist = odist[new_indexes]
    new_ext_dist[np.where(lakeflag == 1)] = 5 #changed from 20 to 5 on 4/19/2023.
    
    check = np.where(new_ext_dist == 1)[0]
    not_single = np.where(nchan[check] > 1)[0]    
    new_ext_dist[check[not_single]] = 2    
         
    # reduce coeficient to 1 everywhere for max width testing. 
    update = np.where(new_ext_dist == 2)[0]
    for ind in list(range(len(update))):
        #divide and round max width by width to get new ext_dist_coef value for multichannel rivers.
        val = np.around(max_wth[update[ind]]/wth[update[ind]])
        if val <= 1:
            new_ext_dist[update[ind]] = 2
        elif val >= 5: #changed from 20 to 5 on 4/19/2023.
            new_ext_dist[update[ind]] = 5 #changed from 20 to 5 on 4/19/2023.
        else:
            new_ext_dist[update[ind]] = val
    
    
    # In the case where ext_dist_coef values are greater than 20 bring them back to 20. 
    new_ext_dist[np.where(new_ext_dist > 5)] = 5 #changed from 20 to 5 on 4/19/2023.
                 
    # assign new coeficients.
    new.groups['nodes'].variables['ext_dist_coef'][:] = new_ext_dist
    new.close()
    
    end = time.time()
    print('Finished Attaching ExtDistCoef in: ' + str(np.round((end-start)/60, 2)) + ' min')

###############################################################################    

def attach_river_names(names_dir, fn_sword):

    """
    FUNCTION:
        Attaches river names to SWORD reaches and nodes based on 
        loaction information.
    
    INPUTS:
        filename -- Directory to the river name files (.shp). 
        
    OUTPUTS:
        SWORD NetCDF file conatining filled river names.
    """
    
    start = time.time()
    
    #format river name filenames.
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
    
    level2 = np.array([int(str(ind)[0:2]) for ind in nid])
    
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
        
        nan = np.where(node_names == 'NaN')[0]
        node_names[nan] = 'NODATA'

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
    print('Finished Attaching River Names in: ' + str(np.round((end-start)/60, 2)) + ' min')
  
###############################################################################

def attach_max_wth(max_wth_dir, raster_dir, fn_sword):
    
    """
     FUNCTION:
        Attaches maximum width values to SWORD nodes based on 
        loaction information. The function also replaces width = 1 values 
        with the reach width value or maximum width value.
    
    INPUTS:
        filename -- Directory to the maximum width files (.csv). 
        
    OUTPUTS:
        SWORD NetCDF file conatining filled maximum width values.
    """
    
    start = time.time()
    
    # read in global data
    sword = nc.Dataset(fn_sword, 'r+')
    
    # make array of node locations.
    cl_lon = sword.groups['centerlines'].variables['x'][:]
    cl_lat = sword.groups['centerlines'].variables['y'][:]
    cl_rch_id = sword.groups['centerlines'].variables['reach_id'][:]
    cl_pts = np.array([cl_lon, cl_lat]).T
    
    node_id = sword.groups['nodes'].variables['node_id'][:]
    n_rch_id = sword.groups['nodes'].variables['reach_id'][:]
    n_chan = sword.groups['nodes'].variables['n_chan_mod'][:]
    n_wth = sword.groups['nodes'].variables['width'][:]
    
    rch_id = sword.groups['reaches'].variables['reach_id'][:]
    rch_wth = sword.groups['reaches'].variables['width'][:]
    rch_id_up = sword.groups['reaches'].variables['rch_id_up'][:]
    rch_id_dn = sword.groups['reaches'].variables['rch_id_dn'][:]
    
    node_max_wth = np.zeros(len(node_id))
    rch_max_wth = np.zeros(len(rch_id))
    
    raster_paths = np.array([f for f in os.listdir(raster_dir) if '.tif' in f])
    raster_names = np.array([name[0:7] for name in raster_paths])
    raster_len = np.array([len(str(name)) for name in raster_paths])
    remove = np.where(raster_len > 11)[0]
    raster_names = np.delete(raster_names, remove)
    raster_paths = np.delete(raster_paths, remove)

    projection = []
    for ras in list(range(len(raster_paths))):
        tif = rasterio.open(raster_dir+raster_paths[ras])
        projection.append(tif.crs)
    
    #format regional data info.
    csv_paths = np.array([f for f in os.listdir(max_wth_dir) if '.csv' in f])
    csv_names = np.array([name[0:7] for name in csv_paths])
    for ind in list(range(len(csv_paths))):
        
        #print(ind)
        
        #read in max_wth csv data.
        csv = pd.read_csv(max_wth_dir+csv_paths[ind])
        csv_x = np.array(csv.x[:])
        csv_y = np.array(csv.y[:])
        csv_max_wth = np.array(csv.bank_wth[:])
        
        #find assiciated raster to current tile, and find projection to calculate
        #lat-lon from utm info.
        raster = np.where(raster_names == csv_names[ind])[0]
        myProj = Proj(projection[int(raster)]) 
        csv_lon, csv_lat = myProj(csv_x, csv_y, inverse=True)
        
        #find sword points within tile extent. 
        ll = np.array([np.min(csv_lon), np.min(csv_lat)])  # lower-left
        ur = np.array([np.max(csv_lon), np.max(csv_lat)])  # upper-right
        
        inidx = np.all(np.logical_and(ll <= cl_pts, cl_pts <= ur), axis=1)
        inbox = cl_pts[inidx]
        
        if len(inbox) == 0:
            #print(ind, csv_names[ind], ' no overlapping data')
            continue
    
        cl_lon_clip = inbox[:,0]
        cl_lat_clip = inbox[:,1]
        cl_rch_id_clip = cl_rch_id[0,inidx]
        
        # find closest points.    
        csv_pts = np.vstack((csv_lon, csv_lat)).T
        cl_pts_clip = np.vstack((cl_lon_clip, cl_lat_clip)).T
        kdt = sp.cKDTree(csv_pts)
        eps_dist, eps_ind = kdt.query(cl_pts_clip, k = 50) 
        
        #assign max width of closest points to new vector
        cl_max_wth = np.max(csv_max_wth[eps_ind[:]], axis = 1)
        
        #if points don't have sword points within 500 m assign value of 1 and give max width below (lines 442-447). 
        too_far = np.where(eps_dist[:,0] > 0.0005)[0]
        cl_max_wth[too_far] = 1

        #assign max width per unique reach to node locations. 
        uniq_rch = np.unique(cl_rch_id_clip)
        for idx in list(range(len(uniq_rch))):
            rch = np.where(cl_rch_id_clip == uniq_rch[idx])[0]
            assign1 = np.where(rch_id == uniq_rch[idx])[0]
            assign2 = np.where(n_rch_id == uniq_rch[idx])[0]
            max_wth1 = np.max(cl_max_wth[rch])
            rch_max_wth[assign1] = max_wth1
            node_max_wth[assign2] = max_wth1
                 
    # filling in reach wth = 1 values. 
    rch_one_wth = np.where(rch_wth == 1)[0]          
    for ind2 in list(range(len(rch_one_wth))): 
        nghs = np.unique(np.array([rch_id_up[:,rch_one_wth[ind2]], rch_id_dn[:,rch_one_wth[ind2]]]))
        zero = np.where(nghs == 0)[0]
        nghs = np.delete(nghs, zero)
        if len(nghs) > 0:
            ngh_wths = np.zeros(len(nghs))
            for ind3 in list(range(len(nghs))):
                r = np.where(rch_id == nghs[ind3])[0]
                ngh_wths[ind3] = rch_wth[r]
            
            #find max width of neighbors
            ngh_max_wth = np.max(ngh_wths)
            
            if ngh_max_wth > 1:
                rch_wth[rch_one_wth[ind2]] = ngh_max_wth
            else:
                rch_wth[rch_one_wth[ind2]] = 1000
        
        else:
            # if the reach has no neighbors assign default width value = 1000. 
            rch_wth[rch_one_wth[ind2]] = 1000
                      
    # filling in node wth = 1 values. 
    one_wth = np.where(n_wth == 1)[0] #32,566/164,2238 for NA
    
    for idy in list(range(len(one_wth))):
        nrch = np.where(rch_id == n_rch_id[one_wth[idy]])[0]
        rwth = rch_wth[nrch]
        n_wth[one_wth[idy]] = rwth
                      
    #fill in nodes with no max width values with regular width values. 
    no_max_wth = np.where(node_max_wth <= 1)[0]
    node_max_wth[no_max_wth] = n_wth[no_max_wth]
    #fill in reaches with no max width values with regular width values. 
    no_max_wth2 = np.where(rch_max_wth <= 1)[0]
    rch_max_wth[no_max_wth2] = rch_wth[no_max_wth2]
    
    # replacing single channel max_wth values with regular widths.
    single_channels = np.where(n_chan <= 1)[0]
    node_max_wth[single_channels] = n_wth[single_channels]
    
    # assign new coeficients.
    sword.groups['nodes'].variables['width'][:] = n_wth
    sword.groups['reaches'].variables['width'][:] = rch_wth
    sword.groups['nodes'].variables['max_width'][:] = node_max_wth
    sword.groups['reaches'].variables['max_width'][:] = rch_max_wth
    
    sword.close()
    
    end = time.time()
    print('Finished Attaching Max Widths in: ' + str(np.round((end-start)/60, 2)) + ' min')

###############################################################################

def attach_sinuosity(sin_dir, fn_sword):
    """
    FUNCTION:
        Attaches meander length and sinuosity values to SWORD reaches based 
        on Node ID. The input files for this function are output from the 
        "CalculateSinuositySWORD.m" found in the src directory under: 
        reach_definition/post_formatting/SWORD-Sinuosity
    
    INPUTS:
        filename -- Directory to meander length and sinuosity file (.nc).
        
    OUTPUTS:
        SWORD NetCDF file conatining filled meander length and sinuosity values.
    """
    
    start = time.time()
    
    # read in global data
    sword = nc.Dataset(fn_sword, 'r+')
    sinuosity = nc.Dataset(sin_dir)
    
    sword_nid = sword.groups['nodes'].variables['node_id'][:]
    sinuosity_nid = sinuosity.groups['nodes'].variables['node_id'][:]
    sinuosity_vals = sinuosity.groups['nodes'].variables['sinuosity'][:]
    meand_vals = sinuosity.groups['nodes'].variables['meanderwavelength'][:]
    
    __, __, ind_sin = np.intersect1d(sword_nid,sinuosity_nid, return_indices=True)
    
    sword.groups['nodes'].variables['sinuosity'][:] = sinuosity_vals[ind_sin]
    sword.groups['nodes'].variables['meander_length'][:] = meand_vals[ind_sin]

    sword.close()
    sinuosity.close()
    
    end = time.time()
    print('Finished Attaching Sinuosity in: ' + str(np.round((end-start)/60, 2)) + ' min')

    
###############################################################################
################################# Main Code ###################################
###############################################################################
    
region = 'OC'
version = 'v15'

#Paths will need to be replaced.
fn_sword = main_dir+'/data/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
#paths for "attach_ice_flag". Will need to be replaced.
fn_iceflag = main_dir+'/data/inputs/IceFlag/SWOT_river_ice_flag_v15.csv'
#paths for "attach_ext_dist". Will need to be replaced.
fn_ext_dist = main_dir+'/data/inputs/ExtremeDist_Nodes/'+region.lower()+'_extdist_nodes.csv'
#paths for "attach_river_names". Will need to be replaced.
names_dir = main_dir+'/data/inputs/RiverNames/SWORD_Node_Names/'+region+'/'
#paths for "attach_max_wth". Will need to be replaced.
max_wth_dir = main_dir+'/data/inputs/MaxWidth_Nodes/'+region+'/'
raster_dir = main_dir+'/data/inputs/GRWL/GRWL_Masks_V01.01_LatLonNames/'
#path for meander_length and sinuosity.
sin_dir = main_dir+'/data/inputs/Sinuosity_Files/netcdf/'+region.lower()+'_sword_v15output.nc'


#functions for fill variables.
attach_ice_flag(fn_iceflag, fn_sword)
# attach_river_names(names_dir, fn_sword)
# attach_max_wth(max_wth_dir, raster_dir, fn_sword)
attach_sinuosity(sin_dir, fn_sword)
# attach_ext_dist(fn_ext_dist, fn_sword)
