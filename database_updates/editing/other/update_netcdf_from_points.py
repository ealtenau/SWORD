import netCDF4 as nc
import geopandas as gp
import numpy as np
import pandas as pd
from shapely.geometry import Point
import os

###############################################################################

region = 'NA'
version = 'v17'
sword_fn = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'\
    +version+'/netcdf_beta/'+region.lower()+'_sword_'+version+'.nc'
sword = nc.Dataset(sword_fn)
# rch_vars = sword.groups['reaches'].variables.keys()
# node_vars = sword.groups['nodes'].variables.keys()

pt_dir = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'\
    +version+'/gpkg_30m/'+region.lower()+'/'
pt_files = np.sort(os.listdir(pt_dir))
pt_files = [fn for fn in pt_files if 'gpkg' in fn]

for f in list(range(len(pt_files))):
    pts = gp.read_file(pt_dir+pt_files[f])

    x = np.array(pts.x)
    y = np.array(pts.y)
    cl_ids = np.array(pts.cl_id)
    rch_ids = np.array(pts.reach_id)
    nd_ids = np.array(pts.node_id)

    geom = [i for i in pts.geometry]
    new_x = np.zeros(len(geom))
    new_y = np.zeros(len(geom))
    for ind in list(range(len(geom))):
        new_x[ind] = np.array(geom[ind].coords.xy[0])
        new_y[ind] = np.array(geom[ind].coords.xy[1])

    pt_diff = np.where(new_x != x)[0]
    updated_ids = cl_ids[pt_diff]


    # variables effected by x-y change 
        # x-y
        # x_min/max y_min/max
        # reach/node dist_out
        # reach/node len
    
    unq_rch = np.unique(rch_ids[pt_diff])
    unq_node = np.unique(nd_ids[pt_diff])
    
