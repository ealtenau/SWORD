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

# cl_vars = sword.groups['centerlines'].variables.keys()
# rch_vars = sword.groups['reaches'].variables.keys()
# node_vars = sword.groups['nodes'].variables.keys()

df = pd.DataFrame(sword.groups['centerlines'].variables['x'][:], sword.groups['centerlines'].variables['y'][:])
dups = df.duplicated()
rmv = np.where(dups == True)[0]

#reach attributes to update: 'cl_ids', 'x', 'x_min', 'x_max', 'y', 'y_min', 'y_max', 'reach_length', 'dist_out'
#node attributes to update: 'cl_ids', 'x', 'y', 'node_length', 'dist_out'
cl_rch_ids = np.array(sword.groups['centerlines'].variables['reach_id'][0,:])
cl_node_ids = np.array(sword.groups['centerlines'].variables['node_id'][0,:])

rch_updates = np.unique(cl_rch_ids[rmv])
node_updates = np.unique(cl_node_ids[rmv])

#need to loop through and redo attributes for reaches and nodes. 