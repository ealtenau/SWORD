"""
Finding GLOW-S IDs associated with SWORD CalVal Reach IDs
(calval_sword_glow-s_matchups.py).
===============================================================

This script finds GLOW-S COMID IDs associated with SWORD 
calibration and validation reach IDs. It is a bespoke script 
written for SWORD v16 to GLOW-S width comparisons. Need to 
specify SWORD region and GLOW-S region in beginning of script. 

Input is a csv file of specific SWORD IDs and output is a csv
file containing SWORD reach ID and associated GLOW-S COMID ID. 

"""

import os
main_dir = os.getcwd()
import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt

region = 'NA'
glows_rg = '7-8'

#read data. 
reaches = pd.read_csv(main_dir+'/data/swot_data/glows_sword_calval_reaches/S2_Validation_reaches.csv')
sword = nc.Dataset(main_dir+'/data/outputs/Reaches_Nodes/v16_glows/netcdf/'+region.lower()+'_sword_v16_glows.nc')

nc_node_rchs = sword['/nodes/reach_id/'][:]
nc_node_ids = sword['/nodes/node_id/'][:]
nc_glows_ids = sword['/nodes/glows_river_id/'][:]
unq_rchs = np.unique(reaches['reach_id'])

#loop through and match GLOW-S COMID to SWORD's reach ID. 
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
df.to_csv(main_dir+'/data/swot_data/glows_sword_calval_reaches/reach_csv_glows_s2validation/sword_glows_s2_validation_regions'+glows_rg+'.csv', index = False)
sword.close()
print('Done with '+region)