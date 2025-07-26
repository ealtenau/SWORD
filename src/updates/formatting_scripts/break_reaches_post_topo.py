# -*- coding: utf-8 -*-
"""
Breaking Reaches (break_reaches_post_topo.py)
==============================================

This script breaks reaches in the SWOT River Database 
(SWORD) at specified locations that are identified in 
the 'find_tributary_breaks.py' script.

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python path/to/break_reaches_post_topo.py NA v17 

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import geopandas as gp
import pandas as pd
import argparse
import glob
import argparse
from src.updates.sword import SWORD
import src.updates.geo_utils as geo 
# import matplotlib.pyplot as plt

###############################################################################   

def aggregate_files(trib_files):
    """
    Aggregates multiple csv files of tributary break locations 
    into one dataframe. 

    Parmeters
    ---------
    trib_files: geopandas.dataframe
        The geodataframes containing spatial and attribute information
        for breaking reaches at specified locations.

    Returns
    -------
    gdf_all: geopandas.dataframe
        Concatenated geodataframe.
    
    """

    for f in list(range(len(trib_files))):
        gdf = gp.read_file(trib_files[f])
        if f == 0:
            gdf_all = gdf.copy()
        else:
            gdf_all = pd.concat([gdf_all, gdf], ignore_index=True)

    return gdf_all
    
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version
multi_file = 'True'

#manual
# region = 'SA'
# version = 'v18'
# reach = np.array([61680300053])
# break_id = np.array([10823001])

#read sword data. 
sword = SWORD(main_dir, region, version)
sword.copy() #copies original file for version control. 

############################ Comment out for manual 
trib_dir = sword.paths['updates_dir']+'/tribs'
trib_files = np.sort(glob.glob(os.path.join(trib_dir, '*.gpkg')))

#automatic
if multi_file == 'True':
    tribs = aggregate_files(trib_files)
else:
    tribs = gp.read_file(trib_files[0]) 
reach = np.array(tribs['reach_id']) 
break_id = np.array(tribs['cl_id']) 
############################ Comment out for manual 

#break reaches.
sword.break_reaches(reach, break_id, verbose=True)

### Write Data. 
sword.save_nc()

