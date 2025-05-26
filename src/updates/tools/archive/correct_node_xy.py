# -*- coding: utf-8 -*-
"""
Created on Wed Oct 09 12:56:58 2019
"""
from __future__ import division
import os
main_dir = os.getcwd()
import numpy as np
import time
import netCDF4 as nc
import geopandas as gp
from shapely.geometry import Point
import pandas as pd
import argparse 

#########################################################NA######################
###############################################################################
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("region", help="continental region", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

outdir  = main_dir+'/data/outputs/Reaches_Nodes/'
outpath = outdir+version+'/'
fn = outpath+'netcdf/'+region.lower()+'_sword_'+version+'.nc'

# read originial data.
data = nc.Dataset(fn, 'r+')

nodes = np.array(data.groups['nodes'].variables['node_id'][:])
cl_nodes = np.array(data.groups['centerlines'].variables['node_id'][0,:])
cl_x = np.array(data.groups['centerlines'].variables['x'][:])
cl_y = np.array(data.groups['centerlines'].variables['y'][:])
node_x = np.zeros(len(nodes))
node_y = np.zeros(len(nodes))
for n in list(range(len(nodes))):
    print(n, len(nodes)-1)
    pts = np.where(cl_nodes == nodes[n])[0]
    node_x[n] = np.median(cl_x[pts])
    node_y[n] = np.median(cl_y[pts])

data.groups['nodes'].variables['x'][:] = node_x
data.groups['nodes'].variables['y'][:] = node_y
data.close()

