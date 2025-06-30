# -*- coding: utf-8 -*-
"""
Producing Centerline Vector Points (level2_sword_30m.py)
===============================================================

This script takes the SWORD netCDF and output the 30 meter 
centerline level points in geopackage format. Geopackage 
files are output at Level 2 Pfafstetter basin scale. 

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python level2_sword_30m.py NA v17

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import geopandas as gp
import numpy as np
import pandas as pd
from shapely.geometry import Point
import argparse
from src.updates.sword import SWORD

parser = argparse.ArgumentParser()
parser.add_argument("region", help="continental region", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

#read data.
sword = SWORD(main_dir, region, version)

#isolate level 2 basin number for every centerline point. 
level2_basins = np.array([int(str(ind)[0:2]) for ind in sword.centerlines.reach_id[0,:]])
unq_l2 = np.unique(level2_basins)

#loop through each level 2 basin and create/write geodataframes. 
for ind in list(range(len(unq_l2))):
    print(unq_l2[ind])
    pts = np.where(level2_basins == unq_l2[ind])[0]
    points = gp.GeoDataFrame([
        sword.centerlines.x[pts],
        sword.centerlines.y[pts],
        sword.centerlines.cl_id[pts],
        sword.centerlines.reach_id[0,pts],
        sword.centerlines.node_id[0,pts],
    ]).T

    #rename columns.
    points.rename(
        columns={
            0:"x",
            1:"y",
            2:"cl_id",
            3:"reach_id",
            4:"node_id",
        },inplace=True)

    #attach geometry and format attributes. 
    points = points.apply(pd.to_numeric, errors='ignore') # points.dtypes
    geom = gp.GeoSeries(map(Point, zip(sword.centerlines.x[pts], sword.centerlines.y[pts])))
    points['geometry'] = geom
    points = gp.GeoDataFrame(points)
    points.set_geometry(col='geometry')
    points = points.set_crs(4326, allow_override=True)

    #write data. 
    outgpkg = sword.paths['pts_gpkg_dir']+'hb'+str(unq_l2[ind])+'_centerlines_'+version+'.gpkg'
    points.to_file(outgpkg, driver='GPKG', layer='points')

