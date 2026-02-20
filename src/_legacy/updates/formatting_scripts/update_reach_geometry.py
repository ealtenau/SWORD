# -*- coding: utf-8 -*-
"""
Updating Reach Centerline Coordinates (update_reach_geometry.py).
===============================================================

This script updates the SWORD netCDF centerline coordinates
based on manually updated geometry in the SWORD reaches geopackage 
files.  

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA), SWORD version (i.e. v17), 
and a csv file containing the Reach IDs that have been
manually adjusted in the geopackage files.

Execution example (terminal):
    python path/to/update_reach_geometry.py NA v17 path/to/reaches.csv

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import geopandas as gp
import geopy.distance
import pandas as pd
import argparse
from src.updates.sword_duckdb import SWORD
import src.updates.geo_utils as geo

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("csv", help="csv file of reaches to update geometry", type = str)

args = parser.parse_args()

region = args.region
version = args.version

### manual 
# region = 'OC'
# version = 'v18'
# reach = [53190500116, 56257300106]

#read sword data.
db_path = os.path.join(main_dir, f'data/duckdb/sword_{version}.duckdb')
sword = SWORD(db_path, region, version)
sword.copy() #copies original file for version control. 

#read csv file of updated reaches. 
rch_dir = args.csv
rch_df = pd.read_csv(rch_dir)
reach = np.unique(np.array(rch_df['reach_id']))

#read geopackage data.
gpkg_fn = sword.paths['gpkg_dir']+sword.paths['gpkg_rch_fn']
gpkg = gp.read_file(gpkg_fn)
geom = [i for i in gpkg.geometry]

#loop through and update reach coordinates in netCDF. 
for r in list(range(len(reach))):
    print(r, len(reach)-1)
    #geopackage geometry 
    gp_rch = np.where(gpkg['reach_id'] == reach[r])[0]
    if len(gp_rch) == 0:
        print('Reach does not exist: ', reach[r])
        continue
    lon = np.array(geom[gp_rch[0]].coords.xy[0])
    lat = np.array(geom[gp_rch[0]].coords.xy[1])

    #filtering out "connecting" coordinates in geopackage file.
    rch = np.where(sword.centerlines.reach_id[0,:] == reach[r])[0]
    if len(rch) != len(lon):
        if np.abs(len(rch)-len(lon)) == 1:
            mn = np.where(sword.centerlines.cl_id[rch] == np.min(sword.centerlines.cl_id[rch]))[0]
            mx = np.where(sword.centerlines.cl_id[rch] == np.max(sword.centerlines.cl_id[rch]))[0]
            coords_0 = (sword.centerlines.y[rch[mn]], sword.centerlines.x[rch[mn]])
            coords_1 = (sword.centerlines.y[rch[mx]], sword.centerlines.x[rch[mx]])
            coords_2 = (lat[0], lon[0])
            coords_3 = (lat[-1], lon[-1])
            end1 = np.min([geopy.distance.geodesic(coords_2, coords_0).m, 
                           geopy.distance.geodesic(coords_2, coords_1).m])
            end2 = np.min([geopy.distance.geodesic(coords_3, coords_0).m, 
                           geopy.distance.geodesic(coords_3, coords_1).m])
            if end1<end2:
                lon = lon[1::]
                lat = lat[1::]
            else:
                lon = lon[0:-1]
                lat = lat[0:-1]
        if np.abs(len(rch)-len(lon)) == 2:
            lon = lon[1:-1]
            lat = lat[1:-1]

    #sort coordinates by centerline IDs. 
    order_ids = np.argsort(sword.centerlines.cl_id[rch])
    sword.centerlines.x[rch[order_ids]] = lon
    sword.centerlines.y[rch[order_ids]] = lat

    #recalculate centerline reach length.
    x_coords = sword.centerlines.x[rch[order_ids]]
    y_coords = sword.centerlines.y[rch[order_ids]]
    diff = geo.get_distances(x_coords,y_coords)
    dist = np.cumsum(diff)

    #update sword.reaches. 
    rind = np.where(sword.reaches.id == reach[r])[0]
    base_val = sword.reaches.dist_out[rind] - sword.reaches.len[rind]
    sword.reaches.len[rind] = np.max(dist)
    sword.reaches.dist_out[rind] = np.max(dist)+base_val
    #reach x-y
    sword.reaches.x[rind] = np.median(sword.centerlines.x[rch])
    sword.reaches.x_min[rind] = np.min(sword.centerlines.x[rch])
    sword.reaches.x_max[rind] = np.max(sword.centerlines.x[rch])
    sword.reaches.y[rind] = np.median(sword.centerlines.y[rch])
    sword.reaches.y_min[rind] = np.min(sword.centerlines.y[rch])
    sword.reaches.y_max[rind] = np.max(sword.centerlines.y[rch])
    
    #update node coordinates. 
    unq_nodes = np.unique(sword.centerlines.node_id[0,rch])
    for n in list(range(len(unq_nodes))):
        nind = np.where(sword.nodes.id == unq_nodes[n])[0]
        pts = np.where(sword.centerlines.node_id[0,rch[order_ids]] == unq_nodes[n])[0]
        sword.nodes.x[nind] = np.median(sword.centerlines.x[rch[order_ids[pts]]])
        sword.nodes.y[nind] = np.median(sword.centerlines.y[rch[order_ids[pts]]])
        sword.nodes.len[nind] = max(np.cumsum(diff[pts]))
        if len(pts) == 1:
            sword.nodes.len[n] = 30
    #node dist_out.
    nrch = np.where(sword.nodes.reach_id == reach[r])[0]
    sort_nodes = nrch[np.argsort(sword.nodes.id[nrch])]
    sword.nodes.dist_out[sort_nodes] = np.cumsum(sword.nodes.len[sort_nodes])+base_val

#write data.
sword.save_nc()
print('DONE')