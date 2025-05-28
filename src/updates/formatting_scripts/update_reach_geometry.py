# -*- coding: utf-8 -*-
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
import src.updates.sword_utils as swd
import src.updates.calc_utils as ct

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

paths = swd.prepare_paths(main_dir, region, version)
gpkg_fn = paths['gpkg_dir']+paths['gpkg_fn']
sword_fn = paths['nc_dir']+paths['nc_fn']

rch_dir = args.csv
rch_df = pd.read_csv(rch_dir)
reach = np.unique(np.array(rch_df['reach_id']))

#read geopackage data.
gpkg = gp.read_file(gpkg_fn)
geom = [i for i in gpkg.geometry]

#read nc data.
centerlines, nodes, reaches = swd.read_nc(sword_fn)

for r in list(range(len(reach))):
    print(r, len(reach)-1)
    #geopackage geometry 
    gp_rch = np.where(gpkg['reach_id'] == reach[r])[0]
    if len(gp_rch) == 0:
        print('Reach does not exist: ', reach[r])
        continue
    lon = np.array(geom[gp_rch[0]].coords.xy[0])
    lat = np.array(geom[gp_rch[0]].coords.xy[1])

    #netcdf updates.
    rch = np.where(centerlines.reach_id[0,:] == reach[r])[0]
    if len(rch) != len(lon):
        if np.abs(len(rch)-len(lon)) == 1:
            mn = np.where(centerlines.cl_id[rch] == np.min(centerlines.cl_id[rch]))[0]
            mx = np.where(centerlines.cl_id[rch] == np.max(centerlines.cl_id[rch]))[0]
            coords_0 = (centerlines.y[rch[mn]], centerlines.x[rch[mn]])
            coords_1 = (centerlines.y[rch[mx]], centerlines.x[rch[mx]])
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

    order_ids = np.argsort(centerlines.cl_id[rch])
    centerlines.x[rch[order_ids]] = lon
    centerlines.y[rch[order_ids]] = lat

    #recalculate centerline reach length.
    x_coords = centerlines.x[rch[order_ids]]
    y_coords = centerlines.y[rch[order_ids]]
    diff = ct.get_distances(x_coords,y_coords)
    dist = np.cumsum(diff)

    #update reaches. 
    rind = np.where(reaches.id == reach[r])[0]
    base_val = reaches.dist_out[rind] - reaches.len[rind]
    reaches.len[rind] = np.max(dist)
    reaches.dist_out[rind] = np.max(dist)+base_val
    #reach x-y
    reaches.x[rind] = np.median(centerlines.x[rch])
    reaches.x_min[rind] = np.min(centerlines.x[rch])
    reaches.x_max[rind] = np.max(centerlines.x[rch])
    reaches.y[rind] = np.median(centerlines.y[rch])
    reaches.y_min[rind] = np.min(centerlines.y[rch])
    reaches.y_max[rind] = np.max(centerlines.y[rch])
    
    #update nodes. 
    unq_nodes = np.unique(centerlines.node_id[0,rch])
    for n in list(range(len(unq_nodes))):
        nind = np.where(nodes.id == unq_nodes[n])[0]
        pts = np.where(centerlines.node_id[0,rch[order_ids]] == unq_nodes[n])[0]
        nodes.x[nind] = np.median(centerlines.x[rch[order_ids[pts]]])
        nodes.y[nind] = np.median(centerlines.y[rch[order_ids[pts]]])
        nodes.len[nind] = max(np.cumsum(diff[pts]))
        if len(pts) == 1:
            nodes.len[n] = 30
    #node dist_out.
    nrch = np.where(nodes.reach_id == reach[r])[0]
    sort_nodes = nrch[np.argsort(nodes.id[nrch])]
    nodes.dist_out[sort_nodes] = np.cumsum(nodes.len[sort_nodes])+base_val

#write data.
swd.discharge_attr_nc(reaches)
swd.write_nc(centerlines, nodes, reaches, sword_fn)
print('DONE')