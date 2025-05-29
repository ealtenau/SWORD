# -*- coding: utf-8 -*-
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
parser.add_argument("csv", help="csv file of reaches to delete", type = str)
args = parser.parse_args()

region = args.region
version = args.version

sword = SWORD(main_dir, region, version)

new_rch_id = np.repeat('NaN', len(sword.centerlines.x))
new_up_nghs = np.repeat('NaN', len(sword.centerlines.x))
new_dn_nghs = np.repeat('NaN', len(sword.centerlines.x))
reverse_topo = np.repeat('NaN', len(sword.centerlines.x))

level2_basins = np.array([int(str(ind)[0:2]) for ind in sword.centerlines.reach_id[0,:]])
unq_l2 = np.unique(level2_basins)

for ind in list(range(len(unq_l2))):
    print(unq_l2[ind])
    pts = np.where(level2_basins == unq_l2[ind])[0]
    points = gp.GeoDataFrame([
        sword.centerlines.x[pts],
        sword.centerlines.y[pts],
        sword.centerlines.cl_id[pts],
        sword.centerlines.reach_id[0,pts],
        sword.centerlines.node_id[0,pts],
        new_rch_id[pts],
        new_up_nghs[pts],
        new_dn_nghs[pts],
        reverse_topo[pts],
    ]).T

    #rename columns.
    points.rename(
        columns={
            0:"x",
            1:"y",
            2:"cl_id",
            3:"reach_id",
            4:"node_id",
            5:"new_rch_id",
            6:"new_up_nghs",
            7:"new_dn_nghs",
            8:"reverse_topo",
        },inplace=True)

    points = points.apply(pd.to_numeric, errors='ignore') # points.dtypes
    geom = gp.GeoSeries(map(Point, zip(sword.centerlines.x[pts], sword.centerlines.y[pts])))
    points['geometry'] = geom
    points = gp.GeoDataFrame(points)
    points.set_geometry(col='geometry')
    points = points.set_crs(4326, allow_override=True)

    outgpkg = sword.paths['pts_gpkg_dir']+'hb'+str(unq_l2[ind])+'_centerlines_'+version+'.gpkg'
    points.to_file(outgpkg, driver='GPKG', layer='points')

