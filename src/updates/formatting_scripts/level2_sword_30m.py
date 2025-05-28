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
import src.updates.sword_utils as swd

parser = argparse.ArgumentParser()
parser.add_argument("region", help="continental region", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("csv", help="csv file of reaches to delete", type = str)
args = parser.parse_args()

region = args.region
version = args.version

paths = swd.prepare_paths(main_dir, region, version)
sword_fn = paths['geom_dir']+paths['geom_fn']

#read sword
centerlines, nodes, reaches = swd.read_nc(sword_fn)

new_rch_id = np.repeat('NaN', len(centerlines.x))
new_up_nghs = np.repeat('NaN', len(centerlines.x))
new_dn_nghs = np.repeat('NaN', len(centerlines.x))
reverse_topo = np.repeat('NaN', len(centerlines.x))

level2_basins = np.array([int(str(ind)[0:2]) for ind in centerlines.reach_id[0,:]])
unq_l2 = np.unique(level2_basins)

for ind in list(range(len(unq_l2))):
    print(unq_l2[ind])
    pts = np.where(level2_basins == unq_l2[ind])[0]
    nodes = gp.GeoDataFrame([
        centerlines.x[pts],
        centerlines.y[pts],
        centerlines.cl_id[pts],
        centerlines.reach_id[0,pts],
        centerlines.node_id[0,pts],
        new_rch_id[pts],
        new_up_nghs[pts],
        new_dn_nghs[pts],
        reverse_topo[pts],
    ]).T

    #rename columns.
    nodes.rename(
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

    nodes = nodes.apply(pd.to_numeric, errors='ignore') # nodes.dtypes
    geom = gp.GeoSeries(map(Point, zip(centerlines.x[pts], centerlines.y[pts])))
    nodes['geometry'] = geom
    nodes = gp.GeoDataFrame(nodes)
    nodes.set_geometry(col='geometry')
    nodes = nodes.set_crs(4326, allow_override=True)

    outgpkg = paths['pts_gpkg_dir']+'hb'+str(unq_l2[ind])+'_centerlines_'+version+'.gpkg'
    nodes.to_file(outgpkg, driver='GPKG', layer='nodes')

