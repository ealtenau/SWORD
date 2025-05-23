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

outpath = main_dir+'/data/outputs/Reaches_Nodes/'+version+'/'
fn = outpath+'netcdf/'+region.lower()+'_sword_'+version+'.nc'

# read originial data.
data = nc.Dataset(fn)
unq_nodes = data.groups['nodes'].variables['node_id'][:]
node_type = np.array([int(str(rch)[-1]) for rch in unq_nodes])

nodes = gp.GeoDataFrame([
    np.array(data.groups['nodes'].variables['x'][:]),
    np.array(data.groups['nodes'].variables['y'][:]),
    np.array(data.groups['nodes'].variables['node_id'][:]),
    np.array(data.groups['nodes'].variables['node_length'][:]),
    np.array(data.groups['nodes'].variables['reach_id'][:]),
    np.array(data.groups['nodes'].variables['wse'][:]),
    np.array(data.groups['nodes'].variables['wse_var'][:]),
    np.array(data.groups['nodes'].variables['width'][:]),
    np.array(data.groups['nodes'].variables['width_var'][:]),
    np.array(data.groups['nodes'].variables['facc'][:]),
    np.array(data.groups['nodes'].variables['n_chan_max'][:]),
    np.array(data.groups['nodes'].variables['n_chan_mod'][:]),
    np.array(data.groups['nodes'].variables['obstr_type'][:]),
    np.array(data.groups['nodes'].variables['grod_id'][:]),
    np.array(data.groups['nodes'].variables['hfalls_id'][:]),
    np.array(data.groups['nodes'].variables['dist_out'][:]),
    np.array(data.groups['nodes'].variables['lakeflag'][:]),
    np.array(data.groups['nodes'].variables['max_width'][:]),
    np.array(data.groups['nodes'].variables['manual_add'][:]),
    np.array(data.groups['nodes'].variables['meander_length'][:]),
    np.array(data.groups['nodes'].variables['sinuosity'][:]),
    node_type,
    np.array(data.groups['nodes'].variables['river_name'][:]),
    np.array(data.groups['nodes'].variables['edit_flag'][:]),
    np.array(data.groups['nodes'].variables['trib_flag'][:]),
]).T

#rename columns.
nodes.rename(
    columns={
        0:"x",
        1:"y",
        2:"node_id",
        3:"node_len",
        4:"reach_id",
        5:"wse",
        6:"wse_var",
        7:"width",
        8:"width_var",
        9:"facc",
        10:"n_chan_max",
        11:"n_chan_mod",
        12:"obstr_type",
        13:"grod_id",
        14:"hfalls_id",
        15:"dist_out",
        16:"lakeflag",
        17:"max_width",
        18:"manual_add",
        19:"meand_len",
        20:"sinuosity",
        21:"type",
        22:"river_name",
        23:"edit_flag",
        24:"trib_flag",
        },inplace=True)

nodes = nodes.apply(pd.to_numeric, errors='ignore') # nodes.dtypes
geom = gp.GeoSeries(map(Point, zip(data.groups['nodes'].variables['x'][:], data.groups['nodes'].variables['y'][:])))
nodes['geometry'] = geom
nodes = gp.GeoDataFrame(nodes)
nodes.set_geometry(col='geometry')
nodes = nodes.set_crs(4326, allow_override=True)

print('Writing GeoPackage File')

#write geopackage (continental scale)
if os.path.exists(outpath+'gpkg/'):
    outgpkg = outpath + 'gpkg/' + region.lower() + '_sword_nodes_' + version + '.gpkg'
else:
    os.makedirs(outpath+'gpkg/')
    outgpkg = outpath + 'gpkg/' + region.lower() + '_sword_nodes_' + version + '.gpkg'

start = time.time()
nodes.to_file(outgpkg, driver='GPKG', layer='nodes')
end = time.time()
print('Finished GPKG in: '+str(np.round((end-start)/60,2))+' min')

#write as shapefile per level2 basin.
print('Writing Shapefiles')
if os.path.exists(outpath + 'shp/' + region + '/'):
    shpdir = outpath + 'shp/' + region + '/'
else:
    os.makedirs(outpath + 'shp/' + region + '/')
    shpdir = outpath + 'shp/' + region + '/'

start = time.time()
level2 = np.array([int(str(n)[0:2]) for n in nodes['node_id']])
unq_l2 = np.unique(level2)
nodes_cp = nodes.copy(); nodes_cp['level2'] = level2
for lvl in list(range(len(unq_l2))):
    print(unq_l2[lvl])
    outshp = shpdir + region.lower() + "_sword_nodes_hb" + str(unq_l2[lvl]) + "_" + version + '.shp'
    subset = nodes_cp[nodes_cp['level2'] == unq_l2[lvl]]
    subset = subset.drop(columns=['level2'])
    subset.to_file(outshp)
    del(subset)
end = time.time()
print('Finished SHPs in: '+str(np.round((end-start)/60,2))+' min')
