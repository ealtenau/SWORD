# -*- coding: utf-8 -*-
"""
Created on Wed Oct 09 12:56:58 2019

@author: ealtenau
"""
from __future__ import division
import numpy as np
import time
import netCDF4 as nc
import geopandas as gp
from shapely.geometry import Point
import pandas as pd

###############################################################################
###############################################################################
###############################################################################

region = 'NA'
version = 'v13'
outdir = '/Users/ealteanau/Documents/SWORD_Dev/outputs/'
outpath = outdir+version+'/'
fn = outpath+'netcdf/'+region.lower()+'_sword_'+version+'.nc'

# read originial data.
data = nc.Dataset(fn)
unq_nodes = data.groups['nodes'].variables['node_id'][:]
node_type = [int(str(rch)[-1]) for rch in unq_nodes]

nodes = gp.GeoDataFrame([
    data.groups['nodes'].variables['x'][:],
    data.groups['nodes'].variables['y'][:],
    data.groups['nodes'].variables['node_id'][:],
    data.groups['nodes'].variables['node_length'][:],
    data.groups['nodes'].variables['reach_id'][:],
    data.groups['nodes'].variables['wse'][:],
    data.groups['nodes'].variables['wse_var'][:],
    data.groups['nodes'].variables['width'][:],
    data.groups['nodes'].variables['width_var'][:],
    data.groups['nodes'].variables['facc'][:],
    data.groups['nodes'].variables['n_chan_max'][:],
    data.groups['nodes'].variables['n_chan_mod'][:],
    data.groups['nodes'].variables['obstr_type'][:],
    data.groups['nodes'].variables['grod_id'][:],
    data.groups['nodes'].variables['hfalls_id'][:],
    data.groups['nodes'].variables['dist_out'][:],
    data.groups['nodes'].variables['lakeflag'][:],
    data.groups['nodes'].variables['max_width'][:],
    data.groups['nodes'].variables['manual_add'][:],
    data.groups['nodes'].variables['meander_length'][:],
    data.groups['nodes'].variables['sinuosity'][:],
    node_type,
    data.groups['nodes'].variables['river_name'][:],
    data.groups['nodes'].variables['edit_flag'][:],
    # data.groups['nodes'].variables['lake_id'][:],
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
        23:"edit_flag"
        # 24:"lake_id",
        },inplace=True)

nodes = nodes.apply(pd.to_numeric, errors='ignore') # nodes.dtypes
geom = gp.GeoSeries(map(Point, zip(data.groups['nodes'].variables['x'][:], data.groups['nodes'].variables['y'][:])))
nodes['geometry'] = geom
nodes.set_geometry(col='geometry', inplace=True)
nodes = nodes.set_crs(4326, allow_override=True)

print('Writing GeoPackage File')
start = time.time()
#write geopackage (continental scale)
outgpkg = outpath + 'gpkg/' + region.lower() + '_sword_nodes_' + version + '.gpkg'
nodes.to_file(outgpkg, driver='GPKG', layer='reaches')
end = time.time()
print('Finished GPKG in: '+str(np.round((end-start)/60,2))+' min')

#write as shapefile per level2 basin.
print('Writing Shapefiles')
start = time.time()
level2 = [int(str(n)[0:2]) for n in nodes['node_id']]
unq_l2 = np.unique(level2)
nodes_cp = nodes.copy(); nodes_cp['level2'] = level2
for lvl in list(range(len(unq_l2))):
    outshp = outpath + 'shp/' + region + '/' + region.lower() + "_sword_nodes_hb" + str(unq_l2[lvl]) + "_" + version + '.shp'
    subset = nodes_cp[nodes_cp['level2'] == unq_l2[lvl]]
    subset = subset.drop(columns=['level2'])
    subset.to_file(outshp)
    del(subset)
end = time.time()
print('Finished SHPs in: '+str(np.round((end-start)/60,2))+' min')
