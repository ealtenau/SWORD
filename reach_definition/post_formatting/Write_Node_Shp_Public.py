# -*- coding: utf-8 -*-
"""
Created on Wed Oct 09 12:56:58 2019

@author: ealtenau
"""
from __future__ import division
import os
import sys
from osgeo import ogr
from osgeo import osr
import numpy as np
import numpy.ma as ma
import time
import netCDF4 as nc

###############################################################################

class Object(object):
    """
    FUNCTION:
        Creates class object to assign attributes to.
    """
    pass

###############################################################################

def read_nodes(filename):

    """
    FUNCTION:
        Reads in node location and attribute information from the SWORD netcdf.

    INPUTS
        filename -- Path to SWORD netcdf file.

    OUTPUTS
        nodes -- Object containing node location and attribute information.
    """

    nodes = Object()
    data = nc.Dataset(filename)

    nodes.id = data.groups['nodes'].variables['node_id'][:]
    nodes.cl_id = data.groups['nodes'].variables['cl_ids'][:]
    nodes.x = data.groups['nodes'].variables['x'][:]
    nodes.y = data.groups['nodes'].variables['y'][:]
    nodes.len = data.groups['nodes'].variables['node_length'][:]
    nodes.wse = data.groups['nodes'].variables['wse'][:]
    nodes.wse_var = data.groups['nodes'].variables['wse_var'][:]
    nodes.wth = data.groups['nodes'].variables['width'][:]
    nodes.wth_var = data.groups['nodes'].variables['width_var'][:]
    nodes.grod = data.groups['nodes'].variables['obstr_type'][:]
    nodes.nchan_max = data.groups['nodes'].variables['n_chan_max'][:]
    nodes.nchan_mod = data.groups['nodes'].variables['n_chan_mod'][:]
    nodes.dist_out = data.groups['nodes'].variables['dist_out'][:]
    nodes.reach_id = data.groups['nodes'].variables['reach_id'][:]
    nodes.wth_coef = data.groups['nodes'].variables['wth_coef'][:]
    nodes.facc = data.groups['nodes'].variables['facc'][:]
    nodes.grod_fid = data.groups['nodes'].variables['grod_id'][:]
    nodes.hfalls_fid = data.groups['nodes'].variables['hfalls_id'][:]
    nodes.lakeflag = data.groups['nodes'].variables['lakeflag'][:]
    nodes.max_wth = data.groups['nodes'].variables['max_width'][:]
    nodes.manual_add = data.groups['nodes'].variables['manual_add'][:]
    nodes.meand_len = data.groups['nodes'].variables['meander_length'][:]
    nodes.sinuosity = data.groups['nodes'].variables['sinuosity'][:]
    nodes.river_name = data.groups['nodes'].variables['river_name'][:]
    #nodes.lake_id = data.groups['nodes'].variables['lake_id'][:]    


    return nodes

###############################################################################

def write_node_shp(nodes, outfile):

    """
    FUNCTION:
        Outputs node locations and attributes in shapefile format at the
        Pfafstetter level 2 basin scale.

    INPUTS
        nodes -- Object containing lcation and attribute information for
            each node.
        outfile -- Path for shapefile to be written.

    OUTPUTS
        SWORD Node Shapefile -- Point shapefile containing locations and attributes
            for the nodes (~ 200 m spacing) at the level 2 Pfafstetter basin scale.
    """

    driver = ogr.GetDriverByName('ESRI Shapefile')

    fshpout = outfile
    if os.path.exists(fshpout):
        driver.DeleteDataSource(fshpout)
    try:
        dataout = driver.CreateDataSource(fshpout)
    except:
        print('Could not create file ' + fshpout)
        sys.exit(1)

    proj = osr.SpatialReference()
    #proj.ImportFromEPSG(4326) # 4326 = EPSG code for lon/lat WGS84 projection
    wkt_text = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,\
                                                          AUTHORITY["EPSG",7030]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG",6326]],\
                                                          PRIMEM["Greenwich",0,AUTHORITY["EPSG",8901]],UNIT["DMSH",0.0174532925199433,\
                                                          AUTHORITY["EPSG",9108]],AXIS["Lat",NORTH],AXIS["Long",EAST],AUTHORITY["EPSG",4326]]'
    proj.ImportFromWkt(wkt_text)

    layerout = dataout.CreateLayer(outfile[:-4] + '_lay.shp',
                                   proj, geom_type=ogr.wkbPoint)

    # Define pixel attributes
    fieldDef1 = ogr.FieldDefn('x', ogr.OFTReal)
    fieldDef2 = ogr.FieldDefn('y', ogr.OFTReal)
    fieldDef3 = ogr.FieldDefn('node_id', ogr.OFTString)
    fieldDef4 = ogr.FieldDefn('node_len', ogr.OFTReal)
    fieldDef5 = ogr.FieldDefn('reach_id', ogr.OFTString)
    fieldDef6 = ogr.FieldDefn('wse', ogr.OFTReal)
    fieldDef7 = ogr.FieldDefn('wse_var', ogr.OFTReal)
    fieldDef8 = ogr.FieldDefn('width', ogr.OFTReal)
    fieldDef9 = ogr.FieldDefn('wth_var', ogr.OFTReal)
    fieldDef10 = ogr.FieldDefn('n_chan_max', ogr.OFTInteger)
    fieldDef11 = ogr.FieldDefn('n_chan_mod', ogr.OFTInteger)
    fieldDef12 = ogr.FieldDefn('obstr_type', ogr.OFTInteger)
    fieldDef13 = ogr.FieldDefn('grod_id', ogr.OFTInteger)
    fieldDef14 = ogr.FieldDefn('hfalls_id', ogr.OFTInteger)
    fieldDef15 = ogr.FieldDefn('dist_out', ogr.OFTReal)
    fieldDef16 = ogr.FieldDefn('type', ogr.OFTInteger)
    fieldDef17 = ogr.FieldDefn('facc', ogr.OFTReal)
    fieldDef18 = ogr.FieldDefn('lakeflag', ogr.OFTString)
    fieldDef19 = ogr.FieldDefn('max_width', ogr.OFTReal)
    fieldDef20 = ogr.FieldDefn('manual_add', ogr.OFTInteger)
    fieldDef21 = ogr.FieldDefn('meand_len', ogr.OFTReal)
    fieldDef22 = ogr.FieldDefn('sinuosity', ogr.OFTReal)
    #fieldDef23 = ogr.FieldDefn('river_name', ogr.OFTString)
    #fieldDef24 = ogr.FieldDefn('lake_id', ogr.OFTString)

    layerout.CreateField(fieldDef1)
    layerout.CreateField(fieldDef2)
    layerout.CreateField(fieldDef3)
    layerout.CreateField(fieldDef4)
    layerout.CreateField(fieldDef5)
    layerout.CreateField(fieldDef6)
    layerout.CreateField(fieldDef7)
    layerout.CreateField(fieldDef8)
    layerout.CreateField(fieldDef9)
    layerout.CreateField(fieldDef10)
    layerout.CreateField(fieldDef11)
    layerout.CreateField(fieldDef12)
    layerout.CreateField(fieldDef13)
    layerout.CreateField(fieldDef14)
    layerout.CreateField(fieldDef15)
    layerout.CreateField(fieldDef16)
    layerout.CreateField(fieldDef17)
    layerout.CreateField(fieldDef18)
    layerout.CreateField(fieldDef19)
    layerout.CreateField(fieldDef20)
    layerout.CreateField(fieldDef21)
    layerout.CreateField(fieldDef22)
    #layerout.CreateField(fieldDef23)
    #layerout.CreateField(fieldDef24)
    
    floutDefn = layerout.GetLayerDefn()
    # Create feature (point) to store pixel
    feature_out = ogr.Feature(floutDefn)

    for ipix in range(len(nodes.x)):
        # Create Geometry Point with pixel coordinates
        pixel_point = ogr.Geometry(ogr.wkbPoint)
        pixel_point.AddPoint(nodes.x[ipix], nodes.y[ipix])
        # Add the geometry to the feature
        feature_out.SetGeometry(pixel_point)
        # Set feature attributes
        # int() is needed because facc.dtype=float64, needs to be saved with all values
        # whereas lat.dtype=float64, which raise an error
        feature_out.SetField('x', nodes.x[ipix])
        feature_out.SetField('y', nodes.y[ipix])
        feature_out.SetField('node_id', str(int(nodes.id[ipix])))
        feature_out.SetField('node_len', nodes.len[ipix])
        feature_out.SetField('reach_id', str(int(nodes.reach_id[ipix])))
        feature_out.SetField('wse', nodes.wse[ipix])
        feature_out.SetField('wse_var', nodes.wse_var[ipix])
        feature_out.SetField('width', nodes.wth[ipix])
        feature_out.SetField('wth_var', nodes.wth_var[ipix])
        feature_out.SetField('n_chan_max', int(nodes.nchan_max[ipix]))
        feature_out.SetField('n_chan_mod', int(nodes.nchan_mod[ipix]))
        feature_out.SetField('obstr_type', int(nodes.grod[ipix]))
        feature_out.SetField('grod_id', int(nodes.grod_fid[ipix]))
        feature_out.SetField('hfalls_id', int(nodes.hfalls_fid[ipix]))
        feature_out.SetField('dist_out', nodes.dist_out[ipix])
        feature_out.SetField('type', int(nodes.type[ipix]))
        feature_out.SetField('facc', nodes.facc[ipix])
        feature_out.SetField('lakeflag', str(int(nodes.lakeflag[ipix])))
        feature_out.SetField('max_width', nodes.max_wth[ipix])
        feature_out.SetField('manual_add', int(nodes.manual_add[ipix]))
        feature_out.SetField('meand_len', nodes.meand_len[ipix])
        feature_out.SetField('sinuosity', nodes.sinuosity[ipix])
        #feature_out.SetField('river_name', str(nodes.river_name[ipix]))
        #feature_out.SetField('lake_id', str(int(nodes.lake_id[ipix])))
        

        # Add the feature to the layer
        layerout.CreateFeature(feature_out)
        # Delete point geometry
        pixel_point.Destroy()

    # Close feature and shape files
    feature_out.Destroy()
    dataout.Destroy()

    return fshpout

###############################################################################
###############################################################################
###############################################################################

region = 'NA'
version = '_v12'
fn = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/outputs/Reaches_Nodes/netcdf/'+region.lower()+'_sword'+version+'.nc'
node_outpath = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/outputs/Reaches_Nodes_Public/shp/'+region+'/'+region.lower()+'_sword_nodes_'

# read originial data.
nodes = read_nodes(fn)

level2_basins = np.array([np.int(np.str(ind)[0:2]) for ind in nodes.id])
uniq_level2 = np.unique(level2_basins)
#uniq_level2 = np.delete(uniq_level2, 0)
for ind in list(range(len(uniq_level2))):

    print('STARTING BASIN: ' + str(uniq_level2[ind]))

    start = time.time()

    # Define objects to assign attributes.
    subnodes = Object()

    # Subset data.
    level2 = np.where(level2_basins == uniq_level2[ind])[0]

    subnodes.id = nodes.id[level2]
    subnodes.cl_id = nodes.cl_id[:,level2]
    subnodes.x = nodes.x[level2]
    subnodes.y = nodes.y[level2]
    subnodes.len = nodes.len[level2]
    subnodes.wse = nodes.wse[level2]
    subnodes.wse_var = nodes.wse_var[level2]
    subnodes.wth = nodes.wth[level2]
    subnodes.wth_var = nodes.wth_var[level2]
    subnodes.grod = nodes.grod[level2]
    subnodes.grod_fid = nodes.grod_fid[level2]
    subnodes.hfalls_fid = nodes.hfalls_fid[level2]
    subnodes.nchan_max = nodes.nchan_max[level2]
    subnodes.nchan_mod = nodes.nchan_mod[level2]
    subnodes.dist_out = nodes.dist_out[level2]
    subnodes.reach_id = nodes.reach_id[level2]
    subnodes.facc = nodes.facc[level2]
    subnodes.lakeflag = nodes.lakeflag[level2]
    subnodes.max_wth = nodes.max_wth[level2]
    subnodes.manual_add = nodes.manual_add[level2]
    subnodes.meand_len = nodes.meand_len[level2]
    subnodes.sinuosity = nodes.sinuosity[level2]
    #subnodes.river_name = nodes.river_name[level2]
    #subnodes.lake_id = nodes.lake_id[level2]
    
    Type = np.zeros(len(subnodes.reach_id))
    for idx in list(range(len(subnodes.reach_id))):
        Type[idx] = np.int(np.str(subnodes.reach_id[idx])[10:11])
    subnodes.type = Type

    nan = np.isnan(subnodes.sinuosity)
    subnodes.sinuosity[nan] = -9999
    
    if ma.is_masked(subnodes.sinuosity):
        subnodes.sinuosity = subnodes.sinuosity.data

    node_outfile = node_outpath + 'hb' + str(uniq_level2[ind]) + '_v2.shp'
    write_node_shp(subnodes, node_outfile)

    end = time.time()
    print('Time to Write Basin: ' + str((end-start)/60) + 'min')
