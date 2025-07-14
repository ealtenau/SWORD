# -*- coding: utf-8 -*-
"""
Delta Utilities (delta_utils.py)
=======================================

Utilities for attaching auxillary data to delta 
polyline and node shapefiles. These tools are 
specific to the delta shapefiles provided by 
Dr. Paola Passalacqua's lab at UT Austin 
(https://sites.google.com/site/passalacquagroup/home). 

"""

from __future__ import division
import os
import sys
main_dir = os.getcwd()
import time
import numpy as np
from scipy import spatial as sp
import glob
import geopandas as gp
import netCDF4 as nc
from shapely.geometry import Point
import pandas as pd
from osgeo import ogr, gdal
from statistics import mode
from itertools import chain
import src.updates.geo_utils as geo 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") #if code stops working may need to comment out to check warnings. 

###############################################################################

def format_topo_attributes(rch_id_up, rch_id_dn, n_rch_up, n_rch_dn):
    """
    Reformats string topological attributes into array format 
    for writing netCDF files.

    Parameters
    ----------
    rch_id_up: numpy.array()
        1-D array of upstream reach IDs in string format.
    rch_id_dn: numpy.array()
        1-D array of downstream reach IDs in string format.
    n_rch_up: numpy.array()
        Number of upstream reaches.
    n_rch_dn: numpy.array()
        Number of downstream reaches.

    Returns
    -------
    id_up_arr: numpy.array() [4,number of reaches]
        Array containing upstream reach IDs in separate rows. 
        Up to 4 upstream reaches. 
    id_dn_arr: numpy.array() [4,number of reaches]
        Array containing downstream reach IDs in separate rows. 
        Up to 4 downstream reaches. 

    """

    id_up_arr = np.zeros((4,len(n_rch_up)), dtype=int)
    id_dn_arr = np.zeros((4,len(n_rch_up)), dtype=int)
    for pt in list(range(len(n_rch_up))):
        if n_rch_up[pt] > 0:
            up_ids = np.array(rch_id_up[pt].split())
            id_up_arr[0:len(up_ids),pt] = up_ids
        if n_rch_dn[pt] > 0:
            dn_ids = np.array(rch_id_dn[pt].split())  
            id_dn_arr[0:len(dn_ids),pt] = dn_ids

    return id_up_arr, id_dn_arr

###############################################################################

def reverse_indexes(reaches, index):
    """
    Reverses indexes from the default upstream-to-downstream 
    order to downstream-to-upstream order.

    Parameters
    ----------
    reaches: numpy.array()
        Reach IDs. 
    index: numpy.array() 
        Unique ID for each point/coordinate.

    Returns
    -------
    new_indexes: numpy.array() 
        Reversed IDs for each point/coordinate. 
       
    """

    new_index = np.zeros(len(reaches))
    unq_rchs = np.unique(reaches)
    for r in list(range(len(unq_rchs))):
        rch = np.where(reaches == unq_rchs[r])[0]
        new_index[rch] = index[rch][::-1]

    return new_index

###############################################################################

def write_data_nc(data_obj, outfile):
    """
    Writes delta attributes as a netCDF file.

    Parameters
    ----------
    data_obj: obj
        Object containing delta centerline data and attached 
        auxillary attributes. 
    outfile: str 
        Filepath to write netCDF file. 

    Returns
    -------
    None. 
       
    """
    
    # global attributes
    root_grp = nc.Dataset(outfile, 'w', format='NETCDF4')
    root_grp.delta_file = data_obj.file
    root_grp.x_min = np.min(data_obj.x)
    root_grp.x_max = np.max(data_obj.x)
    root_grp.y_min = np.min(data_obj.x)
    root_grp.y_max = np.max(data_obj.x)
    root_grp.production_date = time.strftime("%d-%b-%Y %H:%M:%S", time.gmtime()) #utc time

    # subgroups
    cl_grp = root_grp.createGroup('centerlines')

    # dimensions
    cl_grp.createDimension('num_points', len(data_obj.x))
    cl_grp.createDimension('orbit', 200)
    cl_grp.createDimension('nghs', 4)

    ### variables and units
    # centerline variables
    x = cl_grp.createVariable('x', 'f8', ('num_points',), fill_value=-9999.)
    x.units = 'degrees east'
    y = cl_grp.createVariable('y', 'f8', ('num_points',), fill_value=-9999.)
    y.units = 'degrees north'
    segID = cl_grp.createVariable('segID', 'i8', ('num_points',), fill_value=-9999.)
    segInd = cl_grp.createVariable('segInd', 'i8', ('num_points',), fill_value=-9999.)
    #global attributes
    lakeflag = cl_grp.createVariable('lakeflag', 'i4', ('num_points',), fill_value=-9999.)
    deltas = cl_grp.createVariable('deltaflag', 'i4', ('num_points',), fill_value=-9999.)
    grand = cl_grp.createVariable('grand_id', 'i4', ('num_points',), fill_value=-9999.)
    grod_type = cl_grp.createVariable('grod_id', 'i4', ('num_points',), fill_value=-9999.)
    grod_id = cl_grp.createVariable('grod_fid', 'i8', ('num_points',), fill_value=-9999.)
    hfalls_id = cl_grp.createVariable('hfalls_fid', 'i8', ('num_points',), fill_value=-9999.)
    basin = cl_grp.createVariable('basin_code', 'i8', ('num_points',), fill_value=-9999.)
    swot_obs = cl_grp.createVariable('number_obs', 'i4', ('num_points',), fill_value=-9999.)
    orbits = cl_grp.createVariable('orbits', 'i4', ('orbit','num_points'), fill_value=-9999.)
    lake_id = cl_grp.createVariable('lake_id', 'i8', ('num_points',), fill_value=-9999.)
    #regional attributes
    cl_id = cl_grp.createVariable('cl_id', 'i8', ('num_points',), fill_value=-9999.)
    east = cl_grp.createVariable('easting', 'f8', ('num_points',), fill_value=-9999.)
    north = cl_grp.createVariable('northing', 'f8', ('num_points',), fill_value=-9999.)
    dist = cl_grp.createVariable('segDist', 'f8', ('num_points',), fill_value=-9999.)
    length = cl_grp.createVariable('segLen', 'f8', ('num_points',), fill_value=-9999.)
    wth = cl_grp.createVariable('p_width', 'f8', ('num_points',), fill_value=-9999.)
    wse = cl_grp.createVariable('p_height', 'f8', ('num_points',), fill_value=-9999.)
    facc = cl_grp.createVariable('flowacc', 'f8', ('num_points',), fill_value=-9999.)
    nchan = cl_grp.createVariable('nchan', 'i4', ('num_points',), fill_value=-9999.)
    manual = cl_grp.createVariable('manual_add', 'i4', ('num_points',), fill_value=-9999.)
    eps = cl_grp.createVariable('endpoints', 'i4', ('num_points',), fill_value=-9999.)
    tile = cl_grp.createVariable('mh_tile', 'S7', ('num_points',), fill_value=-9999.)
    tile._Encoding = 'ascii'
    #delta specific attributes
    wth_var = cl_grp.createVariable('wth_var', 'f8', ('num_points',), fill_value=-9999.)
    max_width = cl_grp.createVariable('max_width', 'f8', ('num_points',), fill_value=-9999.)
    rch_id_up = cl_grp.createVariable('rch_id_up', 'i8', ('nghs','num_points',), fill_value=-9999.)
    rch_id_dn = cl_grp.createVariable('rch_id_down', 'i8', ('nghs','num_points',), fill_value=-9999.)
    n_rch_up = cl_grp.createVariable('n_rch_up', 'i4', ('num_points',), fill_value=-9999.)
    n_rch_dn = cl_grp.createVariable('n_rch_down', 'i4', ('num_points',), fill_value=-9999.)
    node_id = cl_grp.createVariable('segNode', 'i8', ('num_points',), fill_value=-9999.)
    sinuosity = cl_grp.createVariable('sinuosity', 'f8', ('num_points',), fill_value=-9999.)
        
    # saving data
    # centerline data
    x[:] = data_obj.x
    y[:] = data_obj.y
    segID[:] = data_obj.reach_id_R
    segInd[:] = data_obj.index
    lakeflag[:] = data_obj.lakeflag
    deltas[:] = data_obj.deltaflag
    grand[:] = data_obj.grand
    grod_type[:] = data_obj.obstr_type
    grod_id[:] = data_obj.grod_id
    hfalls_id[:] = data_obj.hfalls_id
    basin[:] = data_obj.basins_filt
    swot_obs[:] = data_obj.swot_obs
    orbits[:,:] = data_obj.swot_orbit
    lake_id[:] = data_obj.lakeid
    cl_id[:] = data_obj.cl_id
    east[:] = data_obj.east
    north[:] = data_obj.north
    dist[:] = data_obj.dist
    length[:] = data_obj.len
    wth[:] = data_obj.width
    wse[:] = data_obj.wse
    facc[:] = data_obj.facc
    nchan[:] = data_obj.nchan
    manual[:] = data_obj.manual_add
    eps[:] = data_obj.ends
    tile[:] = data_obj.mh_tile
    wth_var[:] = data_obj.wth_var
    max_width[:] = data_obj.max_width
    rch_id_up[:,:] = data_obj.fmt_rch_id_up
    rch_id_dn[:,:] = data_obj.fmt_rch_id_dn
    n_rch_up[:] = data_obj.n_rch_up
    n_rch_dn[:] = data_obj.n_rch_down
    node_id[:] = data_obj.node_id
    sinuosity[:] = data_obj.sinuosity
        
    root_grp.close()

###############################################################################

def read_delta(filename):
    """
    Reads delta netCDF file and stores attributes as a class object.

    Parmeters
    ---------
    filename: str
        The directory to the SWORD netCDF file. 
    
    Returns
    -------
    data: obj 
        Object containing attributes associated with the delta 
        centerlines. 
        
        Attributes: type [dimension]
        ----------------------------
        cl_id: numpy.array() [number of points]
            Unique ID associated with each centerline point. 
        x: numpy.array() [number of points]
            Longitude (WGS 84, EPSG:4326). 
        y: numpy.array() [number of points]
            Latitude (WGS 84, EPSG:4326).
        seg: numpy.array() [number of points]
            Unique ID associated with a river reach/segment. 
        seg_node: numpy.array() [number of points]
            Unique ID associated with a river node 
            (~200 m intervals within a river reach/segment).
        ind: numpy.array() [number of points]
            Unique ID associated with each river reach/segment.
        dist: numpy.array() [number of points]
            Cumulative distance along a river reach/segment (meters).
        len: numpy.array() [number of points]
            Reach/segment length (meters).
        wse: numpy.array() [number of points]
            Water surface elevation (meters). 
        wth: numpy.array() [number of points]
            Width (meters).
        wth_var: numpy.array() [number of points]
            Width variance (squared meters).
        grod: numpy.array() [number of points]
            Type of obstruction for each node based on GROD and
            HydroFALLS databases. Obstr_type values: 
            0 - No Dam, 
            1 - Dam, 
            2 - Lock, 
            3 - Low Permeable Dam, 
            4 - Waterfall.
        grod_fid: numpy.array() [number of points]
            GROD database ID. 
        hfalls_fid: numpy.array() [number of points]
            HydroFALLS database ID. 
        facc: numpy.array() [number of points]
            Flow accumulation (squared kilometers).
        lakeflag: numpy.array() [number of points]
            GRWL water body identifier for each node: 
            0 - river, 
            1 - lake/reservoir, 
            2 - canal , 
            3 - tidally influenced river.
        max_wth: numpy.array() [number of points]
            Maximum width value across the channel that
            includes any island and bar areas (meters).
        manual_add: numpy.array() [number of points]
            Binary flag indicating whether the node was manually added. 
            0 - Not manually added. 
            1 - Manually added. 
        n_rch_up: numpy.array() [number of points]
            Number of upstream neighbors.
        n_rch_down: numpy.array() [number of points]
            Number of downstream neighbors.
        rch_id_up: numpy.array() [4, number of points]
            Reach/segment IDs of upstream neighbors (4 maximum).
        rch_id_down: numpy.array() [4, number of points]
            Reach/segment IDs of downstream neighbors (4 maximum).
        sinuosity: numpy.array() [number of points]
            The total length of the segment node divided by the 
            Euclidean distance between the segment node end 
            points (meters).
 
    """
        
    #read file. 
    nc_data = nc.Dataset(filename)
    data = geo.Object()
    #assign attributes. 
    data.lon = np.array(nc_data['/centerlines/x'][:])
    data.lat = np.array(nc_data['/centerlines/y'][:])
    data.cl_id = np.array(nc_data['/centerlines/cl_id'][:])
    data.x = np.array(nc_data['/centerlines/easting'][:])
    data.y = np.array(nc_data['/centerlines/northing'][:])
    data.wth = np.array(nc_data['/centerlines/p_width'][:])
    data.elv = np.array(nc_data['/centerlines/p_height'][:])
    data.facc = np.array(nc_data['/centerlines/flowacc'][:])
    data.nchan = np.array(nc_data['/centerlines/nchan'][:])
    data.manual = np.array(nc_data['/centerlines/manual_add'][:])
    data.eps = np.array(nc_data['/centerlines/endpoints'][:])
    data.lake = np.array(nc_data['/centerlines/lakeflag'][:])
    data.delta = np.array(nc_data['/centerlines/deltaflag'][:])
    data.grand = np.array(nc_data['/centerlines/grand_id'][:])
    data.grod = np.array(nc_data['/centerlines/grod_id'][:])
    data.grod_fid = np.array(nc_data['/centerlines/grod_fid'][:])
    data.hfalls_fid = np.array(nc_data['/centerlines/hfalls_fid'][:])
    data.basins = np.array(nc_data['/centerlines/basin_code'][:])
    data.num_obs = np.array(nc_data['/centerlines/number_obs'][:])
    data.orbits = np.array(nc_data['/centerlines/orbits'][:])
    data.lake_id = np.array(nc_data['/centerlines/lake_id'][:])
    data.seg = np.array(nc_data['/centerlines/segID'][:])
    data.ind = np.array(nc_data['/centerlines/segInd'][:])
    data.dist = np.array(nc_data['/centerlines/segDist'][:])
    data.len = np.array(nc_data['/centerlines/segLen'][:])
    data.wth_var = np.array(nc_data['/centerlines/wth_var'][:])
    data.max_wth = np.array(nc_data['/centerlines/max_width'][:])
    data.n_rch_up = np.array(nc_data['/centerlines/n_rch_up'][:])
    data.n_rch_down = np.array(nc_data['/centerlines/n_rch_down'][:])
    data.rch_id_up = np.array(nc_data['/centerlines/rch_id_up'][:])
    data.rch_id_down = np.array(nc_data['/centerlines/rch_id_down'][:])
    data.sinuosity = np.array(nc_data['/centerlines/sinuosity'][:])
    data.seg_node = np.array(nc_data['/centerlines/segNode'][:])
    return data

###############################################################################
        
def cut_delta_segments(delta_cls, thresh):
    """
    Cuts delta segments into ~10 km segments if greater than 
    a specified threshold length.

    Parameters
    ----------
    delta_cls: obj
        Object containing delta centerline attributes and location.  
    thresh: float
        Length in meters that indicates if a segment should be 
        cut. 

    New Attributes
    --------------
    delta_cls.new_len: numpy.array()
        Updated segment lengths (meters).
    delta_cls.new_seg: numpy.array() 
        Updated segment IDs. 
    delta_cls.new_rch_id_up: numpy.array()
        Updated upstream segment neighbor IDs. 
    delta_cls.new_rch_id_down: numpy.array() 
        Updated downstream segment neighbor IDs.
    delta_cls.new_n_rch_up: numpy.array()
        Updated number of upstream neighbors.
    delta_cls.new_n_rch_down: numpy.array()
        Updated number of downstream neighbors.
    
    Returns
    -------
    None.

    """

    cnt = np.max(delta_cls.seg)+1
    long = np.where(delta_cls.len > thresh)[0]
    cut_rchs = np.unique(delta_cls.seg[long])
    delta_cls.new_len = np.copy(delta_cls.len)
    delta_cls.new_seg = np.copy(delta_cls.seg)
    delta_cls.new_rch_id_up = np.copy(delta_cls.rch_id_up)
    delta_cls.new_rch_id_down = np.copy(delta_cls.rch_id_down)
    delta_cls.new_n_rch_up = np.copy(delta_cls.n_rch_up)
    delta_cls.new_n_rch_down = np.copy(delta_cls.n_rch_down)
    check_topo_down = []; check_topo_up = []
    if len(cut_rchs) == 0:
        print('----> No long segments')
    else:
        for ind in list(range(len(cut_rchs))):
            #Finding current reach id and length.
            rch = np.where(delta_cls.seg == cut_rchs[ind])[0]
            sort_rch = rch[np.argsort(delta_cls.ind[rch])]
            distance = delta_cls.dist[sort_rch]
            new_rch_bounds = np.floor(distance/10000) # 10 km for segment lengths. 
                
            #filter bounds to be at node boundaries. 
            unq_bnds = np.unique(new_rch_bounds)
            for bnd in list(range(len(unq_bnds))):
                b = np.where(new_rch_bounds == unq_bnds[bnd])[0]
                if len(b) < 35:
                    new_rch_bounds[b] = unq_bnds[bnd]-1
                    continue
                #filtering end nodes to be one reach value. 
                max_node = delta_cls.seg_node[sort_rch[b][-1]]
                mx_node_pts = np.where(delta_cls.seg_node[sort_rch] == max_node)[0]
                mx_num_nodes = np.unique(new_rch_bounds[mx_node_pts])
                if len(mx_num_nodes) > 1:
                    new_rch_bounds[mx_node_pts] = unq_bnds[bnd]
                min_node = delta_cls.seg_node[sort_rch[b][0]]
                mn_node_pts = np.where(delta_cls.seg_node[sort_rch] == min_node)[0]
                mn_num_nodes = np.unique(new_rch_bounds[mn_node_pts])
                if len(mn_num_nodes) > 1:
                    new_rch_bounds[mn_node_pts] = unq_bnds[bnd]-1

            #calculate new reach boundary lengths
            #assign new reach ids. 
            unq_bnds = np.unique(new_rch_bounds)
            for bnd in list(range(len(unq_bnds))):
                b = np.where(new_rch_bounds == unq_bnds[bnd])[0]
                diff = np.diff(delta_cls.dist[sort_rch[b]])
                leng = np.max(np.cumsum(diff))          
                delta_cls.new_len[sort_rch[b]] = leng
                delta_cls.new_seg[sort_rch[b]] = cnt 
                ### topology 
                #first reach in breaks.
                if bnd == 0:
                    #zero out existing topology
                    delta_cls.new_rch_id_up[:,sort_rch[b]] = 0
                    #fill in new topology 
                    delta_cls.new_rch_id_up[0,sort_rch[b]] = cnt+1
                    delta_cls.new_n_rch_up[sort_rch[b]] = 1
                    check_topo_down.append(cnt)
                #last reach in breaks.
                elif bnd == len(unq_bnds)-1:
                    #zero out existing topology
                    delta_cls.new_rch_id_down[:,sort_rch[b]] = 0
                    #fill in new topology 
                    delta_cls.new_rch_id_down[0,sort_rch[b]] = cnt-1
                    delta_cls.new_n_rch_down[sort_rch[b]] = 1
                    check_topo_up.append(cnt)
                #middle reaches in breaks.
                else:
                    #zero out existing topology
                    delta_cls.new_rch_id_up[:,sort_rch[b]] = 0
                    delta_cls.new_rch_id_down[:,sort_rch[b]] = 0
                    #fill in new topology
                    delta_cls.new_rch_id_up[0,sort_rch[b]] = cnt+1
                    delta_cls.new_n_rch_up[sort_rch[b]] = 1
                    delta_cls.new_rch_id_down[0,sort_rch[b]] = cnt-1
                    delta_cls.new_n_rch_down[sort_rch[b]] = 1
                #update counter. 
                cnt = cnt+1 
        
        #check and fix topology if needed.  
        unq_new = np.unique(delta_cls.new_seg)
        for s in list(range(len(unq_new))):
            seg = np.where(delta_cls.new_seg == unq_new[s])[0]
            #upstream check. 
            up_segs = np.unique(delta_cls.new_rch_id_up[:,seg])
            up_segs = up_segs[up_segs>0]
            for us in list(range(len(up_segs))):
                if up_segs[us] not in delta_cls.new_seg:
                    old_seg = np.where(delta_cls.seg == up_segs[us])[0]
                    # if len(old_seg) == 0:
                    #     rmv = np.where(delta_cls.new_rch_id_up[:,seg] == up_segs[us])[0]
                    #     delta_cls.new_rch_id_up[rmv,seg] = 0
                    # else:
                    mn = np.where(delta_cls.ind[old_seg] == np.min(delta_cls.ind[old_seg]))[0]
                    replace_seg = delta_cls.new_seg[old_seg[mn]]
                    update = np.where(delta_cls.new_rch_id_up[:,seg] == up_segs[us])[0]
                    delta_cls.new_rch_id_up[update,seg] = replace_seg
            #downstream neighbor fix. 
            dn_segs = np.unique(delta_cls.new_rch_id_down[:,seg])
            dn_segs = dn_segs[dn_segs>0]
            for ds in list(range(len(dn_segs))):
                if dn_segs[ds] not in delta_cls.new_seg:
                    old_seg = np.where(delta_cls.seg == dn_segs[ds])[0]
                    # if len(old_seg) == 0:
                    #     rmv = np.where(delta_cls.new_rch_id_down[:,seg] == dn_segs[ds])[0]
                    #     delta_cls.new_rch_id_down[rmv,seg] = 0
                    # else:
                    mx = np.where(delta_cls.ind[old_seg] == np.max(delta_cls.ind[old_seg]))[0]
                    replace_seg = delta_cls.new_seg[old_seg[mx]]
                    update = np.where(delta_cls.new_rch_id_down[:,seg] == dn_segs[ds])[0]
                    delta_cls.new_rch_id_down[update,seg] = replace_seg
            

###############################################################################        

def find_sword_breaks(delta_cls, sword):
    """
    Finds SWORD reach and centerline IDs that are close to the
    delta's upstream junctions and saves them for breaking SWORD
    reaches.

    Parameters
    ----------
    delta_cls: obj
        Object containing delta centerline attributes.  
    sword: obj
        Object containing SWORD dimensions and attributes. 

    Returns
    -------
    break_rchs: list
        SWORD reach IDs to break.
    break_ids: list
        SWORD centerline IDs indicating where to break the reach.

    """

    zero_up = np.where(delta_cls.new_n_rch_up == 0)[0]
    up_segs = np.unique(delta_cls.new_seg[zero_up])
    #set up spatial query to sword points. 
    sword_pts = np.vstack((sword.centerlines.x, 
                            sword.centerlines.y)).T
    kdt = sp.cKDTree(sword_pts)
    #loop through delta endpoints and find closest 
    #sword point. Determine whether it fits the 
    #criteria to break the sword reach. 
    break_rchs = []
    break_ids = []
    for s in list(range(len(up_segs))):
        seg = np.where(delta_cls.new_seg == up_segs[s])[0]
        mx_pt = np.where(delta_cls.ind[seg] == np.max(delta_cls.ind[seg]))[0]
        dlt_pt = np.vstack((delta_cls.lon[seg[mx_pt]], 
                            delta_cls.lat[seg[mx_pt]])).T
        pt_dist, pt_ind = kdt.query(dlt_pt, k = 1)
        #finding sword break index and determining if it's close to 
        #a current reach boundary. 
        sword_rch = np.where(sword.centerlines.reach_id[0,:] == 
                            sword.centerlines.reach_id[0,pt_ind])[0]
        break_pt = sword.centerlines.cl_id[pt_ind]
        rch_mx_ind = np.max(sword.centerlines.cl_id[sword_rch])
        rch_mn_ind = np.min(sword.centerlines.cl_id[sword_rch])
        #if closest sword point is within 5 points of the reach 
        #boundary ends, don't flag to break. 
        if break_pt < rch_mn_ind+6 or break_pt > rch_mx_ind-6:
            continue
        #if closest sword point is farther from the reach ends
        #flag to break. 
        else:
            break_ids.append(sword.centerlines.cl_id[pt_ind][0])
            break_rchs.append(sword.centerlines.reach_id[0,pt_ind][0])
    
    return break_rchs, break_ids
        
###############################################################################        

def find_tributary_junctions(sword, delta_tribs):

    sword_pts = np.vstack((sword.centerlines.x, 
                           sword.centerlines.y)).T
    kdt = sp.cKDTree(sword_pts)
    pt_dist, pt_ind = kdt.query(sword_pts, 
                                k = 30, 
                                distance_upper_bound=0.005)

    tribs = np.zeros(len(sword.centerlines.reach_id[0,:]))
    # uniq_segs = np.unique(sword.centerlines.reach_id[0,:])
    uniq_segs = delta_tribs
    for ind in list(range(len(uniq_segs))):
        # print(ind, len(uniq_segs)-1)

        # Isolate endpoints for the edited segment.
        seg = np.where(sword.centerlines.reach_id[0,:] == uniq_segs[ind])[0]
        pt1 = seg[np.where(sword.centerlines.cl_id[seg] == np.min(sword.centerlines.cl_id[seg]))[0]]
        pt2 = seg[np.where(sword.centerlines.cl_id[seg] == np.max(sword.centerlines.cl_id[seg]))[0]]
                                
        ep1_ind = pt_ind[pt1,:]
        ep1_dist = pt_dist[pt1,:]
        na1 = np.where(ep1_ind == len(sword.centerlines.reach_id[0,:]))
        ep1_dist = np.delete(ep1_dist, na1)
        ep1_ind = np.delete(ep1_ind, na1)
        na1_1 = np.where(sword.centerlines.reach_id[0,ep1_ind] == uniq_segs[ind])[0]
        ep1_dist = np.delete(ep1_dist, na1_1)
        ep1_ind = np.delete(ep1_ind, na1_1)

        ep2_ind = pt_ind[pt2,:]
        ep2_dist = pt_dist[pt2,:]
        na2 = np.where(ep2_ind == len(sword.centerlines.reach_id[0,:]))
        ep2_dist = np.delete(ep2_dist, na2)
        ep2_ind = np.delete(ep2_ind, na2)
        na2_1 = np.where(sword.centerlines.reach_id[0,ep2_ind] == uniq_segs[ind])[0]
        ep2_dist = np.delete(ep2_dist, na2_1)
        ep2_ind = np.delete(ep2_ind, na2_1)

        ep1_segs = np.unique(sword.centerlines.reach_id[0,[ep1_ind]])
        ep2_segs = np.unique(sword.centerlines.reach_id[0,[ep2_ind]])
        
        if len(ep1_segs) > 0:
            for e1 in list(range(len(ep1_segs))):
                #finding min/max reach cl_ids.
                s1 = np.where(sword.centerlines.reach_id[0,:] == ep1_segs[e1])[0]
                ep1_min = np.min(sword.centerlines.cl_id[s1])
                ep1_max = np.max(sword.centerlines.cl_id[s1])
                #finding the junction point cl_id. 
                con1_ind = np.where(sword.centerlines.reach_id[0,ep1_ind] == ep1_segs[e1])[0]
                con1_pt = ep1_ind[np.where(ep1_dist[con1_ind] == np.min(ep1_dist[con1_ind]))[0]][0]
                ep1_junct = sword.centerlines.cl_id[con1_pt]
                if ep1_junct > ep1_min+5 and ep1_junct < ep1_max-5:
                    if len(seg) >= 15: 
                        tribs[con1_pt] = 1
        
        if len(ep2_segs) > 0:
            for e2 in list(range(len(ep2_segs))):
                #finding min/max reach cl_ids. 
                s2 = np.where(sword.centerlines.reach_id[0,:] == ep2_segs[e2])[0]
                ep2_min = np.min(sword.centerlines.cl_id[s2])
                ep2_max = np.max(sword.centerlines.cl_id[s2])
                #finding the junction point cl_id. 
                con2_ind = np.where(sword.centerlines.reach_id[0,ep2_ind] == ep2_segs[e2])[0]
                con2_pt = ep2_ind[np.where(ep2_dist[con2_ind] == np.min(ep2_dist[con2_ind]))[0]][0]
                ep2_junct = sword.centerlines.cl_id[con2_pt]
                if ep2_junct > ep2_min+5 and ep2_junct < ep2_max-5:
                    if len(seg) >= 15:
                        tribs[con2_pt] = 1

    t = np.where(tribs > 0)[0]
    break_id = sword.centerlines.cl_id[t]
    break_rchs = sword.centerlines.reach_id[0,t]

    return break_rchs, break_id
        
###############################################################################

def find_all_ds_rchs(sword, delete_start):
    """
    Finds all downstream reaches associated with 
    a specified SWORD reach ID. This is a subfunction 
    of  the "find_ds_sword_rchs" function. 

    Parameters
    ----------
    sword: obj
        Class object containing SWORD dimensions and 
        attributes. 
    delete_start: int
        SWORD reach ID for which to find all associated
        downstream reach IDs. 

    Returns
    -------
    del_array: numpy.array()
        Array of SWORD reach IDs downstream of delta 
        junction. 

    """
        
    #find all downstream reaches from the start reach. 
    ds_rchs = np.array([delete_start])
    del_rchs = []
    while len(ds_rchs) > 0:
        del_rchs.append(ds_rchs)
        idx = np.where(np.in1d(sword.reaches.id, ds_rchs)==True)[0]
        nghs = np.unique(sword.reaches.rch_id_down[:,idx])
        ds_rchs = nghs[nghs>0]
    #unnest the list.
    del_array = np.unique(np.array([item for sublist in del_rchs for item in sublist]))
    
    return del_array

###############################################################################

def find_ds_sword_rchs(delta_cls, sword, pt_radius):
    """
    Finds all downstream SWORD reaches of the delta
    upstream junction. 

    Parameters
    ----------
    delta_cls: obj
        Class object containing delta dimensions and 
        attributes. 
    sword: obj
        Class object containing SWORD dimensions and 
        attributes.

    Returns
    -------
    delete_rchs: numpy.array()
        Array of SWORD reach IDs downstream of delta 
        junction.

    """

    #find upstream delta start points. 
    zero_up = np.where(delta_cls.new_n_rch_up == 0)[0]
    up_segs = np.unique(delta_cls.new_seg[zero_up])
    #set up spatial query to sword points. 
    sword_pts = np.vstack((sword.centerlines.x, 
                            sword.centerlines.y)).T
    delta_pts = np.vstack((delta_cls.lon, delta_cls.lat)).T
    kdt = sp.cKDTree(sword_pts)

    #check to make sure there is sufficient percentage overlap between 
    #the delta additions and SWORD to rule out non-overlapping deltas 
    #like the Hay. 
    rch_dn_copy = np.copy(sword.reaches.rch_id_down)
    pt_dist, pt_ind = kdt.query(delta_pts, k = 1)
    perc = (len(np.where(pt_dist < 0.005)[0])/len(pt_dist))*100
    if perc > 10:
        #if overlap is sufficient find existing SWORD reaches to delete. 
        for s in list(range(len(up_segs))):
            seg = np.where(delta_cls.new_seg == up_segs[s])[0]
            mx_pt = np.where(delta_cls.ind[seg] == np.max(delta_cls.ind[seg]))[0]
            dlt_pt = np.vstack((delta_cls.lon[seg[mx_pt]], 
                                delta_cls.lat[seg[mx_pt]])).T
            pt_dist, pt_ind = kdt.query(dlt_pt, k = pt_radius) #very sensitive to this... 
            #find starting reach for downstream deletions. 
            sword_rchs = np.unique(sword.centerlines.reach_id[0,pt_ind])
            del_start = []
            for r in list(range(len(sword_rchs))):
                rch = np.where(rch_dn_copy == sword_rchs[r])[1] #gives reach index.
                if sword.reaches.id[rch] in sword_rchs:
                    del_start.append(sword_rchs[r])
                else:
                    delta_cls.sword_rch_id_up[0,seg] = sword_rchs[r]
                    delta_cls.new_n_rch_up[seg] = 1
                    #sword topology updates. 
                    #reach dimension update. 
                    swd_rch = np.where(sword.reaches.id == sword_rchs[r])[0]
                    sword.reaches.rch_id_down[:,swd_rch] = 0
                    sword.reaches.rch_id_down[0,swd_rch] = np.unique(delta_cls.reach_id[0,seg])
                    #centerline dimension update. 
                    swd_cls = np.where(sword.centerlines.reach_id[0,:] == sword_rchs[r])[0]
                    rw1 = np.where(np.in1d(sword.centerlines.reach_id[1,swd_cls], sword_rchs)==True)
                    rw2 = np.where(np.in1d(sword.centerlines.reach_id[2,swd_cls], sword_rchs)==True)
                    rw3 = np.where(np.in1d(sword.centerlines.reach_id[3,swd_cls], sword_rchs)==True)
                    if len(rw1) > 0:
                        sword.centerlines.reach_id[1,swd_cls[rw1]] = np.unique(delta_cls.reach_id[0,seg])
                    if len(rw2) > 0:
                        sword.centerlines.reach_id[2,swd_cls[rw2]] = np.unique(delta_cls.reach_id[0,seg])
                    if len(rw3) > 0:
                        sword.centerlines.reach_id[3,swd_cls[rw3]] = np.unique(delta_cls.reach_id[0,seg]) #sword.centerlines.reach_id[1:3,swd_cls]

            #find and catalog downstream reaches to delete.
            if len(del_start) > 0:
                #put inside loop
                for dl in list(range(len(del_start))):
                    ds_rchs = find_all_ds_rchs(sword, del_start[dl])
                    if 'delete_rchs' in locals():
                        delete_rchs = np.append(delete_rchs, ds_rchs)
                        delete_rchs = np.unique(delete_rchs)
                    else:
                        delete_rchs = np.unique(ds_rchs)
            #if multiple reaches are detected or no reaches are detected print warnings. 
            else:
                print("!!! WARNING: insufficient junction reaches detected !!!")
                print("----> Check 'find_ds_sword_rchs' function in 'delta_utils.py'")

    else:
        #don't delete reaches. 
        delete_rchs = []
        #assign topology at delta-sword junction. 
        for s in list(range(len(up_segs))):
            seg = np.where(delta_cls.new_seg == up_segs[s])[0]
            mx_pt = np.where(delta_cls.ind[seg] == np.max(delta_cls.ind[seg]))[0]
            dlt_pt = np.vstack((delta_cls.lon[seg[mx_pt]], 
                                delta_cls.lat[seg[mx_pt]])).T
            pt_dist, pt_ind = kdt.query(dlt_pt, k = 20)
            #find starting reach for downstream deletions. 
            sword_rchs = np.unique(sword.centerlines.reach_id[0,pt_ind])
            for r in list(range(len(sword_rchs))):
                rch = np.where(rch_dn_copy == sword_rchs[r])[1] #gives reach index.
                if sword.reaches.id[rch] in sword_rchs:
                    continue
                else:
                    delta_cls.sword_rch_id_up[0,seg] = sword_rchs[r]
                    delta_cls.new_n_rch_up[seg] = 1
                    #sword topology updates. 
                    #reach dimension update. 
                    swd_rch = np.where(sword.reaches.id == sword_rchs[r])[0]
                    val = np.where(sword.reaches.rch_id_down[:,swd_rch]==0)[0]
                    sword.reaches.rch_id_down[val[0],swd_rch] = np.unique(delta_cls.reach_id[0,seg])
                    sword.reaches.n_rch_down[swd_rch] = sword.reaches.n_rch_down[swd_rch]+1
                    #centerline dimension update. 
                    swd_cls = np.where(sword.centerlines.reach_id[0,:] == sword_rchs[r])[0]
                    rw1 = np.where(np.in1d(sword.centerlines.reach_id[1,swd_cls], sword_rchs)==True)[0]
                    rw2 = np.where(np.in1d(sword.centerlines.reach_id[2,swd_cls], sword_rchs)==True)[0]
                    rw3 = np.where(np.in1d(sword.centerlines.reach_id[3,swd_cls], sword_rchs)==True)[0]
                    if len(rw1) > 0:
                        v1 = np.where(sword.centerlines.reach_id[:,swd_cls[rw1]] == 0)[0]
                        sword.centerlines.reach_id[v1[0],swd_cls[rw1]] = np.unique(delta_cls.reach_id[0,seg])
                    if len(rw2) > 0:
                        v2 = np.where(sword.centerlines.reach_id[:,swd_cls[rw2]] == 0)[0]
                        sword.centerlines.reach_id[v2[0],swd_cls[rw2]] = np.unique(delta_cls.reach_id[0,seg])
                    if len(rw3) > 0:
                        v3 = np.where(sword.centerlines.reach_id[:,swd_cls[rw3]] == 0)[0]
                        sword.centerlines.reach_id[v3[0],swd_cls[rw3]] = np.unique(delta_cls.reach_id[0,seg]) #sword.centerlines.reach_id[1:3,swd_cls]            
        
    return delete_rchs

###############################################################################

def number_segments(delta_cls):
    """
    Assigns numbers to delta reaches that are ordered 
    based on topology. 

    Parameters
    ----------
    delta_cls: obj
        Class object containing delta dimensions and 
        attributes. 

    New Attributes
    --------------
    delta_cls.rch_num: numpy.array()
        Delta reach numbers ordered based on topology.

    Returns
    -------
    None. 

    """

    rch_num = np.repeat(-9999, len(delta_cls.new_seg)).astype(np.float64) #filler array for new outlet distance. 
    flag = np.zeros(len(delta_cls.new_seg))
    outlets = delta_cls.new_seg[np.where(delta_cls.new_n_rch_down == 0)[0]]
    start_rchs = np.array([outlets[0]]) #start with any outlet first. 
    loop = 1
    cnt = 1
    ### While loop 
    while len(start_rchs) > 0:
        #for loop to go through all start_rchs, which are the upstream neighbors of 
        #the previously updated reaches. The first reach is any outlet. 
        # print('LOOP:',loop, start_rchs)
        up_ngh_list = []
        for r in list(range(len(start_rchs))):
            rch = np.where(delta_cls.new_seg == start_rchs[r])[0]
            rch_flag = np.max(flag[rch])
            if np.unique(delta_cls.new_n_rch_down[rch]) == 0:
                rch_num[rch] = cnt
                cnt = cnt+1
                up_nghs = np.unique(delta_cls.new_rch_id_up[:,rch]); up_nghs = up_nghs[up_nghs>0]
                up_flag = np.array([np.max(flag[np.where(delta_cls.new_seg == n)[0]]) for n in up_nghs])
                up_nghs = up_nghs[up_flag == 0]
                up_ngh_list.append(up_nghs)
                # loop=loop+1
            else:
                dn_nghs = np.unique(delta_cls.new_rch_id_down[:,rch]); dn_nghs = dn_nghs[dn_nghs>0]
                dn_dist = np.array([rch_num[np.where(delta_cls.new_seg == n)[0]][0] for n in dn_nghs])
                if min(dn_dist) == -9999:
                    if rch_flag == 1:
                        rch_num[rch] = cnt
                        cnt = cnt+1
                        up_nghs = np.unique(delta_cls.new_rch_id_up[:,rch]); up_nghs = up_nghs[up_nghs>0]
                        up_flag = np.array([np.max(flag[np.where(delta_cls.new_seg == n)[0]][0]) for n in up_nghs])
                        up_nghs = up_nghs[up_flag == 0]
                        up_ngh_list.append(up_nghs)
                    else:
                        #set condition to flag downstream, non-outlet reaches (multichannel cases). 
                        flag[rch] = 1
                else:
                    # add_val = max(dn_dist)
                    rch_num[rch] = cnt
                    cnt = cnt+1
                    up_nghs = np.unique(delta_cls.new_rch_id_up[:,rch]); up_nghs = up_nghs[up_nghs>0]
                    up_flag = np.array([np.max(flag[np.where(delta_cls.new_seg == n)[0]][0]) for n in up_nghs])
                    up_nghs = up_nghs[up_flag == 0]
                    up_ngh_list.append(up_nghs) 
        #formatting next start reach.         
        up_ngh_arr = np.array(list(chain.from_iterable(up_ngh_list)))
        start_rchs = np.unique(up_ngh_arr)
        #if no more upstream neighbors move to next outlet. 
        if len(start_rchs) == 0:
            outlets = delta_cls.new_seg[np.where((delta_cls.new_n_rch_down == 0) & (rch_num == -9999))[0]]
            #a case where all downstream reaches have filled but not all upstream.
            if len(outlets) == 0 and min(rch_num) > -9999:
                start_rchs = np.array([])
            elif len(outlets) == 0 and min(rch_num) == -9999:
                flag_rchs = np.unique(delta_cls.new_seg[np.where((rch_num == -9999)&(flag == 1))[0]])
                if len(flag_rchs) > 0:  
                    start_rchs = np.array([flag_rchs[0]])
                else:
                    #find reach with downstream distances filled but a value of -9999
                    print('!!! PROBLEM !!! --> Still -9999 values in reach number array')
                    break
            else:
                start_rchs = np.array([outlets[0]])
        loop = loop+1
        if loop > 5*len(np.unique(delta_cls.new_seg)):
            print('!!! LOOP STUCK !!!')
            break

    #add to object
    delta_cls.rch_num = rch_num   

###############################################################################

def number_nodes(delta_cls):
    """
    Assigns numbers to nodes within delta reaches that 
    are ordered based on flow direction (downstream to 
    upstream order). 

    Parameters
    ----------
    delta_cls: obj
        Class object containing delta dimensions and 
        attributes. 

    New Attributes
    --------------
    delta_cls.node_num: numpy.array()
        Delta node numbers ordered based on flow
        direction.

    Returns
    -------
    None. 

    """

    unq_segs = np.unique(delta_cls.new_seg)
    delta_cls.node_num = np.zeros(len(delta_cls.new_seg))
    for ind in list(range(len(unq_segs))):
        seg = np.where(delta_cls.new_seg == unq_segs[ind])[0]
        unq_nodes = np.unique(delta_cls.seg_node[seg])
        min_index = [np.min(delta_cls.cl_id[seg[np.where(delta_cls.seg_node[seg] == n)[0]]]) for n in unq_nodes]
        order = np.argsort(min_index)
        order_nodes = unq_nodes[order]
        cnt = 1
        for n in list(range(len(order_nodes))):
            node = np.where(delta_cls.seg_node[seg] == order_nodes[n])[0]
            delta_cls.node_num[seg[node]] = cnt 
            cnt = cnt+1

###############################################################################

def create_sword_ids(delta_cls, sword):
    """
    Created official SWORD formatted reach and node
    IDs for delta reaches and nodes. 

    Parameters
    ----------
    delta_cls: obj
        Class object containing delta dimensions and 
        attributes.
    sword: obj
        Class object containing SWORD dimensions and 
        attributes.

    New Attributes type [dimension]
    --------------------------------
    delta_cls.reach_id: numpy.array() [4,number of points]
        SWORD formatted reach IDs.
    delta_cls.node_id: numpy.array() [4,number of points]
        SWORD formatted node IDs. 
    delta_cls.sword_rch_id_up: numpy.array() [4,number of points]
        SWORD formatted upstream reach IDs. 
    delta_cls.sword_rch_id_down: numpy.array() [4,number of points]
        SWORD formatted downstream reach IDs. 
    
    Returns
    -------
    None. 

    """

    basins_l6 = np.array([int(str(b)[0:6]) for b in delta_cls.basins])
    sword_l6 = np.array([int(str(b)[0:6]) for b in sword.reaches.id])
    sword_rch_nums = np.array([int(str(b)[6:10]) for b in sword.reaches.id])
    unq_basins = np.unique(basins_l6)
    sword_rch_ids = np.zeros((4,len(delta_cls.basins)), dtype=int)
    sword_node_ids = np.zeros((4,len(delta_cls.basins)), dtype=int)
    sword_rch_id_up = np.copy(delta_cls.new_rch_id_up)
    sword_rch_id_down = np.copy(delta_cls.new_rch_id_down)
    for ind in list(range(len(unq_basins))):
        # print(ind)
        bsn = np.where(basins_l6 == unq_basins[ind])[0]
        swd_bsn = np.where(sword_l6 == unq_basins[ind])[0]
        unq_rchs = np.unique(delta_cls.rch_num[bsn])
        rch_id_nums = np.array(list(range(len(unq_rchs))))+1
        if len(swd_bsn) > 0:
            max_sword = np.max(sword_rch_nums[swd_bsn])
            rch_id_nums = rch_id_nums + max_sword
        # Create formal reach id.
        for r in list(range(len(unq_rchs))):
            rch = np.where(delta_cls.rch_num == unq_rchs[r])[0]
            if len(str(int(rch_id_nums[r]))) == 1:
                fill = '000'
                reach_id = int(str(unq_basins[ind])+fill+str(int(rch_id_nums[r]))+'5')
            if len(str(int(rch_id_nums[r]))) == 2:
                fill = '00'
                reach_id = int(str(unq_basins[ind])+fill+str(int(rch_id_nums[r]))+'5')
            if len(str(int(rch_id_nums[r]))) == 3:
                fill = '0'
                reach_id = int(str(unq_basins[ind])+fill+str(int(rch_id_nums[r]))+'5')
            if len(str(int(rch_id_nums[r]))) == 4:
                reach_id = int(str(unq_basins[ind])+str(int(rch_id_nums[r]))+'5')
            
            if len(str(reach_id)) != 11:
                break

            sword_rch_ids[0,rch] = reach_id
            
            # Update reach IDs in topology variables. 
            up_ind = np.where(sword_rch_id_up == int(np.unique(delta_cls.new_seg[rch])))
            dn_ind = np.where(sword_rch_id_down == int(np.unique(delta_cls.new_seg[rch])))
            sword_rch_id_up[up_ind[0],up_ind[1]] = reach_id
            sword_rch_id_down[dn_ind[0],dn_ind[1]] = reach_id
            
            # Create formal node id within reach. 
            unq_nodes = np.unique(delta_cls.node_num[rch])
            for n in list(range(len(unq_nodes))):
                nd = np.where(delta_cls.node_num[rch] == unq_nodes[n])[0]
                if len(str(int(unq_nodes[n]))) == 1:
                    fill = '00'
                    node_id = int(str(reach_id)[:-1]+fill+str(int(unq_nodes[n]))+'5')
                if len(str(int(unq_nodes[n]))) == 2:
                    fill = '0'
                    node_id = int(str(reach_id)[:-1]+fill+str(int(unq_nodes[n]))+'5')
                if len(str(int(unq_nodes[n]))) == 3:
                    node_id = int(str(reach_id)[:-1]+str(int(unq_nodes[n]))+'5')
                sword_node_ids[0,rch[nd]] = node_id

        delta_cls.reach_id = sword_rch_ids
        delta_cls.node_id = sword_node_ids
        delta_cls.sword_rch_id_up = sword_rch_id_up
        delta_cls.sword_rch_id_down = sword_rch_id_down

###############################################################################

def find_all_us_rchs(sword, start_rchs):
    """
    Finds all upstream reaches associated with 
    specified SWORD reach IDs. This is a subfunction 
    of  the "find_delta_tribs" function. 

    Parameters
    ----------
    sword: obj
        Class object containing SWORD dimensions and 
        attributes. 
    start_rchs: int
        SWORD reach IDs for which to find all associated
        upstream reach IDs. 

    Returns
    -------
    del_array: numpy.array()
        Array of SWORD reach IDs that are upstream of 
        the specified reaches. 

    """
        
    #find all downstream reaches from the start reach. 
    if isinstance(start_rchs, np.ndarray):
        us_rchs = start_rchs
    else:
        us_rchs = np.array([start_rchs])
    del_rchs = []
    while len(us_rchs) > 0:
        del_rchs.append(us_rchs)
        idx = np.where(np.in1d(sword.reaches.id, us_rchs)==True)[0]
        nghs = np.unique(sword.reaches.rch_id_up[:,idx])
        us_rchs = np.array(nghs[nghs>0])
    #unnest the list.
    del_array = np.unique(np.array([item for sublist in del_rchs for item in sublist]))
    
    return del_array

###############################################################################

def find_delta_tribs(delta_cls, sword):
    """
    Finds SWORD reaches that need to be removed within
    the delta or are tributaries entering the delta. 

    Parameters
    ----------
    sword: obj
        Class object containing SWORD dimensions and 
        attributes. 
    delete_start: int
        SWORD reach ID for which to find all associated
        downstream reach IDs. 

    Returns
    -------
    rmv_rchs: numpy.array()
        SWORD reach IDs to remove. 
    delta_tribs: list
        SWORD reach IDs identified as tributaries 
        along the delta. 

    """

    #subsetting for faster spatial joins. 
    sword_l6 = np.array([int(str(b)[0:6]) for b in sword.centerlines.reach_id[0,:]])
    sword_rch_l6 = np.array([int(str(b)[0:6]) for b in sword.reaches.id])
    basins_l6 = np.array([int(str(b)[0:6]) for b in delta_cls.basins])
    cl_keep = np.where(np.in1d(sword_l6, basins_l6)==True)[0]
    rch_keep = np.where(np.in1d(sword_rch_l6, basins_l6)==True)[0]
    zero_dn = np.where(sword.reaches.n_rch_down[rch_keep] == 0)[0]
    trib_check = sword.reaches.id[rch_keep[zero_dn]]

    #spatial join. 
    sword_pts = np.vstack((sword.centerlines.x[cl_keep], 
                        sword.centerlines.y[cl_keep])).T
    delta_pts = np.vstack((delta_cls.lon, delta_cls.lat)).T
    kdt = sp.cKDTree(delta_pts)
    pt_dist, pt_ind = kdt.query(sword_pts, k = 1)

    #loop through remaining sword reaches with zero downstream neighbors 
    #and see if there is significant overlap. If so, delete, if not, record
    #as tributary. 
    rmv_rchs = []
    delta_tribs = []
    for t in list(range(len(trib_check))):
        rch = np.where(sword.reaches.id == trib_check[t])[0]
        seg = np.where(sword.reaches.path_segs == sword.reaches.path_segs[rch])[0]
        seg_rchs = sword.reaches.id[seg]
        pts = np.where(np.in1d(sword.centerlines.reach_id[0,cl_keep], seg_rchs)==True)[0]
        radius = np.where(pt_dist[pts] < 0.005)[0]
        perc = len(radius)/len(pts)*100
        if perc >= 0:
            #determine if reaches are part of a separate network
            #from delta by looking for outlets. Remove if there 
            #is no outlet. 
            if perc == 0: 
                ends = sword.reaches.end_rch[seg]
                if 2 in ends:
                    continue
                else:
                    up_rchs = find_all_us_rchs(sword, seg_rchs) 
                    rmv_rchs.append(up_rchs)
            #remove reaches if there is more than 5% overlap. 
            elif perc > 5:
                up_rchs = find_all_us_rchs(sword, seg_rchs) 
                rmv_rchs.append(up_rchs)
            #flag as tributary if there is a small overlap. 
            else:
                delta_tribs.append(trib_check[t])

    #unnest list. 
    rmv_rchs = np.unique(np.array([item for sublist in rmv_rchs for item in sublist]))

    return rmv_rchs, delta_tribs

###############################################################################

def create_nodes(delta_cls):
    """
    xxxx
    
    """

    subnodes = geo.Object()
    unq_nodes = np.unique(delta_cls.node_id[0,:])

    #create filler arrays
    subnodes.id = np.zeros(len(np.unique(unq_nodes)), dtype=int)
    subnodes.x = np.zeros(len(np.unique(unq_nodes)))
    subnodes.y = np.zeros(len(np.unique(unq_nodes)))
    subnodes.len = np.zeros(len(np.unique(unq_nodes)))
    subnodes.wse = np.zeros(len(np.unique(unq_nodes)))
    subnodes.wse_var = np.zeros(len(np.unique(unq_nodes)))
    subnodes.wth = np.zeros(len(np.unique(unq_nodes)))
    subnodes.wth_var = np.zeros(len(np.unique(unq_nodes)))
    subnodes.max_wth = np.zeros(len(np.unique(unq_nodes)))
    subnodes.facc = np.zeros(len(np.unique(unq_nodes)))
    subnodes.nchan_max = np.zeros(len(np.unique(unq_nodes)))
    subnodes.nchan_mod = np.zeros(len(np.unique(unq_nodes)))
    subnodes.seg = np.zeros(len(np.unique(unq_nodes)))
    subnodes.grod = np.zeros(len(np.unique(unq_nodes)))
    subnodes.lakeflag = np.zeros(len(np.unique(unq_nodes)))
    subnodes.grod_fid = np.zeros(len(np.unique(unq_nodes)))
    subnodes.hfalls_fid = np.zeros(len(np.unique(unq_nodes)))
    subnodes.lake_id = np.zeros(len(np.unique(unq_nodes)))
    subnodes.sinuosity = np.zeros(len(np.unique(unq_nodes)))
    subnodes.reach_id = np.zeros(len(np.unique(unq_nodes)), dtype=int)
    subnodes.cl_id = np.zeros((2, len(unq_nodes)))

    # Loop through each node ID to create location and attribute information.
    for ind in list(range(len(unq_nodes))):
        nodes = np.where(delta_cls.node_id[0,:] == unq_nodes[ind])[0]
        subnodes.id[ind] = int(unq_nodes[ind])
        subnodes.cl_id[0,ind] = np.min(delta_cls.cl_id[nodes])
        subnodes.cl_id[1,ind] = np.max(delta_cls.cl_id[nodes])
        subnodes.reach_id[ind] = int(np.unique(delta_cls.reach_id[0,nodes])[0])
        subnodes.seg[ind] = np.unique(delta_cls.seg[nodes])[0]
        subnodes.x[ind] = np.median(delta_cls.lon[nodes])
        subnodes.y[ind] = np.median(delta_cls.lat[nodes])
        subnodes.len[ind] = np.max(np.unique(delta_cls.new_len[nodes]))
        subnodes.wse[ind] = np.median(delta_cls.elv[nodes])
        subnodes.wse_var[ind] = np.var(delta_cls.elv[nodes])
        subnodes.facc[ind] = np.max(delta_cls.facc[nodes])
        subnodes.nchan_max[ind] = np.max(delta_cls.nchan[nodes])
        subnodes.wth[ind] = np.max(delta_cls.wth[nodes])
        subnodes.wth_var[ind] = np.max(delta_cls.wth_var[nodes])
        subnodes.max_wth[ind] = np.max(delta_cls.max_wth[nodes])
        subnodes.sinuosity[ind] = np.max(delta_cls.sinuosity[nodes])
        subnodes.nchan_mod[ind] = mode(delta_cls.nchan[nodes])
        subnodes.lakeflag[ind] = np.max(delta_cls.lake[nodes])
        subnodes.lake_id[ind] = np.max(delta_cls.lake_id[nodes])
        
        #dam/obstruction info. 
        grod = np.copy(delta_cls.grod_fid[nodes])
        grod[np.where(grod > 4)] = 0
        subnodes.grod[ind] = np.max(grod)
        # Assign grod and hydrofalls ids to nodes.
        nind = np.where(grod == np.max(grod))[0][0]
        if np.max(grod) == 0:
            subnodes.grod_fid[ind] = 0
        elif np.max(grod) == 4:
            subnodes.hfalls_fid[ind] = delta_cls.hfalls_fid[nodes[nind]]
        else:
            subnodes.grod_fid[ind] = delta_cls.grod_fid[nodes[nind]]

        #filler variables.
        subnodes.ext_dist_coef = np.repeat(3,len(subnodes.id))
        subnodes.wth_coef = np.repeat(0.5, len(subnodes.id))
        subnodes.meand_len = np.repeat(-9999, len(subnodes.id))
        subnodes.river_name = np.repeat('NODATA', len(subnodes.id))
        subnodes.manual_add = np.repeat(1, len(subnodes.id))
        subnodes.edit_flag = np.repeat('7', len(subnodes.id))
        subnodes.trib_flag = np.repeat(0, len(subnodes.id)) 
        subnodes.dist_out = np.repeat(-9999, len(subnodes.id))
        subnodes.path_freq = np.repeat(-9999, len(subnodes.id))
        subnodes.path_order = np.repeat(-9999, len(subnodes.id))
        subnodes.path_segs = np.repeat(-9999, len(subnodes.id))
        subnodes.main_side = np.repeat(2, len(subnodes.id))
        subnodes.strm_order = np.repeat(-9999, len(subnodes.id))
        subnodes.network = np.repeat(0, len(subnodes.id))
        subnodes.end_rch = np.repeat(0, len(subnodes.id))
        subnodes.add_flag = np.repeat(1, len(subnodes.id))

    return(subnodes)

###############################################################################

def create_reaches(delta_cls):

    """
    xxxx
    """

    # Set variables.
    subreaches = geo.Object()
    uniq_rch = np.unique(delta_cls.reach_id[0,:])
    subreaches.id = np.zeros(len(uniq_rch), dtype=int)
    subreaches.x = np.zeros(len(uniq_rch))
    subreaches.y = np.zeros(len(uniq_rch))
    subreaches.x_max = np.zeros(len(uniq_rch))
    subreaches.x_min = np.zeros(len(uniq_rch))
    subreaches.y_max = np.zeros(len(uniq_rch))
    subreaches.y_min = np.zeros(len(uniq_rch))
    subreaches.wse = np.zeros(len(uniq_rch))
    subreaches.wse_var = np.zeros(len(uniq_rch))
    subreaches.wth = np.zeros(len(uniq_rch))
    subreaches.wth_var = np.zeros(len(uniq_rch))
    subreaches.facc = np.zeros(len(uniq_rch))
    subreaches.len = np.zeros(len(uniq_rch))
    subreaches.nchan_max = np.zeros(len(uniq_rch))
    subreaches.nchan_mod = np.zeros(len(uniq_rch))
    subreaches.rch_n_nodes = np.zeros(len(uniq_rch))
    subreaches.slope = np.zeros(len(uniq_rch))
    subreaches.grod = np.zeros(len(uniq_rch))
    subreaches.grod_fid = np.zeros(len(uniq_rch))
    subreaches.hfalls_fid = np.zeros(len(uniq_rch))
    subreaches.lakeflag = np.zeros(len(uniq_rch))
    subreaches.lake_id = np.zeros(len(uniq_rch))
    subreaches.max_wth = np.zeros(len(uniq_rch))
    subreaches.orbits = np.zeros((75, len(uniq_rch)))
    subreaches.max_obs = np.zeros(len(uniq_rch))
    subreaches.cl_id = np.zeros((2, len(uniq_rch)))

    # Loop through and calculate reach locations and attributes for each
    # unique reach ID.
    # print('2')
    for ind in list(range(len(uniq_rch))):
        # print(ind)
        reach = np.where(delta_cls.reach_id[0,:] == uniq_rch[ind])[0]
        subreaches.id[ind] = int(np.unique(delta_cls.reach_id[0,reach]))
        subreaches.cl_id[0,ind] = np.min(delta_cls.cl_id[reach])
        subreaches.cl_id[1,ind] = np.max(delta_cls.cl_id[reach])
        subreaches.x[ind] = np.median(delta_cls.lon[reach])
        subreaches.y[ind] = np.median(delta_cls.lat[reach])
        subreaches.x_max[ind] = np.max(delta_cls.lon[reach])
        subreaches.x_min[ind] = np.min(delta_cls.lon[reach])
        subreaches.y_max[ind] = np.max(delta_cls.lat[reach])
        subreaches.y_min[ind] = np.min(delta_cls.lat[reach])
        subreaches.len[ind] = np.max(delta_cls.len[reach])
        subreaches.wse[ind] = np.median(delta_cls.elv[reach])
        subreaches.wse_var[ind] = np.var(delta_cls.elv[reach])
        subreaches.facc[ind] = np.max(delta_cls.facc[reach])
        subreaches.nchan_max[ind] = np.max(delta_cls.nchan[reach])
        subreaches.nchan_mod[ind] = max(set(list(delta_cls.nchan[reach])), key=list(delta_cls.nchan[reach]).count)
        subreaches.rch_n_nodes[ind] = len(np.unique(delta_cls.node_id[0,reach]))
        subreaches.lakeflag[ind] = max(set(list(delta_cls.lake[reach])), key=list(delta_cls.lake[reach]).count)
        subreaches.lake_id[ind] = max(set(list(delta_cls.lake_id[reach])), key=list(delta_cls.lake_id[reach]).count)
        subreaches.wth[ind] = np.max(delta_cls.wth[reach])
        subreaches.wth_var[ind] = np.max(delta_cls.wth_var[reach])
        subreaches.max_wth[ind] = np.max(delta_cls.max_wth[reach])
        subreaches.max_obs[ind] = np.max(delta_cls.num_obs[reach])
        # swot orbit attributes. 
        uniq_orbits = np.unique(delta_cls.orbits[:,reach])
        uniq_orbits = uniq_orbits[np.where(uniq_orbits>0)[0]]
        if len(uniq_orbits) == 0:
            continue
        for idz in list(range(len(uniq_orbits))):
            subreaches.orbits[idz,ind] = uniq_orbits[idz]

        # Find grod type per reach.
        GROD = np.copy(delta_cls.grod[reach])
        GROD[np.where(GROD > 4)] = 0
        subreaches.grod[ind] = np.max(GROD)
        # Assign grod and hydrofalls ids to reach.
        ID = np.where(GROD == np.max(GROD))[0][0]
        if np.max(GROD) == 0:
            subreaches.grod_fid[ind] = 0
        elif np.max(GROD) == 4:
            subreaches.hfalls_fid[ind] = delta_cls.hfalls_fid[reach[ID]]
        else:
            subreaches.grod_fid[ind] = delta_cls.grod_fid[reach[ID]]

        # Slope calculation.
        slope_pts = np.vstack([delta_cls.dist[reach]/1000, np.ones(len(reach))]).T
        slope, intercept = np.linalg.lstsq(slope_pts, delta_cls.elv[reach], rcond=None)[0]
        subreaches.slope[ind] = abs(slope) # m/km
    
        ### create filler variables.
        subreaches.iceflag = np.zeros([366,len(subreaches.id)])
        subreaches.iceflag[:,:] = -9999
        subreaches.river_name = np.repeat('NODATA', len(subreaches.id))
        subreaches.low_slope = np.repeat(0, len(subreaches.id))
        subreaches.edit_flag = np.repeat('7', len(subreaches.id))
        subreaches.trib_flag = np.repeat(0, len(subreaches.id))
        subreaches.dist_out = np.repeat(-9999, len(subreaches.id))
        subreaches.path_freq = np.repeat(-9999, len(subreaches.id))
        subreaches.path_order = np.repeat(-9999, len(subreaches.id))
        subreaches.path_segs = np.repeat(-9999, len(subreaches.id))
        subreaches.main_side = np.repeat(2, len(subreaches.id))
        subreaches.strm_order = np.repeat(-9999, len(subreaches.id))
        subreaches.network = np.repeat(0, len(subreaches.id))
        subreaches.end_rch = np.repeat(0, len(subreaches.id))
        subreaches.add_flag = np.repeat(1, len(subreaches.id))

    return subreaches
        
###############################################################################

def format_sword_topo_attributes(delta_cls, subreaches):
    """
    xxxx
    
    """

    subreaches.n_rch_up = np.zeros(len(subreaches.id),dtype = int)
    subreaches.n_rch_down = np.zeros(len(subreaches.id),dtype = int)
    subreaches.rch_id_up = np.zeros([4,len(subreaches.id)],dtype = int)
    subreaches.rch_id_down = np.zeros([4,len(subreaches.id)],dtype = int)

    for r in list(range(len(subreaches.id))):
        rch = np.where(delta_cls.reach_id[0,:] == subreaches.id[r])[0]
        #find upstream and downstream reach IDs. 
        up_rchs = np.unique(delta_cls.sword_rch_id_up[:,rch])
        up_rchs = up_rchs[up_rchs>0]
        dn_rchs = np.unique(delta_cls.sword_rch_id_down[:,rch])
        dn_rchs = dn_rchs[dn_rchs>0]
        #fill centerline dimension
        mn_id = np.where(delta_cls.cl_id[rch] == min(delta_cls.cl_id[rch]))[0]
        mx_id = np.where(delta_cls.cl_id[rch] == max(delta_cls.cl_id[rch]))[0]
        num_up = len(up_rchs)
        num_dn = len(dn_rchs)
        if num_up > 0:
            subreaches.n_rch_up[r] = num_up
            subreaches.rch_id_up[0:num_up,r] = up_rchs #subreaches.rch_id_up[:,r]
            if num_up > 1:
                up_rchs = up_rchs.reshape(num_up,1)
            delta_cls.reach_id[1:num_up+1,rch[mx_id]] = up_rchs #delta_cls.reach_id[:,rch[mx_id]]
        if num_dn > 0:
            subreaches.n_rch_down[r] = num_dn
            subreaches.rch_id_down[0:num_dn,r] = dn_rchs #subreaches.rch_id_down[:,r]
            if num_dn > 1:
                dn_rchs = dn_rchs.reshape(num_dn,1)
            delta_cls.reach_id[1:num_dn+1,rch[mn_id]] = dn_rchs #delta_cls.reach_id[:,rch[mx_id]]

###############################################################################

def format_fill_attributes(delta_cls, subnodes, subreaches, sword):
    """
    xxxx

    """
    #updating end reach variable 
    hw = np.where(subreaches.n_rch_up == 0)[0]
    ot = np.where(subreaches.n_rch_down == 0)[0]
    junc1 = np.where(subreaches.n_rch_up > 1)[0]
    junc2 = np.where(subreaches.n_rch_down > 1)[0]
    node_hw = np.where(np.in1d(subnodes.reach_id, subreaches.id[hw]) == True)[0]
    node_ot = np.where(np.in1d(subnodes.reach_id, subreaches.id[ot]) == True)[0]
    node_junc1 = np.where(np.in1d(subnodes.reach_id, subreaches.id[junc1]) == True)[0]
    node_junc2 = np.where(np.in1d(subnodes.reach_id, subreaches.id[junc2]) == True)[0]
    subreaches.end_rch[junc1] = 3
    subreaches.end_rch[junc1] = 3
    subreaches.end_rch[hw] = 1
    subreaches.end_rch[ot] = 2
    subnodes.end_rch[node_junc1] = 3
    subnodes.end_rch[node_junc2] = 3
    subnodes.end_rch[node_hw] = 1
    subnodes.end_rch[node_ot] = 2

    #updating network attribute.
    vals = np.where(np.in1d(sword.reaches.id, 
                            np.unique(subreaches.rch_id_up))==True)[0]
    net = np.unique(sword.reaches.network[vals])
    subreaches.network[:] = net
    subnodes.network[:] = net

    #path segment attribute update. 
    cnt = np.max(sword.reaches.path_segs)+1
    unq_segs = np.unique(delta_cls.seg)
    for s in list(range(len(unq_segs))):
        seg = np.where(delta_cls.seg == unq_segs[s])[0]
        rchs = np.unique(delta_cls.reach_id[0,seg])
        rind = np.where(np.in1d(subreaches.id, rchs)==True)[0]
        nind = np.where(np.in1d(subnodes.reach_id, rchs)==True)[0]
        subreaches.path_segs[rind] = cnt
        subnodes.path_segs[nind] = cnt 
        cnt = cnt+1

###############################################################################

def tributary_topo(sword, delta_tribs, basin):
    """
    xxxx

    """
    #spatial join with self.
    lvl2 = np.array([int(str(b)[0:2]) for b in sword.centerlines.reach_id[0,:]]) 
    l2 = np.where(lvl2 == basin)[0]
    sword_pts = np.vstack((sword.centerlines.x[l2], 
                        sword.centerlines.y[l2])).T
    kdt = sp.cKDTree(sword_pts)
    pt_dist, pt_ind = kdt.query(sword_pts, k = 30)

    #loop through tributaries to identify downstream reach
    #and update topology informatino. 
    for trib in list(range(len(delta_tribs))):
        cl_rch = np.where(sword.centerlines.reach_id[0,l2] 
                        == delta_tribs[trib])[0]
        mn_id = np.where(sword.centerlines.cl_id[l2[cl_rch]] == 
                        np.min(sword.centerlines.cl_id[l2[cl_rch]]))[0]
        nghs = np.unique(sword.centerlines.reach_id[0,l2[pt_ind[cl_rch[mn_id]]]])
        nghs = nghs[nghs != delta_tribs[trib]]
        #find downstream neighbor and update topology. 
        for n in list(range(len(nghs))):
            check = np.where(sword.reaches.rch_id_down == nghs[n])[1]
            if sword.reaches.id[check] in nghs:
                #update upstream topology of downstream neighbor in sword.
                rch_update = np.where(sword.reaches.id == nghs[n])[0]
                add = np.where(sword.reaches.rch_id_up[:,rch_update] == 0)[0][0]
                cl_update = np.where(sword.centerlines.reach_id[0,:] == nghs[n])[0]
                mx_id = np.where(sword.centerlines.cl_id[cl_update] == 
                                np.max(sword.centerlines.cl_id[cl_update]))[0]
                #reach dimension updates. 
                sword.reaches.rch_id_up[add,rch_update] = delta_tribs[trib] #sword.reaches.rch_id_up[:,rch_update]
                sword.reaches.n_rch_up[rch_update] = sword.reaches.n_rch_up[rch_update]+1        
                #centerline dimension updates.  
                add_cl = np.where(sword.centerlines.reach_id[:,cl_update[mx_id]] == 0)[0][0]
                sword.centerlines.reach_id[add_cl,cl_update[mx_id]] = delta_tribs[trib]
                #update end reach value for downstream reach. 
                if sword.reaches.end_rch[rch_update] == 0:
                    sword.reaches.end_rch[rch_update] = 3
                    sword.nodes.end_rch[np.where(np.in1d(sword.nodes.reach_id,nghs[n]))[0]] = 3
            
                #update downstream topology of tributary in sword.
                rch_trib = np.where(sword.reaches.id == delta_tribs[trib])[0]
                add2 = np.where(sword.reaches.rch_id_down[:,rch_trib] == 0)[0][0]
                cl_trib = np.where(sword.centerlines.reach_id[0,:] == delta_tribs[trib])[0]
                mn_idx = np.where(sword.centerlines.cl_id[cl_trib] == 
                                np.min(sword.centerlines.cl_id[cl_trib]))[0]
                #reach dimension updates.
                sword.reaches.rch_id_down[add2,rch_trib] = nghs[n] #sword.reaches.rch_id_down[:,rch_trib]
                sword.reaches.n_rch_down[rch_trib] = sword.reaches.n_rch_down[rch_trib]+1
                #centerline dimension updates.  
                add_cl2 = np.where(sword.centerlines.reach_id[:,cl_trib[mn_idx]] == 0)[0][0]
                sword.centerlines.reach_id[add_cl2,cl_trib[mn_idx]] = nghs[n]

###############################################################################

def plot_sword_deletions(sword, delta_cls, rmv_rchs, delta_tribs, delta_dir):
    
    plt_dir = os.path.dirname(delta_dir)+'/plots/'
    if os.path.isdir(plt_dir) is False:
        os.makedirs(plt_dir)
    
    sword_l4 = np.array([int(str(b)[0:4]) for b in sword.centerlines.reach_id[0,:]])
    basins_l4 = np.array([int(str(b)[0:4]) for b in delta_cls.basins])
    cl_keep = np.where(np.in1d(sword_l4, basins_l4)==True)[0]
    rmv = np.where(np.in1d(sword.centerlines.reach_id[0,cl_keep], rmv_rchs)==True)[0]
    trib = np.where(np.in1d(sword.centerlines.reach_id[0,cl_keep], delta_tribs)==True)[0]

    plt.figure(1, figsize=(8,8))
    plt.scatter(delta_cls.lon, delta_cls.lat, c='gold', s=3, label='Delta Additions')
    plt.scatter(sword.centerlines.x[cl_keep], sword.centerlines.y[cl_keep], c='blue', s=3, label='SWORD')
    plt.scatter(sword.centerlines.x[cl_keep[rmv]], sword.centerlines.y[cl_keep[rmv]], c='red', s=3, label='Flagged Deletions')
    plt.scatter(sword.centerlines.x[cl_keep[trib]], sword.centerlines.y[cl_keep[trib]], c='limegreen', s=3, label='Flagged Tributaries')
    plt.xlim(np.min(delta_cls.lon)-0.1, np.max(delta_cls.lon)+0.1)
    plt.ylim(np.min(delta_cls.lat)-0.1, np.max(delta_cls.lat)+0.1)
    plt.title(os.path.basename(delta_dir)[:-3])
    plt.legend()
    plt.savefig(plt_dir + os.path.basename(delta_dir)[:-3]+'_deletions.png')
    plt.close()

###############################################################################

def plot_sword_additions(sword, delta_cls, delta_dir):

    plt_dir = os.path.dirname(delta_dir)+'/plots/'
    if os.path.isdir(plt_dir) is False:
        os.makedirs(plt_dir)

    sword_l4 = np.array([int(str(b)[0:4]) for b in sword.centerlines.reach_id[0,:]])
    basins_l4 = np.array([int(str(b)[0:4]) for b in delta_cls.basins])
    cl_keep = np.where(np.in1d(sword_l4, basins_l4)==True)[0]
    unq_rchs = np.unique(delta_cls.reach_id[0,:])
    dup = np.where(np.in1d(sword.reaches.id, unq_rchs)==True)[0]

    plt.figure(1, figsize=(8,8))
    plt.scatter(delta_cls.lon, delta_cls.lat, c='gold', s=3, label='Delta Additions')
    plt.scatter(sword.centerlines.x[cl_keep], sword.centerlines.y[cl_keep], c='blue', s=3, label='SWORD')
    if len(dup) > 0:
        plt.scatter(sword.reaches.x[dup], sword.reaches.y[dup], c='limegreen', s=10, label='Dupcliate Reach IDs')
    plt.xlim(np.min(delta_cls.lon)-0.1, np.max(delta_cls.lon)+0.1)
    plt.ylim(np.min(delta_cls.lat)-0.1, np.max(delta_cls.lat)+0.1)
    plt.title(os.path.basename(delta_dir)[:-3])
    plt.legend()
    plt.savefig(plt_dir + os.path.basename(delta_dir)[:-3]+'_additions.png')
    plt.close()

###############################################################################