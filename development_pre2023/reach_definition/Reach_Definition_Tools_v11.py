# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
                            Reach Definition Tools
-------------------------------------------------------------------------------
Copyright (c) 2018-2021 UNC Chapel Hill. All rights reserved.
Created by E. Altenau. Some functions were taken or modified from C. Lion's
"Tools.py" script (c) 2016.

DESCRIPTON:
    This script contains functions to aid in Reach and Topology Definition
    which forms the foundation of the SWOT River Database (SWORD). These
    functions and tools are called by the primarty script:
    Reach_Noce_Definition_v11.py.
-------------------------------------------------------------------------------
"""

from __future__ import division
import utm
import numpy as np
from scipy import spatial as sp
import netCDF4 as nc
from pyproj import Proj

###############################################################################/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/sa_sword_v17.nc
######################## Reach Definition Functions ###########################
###############################################################################

class Object(object):
    """
    FUNCTION:
        Creates class object to assign attributes to.
    """
    pass

###############################################################################

def read_merge_netcdf(filename):

    """
    FUNCTION:
        Reads in attributes from the merged database and assigns them to an
        object.

    INPUTS
        filename -- Merged database netcdf file.

    OUTPUTS
        data -- Object containing attributes from the merged database.
    """

    data = Object()
    new = nc.Dataset(filename)
    data.lon = new.groups['centerlines'].variables['x'][:]
    data.lat = new.groups['centerlines'].variables['y'][:]
    data.x = new.groups['centerlines'].variables['easting'][:]
    data.y = new.groups['centerlines'].variables['northing'][:]
    data.seg = new.groups['centerlines'].variables['segID'][:]
    data.ind = new.groups['centerlines'].variables['segInd'][:]
    data.id = new.groups['centerlines'].variables['cl_id'][:]
    data.segDist = new.groups['centerlines'].variables['segDist'][:]
    data.wth = new.groups['centerlines'].variables['p_width'][:]
    data.elv = new.groups['centerlines'].variables['p_height'][:]
    data.facc = new.groups['centerlines'].variables['flowacc'][:]
    data.lake = new.groups['centerlines'].variables['lakeflag'][:]
    data.delta = new.groups['centerlines'].variables['deltaflag'][:]
    data.nchan = new.groups['centerlines'].variables['nchan'][:]
    data.grand = new.groups['centerlines'].variables['grand_id'][:]
    data.grod = new.groups['centerlines'].variables['grod_id'][:]
    data.grod_fid = new.groups['centerlines'].variables['grod_fid'][:]
    data.hfalls_fid = new.groups['centerlines'].variables['hfalls_fid'][:]
    data.basins = new.groups['centerlines'].variables['basin_code'][:]
    data.manual = new.groups['centerlines'].variables['manual_add'][:]
    data.num_obs = new.groups['centerlines'].variables['number_obs'][:]
    data.orbits = new.groups['centerlines'].variables['orbits'][:]
    data.tile = new.groups['centerlines'].variables['grwl_tile'][:]
    data.eps = new.groups['centerlines'].variables['endpoints'][:]
    data.lake_id = new.groups['centerlines'].variables['lake_id'][:]
    new.close()

    return data

###############################################################################

def append_centerlines(centerlines, subcls, cnt):

    """
    FUNCTION:
        Appends sub-attributes within a loop to an object containing the final
        SWORD high-resolution centerline attributes for an entire specified
        region (in most cases a continent).

    INPUTS
        centerlines -- Object to be appended with sub-attribute data.
        subcls -- Object containing current attribute information for a single
            level 2 Pfafstetter basin at the high-resolution centerline loctions.
        cnt -- Specifies the current loop iteration.
    """

    # Copy the very first sub-attributes.
    if cnt == 0:
        centerlines.id = np.copy(subcls.id)
        centerlines.cl_id = np.copy(subcls.cl_id)
        centerlines.x = np.copy(subcls.lon)
        centerlines.y = np.copy(subcls.lat)
        centerlines.reach_id = np.copy(subcls.reach_id)
        centerlines.node_id = np.copy(subcls.node_id)

    # Otherwise, append the sub-attributes.
    else:
        centerlines.id = np.insert(centerlines.id, len(centerlines.id), np.copy(subcls.id))
        centerlines.cl_id = np.insert(centerlines.cl_id, len(centerlines.cl_id), np.copy(subcls.cl_id))
        centerlines.x = np.insert(centerlines.x, len(centerlines.x), np.copy(subcls.lon))
        centerlines.y = np.insert(centerlines.y, len(centerlines.y), np.copy(subcls.lat))
        centerlines.reach_id = np.insert(centerlines.reach_id, len(centerlines.reach_id), np.copy(subcls.reach_id), axis = 0)
        centerlines.node_id = np.insert(centerlines.node_id, len(centerlines.node_id), np.copy(subcls.node_id), axis = 0)

###############################################################################

def append_reaches(reaches, subreaches, cnt):

    """
    FUNCTION:
        Appends sub-attributes within a loop to an object containing the final
        SWORD reach attributes for an entire specified region
        (in most cases a continent).

    INPUTS
        reaches -- Object to be appended with sub-attribute data.
        subreaches -- Object containing current attribute information for a
            single level 2 Pfafstetter basin at the reach loctions.
        cnt -- Specifies the current loop iteration.
    """

    # Copy the very first sub-attributes.
    if cnt == 0:
        reaches.id = np.copy(subreaches.id)
        reaches.cl_id = np.copy(subreaches.cl_id)
        reaches.x = np.copy(subreaches.x)
        reaches.x_min = np.copy(subreaches.x_min)
        reaches.x_max = np.copy(subreaches.x_max)
        reaches.y = np.copy(subreaches.y)
        reaches.y_min = np.copy(subreaches.y_min)
        reaches.y_max = np.copy(subreaches.y_max)
        reaches.len = np.copy(subreaches.len)
        reaches.wse = np.copy(subreaches.wse)
        reaches.wse_var = np.copy(subreaches.wse_var)
        reaches.wth = np.copy(subreaches.wth)
        reaches.wth_var = np.copy(subreaches.wth_var)
        reaches.slope = np.copy(subreaches.slope)
        reaches.rch_n_nodes = np.copy(subreaches.rch_n_nodes)
        reaches.grod = np.copy(subreaches.grod)
        reaches.grod_fid = np.copy(subreaches.grod_fid)
        reaches.hfalls_fid = np.copy(subreaches.hfalls_fid)
        reaches.nchan_max = np.copy(subreaches.nchan_max)
        reaches.nchan_mod = np.copy(subreaches.nchan_mod)
        reaches.dist_out = np.copy(subreaches.dist_out)
        reaches.n_rch_up = np.copy(subreaches.n_rch_up_filt)
        reaches.n_rch_down = np.copy(subreaches.n_rch_down_filt)
        reaches.rch_id_up = np.copy(subreaches.rch_id_up_filt)
        reaches.rch_id_down = np.copy(subreaches.rch_id_down_filt)
        reaches.lakeflag = np.copy(subreaches.lakeflag)
        reaches.facc = np.copy(subreaches.facc)
        reaches.max_obs = np.copy(subreaches.max_obs)
        reaches.median_obs = np.copy(subreaches.median_obs)
        reaches.mean_obs = np.copy(subreaches.mean_obs)
        reaches.orbits = np.copy(subreaches.orbits)
        reaches.coverage = np.copy(subreaches.coverage)
        reaches.lake_id = np.copy(subreaches.lake_id)

	# Otherwise, append the sub-attributes.
    else:
        reaches.id = np.insert(reaches.id, len(reaches.id), np.copy(subreaches.id))
        reaches.cl_id = np.insert(reaches.cl_id, len(reaches.cl_id), np.copy(subreaches.cl_id), axis = 0)
        reaches.x = np.insert(reaches.x, len(reaches.x), np.copy(subreaches.x))
        reaches.x_min = np.insert(reaches.x_min, len(reaches.x_min), np.copy(subreaches.x_min))
        reaches.x_max = np.insert(reaches.x_max, len(reaches.x_max), np.copy(subreaches.x_max))
        reaches.y = np.insert(reaches.y, len(reaches.y), np.copy(subreaches.y))
        reaches.y_min = np.insert(reaches.y_min, len(reaches.y_min), np.copy(subreaches.y_min))
        reaches.y_max = np.insert(reaches.y_max, len(reaches.y_max), np.copy(subreaches.y_max))
        reaches.len = np.insert(reaches.len, len(reaches.len), np.copy(subreaches.len))
        reaches.wse = np.insert(reaches.wse, len(reaches.wse), np.copy(subreaches.wse))
        reaches.wse_var = np.insert(reaches.wse_var, len(reaches.wse_var), np.copy(subreaches.wse_var))
        reaches.wth = np.insert(reaches.wth, len(reaches.wth), np.copy(subreaches.wth))
        reaches.wth_var = np.insert(reaches.wth_var, len(reaches.wth_var), np.copy(subreaches.wth_var))
        reaches.slope = np.insert(reaches.slope, len(reaches.slope), np.copy(subreaches.slope))
        reaches.rch_n_nodes = np.insert(reaches.rch_n_nodes, len(reaches.rch_n_nodes), np.copy(subreaches.rch_n_nodes))
        reaches.grod = np.insert(reaches.grod, len(reaches.grod), np.copy(subreaches.grod))
        reaches.grod_fid = np.insert(reaches.grod_fid, len(reaches.grod_fid), np.copy(subreaches.grod_fid))
        reaches.hfalls_fid = np.insert(reaches.hfalls_fid, len(reaches.hfalls_fid), np.copy(subreaches.hfalls_fid))
        reaches.nchan_max = np.insert(reaches.nchan_max, len(reaches.nchan_max), np.copy(subreaches.nchan_max))
        reaches.nchan_mod = np.insert(reaches.nchan_mod, len(reaches.nchan_mod), np.copy(subreaches.nchan_mod))
        reaches.dist_out = np.insert(reaches.dist_out, len(reaches.dist_out), np.copy(subreaches.dist_out))
        reaches.n_rch_up = np.insert(reaches.n_rch_up, len(reaches.n_rch_up), np.copy(subreaches.n_rch_up_filt))
        reaches.n_rch_down = np.insert(reaches.n_rch_down, len(reaches.n_rch_down), np.copy(subreaches.n_rch_down_filt))
        reaches.rch_id_up = np.insert(reaches.rch_id_up, len(reaches.rch_id_up), np.copy(subreaches.rch_id_up_filt), axis = 0)
        reaches.rch_id_down = np.insert(reaches.rch_id_down, len(reaches.rch_id_down), np.copy(subreaches.rch_id_down_filt), axis = 0)
        reaches.lakeflag = np.insert(reaches.lakeflag, len(reaches.lakeflag), np.copy(subreaches.lakeflag))
        reaches.facc = np.insert(reaches.facc, len(reaches.facc), np.copy(subreaches.facc))
        reaches.max_obs = np.insert(reaches.max_obs, len(reaches.max_obs), np.copy(subreaches.max_obs))
        reaches.median_obs = np.insert(reaches.median_obs, len(reaches.median_obs), np.copy(subreaches.median_obs))
        reaches.mean_obs = np.insert(reaches.mean_obs, len(reaches.mean_obs), np.copy(subreaches.mean_obs))
        reaches.orbits = np.insert(reaches.orbits, len(reaches.orbits), np.copy(subreaches.orbits), axis = 0)
        reaches.coverage = np.insert(reaches.coverage, len(reaches.coverage), np.copy(subreaches.coverage), axis = 0)
        reaches.lake_id = np.insert(reaches.lake_id, len(reaches.lake_id), np.copy(subreaches.lake_id))

###############################################################################

def append_nodes(nodes, subnodes, cnt):

    """
    FUNCTION:
        Appends sub-attributes within a loop to an object containing the final
        SWORD node attributes for an entire specified region
        (in most cases a continent).

    INPUTS
        nodes -- Object to be appended with sub-attribute data.
        subnodes -- Object containing current attribute information for a
            single level 2 Pfafstetter basin at the node loctions.
        cnt -- Specifies the current loop iteration.
    """
    # Copy the very first sub-attributes.
    if cnt == 0:
        nodes.id = np.copy(subnodes.id)
        nodes.cl_id = np.copy(subnodes.cl_id)
        nodes.x = np.copy(subnodes.x)
        nodes.y = np.copy(subnodes.y)
        nodes.len = np.copy(subnodes.len)
        nodes.wse = np.copy(subnodes.wse)
        nodes.wse_var = np.copy(subnodes.wse_var)
        nodes.wth = np.copy(subnodes.wth)
        nodes.wth_var = np.copy(subnodes.wth_var)
        nodes.grod = np.copy(subnodes.grod)
        nodes.grod_fid = np.copy(subnodes.grod_fid)
        nodes.hfalls_fid = np.copy(subnodes.hfalls_fid)
        nodes.nchan_max = np.copy(subnodes.nchan_max)
        nodes.nchan_mod = np.copy(subnodes.nchan_mod)
        nodes.dist_out = np.copy(subnodes.dist_out)
        nodes.reach_id = np.copy(subnodes.reach_id)
        nodes.wth_coef = np.copy(subnodes.wth_coef)
        nodes.lakeflag = np.copy(subnodes.lakeflag)
        nodes.facc = np.copy(subnodes.facc)
        nodes.lake_id = np.copy(subnodes.lake_id)
    # Otherwise, append the sub-attributes.
    else:
        nodes.id = np.insert(nodes.id, len(nodes.id), np.copy(subnodes.id))
        nodes.cl_id = np.insert(nodes.cl_id, len(nodes.cl_id), np.copy(subnodes.cl_id), axis = 0)
        nodes.x = np.insert(nodes.x, len(nodes.x), np.copy(subnodes.x))
        nodes.y = np.insert(nodes.y, len(nodes.y), np.copy(subnodes.y))
        nodes.len = np.insert(nodes.len, len(nodes.len), np.copy(subnodes.len))
        nodes.wse = np.insert(nodes.wse, len(nodes.wse), np.copy(subnodes.wse))
        nodes.wse_var = np.insert(nodes.wse_var, len(nodes.wse_var), np.copy(subnodes.wse_var))
        nodes.wth = np.insert(nodes.wth, len(nodes.wth), np.copy(subnodes.wth))
        nodes.wth_var = np.insert(nodes.wth_var, len(nodes.wth_var), np.copy(subnodes.wth_var))
        nodes.grod = np.insert(nodes.grod, len(nodes.grod), np.copy(subnodes.grod))
        nodes.grod_fid = np.insert(nodes.grod_fid, len(nodes.grod_fid), np.copy(subnodes.grod_fid))
        nodes.hfalls_fid = np.insert(nodes.hfalls_fid, len(nodes.hfalls_fid), np.copy(subnodes.hfalls_fid))
        nodes.nchan_max = np.insert(nodes.nchan_max, len(nodes.nchan_max), np.copy(subnodes.nchan_max))
        nodes.nchan_mod = np.insert(nodes.nchan_mod, len(nodes.nchan_mod), np.copy(subnodes.nchan_mod))
        nodes.dist_out = np.insert(nodes.dist_out, len(nodes.dist_out), np.copy(subnodes.dist_out))
        nodes.reach_id = np.insert(nodes.reach_id, len(nodes.reach_id), np.copy(subnodes.reach_id))
        nodes.wth_coef = np.insert(nodes.wth_coef, len(nodes.wth_coef), np.copy(subnodes.wth_coef))
        nodes.lakeflag = np.insert(nodes.lakeflag, len(nodes.lakeflag), np.copy(subnodes.lakeflag))
        nodes.facc = np.insert(nodes.facc, len(nodes.facc), np.copy(subnodes.facc))
        nodes.lake_id = np.insert(nodes.lake_id, len(nodes.lake_id), np.copy(subnodes.lake_id))

###################################################w############################

def reproject_utm(latitude, longitude):

    """
    Modified from C. Lion's function by E. Altenau
    Copyright (c) 2018 UNC Chapel Hill. All rights reserved.

    FUNCTION:
        Projects all points in UTM.

    INPUTS
        latitude -- latitude in degrees (1-D array)
        longitude -- longitude in degrees (1-D array)

    OUTPUTS
        east -- easting in UTM (1-D array)
        north -- northing in UTM (1-D array)
        utm_num -- UTM zone number (1-D array of utm zone numbers for each point)
        utm_let -- UTM zone letter (1-D array of utm zone letters for each point)
    """

    east = np.zeros(len(latitude))
    north = np.zeros(len(latitude))
    east_int = np.zeros(len(latitude))
    north_int = np.zeros(len(latitude))
    zone_num = np.zeros(len(latitude))
    zone_let = []

	# Finds UTM letter and zone for each lat/lon pair.

    for ind in list(range(len(latitude))):
        (east_int[ind], north_int[ind],
	 zone_num[ind], zone_let_int) = utm.from_latlon(latitude[ind],
	                                                longitude[ind])
        zone_let.append(zone_let_int)

    # Finds the unique UTM zones and converts the lat/lon pairs to UTM.
    unq_zones = np.unique(zone_num)
    utm_let = np.unique(zone_let)[0]

    for idx in list(range(len(unq_zones))):
        pt_len = len(np.where(zone_num == unq_zones[idx])[0])

    idx = np.where(pt_len == np.max(pt_len))

    # Set the projection

    if np.sum(latitude) > 0:
        myproj = Proj(
		"+proj=utm +zone=" + str(int(unq_zones[idx])) + utm_let +
		" +ellips=WGS84 +datum=WGS84 +units=m")
    else:
        myproj = Proj(
		"+proj=utm +south +zone=" + str(int(unq_zones[idx])) + utm_let +
		" +ellips=WGS84 +datum=WGS84 +units=m")

    # Convert all the lon/lat to the main UTM zone
    (east, north) = myproj(longitude, latitude)

    return east, north, zone_num, zone_let

###############################################################################

def calc_segDist(subcls_lon, subcls_lat, subcls_rch_id, subcls_facc,
                 subcls_rch_ind):

    """
    FUNCTION:
        Creates a 1-D array of flow distances for each specified reach. Flow
        distance is build to start at 0 and increases in the upstream direction.

    INPUTS
        subcls -- Object containing reach and node attributes for the
            high-resolution centerline.
            [attributes used]:
                lon -- Longitude values along the high-resolution centerline.
                lat -- Latitude values along the high-resolution centerline.
                facc -- Flow accumulation along the high-resolution centerline.
                rch_id5 -- Reach numbers after aggregating short river, lake,
                    and dam reaches.

    OUTPUTS
        segDist -- Flow distance per reach (meters).
    """

    #loop through each reach and calculate flow distance.
    segDist = np.zeros(len(subcls_lon))
    uniq_segs = np.unique(subcls_rch_id)
    for ind in list(range(len(uniq_segs))):
        #for a single reach, reproject lat-lon coordinates to utm.
        #print(ind, uniq_segs[ind])
        seg = np.where(subcls_rch_id == uniq_segs[ind])[0]
        seg_lon = subcls_lon[seg]
        seg_lat = subcls_lat[seg]
        seg_x, seg_y, __, __ = reproject_utm(seg_lat, seg_lon)
        upa = subcls_facc[seg]

        #order the reach points based on index values, then calculate the
        #eculdean distance bewteen each ordered point.
        order_ids = np.argsort(subcls_rch_ind[seg])
        dist = np.zeros(len(seg))
        dist[order_ids[0]] = 0
        for idx in list(range(len(order_ids)-1)):
            d = np.sqrt((seg_x[order_ids[idx]]-seg_x[order_ids[idx+1]])**2 +
                        (seg_y[order_ids[idx]]-seg_y[order_ids[idx+1]])**2)
            dist[order_ids[idx+1]] = d + dist[order_ids[idx]]

        #format flow distance as array and determine flow direction by flowacc.
        dist = np.array(dist)
        start = upa[np.where(dist == np.min(dist))[0][0]]
        end = upa[np.where(dist == np.max(dist))[0][0]]

        if end > start:
            segDist[seg] = abs(dist-np.max(dist))

        else:
            segDist[seg] = dist

    return segDist

###############################################################################

def find_swot_bounds(seg, orbs, ID, dist, cnt):

    """
    FUNCTION:
        Finds the boundaries of the SWOT orbits along the high-resolution
        centerline locations. This is a sub-function of the "find_all_bounds"
        function.

    INPUTS
        seg -- Current GRWL segment.
        orbs -- SWOT orbits that cover the current segment.
        ID -- Point indexes for the current segment.
        dist -- Flow distance for the current segment.
        cnt -- Count that numbers the various SWOT intersections.

    OUTPUTS
        binary_orbits -- A binary 1-D array containing SWOT orbit overpass
            locations (1 = overpass boundary, 0 = no overpass boundary).
        swot_id -- 1-D array containing reaches defined by the overpass
            boundaries.
        swot_dist -- 1-D array containing reach lengths for the SWOT defined
            reaches.
    """

    # Identifying all swot orbit boundaries.
    uniq_orbits = np.unique(orbs)
    uniq_orbits = uniq_orbits[np.where(uniq_orbits>0)[0]]
    binary_orbits = np.zeros(len(seg))
    for idz in list(range(len(uniq_orbits))):
        temp_array = np.zeros(len(seg))
        rows = np.where(orbs == uniq_orbits[idz])[0]
        temp_array[rows] = 1
        bounds = np.where(np.diff(temp_array) != 0)[0]
        binary_orbits[bounds] = 1

    binary_orbits[np.where(ID == np.min(ID))] = 1
    binary_orbits[np.where(ID == np.max(ID))] = 1

    # Assigning an ID to the reaches defined by the swot orbit boundaries.
    border_ids = ID[np.where(binary_orbits == 1)[0]]
    borders = np.where(binary_orbits == 1)[0]
    borders = borders[np.argsort(border_ids)]

    # Empty arrays to fill.
    swot_dist = np.zeros(len(seg))
    swot_id = np.zeros(len(seg))
    cnt=cnt

    #condition for only one swot orbit.
    if len(borders) <= 2:
        swot_dist = np.max(dist)
        swot_id = cnt
        cnt=cnt+1

    #condition for multiple swot orbits.
    else:
        for idy in list(range(len(borders)-1)):
            index1 = borders[idy]
            index2 = borders[idy+1]

            ID1 = ID[index1]
            ID2 = ID[index2]

            if ID1 > ID2:
                vals = np.where((ID2 <= ID) &  (ID <= ID1))[0]
            else:
                vals = np.where((ID1 <= ID) &  (ID <= ID2))[0]

            swot_dist[vals] = abs(np.max(dist[vals])-np.min(dist[vals]))
            swot_id[vals] = cnt
            cnt=cnt+1
            #print(np.unique(swot_dist[vals]), np.unique(swot_id[vals]))

    return binary_orbits, swot_id, swot_dist

###############################################################################

def find_lake_bounds(seg, lakes, ID, dist, cnt):

    """
    FUNCTION:
        Finds the boundaries of lakes residing along the high-resolution
        centerline locations. This is a sub-function of the "find_all_bounds"
        function.

    INPUTS
        seg -- Current GRWL segment.
        lakes -- Lake IDs within the current segment.
        ID -- Point indexes for the current segment.
        dist -- Flow distance for the current segment.
        cnt -- Count that numbers the various lake intersections.

    OUTPUTS
        lake_bounds -- A binary 1-D array containing lake intersection
            locations (1 = lake boundary, 0 = no lake boundary).
        lake_id -- 1-D array containing reaches defined by the intersecting
            lakes.
        lake_dist -- 1-D array containing reach lengths for the lake defined
            reaches.
    """

    # Finding all lake intersection boundaries.
    binary_lakes = np.zeros(len(seg))
    bounds = np.where(np.diff(lakes) != 0)[0]
    binary_lakes[bounds] = 1
    binary_lakes[np.where(ID == np.min(ID))] = 1
    binary_lakes[np.where(ID == np.max(ID))] = 1

    # Assiging IDs to reaches defined by lake boundaries.
    borders = np.where(binary_lakes == 1)[0]
    # Empty arrays to fill.
    lake_dist = np.zeros(len(seg))
    lake_id = np.zeros(len(seg))
    lake_bounds = np.zeros(len(seg))
    cnt=cnt

    #condition for one lake boundary.
    if len(borders) <= 2:
        mode = max(set(list(lakes)), key=list(lakes).count)
        if mode > 0:
            lake_dist = np.max(dist)
            lake_id = cnt
            lake_bounds[borders] = 1
            cnt=cnt+1

    #condition for multiple lake boundaries.
    else:
        for idy in list(range(len(borders)-1)):
            index1 = borders[idy]
            index2 = borders[idy+1]
            id1 = ID[index1]
            id2 = ID[index2]
            if index2 == max(borders):
                index2 = index2+1
            #find values based on indexes.
            if id1 > id2:
                vals = np.where((ID >= id2) & (ID <= id1))[0]
            else:
                vals = np.where((ID >= id1) & (ID <= id2))[0]
            # find mode values of grwl flag.
            mode = max(set(list(lakes[vals])),
                       key=list(lakes[vals]).count)
            if mode > 0:
                lake_dist[vals] = abs(dist[borders[idy+1]]-dist[borders[idy]])
                lake_id[vals] = cnt
                lake_bounds[index1] = 1
                if index2 > np.max(borders):
                    lake_bounds[index2-1] = 1
                if index2 < np.max(borders):
                    lake_bounds[index2] = 1
                cnt=cnt+1

    return lake_bounds, lake_id, lake_dist

###############################################################################

def find_dam_bounds(seg, dams, dist, cnt, ID, radius):

    """
    FUNCTION:
        Finds the boundaries of dams residing along the high-resolution
        centerline locations. This is a sub-function of the "find_all_bounds"
        function.

    INPUTS
        seg -- Current GRWL segment.
        dams -- Dam IDs within the current segment.
        dist -- Flow distance for the current segment.
        cnt -- Count that numbers the various dam intersections
        radius -- Desired length of the reach to be formed around the dam.

    OUTPUTS
        dam_bounds -- A binary 1-D array containing dam reach boundaries
            (1 = dam boundary, 0 = no dam boundary).
        dam_id -- 1-D array containing reaches defined by the intersecting
            dams.
        dam_dist -- 1-D array containing reach lengths for the dam defined
            reaches.
    """

    # Empty arrays to fill.
    dam_bounds = np.zeros(len(seg))
    dam_id = np.zeros(len(seg))
    dam_dist = np.zeros(len(seg))
    cnt = cnt
    r = radius

    # Loop through each dam location and define surrounding reach locations
    # based on the specified radius.
    for ind in list(range(len(dams))):
        # Defining flow distances around dam location based on radius (aka desired dam length).
        if dist[dams[ind]] <= r:
            new_r = 400 - dist[dams[ind]]
            lower_bound = np.min(dist)
            upper_bound = dist[dams[ind]]+new_r

        if dist[dams[ind]] >= (dist[dams[ind]]+r):
            new_r = np.max(dist) - 400
            lower_bound = new_r
            upper_bound = np.max(dist)

        elif r < dist[dams[ind]] < (dist[dams[ind]]+r):
            lower_bound = dist[dams[ind]]-r
            upper_bound = dist[dams[ind]]+r

        # Finding existing flow distances closest to the calculated boundaries.
        if dist[dams[ind]] == 0:
            lower_dist = 0
        else:
            lower_dist = np.min(dist[np.where((dist-lower_bound) > 0)[0]])
        upper_dist = np.max(dist[np.where((dist-upper_bound) < 0)[0]])
        bound1 = np.int(np.where(dist == lower_dist)[0][0])
        bound2 = np.int(np.where(dist == upper_dist)[0][0])
        # Assigning a binary value to the identified locations.
        dam_bounds[bound1] = 1
        dam_bounds[bound2] = 1
        index1 = ID[bound1]
        index2 = ID[bound2]

        # Assign IDs to reaches defined by the dam locations.
        if index1 > index2:
            vals = np.where((ID >= index2) & (ID <= index1))[0]
            dam_dist[vals] = abs(dist[bound2] - dist[bound1])
            dam_id[vals] = cnt
            cnt = cnt+1
        else:
            vals = np.where((ID >= index1) & (ID <= index2))[0]
            dam_dist[vals] = abs(dist[bound1] - dist[bound2])
            dam_id[vals] = cnt
            cnt = cnt+1

    return dam_bounds, dam_id, dam_dist

###############################################################################

def find_all_bounds(subcls, radius):

    """
    FUNCTION:
        Loops through each basin and finds SWOT orbit, lake, and dam boundaries,
        and creates a type flag based on each boundary.

    INPUTS
        subcls -- Object containing attributes for the high-resolution
            centerline.
            [attributes used]:
                seg -- GRWL segment values along the high-resolution centerline.
                ind -- Point indexes for each GRWL segment along the
                    high-resolution centerline.
                basins -- Pfafstetter basin codes along the high-resolution
                    centerline.
                lake -- Water body type values along the high-resolution
                    centerline (0 = river, 1 = lake/reservior, 2 = canal,
                    3 = tidal).
                delta -- Delta flag along the high-resolution centerline
                    (0 = no delta, 1 = delta)
                dist -- Flow distance along the high-resolution centerline.
                orbits -- SWOT orbit locations along the high-resolution
                    centerline.
                grod -- GROD IDs along the high-resolution centerline.

        radius -- Desired length for the dam reaches.

    OUTPUTS
        swot_bounds -- A binary 1-D array containing SWOT orbit overpass
            locations (1 = overpass boundary, 0 = no overpass boundary).
        swot_id -- 1-D array containing reaches defined by the overpass
            boundaries.
        swot_dist -- 1-D array containing reach lengths for the SWOT defined
            reaches.
        lake_bounds -- A binary 1-D array containing lake intersection
            locations (1 = lake boundary, 0 = no lake boundary).
        lake_id -- 1-D array containing reaches defined by the intersecting
            lakes.
        lake_dist -- 1-D array containing reach lengths for the lake defined
            reaches.
        dam_bounds -- A binary 1-D array containing dam reach boundaries
            (1 = dam boundary, 0 = no dam boundary).
        dam_id -- 1-D array containing reaches defined by the intersecting
            dams.
        dam_dist -- 1-D array containing reach lengths for the dam defined
            reaches.
        all_bounds -- All boundary locations.
        Type -- Type of water body between each boundary (1 = river, 2 = lake,
            3 = lake on river, 4 = dam, 5 = no topology)
    """

    # Combine lake and delta flags into one vector.
    lake_coast_flag = np.copy(subcls.lake)
    lake_coast_flag[np.where(subcls.delta > 0)] = 3

    # Set variables.
    uniq_basins = np.unique(subcls.basins)
    lake_bounds = np.zeros(len(subcls.ind))
    lake_dist = np.zeros(len(subcls.ind))
    lake_id = np.zeros(len(subcls.ind))
    swot_bounds = np.zeros(len(subcls.ind))
    swot_dist = np.zeros(len(subcls.ind))
    swot_id = np.zeros(len(subcls.ind))
    dam_bounds = np.zeros(len(subcls.ind))
    dam_dist = np.zeros(len(subcls.ind))
    dam_id = np.zeros(len(subcls.ind))
    swot_cnt = 1
    lake_cnt = 1
    dam_cnt = 1
    radius = radius

    # Loop through each basin and identify SWOT orbit, lake, and dam boundaries.
    for ind in list(range(len(uniq_basins))):
        basin = np.where(subcls.basins == uniq_basins[ind])[0]
        uniq_segs = np.unique(subcls.seg[basin])
        for idx in list(range(len(uniq_segs))):
            seg = np.where(subcls.seg[basin] == uniq_segs[idx])[0]
            ID = subcls.ind[basin[seg]]
            dist = subcls.dist[basin[seg]]
            lakes = lake_coast_flag[basin[seg]]
            orbs = subcls.orbits[basin[seg]]
            grod = subcls.grod[basin[seg]]
            dams = np.where((grod > 0) & (grod <= 4))[0]

            # Lake boundaries.
            lb, li, ld = find_lake_bounds(seg, lakes, ID, dist, lake_cnt)
            lake_bounds[basin[seg]] = lb
            lake_dist[basin[seg]] = ld
            lake_id[basin[seg]] = li
            lake_cnt = np.max(lake_id)+1
            # SWOT orbit boundaries.
            sb, si, sd = find_swot_bounds(seg, orbs, ID, dist, swot_cnt)
            swot_bounds[basin[seg]] = sb
            swot_dist[basin[seg]] = sd
            swot_id[basin[seg]] = si
            swot_cnt = np.max(swot_id)+1
            # Dam boundaries.
            db, di, dd = find_dam_bounds(seg, dams, dist, dam_cnt, ID, radius)
            dam_bounds[basin[seg]] = db
            dam_dist[basin[seg]] = dd
            dam_id[basin[seg]] = di
            dam_cnt = np.max(dam_id)+1
            #if np.min(dd) == 0:
                #print(ind, idx)

        # Erase lake boundaries within dam boundaries.
        lake_dam_bounds =  np.copy(lake_bounds)
        lake_dam_bounds[np.where(dam_id > 0)] = 0
        lake_dam_bounds[np.where(dam_bounds > 0)] = 1

        # Combine all the boundaries into one binary, 1-D array.
        all_bounds = np.copy(swot_bounds)
        all_bounds[np.where(lake_id > 0)] = 0
        all_bounds[np.where(dam_id > 0)] = 0
        all_bounds[np.where(lake_dam_bounds > 0)] = 1

        # Create reach "type" flag based on boundaries.
        Type = np.zeros(len(subcls.ind))
        Type[np.where(Type == 0)] = 1
        Type[np.where(lake_id > 0)] = 3
        Type[np.where(subcls.lake == 2)] = 1
        Type[np.where(subcls.delta > 0)] = 5
        Type[np.where(dam_id > 0)] = 4

    return(lake_bounds, lake_dist, lake_id, dam_bounds, dam_dist, dam_id,
           swot_bounds, swot_dist, swot_id, all_bounds, Type)

###############################################################################

def find_initial_reaches(seg, bounds, ID, dist, cnt):

    """
    FUNCTION:
        Assigns a unique ID to each reach falling between a boundary. This is
        a sub-function of the "number_reaches" function.

    INPUTS
        seg -- Current GRWL segment (1-D array).
        bounds -- Identifies boundaries along the segment (1-D array).
        ID -- Point indexes for the segment (1-D array).
        dist -- Flow distance along the segment (1-D array).
        cnt -- Start value for the unique IDs assigned to the reaches (value).

    OUTPUTS
        rch_id -- 1-D array of unique reach IDs.
        rch_dist -- 1-D array of reach lengths (meters).
    """

    # Formatting boundaries.
    binary_orbits = np.copy(bounds)
    binary_orbits[np.where(ID == np.min(ID))] = 1
    binary_orbits[np.where(ID == np.max(ID))] = 1
    border_ids = ID[np.where(binary_orbits == 1)[0]]
    # Adding first and last segment points to the boundaries array.
    borders = np.where(binary_orbits == 1)[0]
    borders = borders[np.argsort(border_ids)]

    # Set variables.
    rch_dist = np.zeros(len(seg))
    rch_id = np.zeros(len(seg))
    cnt=cnt

    # Create ID for each reach between the boundaries.
    if len(borders) <= 2:
        rch_dist = np.max(dist)
        if rch_dist == 0:
            rch_dist = 30.0
        rch_id = cnt
        cnt=cnt+1

    else:
        for idy in list(range(len(borders)-1)):
            index1 = borders[idy]
            index2 = borders[idy+1]

            ID1 = ID[index1]
            ID2 = ID[index2]

            if ID1 > ID2:
                vals = np.where((ID2 <= ID) &  (ID <= ID1))[0]
            else:
                vals = np.where((ID1 <= ID) &  (ID <= ID2))[0]

            #if index2 == max(borders):
                #index2 = index2+1

            dist_diff = abs(np.max(dist[vals])-np.min(dist[vals]))

            if dist_diff == 0:
                rch_dist[vals] = 30.0
            else:
                rch_dist[vals] = dist_diff

            rch_id[vals] = cnt
            cnt=cnt+1

    return rch_id, rch_dist

###############################################################################

def number_reaches(subcls):

    """
    FUNCTION:
        Loops through each basin and saves a unique ID to each reach defined
        by the SWOT, lake, and dam boundaries.

    INPUTS
        subcls -- Object containing attributes along the high-resolution
            centerline.
            [attributes used]:
                basins -- Pfafstetter basin codes along the high-resolution
                    centerline.
                seg -- GRWL segment values along the high-resolution centerline.
                dist -- Flow distance along the high-resolution centerline.
                ind -- Point indexes for each GRWL segment along the
                    high-resolution centerline.
                all_bnds -- All boundary locations for lakes, dams, and SWOT
                    overpasses.

    OUTPUTS
        rch_id -- 1-D array of unique reach IDs.
        rch_dist -- 1-D array of reach lengths (meters).
    """

    uniq_basins = np.unique(subcls.basins)
    rch_dist = np.zeros(len(subcls.ind))
    rch_id = np.zeros(len(subcls.ind))
    all_cnt = 1
    for ind in list(range(len(uniq_basins))):
        basin = np.where(subcls.basins == uniq_basins[ind])[0]
        uniq_segs = np.unique(subcls.seg[basin])
        for idx in list(range(len(uniq_segs))):
            seg = np.where(subcls.seg[basin] == uniq_segs[idx])[0]
            ID = subcls.ind[basin[seg]]
            dist = subcls.dist[basin[seg]]
            all_bnds = subcls.all_bnds[basin[seg]]

            ri, rd = find_initial_reaches(seg, all_bnds, ID, dist, all_cnt)
            rch_dist[basin[seg]] = rd
            rch_id[basin[seg]] = ri
            all_cnt = np.max(rch_id)+1
            #if np.min(rd) == 0:
                #print(ind, idx, 'Reach Distance = 0')

    return(rch_id, rch_dist)

###############################################################################

def cut_reaches(subcls_rch_id0, subcls_rch_len0, subcls_dist,
                subcls_ind, max_dist):

    """
    FUNCTION:
        Divides reaches with lengths greater than a specified maximum distance
        into smaller reaches of similar length.

    INPUTS
        subcls -- Object containing attributes along the high-resolution
            centerline.
            [attributes used]:
                rch_id1 -- Reach numbers for the original reach boundaries.
                rch_len1 -- Reach lengths for the original reach boundaries.
                dist -- Flow distance along the high-resolution centerline.
                ind -- Point indexes for each GRWL segment along the
                    high-resolution centerline.
        max_dist -- Desired maximum reach length. The script will cut reaches
            greater than the specified max_dist.

    OUTPUTS
        new_rch_id -- 1-D array of updated reach IDs.
        new_rch_dist -- 1-D array of updated reach lengths (meters).
    """

    # Setting variables.
    cnt = np.max(subcls_rch_id0)+1
    new_rch_id = np.copy(subcls_rch_id0)
    new_rch_dist = np.copy(subcls_rch_len0)
    uniq_rch = np.unique(subcls_rch_id0[np.where(subcls_rch_len0 >= max_dist)])

    # Loop through each reach that is greater than the maximum distance and
    # divide it into smaller reaches.
    for ind in list(range(len(uniq_rch))):

        # Finding current reach id and length.
        rch = np.where(subcls_rch_id0 == uniq_rch[ind])[0]
        distance = subcls_dist[rch]
        ID = subcls_ind[rch]
        # Setting temporary variables.
        temp_rch_id = np.zeros(len(rch))
        temp_rch_dist = np.zeros(len(rch))
        # Determining the number of divisions to cut the reach.
        d = np.unique(subcls_rch_len0[rch])
        divs = np.around(d/10000)
        divs_dist = d/divs

        # Determining new boundaries to cut the reaches.
        break_index = np.zeros(np.int(divs-1))
        for idx in range(int(divs)-1):
            dist = divs_dist*(range(int(divs)-1)[idx]+1)+np.min(distance)
            cut = np.where(abs(distance - dist) == np.min(abs(distance - dist)))[0][0]
            break_index[idx] = cut
        div_ends = np.array([np.where(ID == np.min(ID))[0][0],np.where(ID == np.max(ID))[0][0]])
        borders = np.insert(div_ends, 0, break_index)
        border_ids = ID[borders]
        borders = borders[np.argsort(border_ids)]

        # Numbering the new cut reaches.
        for idy in list(range(len(borders)-1)):
            index1 = borders[idy]
            index2 = borders[idy+1]

            ID1 = ID[index1]
            ID2 = ID[index2]

            if ID1 > ID2:
                vals = np.where((ID2 <= ID) &  (ID <= ID1))[0]
            else:
                vals = np.where((ID1 <= ID) &  (ID <= ID2))[0]

            avg_dist = abs(np.max(distance[vals])-np.min(distance[vals]))
            if avg_dist == 0:
                temp_rch_dist[vals] = 30.0
            else:
                temp_rch_dist[vals] = avg_dist

            temp_rch_id[vals] = cnt
            cnt=cnt+1

        new_rch_id[rch] = temp_rch_id
        new_rch_dist[rch] = temp_rch_dist
        #if np.max(new_rch_dist[rch])>max_dist:
            #print(ind, 'max distance too long - likely an index problem')

    return new_rch_id, new_rch_dist

###############################################################################

def find_neighbors(basin_rch, basin_dist, basin_flag, basin_acc, basin_wse,
                   basin_x, basin_y, rch_x, rch_y, rch_ind, rch_len, rch_id, rch):

    """
    FUNCTION:
        Finds neighboring reaches that are within the current reach's basin.
        This is a sub-function used in the following functions:
        "aggregate_rivers", "aggregate_lakes", "aggregate_dams", and
        "order_reaches."

    INPUTS
        basin_rch -- All reach IDs within the basin.
        basin_dist -- All reach lengths for the reaches in the basin.
        basin_flag -- All reach types for the basin.
        basin_acc -- All flow accumulation values for the reaches in the basin.
        basin_wse -- All elevation values for the reaches in the basin.
        basin_x -- Easting values for all points in the basin.
        basin_y -- Northing values for all points in the basin.
        rch_x -- Easting values for the current reach.
        rch_y -- Northing values for the current reach.
        rch_ind -- Point indexes for the current reach.
        rch_len -- Reach length for the current reach.
        rch_id -- ID of the current reach.
        rch -- Index locations for the current reach.

    OUTPUTS
        ep1 -- Array of neighboring reach attributes. for the first segment
            endpoint. The array row dimensions will depend on the number of
            neighbors for that endpoint, while the column dimension will always
            be equal to five and contain following attributes for each
            neighbor: (1) reach ID, (2) reach length, (3) reach type,
            (4) reach flow accumulation, and (5) reach water surface elevation).
            For example, if the segment endpoint has two neighbors the array
            size would be [2,5], and if the segment endpoint only has one
            neighbor the array size would be [1,5]).
        ep2 -- Array of neighboring reach attributes for the second segment
            endpoint. The array row dimensions will depend on the number of
            neighbors for that endpoint, while the column dimension will always
            be equal to five and contain following attributes for each
            neighbor: (1) reach ID, (2) reach length, (3) reach type,
            (4) reach flow accumulation, and (5) reach water surface elevation).
            For example, if the segment endpoint has two neighbors the array
            size would be [2,5], and if the segment endpoint only has one
            neighbor the array size would be [1,5]).
    """

    # Formatting all basin coordinate values.
    basin_pts = np.vstack((basin_x, basin_y)).T
    # Formatting the current reach's endpoint coordinates.
    if len(rch) == 1:
        eps = np.array([0,0])
    else:
        pt1 = np.where(rch_ind == np.min(rch_ind))[0][0]
        pt2 = np.where(rch_ind == np.max(rch_ind))[0][0]
        eps = np.array([pt1,pt2]).T

    # Performing a spatial query to get the closest points within the basin
    # to the current reach's endpoints.
    ep_pts = np.vstack((rch_x[eps], rch_y[eps])).T
    kdt = sp.cKDTree(basin_pts)

    if rch_len < 100:
        pt_dist, pt_ind = kdt.query(ep_pts, k = 4, distance_upper_bound = 100.0)
    elif 100 <= rch_len and rch_len <= 200:
        pt_dist, pt_ind = kdt.query(ep_pts, k = 10, distance_upper_bound = 100.0)
    elif rch_len > 200:
        pt_dist, pt_ind = kdt.query(ep_pts, k = 10, distance_upper_bound = 200.0)

    # Identifying endpoint neighbors.
    ep1_ind = pt_ind[0,:]
    ep1_dist = pt_dist[0,:]
    na1 = np.where(ep1_ind == len(basin_rch))
    ep1_dist = np.delete(ep1_dist, na1)
    ep1_ind = np.delete(ep1_ind, na1)
    s1 = np.where(basin_rch[ep1_ind] == rch_id)
    ep1_dist = np.delete(ep1_dist, s1)
    ep1_ind = np.delete(ep1_ind, s1)
    ep1_ngb = np.unique(basin_rch[ep1_ind])

    ep2_ind = pt_ind[1,:]
    ep2_dist = pt_dist[1,:]
    na2 = np.where(ep2_ind == len(basin_rch))
    ep2_dist = np.delete(ep2_dist, na2)
    ep2_ind = np.delete(ep2_ind, na2)
    s2 = np.where(basin_rch[ep2_ind] == rch_id)
    ep2_dist = np.delete(ep2_dist, s2)
    ep2_ind = np.delete(ep2_ind, s2)
    ep2_ngb = np.unique(basin_rch[ep2_ind])

    # Pulling attribute information for the endpoint neighbors.
    ep1_len = np.zeros(len(ep1_ngb))
    ep1_flg = np.zeros(len(ep1_ngb))
    ep1_acc = np.zeros(len(ep1_ngb))
    ep1_wse = np.zeros(len(ep1_ngb))
    for idy in list(range(len(ep1_ngb))):
        ep1_len[idy] = np.unique(basin_dist[np.where(basin_rch == ep1_ngb[idy])])
        ep1_flg[idy] = np.max(basin_flag[np.where(basin_rch == ep1_ngb[idy])])
        ep1_acc[idy] = np.median(basin_acc[np.where(basin_rch == ep1_ngb[idy])])
        ep1_wse[idy] = np.median(basin_wse[np.where(basin_rch == ep1_ngb[idy])])

    ep2_len = np.zeros(len(ep2_ngb))
    ep2_flg = np.zeros(len(ep2_ngb))
    ep2_acc = np.zeros(len(ep2_ngb))
    ep2_wse = np.zeros(len(ep2_ngb))
    for idy in list(range(len(ep2_ngb))):
        ep2_len[idy] = np.unique(basin_dist[np.where(basin_rch == ep2_ngb[idy])])
        ep2_flg[idy] = np.max(basin_flag[np.where(basin_rch == ep2_ngb[idy])])
        ep2_acc[idy] = np.median(basin_acc[np.where(basin_rch == ep2_ngb[idy])])
        ep2_wse[idy] = np.median(basin_wse[np.where(basin_rch == ep2_ngb[idy])])

    # Creating final arrays.
    ep1 = np.array([ep1_ngb, ep1_len, ep1_flg, ep1_acc, ep1_wse]).T
    ep2 = np.array([ep2_ngb, ep2_len, ep2_flg, ep2_acc, ep2_wse]).T

    return ep1, ep2

###############################################################################

def find_all_neighbors(basin_rch, basin_dist, basin_flag, basin_acc, basin_wse,
                       rch_len, rch_id, eps_ind, eps_dist):

    """
    FUNCTION:
        Finds neighbors for all reaches within a region or large basin
        (usually a Pfafstetter level 2 basin). This is a sub-function
        of the "reach_attributes" function where all neighboring reaches, even
        those in different level 6 basins are needed to define upstream and
        downstream reaches.

    INPUTS
        basin_rch -- All reach IDs within the basin.
        basin_dist -- All reach lengths for the reaches in the basin.
        basin_flag -- All reach types for the basin.
        basin_acc -- All flow accumulation values for the reaches in the basin.
        basin_wse -- All elevation values for the reaches in the basin.
        rch_len -- Reach length for the current reach.
        rch_id -- ID of the current reach.
        eps_ind -- Spatial query array containing closest point indexes
            for each point in a basin.
        eps_dist -- Spatial query array containing closest point distances
            for each point in a basin.

    OUTPUTS
        ep1 -- Array of neighboring reach attributes. for the first segment
            endpoint. The array row dimensions will depend on the number of
            neighbors for that endpoint, while the column dimension will always
            be equal to five and contain following attributes for each
            neighbor: (1) reach ID, (2) reach length, (3) reach type,
            (4) reach flow accumulation, and (5) reach water surface elevation).
            For example, if the segment endpoint has two neighbors the array
            size would be [2,5], and if the segment endpoint only has one
            neighbor the array size would be [1,5]).
        ep2 -- Array of neighboring reach attributes for the second segment
            endpoint. The array row dimensions will depend on the number of
            neighbors for that endpoint, while the column dimension will always
            be equal to five and contain following attributes for each
            neighbor: (1) reach ID, (2) reach length, (3) reach type,
            (4) reach flow accumulation, and (5) reach water surface elevation).
            For example, if the segment endpoint has two neighbors the array
            size would be [2,5], and if the segment endpoint only has one
            neighbor the array size would be [1,5]).
    """

    # Formatting spatial query arrays.
    if len(eps_ind) == 0:
        ep1 = []
        ep2 = np.array([])
    else:
        if rch_len < 100:
            eps_ind[np.where(eps_dist > 100)] = len(basin_rch)
            eps_dist[np.where(eps_dist > 100)] = 'inf'
            pt_ind = eps_ind
            pt_dist = eps_dist
        elif 100 <= rch_len and rch_len <= 300: #use to be 200 before ghost reaches.
            eps_ind[np.where(eps_dist > 100)] = len(basin_rch)
            eps_dist[np.where(eps_dist > 100)] = 'inf'
            pt_ind = eps_ind
            pt_dist = eps_dist
        elif rch_len > 300: #use to be 200 before ghost reaches.
            pt_ind = eps_ind
            pt_dist = eps_dist

        # Identifying first endpoint neighbors.
        ep1_ind = pt_ind[0,:]
        ep1_dist = pt_dist[0,:]
        na1 = np.where(ep1_ind == len(basin_rch))
        ep1_dist = np.delete(ep1_dist, na1)
        ep1_ind = np.delete(ep1_ind, na1)
        s1 = np.where(basin_rch[ep1_ind] == rch_id)
        ep1_dist = np.delete(ep1_dist, s1)
        ep1_ind = np.delete(ep1_ind, s1)
        ep1_ngb = np.unique(basin_rch[ep1_ind])

        # Pulling first endpoint neighbors' attributes.
        ep1_len = np.zeros(len(ep1_ngb))
        ep1_flg = np.zeros(len(ep1_ngb))
        ep1_acc = np.zeros(len(ep1_ngb))
        ep1_wse = np.zeros(len(ep1_ngb))
        for idy in list(range(len(ep1_ngb))):
            ep1_len[idy] = np.unique(basin_dist[np.where(basin_rch == ep1_ngb[idy])])
            ep1_flg[idy] = np.max(basin_flag[np.where(basin_rch == ep1_ngb[idy])])
            ep1_acc[idy] = np.median(basin_acc[np.where(basin_rch == ep1_ngb[idy])])
            ep1_wse[idy] = np.median(basin_wse[np.where(basin_rch == ep1_ngb[idy])])

        ep1 = np.array([ep1_ngb, ep1_len, ep1_flg, ep1_acc, ep1_wse]).T

        if len(eps_ind) > 1:
            # Identifying second endpoint neighbors.
            ep2_ind = pt_ind[1,:]
            ep2_dist = pt_dist[1,:]
            na2 = np.where(ep2_ind == len(basin_rch))
            ep2_dist = np.delete(ep2_dist, na2)
            ep2_ind = np.delete(ep2_ind, na2)
            s2 = np.where(basin_rch[ep2_ind] == rch_id)
            ep2_dist = np.delete(ep2_dist, s2)
            ep2_ind = np.delete(ep2_ind, s2)
            ep2_ngb = np.unique(basin_rch[ep2_ind])

            # Pulling second endpoint neighbors' attributes.
            ep2_len = np.zeros(len(ep2_ngb))
            ep2_flg = np.zeros(len(ep2_ngb))
            ep2_acc = np.zeros(len(ep2_ngb))
            ep2_wse = np.zeros(len(ep2_ngb))
            for idy in list(range(len(ep2_ngb))):
                ep2_len[idy] = np.unique(basin_dist[np.where(basin_rch == ep2_ngb[idy])])
                ep2_flg[idy] = np.max(basin_flag[np.where(basin_rch == ep2_ngb[idy])])
                ep2_acc[idy] = np.median(basin_acc[np.where(basin_rch == ep2_ngb[idy])])
                ep2_wse[idy] = np.median(basin_wse[np.where(basin_rch == ep2_ngb[idy])])

            ep2 = np.array([ep2_ngb, ep2_len, ep2_flg, ep2_acc, ep2_wse]).T

        else:
            ep2 = np.array([])

    return ep1, ep2

###############################################################################

def update_rch_indexes(subcls, new_rch_id):

    """
    FUNCTION:
        Re-orders the point indexes within a reach and defines reach endpoints.

    INPUTS
        subcls -- Object containing attributes for the high-resolution
            centerline.
            [attributes used]:
                lon -- Longitude values along the high-resolution centerline.
                lat -- Latitude values along the high-resolution centerline.
                seg -- GRWL segment values along the high-resolution centerline.
                ind -- Point indexes for each GRWL segment along the
                    high-resolution centerline.
        new_rch_id -- 1-D array of the reach IDs to re-format the point
            indexes.

    OUTPUTS
        new_rch_ind -- Updated reach indexes (1-D array).
        new_rch_eps -- Updated reach endpoints (1-D array).
    """

    # Set variables and find unique reaches.
    uniq_rch = np.unique(new_rch_id)
    new_rch_ind = np.zeros(len(subcls.ind))
    new_rch_eps = np.zeros(len(subcls.ind))

    # Loop through each reach and re-order indexes.
    for ind in list(range(len(uniq_rch))):
        rch = np.where(new_rch_id == uniq_rch[ind])[0]
        rch_lon = subcls.lon[rch]
        rch_lat = subcls.lat[rch]
        rch_x, rch_y, __, __ = reproject_utm(rch_lat, rch_lon)
        rch_pts = np.vstack((rch_x, rch_y)).T
        rch_segs = subcls.seg[rch]
        segs = np.unique(subcls.seg[rch])
        new_ind = np.zeros(len(rch))
        eps = np.zeros(len(rch))

        # Reformat indexes if multiple segments are within a reach.
        if len(segs) > 1:
            #print(ind)
            for idx in list(range(len(segs))):
                s = np.where(subcls.seg[rch] == segs[idx])[0]
                mn = np.where(subcls.ind[rch[s]] == np.min(subcls.ind[rch[s]]))[0]
                mx = np.where(subcls.ind[rch[s]] == np.max(subcls.ind[rch[s]]))[0]
                eps[s[mn]] = 1
                eps[s[mx]] = 1

            # Finding true endpoints from orginal GRWL segment extents within
            # the new reach extent.
            eps_ind = np.where(eps>0)[0]
            ep_pts = np.vstack((rch_x[eps_ind], rch_y[eps_ind])).T
            kdt = sp.cKDTree(rch_pts)
            if len(rch_segs) < 10: #use to be 5.
                pt_dist, pt_ind = kdt.query(ep_pts, k = len(rch_segs))
            else:
                pt_dist, pt_ind = kdt.query(ep_pts, k = 10) #use to be 5

            real_eps = []
            for idy in list(range(len(eps_ind))):
                neighbors = len(np.unique(rch_segs[pt_ind[idy,:]]))
                if neighbors == 1:
                    real_eps.append(eps_ind[idy])
            real_eps = np.array(real_eps)

            if len(real_eps) == 1 or len(real_eps) == 2:
                final_eps = real_eps

            else:
                kdt2 = sp.cKDTree(ep_pts)
                if len(ep_pts) < 4:
                    pt_dist2, pt_ind2 = kdt2.query(ep_pts, k = len(ep_pts))
                else:
                    pt_dist2, pt_ind2 = kdt2.query(ep_pts, k = 4)
                real_eps2 = np.where(pt_dist2== np.max(pt_dist2))[0]
                final_eps = real_eps2

            if len(final_eps) == 0 or len(final_eps) > 2:
                print(uniq_rch[ind], 'problem with indexes')

            # Re-ordering points based on updated endpoints.
            new_ind[final_eps[0]]=1
            idz = final_eps[0]
            count = 2
            while np.min(new_ind) == 0:
                d = np.sqrt((rch_x[idz]-rch_x)**2 + (rch_y[idz]-rch_y)**2)
                dzero = np.where(new_ind == 0)[0]
                #vals = np.where(edits_segInd[dzero] eq 0)
                next_pt = dzero[np.where(d[dzero] == np.min(d[dzero]))[0]][0]
                new_ind[next_pt] = count
                count = count+1
                idz = next_pt

            new_rch_ind[rch] = new_ind
            ep1 = np.where(new_ind == np.min(new_ind))[0]
            ep2 = np.where(new_ind == np.max(new_ind))[0]
            new_rch_eps[rch[ep1]] = 1
            new_rch_eps[rch[ep2]] = 1
            #reverse index order to have indexes increasing in the upstream direction.
            if subcls.facc[rch[ep1]] < subcls.facc[rch[ep2]]:
                new_rch_ind[rch] = abs(new_ind - np.max(new_ind))

        # If there are no combined segments within reach keep current indexes.
        else:
            new_rch_ind[rch] = subcls.ind[rch]
            ep1 = np.where(subcls.ind[rch] == np.min(subcls.ind[rch]))[0]
            ep2 = np.where(subcls.ind[rch] == np.max(subcls.ind[rch]))[0]
            new_rch_eps[rch[ep1]] = 1
            new_rch_eps[rch[ep2]] = 1
            #reverse index order to have indexes increasing in the upstream direction.
            if subcls.facc[rch[ep1]] < subcls.facc[rch[ep2]]:
                new_rch_ind[rch] = abs(new_rch_ind[rch] - np.max(new_rch_ind[rch]))

    return new_rch_ind, new_rch_eps

###############################################################################

def aggregate_rivers(subcls, min_dist):

    """
    FUNCTION:
        Aggregates river reach types with reach lengths less than a specified
        minimum distance.

    INPUTS
        subcls -- Object containing attributes for the high-resolution
            centerline.
            [attributes used]:
                lon - Longtitude values along the high-resolution centerline.
                lat - Latitude values along the high-resolution centerline.
                rch_id2 -- Numbered reaches after cutting long river reaches.
                rch_len2 -- Reach lengths aftercutting long river reaches (meters).
                rch_ind2 -- Point indexes after cutting long river reaches.
                type2 -- Type flag after cutting long river reaches (1 = river,
                    2 = lake, 3 = lake on river, 4 = dam, 5 = no topology).
                elv -- Elevation values along the high-resolution centerline
                    (meters).
                facc -- Flow accumulation along the high-resolution ceterline
                    (km^2).
        min_dist -- Minimum reach length.

    OUTPUTS
        new_rch_id -- Updated reach IDs (1-D array).
        new_rch_dist -- Updated reach lengths (1-D array).
        new_rch_flag -- Updated reach type (1-D array)
    """

    # Set variables.
    new_rch_id = np.copy(subcls.rch_id1)
    new_rch_dist = np.copy(subcls.rch_len1)
    new_flag = np.copy(subcls.type1)
    level4 = np.array([int(str(point)[0:4]) for point in subcls.basins])
    uniq_basins = np.unique(level4) #np.unique(subcls.basins)
    for ind in list(range(len(uniq_basins))):
        #print(ind)
        basin = np.where(level4 == uniq_basins[ind])[0]
        basin_l6 =  subcls.basins[basin]
        basin_rch = subcls.rch_id1[basin]
        basin_dist = subcls.rch_len1[basin]
        basin_flag = subcls.type1[basin]
        basin_acc = subcls.facc[basin]
        basin_wse = subcls.elv[basin]
        basin_lon = subcls.lon[basin]
        basin_lat = subcls.lat[basin]
        basin_ind = subcls.ind[basin]
        basin_x, basin_y, __, __ = reproject_utm(basin_lat, basin_lon)

        # creating dummy vectors to help keep track of changes.
        dummy_basin_rch = np.copy(basin_rch)
        dummy_basin_dist = np.copy(basin_dist)
        dummy_type = np.copy(basin_flag)

        # finding intial small reaches.
        small_rch = np.unique(basin_rch[np.where((basin_dist > 0) & (basin_dist < min_dist))[0]])
        small_flag = np.array([np.unique(basin_flag[np.where(basin_rch == index)][0]) for index in small_rch]).flatten()
        small_dist = np.array([np.unique(basin_dist[np.where(basin_rch == index)][0]) for index in small_rch]).flatten()
        small_rivers = small_rch[np.where(small_flag == 1)]
        small_rivers_dist = small_dist[np.where(small_flag == 1)]
        #print(ind, len(small_rivers))

        #looping through short reaches and aggregating them based on a series of
        #logical conditions based on reach type and hydrological boundaries.
        loop = 1
        while len(small_rivers_dist) > 0:
            for idx in list(range(len(small_rivers))):
                # print(idx, small_rivers[idx])

                if small_rivers[idx] == -9999:
                    continue

                rch = np.where(basin_rch == small_rivers[idx])[0]
                rch_id = small_rivers[idx]
                rch_len = np.unique(basin_dist[rch])
                rch_l6 = np.unique(basin_l6[rch])[0]
                #rch_flag = np.unique(basin_flag[rch])
                rch_x = basin_x[rch]
                rch_y = basin_y[rch]
                rch_ind = basin_ind[rch]
                end1, end2 = find_neighbors(basin_rch, basin_dist, basin_flag,
                                            basin_acc, basin_wse, basin_x, basin_y, rch_x,
                                            rch_y, rch_ind, rch_len, rch_id, rch)

                # filtering out single neighbors that cross level 6 basin lines.
                if len(end1) == 1:
                    end1_l6 = np.unique(basin_l6[np.where(basin_rch == end1[0,0])[0]])[0]
                    if end1_l6 == rch_l6:
                        end1 = end1
                    else:
                        end1 = []

                if len(end2) == 1:
                    end2_l6 = np.unique(basin_l6[np.where(basin_rch == end2[0,0])[0]])[0]
                    if end2_l6 == rch_l6:
                        end2 = end2
                    else:
                        end2 = []

                ###############################################################
                # no bounding reaches.
                ###############################################################

                if len(end1) == 0 and len(end2) == 0:
                    # print(idx, 'cond 1 - no bordering reaches')
                    dummy_basin_rch[rch] = 0
                    dummy_basin_dist[rch] = 0
                    dummy_type[rch] = 0
                    continue

                ###############################################################
                # only one reach on one end.
                ###############################################################

                elif len(end1) == 0 and len(end2) == 1:
                    # print(idx, 'cond 2')
                    # river
                    if end2[:,2] == 1:
                        #print('    subcond. 1 - river')
                        new_id = end2[:,0]
                        new_dist = rch_len+end2[:,1]
                        new_type = end2[:,2]
                        rch2 = np.where(basin_rch == new_id)[0]
                        dummy_basin_dist[rch2] = new_dist
                        basin_dist[rch2] = new_dist
                        dummy_basin_rch[rch] = new_id
                        dummy_basin_dist[rch] = new_dist
                        dummy_type[rch] = new_type
                        basin_rch[rch] = new_id
                        basin_dist[rch] = new_dist
                        basin_flag[rch] =   new_type
                        if new_id in small_rivers:
                            update = np.where(small_rivers == new_id)[0]
                            small_rivers[update] = -9999
                            small_rivers_dist[update] = -9999

                    # dam
                    if end2[:,2] == 4:
                        #print('    subcond. 2 - dam')
                        if rch_len < 1000:
                            new_id = end2[:,0]
                            new_dist = rch_len+end2[:,1]
                            new_type = end2[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                            dummy_basin_rch[rch] = 0
                            dummy_basin_dist[rch] = 0
                            dummy_type[rch] = 0
                            continue
                    # lake
                    if end2[:,2] == 3:
                        #print('    subcond. 3 - lake')
                        if rch_len < 1000:
                            new_id = end2[:,0]
                            new_dist = rch_len+end2[:,1]
                            new_type = end2[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                            dummy_basin_rch[rch] = 0
                            dummy_basin_dist[rch] = 0
                            dummy_type[rch] = 0
                            continue
                    # coast
                    if end2[:,2] == 5:
                        #print('    subcond. 4 - coast')
                        if rch_len < 1000:
                            new_id = end2[:,0]
                            new_dist = rch_len+end2[:,1]
                            new_type = end2[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                            dummy_basin_rch[rch] = 0
                            dummy_basin_dist[rch] = 0
                            dummy_type[rch] = 0
                            continue

                ###############################################################
                # multiple reaches on one end only.
                ###############################################################

                elif len(end1) == 0 and len(end2) > 1:
                    # print(idx, 'cond 3 - short river with two tributaries')
                    dummy_basin_rch[rch] = 0
                    dummy_basin_dist[rch] = 0
                    dummy_type[rch] = 0
                    continue

                ###############################################################
                # only one reach on one end.
                ###############################################################

                elif len(end1) == 1 and len(end2) == 0:
                    # print(idx, 'cond 4')
                    # river
                    if end1[:,2] == 1:
                        #print('    subcond. 1 - river')
                        new_id = end1[:,0]
                        new_dist = rch_len+end1[:,1]
                        new_type = end1[:,2]
                        rch2 = np.where(basin_rch == new_id)[0]
                        dummy_basin_dist[rch2] = new_dist
                        basin_dist[rch2] = new_dist
                        dummy_basin_rch[rch] = new_id
                        dummy_basin_dist[rch] = new_dist
                        dummy_type[rch] = new_type
                        basin_rch[rch] = new_id
                        basin_dist[rch] = new_dist
                        basin_flag[rch] =   new_type
                        if new_id in small_rivers:
                            update = np.where(small_rivers == new_id)[0]
                            small_rivers[update] = -9999
                            small_rivers_dist[update] = -9999
                    # dam
                    if end1[:,2] == 4:
                        #print('    subcond. 2 - dam')
                        if rch_len < 1000:
                            new_id = end1[:,0]
                            new_dist = rch_len+end1[:,1]
                            new_type = end1[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                            dummy_basin_rch[rch] = 0
                            dummy_basin_dist[rch] = 0
                            dummy_type[rch] = 0
                            continue
                    # lake
                    if end1[:,2] == 3:
                        #print('    subcond. 3 - lake')
                        if rch_len < 1000:
                            new_id = end1[:,0]
                            new_dist = rch_len+end1[:,1]
                            new_type = end1[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                            dummy_basin_rch[rch] = 0
                            dummy_basin_dist[rch] = 0
                            dummy_type[rch] = 0
                            continue
                    # coast
                    if end1[:,2] == 5:
                        #print('    subcond. 4 - coast')
                        if rch_len < 1000:
                            new_id = end1[:,0]
                            new_dist = rch_len+end1[:,1]
                            new_type = end1[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                            dummy_basin_rch[rch] = 0
                            dummy_basin_dist[rch] = 0
                            dummy_type[rch] = 0
                            continue

                ###############################################################
                # one reach is on each end.
                ###############################################################

                elif len(end1) == 1 and len(end2) == 1:
                    # print(idx, 'cond 5')
                    # two rivers bordering the reach.
                    if end1[:,2] == 1 and end2[:,2] == 1:
                        #print('    subcond. 1 -> 2 bordering rivers')
                        # put with shorter river reach.
                        if end1[:,1] > end2[:,1]:
                            new_id = end2[:,0]
                            new_dist = rch_len+end2[:,1]
                            new_type = end2[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                            new_id = end1[:,0]
                            new_dist = rch_len+end1[:,1]
                            new_type = end1[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] = new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                    # river and non river bordering the reach.
                    elif end1[:,2] == 1 and end2[:,2] > 1:
                        #print('    subcond. 2 -> 1 bordering river')
                        # Attach to river if longer than 500 m.
                        if end1[:,1] > 100:
                            new_id = end1[:,0]
                            new_dist = rch_len+end1[:,1]
                            new_type = end1[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                            if end2[:,2] != 4:
                                new_id = end2[:,0]
                                new_dist = rch_len+end2[:,1]
                                new_type = end2[:,2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                            else:
                                if rch_len < 1000:
                                    new_id = end2[:,0]
                                    new_dist = rch_len+end2[:,1]
                                    new_type = end2[:,2]
                                    rch2 = np.where(basin_rch == new_id)[0]
                                    dummy_basin_dist[rch2] = new_dist
                                    basin_dist[rch2] = new_dist
                                    dummy_basin_rch[rch] = new_id
                                    dummy_basin_dist[rch] = new_dist
                                    dummy_type[rch] = new_type
                                    basin_rch[rch] = new_id
                                    basin_dist[rch] = new_dist
                                    basin_flag[rch] =   new_type
                                    if new_id in small_rivers:
                                        update = np.where(small_rivers == new_id)[0]
                                        small_rivers[update] = -9999
                                        small_rivers_dist[update] = -9999
                                else:
                                    dummy_basin_rch[rch] = 0
                                    dummy_basin_dist[rch] = 0
                                    dummy_type[rch] = 0
                                    continue
                    # river and non river bordering the reach.
                    elif end1[:,2] > 1 and end2[:,2] == 1:
                        #print('    subcond. 3 -> 1 bordering river')
                        # Attach to river if longer than 500 m.
                        if end2[:,1] > 100:
                            new_id = end2[:,0]
                            new_dist = rch_len+end2[:,1]
                            new_type = end2[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                            if end1[:,2] != 4:
                                new_id = end1[:,0]
                                new_dist = rch_len+end1[:,1]
                                new_type = end1[:,2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                            else:
                                if rch_len < 1000:
                                    new_id = end1[:,0]
                                    new_dist = rch_len+end1[:,1]
                                    new_type = end1[:,2]
                                    rch2 = np.where(basin_rch == new_id)[0]
                                    dummy_basin_dist[rch2] = new_dist
                                    basin_dist[rch2] = new_dist
                                    dummy_basin_rch[rch] = new_id
                                    dummy_basin_dist[rch] = new_dist
                                    dummy_type[rch] = new_type
                                    basin_rch[rch] = new_id
                                    basin_dist[rch] = new_dist
                                    basin_flag[rch] =   new_type
                                    if new_id in small_rivers:
                                        update = np.where(small_rivers == new_id)[0]
                                        small_rivers[update] = -9999
                                        small_rivers_dist[update] = -9999
                                else:
                                    dummy_basin_rch[rch] = 0
                                    dummy_basin_dist[rch] = 0
                                    dummy_type[rch] = 0
                                    continue
                    # two non rivers bordering the reach.
                    elif end1[:,2] > 1 and end2[:,2] > 1:
                        #print('    subcond. 4 - no bordering rivers')
                        vals = np.array([end1[:,2], end2[:,2]]).flatten()
                        # two borering lakes.
                        if np.max(vals) == 3:
                            #print('    two bordering lakes')
                            if rch_len < 1000:
                                new_id = end1[:,0]
                                new_dist = rch_len+end1[:,1]
                                new_type = end1[:,2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                            else:
                                dummy_basin_rch[rch] = 0
                                dummy_basin_dist[rch] = 0
                                dummy_type[rch] = 0
                                continue
                        # lake and dam.
                        elif np.min(vals) == 3 and np.max(vals) == 4:
                            #print('    lake and dam')
                            if rch_len < 1000:
                                keep = np.where(vals == 3)[0]
                                if keep == 0:
                                    new_id = end1[:,0]
                                    new_dist = rch_len+end1[:,1]
                                    new_type = end1[:,2]
                                    rch2 = np.where(basin_rch == new_id)[0]
                                    dummy_basin_dist[rch2] = new_dist
                                    basin_dist[rch2] = new_dist
                                    dummy_basin_rch[rch] = new_id
                                    dummy_basin_dist[rch] = new_dist
                                    dummy_type[rch] = new_type
                                    basin_rch[rch] = new_id
                                    basin_dist[rch] = new_dist
                                    basin_flag[rch] =   new_type
                                    if new_id in small_rivers:
                                        update = np.where(small_rivers == new_id)[0]
                                        small_rivers[update] = -9999
                                        small_rivers_dist[update] = -9999
                                else:
                                    new_id = end2[:,0]
                                    new_dist = rch_len+end2[:,1]
                                    new_type = end2[:,2]
                                    rch2 = np.where(basin_rch == new_id)[0]
                                    dummy_basin_dist[rch2] = new_dist
                                    basin_dist[rch2] = new_dist
                                    dummy_basin_rch[rch] = new_id
                                    dummy_basin_dist[rch] = new_dist
                                    dummy_type[rch] = new_type
                                    basin_rch[rch] = new_id
                                    basin_dist[rch] = new_dist
                                    basin_flag[rch] =   new_type
                                    if new_id in small_rivers:
                                        update = np.where(small_rivers == new_id)[0]
                                        small_rivers[update] = -9999
                                        small_rivers_dist[update] = -9999
                            else:
                                dummy_basin_rch[rch] = 0
                                dummy_basin_dist[rch] = 0
                                dummy_type[rch] = 0
                                continue
                        # lake and coast
                        elif np.min(vals) == 3 and np.max(vals) == 5:
                            #print('    lake and coast')
                            if rch_len < 1000:
                                keep = np.where(vals == 3)[0]
                                if keep == 0:
                                    new_id = end1[:,0]
                                    new_dist = rch_len+end1[:,1]
                                    new_type = end1[:,2]
                                    rch2 = np.where(basin_rch == new_id)[0]
                                    dummy_basin_dist[rch2] = new_dist
                                    basin_dist[rch2] = new_dist
                                    dummy_basin_rch[rch] = new_id
                                    dummy_basin_dist[rch] = new_dist
                                    dummy_type[rch] = new_type
                                    basin_rch[rch] = new_id
                                    basin_dist[rch] = new_dist
                                    basin_flag[rch] =   new_type
                                    if new_id in small_rivers:
                                        update = np.where(small_rivers == new_id)[0]
                                        small_rivers[update] = -9999
                                        small_rivers_dist[update] = -9999
                                else:
                                    new_id = end2[:,0]
                                    new_dist = rch_len+end2[:,1]
                                    new_type = end2[:,2]
                                    rch2 = np.where(basin_rch == new_id)[0]
                                    dummy_basin_dist[rch2] = new_dist
                                    basin_dist[rch2] = new_dist
                                    dummy_basin_rch[rch] = new_id
                                    dummy_basin_dist[rch] = new_dist
                                    dummy_type[rch] = new_type
                                    basin_rch[rch] = new_id
                                    basin_dist[rch] = new_dist
                                    basin_flag[rch] =   new_type
                                    if new_id in small_rivers:
                                        update = np.where(small_rivers == new_id)[0]
                                        small_rivers[update] = -9999
                                        small_rivers_dist[update] = -9999
                            else:
                                dummy_basin_rch[rch] = 0
                                dummy_basin_dist[rch] = 0
                                dummy_type[rch] = 0
                                continue
                        # dam and dam
                        elif np.min(vals) == 4 and np.max(vals) == 4:
                            #print('    two dams')
                            if rch_len < 1000:
                                new_id = end1[:,0]
                                new_dist = rch_len+end1[:,1]
                                new_type = end1[:,2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                            else:
                                dummy_basin_rch[rch] = 0
                                dummy_basin_dist[rch] = 0
                                dummy_type[rch] = 0
                                continue
                        # dam and coast
                        elif np.min(vals) == 4 and np.max(vals) == 5:
                            #print('    dam and coast')
                            if rch_len < 1000:
                                keep = np.where(vals == 5)[0]
                                if keep == 0:
                                    new_id = end1[:,0]
                                    new_dist = rch_len+end1[:,1]
                                    new_type = end1[:,2]
                                    rch2 = np.where(basin_rch == new_id)[0]
                                    dummy_basin_dist[rch2] = new_dist
                                    basin_dist[rch2] = new_dist
                                    dummy_basin_rch[rch] = new_id
                                    dummy_basin_dist[rch] = new_dist
                                    dummy_type[rch] = new_type
                                    basin_rch[rch] = new_id
                                    basin_dist[rch] = new_dist
                                    basin_flag[rch] =   new_type
                                    if new_id in small_rivers:
                                        update = np.where(small_rivers == new_id)[0]
                                        small_rivers[update] = -9999
                                        small_rivers_dist[update] = -9999
                                else:
                                    new_id = end2[:,0]
                                    new_dist = rch_len+end2[:,1]
                                    new_type = end2[:,2]
                                    rch2 = np.where(basin_rch == new_id)[0]
                                    dummy_basin_dist[rch2] = new_dist
                                    basin_dist[rch2] = new_dist
                                    dummy_basin_rch[rch] = new_id
                                    dummy_basin_dist[rch] = new_dist
                                    dummy_type[rch] = new_type
                                    basin_rch[rch] = new_id
                                    basin_dist[rch] = new_dist
                                    basin_flag[rch] =   new_type
                                    if new_id in small_rivers:
                                        update = np.where(small_rivers == new_id)[0]
                                        small_rivers[update] = -9999
                                        small_rivers_dist[update] = -9999
                            else:
                                dummy_basin_rch[rch] = 0
                                dummy_basin_dist[rch] = 0
                                dummy_type[rch] = 0
                                continue
                        # coast and coast
                        elif np.min(vals) == 5 and np.max(vals) == 5:
                            #print('    two coasts')
                            if rch_len < 1000:
                                new_id = end1[:,0]
                                new_dist = rch_len+end1[:,1]
                                new_type = end1[:,2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                            else:
                                dummy_basin_rch[rch] = 0
                                dummy_basin_dist[rch] = 0
                                dummy_type[rch] = 0
                                continue

                ###############################################################
                # one reach at one end and multiple reaches on the other end.
                ###############################################################

                elif len(end1) == 1 and len(end2) > 1:
                    # print(idx, 'cond 6')
                    if end1[:,2] == 1:
                        #print('    subcond 1 - river on single end.')
                        new_id = end1[:,0]
                        new_dist = rch_len+end1[:,1]
                        new_type = end1[:,2]
                        rch2 = np.where(basin_rch == new_id)[0]
                        dummy_basin_dist[rch2] = new_dist
                        basin_dist[rch2] = new_dist
                        dummy_basin_rch[rch] = new_id
                        dummy_basin_dist[rch] = new_dist
                        dummy_type[rch] = new_type
                        basin_rch[rch] = new_id
                        basin_dist[rch] = new_dist
                        basin_flag[rch] =   new_type
                        if new_id in small_rivers:
                            update = np.where(small_rivers == new_id)[0]
                            small_rivers[update] = -9999
                            small_rivers_dist[update] = -9999
                    elif end1[:,2] > 1:
                        #print('    subcond 2 - non river on single end.')
                        if rch_len < 1000:
                            new_id = end1[:,0]
                            new_dist = rch_len+end1[:,1]
                            new_type = end1[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                            dummy_basin_rch[rch] = 0
                            dummy_basin_dist[rch] = 0
                            dummy_type[rch] = 0
                            continue

                ###############################################################
                # multiple reaches on one end only.
                ###############################################################

                elif len(end1) > 1 and len(end2) == 0:
                    # print(idx, 'cond 7 - short river with two tributaries')
                    dummy_basin_rch[rch] = 0
                    dummy_basin_dist[rch] = 0
                    dummy_type[rch] = 0
                    continue

                ###############################################################
                # one reach at one end and multiple reaches on the other end.
                ###############################################################

                elif len(end1) > 1 and len(end2) == 1:
                    # print(idx, 'cond 8')
                    if end2[:,2] == 1:
                        #print('    subcond 1 - river on single end.')
                        new_id = end2[:,0]
                        new_dist = rch_len+end2[:,1]
                        new_type = end2[:,2]
                        rch2 = np.where(basin_rch == new_id)[0]
                        dummy_basin_dist[rch2] = new_dist
                        basin_dist[rch2] = new_dist
                        dummy_basin_rch[rch] = new_id
                        dummy_basin_dist[rch] = new_dist
                        dummy_type[rch] = new_type
                        basin_rch[rch] = new_id
                        basin_dist[rch] = new_dist
                        basin_flag[rch] =   new_type
                        if new_id in small_rivers:
                            update = np.where(small_rivers == new_id)[0]
                            small_rivers[update] = -9999
                            small_rivers_dist[update] = -9999
                    elif end2[:,2] > 1:
                        #print('    subcond 2 - non river on single end.')
                        if rch_len < 1000:
                            new_id = end2[:,0]
                            new_dist = rch_len+end2[:,1]
                            new_type = end2[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                            dummy_basin_rch[rch] = 0
                            dummy_basin_dist[rch] = 0
                            dummy_type[rch] = 0
                            continue
                ###############################################################
                # multiple reaches at both ends.
                ###############################################################

                elif len(end1) > 1 and len(end2) > 1:
                    # print(idx, 'cond 9')
                    if rch_len <= 100:
                        singles = np.unique(np.concatenate([end1[:,0], end2[:,0]]))
                        if len(singles) == 1:
                            #print('    subcond 1. - all duplicates, one real neighbor')
                            new_id = end1[0,0]
                            new_dist = rch_len+end1[0,1]
                            new_type = end1[0,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        elif len(singles) == 2:
                            #print('    subcond 2. - only two real neighbors')
                            new_id = end1[0,0]
                            new_dist = rch_len+end1[0,1]
                            new_type = end1[0,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        elif len(singles) > 2:
                            #print('    subcond 3. - at least three real neighbors')
                            vals = np.unique(np.concatenate([end1[:,2], end2[:,2]]))
                            mv1 = np.where(end1[:,2] == np.max(vals))[0]
                            mv2 = np.where(end2[:,2] == np.max(vals))[0]
                            if len(mv1) == 0 and len(mv2) > 0:
                                new_id = end2[mv2[0],0]
                                new_dist = rch_len+end2[mv2[0],1]
                                new_type = end2[mv2[0],2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                            elif len(mv1) > 0 and len(mv2) == 0:
                                new_id = end1[mv1[0],0]
                                new_dist = rch_len+end1[mv1[0],1]
                                new_type = end1[mv1[0],2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                            elif len(mv1) > 0 and len(mv2) > 0:
                                new_id = end1[mv1[0],0]
                                new_dist = rch_len+end1[mv1[0],1]
                                new_type = end1[mv1[0],2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                    else:
                        dummy_basin_rch[rch] = 0
                        dummy_basin_dist[rch] = 0
                        dummy_type[rch] = 0
                        continue

                ###############################################################

            ### should be re-calculated within while loop.
            small_rch = np.unique(dummy_basin_rch[np.where((dummy_basin_dist > 0) & (dummy_basin_dist < min_dist))[0]])
            small_flag = np.array([np.unique(dummy_type[np.where(dummy_basin_rch == index)[0][0]]) for index in small_rch]).flatten()
            small_dist = np.array([np.unique(dummy_basin_dist[np.where(dummy_basin_rch == index)[0][0]]) for index in small_rch]).flatten()
            small_rivers = small_rch[np.where(small_flag == 1)[0]]
            small_rivers_dist = small_dist[np.where(small_flag == 1)[0]]
            loop = loop+1
            if loop == 200:
                print(ind, 'LOOP STUCK')

        ### should be replaced after while loop.
        new_rch_id[basin] = basin_rch
        new_rch_dist[basin] = basin_dist
        new_flag[basin] = basin_flag

    return new_rch_id, new_rch_dist, new_flag

###############################################################################

def aggregate_lakes(subcls, min_dist):

    """
    FUNCTION:
        Aggregates lake reach types with reach lengths less than a specified
        minimum distance.

    INPUTS
        subcls -- Object containing attributes for the high-resolution
            centerline.
            [attributes used]:
                lon - Longtitude values along the high-resolution centerline.
                lat - Latitude values along the high-resolution centerline.
                rch_id3 -- Numbered reaches after aggregating short river reaches.
                rch_len3 -- Reach lengths after aggregating short river reaches
                    (meters).
                rch_ind3 -- Point indexes after aggregating short river reaches.
                type3 -- Type flag after aggregating short river reaches
                    (1 = river, 2 = lake, 3 = lake on river, 4 = dam
                    5 = no topology).
                elv -- Elevation values along the high-resolution centerline
                    (meters).
                facc -- Flow accumulation along the high-resolution ceterline
                    (km^2).
        min_dist -- Minimum reach length.

    OUTPUTS
        new_rch_id -- Updated reach IDs (1-D array).
        new_rch_dist -- Updated reach lengths (1-D array).
        new_rch_flag -- Updated reach type (1-D array)
    """
    # Set variables.
    new_rch_id = np.copy(subcls.rch_id2)
    new_rch_dist = np.copy(subcls.rch_len2)
    new_flag = np.copy(subcls.type2)
    level4 = np.array([np.int(np.str(point)[0:4]) for point in subcls.basins])
    uniq_basins = np.unique(level4) #np.unique(subcls.basins)
    for ind in list(range(len(uniq_basins))):
        #print(ind)
        basin = np.where(level4 == uniq_basins[ind])[0]
        basin_l6 =  subcls.basins[basin]
        basin_rch = subcls.rch_id2[basin]
        basin_dist = subcls.rch_len2[basin]
        basin_flag = subcls.type2[basin]
        basin_acc = subcls.facc[basin]
        basin_wse = subcls.elv[basin]
        basin_lon = subcls.lon[basin]
        basin_lat = subcls.lat[basin]
        basin_ind = subcls.rch_ind2[basin]
        basin_x, basin_y, __, __ = reproject_utm(basin_lat, basin_lon)

        #creating dummy vectors to help keep track of changes.
        dummy_basin_rch = np.copy(basin_rch)
        dummy_basin_dist = np.copy(basin_dist)
        dummy_type = np.copy(basin_flag)

        # finding intial small reaches.
        small_rch = np.unique(basin_rch[np.where((basin_dist > 0) & (basin_dist < min_dist))[0]])
        small_flag = np.array([np.unique(basin_flag[np.where(basin_rch == index)][0]) for index in small_rch]).flatten()
        small_dist = np.array([np.unique(basin_dist[np.where(basin_rch == index)][0]) for index in small_rch]).flatten()
        small_rivers = small_rch[np.where((small_flag > 1) & (small_flag !=4))]
        small_rivers_dist = small_dist[np.where((small_flag > 1) & (small_flag !=4))]
        #print(ind, len(small_rivers))

        #looping through short reaches and aggregating them based on a series of
        #logical conditions based on reach type and hydrological boundaries.
        loop = 1
        while len(small_rivers_dist) > 0:
            for idx in list(range(len(small_rivers))):
                #print(idx)

                if small_rivers[idx] == -9999:
                    continue

                rch = np.where(basin_rch == small_rivers[idx])[0]
                rch_id = small_rivers[idx]
                rch_len = np.unique(basin_dist[rch])
                rch_l6 = np.unique(basin_l6[rch])[0]
                rch_flag = np.max(np.unique(basin_flag[rch]))
                rch_x = basin_x[rch]
                rch_y = basin_y[rch]
                rch_ind = basin_ind[rch]
                end1, end2 = find_neighbors(basin_rch, basin_dist, basin_flag,
                                            basin_acc, basin_wse, basin_x, basin_y, rch_x,
                                            rch_y, rch_ind, rch_len, rch_id, rch)

                # filtering out single neighbors that cross level 6 basin lines.
                if len(end1) == 1:
                    end1_l6 = np.unique(basin_l6[np.where(basin_rch == end1[0,0])[0]])[0]
                    if end1_l6 == rch_l6:
                        end1 = end1
                    else:
                        end1 = []

                if len(end2) == 1:
                    end2_l6 = np.unique(basin_l6[np.where(basin_rch == end2[0,0])[0]])[0]
                    if end2_l6 == rch_l6:
                        end2 = end2
                    else:
                        end2 = []

                ###############################################################
                # no bounding reaches.
                ###############################################################

                if len(end1) == 0 and len(end2) == 0:
                    #print(idx, 'cond 1 - no bordering reaches')
                    dummy_basin_rch[rch] = 0
                    dummy_basin_dist[rch] = 0
                    dummy_type[rch] = 0
                    continue

                ###############################################################
                # only one reach on one end.
                ###############################################################

                elif len(end1) == 0 and len(end2) == 1:
                    #print(idx, 'cond 2')
                    # lake
                    if end2[:,2] == rch_flag:
                        #print('    subcond. 1 - lake')
                        new_id = end2[:,0]
                        new_dist = rch_len+end2[:,1]
                        new_type = end2[:,2]
                        rch2 = np.where(basin_rch == new_id)[0]
                        dummy_basin_dist[rch2] = new_dist
                        basin_dist[rch2] = new_dist
                        dummy_basin_rch[rch] = new_id
                        dummy_basin_dist[rch] = new_dist
                        dummy_type[rch] = new_type
                        basin_rch[rch] = new_id
                        basin_dist[rch] = new_dist
                        basin_flag[rch] =   new_type
                        if new_id in small_rivers:
                            update = np.where(small_rivers == new_id)[0]
                            small_rivers[update] = -9999
                            small_rivers_dist[update] = -9999

                    # not a lake. combine if < 1000 m.
                    if end2[:,2] != rch_flag:
                        #print('    subcond. 2 - non-lake')
                        if rch_len < 1000:
                            new_id = end2[:,0]
                            new_dist = rch_len+end2[:,1]
                            new_type = end2[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                            dummy_basin_rch[rch] = 0
                            dummy_basin_dist[rch] = 0
                            dummy_type[rch] = 0
                            continue

                ###############################################################
                # multiple reaches on one end only.
                ###############################################################

                elif len(end1) == 0 and len(end2) > 1:
                    #print(idx, 'cond 3 - short reach with two tributaries on one end')
                    dummy_basin_rch[rch] = 0
                    dummy_basin_dist[rch] = 0
                    dummy_type[rch] = 0
                    continue

                ###############################################################
                # only one reach on one end.
                ###############################################################

                elif len(end1) == 1 and len(end2) == 0:
                    #print(idx, 'cond 4')
                    # lake
                    if end1[:,2] == rch_flag:
                        #print('    subcond. 1 - lake')
                        new_id = end1[:,0]
                        new_dist = rch_len+end1[:,1]
                        new_type = end1[:,2]
                        rch2 = np.where(basin_rch == new_id)[0]
                        dummy_basin_dist[rch2] = new_dist
                        basin_dist[rch2] = new_dist
                        dummy_basin_rch[rch] = new_id
                        dummy_basin_dist[rch] = new_dist
                        dummy_type[rch] = new_type
                        basin_rch[rch] = new_id
                        basin_dist[rch] = new_dist
                        basin_flag[rch] =   new_type
                        if new_id in small_rivers:
                            update = np.where(small_rivers == new_id)[0]
                            small_rivers[update] = -9999
                            small_rivers_dist[update] = -9999
                    # not a lake. combine if < 1000 m.
                    if end1[:,2] != rch_flag:
                        #print('    subcond. 2 - non lake')
                        if rch_len < 1000:
                            new_id = end1[:,0]
                            new_dist = rch_len+end1[:,1]
                            new_type = end1[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                            dummy_basin_rch[rch] = 0
                            dummy_basin_dist[rch] = 0
                            dummy_type[rch] = 0
                            continue

                ###############################################################
                # one reach is on each end.
                ###############################################################

                elif len(end1) == 1 and len(end2) == 1:
                    #print(idx, 'cond 5')
                    # One bordering reach on each end.
                    if end1[:,2] == 1 and end2[:,2] == 1:
                        #print('    subcond. 1 -> 2 bordering rivers')
                        # put with shorter river reach.
                        if rch_len < 1000:
                            new_id = end2[:,0]
                            new_dist = rch_len+end2[:,1]
                            new_type = end2[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                                dummy_basin_rch[rch] = 0
                                dummy_basin_dist[rch] = 0
                                dummy_type[rch] = 0
                                continue
                    # river and non river bordering the reach.
                    elif end1[:,2] == 1 and end2[:,2] > 1:
                        #print('    subcond. 2 -> 1 bordering river')
                        # Attach to river if longer than 1000 m.
                        if end2[:,2] == rch_flag:
                            new_id = end2[:,0]
                            new_dist = rch_len+end2[:,1]
                            new_type = end2[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                            if rch_len < 1000:
                                new_id = end2[:,0]
                                new_dist = rch_len+end2[:,1]
                                new_type = end2[:,2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                            else:
                                dummy_basin_rch[rch] = 0
                                dummy_basin_dist[rch] = 0
                                dummy_type[rch] = 0
                                continue
                    # river and non river bordering the reach.
                    elif end1[:,2] > 1 and end2[:,2] == 1:
                        #print('    subcond. 3 -> 1 bordering river')
                        # Attach to river if longer than 1000 m.
                        if end1[:,2] == rch_flag:
                            new_id = end1[:,0]
                            new_dist = rch_len+end1[:,1]
                            new_type = end1[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                            if rch_len < 1000:
                                new_id = end1[:,0]
                                new_dist = rch_len+end1[:,1]
                                new_type = end1[:,2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                            else:
                                dummy_basin_rch[rch] = 0
                                dummy_basin_dist[rch] = 0
                                dummy_type[rch] = 0
                                continue
                    # two non-rivers bordering the reach.
                    elif end1[:,2] > 1 and end2[:,2] > 1:
                        #print('    subcond. 4 - no bordering rivers')
                        vals = np.array([end1[:,2], end2[:,2]]).flatten()
                        # two borering lakes.
                        if np.max(vals) == 3 and np.min(vals) == 3:
                            #print('    two bordering lakes')
                            new_id = end1[:,0]
                            new_dist = rch_len+end1[:,1]
                            new_type = end1[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        # two bordering coasts.
                        if np.max(vals) == 5 and np.min(vals) == 5:
                            #print('    two bordering lakes')
                            new_id = end1[:,0]
                            new_dist = rch_len+end1[:,1]
                            new_type = end1[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        # lake and dam.
                        elif np.min(vals) == 3 and np.max(vals) == 4:
                            #print('    lake and dam')
                            keep = np.where(vals == rch_flag)[0]
                            if keep == 0:
                                new_id = end1[:,0]
                                new_dist = rch_len+end1[:,1]
                                new_type = end1[:,2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                            else:
                                new_id = end2[:,0]
                                new_dist = rch_len+end2[:,1]
                                new_type = end2[:,2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                        # dam and coast
                        elif np.min(vals) == 4 and np.max(vals) == 5:
                            #print('    lake and dam')
                            keep = np.where(vals == rch_flag)[0]
                            if keep == 0:
                                new_id = end1[:,0]
                                new_dist = rch_len+end1[:,1]
                                new_type = end1[:,2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                            else:
                                new_id = end2[:,0]
                                new_dist = rch_len+end2[:,1]
                                new_type = end2[:,2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                        # lake and coast
                        elif np.min(vals) == 3 and np.max(vals) == 5:
                            #print('    lake and coast')
                            if rch_flag == 3:
                                keep = np.where(vals == 3)[0]
                                if keep == 0:
                                    new_id = end1[:,0]
                                    new_dist = rch_len+end1[:,1]
                                    new_type = end1[:,2]
                                    rch2 = np.where(basin_rch == new_id)[0]
                                    dummy_basin_dist[rch2] = new_dist
                                    basin_dist[rch2] = new_dist
                                    dummy_basin_rch[rch] = new_id
                                    dummy_basin_dist[rch] = new_dist
                                    dummy_type[rch] = new_type
                                    basin_rch[rch] = new_id
                                    basin_dist[rch] = new_dist
                                    basin_flag[rch] =   new_type
                                    if new_id in small_rivers:
                                        update = np.where(small_rivers == new_id)[0]
                                        small_rivers[update] = -9999
                                        small_rivers_dist[update] = -9999
                                else:
                                    new_id = end2[:,0]
                                    new_dist = rch_len+end2[:,1]
                                    new_type = end2[:,2]
                                    rch2 = np.where(basin_rch == new_id)[0]
                                    dummy_basin_dist[rch2] = new_dist
                                    basin_dist[rch2] = new_dist
                                    dummy_basin_rch[rch] = new_id
                                    dummy_basin_dist[rch] = new_dist
                                    dummy_type[rch] = new_type
                                    basin_rch[rch] = new_id
                                    basin_dist[rch] = new_dist
                                    basin_flag[rch] =   new_type
                                    if new_id in small_rivers:
                                        update = np.where(small_rivers == new_id)[0]
                                        small_rivers[update] = -9999
                                        small_rivers_dist[update] = -9999
                            if rch_flag == 5:
                                keep = np.where(vals == 3)[0]
                                if keep == 0:
                                    new_id = end1[:,0]
                                    new_dist = rch_len+end1[:,1]
                                    new_type = end1[:,2]
                                    rch2 = np.where(basin_rch == new_id)[0]
                                    dummy_basin_dist[rch2] = new_dist
                                    basin_dist[rch2] = new_dist
                                    dummy_basin_rch[rch] = new_id
                                    dummy_basin_dist[rch] = new_dist
                                    dummy_type[rch] = new_type
                                    basin_rch[rch] = new_id
                                    basin_dist[rch] = new_dist
                                    basin_flag[rch] =   new_type
                                    if new_id in small_rivers:
                                        update = np.where(small_rivers == new_id)[0]
                                        small_rivers[update] = -9999
                                        small_rivers_dist[update] = -9999
                                else:
                                    new_id = end2[:,0]
                                    new_dist = rch_len+end2[:,1]
                                    new_type = end2[:,2]
                                    rch2 = np.where(basin_rch == new_id)[0]
                                    dummy_basin_dist[rch2] = new_dist
                                    basin_dist[rch2] = new_dist
                                    dummy_basin_rch[rch] = new_id
                                    dummy_basin_dist[rch] = new_dist
                                    dummy_type[rch] = new_type
                                    basin_rch[rch] = new_id
                                    basin_dist[rch] = new_dist
                                    basin_flag[rch] =   new_type
                                    if new_id in small_rivers:
                                        update = np.where(small_rivers == new_id)[0]
                                        small_rivers[update] = -9999
                                        small_rivers_dist[update] = -9999
                        # dam and dam
                        elif np.min(vals) == 4 and np.max(vals) == 4:
                            #print('    two dams')
                            if rch_len < 1000:
                                new_id = end1[:,0]
                                new_dist = rch_len+end1[:,1]
                                new_type = end1[:,2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                            else:
                                dummy_basin_rch[rch] = 0
                                dummy_basin_dist[rch] = 0
                                dummy_type[rch] = 0
                                continue

                ###############################################################
                # one reach at one end and multiple reaches on the other end.
                ###############################################################

                elif len(end1) == 1 and len(end2) > 1:
                    #print(idx, 'cond 6')
                    if end1[:,2] == rch_flag:
                        #print('    subcond 1 - lake on single end.')
                        new_id = end1[:,0]
                        new_dist = rch_len+end1[:,1]
                        new_type = end1[:,2]
                        rch2 = np.where(basin_rch == new_id)[0]
                        dummy_basin_dist[rch2] = new_dist
                        basin_dist[rch2] = new_dist
                        dummy_basin_rch[rch] = new_id
                        dummy_basin_dist[rch] = new_dist
                        dummy_type[rch] = new_type
                        basin_rch[rch] = new_id
                        basin_dist[rch] = new_dist
                        basin_flag[rch] =   new_type
                        if new_id in small_rivers:
                            update = np.where(small_rivers == new_id)[0]
                            small_rivers[update] = -9999
                            small_rivers_dist[update] = -9999
                    elif end1[:,2] != rch_flag:
                        #print('    subcond 2 - non lake on single end.')
                        if rch_len < 1000:
                            new_id = end1[:,0]
                            new_dist = rch_len+end1[:,1]
                            new_type = end1[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                            dummy_basin_rch[rch] = 0
                            dummy_basin_dist[rch] = 0
                            dummy_type[rch] = 0
                            continue

                ###############################################################
                # multiple reaches on one end only.
                ###############################################################

                elif len(end1) > 1 and len(end2) == 0:
                    #print(idx, 'cond 7 - short reach with two tributaries on one end.')
                    dummy_basin_rch[rch] = 0
                    dummy_basin_dist[rch] = 0
                    dummy_type[rch] = 0
                    continue

                ###############################################################
                # one reach at one end and multiple reaches on the other end.
                ###############################################################

                elif len(end1) > 1 and len(end2) == 1:
                    #print(idx, 'cond 8')
                    if end2[:,2] == rch_flag:
                        #print('    subcond 1 - lake on single end.')
                        new_id = end2[:,0]
                        new_dist = rch_len+end2[:,1]
                        new_type = end2[:,2]
                        rch2 = np.where(basin_rch == new_id)[0]
                        dummy_basin_dist[rch2] = new_dist
                        basin_dist[rch2] = new_dist
                        dummy_basin_rch[rch] = new_id
                        dummy_basin_dist[rch] = new_dist
                        dummy_type[rch] = new_type
                        basin_rch[rch] = new_id
                        basin_dist[rch] = new_dist
                        basin_flag[rch] =   new_type
                        if new_id in small_rivers:
                            update = np.where(small_rivers == new_id)[0]
                            small_rivers[update] = -9999
                            small_rivers_dist[update] = -9999
                    elif end2[:,2] != rch_flag:
                        #print('    subcond 2 - non lake on single end.')
                        if rch_len < 1000:
                            new_id = end2[:,0]
                            new_dist = rch_len+end2[:,1]
                            new_type = end2[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                            dummy_basin_rch[rch] = 0
                            dummy_basin_dist[rch] = 0
                            dummy_type[rch] = 0
                            continue
                ###############################################################
                # multiple reaches at both ends.
                ###############################################################

                elif len(end1) > 1 and len(end2) > 1:
                    #print(idx, 'cond 9')
                    if rch_len <= 100:
                        singles = np.unique(np.concatenate([end1[:,0], end2[:,0]]))
                        if len(singles) == 1:
                            #print('    subcond 1. - all duplicates, one real neighbor')
                            new_id = end1[0,0]
                            new_dist = rch_len+end1[0,1]
                            new_type = end1[0,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        elif len(singles) == 2:
                            #print('    subcond 2. - only two real neighbors')
                            new_id = end1[0,0]
                            new_dist = rch_len+end1[0,1]
                            new_type = end1[0,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        elif len(singles) > 2:
                            #print('    subcond 3. - at least three real neighbors')
                            vals = np.unique(np.concatenate([end1[:,2], end2[:,2]]))
                            mv1 = np.where(end1[:,2] == np.max(vals))[0]
                            mv2 = np.where(end2[:,2] == np.max(vals))[0]
                            if len(mv1) == 0 and len(mv2) > 0:
                                new_id = end2[mv2[0],0]
                                new_dist = rch_len+end2[mv2[0],1]
                                new_type = end2[mv2[0],2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                            elif len(mv1) > 0 and len(mv2) == 0:
                                new_id = end1[mv1[0],0]
                                new_dist = rch_len+end1[mv1[0],1]
                                new_type = end1[mv1[0],2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                            elif len(mv1) > 0 and len(mv2) > 0:
                                new_id = end1[mv1[0],0]
                                new_dist = rch_len+end1[mv1[0],1]
                                new_type = end1[mv1[0],2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                    else:
                        dummy_basin_rch[rch] = 0
                        dummy_basin_dist[rch] = 0
                        dummy_type[rch] = 0
                        continue

                ###############################################################

            ### should be re-calculated within while loop.
            small_rch = np.unique(dummy_basin_rch[np.where((dummy_basin_dist > 0) & (dummy_basin_dist < min_dist))[0]])
            small_flag = np.array([np.unique(dummy_type[np.where(dummy_basin_rch == index)[0][0]]) for index in small_rch]).flatten()
            small_dist = np.array([np.unique(dummy_basin_dist[np.where(dummy_basin_rch == index)[0][0]]) for index in small_rch]).flatten()
            small_rivers = small_rch[np.where((small_flag > 1) & (small_flag !=4))[0]]
            small_rivers_dist = small_dist[np.where((small_flag > 1) & (small_flag !=4))[0]]
            loop = loop+1
            if loop == 200:
                print(ind, 'LOOP STUCK')

        ### should be replaced after while loop.
        new_rch_id[basin] = basin_rch
        new_rch_dist[basin] = basin_dist
        new_flag[basin] = basin_flag

    return new_rch_id, new_rch_dist, new_flag

###############################################################################

def aggregate_dams(subcls, min_dist):

    """
    FUNCTION:
        Aggregates dam reach types with reach lengths less than a specified
        minimum distance.

    INPUTS
        subcls -- Object containing attributes for the high-resolution
            centerline.
            [attributes used]:
                lon - Longtitude values along the high-resolution centerline.
                lat - Latitude values along the high-resolution centerline.
                rch_id4 -- Numbered reaches after aggregating short lake and
                    river reaches.
                rch_len4 -- Reach lengths after aggregating short lake and
                    river reaches (meters).
                rch_ind4 -- Point indexes after aggregating short lake and
                    river reaches.
                type4 -- Type flag after aggregating short lake and river
                    reaches (1 = river, 2 = lake, 3 = lake on river, 4 = dam
                    5 = no topology).
                elv -- Elevation values along the high-resolution centerline
                    (meters).
                facc -- Flow accumulation along the high-resolution ceterline
                    (km^2).
        min_dist -- Minimum reach length.

    OUTPUTS
        new_rch_id -- Updated reach IDs (1-D array).
        new_rch_dist -- Updated reach lengths (1-D array).
        new_rch_flag -- Updated reach type (1-D array)
    """

    # Set variables.
    new_rch_id = np.copy(subcls.rch_id3)
    new_rch_dist = np.copy(subcls.rch_len3)
    new_flag = np.copy(subcls.type3)
    level4 = np.array([np.int(np.str(point)[0:4]) for point in subcls.basins])
    uniq_basins = np.unique(level4) #np.unique(subcls.basins)
    for ind in list(range(len(uniq_basins))):
        #print(ind)
        basin = np.where(level4 == uniq_basins[ind])[0]
        basin_l6 =  subcls.basins[basin]
        basin_rch = subcls.rch_id3[basin]
        basin_dist = subcls.rch_len3[basin]
        basin_flag = subcls.type3[basin]
        basin_acc = subcls.facc[basin]
        basin_wse = subcls.elv[basin]
        basin_lon = subcls.lon[basin]
        basin_lat = subcls.lat[basin]
        basin_ind = subcls.rch_ind3[basin]
        basin_x, basin_y, __, __ = reproject_utm(basin_lat, basin_lon)

        #creating dummy vectors to help keep track of changes.
        dummy_basin_rch = np.copy(basin_rch)
        dummy_basin_dist = np.copy(basin_dist)
        dummy_type = np.copy(basin_flag)

        # finding intial small reaches.
        small_rch = np.unique(basin_rch[np.where((basin_dist > 0) & (basin_dist < min_dist))[0]])
        small_flag = np.array([np.unique(basin_flag[np.where(basin_rch == index)][0]) for index in small_rch]).flatten()
        small_dist = np.array([np.unique(basin_dist[np.where(basin_rch == index)][0]) for index in small_rch]).flatten()
        small_rivers = small_rch[np.where(small_flag == 4)]
        small_rivers_dist = small_dist[np.where(small_flag == 4)]
        #print(ind, len(small_rivers))

        #looping through short reaches and aggregating them based on a series of
        #logical conditions based on reach type and hydrological boundaries.
        loop = 1
        while len(small_rivers_dist) > 0:
            for idx in list(range(len(small_rivers))):
                #print(idx)

                if small_rivers[idx] == -9999:
                    continue

                rch = np.where(basin_rch == small_rivers[idx])[0]
                rch_id = small_rivers[idx]
                rch_len = np.unique(basin_dist[rch])
                rch_l6 = np.unique(basin_l6[rch])[0]
                rch_flag = np.max(np.unique(basin_flag[rch]))
                rch_x = basin_x[rch]
                rch_y = basin_y[rch]
                rch_ind = basin_ind[rch]
                end1, end2 = find_neighbors(basin_rch, basin_dist, basin_flag,
                                            basin_acc, basin_wse, basin_x, basin_y, rch_x,
                                            rch_y, rch_ind, rch_len, rch_id, rch)

                # filtering out single neighbors that cross level 6 basin lines.
                if len(end1) == 1:
                    end1_l6 = np.unique(basin_l6[np.where(basin_rch == end1[0,0])[0]])[0]
                    if end1_l6 == rch_l6:
                        end1 = end1
                    else:
                        end1 = []

                if len(end2) == 1:
                    end2_l6 = np.unique(basin_l6[np.where(basin_rch == end2[0,0])[0]])[0]
                    if end2_l6 == rch_l6:
                        end2 = end2
                    else:
                        end2 = []

                ###############################################################
                # no bounding reaches.
                ###############################################################

                if len(end1) == 0 and len(end2) == 0:
                    #print(idx, 'cond 1 - no bordering reaches')
                    dummy_basin_rch[rch] = 0
                    dummy_basin_dist[rch] = 0
                    dummy_type[rch] = 0
                    continue

                ###############################################################
                # only one reach on one end.
                ###############################################################

                elif len(end1) == 0 and len(end2) == 1:
                    #print(idx, 'cond 2')
                    # dam.
                    if end2[:,2] == rch_flag:
                        #print('    subcond. 1 - dam')
                        new_id = end2[:,0]
                        new_dist = rch_len+end2[:,1]
                        new_type = end2[:,2]
                        rch2 = np.where(basin_rch == new_id)[0]
                        dummy_basin_dist[rch2] = new_dist
                        basin_dist[rch2] = new_dist
                        dummy_basin_rch[rch] = new_id
                        dummy_basin_dist[rch] = new_dist
                        dummy_type[rch] = new_type
                        basin_rch[rch] = new_id
                        basin_dist[rch] = new_dist
                        basin_flag[rch] =   new_type
                        if new_id in small_rivers:
                            update = np.where(small_rivers == new_id)[0]
                            small_rivers[update] = -9999
                            small_rivers_dist[update] = -9999

                    # non-dam - attach is reach is < 100 m.
                    if end2[:,2] != rch_flag:
                        #print('    subcond. 2 - non-dam')
                        if rch_len < 100:
                            new_id = end2[:,0]
                            new_dist = rch_len+end2[:,1]
                            new_type = end2[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                            dummy_basin_rch[rch] = 0
                            dummy_basin_dist[rch] = 0
                            dummy_type[rch] = 0
                            continue

                ###############################################################
                # multiple reaches on one end only.
                ###############################################################

                elif len(end1) == 0 and len(end2) > 1:
                    #print(idx, 'cond 3 - short reach with two tributaries on one end.')
                    dummy_basin_rch[rch] = 0
                    dummy_basin_dist[rch] = 0
                    dummy_type[rch] = 0
                    continue

                ###############################################################
                # only one reach on one end.
                ###############################################################

                elif len(end1) == 1 and len(end2) == 0:
                    #print(idx, 'cond 4')
                    # dam
                    if end1[:,2] == rch_flag:
                        #print('    subcond. 1 - dam')
                        new_id = end1[:,0]
                        new_dist = rch_len+end1[:,1]
                        new_type = end1[:,2]
                        rch2 = np.where(basin_rch == new_id)[0]
                        dummy_basin_dist[rch2] = new_dist
                        basin_dist[rch2] = new_dist
                        dummy_basin_rch[rch] = new_id
                        dummy_basin_dist[rch] = new_dist
                        dummy_type[rch] = new_type
                        basin_rch[rch] = new_id
                        basin_dist[rch] = new_dist
                        basin_flag[rch] =   new_type
                        if new_id in small_rivers:
                            update = np.where(small_rivers == new_id)[0]
                            small_rivers[update] = -9999
                            small_rivers_dist[update] = -9999
                    # non-dam - attach if reach < 100 m.
                    if end1[:,2] != rch_flag:
                        #print('    subcond. 2 - non-dam')
                        if rch_len < 100:
                            new_id = end1[:,0]
                            new_dist = rch_len+end1[:,1]
                            new_type = end1[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                            dummy_basin_rch[rch] = 0
                            dummy_basin_dist[rch] = 0
                            dummy_type[rch] = 0
                            continue

                ###############################################################
                # one reach is on each end.
                ###############################################################

                elif len(end1) == 1 and len(end2) == 1:
                    #print(idx, 'cond 5')
                    # two rivers bordering the reach.
                    if end1[:,2] == 1 and end2[:,2] == 1:
                        #print('    subcond. 1 -> 2 bordering rivers')
                        # put river reach if <100 m.
                        if rch_len < 100:
                            new_id = end2[:,0]
                            new_dist = rch_len+end2[:,1]
                            new_type = end2[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                                dummy_basin_rch[rch] = 0
                                dummy_basin_dist[rch] = 0
                                dummy_type[rch] = 0
                                continue
                    # river and non river bordering the reach.
                    elif end1[:,2] == 1 and end2[:,2] > 1:
                        #print('    subcond. 2 -> 1 bordering river')
                        # Attach to non-dam if shorter than 100 m.
                        if end2[:,2] == rch_flag:
                            new_id = end2[:,0]
                            new_dist = rch_len+end2[:,1]
                            new_type = end2[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                            if rch_len < 100:
                                new_id = end2[:,0]
                                new_dist = rch_len+end2[:,1]
                                new_type = end2[:,2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                            else:
                                dummy_basin_rch[rch] = 0
                                dummy_basin_dist[rch] = 0
                                dummy_type[rch] = 0
                                continue
                    # river and non river bordering the reach.
                    elif end1[:,2] > 1 and end2[:,2] == 1:
                        #print('    subcond. 3 -> 1 bordering river')
                        # Attach to non-dam if shorter than 100 m.
                        if end1[:,2] == rch_flag:
                            new_id = end1[:,0]
                            new_dist = rch_len+end1[:,1]
                            new_type = end1[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                            if rch_len < 100:
                                new_id = end1[:,0]
                                new_dist = rch_len+end1[:,1]
                                new_type = end1[:,2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                            else:
                                dummy_basin_rch[rch] = 0
                                dummy_basin_dist[rch] = 0
                                dummy_type[rch] = 0
                                continue
                    # two non-rivers bordering the reach.
                    elif end1[:,2] > 1 and end2[:,2] > 1:
                        #print('    subcond. 4 - no bordering rivers')
                        vals = np.array([end1[:,2], end2[:,2]]).flatten()
                        # two borering of same flag.
                        if end1[:,2] == rch_flag and end2[:,2] == rch_flag:
                            #print('    two bordering dams')
                            new_id = end1[:,0]
                            new_dist = rch_len+end1[:,1]
                            new_type = end1[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        # lake and dam.
                        elif np.min(vals) == 3 and np.max(vals) == 4:
                            #print('    lake and dam')
                            keep = np.where(vals == 4)[0]
                            if keep == 0:
                                new_id = end1[:,0]
                                new_dist = rch_len+end1[:,1]
                                new_type = end1[:,2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                            else:
                                new_id = end2[:,0]
                                new_dist = rch_len+end2[:,1]
                                new_type = end2[:,2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                        # dam and coast
                        elif np.min(vals) == 4 and np.max(vals) == 5:
                            #print('    dam and coast')
                            keep = np.where(vals == 4)[0]
                            if keep == 0:
                                new_id = end1[:,0]
                                new_dist = rch_len+end1[:,1]
                                new_type = end1[:,2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                            else:
                                new_id = end2[:,0]
                                new_dist = rch_len+end2[:,1]
                                new_type = end2[:,2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                        # lake and coast
                        elif np.min(vals) == 3 and np.max(vals) == 5:
                            #print('    lake and coast')
                            if rch_len < 100:
                                new_id = end1[:,0]
                                new_dist = rch_len+end1[:,1]
                                new_type = end1[:,2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                            else:
                                dummy_basin_rch[rch] = 0
                                dummy_basin_dist[rch] = 0
                                dummy_type[rch] = 0
                                continue
                        # coast and coast
                        elif np.min(vals) == 5 and np.max(vals) == 5:
                            #print('    two coasts')
                            if rch_len < 100:
                                new_id = end1[:,0]
                                new_dist = rch_len+end1[:,1]
                                new_type = end1[:,2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                            else:
                                dummy_basin_rch[rch] = 0
                                dummy_basin_dist[rch] = 0
                                dummy_type[rch] = 0
                                continue
                        # lake and lake
                        elif np.min(vals) == 3 and np.max(vals) == 3:
                            #print('    two lakes')
                            if rch_len < 100:
                                new_id = end1[:,0]
                                new_dist = rch_len+end1[:,1]
                                new_type = end1[:,2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                            else:
                                dummy_basin_rch[rch] = 0
                                dummy_basin_dist[rch] = 0
                                dummy_type[rch] = 0
                                continue

                ###############################################################
                # one reach at one end and multiple reaches on the other end.
                ###############################################################

                elif len(end1) == 1 and len(end2) > 1:
                    #print(idx, 'cond 6')
                    if end1[:,2] == rch_flag:
                        #print('    subcond 1 - dam on single end.')
                        new_id = end1[:,0]
                        new_dist = rch_len+end1[:,1]
                        new_type = end1[:,2]
                        rch2 = np.where(basin_rch == new_id)[0]
                        dummy_basin_dist[rch2] = new_dist
                        basin_dist[rch2] = new_dist
                        dummy_basin_rch[rch] = new_id
                        dummy_basin_dist[rch] = new_dist
                        dummy_type[rch] = new_type
                        basin_rch[rch] = new_id
                        basin_dist[rch] = new_dist
                        basin_flag[rch] =   new_type
                        if new_id in small_rivers:
                            update = np.where(small_rivers == new_id)[0]
                            small_rivers[update] = -9999
                            small_rivers_dist[update] = -9999
                    elif end1[:,2] != rch_flag:
                        #print('    subcond 2 - non-dam on single end.')
                        if rch_len < 100:
                            new_id = end1[:,0]
                            new_dist = rch_len+end1[:,1]
                            new_type = end1[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                            dummy_basin_rch[rch] = 0
                            dummy_basin_dist[rch] = 0
                            dummy_type[rch] = 0
                            continue

                ###############################################################
                # multiple reaches on one end only.
                ###############################################################

                elif len(end1) > 1 and len(end2) == 0:
                    #print(idx, 'cond 7 - short reach with two tributaries on one end.')
                    dummy_basin_rch[rch] = 0
                    dummy_basin_dist[rch] = 0
                    dummy_type[rch] = 0
                    continue

                ###############################################################
                # one reach at one end and multiple reaches on the other end.
                ###############################################################

                elif len(end1) > 1 and len(end2) == 1:
                    #print(idx, 'cond 8')
                    if end2[:,2] == rch_flag:
                        #print('    subcond 1 - dam on single end.')
                        new_id = end2[:,0]
                        new_dist = rch_len+end2[:,1]
                        new_type = end2[:,2]
                        rch2 = np.where(basin_rch == new_id)[0]
                        dummy_basin_dist[rch2] = new_dist
                        basin_dist[rch2] = new_dist
                        dummy_basin_rch[rch] = new_id
                        dummy_basin_dist[rch] = new_dist
                        dummy_type[rch] = new_type
                        basin_rch[rch] = new_id
                        basin_dist[rch] = new_dist
                        basin_flag[rch] =   new_type
                        if new_id in small_rivers:
                            update = np.where(small_rivers == new_id)[0]
                            small_rivers[update] = -9999
                            small_rivers_dist[update] = -9999
                    elif end2[:,2] != rch_flag:
                        #print('    subcond 2 - non-dam on single end.')
                        if rch_len < 100:
                            new_id = end2[:,0]
                            new_dist = rch_len+end2[:,1]
                            new_type = end2[:,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        else:
                            dummy_basin_rch[rch] = 0
                            dummy_basin_dist[rch] = 0
                            dummy_type[rch] = 0
                            continue
                ###############################################################
                # multiple reaches at both ends.
                ###############################################################

                elif len(end1) > 1 and len(end2) > 1:
                    #print(idx, 'cond 9')
                    if rch_len < 100:
                        singles = np.unique(np.concatenate([end1[:,0], end2[:,0]]))
                        if len(singles) == 1:
                            #print('    subcond 1. - all duplicates, one real neighbor')
                            new_id = end1[0,0]
                            new_dist = rch_len+end1[0,1]
                            new_type = end1[0,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        elif len(singles) == 2:
                            #print('    subcond 2. - only two real neighbors')
                            new_id = end1[0,0]
                            new_dist = rch_len+end1[0,1]
                            new_type = end1[0,2]
                            rch2 = np.where(basin_rch == new_id)[0]
                            dummy_basin_dist[rch2] = new_dist
                            basin_dist[rch2] = new_dist
                            dummy_basin_rch[rch] = new_id
                            dummy_basin_dist[rch] = new_dist
                            dummy_type[rch] = new_type
                            basin_rch[rch] = new_id
                            basin_dist[rch] = new_dist
                            basin_flag[rch] =   new_type
                            if new_id in small_rivers:
                                update = np.where(small_rivers == new_id)[0]
                                small_rivers[update] = -9999
                                small_rivers_dist[update] = -9999
                        elif len(singles) > 2:
                            #print('    subcond 3. - at least three real neighbors')
                            vals = np.unique(np.concatenate([end1[:,2], end2[:,2]]))
                            mv1 = np.where(end1[:,2] == np.max(vals))[0]
                            mv2 = np.where(end2[:,2] == np.max(vals))[0]
                            if len(mv1) == 0 and len(mv2) > 0:
                                new_id = end2[mv2[0],0]
                                new_dist = rch_len+end2[mv2[0],1]
                                new_type = end2[mv2[0],2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                            elif len(mv1) > 0 and len(mv2) == 0:
                                new_id = end1[mv1[0],0]
                                new_dist = rch_len+end1[mv1[0],1]
                                new_type = end1[mv1[0],2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                            elif len(mv1) > 0 and len(mv2) > 0:
                                new_id = end1[mv1[0],0]
                                new_dist = rch_len+end1[mv1[0],1]
                                new_type = end1[mv1[0],2]
                                rch2 = np.where(basin_rch == new_id)[0]
                                dummy_basin_dist[rch2] = new_dist
                                basin_dist[rch2] = new_dist
                                dummy_basin_rch[rch] = new_id
                                dummy_basin_dist[rch] = new_dist
                                dummy_type[rch] = new_type
                                basin_rch[rch] = new_id
                                basin_dist[rch] = new_dist
                                basin_flag[rch] =   new_type
                                if new_id in small_rivers:
                                    update = np.where(small_rivers == new_id)[0]
                                    small_rivers[update] = -9999
                                    small_rivers_dist[update] = -9999
                    else:
                        dummy_basin_rch[rch] = 0
                        dummy_basin_dist[rch] = 0
                        dummy_type[rch] = 0
                        continue

                ###############################################################

            ### should be re-calculated within while loop.
            small_rch = np.unique(dummy_basin_rch[np.where((dummy_basin_dist > 0) & (dummy_basin_dist < min_dist))[0]])
            small_flag = np.array([np.unique(dummy_type[np.where(dummy_basin_rch == index)[0][0]]) for index in small_rch]).flatten()
            small_dist = np.array([np.unique(dummy_basin_dist[np.where(dummy_basin_rch == index)[0][0]]) for index in small_rch]).flatten()
            small_rivers = small_rch[np.where(small_flag == 4)[0]]
            small_rivers_dist = small_dist[np.where(small_flag == 4)[0]]
            loop = loop+1
            if loop == 200:
                print(ind, 'LOOP STUCK')

        ### should be replaced after while loop.
        new_rch_id[basin] = basin_rch
        new_rch_dist[basin] = basin_dist
        new_flag[basin] = basin_flag

    return new_rch_id, new_rch_dist, new_flag

###############################################################################
##################### Topology and Attribute Functions ########################
###############################################################################

def order_reaches(basin, basin_rch, basin_acc, basin_wse, basin_dist, basin_flag,
                  basin_lon, basin_lat, basin_x, basin_y, segInd, start_id):

    """
    FUNCTION:
        Orders reaches within a basin based on flow accumulation and elevation.
        Reach order will start at the downstream end and increase going
        upstream. This is a sub-function of the "reach_topology" function.

    INPUTS
        basin -- Index locations for points within the basin.
        basin_rch -- Reach IDs within the basin.
        basin_dist -- Reach lengths for the reaches in the basin.
        basin_flag -- Reach types for the basin.
        basin_acc -- Flow accumulation values for the reaches in the basin.
        basin_wse -- Elevation values for the reaches in the basin.
        basin_lon -- Longitude values for all points in the basin.
        basin_lat -- Latitude values for all points in the basin.
        basin_x -- Easting values for all points in the basin.
        basin_y -- Northing values for all points in the basin.
        segInd -- Point indexes for all high-resolution centerline points.
        start_id -- Starting value to start ordering the reaches.

    OUTPUTS
        basin_topo -- 1-D array of ordered reaches within a basin.
    """

    # Set variables.
    basin_topo =  np.zeros(len(basin))
    upa_topo = np.copy(basin_acc)
    start_upa = np.where(upa_topo == np.max(upa_topo))[0]
    start_id = start_id
    loop = 1

    # Loop that "walks" up the river based on flow accumulation until the
    # headwaters are reached.
    while np.max(upa_topo) > 0:

        # Current reach information.
        rch_id = np.unique(basin_rch[start_upa])[0]

        rch = np.where(basin_rch == rch_id)[0]
        basin_topo[rch] = start_id
        start_id = start_id+1
        upa_topo[rch] = 0

        rch_ind = segInd[basin[rch]]
        rch_x = basin_x[rch]
        rch_y = basin_y[rch]
        rch_len = np.unique(basin_dist[rch])
        rch_acc = np.max(basin_acc[rch])
        rch_wse = np.min(basin_wse[rch])

        # Find upstream neighboring reaches.
        ngh1, ngh2 = find_neighbors(basin_rch, basin_dist, basin_flag, basin_acc,
                          basin_wse, basin_x, basin_y, rch_x, rch_y, rch_ind, rch_len,
                          rch_id, rch)

        # No neighbors. Go to next reach with highest flow accumulation.
        if len(ngh1) == 0 and len(ngh2) == 0:
            start_upa = np.where(upa_topo == np.max(upa_topo))[0]
            continue
        # One neighbor on one end.
        elif len(ngh1) == 0 and len(ngh2) > 0:
            # neighbor elevation is less than current reach.
            if np.min(ngh2[:,4]) < rch_wse:
                start_upa = np.where(upa_topo == np.max(upa_topo))[0]
                continue
            # neighbor elevation is equal to current reach.
            elif np.min(ngh2[:,4]) == rch_wse:
                if np.max(ngh2[:,3]) >= rch_acc:
                    start_upa = np.where(upa_topo == np.max(upa_topo))[0]
                    continue
                else:
                    ngh_rch = ngh2[:,0]
                    ngh_acc = ngh2[:,3]
                    ngh_wse = ngh2[:,4]
            # neighbor elevation is greater than current reach.
            else:
                ngh_rch = ngh2[:,0]
                ngh_acc = ngh2[:,3]
                ngh_wse = ngh2[:,4]

        # One neighbor on one end.
        elif len(ngh1) > 0 and len(ngh2) == 0:
            # neighbor elevation is less than current reach.
            if np.min(ngh1[:,4]) < rch_wse:
                start_upa = np.where(upa_topo == np.max(upa_topo))[0]
                continue
            # neighbor elevation is equal to current reach.
            elif np.min(ngh1[:,4]) == rch_wse:
                if np.max(ngh1[:,3]) >= rch_acc:
                    start_upa = np.where(upa_topo == np.max(upa_topo))[0]
                    continue
                else:
                    ngh_rch = ngh1[:,0]
                    ngh_acc = ngh1[:,3]
                    ngh_wse = ngh1[:,4]
            # neighbor elevation is greater than current reach.
            else:
                ngh_rch = ngh1[:,0]
                ngh_acc = ngh1[:,3]
                ngh_wse = ngh1[:,4]

        # One neighbor on each end.
        elif len(ngh1) == 1 and len(ngh2) == 1:
            if np.unique(ngh1[:,0]) == np.unique(ngh2[:,0]):
                ngh_rch = ngh1[:,0]
                ngh_acc = ngh1[:,3]
                ngh_wse = ngh1[:,4]
            else:
                if np.min(ngh1[:,4]) > rch_wse and np.min(ngh2[:,4]) < rch_wse:
                    ngh_rch = ngh1[:,0]
                    ngh_acc = ngh1[:,3]
                    ngh_wse = ngh1[:,4]

                elif np.min(ngh1[:,4]) == rch_wse and np.min(ngh2[:,4]) > rch_wse:
                    ngh_rch = ngh2[:,0]
                    ngh_acc = ngh2[:,3]
                    ngh_wse = ngh2[:,4]

                elif np.min(ngh1[:,4]) < rch_wse and np.min(ngh2[:,4]) == rch_wse:
                    start_upa = np.where(upa_topo == np.max(upa_topo))[0]
                    continue

                elif np.min(ngh1[:,4]) < rch_wse and np.min(ngh2[:,4]) > rch_wse:
                    ngh_rch = ngh2[:,0]
                    ngh_acc = ngh2[:,3]
                    ngh_wse = ngh2[:,4]

                elif np.min(ngh1[:,4]) == rch_wse and np.min(ngh2[:,4]) < rch_wse:
                    start_upa = np.where(upa_topo == np.max(upa_topo))[0]
                    continue

                elif np.min(ngh1[:,4]) > rch_wse and np.min(ngh2[:,4]) == rch_wse:
                    ngh_rch = ngh1[:,0]
                    ngh_acc = ngh1[:,3]
                    ngh_wse = ngh1[:,4]

                elif np.min(ngh1[:,4]) < rch_wse and np.min(ngh2[:,4]) < rch_wse:
                    start_upa = np.where(upa_topo == np.max(upa_topo))[0]
                    continue

                elif np.min(ngh1[:,4]) == rch_wse and np.min(ngh2[:,4]) == rch_wse:
                    if np.max(ngh1[:,3]) > np.max(ngh2[:,3]):
                        ngh_rch = ngh2[:,0]
                        ngh_acc = ngh2[:,3]
                        ngh_wse = ngh2[:,4]
                    elif np.max(ngh1[:,3]) < np.max(ngh2[:,3]):
                        ngh_rch = ngh1[:,0]
                        ngh_acc = ngh1[:,3]
                        ngh_wse = ngh1[:,4]
                    else:
                        start_upa = np.where(upa_topo == np.max(upa_topo))[0]
                        continue

                elif np.min(ngh1[:,4]) > rch_wse and np.min(ngh2[:,4]) > rch_wse:
                    if np.max(ngh1[:,3]) > np.max(ngh2[:,3]):
                        ngh_rch = ngh2[:,0]
                        ngh_acc = ngh2[:,3]
                        ngh_wse = ngh2[:,4]
                    elif np.max(ngh1[:,3]) < np.max(ngh2[:,3]):
                        ngh_rch = ngh1[:,0]
                        ngh_acc = ngh1[:,3]
                        ngh_wse = ngh1[:,4]
                    else:
                        start_upa = np.where(upa_topo == np.max(upa_topo))[0]
                        continue

        # Multiple neighbors on one end and one neighbor on the other end.
        elif len(ngh1) > 1 and len(ngh2) == 1:
            if np.min(ngh1[:,4]) > rch_wse and np.min(ngh2[:,4]) < rch_wse:
                ngh_rch = ngh1[:,0]
                ngh_acc = ngh1[:,3]
                ngh_wse = ngh1[:,4]

            elif np.min(ngh1[:,4]) == rch_wse and np.min(ngh2[:,4]) > rch_wse:
                ngh_rch = ngh2[:,0]
                ngh_acc = ngh2[:,3]
                ngh_wse = ngh2[:,4]

            elif np.min(ngh1[:,4]) < rch_wse and np.min(ngh2[:,4]) == rch_wse:
                start_upa = np.where(upa_topo == np.max(upa_topo))[0]
                continue

            elif np.min(ngh1[:,4]) < rch_wse and np.min(ngh2[:,4]) > rch_wse:
                ngh_rch = ngh2[:,0]
                ngh_acc = ngh2[:,3]
                ngh_wse = ngh2[:,4]

            elif np.min(ngh1[:,4]) == rch_wse and np.min(ngh2[:,4]) < rch_wse:
                start_upa = np.where(upa_topo == np.max(upa_topo))[0]
                continue

            elif np.min(ngh1[:,4]) > rch_wse and np.min(ngh2[:,4]) == rch_wse:
                ngh_rch = ngh1[:,0]
                ngh_acc = ngh1[:,3]
                ngh_wse = ngh1[:,4]

            elif np.min(ngh1[:,4]) < rch_wse and np.min(ngh2[:,4]) < rch_wse:
                start_upa = np.where(upa_topo == np.max(upa_topo))[0]
                continue

            elif np.min(ngh1[:,4]) == rch_wse and np.min(ngh2[:,4]) == rch_wse:
                if np.max(ngh1[:,3]) > np.max(ngh2[:,3]):
                    ngh_rch = ngh2[:,0]
                    ngh_acc = ngh2[:,3]
                    ngh_wse = ngh2[:,4]
                elif np.max(ngh1[:,3]) < np.max(ngh2[:,3]):
                    ngh_rch = ngh1[:,0]
                    ngh_acc = ngh1[:,3]
                    ngh_wse = ngh1[:,4]
                else:
                    start_upa = np.where(upa_topo == np.max(upa_topo))[0]
                    continue

            elif np.min(ngh1[:,4]) > rch_wse and np.min(ngh2[:,4]) > rch_wse:
                if np.max(ngh1[:,3]) > np.max(ngh2[:,3]):
                    ngh_rch = ngh2[:,0]
                    ngh_acc = ngh2[:,3]
                    ngh_wse = ngh2[:,4]
                elif np.max(ngh1[:,3]) < np.max(ngh2[:,3]):
                    ngh_rch = ngh1[:,0]
                    ngh_acc = ngh1[:,3]
                    ngh_wse = ngh1[:,4]
                else:
                    start_upa = np.where(upa_topo == np.max(upa_topo))[0]
                    continue

        # Multiple neighbors on one end and one neighbor on the other end.
        elif len(ngh1) == 1 and len(ngh2) > 1:
            if np.min(ngh1[:,4]) > rch_wse and np.min(ngh2[:,4]) < rch_wse:
                ngh_rch = ngh1[:,0]
                ngh_acc = ngh1[:,3]
                ngh_wse = ngh1[:,4]

            elif np.min(ngh1[:,4]) == rch_wse and np.min(ngh2[:,4]) > rch_wse:
                ngh_rch = ngh2[:,0]
                ngh_acc = ngh2[:,3]
                ngh_wse = ngh2[:,4]

            elif np.min(ngh1[:,4]) < rch_wse and np.min(ngh2[:,4]) == rch_wse:
                start_upa = np.where(upa_topo == np.max(upa_topo))[0]
                continue

            elif np.min(ngh1[:,4]) < rch_wse and np.min(ngh2[:,4]) > rch_wse:
                ngh_rch = ngh2[:,0]
                ngh_acc = ngh2[:,3]
                ngh_wse = ngh2[:,4]

            elif np.min(ngh1[:,4]) == rch_wse and np.min(ngh2[:,4]) < rch_wse:
                start_upa = np.where(upa_topo == np.max(upa_topo))[0]
                continue

            elif np.min(ngh1[:,4]) > rch_wse and np.min(ngh2[:,4]) == rch_wse:
                ngh_rch = ngh1[:,0]
                ngh_acc = ngh1[:,3]
                ngh_wse = ngh1[:,4]

            elif np.min(ngh1[:,4]) < rch_wse and np.min(ngh2[:,4]) < rch_wse:
                start_upa = np.where(upa_topo == np.max(upa_topo))[0]
                continue

            elif np.min(ngh1[:,4]) == rch_wse and np.min(ngh2[:,4]) == rch_wse:
                if np.max(ngh1[:,3]) > np.max(ngh2[:,3]):
                    ngh_rch = ngh2[:,0]
                    ngh_acc = ngh2[:,3]
                    ngh_wse = ngh2[:,4]
                elif np.max(ngh1[:,3]) < np.max(ngh2[:,3]):
                    ngh_rch = ngh1[:,0]
                    ngh_acc = ngh1[:,3]
                    ngh_wse = ngh1[:,4]
                else:
                    start_upa = np.where(upa_topo == np.max(upa_topo))[0]
                    continue

            elif np.min(ngh1[:,4]) > rch_wse and np.min(ngh2[:,4]) > rch_wse:
                if np.max(ngh1[:,3]) > np.max(ngh2[:,3]):
                    ngh_rch = ngh2[:,0]
                    ngh_acc = ngh2[:,3]
                    ngh_wse = ngh2[:,4]
                elif np.max(ngh1[:,3]) < np.max(ngh2[:,3]):
                    ngh_rch = ngh1[:,0]
                    ngh_acc = ngh1[:,3]
                    ngh_wse = ngh1[:,4]
                else:
                    start_upa = np.where(upa_topo == np.max(upa_topo))[0]
                    continue

        # Multiple neighbors on both ends.
        elif len(ngh1) > 1 and len(ngh2) > 1:
            if np.min(ngh1[:,4]) > rch_wse and np.min(ngh2[:,4]) < rch_wse:
                ngh_rch = ngh1[:,0]
                ngh_acc = ngh1[:,3]
                ngh_wse = ngh1[:,4]

            elif np.min(ngh1[:,4]) == rch_wse and np.min(ngh2[:,4]) > rch_wse:
                ngh_rch = ngh2[:,0]
                ngh_acc = ngh2[:,3]
                ngh_wse = ngh2[:,4]

            elif np.min(ngh1[:,4]) < rch_wse and np.min(ngh2[:,4]) == rch_wse:
                start_upa = np.where(upa_topo == np.max(upa_topo))[0]
                continue

            elif np.min(ngh1[:,4]) < rch_wse and np.min(ngh2[:,4]) > rch_wse:
                ngh_rch = ngh2[:,0]
                ngh_acc = ngh2[:,3]
                ngh_wse = ngh2[:,4]

            elif np.min(ngh1[:,4]) == rch_wse and np.min(ngh2[:,4]) < rch_wse:
                start_upa = np.where(upa_topo == np.max(upa_topo))[0]
                continue

            elif np.min(ngh1[:,4]) > rch_wse and np.min(ngh2[:,4]) == rch_wse:
                ngh_rch = ngh1[:,0]
                ngh_acc = ngh1[:,3]
                ngh_wse = ngh1[:,4]

            elif np.min(ngh1[:,4]) < rch_wse and np.min(ngh2[:,4]) < rch_wse:
                start_upa = np.where(upa_topo == np.max(upa_topo))[0]
                continue

            elif np.min(ngh1[:,4]) == rch_wse and np.min(ngh2[:,4]) == rch_wse:
                if np.max(ngh1[:,3]) > np.max(ngh2[:,3]):
                    ngh_rch = ngh2[:,0]
                    ngh_acc = ngh2[:,3]
                    ngh_wse = ngh2[:,4]
                elif np.max(ngh1[:,3]) < np.max(ngh2[:,3]):
                    ngh_rch = ngh1[:,0]
                    ngh_acc = ngh1[:,3]
                    ngh_wse = ngh1[:,4]
                else:
                    start_upa = np.where(upa_topo == np.max(upa_topo))[0]
                    continue

            elif np.min(ngh1[:,4]) > rch_wse and np.min(ngh2[:,4]) > rch_wse:
                if np.max(ngh1[:,3]) > np.max(ngh2[:,3]):
                    ngh_rch = ngh2[:,0]
                    ngh_acc = ngh2[:,3]
                    ngh_wse = ngh2[:,4]
                elif np.max(ngh1[:,3]) < np.max(ngh2[:,3]):
                    ngh_rch = ngh1[:,0]
                    ngh_acc = ngh1[:,3]
                    ngh_wse = ngh1[:,4]
                else:
                    start_upa = np.where(upa_topo == np.max(upa_topo))[0]
                    continue

       ##################################################################

        # Assign next reach based on upstream neighbors.

        # No upstream neighbors - go to reach with highest flow accumulation.
        if len(np.where(ngh_rch > 0)[0]) == 0:
            start_upa = np.where(upa_topo == np.max(upa_topo))[0]
        # One upstream neighbor.
        elif len(np.where(ngh_rch > 0)[0]) == 1:
            if ngh_wse > rch_wse:
                val = np.where(ngh_rch[:] > 0)[0]
                start_upa = np.where(basin_rch == ngh_rch[val])[0]
            elif ngh_wse == rch_wse:
                if ngh_acc < rch_acc:
                    val = np.where(ngh_rch[:] > 0)[0]
                    start_upa = np.where(basin_rch == ngh_rch[val])[0]
                else:
                    start_upa = np.where(upa_topo == np.max(upa_topo))[0]
            else:
                start_upa = np.where(upa_topo == np.max(upa_topo))[0]
        # Multiple upstream neighbors.
        elif len(np.where(ngh_rch > 0)[0]) > 1:
            val = np.where(ngh_wse > rch_wse)[0]

            if len(val) == 0:
                start_upa = np.where(upa_topo == np.max(upa_topo))[0]
            elif len(val) == 1:
                start_upa = np.where(basin_rch == ngh_rch[val])[0]
            elif len(val) > 1:
                val2 = np.where(ngh_acc[val] == np.max(ngh_acc[val]))[0]
                if len(val2) == 1:
                    start_upa = np.where(basin_rch == ngh_rch[val[val2]])[0]
                else:
                    val2 = val[val2[0]]
                    start_upa = np.where(basin_rch == ngh_rch[val2])[0]

        #print(np.unique(basin_rch[start_upa]), loop)
        loop = loop+1
        if loop == 5000:
            print('LOOP STUCK')

    return basin_topo

###############################################################################

def reach_topology(subcls):

    """
    FUNCTION:
        Orders reaches within a basin and constructs the final Reach ID.

    INPUTS
        subcls -- Object that contains attributes along the high-resolution
            centerline locations.
            [attributes used]:
                lon -- Longitude values for the high-resolution centerline.
                lat -- Latitude values for the high-resolution centerline.
                rch_id5 -- Reach numbers along the high-resolution centerline.
                rch_len5 -- Reach length along the high-resolution centerline.
                rch_ind5 -- Point indexes for each reach along the
                    high-resolution centerline.
                type5 -- Type flag for each point in the high-resolution
                    centerline (1 = river, 2 = lake, 3 = lake on river,
                    4 = dam, 5 = no topology).
                elv -- Elevations along the high-resolution centerline (meters).
                facc -- Flow accumulation along the high-resolution ceterline
                    (km^2).
                basins -- Pfafstetter basin codes along the high-resolution
                    centerline.

    OUTPUTS
        reach_id -- 1-D array of reach IDs within a basin. Reach ID format is
            CBBBBBRRRRT (C = Continent, B = Pfafstetter Basin Code,
            R = Reach Number, T = Type).
        rch_topo -- 1-D array of ordered reach values within a basin.
    """

    # Calculate flow accmulation and elevation values for each reach.
    rch_acc = np.zeros(len(subcls.rch_id5))
    rch_wse = np.zeros(len(subcls.rch_id5))
    uniq_rch_id = np.unique(subcls.rch_id5)
    for ind in list(range(len(uniq_rch_id))):
        rch = np.where(subcls.rch_id5 == uniq_rch_id[ind])
        rch_acc[rch] = np.max(subcls.facc[rch])
        rch_wse[rch] = np.median(subcls.elv[rch])

    # Set variables.
    rch_topo = np.zeros(len(subcls.rch_id5))
    uniq_basins = np.unique(subcls.basins)

    # Loop through each basin and order the reaches.
    for ind in list(range(len(uniq_basins))):
        #print(ind)
        basin = np.where(subcls.basins == uniq_basins[ind])[0]
        basin_rch = subcls.rch_id5[basin]
        basin_acc = rch_acc[basin]
        basin_wse = rch_wse[basin]
        basin_dist = subcls.rch_len5[basin]
        basin_flag = subcls.type5[basin]
        basin_lon = subcls.lon[basin]
        basin_lat = subcls.lat[basin]
        basin_x, basin_y, __, __ = reproject_utm(basin_lat, basin_lon)

        basin_rivers = np.where(basin_flag < 5)[0]
        basin_notopo = np.where(basin_flag == 5)[0]
        basin_acc_rivers = np.copy(basin_acc)
        basin_acc_notopo = np.copy(basin_acc)
        basin_wse_rivers = np.copy(basin_wse)
        basin_wse_notopo = np.copy(basin_wse)
        basin_acc_rivers[basin_notopo] = 0
        basin_acc_notopo[basin_rivers] = 0

        # Order the reaches that are not in deltas.
        start_id = 1
        river_order = order_reaches(basin, basin_rch, basin_acc_rivers, basin_wse_rivers,\
                                    basin_dist, basin_flag, basin_lon, basin_lat, basin_x,\
                                    basin_y, subcls.rch_ind5, start_id)

        # Order the reaches with difficult topology (i.e. deltas).
        start_id = np.max(river_order)+1
        notopo_order = order_reaches(basin, basin_rch, basin_acc_notopo, basin_wse_notopo,\
                                     basin_dist, basin_flag, basin_lon, basin_lat, basin_x,\
                                     basin_y, subcls.rch_ind5, start_id)

        rch_topo[basin[np.where(river_order > 0)]] = river_order[np.where(river_order > 0)]
        rch_topo[basin[np.where(notopo_order > 0)]] = notopo_order[np.where(notopo_order > 0)]

    # Create formal reach id.
    reach_id = np.zeros(len(rch_topo), dtype = np.int64)
    for ind in list(range(len(rch_topo))):
        if len(np.str(np.int(rch_topo[ind]))) == 1:
            fill = '000'
            reach_id[ind] = np.int(np.str(subcls.basins[ind])+fill+np.str(np.int(rch_topo[ind]))+np.str(np.int(subcls.type5[ind])))
        if len(np.str(np.int(rch_topo[ind]))) == 2:
            fill = '00'
            reach_id[ind] = np.int(np.str(subcls.basins[ind])+fill+np.str(np.int(rch_topo[ind]))+np.str(np.int(subcls.type5[ind])))
        if len(np.str(np.int(rch_topo[ind]))) == 3:
            fill = '0'
            reach_id[ind] = np.int(np.str(subcls.basins[ind])+fill+np.str(np.int(rch_topo[ind]))+np.str(np.int(subcls.type5[ind])))
        if len(np.str(np.int(rch_topo[ind]))) == 4:
            reach_id[ind] = np.int(np.str(subcls.basins[ind])+np.str(np.int(rch_topo[ind]))+np.str(np.int(subcls.type5[ind])))

    return reach_id, rch_topo

###############################################################################

def node_reaches(subcls, node_len):

    """
    FUNCTION:
        Divides reaches up into nodes based on the specified node length and
        constructs the final Node IDs. Node IDs increase going upstream within
        a reach.

    INPUTS
        subcls -- Object that contains attributes along the high-resolution
            centerline locations.
            [attributes used]:
                reach_id -- Reach IDs for along the high-resolution centerline.
                rch_id5 -- Reach numbers along the high-resolution centerline.
                rch_len5 -- Reach length along the high-resolution centerline.
                rch_dist5 -- Flow distance along the high-resolution centerline
                    (meters).
                facc -- Flow accumulation along the high-resolution ceterline
                    (km^2).
        node_len -- Desired node length (meters).

    OUTPUTS
        node_id -- 1-D array of node IDs within a basin. Node ID format is
            CBBBBBRRRRNNNT (C = Continent, B = Pfafstetter Basin Code,
            R = Reach Number, N - Node Number, T = Type).
        node_dist -- Node lengths (meters).
        node_num -- 1-D array of ordered node values within a basin.
    """

    # Set variables.
    node_num = np.zeros(len(subcls.rch_id5))
    node_dist = np.zeros(len(subcls.rch_id5), dtype = 'f8')
    node_id = np.zeros(len(subcls.rch_id5), dtype = np.int64)
    uniq_rch = np.unique(subcls.rch_id5)

    # Loop through each reach and divide it up based on the specified node
    # length, then number the nodes in order of flow accumulation.
    for ind in list(range(len(uniq_rch))):

        # Current reach information.
        rch = np.where(subcls.rch_id5 == uniq_rch[ind])[0]
        distance = subcls.rch_dist5[rch]
        ID = subcls.rch_ind5[rch]

        # Temporary fill variables.
        temp_node_id = np.zeros(len(rch))
        temp_node_dist = np.zeros(len(rch))

        # Find node division points along the reach.
        d = np.unique(subcls.rch_len5[rch])
        if d <= node_len:
            divs = 1
        else:
            divs = np.round(d/node_len)

        divs_dist = d/divs

        break_index = np.zeros(np.int(divs-1))
        for idx in range(int(divs)-1):
            dist = divs_dist*(range(int(divs)-1)[idx]+1)+np.min(distance)
            cut = np.where(abs(distance - dist) == np.min(abs(distance - dist)))[0][0]
            break_index[idx] = cut

        div_ends = np.array([np.where(ID == np.min(ID))[0][0],np.where(ID == np.max(ID))[0][0]])
        borders = np.insert(div_ends, 0, break_index)
        border_ids = ID[borders]
        borders = borders[np.argsort(border_ids)]

        # Assign and order nodes.
        cnt = 1
        for idy in list(range(len(borders)-1)):
            index1 = borders[idy]
            index2 = borders[idy+1]

            ID1 = ID[index1]
            ID2 = ID[index2]

            if ID1 > ID2:
                vals = np.where((ID2 <= ID) &  (ID <= ID1))[0]
            else:
                vals = np.where((ID1 <= ID) &  (ID <= ID2))[0]

            temp_node_dist[vals] = abs(np.max(distance[vals])-np.min(distance[vals]))
            temp_node_id[vals] = cnt
            cnt=cnt+1

        first_node = np.where(temp_node_id == np.min(temp_node_id))[0]
        last_node = np.where(temp_node_id == np.max(temp_node_id))[0]

        if np.median(subcls.facc[rch[first_node]]) < np.median(subcls.facc[rch[last_node]]):
            node_num[rch] = abs(temp_node_id - np.max(temp_node_id))+1
            node_dist[rch] = temp_node_dist
        else:
            node_num[rch] = temp_node_id
            node_dist[rch] = temp_node_dist

        #if np.max(node_dist[rch])>node_len*2:
            #print(ind, 'max distance too long - likely an index problem')

        # Create formal Node ID.
        for inz in list(range(len(rch))):
            #if len(np.str(np.int(node_num[rch[inz]]))) > 3:
                #print(ind)
            if len(np.str(np.int(node_num[rch[inz]]))) == 1:
                fill = '00'
                node_id[rch[inz]] = np.int(np.str(subcls.reach_id[rch[inz]])[:-1]+fill+np.str(np.int(node_num[rch[inz]]))+np.str(subcls.reach_id[rch[inz]])[10:11])
            if len(np.str(np.int(node_num[rch[inz]]))) == 2:
                fill = '0'
                node_id[rch[inz]] = np.int(np.str(subcls.reach_id[rch[inz]])[:-1]+fill+np.str(np.int(node_num[rch[inz]]))+np.str(subcls.reach_id[rch[inz]])[10:11])
            if len(np.str(np.int(node_num[rch[inz]]))) == 3:
                node_id[rch[inz]] = np.int(np.str(subcls.reach_id[rch[inz]])[:-1]+np.str(np.int(node_num[rch[inz]]))+np.str(subcls.reach_id[rch[inz]])[10:11])

    return node_id, node_dist, node_num

###############################################################################

def basin_node_attributes(node_id, node_dist, height, width, facc, nchan, lon,
                          lat, reach_id, grod_id, lakes, grod_fid, hfalls_fid,
                          lake_id):

    """
    FUNCTION:
        Creates node locations and attributes from the high-resolution
        centerline points within an individual basin. This is a sub-function
        of the "node_attributes" function.

    INPUTS
        node_id -- Node IDs along the high-resolution centerline within a
            single basin.
        node_dist -- Node lengths along the high-resolution centerline within
            a single basin.
        height -- Elevations along the high-resolution centerline within a
            single basin (meters).
        width -- Widths along the high_resolution centerline within a single
            basin (meters).
        nchan -- Number of channels along the high_resolution centerline within
            a single basin.
        lon -- Longitude values within a single basin.
        lat -- Latitude values within a single basin.
        reach_id -- Reach IDs along the high-resolution centerline within a
            single basin.
        grod_id -- GROD dam locations along the high-resolution centerline within a
            single basin.
        lakes -- Lakeflag IDs along the high_resolution centerline within a
            single basin.
        grod_fid -- GROD IDs along the high_resolution centerline.
        hfalls_fid -- HydroFALLS IDs along the high_resolution centerline.
        lake_id -- Prior Lake Databse IDs along the high_resolution centerline.
        facc -- Flow accumulation values along the high_resolution centerline.

    OUTPUTS
        Node_ID -- Node ID respresenting a single node location.
        node_x -- Average longitude value calculated from the
            high-resolution centerline points associated with a node.
        node_y -- Average latitude value calculated from the
            high-resolution centerline points associated with a node.
        node_len -- Node length for a single node (meters).
        node_wse -- Average water surface elevation value calculated from the
            high-resolution centerlines points assosicated with a node (meters).
        node_wse_var -- Water surface elevation variablity calculated from the
            high-resolution centerlines points assosicated with a node (meters).
        node_wth -- Average width value calculated from the high-resolution
            centerlines points assosicated with a node ID (meters).
        node_wth_var -- Width variablity calculated from the high-resolution
            centerlines points assosicated with a node (meters).
        node_nchan_max -- Maximum number of channels calculated from
            the high-resolution centerline points associated with a node.
        node_nchan_mod -- Mode of the number of channels calculated from the
            high-resolution centerline points associated with a node.
        node_rch_id --  Reach ID that a particular node belongs to.
        node_grod_id -- GROD dam locations associated with a node.
        node_lakeflag -- GRWL lakeflag associated with a node.
        node_lake_id = Prior Lake Database ID associated with a node.
        node_facc = Flow Accumulation value associated with a node.
        node_grod_fid = GROD ID associated with a node.
        node_hfalls_fid = HydroFALLS ID associated with a node.
    """

    # Set variables.
    Node_ID = np.zeros(len(np.unique(node_id)))
    node_x = np.zeros(len(np.unique(node_id)))
    node_y = np.zeros(len(np.unique(node_id)))
    node_wse = np.zeros(len(np.unique(node_id)))
    node_wse_var = np.zeros(len(np.unique(node_id)))
    node_wth = np.zeros(len(np.unique(node_id)))
    node_wth_var = np.zeros(len(np.unique(node_id)))
    node_len = np.zeros(len(np.unique(node_id)))
    node_nchan_max = np.zeros(len(np.unique(node_id)))
    node_nchan_mod = np.zeros(len(np.unique(node_id)))
    node_rch_id = np.zeros(len(np.unique(node_id)))
    node_grod_id = np.zeros(len(np.unique(node_id)))
    node_lakeflag = np.zeros(len(np.unique(node_id)))
    node_lake_id = np.zeros(len(np.unique(node_id)))
    node_facc = np.zeros(len(np.unique(node_id)))
    node_grod_fid = np.zeros(len(np.unique(node_id)))
    node_hfalls_fid = np.zeros(len(np.unique(node_id)))

    uniq_nodes = np.unique(node_id)
    # Loop through each node ID to create location and attribute information.
    for ind in list(range(len(uniq_nodes))):
        nodes = np.where(node_id == uniq_nodes[ind])[0]
        Node_ID[ind] = int(np.unique(node_id[nodes])[0])
        node_rch_id[ind] = np.unique(reach_id[nodes])[0]
        node_x[ind] = np.mean(lon[nodes])
        node_y[ind] = np.mean(lat[nodes])
        node_len[ind] = np.unique(node_dist[nodes])[0]
        node_wse[ind] = np.median(height[nodes])
        node_wse_var[ind] = np.var(height[nodes])
        node_wth[ind] = np.median(width[nodes])
        node_wth_var[ind] = np.var(width[nodes])
        node_facc[ind] = np.max(facc[nodes])
        node_nchan_max[ind] = np.max(nchan[nodes])
        node_nchan_mod[ind] = max(set(list(nchan[nodes])), key=list(nchan[nodes]).count)
        node_lakeflag[ind] = max(set(list(lakes[nodes])), key=list(lakes[nodes]).count)
        node_lake_id[ind] = max(set(list(lake_id[nodes])), key=list(lake_id[nodes]).count)

        GROD = np.copy(grod_id[nodes])
        GROD[np.where(GROD > 4)] = 0
        node_grod_id[ind] = np.max(GROD)
        # Assign grod and hydrofalls ids to nodes.
        ID = np.where(GROD == np.max(GROD))[0][0]
        if np.max(GROD) == 0:
            node_grod_fid[ind] = 0
        elif np.max(GROD) == 4:
            node_hfalls_fid[ind] = hfalls_fid[nodes[ID]]
        else:
            node_grod_fid[ind] = grod_fid[nodes[ID]]

    return(Node_ID, node_x, node_y, node_len, node_wse, node_wse_var, node_wth,
           node_wth_var, node_facc, node_nchan_max, node_nchan_mod, node_rch_id,
           node_grod_id, node_lakeflag, node_grod_fid, node_hfalls_fid, node_lake_id)

###############################################################################

def node_attributes(subcls):

    """
    FUNCTION:
        Creates node locations and attributes from the high-resolution
        centerline points for each unique Node ID.

    INPUTS
        subcls -- Object containing attributes for the high-resolution centerline.
            [attributes used]:
                node_id -- Node IDs along the high-resolution centerline.
                node_len -- Node lengths along the high-resolution centerline.
                elv -- Elevations along the high-resolution centerline (meters).
                wth -- Widths along the high_resolution centerline (meters).
                nchan -- Number of channels along the high_resolution centerline.
                lon -- Longitude values.
                lat -- Latitude values.
                reach_id -- Reach IDs along the high-resolution centerline.
                grod -- GROD dam locations along the high-resolution centerline.
                grod_fid -- GROD IDs along the high_resolution centerline.
                hfalls_fid -- HydroFALLS IDs along the high_resolution centerline.
                lake_id -- Prior Lake Databse IDs along the high_resolution centerline.
                facc -- Flow accumulation values along the high_resolution centerline.

    OUTPUTS
        Node_ID -- Node ID respresenting a single node location.
        node_x -- Average longitude value calculated from the
            high-resolution centerline points associated with a node.
        node_y -- Average latitude value calculated from the
            high-resolution centerline points associated with a node.
        node_len -- Node length for a single node (meters).
        node_wse -- Average water surface elevation value calculated from the
            high-resolution centerlines points assosicated with a node (meters).
        node_wse_var -- Water surface elevation variablity calculated from the
            high-resolution centerlines points assosicated with a node (meters).
        node_wth -- Average width value calculated from the high-resolution
            centerlines points assosicated with a node ID (meters).
        node_wth_var -- Width variablity calculated from the high-resolution
            centerlines points assosicated with a node (meters).
        node_nchan_max -- Maximum number of channels calculated from
            the high-resolution centerline points associated with a node.
        node_nchan_mod -- Mode of the number of channels calculated from the
            high-resolution centerline points associated with a node.
        node_rch_id --  Reach ID that a particular node belongs to.
        node_grod_id -- GROD dam location associated with a node.
        node_lakeflag -- GRWL lakeflag ID associated with a node.
        node_lake_id = Prior Lake Database ID associated with a node.
        node_facc = Flow Accumulation value associated with a node.
        node_grod_fid = GROD ID associated with a node.
        node_hfalls_fid = HydroFALLS ID associated with a node.
    """

    # Set variables.
    Node_ID = np.zeros(len(np.unique(subcls.node_id)))
    node_x = np.zeros(len(np.unique(subcls.node_id)))
    node_y = np.zeros(len(np.unique(subcls.node_id)))
    node_wse = np.zeros(len(np.unique(subcls.node_id)))
    node_wse_var = np.zeros(len(np.unique(subcls.node_id)))
    node_wth = np.zeros(len(np.unique(subcls.node_id)))
    node_wth_var = np.zeros(len(np.unique(subcls.node_id)))
    node_len = np.zeros(len(np.unique(subcls.node_id)))
    node_nchan_max = np.zeros(len(np.unique(subcls.node_id)))
    node_nchan_mod = np.zeros(len(np.unique(subcls.node_id)))
    node_rch_id = np.zeros(len(np.unique(subcls.node_id)))
    node_grod_id = np.zeros(len(np.unique(subcls.node_id)))
    node_lakeflag = np.zeros(len(np.unique(subcls.node_id)))
    node_lake_id = np.zeros(len(np.unique(subcls.node_id)))
    node_facc = np.zeros(len(np.unique(subcls.node_id)))
    node_grod_fid = np.zeros(len(np.unique(subcls.node_id)))
    node_hfalls_fid = np.zeros(len(np.unique(subcls.node_id)))

    # Loop through and calculate node locations and attributes within a basin.
    # Looping through a basin versus all of the data is computationally faster.
    uniq_basins = np.unique(subcls.basins)
    start = 0
    for ind in list(range(len(uniq_basins))):
        #print(ind)
        basin = np.where(subcls.basins == uniq_basins[ind])[0]
        end = start+len(np.unique(subcls.node_id[basin]))

        if end > len(Node_ID):
            start = start-(end-len(Node_ID))

        v1, v2, v3, v4, v5, v6, v7,\
            v8, v9, v10, v11, \
            v12, v13, v14, v15, \
            v16, v17 = basin_node_attributes(subcls.node_id[basin],
                                                            subcls.node_len[basin],
                                                            subcls.elv[basin], subcls.wth[basin],
                                                            subcls.facc[basin], subcls.nchan[basin],
                                                            subcls.lon[basin], subcls.lat[basin],
                                                            subcls.reach_id[basin], subcls.grod[basin],
                                                            subcls.lake[basin], subcls.grod_fid[basin],
                                                            subcls.hfalls_fid[basin], subcls.lake_id[basin])

        # Append the data from each basin to one vector.
        Node_ID[start:end] = v1
        node_x[start:end] = v2
        node_y[start:end] = v3
        node_len[start:end] = v4
        node_wse[start:end] = v5
        node_wse_var[start:end] = v6
        node_wth[start:end] = v7
        node_wth_var[start:end] = v8
        node_facc[start:end] = v9
        node_nchan_max[start:end] = v10
        node_nchan_mod[start:end] = v11
        node_rch_id[start:end] = v12
        node_grod_id[start:end] = v13
        node_lakeflag[start:end] = v14
        node_grod_fid[start:end] = v15
        node_hfalls_fid[start:end] = v16
        node_lake_id[start:end] = v17

        start = end

    return(Node_ID, node_x, node_y, node_len, node_wse, node_wse_var, node_wth,
           node_wth_var, node_facc, node_nchan_max, node_nchan_mod, node_rch_id,
           node_grod_id, node_lakeflag, node_grod_fid, node_hfalls_fid, node_lake_id)

###############################################################################

def order_neighboring_reaches(end1, end2, reach_id, flowacc, wse, reach):

    """
    FUNCTION:
        Finds upstream and downstream reach IDs for each individual reach. This
        is a sub-function of the "reach_attributes" function.

    INPUTS
        end1 -- Neighboring reaches found for the first reach endpoint.
        end2 -- Neighboring reaches found for the second reach endpoint.
        reach_id -- Reach ID.
        flowacc -- Flow accumulation (km^2).
        reach -- High-resolution centerline points associated with a reach ID.

    OUTPUTS
        rch_id_up -- List of upstream reaches for each reach.
        rch_id_down -- List of downstream reaches for each reach.
    """

    # Set variables.
    rch_id_up = np.zeros(10)
    rch_id_down = np.zeros(10)
    rch_facc = np.max(flowacc[reach])
    rch_elv = np.median(wse[reach])
    rch = np.unique(reach_id[reach])

    # No neighboring reaches.
    if len(end1) == 0 and len(end2) == 0:
        rch_id_up[:] = 0
        rch_id_down[:] = 0

    # Neighboring reaches on one end only.
    elif len(end1) == 0 and len(end2) > 0:
        neighbors_end2 = np.unique(end2[:,0].flatten())
        facc = np.unique(end2[:,3].flatten())
        elv = np.unique(end2[:,4].flatten())
        if np.max(facc) == rch_facc:
            if np.max(elv) == rch_elv:
                ngbs = np.unique(end2[:,0].flatten())
                if np.max(ngbs) < rch:
                    rch_id_down[0:len(neighbors_end2)] = neighbors_end2
                    rch_id_up[:] = 0
                else:
                    rch_id_up[0:len(neighbors_end2)] = neighbors_end2
                    rch_id_down[:] = 0
            elif np.max(elv) < rch_elv:
                rch_id_down[0:len(neighbors_end2)] = neighbors_end2
                rch_id_up[:] = 0
            else:
                rch_id_up[0:len(neighbors_end2)] = neighbors_end2
                rch_id_down[:] = 0

        elif np.max(facc) > rch_facc:
            rch_id_down[0:len(neighbors_end2)] = neighbors_end2
            rch_id_up[:] = 0
        else:
            rch_id_up[0:len(neighbors_end2)] = neighbors_end2
            rch_id_down[:] = 0

    # Neighboring reaches on one end only.
    elif len(end1) > 0 and len(end2) == 0:
        neighbors_end1 = np.unique(end1[:,0].flatten())
        facc = np.unique(end1[:,3].flatten())
        elv = np.unique(end1[:,4].flatten())
        if np.max(facc) == rch_facc:
            if np.max(elv) == rch_elv:
                ngbs = np.unique(end1[:,0].flatten())
                if np.max(ngbs) < rch:
                    rch_id_down[0:len(neighbors_end1)] = neighbors_end1
                    rch_id_up[:] = 0
                else:
                    rch_id_up[0:len(neighbors_end1)] = neighbors_end1
                    rch_id_down[:] = 0
            elif np.max(elv) < rch_elv:
                rch_id_down[0:len(neighbors_end1)] = neighbors_end1
                rch_id_up[:] = 0
            else:
                rch_id_up[0:len(neighbors_end1)] = neighbors_end1
                rch_id_down[:] = 0

        elif np.max(facc) > rch_facc:
            rch_id_down[0:len(neighbors_end1)] = neighbors_end1
            rch_id_up[:] = 0
        else:
            rch_id_up[0:len(neighbors_end1)] = neighbors_end1
            rch_id_down[:] = 0

    # Neighboring reaches on both ends.
    else:
        neighbors_end1 = np.unique(end1[:,0].flatten())
        neighbors_end2 = np.unique(end2[:,0].flatten())
        # Figure out first end.
        facc1 = np.unique(end1[:,3].flatten())
        elv1 = np.unique(end1[:,4].flatten())
        if np.max(facc1) == rch_facc:
            if np.max(elv1) == rch_elv:
                ngbs1 = np.unique(end1[:,0].flatten())
                if np.max(ngbs1) < rch:
                    rch_id_down[0:len(neighbors_end1)] = neighbors_end1
                else:
                    rch_id_up[0:len(neighbors_end1)] = neighbors_end1
            elif np.max(elv1) < rch_elv:
                rch_id_down[0:len(neighbors_end1)] = neighbors_end1
            else:
                rch_id_up[0:len(neighbors_end1)] = neighbors_end1

        elif np.max(facc1) > rch_facc:
            rch_id_down[0:len(neighbors_end1)] = neighbors_end1
        else:
            rch_id_up[0:len(neighbors_end1)] = neighbors_end1
        # Figure out second end.
        up_start_index = np.min(np.where(rch_id_up[:] == 0)[0])
        dn_start_index = np.min(np.where(rch_id_down[:] == 0)[0])
        facc2 = np.unique(end2[:,3].flatten())
        elv2 = np.unique(end2[:,4].flatten())
        if np.max(facc2) == rch_facc:
            if np.max(elv2) == rch_elv:
                ngbs2 = np.unique(end2[:,0].flatten())
                if np.max(ngbs2) < rch:
                    rch_id_down[0:len(neighbors_end2)] = neighbors_end2
                else:
                    rch_id_up[0:len(neighbors_end2)] = neighbors_end2
            elif np.max(elv2) < rch_elv:
                rch_id_down[0:len(neighbors_end2)] = neighbors_end2
            else:
                rch_id_up[0:len(neighbors_end2)] = neighbors_end2

        elif np.max(facc2) > rch_facc:
            rch_id_down[dn_start_index:len(neighbors_end2)+dn_start_index] = neighbors_end2
        else:
            rch_id_up[up_start_index:len(neighbors_end2)+up_start_index] = neighbors_end2

    return rch_id_up, rch_id_down

###############################################################################

def reach_attributes(subcls):

    """
    FUNCTION:
        Creates reach locations and attributes from the high-resolution
        centerline points for each unique Reach ID.

    INPUTS
        rch_eps -- List of indexes for all reach endpoints.
        eps_ind -- Spatial query array containing closest point indexes
            for each reach endpoint.
        eps_dist -- Spatial query array containing closest point distances
            for each reach endpoint.
        subcls -- Object containing attributes for the high-resolution centerline.
            [attributes used]:
                reach_id -- Reach IDs for along the high-resolution centerline.
                rch_len5 -- Reach lengths along the high-resolution centerline.
                elv -- Elevations along the high-resolution centerline (meters).
                wth -- Widths along the high_resolution centerline (meters).
                nchan -- Number of channels along the high_resolution centerline.
                lon -- Longitude values.
                lat -- Latitude values.
                grod -- GROD IDs along the high-resolution centerline.
                node_id -- Node IDs along the high_resolution centerline.
                rch_dist5 -- Flow distance along the high-resolution centerline
                    (meters).
                type5 -- Type flag for each point in the high-resolution
                    centerline (1 = river, 2 = lake, 3 = lake on river,
                    4 = dam, 5 = no topology).
                facc -- Flow accumulation along the high-resolution ceterline
                    (km^2).

    OUTPUTS
        Reach_ID -- Reach ID respresenting a single node location.
        reach_x -- Average longitude value calculated from the
            high-resolution centerline points associated with a reach.
        reach_y -- Average latitude value calculated from the
            high-resolution centerline points associated with a reach.
        reach_len -- Reach length for a single reach (meters).
        reach_wse -- Average water surface elevation value calculated from the
            high-resolution centerlines points assosicated with a reach (meters).
        reach_wse_var -- Water surface elevation variablity calculated from the
            high-resolution centerlines points assosicated with a reach (meters).
        reach_wth -- Average width value calculated from the high-resolution
            centerlines points assosicated with a reach (meters).
        reach_wth_var -- Width variablity calculated from the high-resolution
            centerlines points assosicated with a reach (meters).
        reach_nchan_max -- Maximum number of channels calculated from
            the high-resolution centerline points associated with a reach.
        reach_nchan_mod -- Mode of the number of channels calculated from the
            high-resolution centerline points associated with a reach.
        reach_grod_id -- GROD ID associated with a reach.
        reach_n_nodes -- Number of nodes associated with a reach.
        reach_slope -- Slope calculated from the high_resolution centerline
            points associated with a reach (m/km).
        rch_id_up -- List of reach IDs for the upstream reaches.
        rch_id_down -- List of reach IDs for the downstream reaches.
        n_rch_up -- Number of upstream reaches.
        n_rch_down -- Number of dowstream reaches.
        rch_lakeflag -- GRWL lakeflag ID associated with a reach.
        rch_grod_id -- GROD dam location associated with a reach.
        rch_grod_fid -- GROD ID associated with a reach.
        rch_hfalls_fid -- HydroFALLS ID associated with a reach.
        rch_lake_id -- Prior Lake Database ID associated with a reach.
        reach_facc -- Flow accumulation associated with a reach.
    """

    # Set variables.
    Reach_ID = np.zeros(len(np.unique(subcls.reach_id)))
    reach_x = np.zeros(len(np.unique(subcls.reach_id)))
    reach_y = np.zeros(len(np.unique(subcls.reach_id)))
    reach_x_max = np.zeros(len(np.unique(subcls.reach_id)))
    reach_x_min = np.zeros(len(np.unique(subcls.reach_id)))
    reach_y_max = np.zeros(len(np.unique(subcls.reach_id)))
    reach_y_min = np.zeros(len(np.unique(subcls.reach_id)))
    reach_wse = np.zeros(len(np.unique(subcls.reach_id)))
    reach_wse_var = np.zeros(len(np.unique(subcls.reach_id)))
    reach_wth = np.zeros(len(np.unique(subcls.reach_id)))
    reach_wth_var = np.zeros(len(np.unique(subcls.reach_id)))
    reach_facc = np.zeros(len(np.unique(subcls.reach_id)))
    reach_len = np.zeros(len(np.unique(subcls.reach_id)))
    reach_nchan_max = np.zeros(len(np.unique(subcls.reach_id)))
    reach_nchan_mod = np.zeros(len(np.unique(subcls.reach_id)))
    reach_n_nodes = np.zeros(len(np.unique(subcls.reach_id)))
    reach_slope = np.zeros(len(np.unique(subcls.reach_id)))
    rch_grod_id = np.zeros(len(np.unique(subcls.reach_id)))
    rch_grod_fid = np.zeros(len(np.unique(subcls.reach_id)))
    rch_hfalls_fid = np.zeros(len(np.unique(subcls.reach_id)))
    rch_lakeflag = np.zeros(len(np.unique(subcls.reach_id)))
    rch_lake_id = np.zeros(len(np.unique(subcls.reach_id)))

    # Loop through and calculate reach locations and attributes for each
    # unique reach ID.
    uniq_rch = np.unique(subcls.reach_id)
    for ind in list(range(len(uniq_rch))):
        #print(ind)
        reach = np.where(subcls.reach_id == uniq_rch[ind])[0]
        Reach_ID[ind] = int(np.unique(subcls.reach_id[reach]))
        reach_x[ind] = np.mean(subcls.lon[reach])
        reach_y[ind] = np.mean(subcls.lat[reach])
        reach_x_max[ind] = np.max(subcls.lon[reach])
        reach_x_min[ind] = np.min(subcls.lon[reach])
        reach_y_max[ind] = np.max(subcls.lat[reach])
        reach_y_min[ind] = np.min(subcls.lat[reach])
        reach_len[ind] = np.unique(subcls.rch_len6[reach])
        reach_wse[ind] = np.median(subcls.elv[reach])
        reach_wse_var[ind] = np.var(subcls.elv[reach])
        reach_wth[ind] = np.median(subcls.wth[reach])
        reach_wth_var[ind] = np.var(subcls.wth[reach])
        reach_facc[ind] = np.max(subcls.facc[reach])
        reach_nchan_max[ind] = np.max(subcls.nchan[reach])
        reach_nchan_mod[ind] = max(set(list(subcls.nchan[reach])), key=list(subcls.nchan[reach]).count)
        reach_n_nodes[ind] = len(np.unique(subcls.node_id[reach]))
        rch_lakeflag[ind] = max(set(list(subcls.lake[reach])), key=list(subcls.lake[reach]).count)
        rch_lake_id[ind] = max(set(list(subcls.lake_id[reach])), key=list(subcls.lake_id[reach]).count)

        # Find grod type per reach.
        GROD = np.copy(subcls.grod[reach])
        GROD[np.where(GROD > 4)] = 0
        rch_grod_id[ind] = np.max(GROD)
        # Assign grod and hydrofalls ids to reach.
        ID = np.where(GROD == np.max(GROD))[0][0]
        if np.max(GROD) == 0:
            rch_grod_fid[ind] = 0
        elif np.max(GROD) == 4:
            rch_hfalls_fid[ind] = subcls.hfalls_fid[reach[ID]]
        else:
            rch_grod_fid[ind] = subcls.grod_fid[reach[ID]]

        # Slope calculation.
        slope_pts = np.vstack([subcls.rch_dist6[reach]/1000, np.ones(len(reach))]).T
        slope, intercept = np.linalg.lstsq(slope_pts, subcls.elv[reach], rcond=None)[0]
        reach_slope[ind] = abs(slope) # m/km

    return(Reach_ID, reach_x, reach_y, reach_x_max, reach_x_min, reach_y_max,
           reach_y_min, reach_len, reach_wse, reach_wse_var, reach_wth,
           reach_wth_var, reach_nchan_max, reach_nchan_mod, reach_n_nodes,
           reach_slope, rch_grod_id, rch_lakeflag, reach_facc, rch_grod_fid,
           rch_hfalls_fid, rch_lake_id)

###############################################################################

def filter_neighbors(subreaches):

    """
        FUNCTION:
            Filters reach neighbors to exclude upstream or downstream
            neighbors that do not belong in the associated category. For example,
            if two tributaries are joining together, one may be miscategorized
            as being downstream of the other when it is just a neighboring
            reach with a common downstream neighbor. This function helps
            remove these misclassifications.

        INPUTS
            subreaches -- Object containing attributes for the reache locations.
                [attributes used]:
                    rch_id_down -- List of reach IDs for the downstream reaches.
                    n_rch_down -- Number of downstream reaches.
                    rch_id_up -- List of reach IDs for the upstream reaches.
                    n_rch_up -- Number of downstream reaches.

        OUTPUTS
            copy_rchs_up -- Filtered list of reach IDs for upstream reaches.
            copy_nrch_up -- Filtered number of upstream reaches.
            copy_rchs_dn -- Filtered list of reach IDs for downstream reaches.
            copy_nrch_dn -- Filtered number of downstream reaches.
    """

    #make copies of existing attributes.
    copy_rchs_dn = np.copy(subreaches.rch_id_down)
    copy_nrch_dn = np.copy(subreaches.n_rch_down)
    copy_rchs_up = np.copy(subreaches.rch_id_up)
    copy_nrch_up = np.copy(subreaches.n_rch_up)

    #Filter downstream reach neighbors.
    multi_downstream_tribs = np.where(subreaches.n_rch_down > 1)[0]
    for ind in list(range(len(multi_downstream_tribs))):
        current_rch_ngh_dn = np.unique(copy_rchs_dn[multi_downstream_tribs[ind]])
        current_rch_ngh_dn = np.delete(current_rch_ngh_dn, np.where(current_rch_ngh_dn == 0)[0])
        for idx in list(range(len(current_rch_ngh_dn))):
            ds_rch_ngh = np.unique(copy_rchs_dn[np.where(subreaches.id == current_rch_ngh_dn[idx])[0]])
            ds_rch_ngh = np.delete(ds_rch_ngh, np.where(ds_rch_ngh == 0)[0])
            matches = np.intersect1d(ds_rch_ngh, current_rch_ngh_dn)
            if len(matches) > 0:
                #print(ind)
                rmv_index = np.where(copy_rchs_dn[multi_downstream_tribs[ind]] == current_rch_ngh_dn[idx])[0]
                copy_rchs_dn[multi_downstream_tribs[ind]][rmv_index] = 0

        copy_rchs_dn[multi_downstream_tribs[ind]][::-1].sort()
        copy_nrch_dn[multi_downstream_tribs[ind]] = len(np.where(copy_rchs_dn[multi_downstream_tribs[ind]]>0)[0])

    #Filter upstream reach neighbors.
    multi_upstream_tribs = np.where(subreaches.n_rch_up > 1)[0]
    for ind in list(range(len(multi_upstream_tribs))):
        current_rch_ngh_up = np.unique(copy_rchs_up[multi_upstream_tribs[ind]])
        current_rch_ngh_up = np.delete(current_rch_ngh_up, np.where(current_rch_ngh_up == 0)[0])
        for idx in list(range(len(current_rch_ngh_up))):
            us_rch_ngh = np.unique(copy_rchs_up[np.where(subreaches.id == current_rch_ngh_up[idx])[0]])
            us_rch_ngh = np.delete(us_rch_ngh, np.where(us_rch_ngh == 0)[0])
            matches = np.intersect1d(us_rch_ngh, current_rch_ngh_up)
            if len(matches) > 0:
                #print(ind)
                rmv_index = np.where(copy_rchs_up[multi_upstream_tribs[ind]] == current_rch_ngh_up[idx])[0]
                copy_rchs_up[multi_upstream_tribs[ind]][rmv_index] = 0

        copy_rchs_up[multi_upstream_tribs[ind]][::-1].sort()
        copy_nrch_up[multi_upstream_tribs[ind]] = len(np.where(copy_rchs_up[multi_upstream_tribs[ind]]>0)[0])

    return copy_rchs_up, copy_nrch_up, copy_rchs_dn, copy_nrch_dn

###############################################################################

def neigboring_reaches(subcls, subreaches):

    """
    FUNCTION:
        This function finds all reach neighbors within the larger level 2
        Pfafstetter basin. This output is used in "find_pathways" function.

    INPUTS
        subcls -- Object containing attributes for the high-resolution centerlines.
            [attributes used]:
                lat -- Latitude (wgs84) along the high-resolution centerline.
                lon -- Longitude (wgs84) along the high-reasolution centerline.
                reach_id -- Reach IDs along the high-resolution centerline.
        subreaches -- Object containing attributes for the reach locations.
            [attributes used]:
                id -- Reach IDs per reach.

    OUTPUTS
        neighbors -- List of neighboring reach ids for each reach.
    """

    #reproject lat-lon coordinates to utm values and finding 20 closest points
    #to each point.
    x, y, __, __ = reproject_utm(subcls.lat, subcls.lon)
    all_pts = np.vstack((x, y)).T
    kdt = sp.cKDTree(all_pts)
    pt_dist, pt_ind = kdt.query(all_pts, k = 20, distance_upper_bound = 500)

    #loop through each reach and find the neighboring reaches.
    neighbors = np.zeros((len(subreaches.id),20))
    uniq_rch = np.unique(subcls.reach_id)
    for ind in list(range(len(uniq_rch))):
        #print(ind)
        rch = np.where(subcls.reach_id == uniq_rch[ind])

        id_filter = np.where(pt_ind[rch,:] != len(subcls.x))
        new_ids = pt_ind[rch,:][id_filter]

        all_nghbrs = np.unique(subcls.reach_id[new_ids])
        nghbrs = all_nghbrs[np.where(all_nghbrs != uniq_rch[ind])[0]]

        num_ngh = len(nghbrs)
        rch_loc = np.where(subreaches.id == uniq_rch[ind])[0]
        if num_ngh == 0:
            continue
        else:
            neighbors[rch_loc,0:num_ngh] = nghbrs

    return neighbors

###############################################################################

def find_pathways(subcls, subreaches):

    """
    FUNCTION:
        Defining complete pathways of rivers from the downstream outlet to
        upstream headwaters. It is a crude version of the "marching" method.
        These pathways are used to help calculate the "distance from outlet"
        attribute which is the flow distance from the coastal outlet of a river
        basin to the headwaters.

    INPUTS
        subcls -- Object containing attributes for the high-resolution centerlines.
            [attributes used]:
                lat -- Latitude (wgs84) along the high-resolution centerline.
                lon -- Longitude (wgs84) along the high-resolution centerline.
                reach_id -- Reach IDs along the high-resolution centerline.
                facc -- Flow accumulation along the high-resolution centerline.
        subreaches -- Object containing attributes for the reach locations.
            [attributes used]:
                id -- Reach ID for each reach.
                neighbors -- Reach neighbors for each reach.
                len -- Reach length (m) for each reach.

    OUTPUTS
        pathways -- Unique ID for each pathway along the high-resolution centerline.
        rch_paths -- Unique ID for each pathway for each reach.
        order -- Ordered number per reach within a pathway increasing in the
                 upstream direction.
        path_dist -- Flow distance along a pathway at the reach scale.
    """

    # create empty array to fill and perfor spatial query to find distance
    # to all other points for each individual points.
    x, y, __, __ = reproject_utm(subcls.lat, subcls.lon)
    all_pts = np.vstack((x, y)).T
    kdt = sp.cKDTree(all_pts)
    start_pt = np.where(subcls.facc == np.max(subcls.facc))[0][0]
    start_rch = subreaches.id[np.where(subreaches.id == subcls.reach_id[start_pt])]
    checked = np.zeros(len(subcls.reach_id))
    pathways = np.zeros(len(subcls.reach_id))
    order = np.zeros(len(subreaches.id))
    rch_paths = np.zeros(len(subreaches.id))
    end = np.vstack((x[start_pt], y[start_pt])).T #think about moving this to first condition...
    dist, index = kdt.query(end, k = len(x))
    index = index[0,:]
    dist = dist[0,:]
    loop = 1
    cnt = 1
    paths = 1

    #loop that "martches" up a river from the farthest point downstream defined
    #by flow accumulation. Once the headwaters are reached, the process starts
    #at the next downstream outlet and the "marching" repeats until all rivers
    #have an assigned path.
    while np.min(checked) == 0:
        #print(loop, int(start_rch), paths)
        if loop > len(subreaches.id) + 500:
            print('LOOP STUCK')

        cls_rch = np.where(subcls.reach_id == start_rch)[0]
        rch = np.where(subreaches.id == start_rch)[0]
        all_nghs = subreaches.neighbors[rch,:]
        nghs = all_nghs[np.where(all_nghs > 0)]

        ### need to delete nghs where order[ngh] > 0
        val = np.zeros(len(nghs))
        for ind in list(range(len(nghs))):
            val[ind] = order[np.where(subreaches.id == nghs[ind])]
        delete = np.where(val > 0)[0]
        nghs = np.delete(nghs, delete)

        order[rch] = cnt
        rch_paths[rch] = paths
        checked[cls_rch] = 1
        pathways[cls_rch] = paths

        if len(nghs) == 0:
            cnt = 1
            paths = paths+1
            zero = np.where(checked == 0)[0]
            if len(zero) == 0:
                continue
            else:
                start_pt = zero[np.where(subcls.facc[zero] == np.max(subcls.facc[zero]))[0][0]]
                start_rch = subreaches.id[np.where(subreaches.id == subcls.reach_id[start_pt])]
                end = np.vstack((x[start_pt], y[start_pt])).T #think about moving this to first condition...
                dist, index = kdt.query(end, k = len(x))
                index = index[0,:]
                dist = dist[0,:]
                loop=loop+1
                continue

        elif len(nghs) == 1:
            cnt = cnt+1
            cls_rch2 = np.where(subcls.reach_id == nghs)[0]
            start_pt = cls_rch2[np.where(subcls.facc[cls_rch2] == np.max(subcls.facc[cls_rch2]))[0][0]]
            start_rch = nghs
            loop=loop+1
            continue

        else:
            cnt = cnt+1
            ngh_dist = np.zeros(len(nghs))
            for idy in list(range(len(nghs))):
                rch2 = np.where(subcls.reach_id[index] == nghs[idy])[0]
                ngh_dist[idy] = np.max(dist[rch2])

            nxt = np.where(ngh_dist == np.max(ngh_dist))[0][0]
            cls_rch2 = np.where(subcls.reach_id == nghs[nxt])[0]
            start_pt = cls_rch2[np.where(subcls.facc[cls_rch2] == np.max(subcls.facc[cls_rch2]))[0][0]]
            start_rch = nghs[nxt]
            loop=loop+1
            continue

    #calculate path distance.
    path_dist = np.zeros(len(subreaches.id))
    for idx in list(range(len(rch_paths))):
        path = np.where(rch_paths == rch_paths[idx])[0]
        order_ids = np.argsort(order[path])
        d = np.cumsum(subreaches.len[path[order_ids]])
        path_dist[path[order_ids]] = d

    return pathways, rch_paths, order, path_dist

###############################################################################

def calc_basin_dist(subreaches):

    """
    FUNCTION:
        Calculates the distance from the river outlet for each reach.

    INPUTS
        subreaches -- Object containing attributes for the reach locations.
            [attributes used]:
                id -- Reach ID for each reach.
                path_dist -- Flow distance along a pathway at the reach scale.
                path -- Unique ID for each pathway for each reach.
                path_order -- Ordered number per reach within a pathway
                              increasing in the upstream direction.
                facc --Flow accumulation for each reach.
                neighbors -- Reach neighbors for each reach.
                len -- Reach length (m).

    OUTPUTS
        rch_dist_out -- Distance from outlet (m) for each reach.
    """

    rch_dist_out = np.copy(subreaches.path_dist)
    start_rch = subreaches.id[np.where(subreaches.facc == np.max(subreaches.facc))[0]][0]
    checked = np.zeros(len(subreaches.id))
    loop = 1

    while np.min(checked) < 1:

        #print(loop, int(start_rch))
        if loop > len(subreaches.id) + 500:
            print('LOOP STUCK')

        rch = np.where(subreaches.id == start_rch)[0]

        #identifying what path the current reach is on, & the path end-reaches.
        rch_path = subreaches.path[rch]
        #if rch_path == 104:
            #break
        all_rch_path_pts = np.where(subreaches.path == rch_path)[0]
        path_min = all_rch_path_pts[np.where(subreaches.path_order[all_rch_path_pts] == np.min(subreaches.path_order[all_rch_path_pts]))]
        path_max = all_rch_path_pts[np.where(subreaches.path_order[all_rch_path_pts] == np.max(subreaches.path_order[all_rch_path_pts]))]

        #finding all neighboring paths to current path.
        all_path_nghs_list = []
        for a in list(range(len(all_rch_path_pts))):
            rch2 = np.where(subreaches.id == subreaches.id[all_rch_path_pts[a]])[0]
            all_nghs = subreaches.neighbors[rch2,:]
            all_path_nghs_list.append(all_nghs[np.where(all_nghs > 0)])
        all_path_nghs = np.unique(np.array([x for sublist in all_path_nghs_list for x in sublist]))

        adjacent_paths = np.zeros(len(all_path_nghs))
        for b in list(range(len(all_path_nghs))):
            rch3 = np.where(subreaches.id == all_path_nghs[b])[0]
            adjacent_paths[b] = subreaches.path[rch3]
        adj_paths = np.unique(adjacent_paths[np.where(adjacent_paths != rch_path)])

        if len(adj_paths) == 0:
            checked[all_rch_path_pts] = 1
            z = np.where(checked < 1)[0]
            if len(z) > 0:
                loop = loop+1
                start_rch = subreaches.id[z[np.where(subreaches.path[z] == np.min(subreaches.path[z]))[0]]][0]
            else:
                loop = loop+1
                continue

        else:
            adj_path_checks = np.zeros(len(adj_paths))
            for ind in list(range(len(adj_paths))):
                val = np.max(checked[np.where(subreaches.path == int(adj_paths[ind]))])
                adj_path_checks[ind] = val

            yes_checks = adj_paths[np.where(adj_path_checks > 0)]
            #no_checks = adj_paths[np.where(adj_path_checks == 0)]

            if len(yes_checks) == 0:

                min_nghs = subreaches.neighbors[path_min,:]
                min_nghs = min_nghs[np.where(min_nghs > 0)]
                max_nghs = subreaches.neighbors[path_max,:]
                max_nghs = max_nghs[np.where(max_nghs > 0)]
                min_max_ngh_rchs = np.insert(min_nghs, len(min_nghs), max_nghs)
                # find paths for all endpoint-path neighbors.
                min_max_ngh_paths = np.zeros(len(min_max_ngh_rchs))
                for m in list(range(len(min_max_ngh_rchs))):
                    min_max_ngh_paths[m] = subreaches.path[np.where(subreaches.id == min_max_ngh_rchs[m])]
                # filter out reaches that are in the same path.
                keep = np.where(min_max_ngh_paths != rch_path)[0]
                min_max_ngh_paths = min_max_ngh_paths[keep]
                min_max_ngh_rchs = min_max_ngh_rchs[keep]

                # if there are no upstream or downstream path neighbors.
                if len(min_max_ngh_paths) == 0:
                    checked[all_rch_path_pts] = 1
                    z = np.where(checked < 1)[0]
                    if len(z) > 0:
                        loop = loop+1
                        start_rch = subreaches.id[z[np.where(subreaches.path[z] == np.min(subreaches.path[z]))[0]]][0]
                    else:
                        loop = loop+1
                        continue
                # if there are neighbors.
                else:
                    loop = loop+1
                    if np.min(checked[all_rch_path_pts]) == 0:
                        checked[all_rch_path_pts] = -1 #ideally would not have this here becuase it could be an upstream reach, but oh well...
                    else:
                        checked[all_rch_path_pts] = 1

                    checks = np.zeros(len(min_max_ngh_paths))
                    for c1 in list(range(len(min_max_ngh_paths))):
                        checks[c1] = np.max(checked[np.where(subreaches.path == min_max_ngh_paths[c1])])
                    min_max_ngh_paths = np.unique(min_max_ngh_paths[np.where(checks == 0)])

                    if len(min_max_ngh_paths) == 0: #may need to adjust....
                        z = np.where(checked == 0)[0]
                        if len(z) > 0:
                            loop = loop+1
                            start_rch = subreaches.id[z[np.where(subreaches.path[z] == np.min(subreaches.path[z]))[0]]][0]
                        else:
                            z2 = np.where(checked < 0)[0]
                            if len(z2) > 0:
                                loop = loop+1
                                start_rch = subreaches.id[z2[np.where(subreaches.path[z2] == np.min(subreaches.path[z2]))[0]]][0]
                            else:
                                loop = loop+1
                                continue

                    elif len(min_max_ngh_paths) > 1:
                        min_ngh_path_facc = np.zeros(len(min_max_ngh_paths))
                        for facc in list(range(len(min_max_ngh_paths))):
                            min_ngh_path_facc[facc] = np.max(subreaches.facc[np.where(subreaches.path == min_max_ngh_paths[facc])])
                        facc_max = np.where(min_ngh_path_facc == np.max(min_ngh_path_facc))[0][0]
                        ngh_path_pts = np.where(subreaches.path == min_max_ngh_paths[facc_max])[0]
                        nxt_pt = ngh_path_pts[np.where(subreaches.path_order[ngh_path_pts] == np.min(subreaches.path_order[ngh_path_pts]))]
                        start_rch = subreaches.id[nxt_pt]
                    else:
                        ngh_path_pts = np.where(subreaches.path == min_max_ngh_paths[0])[0]
                        nxt_pt = ngh_path_pts[np.where(subreaches.path_order[ngh_path_pts] == np.min(subreaches.path_order[ngh_path_pts]))]
                        start_rch = subreaches.id[nxt_pt]

            else:
                # find all neighbors at the current path's end-reaches.
                min_nghs = subreaches.neighbors[path_min,:]
                min_nghs = min_nghs[np.where(min_nghs > 0)]
                max_nghs = subreaches.neighbors[path_max,:]
                max_nghs = max_nghs[np.where(max_nghs > 0)]
                min_max_ngh_rchs = np.insert(min_nghs, len(min_nghs), max_nghs)
                # find paths for all endpoint-path neighbors.
                min_max_ngh_paths = np.zeros(len(min_max_ngh_rchs))
                for m in list(range(len(min_max_ngh_rchs))):
                    min_max_ngh_paths[m] = subreaches.path[np.where(subreaches.id == min_max_ngh_rchs[m])]
                # filter out reaches that are in the same path.
                keep = np.where(min_max_ngh_paths != rch_path)[0]
                min_max_ngh_paths = min_max_ngh_paths[keep]
                min_max_ngh_rchs = min_max_ngh_rchs[keep]

                # check to see if any neighboring reaches/paths are already checked.
                checks = np.zeros(len(min_max_ngh_paths))
                for c2 in list(range(len(min_max_ngh_paths))):
                    if min_max_ngh_paths[c2] in yes_checks:
                        checks[c2] = 1
                    else:
                        checks[c2] = 0

                checked_rchs = min_max_ngh_rchs[np.where(checks > 0)]
                unchecked_rchs = min_max_ngh_rchs[np.where(checks == 0)]

                if len(checked_rchs) == 0:
                    checked[all_rch_path_pts] = 1
                    z = np.where(checked < 1)[0]
                    if len(z) > 0:
                        loop = loop+1
                        start_rch = subreaches.id[z[np.where(subreaches.path[z] == np.min(subreaches.path[z]))[0]]][0]
                    else:
                        loop = loop+1
                        continue
                elif len(checked_rchs) == 1:
                    #path_id = np.where(min_ngh_path == yes_checks[np.where(min_checks > 0)])[0]
                    ngh_rch_pts = np.where(subreaches.id == checked_rchs[0])[0]
                    d = rch_dist_out[ngh_rch_pts]
                    d_len = subreaches.len[ngh_rch_pts]/2 #trying not to have weird transitions at multichannel junctions.
                    d_final = d-d_len
                    rch_vals = subreaches.path_dist[all_rch_path_pts] + d_final
                    rch_dist_out[all_rch_path_pts] = rch_vals
                    #assign next reach.
                    if len(unchecked_rchs) > 0:
                        loop=loop+1
                        checked[all_rch_path_pts] = 1
                        start_rch = unchecked_rchs[0]
                    else:
                        checked[all_rch_path_pts] = 1
                        z = np.where(checked < 1)[0]
                        if len(z) > 0:
                            loop = loop+1
                            start_rch = subreaches.id[z[np.where(subreaches.path[z] == np.min(subreaches.path[z]))[0]]][0]
                        else:
                            loop = loop+1
                            continue
                else:
                    #ds_paths = yes_checks[np.where(min_checks > 0)]
                    d = np.zeros(len(checked_rchs))
                    for o in list(range(len(checked_rchs))):
                        #path_id = np.where(min_ngh_path == ds_paths[o])[0]
                        ngh_rch_pts = np.where(subreaches.id == checked_rchs[o])[0]
                        d[o] = rch_dist_out[ngh_rch_pts]
                    #choose maximum distance below
                    d_max = np.max(d)
                    d_len = subreaches.len[ngh_rch_pts]/2 #trying not to have weird transitions at multichannel junctions.
                    d_final = d_max-d_len
                    rch_vals = subreaches.path_dist[all_rch_path_pts] + d_final
                    rch_vals = subreaches.path_dist[all_rch_path_pts] + d_max
                    rch_dist_out[all_rch_path_pts] = rch_vals
                    #assign next reach.
                    if len(unchecked_rchs) > 0:
                        loop=loop+1
                        checked[all_rch_path_pts] = 1
                        start_rch = unchecked_rchs[0]
                    else:
                        checked[all_rch_path_pts] = 1
                        z = np.where(checked < 1)[0]
                        if len(z) > 0:
                            loop = loop+1
                            start_rch = subreaches.id[z[np.where(subreaches.path[z] == np.min(subreaches.path[z]))[0]]][0]
                        else:
                            loop = loop+1
                            continue

    return rch_dist_out

###############################################################################

def filter_basin_dist(subreaches):

    """
    FUNCTION:
        Filters the intial distance from outlet to remove outliers.

    INPUTS
        subreaches -- Object containing attributes for the reach locations.
            [attributes used]:
                id -- Reach ID for each reach.
                path_dist -- Flow distance along a pathway at the reach scale.
                path -- Unique ID for each pathway for each reach.
                path_order -- Ordered number per reach within a pathway
                              increasing in the upstream direction.
                neighbors -- Reach neighbors for each reach.
                dist_out_prior -- Inital distance from outlet for each reach.

    OUTPUTS
        new_rch_dist_out -- Filtered distance from outlet (m) for each reach.
    """

    new_rch_dist_out = np.copy(subreaches.dist_out_prior)
    start_rch = subreaches.id[np.where(subreaches.dist_out_prior == np.min(subreaches.dist_out_prior))[0]][0]
    #updated = np.zeros(len(subreaches.id))
    checked = np.zeros(len(subreaches.id))
    loop = 1

    while np.min(checked) < 1:

        #print(loop, int(start_rch))
        if loop > len(subreaches.id) + 500:
            print('LOOP STUCK')

        rch = np.where(subreaches.id == start_rch)[0]

        #identifying what path the current reach is on, & the path end-reaches.
        rch_path = subreaches.path[rch]
        #if rch_path == 1211:
            #break
        all_rch_path_pts = np.where(subreaches.path == rch_path)[0]
        path_min = all_rch_path_pts[np.where(subreaches.path_order[all_rch_path_pts] == np.min(subreaches.path_order[all_rch_path_pts]))]
        path_max = all_rch_path_pts[np.where(subreaches.path_order[all_rch_path_pts] == np.max(subreaches.path_order[all_rch_path_pts]))]
        current_dist = np.min(new_rch_dist_out[all_rch_path_pts])

        #finding all neighboring paths to current path.
        all_path_nghs_list = []
        for a in list(range(len(all_rch_path_pts))):
            rch2 = np.where(subreaches.id == subreaches.id[all_rch_path_pts[a]])[0]
            all_nghs = subreaches.neighbors[rch2,:]
            all_path_nghs_list.append(all_nghs[np.where(all_nghs > 0)])
        all_path_nghs = np.unique(np.array([x for sublist in all_path_nghs_list for x in sublist]))

        adjacent_paths = np.zeros(len(all_path_nghs))
        for b in list(range(len(all_path_nghs))):
            rch3 = np.where(subreaches.id == all_path_nghs[b])[0]
            adjacent_paths[b] = subreaches.path[rch3]
        adj_paths = np.unique(adjacent_paths[np.where(adjacent_paths != rch_path)])

        if len(adj_paths) == 0:
            checked[all_rch_path_pts] = 1
            z = np.where(checked < 1)[0]
            if len(z) > 0:
                loop = loop+1
                start_rch = subreaches.id[z[np.where(new_rch_dist_out[z] == np.min(new_rch_dist_out[z]))[0]]][0]
            else:
                loop = loop+1
                continue

        else:
            adj_path_dist = np.zeros(len(adj_paths))
            for ind in list(range(len(adj_paths))):
                val = np.max(new_rch_dist_out[np.where(subreaches.path == int(adj_paths[ind]))]) #change to new_dist_out...
                adj_path_dist[ind] = val
            # find all neighbors at the ends of the paths.
            min_nghs = subreaches.neighbors[path_min,:]
            min_nghs = min_nghs[np.where(min_nghs > 0)]
            max_nghs = subreaches.neighbors[path_max,:]
            max_nghs = max_nghs[np.where(max_nghs > 0)]
            min_max_ngh_rchs = np.insert(min_nghs, len(min_nghs), max_nghs)
            # find paths for all endpoint-path neighbors.
            min_max_ngh_paths = np.zeros(len(min_max_ngh_rchs))
            for m in list(range(len(min_max_ngh_rchs))):
                min_max_ngh_paths[m] = subreaches.path[np.where(subreaches.id == min_max_ngh_rchs[m])]
            # filter out reaches that are in the same path.
            keep = np.where(min_max_ngh_paths != rch_path)[0]
            min_max_ngh_paths = min_max_ngh_paths[keep]
            min_max_ngh_rchs = min_max_ngh_rchs[keep]

            if len(min_max_ngh_rchs) == 0: #### need to think about this more...

                adj_path_checks = np.zeros(len(adj_paths))
                for idz in list(range(len(adj_paths))):
                    val0 = np.max(checked[np.where(subreaches.path == int(adj_paths[idz]))])
                    adj_path_checks[idz] = val0

                #yes_checks = np.where(adj_path_checks > 0)[0]
                no_checks = np.where(adj_path_checks == 0)[0]

                if len(no_checks) == 0:
                    checked[all_rch_path_pts] = 1
                    z = np.where(checked < 1)[0]
                    if len(z) > 0:
                        loop = loop+1
                        start_rch = subreaches.id[z[np.where(new_rch_dist_out[z] == np.min(new_rch_dist_out[z]))[0]]][0]
                    else:
                        loop = loop+1
                        continue
                else:
                    checked[all_rch_path_pts] = 1
                    unchecked_paths = adj_paths[no_checks]
                    unchecked_path_dist = adj_path_dist[no_checks]
                    nxt_path = unchecked_paths[np.where(unchecked_path_dist == np.min(np.sort(unchecked_path_dist)))][0]
                    ngh_path_pts = np.where(subreaches.path == nxt_path)[0]
                    nxt_pt = ngh_path_pts[np.where(subreaches.path_order[ngh_path_pts] == np.min(subreaches.path_order[ngh_path_pts]))]
                    start_rch = subreaches.id[nxt_pt]
                    loop = loop+1

            else:
                ngh_path_dist = np.zeros(len(min_max_ngh_rchs))
                for idx in list(range(len(min_max_ngh_rchs))):
                    val2 = new_rch_dist_out[np.where(subreaches.id == int(min_max_ngh_rchs[idx]))] #change to new_dist_out...
                    ngh_path_dist[idx] = val2

                ngh_dist_max = np.max(ngh_path_dist)
                thresh = current_dist+(50000)

                if ngh_dist_max > thresh:
                    # update path distances.
                    #updated[all_rch_path_pts] = 1
                    d = ngh_dist_max
                    rch_vals = subreaches.path_dist[all_rch_path_pts] + d
                    new_rch_dist_out[all_rch_path_pts] = rch_vals

                    # find out if any adjacent paths are flagged....
                    remaining_paths = adj_paths[np.where(adj_paths != min_max_ngh_paths[np.where(ngh_path_dist == ngh_dist_max)])]
                    remaining_path_dist = adj_path_dist[np.where(adj_paths != min_max_ngh_paths[np.where(ngh_path_dist == ngh_dist_max)])]

                    remaining_path_checks = np.zeros(len(remaining_paths))
                    for idy in list(range(len(remaining_paths))):
                        val3 = np.max(checked[np.where(subreaches.path == int(remaining_paths[idy]))])
                        remaining_path_checks[idy] = val3

                    #yes_checks = np.where(remaining_path_checks > 0)[0]
                    no_checks = np.where(remaining_path_checks == 0)[0]

                    if len(no_checks) == 0:
                        checked[all_rch_path_pts] = 1
                        z = np.where(checked < 1)[0]
                        if len(z) > 0:
                            loop = loop+1
                            start_rch = subreaches.id[z[np.where(new_rch_dist_out[z] == np.min(new_rch_dist_out[z]))[0]]][0]
                        else:
                            loop = loop+1
                            continue
                    else:
                        checked[all_rch_path_pts] = 1
                        unchecked_paths = remaining_paths[no_checks]
                        unchecked_path_dist = remaining_path_dist[no_checks]
                        nxt_path = unchecked_paths[np.where(unchecked_path_dist == np.min(np.sort(unchecked_path_dist)))][0]
                        ngh_path_pts = np.where(subreaches.path == nxt_path)[0]
                        nxt_pt = ngh_path_pts[np.where(subreaches.path_order[ngh_path_pts] == np.min(subreaches.path_order[ngh_path_pts]))]
                        start_rch = subreaches.id[nxt_pt]
                        loop = loop+1

                else:
                    checked[all_rch_path_pts] = 1
                    z = np.where(checked < 1)[0]
                    if len(z) > 0:
                        loop = loop+1
                        start_rch = subreaches.id[z[np.where(new_rch_dist_out[z] == np.min(new_rch_dist_out[z]))[0]]][0]
                    else:
                        loop = loop+1
                        continue

    return new_rch_dist_out

###############################################################################

def calc_node_dist_out(subreaches, subnodes):

    """
    FUNCTION:
        Calculates the distance from the river outlet for each node.

    INPUTS
        subreaches -- Object containing attributes for the reach locations.
            [attributes used]:
                id -- Reach ID for each reach.
                len -- Reach length (m).
                dist_out -- Distance from outlet for each reach (m).
        subnodes -- Object containing attributes for the node locations.
            [attributes used]:
                id -- Node ID for each node.
                reach_id -- Reach ID for each node.
                len -- Node Length (m).

    OUTPUTS
        node_dist_out -- Distance from outlet (m) for each node.
    """

    node_dist_out = np.unique(subnodes.id)
    uniq_rchs = np.unique(subreaches.id)
    for ind in list(range(len(uniq_rchs))):
        rch = np.where(subreaches.id == uniq_rchs[ind])[0]
        node_locs = np.where(subnodes.reach_id == uniq_rchs[ind])[0]
        node_order = np.argsort(subnodes.id[node_locs])
        node_cdist = np.cumsum(subnodes.len[node_locs[node_order]])


        rch_len = subreaches.len[rch]
        rch_dist_out = subreaches.dist_out[rch]
        d = rch_dist_out - rch_len

        node_vals = node_cdist+d
        node_dist_out[node_locs[node_order]] = node_vals

    return node_dist_out

###############################################################################

def local_topology(subcls, subreaches):

    """
    FUNCTION:
        Identifies the upstream and downstream reach neighbors, and number of
        neighbors for each reach.

    INPUTS
        subreaches -- Object containing attributes for the reach locations.
            [attributes used]:
                id -- Reach ID for each reach.
                dist_out -- Distance from outlet for each reach (m).
                neighbors -- All neighbors, without direction, for each reach.
                wse -- Water surface elevation (m) for each reach.
                facc -- Flow accumulatin (km^2) for each reach.
        subcls -- Object containing attributes for the node locations.
            [attributes used]:
                reach_id -- Reach ID along the high-resolution centerline.
                rch_ind6 -- Point indexes after aggregating short reaches and
                            cutting long reaches.
                lon -- Latitude (wgs84).
                lat -- Logitude (wgs84).

    OUTPUTS
        n_rch_up -- Number of upstream reaches per reach.
        n_rch_down -- Number of downstream reaches per reach.
        rch_id_up -- Reach ids of the upstream reach neighbors per reach.
        rch_id_down -- Reach ids of the downstream reach neighbors per reach.
    """

    n_rch_up = np.zeros(len(np.unique(subcls.reach_id)))
    n_rch_down = np.zeros(len(np.unique(subcls.reach_id)))
    rch_id_up = np.zeros((len(np.unique(subcls.reach_id)),10))
    rch_id_down = np.zeros((len(np.unique(subcls.reach_id)),10))
    #flagged = np.zeros(len(np.unique(subcls.reach_id)))

    for ind in list(range(len(subreaches.id))):
        #print(ind)
        #identifying the current reach neighbors.
        nghs = subreaches.neighbors[ind,np.where(subreaches.neighbors[ind,:] > 0)[0]]

        if len(nghs) == 0:
            continue

        else:
            #determining important reach attributes.
            rch_cls = np.where(subcls.reach_id == subreaches.id[ind])[0]
            rch = np.where(subreaches.id == subreaches.id[ind])[0]
            rch_dist = subreaches.dist_out[rch]
            rch_wse = subreaches.wse[rch]
            rch_facc = subreaches.facc[rch]

            #identifying reach endpoints based on index values.
            end1 = rch_cls[np.where(subcls.rch_ind6[rch_cls] == np.min(subcls.rch_ind6[rch_cls]))[0][0]]
            end2 = rch_cls[np.where(subcls.rch_ind6[rch_cls] == np.max(subcls.rch_ind6[rch_cls]))[0][0]]

            #finding the coordination of the reach endpoints.
            ep1_lon, ep1_lat = subcls.lon[end1], subcls.lat[end1]
            ep2_lon, ep2_lat = subcls.lon[end2], subcls.lat[end2]

            #looping through each reach neighbor and determining if it is
            #upstream or downstream based on flow accumulation or wse.
            ep1_nghs = np.zeros([len(nghs), 4])
            ep2_nghs = np.zeros([len(nghs), 4])
            for idx in list(range(len(nghs))):
                rch2 = np.where(subreaches.id == nghs[idx])[0]
                rch2_cls = np.where(subcls.reach_id == nghs[idx])[0]
                ngh_dist = subreaches.dist_out[rch2]
                ngh_wse = subreaches.wse[rch2]
                ngh_facc = subreaches.facc[rch2]

                # finding high-res centerline endpoints for neighboring reach.
                ngh_end1 = rch2_cls[np.where(subcls.rch_ind6[rch2_cls] == np.min(subcls.rch_ind6[rch2_cls]))[0][0]]
                ngh_end2 = rch2_cls[np.where(subcls.rch_ind6[rch2_cls] == np.max(subcls.rch_ind6[rch2_cls]))[0][0]]
                ngh_end1_lon, ngh_end1_lat = subcls.lon[ngh_end1], subcls.lat[ngh_end1]
                ngh_end2_lon, ngh_end2_lat = subcls.lon[ngh_end2], subcls.lat[ngh_end2]

                # reprojecting all current reach and neighboring reach coordinates to utm.
                all_lon = np.array([ep1_lon, ep2_lon, ngh_end1_lon, ngh_end2_lon])
                all_lat = np.array([ep1_lat, ep2_lat, ngh_end1_lat, ngh_end2_lat])
                all_x, all_y, __, __ = reproject_utm(all_lat, all_lon)

                # finding distance between neighboring endpoints and current reach.
                ep1_dist_ngh1 = np.sqrt((all_x[2]-all_x[0])**2 + (all_y[2]-all_y[0])**2)
                ep1_dist_ngh2 = np.sqrt((all_x[3]-all_x[0])**2 + (all_y[3]-all_y[0])**2)
                ep2_dist_ngh1 = np.sqrt((all_x[2]-all_x[1])**2 + (all_y[2]-all_y[1])**2)
                ep2_dist_ngh2 = np.sqrt((all_x[3]-all_x[1])**2 + (all_y[3]-all_y[1])**2)
                ep1_dist = np.min([ep1_dist_ngh1, ep1_dist_ngh2])
                ep2_dist = np.min([ep2_dist_ngh1, ep2_dist_ngh2])

                if ep1_dist <= ep2_dist: #if equal distance then assign to end1 for now...
                    fill_row = np.min(np.where(ep1_nghs[:,0] == 0)[0])
                    ep1_nghs[fill_row,0] = nghs[idx]
                    ep1_nghs[fill_row,1] = ngh_facc
                    ep1_nghs[fill_row,2] = ngh_wse
                    ep1_nghs[fill_row,3] = ngh_dist
                else:
                    fill_row = np.min(np.where(ep2_nghs[:,0] == 0)[0])
                    ep2_nghs[fill_row,0] = nghs[idx]
                    ep2_nghs[fill_row,1] = ngh_facc
                    ep2_nghs[fill_row,2] = ngh_wse
                    ep2_nghs[fill_row,3] = ngh_dist

            ep1_delete = np.where(ep1_nghs[:,0] == 0)[0]
            ep2_delete = np.where(ep2_nghs[:,0] == 0)[0]
            ep1_nghs = np.delete(ep1_nghs, ep1_delete, axis = 0)
            ep2_nghs = np.delete(ep2_nghs, ep2_delete, axis = 0)

            if len(ep1_nghs) > 0:
                min_dist = np.min(ep1_nghs[:,3])
                min_wse = np.min(ep1_nghs[:,2])
                max_facc = np.max(ep1_nghs[:,1])

                if min_dist < rch_dist:
                    ## assign to downstream end
                    start_col = np.min(np.where(rch_id_down[ind,:] == 0)[0])
                    end_col = start_col + len(ep1_nghs)
                    rch_id_down[ind,start_col:end_col] = ep1_nghs[:,0]
                elif min_dist > rch_dist:
                    ### assign to upstream end
                    start_col = np.min(np.where(rch_id_up[ind,:] == 0)[0])
                    end_col = start_col + len(ep1_nghs)
                    rch_id_up[ind,start_col:end_col] = ep1_nghs[:,0]
                else:
                    if max_facc > rch_facc:
                        ### assign to downstream end
                        start_col = np.min(np.where(rch_id_down[ind,:] == 0)[0])
                        end_col = start_col + len(ep1_nghs)
                        rch_id_down[ind,start_col:end_col] = ep1_nghs[:,0]
                    elif max_facc < rch_facc:
                        ### assign to upstream end
                        start_col = np.min(np.where(rch_id_up[ind,:] == 0)[0])
                        end_col = start_col + len(ep1_nghs)
                        rch_id_up[ind,start_col:end_col] = ep1_nghs[:,0]
                    else:
                        if min_wse > rch_wse:
                            ### assign to upstream end
                            start_col = np.min(np.where(rch_id_up[ind,:] == 0)[0])
                            end_col = start_col + len(ep1_nghs)
                            rch_id_up[ind,start_col:end_col] = ep1_nghs[:,0]
                        else:
                            ### assign to downstream end
                            start_col = np.min(np.where(rch_id_down[ind,:] == 0)[0])
                            end_col = start_col + len(ep1_nghs)
                            rch_id_down[ind,start_col:end_col] = ep1_nghs[:,0]

            if len(ep2_nghs) > 0:
                min_dist = np.min(ep2_nghs[:,3])
                min_wse = np.min(ep2_nghs[:,2])
                max_facc = np.max(ep2_nghs[:,1])

                if min_dist < rch_dist:
                    ## assign to downstream end
                    start_col = np.min(np.where(rch_id_down[ind,:] == 0)[0])
                    end_col = start_col + len(ep2_nghs)
                    rch_id_down[ind,start_col:end_col] = ep2_nghs[:,0]
                elif min_dist > rch_dist:
                    ### assign to upstream end
                    start_col = np.min(np.where(rch_id_up[ind,:] == 0)[0])
                    end_col = start_col + len(ep2_nghs)
                    rch_id_up[ind,start_col:end_col] = ep2_nghs[:,0]
                else:
                    if max_facc > rch_facc:
                        ### assign to downstream end
                        start_col = np.min(np.where(rch_id_down[ind,:] == 0)[0])
                        end_col = start_col + len(ep2_nghs)
                        rch_id_down[ind,start_col:end_col] = ep2_nghs[:,0]
                    elif max_facc < rch_facc:
                        ### assign to upstream end
                        start_col = np.min(np.where(rch_id_up[ind,:] == 0)[0])
                        end_col = start_col + len(ep2_nghs)
                        rch_id_up[ind,start_col:end_col] = ep2_nghs[:,0]
                    else:
                        if min_wse > rch_wse:
                            ### assign to upstream end
                            start_col = np.min(np.where(rch_id_up[ind,:] == 0)[0])
                            end_col = start_col + len(ep2_nghs)
                            rch_id_up[ind,start_col:end_col] = ep2_nghs[:,0]
                        else:
                            ### assign to downstream end
                            start_col = np.min(np.where(rch_id_down[ind,:] == 0)[0])
                            end_col = start_col + len(ep2_nghs)
                            rch_id_down[ind,start_col:end_col] = ep2_nghs[:,0]

        ### count number of upstream and downstream neighbors.
        n_rch_down[ind] = len(np.where(rch_id_down[ind,:] > 0)[0])
        n_rch_up[ind] = len(np.where(rch_id_up[ind,:] > 0)[0])

    return n_rch_up, n_rch_down, rch_id_up, rch_id_down

###############################################################################

def ghost_reaches(subcls):

    """
    FUNCTION:
        Determines "ghost" reaches and nodes which are located at the headwaters
        and outlets of all river systems.

    INPUTS
        subcls -- Object containing attributes for the high-resolution centerline.
            [attributes used]:
                reach_id -- Reach IDs along the high-resolution centerline.
                lon -- Longitude (wgs84) for the high-resolution centerline points.
                lat -- Latitude (wgs84) for the high-resolution centerline points.
                node_id -- Node IDs along the high-resolution centerline.
                node_num -- 1-D array of ordered node values within a basin.
                basin -- Pfafstetter level 6 basin codes along the
                         high-resolution centerline.
                rch_topo -- 1-D array of ordered reach values within a basin.
                rch_dist5 -- Flow distance along the high-resolution centerline
                             (meters).
                rch_len5 -- Reach length along the high-resolution centerline.
                rch_ind5 -- Point indexes for each reach along the
                    high-resolution centerline.
                rch_eps5 -- List of indexes for all reach endpoints.

    OUTPUTS
        new_reaches -- Updated reach definitions and IDs including ghost reaches.
        new_nodes -- Updated node definitions and IDs including ghost nodes.
        new_len -- Updated reach lengths including ghost reaches.
    """

    # Pre-defining all reach endpoint neighbors.
    x, y, __, __ = reproject_utm(subcls.lat, subcls.lon)
    rch_eps = np.where(subcls.rch_eps5 == 1)[0]
    all_pts = np.vstack((x, y)).T
    eps_pts = np.vstack((x[rch_eps], y[rch_eps])).T
    kdt = sp.cKDTree(all_pts)
    eps_dist, eps_ind = kdt.query(eps_pts, k = 10, distance_upper_bound = 500)
    #actual ghost node identification.
    ghost_dist = np.copy(eps_dist)
    ghost_dist[np.where(ghost_dist[:,1] == 0),2] = ghost_dist[np.where(ghost_dist[:,1] == 0),3]
    ghost_ids = eps_ind[np.where(ghost_dist[:,2] >= 60)[0],0]
    #added to attempt to filter out unnecessary ghost nodes.
    ghost_pts = np.vstack((x[ghost_ids], y[ghost_ids])).T
    gst_dist, gst_ind = kdt.query(ghost_pts, k = 5, distance_upper_bound = 500)
    gst_dist[np.where(gst_dist[:,1] == 0),1] = gst_dist[np.where(gst_dist[:,1] == 0),2]
    rmv_ids = gst_ind[np.where(gst_dist[:,4] < 100)[0]]
    #creating binary ghost node array.
    subcls.ghost = np.zeros(len(subcls.id))
    subcls.ghost[ghost_ids] = 1
    #removes unnecessary ghost nodes from original list.
    subcls.ghost[rmv_ids] = 0

    # Renumbering/Reformating ghost reaches and node ids.
    new_reaches = np.copy(subcls.reach_id)
    new_nodes = np.copy(subcls.node_id)
    new_len = np.copy(subcls.rch_len5)
    uniq_basins = np.unique(subcls.basins[np.where(subcls.ghost == 1)[0]])
    for idy in list(range(len(uniq_basins))):
        basin = np.where(subcls.basins == uniq_basins[idy])[0]
        max_node = np.max(subcls.node_num[basin])
        max_rch = np.max(subcls.rch_topo[basin])
        rch_num = int(max_rch+1)
        nd_num = int(max_node+1)
        uniq_nodes = np.unique(new_nodes[basin[np.where(subcls.ghost[basin] == 1)[0]]])
        for ind in list(range(len(uniq_nodes))):
            ghost = np.where(new_nodes == uniq_nodes[ind])[0]
            for idx in list(range(len(ghost))):
                #new_nodes[ghost[idx]] = np.int(np.str(new_nodes[ghost[idx]])[:-1]+str(6))
                #new_reaches[ghost[idx]] = np.int(np.str(new_reaches[ghost[idx]])[:-1]+str(6))

                #new reach id
                if len(np.str(rch_num)) == 1:
                    fill = '000'
                    new_reaches[ghost[idx]] = np.int(np.str(new_reaches[ghost[idx]])[0:6]+fill+str(rch_num)+str(6))
                if len(np.str(rch_num)) == 2:
                    fill = '00'
                    new_reaches[ghost[idx]] = np.int(np.str(new_reaches[ghost[idx]])[0:6]+fill+str(rch_num)+str(6))
                if len(np.str(rch_num)) == 3:
                    fill = '0'
                    new_reaches[ghost[idx]] = np.int(np.str(new_reaches[ghost[idx]])[0:6]+fill+str(rch_num)+str(6))
                if len(np.str(rch_num)) == 4:
                    new_reaches[ghost[idx]] = np.int(np.str(new_reaches[ghost[idx]])[0:6]+str(rch_num)+str(6))

                #new node id
                if len(np.str(nd_num)) == 1:
                    fill = '00'
                    new_nodes[ghost[idx]] = np.int(np.str(new_reaches[ghost[idx]])[:-1]+fill+str(nd_num)+str(6))
                if len(np.str(nd_num)) == 2:
                    fill = '0'
                    new_nodes[ghost[idx]] = np.int(np.str(new_reaches[ghost[idx]])[:-1]+fill+str(nd_num)+str(6))
                if len(np.str(nd_num)) == 3:
                    new_nodes[ghost[idx]] = np.int(np.str(new_reaches[ghost[idx]])[:-1]+str(nd_num)+str(6))

                #updating length
                old_rch_id = np.where(new_reaches == np.unique(subcls.reach_id[ghost[idx]]))[0]
                new_rch_id = np.where(new_reaches == np.unique(new_reaches[ghost[idx]]))[0]
                if len(old_rch_id) == 0:
                    new_len[new_rch_id] = np.max(subcls.rch_dist5[new_rch_id])-np.min(subcls.rch_dist5[new_rch_id])
                else:
                    new_len[old_rch_id] = np.max(subcls.rch_dist5[old_rch_id])-np.min(subcls.rch_dist5[old_rch_id])
                    new_len[new_rch_id] = np.max(subcls.rch_dist5[new_rch_id])-np.min(subcls.rch_dist5[new_rch_id])

            rch_num = rch_num+1
            nd_num = nd_num+1

    return new_reaches, new_nodes, new_len

###############################################################################

def swot_obs_percentage(subcls, subreaches):

    """
    FUNCTION:
        Calculating the SWOT coverage for each overpass along a reach.

    INPUTS
        subcls -- Object containing attributes for the high-resolution centerline.
            [attributes used]:
                reach_id -- Reach IDs for along the high-resolution centerline.
                orbits -- SWOT orbit locations along the high-resolution
                    centerline.

    OUTPUTS
        swot_coverage -- 2-D array of the minimum and maximum swot coverage
            per reach (%). Dimension is [number of reaches, 2] with the first
            column representing the minimum swot coverage for a particular
            reach and the second column representing the maximum swot coverage
            for a particular reach.
    """

    # Set variables.
    uniq_rch = np.unique(subreaches.id)
    swot_coverage = np.zeros((len(uniq_rch), 75))
    swot_orbits = np.zeros((len(uniq_rch), 75))
    max_obs = np.zeros(len(uniq_rch))
    med_obs = np.zeros(len(uniq_rch))
    mean_obs = np.zeros(len(uniq_rch))

    # Loop through each reach and calculate the coverage for each swot overpass.
    for ind in list(range(len(uniq_rch))):
        rch = np.where(subcls.reach_id == uniq_rch[ind])[0]
        orbs = subcls.orbits[rch]
        max_obs[ind] = np.max(subcls.num_obs[rch])
        med_obs[ind] = np.median(subcls.num_obs[rch])
        mean_obs[ind] = np.mean(subcls.num_obs[rch])
        uniq_orbits = np.unique(orbs)
        uniq_orbits = uniq_orbits[np.where(uniq_orbits>0)[0]]

        if len(uniq_orbits) == 0:
            continue

        for idz in list(range(len(uniq_orbits))):
            rows = np.where(orbs == uniq_orbits[idz])[0]
            swot_coverage[ind,idz] = (len(rows)/len(rch))*100
            swot_orbits[ind,idz] = uniq_orbits[idz]

    return swot_coverage, swot_orbits, max_obs, med_obs, mean_obs

###############################################################################

def centerline_rch_vars(subreaches, subcls):

    """
    FUNCTION:
        Mapping reach-only variables onto the high-resolution centerline points.

    INPUTS
        subreaches -- Object containing attributes for the reaches.
        subcls -- Object containing attributes for the high-resolution centerline.

    OUTPUTS
        cl_rch_wse -- Average water surface elelvation values per reach mapped
            onto the high-resolution centerline points (meters).
        cl_rch_wse_var -- Water surface elelvation vairability per reach mapped
            onto the high-resolution centerline points (meters).
        cl_rch_wth -- Average width valuesper reach mapped
            onto the high-resolution centerline points (meters).
        cl_rch_wth_var --  Width vairability per reach mapped
            onto the high-resolution centerline points (meters).
        cl_rch_nchan_max -- Maximum number of channels per reach mapped
            onto the high-resolution centerline points.
        cl_rch_nchan_mod -- Number of channels mode per reach mapped
            onto the high-resolution centerline points.
        cl_rch_n_nodes -- Number of nodes per reach mapped onto the
            high-resolution centerline points.
        cl_rch_slope -- Reach slope values per reach mapped onto the
            high-resolution centerline points (m/km).
        cl_rch_dist_out -- Distance from outlet per reach mapped
            onto the high-resolution centerline points (meters).
        cl_rch_grod_id -- GROD IDs associated with each reach mapped onto the
            high-resolution centerline points.
        cl_n_rch_up -- Number of upstream reaches per reach mapped onto the
            high-resolution centerline points.
        cl_n_rch_down -- Number of downstream reaches per reach mapped onto the
            high-resolution centerline points.
        cl_rch_id_up -- List of upstream reach IDs per reach mapped onto the
            high-resolution centerline points.
        cl_rch_id_down -- List of downstream reach IDs per reach mapped onto the
            high-resolution centerline points.
    """

    cl_rch_wse = np.zeros(len(subcls.reach_id))
    cl_rch_wse_var = np.zeros(len(subcls.reach_id))
    cl_rch_wth = np.zeros(len(subcls.reach_id))
    cl_rch_wth_var = np.zeros(len(subcls.reach_id))
    cl_rch_facc = np.zeros(len(subcls.reach_id))
    cl_rch_nchan_max = np.zeros(len(subcls.reach_id))
    cl_rch_nchan_mod = np.zeros(len(subcls.reach_id))
    cl_rch_n_nodes = np.zeros(len(subcls.reach_id))
    cl_rch_slope = np.zeros(len(subcls.reach_id))
    cl_rch_dist_out = np.zeros(len(subcls.reach_id))
    cl_rch_grod_id = np.zeros(len(subcls.reach_id))
    cl_n_rch_up = np.zeros(len(subcls.reach_id))
    cl_n_rch_down = np.zeros(len(subcls.reach_id))
    cl_rch_id_up = np.zeros(len(subcls.reach_id), dtype=object)
    cl_rch_id_down = np.zeros(len(subcls.reach_id), dtype=object)

    for ind in list(range(len(subreaches.id))):
        rch = np.where(subcls.reach_id == subreaches.id[ind])[0]
        cl_rch_wse[rch] = subreaches.wse[ind]
        cl_rch_wse_var[rch] = subreaches.wse_var[ind]
        cl_rch_wth[rch] = subreaches.wth[ind]
        cl_rch_wth_var[rch] = subreaches.wth_var[ind]
        cl_rch_facc[rch] = subreaches.facc[ind]
        cl_rch_nchan_max[rch] = subreaches.nchan_max[ind]
        cl_rch_nchan_mod[rch] = subreaches.nchan_mod[ind]
        cl_rch_n_nodes[rch] = subreaches.rch_n_nodes[ind]
        cl_rch_slope[rch] = subreaches.slope[ind]
        cl_rch_dist_out[rch] = subreaches.dist_out[ind]
        cl_rch_grod_id[rch] = subreaches.grod[ind]
        cl_n_rch_up[rch] = subreaches.n_rch_up_filt[ind]
        cl_n_rch_down[rch] = subreaches.n_rch_down_filt[ind]
        cl_rch_id_up[rch] = str(subreaches.rch_id_up_filt[ind][np.where(subreaches.rch_id_up_filt[ind] > 0)[0]])
        cl_rch_id_down[rch] = str(subreaches.rch_id_down_filt[ind][np.where(subreaches.rch_id_down_filt[ind] > 0)[0]])

    return(cl_rch_wse, cl_rch_wse_var, cl_rch_wth, cl_rch_wth_var, cl_rch_facc,
           cl_rch_nchan_max, cl_rch_nchan_mod, cl_rch_n_nodes, cl_rch_slope,
           cl_rch_dist_out, cl_rch_grod_id, cl_n_rch_up, cl_n_rch_down,
           cl_rch_id_up, cl_rch_id_down)

###############################################################################

def centerline_ids_pt2(subreaches, subnodes, subcls, cl_basins, rch_basins,
                       node_basins, cnt):

    """
    FUNCTION:
        Creating unique centerline ids that are ordered per reach and per node
        within a level 6 basin.
        This is a subfunction of "centerline_ids."

    INPUTS
        subreaches -- Object containing attributes for the reaches.
        subcls -- Object containing attributes for the high-resolution centerline.
        cnt -- starting number to add to existing indexes.
        cl_basins -- High-resolution centerline point indexes for the current level
            6 basin.
        rch_basins --  Reach indexes for the current level 6 basin.
        node_basins --  Node indexes for the current level 6 basin.

    OUTPUTS
        cl_id - Unique IDs for each high-resolution centerline point that are
            ordered within the reaches of a level 6 basin.
        rch_cl_id - Array containing the max and min high-resolution centerline
            IDs for each reach in a level 6 basin.
        node_cl_id - Array containing the max and min high_resolution centerline
            IDs for each node in a level 6 basin.
    """

    ID = np.copy(subcls.id[cl_basins])
    cl_reach_id = np.copy(subcls.reach_id[cl_basins])
    cl_reach_ind = np.copy(subcls.rch_ind6[cl_basins])
    cl_node_id = np.copy(subcls.node_id[cl_basins])
    reach_id = np.copy(subreaches.id[rch_basins])
    node_id = np.copy(subnodes.id[node_basins])

    cl_id = np.zeros(len(ID))
    rch_cl_id = np.full((len(reach_id), 2), 0)
    node_cl_id = np.full((len(node_id), 2), 0)
    cnt = cnt
    for ind in list(range(len(reach_id))):
        rch = np.where(cl_reach_id == reach_id[ind])[0]
        cl_id[rch] = cl_reach_ind[rch] + cnt
        rch_cl_id[ind, 0] = np.min(cl_id[rch])
        rch_cl_id[ind, 1] = np.max(cl_id[rch])
        cnt = np.max(cl_id)+1
        uniq_nodes = np.unique(cl_node_id[rch])
        for idx in list(range(len(uniq_nodes))):
            Nodes = np.where(cl_node_id == uniq_nodes[idx])[0]
            index = np.where(node_id == uniq_nodes[idx])[0]
            node_cl_id[index,0] = np.min(cl_id[Nodes])
            node_cl_id[index,1] = np.max(cl_id[Nodes])

    return(cl_id, rch_cl_id, node_cl_id)

###############################################################################

def centerline_ids(subreaches, subnodes, subcls, cnt):
    """
    FUNCTION:
        Creating unique centerline ids that are ordered per reach and per node.

    INPUTS
        subreaches -- Object containing attributes for the reaches.
        subcls -- Object containing attributes for the high-resolution centerline.
        cnt -- starting number to add to existing indexes.

    OUTPUTS
        cl_id - Unique IDs for each high-resolution centerline point that are
            ordered within the reaches.
        rch_cl_id - Array containing the max and min high-resolution centerline
            IDs for each reach.
        node_cl_id - Array containing the max and min high_resolution centerline
            IDs for each node.
    """

    cl_id = np.zeros(len(subcls.id))
    rch_cl_id = np.full((len(subreaches.id), 2), 0)
    node_cl_id = np.full((len(subnodes.id), 2), 0)
    uniq_basins = np.unique(subcls.basins)

    reach_basins = np.zeros(len(subreaches.id))
    for ind in list(range(len(subreaches.id))):
        reach_basins[ind] = np.int(np.str(subreaches.id[ind])[0:6])

    nbasins = np.zeros(len(subnodes.id))
    for ind in list(range(len(subnodes.id))):
        nbasins[ind] = np.int(np.str(subnodes.id[ind])[0:6])

    cnt = cnt
    for ind in list(range(len(uniq_basins))):
        cl_basins = np.where(subcls.basins == uniq_basins[ind])[0]
        rch_basins = np.where(reach_basins == uniq_basins[ind])[0]
        node_basins = np.where(nbasins == uniq_basins[ind])[0]
        new_cl_id, new_rch_cl_id, \
         new_node_cl_id = centerline_ids_pt2(subreaches, subnodes, subcls,
                                             cl_basins, rch_basins, node_basins, cnt)

        cl_id[cl_basins] = new_cl_id
        rch_cl_id[rch_basins,:] = new_rch_cl_id
        node_cl_id[node_basins,:] = new_node_cl_id
        cnt = np.max(cl_id)+1

    return(cl_id, rch_cl_id, node_cl_id)

###############################################################################

def format_cl_node_ids_pt2(nodes, centerlines):

    """
    FUNCTION:
        Assigns and formats the node ids along high_resolution centerline
        at the continental scale. This is a subfunction of "format_cl_node_ids"
        in order to speed up the code.

    INPUTS
        nodes -- Object containing attributes for the nodes.
        centerlines -- Object containing attributes for the high-resolution centerline.

    OUTPUTS
        cl_nodes_id - Node IDs for each high-resolution centerline point.
    """

    nodes_x = np.zeros(len(nodes.id))
    nodes_y = np.zeros(len(nodes.id))
    for idx in list(range(len(nodes.id))):
        nodes_x[idx], nodes_y[idx], __ , __ = utm.from_latlon(nodes.y[idx], nodes.x[idx])

    all_pts = np.vstack((nodes_x, nodes_y)).T
    kdt = sp.cKDTree(all_pts)

    cl_nodes_id= np.zeros([len(centerlines.cl_id),4])
    for ind in list(range(len(nodes.id))):
        #print(ind)

        # converting coordinates for centerlines points.
        cp1 = np.where(centerlines.cl_id == nodes.cl_id[ind][0])[0]
        cp2 = np.where(centerlines.cl_id == nodes.cl_id[ind][1])[0]
        cp1_x, cp1_y, __, __ = utm.from_latlon(centerlines.y[cp1], centerlines.x[cp1])
        cp2_x, cp2_y, __, __ = utm.from_latlon(centerlines.y[cp2], centerlines.x[cp2])

        cp1_pts = np.vstack((cp1_x, cp1_y)).T
        cp1_dist, cp1_ind = kdt.query(cp1_pts, k = 4, distance_upper_bound = 200)
        cp2_pts = np.vstack((cp2_x, cp2_y)).T
        cp2_dist, cp2_ind = kdt.query(cp2_pts, k = 4, distance_upper_bound = 200)

        rmv1 = np.where(cp1_ind[0] == len(nodes.id))[0]
        rmv2 = np.where(cp2_ind[0] == len(nodes.id))[0]
        cp1_ind = np.delete(cp1_ind[0], rmv1)
        cp2_ind = np.delete(cp2_ind[0], rmv2)

        cp1_nodes = nodes.id[np.where(nodes.id[cp1_ind] != nodes.id[ind])[0]]
        cp2_nodes = nodes.id[np.where(nodes.id[cp2_ind] != nodes.id[ind])[0]]

        cl_nodes_id[cp1,0:len(cp1_nodes)] = cp1_nodes
        cl_nodes_id[cp2,0:len(cp2_nodes)] = cp2_nodes

    return cl_nodes_id

###############################################################################

def format_cl_node_ids(nodes, centerlines):

    """
    FUNCTION:
        Assigns and formats the node ids along high_resolution centerline
        at the continental scale.

    INPUTS
        nodes -- Object containing attributes for the nodes.
        centerlines -- Object containing attributes for the high-resolution centerline.

    OUTPUTS
        cl_nodes_id - Node IDs for each high-resolution centerline point.
    """

    cl_nodes_id = np.zeros([len(centerlines.cl_id),4])
    # divide up into basin level 6 to see if it makes things faster...
    level_nodes = np.array([np.int(np.str(ind)[0:6]) for ind in nodes.id])
    level_cl = np.array([np.int(np.str(ind)[0:6]) for ind in centerlines.node_id])
    uniq_level = np.unique(level_cl)
    for ind in list(range(len(uniq_level))):

        #print(uniq_level[ind])

        cl_ind = np.where(level_cl == uniq_level[ind])[0]
        nodes_ind = np.where(level_nodes == uniq_level[ind])[0]

        Subnodes = Object()
        Subcls = Object()

        Subnodes.id = nodes.id[nodes_ind]
        Subnodes.x = nodes.x[nodes_ind]
        Subnodes.y = nodes.y[nodes_ind]
        Subnodes.cl_id = nodes.cl_id[nodes_ind]
        Subcls.cl_id = centerlines.cl_id[cl_ind]
        Subcls.x = centerlines.x[cl_ind]
        Subcls.y = centerlines.y[cl_ind]

        cl_nodes_id[cl_ind,:] = format_cl_node_ids_pt2(Subnodes, Subcls)

    return cl_nodes_id

################################################################################

def format_cl_rch_ids(reaches, centerlines):

    """
    FUNCTION:
        Assigns and formats the reach ids along high_resolution centerline
        at the continental scale.

    INPUTS
        reaches -- Object containing attributes for the reaches.
        centerlines -- Object containing attributes for the high-resolution centerline.

    OUTPUTS
        cl_rch_id - Reach IDs for each high-resolution centerline point.
    """

    cl_rch_id= np.zeros([len(centerlines.cl_id),4])
    for ind in list(range(len(reaches.id))):
        #print(ind)
        up = np.where(reaches.rch_id_up[ind,:] > 0)[0]
        down = np.where(reaches.rch_id_down[ind,:] > 0)[0]

        # converting coordinates for centerlines points.
        cp1 = np.where(centerlines.cl_id == reaches.cl_id[ind][0])[0]
        cp2 = np.where(centerlines.cl_id == reaches.cl_id[ind][1])[0]
        cp1_x, cp1_y, __, __ = utm.from_latlon(centerlines.y[cp1], centerlines.x[cp1])
        cp2_x, cp2_y, __, __ = utm.from_latlon(centerlines.y[cp2], centerlines.x[cp2])

        if len(up) == 0 and len(down) == 0:
            continue

        if len(up) > 0 and len(down) == 0:

            up_lon = reaches.x[np.where(reaches.id == reaches.rch_id_up[ind,:][up[0]])]
            up_lat = reaches.y[np.where(reaches.id == reaches.rch_id_up[ind,:][up[0]])]
            up_x, up_y, __, __ = utm.from_latlon(up_lat, up_lon)

            d1 = np.sqrt(((cp1_x - up_x)**2 + (cp1_y - up_y)**2))
            d2 = np.sqrt(((cp2_x - up_x)**2 + (cp2_y - up_y)**2))

            if d1 > d2:
                cl_rch_id[cp2,:] = reaches.rch_id_up[ind,0:4]
            if d1 < d2:
                cl_rch_id[cp1,:] = reaches.rch_id_up[ind,0:4]

        if len(up) == 0 and len(down) > 0:
            dn_lon = reaches.x[np.where(reaches.id == reaches.rch_id_down[ind,:][down[0]])]
            dn_lat = reaches.y[np.where(reaches.id == reaches.rch_id_down[ind,:][down[0]])]
            dn_x, dn_y, __, __ = utm.from_latlon(dn_lat, dn_lon)

            d1 = np.sqrt(((cp1_x - dn_x)**2 + (cp1_y - dn_y)**2))
            d2 = np.sqrt(((cp2_x - dn_x)**2 + (cp2_y - dn_y)**2))

            if d1 > d2:
                cl_rch_id[cp2,:] = reaches.rch_id_down[ind,0:4]
            if d1 < d2:
                cl_rch_id[cp1,:] = reaches.rch_id_down[ind,0:4]

        if len(up) > 0 and len(down) > 0:

            up_lon = reaches.x[np.where(reaches.id == reaches.rch_id_up[ind,:][up[0]])] #changed 0 to ind in rch_id_up rows.
            up_lat = reaches.y[np.where(reaches.id == reaches.rch_id_up[ind,:][up[0]])]
            up_x, up_y, __, __ = utm.from_latlon(up_lat, up_lon)

            dn_lon = reaches.x[np.where(reaches.id == reaches.rch_id_down[ind,:][down[0]])]
            dn_lat = reaches.y[np.where(reaches.id == reaches.rch_id_down[ind,:][down[0]])]
            dn_x, dn_y, __, __ = utm.from_latlon(dn_lat, dn_lon)

            d1 = np.sqrt(((cp1_x - up_x)**2 + (cp1_y - up_y)**2))
            d2 = np.sqrt(((cp1_x - dn_x)**2 + (cp1_y - dn_y)**2))

            if d1 > d2:
                cl_rch_id[cp1,:] = reaches.rch_id_down[ind,0:4]
                cl_rch_id[cp2,:] = reaches.rch_id_up[ind,0:4]
            if d1 < d2:
                cl_rch_id[cp1,:] = reaches.rch_id_up[ind,0:4]
                cl_rch_id[cp2,:] = reaches.rch_id_down[ind,0:4] #feel like this should be down...

    return cl_rch_id
