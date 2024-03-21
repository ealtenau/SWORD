from __future__ import division
import utm
import numpy as np
from scipy import spatial as sp
import netCDF4 as nc
from pyproj import Proj
import math
from statistics import mode
import time

###############################################################################

class Object(object):
    """
    FUNCTION:
        Creates class object to assign attributes to.
    """
    pass

###############################################################################

def read_data(filename):

    centerlines = Object()
    nodes = Object()
    reaches = Object ()
    
    data = nc.Dataset(filename)
    
    centerlines.cl_id = np.array(data.groups['centerlines'].variables['cl_id'][:])
    centerlines.x = np.array(data.groups['centerlines'].variables['x'][:])
    centerlines.y = np.array(data.groups['centerlines'].variables['y'][:])
    centerlines.reach_id = np.array(data.groups['centerlines'].variables['reach_id'][:])
    centerlines.node_id = np.array(data.groups['centerlines'].variables['node_id'][:])
    
    nodes.id = np.array(data.groups['nodes'].variables['node_id'][:])
    nodes.cl_id = np.array(data.groups['nodes'].variables['cl_ids'][:])
    nodes.x = np.array(data.groups['nodes'].variables['x'][:])
    nodes.y = np.array(data.groups['nodes'].variables['y'][:])
    nodes.len = np.array(data.groups['nodes'].variables['node_length'][:])
    nodes.wse = np.array(data.groups['nodes'].variables['wse'][:])
    nodes.wse_var = np.array(data.groups['nodes'].variables['wse_var'][:])
    nodes.wth = np.array(data.groups['nodes'].variables['width'][:])
    nodes.wth_var = np.array(data.groups['nodes'].variables['width_var'][:])
    nodes.grod = np.array(data.groups['nodes'].variables['obstr_type'][:])
    nodes.grod_fid = np.array(data.groups['nodes'].variables['grod_id'][:])
    nodes.hfalls_fid = np.array(data.groups['nodes'].variables['hfalls_id'][:])
    nodes.nchan_max = np.array(data.groups['nodes'].variables['n_chan_max'][:])
    nodes.nchan_mod = np.array(data.groups['nodes'].variables['n_chan_mod'][:])
    nodes.dist_out = np.array(data.groups['nodes'].variables['dist_out'][:])
    nodes.reach_id = np.array(data.groups['nodes'].variables['reach_id'][:])
    nodes.facc = np.array(data.groups['nodes'].variables['facc'][:])
    nodes.lakeflag = np.array(data.groups['nodes'].variables['lakeflag'][:])
    nodes.wth_coef = np.array(data.groups['nodes'].variables['wth_coef'][:])
    nodes.ext_dist_coef = np.array(data.groups['nodes'].variables['ext_dist_coef'][:])
    nodes.max_wth = np.array(data.groups['nodes'].variables['max_width'][:])
    nodes.meand_len = np.array(data.groups['nodes'].variables['meander_length'][:])
    nodes.river_name = np.array(data.groups['nodes'].variables['river_name'][:])
    nodes.manual_add = np.array(data.groups['nodes'].variables['manual_add'][:])
    nodes.sinuosity = np.array(data.groups['nodes'].variables['sinuosity'][:])
    nodes.edit_flag = np.array(data.groups['nodes'].variables['edit_flag'][:])
    nodes.trib_flag = np.array(data.groups['nodes'].variables['trib_flag'][:])

    reaches.id = np.array(data.groups['reaches'].variables['reach_id'][:])
    reaches.cl_id = np.array(data.groups['reaches'].variables['cl_ids'][:])
    reaches.x = np.array(data.groups['reaches'].variables['x'][:])
    reaches.x_min = np.array(data.groups['reaches'].variables['x_min'][:])
    reaches.x_max = np.array(data.groups['reaches'].variables['x_max'][:])
    reaches.y = np.array(data.groups['reaches'].variables['y'][:])
    reaches.y_min = np.array(data.groups['reaches'].variables['y_min'][:])
    reaches.y_max = np.array(data.groups['reaches'].variables['y_max'][:])
    reaches.len = np.array(data.groups['reaches'].variables['reach_length'][:])
    reaches.wse = np.array(data.groups['reaches'].variables['wse'][:])
    reaches.wse_var = np.array(data.groups['reaches'].variables['wse_var'][:])
    reaches.wth = np.array(data.groups['reaches'].variables['width'][:])
    reaches.wth_var = np.array(data.groups['reaches'].variables['width_var'][:])
    reaches.slope = np.array(data.groups['reaches'].variables['slope'][:])
    reaches.rch_n_nodes = np.array(data.groups['reaches'].variables['n_nodes'][:])
    reaches.grod = np.array(data.groups['reaches'].variables['obstr_type'][:])
    reaches.grod_fid = np.array(data.groups['reaches'].variables['grod_id'][:])
    reaches.hfalls_fid = np.array(data.groups['reaches'].variables['hfalls_id'][:])
    reaches.lakeflag = np.array(data.groups['reaches'].variables['lakeflag'][:])
    reaches.nchan_max = np.array(data.groups['reaches'].variables['n_chan_max'][:])
    reaches.nchan_mod = np.array(data.groups['reaches'].variables['n_chan_mod'][:])
    reaches.dist_out = np.array(data.groups['reaches'].variables['dist_out'][:])
    reaches.n_rch_up = np.array(data.groups['reaches'].variables['n_rch_up'][:])
    reaches.n_rch_down = np.array(data.groups['reaches'].variables['n_rch_down'][:])
    reaches.rch_id_up = np.array(data.groups['reaches'].variables['rch_id_up'][:])
    reaches.rch_id_down = np.array(data.groups['reaches'].variables['rch_id_dn'][:])
    reaches.max_obs = np.array(data.groups['reaches'].variables['swot_obs'][:])
    reaches.orbits = np.array(data.groups['reaches'].variables['swot_orbits'][:])
    reaches.facc = np.array(data.groups['reaches'].variables['facc'][:])
    reaches.iceflag = np.array(data.groups['reaches'].variables['iceflag'][:])
    reaches.max_wth = np.array(data.groups['reaches'].variables['max_width'][:])
    reaches.river_name = np.array(data.groups['reaches'].variables['river_name'][:])
    reaches.low_slope = np.array(data.groups['reaches'].variables['low_slope_flag'][:])
    reaches.edit_flag= np.array(data.groups['reaches'].variables['edit_flag'][:])
    reaches.trib_flag = np.array(data.groups['reaches'].variables['trib_flag'][:])

    data.close()    

    return centerlines, nodes, reaches
    
###############################################################################

def meters_to_degrees(meters, latitude):
    deg = np.round(meters/(111.32 * 1000 * math.cos(latitude * (math.pi / 180))),5)
    return deg

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
    data.lon = np.array(new.groups['centerlines'].variables['x'][:])
    data.lat = np.array(new.groups['centerlines'].variables['y'][:])
    data.seg = np.array(new.groups['centerlines'].variables['segID'][:])
    data.ind = np.array(new.groups['centerlines'].variables['segInd'][:])
    data.dist = np.array(new.groups['centerlines'].variables['segDist'][:])
    data.wth = np.array(new.groups['centerlines'].variables['p_width'][:])
    data.elv = np.array(new.groups['centerlines'].variables['p_height'][:])
    data.facc = np.array(new.groups['centerlines'].variables['flowacc'][:])
    data.lake = np.array(new.groups['centerlines'].variables['lakeflag'][:])
    data.delta = np.array(new.groups['centerlines'].variables['deltaflag'][:])
    data.nchan = np.array(new.groups['centerlines'].variables['nchan'][:])
    data.grand = np.array(new.groups['centerlines'].variables['grand_id'][:])
    data.grod = np.array(new.groups['centerlines'].variables['grod_id'][:])
    data.grod_fid = np.array(new.groups['centerlines'].variables['grod_fid'][:])
    data.hfalls_fid = np.array(new.groups['centerlines'].variables['hfalls_fid'][:])
    data.basins = np.array(new.groups['centerlines'].variables['basin_code'][:])
    data.manual = np.array(new.groups['centerlines'].variables['manual_add'][:])
    data.num_obs = np.array(new.groups['centerlines'].variables['number_obs'][:])
    data.orbits = np.array(new.groups['centerlines'].variables['orbits'][:])
    data.eps = np.array(new.groups['centerlines'].variables['endpoints'][:])
    data.lake_id = np.array(new.groups['centerlines'].variables['lake_id'][:])
    new.close()

    return data

###############################################################################

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
         zone_num[ind], zone_let_int) = utm.from_latlon(latitude[ind],longitude[ind])
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
            "+proj=utm +zone=" + str(int(unq_zones[idx])) +
            " +ellips=WGS84 +datum=WGS84 +units=m")
    else:
        myproj = Proj(
            "+proj=utm +south +zone=" + str(int(unq_zones[idx])) +
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
        # print(ind, uniq_segs[ind])
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
        break_index = np.zeros(int(divs-1))
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
        if np.max(new_rch_dist[rch])>max_dist:
            print(uniq_rch[ind], 'max distance too long - likely an index problem')

    return new_rch_id, new_rch_dist

###############################################################################
##################### Topology and Attribute Functions ########################
###############################################################################
def simple_rch_ids(centerlines, sword_basins, sword_rch_nums):
    unq_rchs = np.unique(centerlines.rch_id5)
    reach_id = np.zeros(len(centerlines.lon), dtype=int)
    for r in list(range(len(unq_rchs))):
        rch = np.where(centerlines.rch_id5 == unq_rchs[r])[0]
        l6 = mode(centerlines.basins[rch])
        flag = mode(centerlines.lake[rch])
        if flag == 0:
            tp = 1
        if flag != 0:
            tp = 3
        sb = np.where(sword_basins == l6)[0]
        if len(sb) == 0:
            add = 0
        else:
            add = np.max(sword_rch_nums[sb])+1
        rch_num = unq_rchs[r]+add
        if len(str(rch_num)) == 1:
            fill = '000'
            reach_id[rch] = int(str(l6)+fill+str(rch_num)+str(tp))
        if len(str(rch_num)) == 2:
            fill = '00'
            reach_id[rch] = int(str(l6)+fill+str(rch_num)+str(tp))
        if len(str(rch_num)) == 3:
            fill = '0'
            reach_id[rch] = int(str(l6)+fill+str(rch_num)+str(tp))
        if len(str(rch_num)) == 4:
            reach_id[rch] = int(str(l6)+fill+str(rch_num)+str(tp))
    return reach_id

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
    node_id = np.zeros(len(subcls.rch_id5), dtype = int)
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

        break_index = np.zeros(int(divs-1))
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
            if len(str(int(node_num[rch[inz]]))) == 1:
                fill = '00'
                node_id[rch[inz]] = int(str(subcls.reach_id[rch[inz]])[:-1]+fill+str(int(node_num[rch[inz]]))+str(subcls.reach_id[rch[inz]])[10:11])
            if len(str(int(node_num[rch[inz]]))) == 2:
                fill = '0'
                node_id[rch[inz]] = int(str(subcls.reach_id[rch[inz]])[:-1]+fill+str(int(node_num[rch[inz]]))+str(subcls.reach_id[rch[inz]])[10:11])
            if len(str(int(node_num[rch[inz]]))) == 3:
                node_id[rch[inz]] = int(str(subcls.reach_id[rch[inz]])[:-1]+str(int(node_num[rch[inz]]))+str(subcls.reach_id[rch[inz]])[10:11])

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

def centerline_ids(subreaches, subnodes, subcls):

    rch_cl_id = np.full((len(subreaches.id), 2), 0)
    node_cl_id = np.full((len(subnodes.id), 2), 0)

    unq_rchs = np.unique(subreaches.id)
    for ind in list(range(len(unq_rchs))):
        cl_rch = np.where(subcls.reach_id == unq_rchs[ind])[0]
        rch = np.where(subreaches.id == unq_rchs[ind])[0]
        rch_cl_id[rch,0] = np.min(subcls.id[cl_rch])
        rch_cl_id[rch,1] = np.max(subcls.id[cl_rch])

    unq_nodes = np.unique(subnodes.id)
    for idx in list(range(len(unq_nodes))):
        cl_nodes = np.where(subcls.node_id == unq_nodes[idx])[0]
        nds = np.where(subnodes.id == unq_nodes[idx])[0]
        node_cl_id[nds,0] = np.min(subcls.id[cl_nodes])
        node_cl_id[nds,1] = np.max(subcls.id[cl_nodes])
    
    return(rch_cl_id, node_cl_id)

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
    level_nodes = np.array([int(str(ind)[0:6]) for ind in nodes.id])
    level_cl = np.array([int(str(ind)[0:6]) for ind in centerlines.node_id])
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
        Subnodes.cl_id = nodes.cl_id[nodes_ind,:]
        Subcls.cl_id = centerlines.cl_id[cl_ind]
        Subcls.x = centerlines.lon[cl_ind]
        Subcls.y = centerlines.lat[cl_ind]

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
        cp1_x, cp1_y, __, __ = utm.from_latlon(centerlines.lat[cp1], centerlines.lon[cp1])
        cp2_x, cp2_y, __, __ = utm.from_latlon(centerlines.lat[cp2], centerlines.lon[cp2])

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

################################################################################
    
def append_data(centerlines, nodes, reaches, 
                subcls, subnodes, subreaches):

    centerlines.cl_id = np.append(centerlines.cl_id, subcls.cl_id)
    centerlines.x = np.append(centerlines.x, subcls.lon)
    centerlines.y = np.append(centerlines.y, subcls.lat)
    centerlines.reach_id = np.append(centerlines.reach_id, subcls.reach_id, axis=1)
    centerlines.node_id = np.append(centerlines.node_id, subcls.node_id, axis=1)
    
    nodes.id = np.append(nodes.id, subnodes.id)
    nodes.cl_id = np.append(nodes.cl_id, subnodes.cl_id, axis=1)
    nodes.x = np.append(nodes.x, subnodes.x)
    nodes.y = np.append(nodes.y, subnodes.y)
    nodes.len = np.append(nodes.len, subnodes.len)
    nodes.wse = np.append(nodes.wse, subnodes.wse)
    nodes.wse_var = np.append(nodes.wse_var, subnodes.wse_var)
    nodes.wth = np.append(nodes.wth, subnodes.wth)
    nodes.wth_var = np.append(nodes.wth_var, subnodes.wth_var)
    nodes.grod = np.append(nodes.grod, subnodes.grod)
    nodes.grod_fid = np.append(nodes.grod_fid, subnodes.grod_fid)
    nodes.hfalls_fid = np.append(nodes.hfalls_fid, subnodes.hfalls_fid)
    nodes.nchan_max = np.append(nodes.nchan_max, subnodes.nchan_max)
    nodes.nchan_mod = np.append(nodes.nchan_mod, subnodes.nchan_mod)
    nodes.dist_out = np.append(nodes.dist_out, subnodes.dist_out)
    nodes.reach_id = np.append(nodes.reach_id, subnodes.reach_id)
    nodes.facc = np.append(nodes.facc, subnodes.facc)
    nodes.lakeflag = np.append(nodes.lakeflag, subnodes.lakeflag)
    nodes.wth_coef = np.append(nodes.wth_coef, subnodes.wth_coef)
    nodes.ext_dist_coef = np.append(nodes.ext_dist_coef, subnodes.ext_dist_coef)
    nodes.max_wth = np.append(nodes.max_wth, subnodes.max_wth)
    nodes.meand_len = np.append(nodes.meand_len, subnodes.meand_len)
    nodes.river_name = np.append(nodes.river_name, subnodes.river_name)
    nodes.manual_add = np.append(nodes.manual_add, subnodes.manual_add)
    nodes.sinuosity = np.append(nodes.sinuosity, subnodes.sinuosity)
    nodes.edit_flag = np.append(nodes.edit_flag, subnodes.edit_flag)
    nodes.trib_flag = np.append(nodes.trib_flag, subnodes.trib_flag)

    reaches.id = np.append(reaches.id, subreaches.id)
    reaches.cl_id = np.append(reaches.cl_id, subreaches.cl_id, axis=1)
    reaches.x = np.append(reaches.x, subreaches.x)
    reaches.x_min = np.append(reaches.x_min, subreaches.x_min)
    reaches.x_max = np.append(reaches.x_max, subreaches.x_max)
    reaches.y = np.append(reaches.y, subreaches.y)
    reaches.y_min = np.append(reaches.y_min, subreaches.y_min)
    reaches.y_max = np.append(reaches.y_max, subreaches.y_max)
    reaches.len = np.append(reaches.len, subreaches.len)
    reaches.wse = np.append(reaches.wse, subreaches.wse)
    reaches.wse_var = np.append(reaches.wse_var, subreaches.wse_var)
    reaches.wth = np.append(reaches.wth, subreaches.wth)
    reaches.wth_var = np.append(reaches.wth_var, subreaches.wth_var)
    reaches.slope = np.append(reaches.slope, subreaches.slope)
    reaches.rch_n_nodes = np.append(reaches.rch_n_nodes, subreaches.rch_n_nodes)
    reaches.grod = np.append(reaches.grod, subreaches.grod)
    reaches.grod_fid = np.append(reaches.grod_fid, subreaches.grod_fid)
    reaches.hfalls_fid = np.append(reaches.hfalls_fid, subreaches.hfalls_fid)
    reaches.lakeflag = np.append(reaches.lakeflag, subreaches.lakeflag)
    reaches.nchan_max = np.append(reaches.nchan_max, subreaches.nchan_max)
    reaches.nchan_mod = np.append(reaches.nchan_mod, subreaches.nchan_mod)
    reaches.dist_out = np.append(reaches.dist_out, subreaches.dist_out)
    reaches.n_rch_up = np.append(reaches.n_rch_up, subreaches.n_rch_up)
    reaches.n_rch_down = np.append(reaches.n_rch_down, subreaches.n_rch_down)
    reaches.rch_id_up = np.append(reaches.rch_id_up, subreaches.rch_id_up, axis=1)
    reaches.rch_id_down = np.append(reaches.rch_id_down, subreaches.rch_id_down, axis=1)
    reaches.max_obs = np.append(reaches.max_obs, subreaches.max_obs)
    reaches.orbits = np.append(reaches.orbits, subreaches.orbits, axis=1)
    reaches.facc = np.append(reaches.facc, subreaches.facc)
    reaches.iceflag = np.append(reaches.iceflag, subreaches.iceflag, axis=1)
    reaches.max_wth = np.append(reaches.max_wth, subreaches.max_wth)
    reaches.river_name = np.append(reaches.river_name, subreaches.river_name)
    reaches.low_slope = np.append(reaches.low_slope, subreaches.low_slope)
    reaches.edit_flag = np.append(reaches.edit_flag, subreaches.edit_flag)
    reaches.trib_flag = np.append(reaches.trib_flag, subreaches.trib_flag)
    
###############################################################################

def write_database_nc(centerlines, reaches, nodes, region, outfile):

    """
    FUNCTION:
        Outputs the SWOT River Database (SWORD) information in netcdf
        format. The file contains attributes for the high-resolution centerline,
        nodes, and reaches.

    INPUTS
        centerlines -- Object containing lcation and attribute information
            along the high-resolution centerline.
        reaches -- Object containing lcation and attribute information for
            each reach.
        nodes -- Object containing lcation and attribute information for
            each node.
        outfile -- Path for netcdf to be written.

    OUTPUTS
        SWORD NetCDF -- NetCDF file containing attributes for the high-resolution
            centerline, node, and reach locations.
    """

    start = time.time()

    # global attributes
    root_grp = nc.Dataset(outfile, 'w', format='NETCDF4')
    root_grp.x_min = np.min(centerlines.x)
    root_grp.x_max = np.max(centerlines.x)
    root_grp.y_min = np.min(centerlines.y)
    root_grp.y_max = np.max(centerlines.y)
    root_grp.Name = region
    root_grp.production_date = time.strftime("%d-%b-%Y %H:%M:%S", time.gmtime()) #utc time
    #root_grp.history = 'Created ' + time.ctime(time.time())

    # groups
    cl_grp = root_grp.createGroup('centerlines')
    node_grp = root_grp.createGroup('nodes')
    rch_grp = root_grp.createGroup('reaches')
    # subgroups
    sub_grp1 = rch_grp.createGroup('area_fits')
    sub_grp2 = rch_grp.createGroup('discharge_models')
    # discharge subgroups
    qgrp1 = sub_grp2.createGroup('unconstrained')
    qgrp2 = sub_grp2.createGroup('constrained')
    # unconstrained discharge models
    ucmod1 = qgrp1.createGroup('MetroMan')
    ucmod2 = qgrp1.createGroup('BAM')
    ucmod3 = qgrp1.createGroup('HiVDI')
    ucmod4 = qgrp1.createGroup('MOMMA')
    ucmod5 = qgrp1.createGroup('SADS')
    ucmod6 = qgrp1.createGroup('SIC4DVar')
    # constrained discharge models
    cmod1 = qgrp2.createGroup('MetroMan')
    cmod2 = qgrp2.createGroup('BAM')
    cmod3 = qgrp2.createGroup('HiVDI')
    cmod4 = qgrp2.createGroup('MOMMA')
    cmod5 = qgrp2.createGroup('SADS')
    cmod6 = qgrp2.createGroup('SIC4DVar')

    # dimensions
    #root_grp.createDimension('d1', 2)
    cl_grp.createDimension('num_points', len(centerlines.cl_id))
    cl_grp.createDimension('num_domains', 4)

    node_grp.createDimension('num_nodes', len(nodes.id))
    node_grp.createDimension('num_ids', 2)

    rch_grp.createDimension('num_reaches', len(reaches.id))
    rch_grp.createDimension('num_ids', 2)
    rch_grp.createDimension('num_domains', 4)
    rch_grp.createDimension('julian_day', 366)
    rch_grp.createDimension('orbits', 75)
    sub_grp1.createDimension('nCoeffs', 2)
    sub_grp1.createDimension('nReg', 3)
    sub_grp1.createDimension('num_domains', 4)

    # centerline variables
    cl_id = cl_grp.createVariable(
        'cl_id', 'i8', ('num_points',), fill_value=-9999.)
    cl_x = cl_grp.createVariable(
        'x', 'f8', ('num_points',), fill_value=-9999.)
    cl_x.units = 'degrees east'
    cl_y = cl_grp.createVariable(
        'y', 'f8', ('num_points',), fill_value=-9999.)
    cl_y.units = 'degrees north'
    reach_id = cl_grp.createVariable(
        'reach_id', 'i8', ('num_domains','num_points'), fill_value=-9999.)
    reach_id.format = 'CBBBBBRRRRT'
    node_id = cl_grp.createVariable(
        'node_id', 'i8', ('num_domains','num_points'), fill_value=-9999.)
    node_id.format = 'CBBBBBRRRRNNNT'

    # node variables
    Node_ID = node_grp.createVariable(
        'node_id', 'i8', ('num_nodes',), fill_value=-9999.)
    Node_ID.format = 'CBBBBBRRRRNNNT'
    node_cl_id = node_grp.createVariable(
        'cl_ids', 'i8', ('num_ids','num_nodes'), fill_value=-9999.)
    node_x = node_grp.createVariable(
        'x', 'f8', ('num_nodes',), fill_value=-9999.)
    node_x.units = 'degrees east'
    node_y = node_grp.createVariable(
        'y', 'f8', ('num_nodes',), fill_value=-9999.)
    node_y.units = 'degrees north'
    node_len = node_grp.createVariable(
        'node_length', 'f8', ('num_nodes',), fill_value=-9999.)
    node_len.units = 'meters'
    node_rch_id = node_grp.createVariable(
        'reach_id', 'i8', ('num_nodes',), fill_value=-9999.)
    node_rch_id.format = 'CBBBBBRRRRT'
    node_wse = node_grp.createVariable(
        'wse', 'f8', ('num_nodes',), fill_value=-9999.)
    node_wse.units = 'meters'
    node_wse_var = node_grp.createVariable(
        'wse_var', 'f8', ('num_nodes',), fill_value=-9999.)
    node_wse_var.units = 'meters^2'
    node_wth = node_grp.createVariable(
        'width', 'f8', ('num_nodes',), fill_value=-9999.)
    node_wth.units = 'meters'
    node_wth_var = node_grp.createVariable(
        'width_var', 'f8', ('num_nodes',), fill_value=-9999.)
    node_wth_var.units = 'meters^2'
    node_chan_max = node_grp.createVariable(
        'n_chan_max', 'i4', ('num_nodes',), fill_value=-9999.)
    node_chan_mod = node_grp.createVariable(
        'n_chan_mod', 'i4', ('num_nodes',), fill_value=-9999.)
    node_grod_id = node_grp.createVariable(
        'obstr_type', 'i4', ('num_nodes',), fill_value=-9999.)
    node_grod_fid = node_grp.createVariable(
        'grod_id', 'i8', ('num_nodes',), fill_value=-9999.)
    node_hfalls_fid = node_grp.createVariable(
        'hfalls_id', 'i8', ('num_nodes',), fill_value=-9999.)
    node_dist_out = node_grp.createVariable(
        'dist_out', 'f8', ('num_nodes',), fill_value=-9999.)
    node_dist_out.units = 'meters'
    node_wth_coef = node_grp.createVariable(
        'wth_coef', 'f8', ('num_nodes',), fill_value=-9999.)
    node_ext_dist_coef = node_grp.createVariable(
        'ext_dist_coef', 'f8', ('num_nodes',), fill_value=-9999.)
    node_facc = node_grp.createVariable(
        'facc', 'f8', ('num_nodes',), fill_value=-9999.)
    node_facc.units = 'km^2'
    node_lakeflag = node_grp.createVariable(
        'lakeflag', 'i8', ('num_nodes',), fill_value=-9999.)
    #node_lake_id = node_grp.createVariable(
        #'lake_id', 'i8', ('num_nodes',), fill_value=-9999.)
    node_max_wth = node_grp.createVariable(
        'max_width', 'f8', ('num_nodes',), fill_value=-9999.)
    node_max_wth.units = 'meters'
    node_meand_len = node_grp.createVariable(
        'meander_length', 'f8', ('num_nodes',), fill_value=-9999.)
    node_sinuosity = node_grp.createVariable(
        'sinuosity', 'f8', ('num_nodes',), fill_value=-9999.)
    node_manual_add = node_grp.createVariable(
        'manual_add', 'i4', ('num_nodes',), fill_value=-9999.)
    node_river_name = node_grp.createVariable(
        'river_name', 'S50', ('num_nodes',))
    node_river_name._Encoding = 'ascii'
    node_edit_flag = node_grp.createVariable(
        'edit_flag',  'S50', ('num_nodes',))
    node_edit_flag._Encoding = 'ascii'
    node_trib_flag = node_grp.createVariable(
        'trib_flag', 'i4', ('num_nodes',), fill_value=-9999.)

    # reach variables
    Reach_ID = rch_grp.createVariable(
        'reach_id', 'i8', ('num_reaches',), fill_value=-9999.)
    Reach_ID.format = 'CBBBBBRRRRT'
    rch_cl_id = rch_grp.createVariable(
        'cl_ids', 'i8', ('num_ids','num_reaches'), fill_value=-9999.)
    rch_x = rch_grp.createVariable(
        'x', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_x.units = 'degrees east'
    rch_x_min = rch_grp.createVariable(
        'x_min', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_x_min.units = 'degrees east'
    rch_x_max = rch_grp.createVariable(
        'x_max', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_x_max.units = 'degrees east'
    rch_y = rch_grp.createVariable(
        'y', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_y.units = 'degrees north'
    rch_y_min = rch_grp.createVariable(
        'y_min', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_y_min.units = 'degrees north'
    rch_y_max = rch_grp.createVariable(
        'y_max', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_y_max.units = 'degrees north'
    rch_len = rch_grp.createVariable(
        'reach_length', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_len.units = 'meters'
    num_nodes = rch_grp.createVariable(
        'n_nodes', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_wse = rch_grp.createVariable(
        'wse', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_wse.units = 'meters'
    rch_wse_var = rch_grp.createVariable(
        'wse_var', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_wse_var.units = 'meters^2'
    rch_wth = rch_grp.createVariable(
        'width', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_wth.units = 'meters'
    rch_wth_var = rch_grp.createVariable(
        'width_var', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_wth_var.units = 'meters^2'
    rch_facc = rch_grp.createVariable(
        'facc', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_facc.units = 'km^2'
    rch_chan_max = rch_grp.createVariable(
        'n_chan_max', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_chan_mod = rch_grp.createVariable(
        'n_chan_mod', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_grod_id = rch_grp.createVariable(
        'obstr_type', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_grod_fid = rch_grp.createVariable(
        'grod_id', 'i8', ('num_reaches',), fill_value=-9999.)
    rch_hfalls_fid = rch_grp.createVariable(
        'hfalls_id', 'i8', ('num_reaches',), fill_value=-9999.)
    rch_slope = rch_grp.createVariable(
        'slope', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_slope.units = 'meters/kilometers'
    rch_dist_out = rch_grp.createVariable(
        'dist_out', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_dist_out.units = 'meters'
    n_rch_up = rch_grp.createVariable(
        'n_rch_up', 'i4', ('num_reaches',), fill_value=-9999.)
    n_rch_down= rch_grp.createVariable(
        'n_rch_down', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_id_up = rch_grp.createVariable(
        'rch_id_up', 'i8', ('num_domains','num_reaches'), fill_value=-9999.)
    rch_id_down = rch_grp.createVariable(
        'rch_id_dn', 'i8', ('num_domains','num_reaches'), fill_value=-9999.)
    rch_lakeflag = rch_grp.createVariable(
        'lakeflag', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_iceflag = rch_grp.createVariable(
        'iceflag', 'i4', ('julian_day','num_reaches'), fill_value=-9999.)
    #rch_lake_id = rch_grp.createVariable(
        #'lake_id', 'i8', ('num_reaches',), fill_value=-9999.)
    rch_swot_obs = rch_grp.createVariable(
        'swot_obs', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_orbits = rch_grp.createVariable(
        'swot_orbits', 'i8', ('orbits','num_reaches'), fill_value=-9999.)
    rch_river_name = rch_grp.createVariable(
        'river_name', 'S50', ('num_reaches',))
    rch_river_name._Encoding = 'ascii'
    rch_max_wth = rch_grp.createVariable(
        'max_width', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_max_wth.units = 'meters'
    rch_low_slope = rch_grp.createVariable(
        'low_slope_flag', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_edit_flag = rch_grp.createVariable(
        'edit_flag', 'S50', ('num_reaches',), fill_value=-9999.)
    rch_edit_flag._Encoding = 'ascii'
    rch_trib_flag = rch_grp.createVariable(
        'trib_flag', 'i4', ('num_reaches',), fill_value=-9999.)
    # subgroup 1 - 'area_fits'
    h_break = sub_grp1.createVariable(
        'h_break', 'f8', ('num_domains','num_reaches'), fill_value=-9999.)
    h_break.units = 'meters'
    w_break = sub_grp1.createVariable(
        'w_break', 'f8', ('num_domains','num_reaches'), fill_value=-9999.)
    w_break.units = 'meters'
    h_variance = sub_grp1.createVariable(
        'h_variance', 'f8', ('num_reaches',), fill_value=-9999.)
    h_variance.units = 'meters^2'
    w_variance = sub_grp1.createVariable(
        'w_variance', 'f8', ('num_reaches',), fill_value=-9999.)
    w_variance.units = 'meters^2'
    hw_covariance = sub_grp1.createVariable(
        'hw_covariance', 'f8', ('num_reaches',), fill_value=-9999.)
    hw_covariance.units = 'meters^2'
    h_err_stdev = sub_grp1.createVariable(
        'h_err_stdev', 'f8', ('num_reaches',), fill_value=-9999.)
    h_err_stdev.units = 'meters'
    w_err_stdev = sub_grp1.createVariable(
        'w_err_stdev', 'f8', ('num_reaches',), fill_value=-9999.)
    w_err_stdev.units = 'meters'
    h_w_nobs = sub_grp1.createVariable(
        'h_w_nobs', 'f8', ('num_reaches',), fill_value=-9999.)
    fit_coeffs = sub_grp1.createVariable(
        'fit_coeffs', 'f8', ('nCoeffs','nReg','num_reaches'), fill_value=-9999.)
    med_flow_area = sub_grp1.createVariable(
        'med_flow_area', 'f8', ('num_reaches',), fill_value=-9999.)
    # unconstrained discharge subgroups
    # MetroMan (ucmod1)
    uc_metroman_Abar = ucmod1.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_Abar.units = 'meters'
    uc_metroman_Abar_stdev = ucmod1.createVariable(
        'Abar_stdev', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_Abar_stdev.units = 'meters'
    uc_metroman_ninf = ucmod1.createVariable(
        'ninf', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_ninf_stdev = ucmod1.createVariable(
        'ninf_stdev', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_p = ucmod1.createVariable(
        'p', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_p_stdev = ucmod1.createVariable(
        'p_stdev', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_ninf_p_cor = ucmod1.createVariable(
        'ninf_p_cor', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_p_Abar_cor = ucmod1.createVariable(
        'p_Abar_cor', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_ninf_Abar_cor = ucmod1.createVariable(
        'ninf_Abar_cor', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_sbQ_rel = ucmod1.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # BAM (ucmod2)
    uc_bam_Abar = ucmod2.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_bam_Abar.units = 'meters'
    uc_bam_n = ucmod2.createVariable(
        'n', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_bam_sbQ_rel = ucmod2.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # HiVDI (ucmod3)
    uc_hivdi_Abar = ucmod3.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_hivdi_Abar.units = 'meters'
    uc_hivdi_alpha = ucmod3.createVariable(
        'alpha', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_hivdi_beta = ucmod3.createVariable(
        'beta', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_hivdi_sbQ_rel = ucmod3.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # MOMMA (ucmod4)
    uc_momma_B = ucmod4.createVariable(
        'B', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_momma_B.units = 'meters'
    uc_momma_H = ucmod4.createVariable(
        'H', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_momma_H.units = 'meters'
    uc_momma_Save = ucmod4.createVariable(
        'Save', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_momma_Save.units = 'meters/kilometers'
    uc_momma_sbQ_rel = ucmod4.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # SADS (ucmod5)
    uc_sads_Abar = ucmod5.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_sads_Abar.units = 'meters'
    uc_sads_n = ucmod5.createVariable(
        'n', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_sads_sbQ_rel = ucmod5.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # SIC4DVar (ucmod6)
    uc_sic4d_Abar = ucmod6.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_sic4d_Abar.units = 'meters'
    uc_sic4d_n = ucmod6.createVariable(
        'n', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_sic4d_sbQ_rel = ucmod6.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # constrained discharge subgroups
    # MetroMan (cmod1)
    c_metroman_Abar = cmod1.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_Abar.units = 'meters'
    c_metroman_Abar_stdev = cmod1.createVariable(
        'Abar_stdev', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_Abar_stdev.units = 'meters'
    c_metroman_ninf = cmod1.createVariable(
        'ninf', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_ninf_stdev = cmod1.createVariable(
        'ninf_stdev', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_p = cmod1.createVariable(
        'p', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_p_stdev = cmod1.createVariable(
        'p_stdev', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_ninf_p_cor = cmod1.createVariable(
        'ninf_p_cor', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_p_Abar_cor = cmod1.createVariable(
        'p_Abar_cor', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_ninf_Abar_cor = cmod1.createVariable(
        'ninf_Abar_cor', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_sbQ_rel = cmod1.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # BAM (cmod2)
    c_bam_Abar = cmod2.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    c_bam_Abar.units = 'meters'
    c_bam_n = cmod2.createVariable(
        'n', 'f8', ('num_reaches',), fill_value=-9999.)
    c_bam_sbQ_rel = cmod2.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # HiDVI (cmod3)
    c_hivdi_Abar = cmod3.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    c_hivdi_Abar.units = 'meters'
    c_hivdi_alpha = cmod3.createVariable(
        'alpha', 'f8', ('num_reaches',), fill_value=-9999.)
    c_hivdi_beta = cmod3.createVariable(
        'beta', 'f8', ('num_reaches',), fill_value=-9999.)
    c_hivdi_sbQ_rel = cmod3.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # MOMMA (cmod4)
    c_momma_B = cmod4.createVariable(
        'B', 'f8', ('num_reaches',), fill_value=-9999.)
    c_momma_B.units = 'meters'
    c_momma_H = cmod4.createVariable(
        'H', 'f8', ('num_reaches',), fill_value=-9999.)
    c_momma_H.units = 'meters'
    c_momma_Save = cmod4.createVariable(
        'Save', 'f8', ('num_reaches',), fill_value=-9999.)
    c_momma_Save.units = 'meters/kilometers'
    c_momma_sbQ_rel = cmod4.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # SADS (cmod5)
    c_sads_Abar = cmod5.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    c_sads_Abar.units = 'meters'
    c_sads_n = cmod5.createVariable(
        'n', 'f8', ('num_reaches',), fill_value=-9999.)
    c_sads_sbQ_rel = cmod5.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # SIC4DVar (cmod6)
    c_sic4d_Abar = cmod6.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    c_sic4d_Abar.units = 'meters'
    c_sic4d_n = cmod6.createVariable(
        'n', 'f8', ('num_reaches',), fill_value=-9999.)
    c_sic4d_sbQ_rel = cmod6.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)

    # saving data
    print("saving nc")

    # root group data
    #cont_str = nc.stringtochar(np.array(['NA'], 'S2'))
    #Name[:] = cont_str

    # centerline data
    cl_id[:] = centerlines.cl_id
    cl_x[:] = centerlines.x
    cl_y[:] = centerlines.y
    reach_id[:,:] = centerlines.reach_id
    node_id[:,:] = centerlines.node_id

    # node data
    Node_ID[:] = nodes.id
    node_cl_id[:,:] = nodes.cl_id
    node_x[:] = nodes.x
    node_y[:] = nodes.y
    node_len[:] = nodes.len
    node_rch_id[:] = nodes.reach_id
    node_wse[:] = nodes.wse
    node_wse_var[:] = nodes.wse_var
    node_wth[:] = nodes.wth
    node_wth_var[:] = nodes.wth_var
    node_chan_max[:] = nodes.nchan_max
    node_chan_mod[:] = nodes.nchan_mod
    node_grod_id[:] = nodes.grod
    node_grod_fid[:] = nodes.grod_fid
    node_hfalls_fid[:] = nodes.hfalls_fid
    node_dist_out[:] = nodes.dist_out
    node_wth_coef[:] = nodes.wth_coef
    node_ext_dist_coef[:] = nodes.ext_dist_coef
    node_facc[:] = nodes.facc
    node_lakeflag[:] = nodes.lakeflag
    #node_lake_id[:] = nodes.lake_id
    node_max_wth[:] = nodes.max_wth
    node_meand_len[:] = nodes.meand_len
    node_sinuosity[:] = nodes.sinuosity
    node_river_name[:] = nodes.river_name
    node_manual_add[:] = nodes.manual_add
    node_edit_flag[:] = nodes.edit_flag
    node_trib_flag[:] = nodes.trib_flag

    # reach data
    Reach_ID[:] = reaches.id
    rch_cl_id[:,:] = reaches.cl_id
    rch_x[:] = reaches.x
    rch_x_min[:] = reaches.x_min
    rch_x_max[:] = reaches.x_max
    rch_y[:] = reaches.y
    rch_y_min[:] = reaches.y_min
    rch_y_max[:] = reaches.y_max
    rch_len[:] = reaches.len
    num_nodes[:] = reaches.rch_n_nodes
    rch_wse[:] = reaches.wse
    rch_wse_var[:] = reaches.wse_var
    rch_wth[:] = reaches.wth
    rch_wth_var[:] = reaches.wth_var
    rch_facc[:] = reaches.facc
    rch_chan_max[:] = reaches.nchan_max
    rch_chan_mod[:] = reaches.nchan_mod
    rch_grod_id[:] = reaches.grod
    rch_grod_fid[:] = reaches.grod_fid
    rch_hfalls_fid[:] = reaches.hfalls_fid
    rch_slope[:] = reaches.slope
    rch_dist_out[:] = reaches.dist_out
    n_rch_up[:] = reaches.n_rch_up
    n_rch_down[:] = reaches.n_rch_down
    rch_id_up[:,:] = reaches.rch_id_up
    rch_id_down[:,:] = reaches.rch_id_down
    rch_lakeflag[:] = reaches.lakeflag
    rch_iceflag[:,:] = reaches.iceflag
    #rch_lake_id[:] = reaches.lake_id
    rch_swot_obs[:] = reaches.max_obs
    rch_orbits[:,:] = reaches.orbits
    rch_river_name[:] = reaches.river_name
    rch_max_wth[:] = reaches.max_wth
    rch_low_slope[:] = reaches.low_slope
    rch_edit_flag[:] = reaches.edit_flag
    rch_trib_flag[:] = reaches.trib_flag
    # subgroup1 - area fits
    h_break[:,:] = reaches.h_break
    w_break[:,:] = reaches.w_break
    h_variance[:] = reaches.wse_var
    w_variance[:] = reaches.wth_var
    hw_covariance[:] = reaches.hw_covariance
    h_err_stdev[:] = reaches.h_err_stdev
    w_err_stdev[:] = reaches.w_err_stdev
    h_w_nobs[:] = reaches.h_w_nobs
    fit_coeffs[:,:,:] = reaches.fit_coeffs
    med_flow_area[:] = reaches.med_flow_area
    # ucmod1
    uc_metroman_Abar[:] = reaches.metroman_abar
    uc_metroman_ninf[:] = reaches.metroman_ninf
    uc_metroman_p[:] = reaches.metroman_p
    uc_metroman_Abar_stdev[:] = reaches.metroman_abar_stdev
    uc_metroman_ninf_stdev[:] = reaches.metroman_ninf_stdev
    uc_metroman_p_stdev[:] = reaches.metroman_p_stdev
    uc_metroman_ninf_p_cor[:] = reaches.metroman_ninf_p_cor
    uc_metroman_ninf_Abar_cor[:] = reaches.metroman_ninf_abar_cor
    uc_metroman_p_Abar_cor[:] = reaches.metroman_p_abar_cor
    uc_metroman_sbQ_rel[:] = reaches.metroman_sbQ_rel
    # ucmod2
    uc_bam_Abar[:] = reaches.bam_abar
    uc_bam_n[:] = reaches.bam_n
    uc_bam_sbQ_rel[:] = reaches.bam_sbQ_rel
    # ucmod3
    uc_hivdi_Abar[:] = reaches.hivdi_abar
    uc_hivdi_alpha[:] = reaches.hivdi_alpha
    uc_hivdi_beta[:] = reaches.hivdi_beta
    uc_hivdi_sbQ_rel[:] = reaches.hivdi_sbQ_rel
    # ucmod4
    uc_momma_B[:] = reaches.momma_b
    uc_momma_H[:] = reaches.momma_h
    uc_momma_Save[:] = reaches.momma_save
    uc_momma_sbQ_rel[:] = reaches.momma_sbQ_rel
    # ucmod5
    uc_sads_Abar[:] = reaches.sads_abar
    uc_sads_n[:] = reaches.sads_n
    uc_sads_sbQ_rel[:] = reaches.sads_sbQ_rel
    # ucmod6
    uc_sic4d_Abar[:] = reaches.sic4d_abar
    uc_sic4d_n[:] = reaches.sic4d_n
    uc_sic4d_sbQ_rel[:] = reaches.sic4d_sbQ_rel
    # cmod1
    c_metroman_Abar[:] = reaches.metroman_abar
    c_metroman_ninf[:] = reaches.metroman_ninf
    c_metroman_p[:] = reaches.metroman_p
    c_metroman_Abar_stdev[:] = reaches.metroman_abar_stdev
    c_metroman_ninf_stdev[:] = reaches.metroman_ninf_stdev
    c_metroman_p_stdev[:] = reaches.metroman_p_stdev
    c_metroman_ninf_p_cor[:] = reaches.metroman_ninf_p_cor
    c_metroman_ninf_Abar_cor[:] = reaches.metroman_ninf_abar_cor
    c_metroman_p_Abar_cor[:] = reaches.metroman_p_abar_cor
    c_metroman_sbQ_rel[:] = reaches.metroman_sbQ_rel
    # cmod2
    c_bam_Abar[:] = reaches.bam_abar
    c_bam_n[:] = reaches.bam_n
    c_bam_sbQ_rel[:] = reaches.bam_sbQ_rel
    # cmod3
    c_hivdi_Abar[:] = reaches.hivdi_abar
    c_hivdi_alpha[:] = reaches.hivdi_alpha
    c_hivdi_beta[:] = reaches.hivdi_beta
    c_hivdi_sbQ_rel[:] = reaches.hivdi_sbQ_rel
    # cmod4
    c_momma_B[:] = reaches.momma_b
    c_momma_H[:] = reaches.momma_h
    c_momma_Save[:] = reaches.momma_save
    c_momma_sbQ_rel[:] = reaches.momma_sbQ_rel
    # cmod5
    c_sads_Abar[:] = reaches.sads_abar
    c_sads_n[:] = reaches.sads_n
    c_sads_sbQ_rel[:] = reaches.sads_sbQ_rel
    # cmod6
    c_sic4d_Abar[:] = reaches.sic4d_abar
    c_sic4d_n[:] = reaches.sic4d_n
    c_sic4d_sbQ_rel[:] = reaches.sic4d_sbQ_rel

    root_grp.close()

    end = time.time()

    print("Ended Saving Main NetCDF in: ", str(np.round((end-start)/60, 2)), " min")

    return outfile

###############################################################################