from __future__ import division
import utm
import numpy as np
from scipy import spatial as sp
import netCDF4 as nc
from pyproj import Proj
import math
from geopy import distance

###############################################################################

class Object(object):
    """
    FUNCTION:
        Creates class object to assign attributes to.
    """
    pass

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
    data.x = np.array(new.groups['centerlines'].variables['easting'][:])
    data.y = np.array(new.groups['centerlines'].variables['northing'][:])
    data.seg = np.array(new.groups['centerlines'].variables['new_segs'][:]) #use to be segID
    data.ind = np.array(new.groups['centerlines'].variables['new_segs_ind'][:]) #use to be segInd
    data.id = np.array(new.groups['centerlines'].variables['cl_id'][:])
    data.dist = np.array(new.groups['centerlines'].variables['new_segDist'][:]) #use to be segDist
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
    data.tile = np.array(new.groups['centerlines'].variables['mh_tile'][:])
    data.eps = np.array(new.groups['centerlines'].variables['endpoints'][:])
    data.lake_id = np.array(new.groups['centerlines'].variables['lake_id'][:])
    data.strorder = np.array(new.groups['centerlines'].variables['strmorder'][:])
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

def get_distances(lon,lat):
    traces = len(lon) -1
    distances = np.zeros(traces)
    for i in range(traces):
        start = (lat[i], lon[i])
        finish = (lat[i+1], lon[i+1])
        distances[i] = distance.geodesic(start, finish).m
    distances = np.append(0,distances)
    return distances

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
        sort_ind = np.argsort(subcls_rch_ind[seg])
        seg_lon = subcls_lon[seg[sort_ind]]
        seg_lat = subcls_lat[seg[sort_ind]]
        # upa = subcls_facc[seg[sort_ind]]

        #segment distance
        x_coords = seg_lon
        y_coords = seg_lat
        diff = get_distances(x_coords,y_coords)
        dist = np.cumsum(diff)

        segDist[seg[sort_ind]] = dist
        
        #format flow distance as array and determine flow direction by flowacc.
        # start = upa[np.where(dist == np.min(dist))[0][0]]
        # end = upa[np.where(dist == np.max(dist))[0][0]]

        # if end > start:
        #     segDist[seg] = abs(dist-np.max(dist))

        # else:
        #     segDist[seg] = dist

    return segDist

###############################################################################

def find_boundaries(subcls):

    # Combine lake and delta flags into one vector.
    lake_coast_flag = np.copy(subcls.lake)
    lake_coast_flag[np.where(subcls.delta > 0)] = 3

    # Set variables.
    reach_nums = np.zeros(len(subcls.ind))
    Type = np.repeat(1,len(subcls.ind))
    Len = np.zeros(len(subcls.ind))
    cnt = 1

    # Loop through each basin and identify SWOT orbit, lake, and dam boundaries.
    uniq_basins = np.unique(subcls.basins)
    for ind in list(range(len(uniq_basins))):
        basin = np.where(subcls.basins == uniq_basins[ind])[0]
        uniq_segs = np.unique(subcls.seg[basin])
        for idx in list(range(len(uniq_segs))):
            seg = np.where(subcls.seg[basin] == uniq_segs[idx])[0]
            sort_ids = np.argsort(subcls.ind[basin[seg]])
            ID = subcls.ind[basin[seg[sort_ids]]]
            dist = subcls.dist[basin[seg[sort_ids]]]
            lakes = lake_coast_flag[basin[seg[sort_ids]]]
            grod = subcls.grod[basin[seg[sort_ids]]]
            dams = np.where((grod > 0) & (grod <= 4))[0]

            # Find lake and dam boundaries.
            bounds = []
            if max(lakes) > 0:
                # print(idx)
                bounds.extend(np.where(np.diff(lakes) != 0)[0])

            if len(dams) > 0:
                # print(idx)
                for d in list(range(len(dams))):
                    if dams[d] < 3:
                        bounds.extend(np.array([5])) #first 5ish points are a dam
                    if dams[d] > len(ID)-4:
                        bounds.extend(np.array([len(ID) - 6])) #last 5ish points are a dam 
                    else:
                        b1 = np.array([dams[d] - 2])
                        b2 = np.array([dams[d] + 2])
                        bounds.extend(b1)
                        bounds.extend(b2)
            
            # Account for odd basin boundaries
            basin_breaks = np.where(np.diff(dist) > 250)[0]
            if len(basin_breaks) > 0:
                bounds.extend(basin_breaks+1)

            if len(bounds) > 0:
                bounds.extend(np.where(ID == np.min(ID))[0])
                bounds.extend(np.where(ID == np.max(ID))[0])
                bounds = np.sort(bounds)
                ### number between boundaries
                for b in list(range(len(bounds)-1)):
                    reach_nums[basin[seg[sort_ids[bounds[b]:bounds[b+1]+1]]]] = cnt 
                    cnt = cnt+1
                    # reach_nums[basin[seg[sort_ids]]]
            else:
                reach_nums[basin[seg[sort_ids]]] = cnt 
                cnt = cnt+1

            # Create reach "type" flag based on boundaries.
            unq_rchs = np.unique(reach_nums[basin[seg[sort_ids]]])
            for r in list(range(len(unq_rchs))):
                rind = np.where(reach_nums[basin[seg[sort_ids]]] == unq_rchs[r])[0]
                Len[basin[seg[sort_ids[rind]]]] = max(dist[rind]) - min(dist[rind])
                if max(lakes[rind]) > 0:
                    Type[basin[seg[sort_ids[rind]]]] = 3
                if max(lakes[rind]) == 3:
                    Type[basin[seg[sort_ids[rind]]]] = 5
                if max(grod[rind]) > 0:
                    Type[basin[seg[sort_ids[rind]]]] = 4

    return(reach_nums, Type, Len)

###############################################################################

def cut_reaches(subcls_rch_id0, subcls_rch_len0, subcls_dist,
                subcls_ind, max_dist):

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
        sort_ids = np.argsort(subcls_ind[rch])
        distance = subcls_dist[rch[sort_ids]]
        ID = subcls_ind[rch[sort_ids]]
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
        borders = np.sort(borders)

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
                temp_rch_dist[vals] = 90.0
            else:
                temp_rch_dist[vals] = avg_dist

            temp_rch_id[vals] = cnt
            cnt=cnt+1

        new_rch_id[rch[sort_ids]] = temp_rch_id
        new_rch_dist[rch[sort_ids]] = temp_rch_dist
        #if np.max(new_rch_dist[rch])>max_dist:
            #print(ind, 'max distance too long - likely an index problem')

    return new_rch_id, new_rch_dist

###############################################################################

def find_neighbors(basin_rch, basin_dist, basin_flag, basin_acc, basin_wse,
                   basin_x, basin_y, rch_x, rch_y, rch_ind, rch_id, rch):

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
    #getting radius for finding neighbors.
    radius = meters_to_degrees(128, np.median(basin_y))
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

    #for grwl the values were 100 and 200 
    if len(rch) < 10:
        pt_dist, pt_ind = kdt.query(ep_pts, k = 6, distance_upper_bound = radius)
    else:
        pt_dist, pt_ind = kdt.query(ep_pts, k = 10, distance_upper_bound = radius)

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
##### TAKE 3

def update_rch_indexes(subcls, new_rch_id):
    # Set variables and find unique reaches.
    uniq_rch = np.unique(new_rch_id)
    new_rch_ind = np.zeros(len(subcls.ind))
    new_rch_eps = np.zeros(len(subcls.ind))

    # Loop through each reach and re-order indexes.
    for ind in list(range(len(uniq_rch))):
        rch = np.where(new_rch_id == uniq_rch[ind])[0]
        sort_ids = np.argsort(subcls.ind[rch])
        rch_ind = subcls.ind[rch[sort_ids]]
        rch_segs = subcls.seg[rch[sort_ids]]
        unq_segs = np.unique(rch_segs) #subcls.seg[rch[index]]
        if len(unq_segs) == 1:
            new_rch_ind[rch[sort_ids]] = subcls.ind[rch[sort_ids]]
            ep1 = np.where(subcls.ind[rch[sort_ids]] == np.min(subcls.ind[rch[sort_ids]]))[0]
            ep2 = np.where(subcls.ind[rch[sort_ids]] == np.max(subcls.ind[rch[sort_ids]]))[0]
            new_rch_eps[rch[sort_ids[ep1]]] = 1
            new_rch_eps[rch[sort_ids[ep2]]] = 1
            #reverse index order to have indexes increasing in the upstream direction.
            # if subcls.facc[rch[ep1]] < subcls.facc[rch[ep2]]:
            #     new_rch_ind[rch] = abs(new_rch_ind[rch] - np.max(new_rch_ind[rch]))   
        else:
            print(ind, len(unq_segs))
            rch_ind_temp = np.zeros(len(rch))
            for r in list(range(len(unq_segs))):
                seg = np.where(rch_segs == unq_segs[r])[0]
                rch_ind_temp[seg] = rch_ind[seg] - np.min(rch_ind[seg]-1)
                  
            diff = np.diff(rch_ind_temp)
            divs = np.where(diff != 1)[0]
            start_add = rch_ind_temp[divs[0]]
            for d in list(range(len(divs))):
                if d+1 >= len(divs):
                    new_ind = rch_ind_temp[divs[d]+1::]+start_add
                    rch_ind_temp[divs[d]+1::] = new_ind
                else:
                    new_ind = rch_ind_temp[divs[d]+1:divs[d+1]+1]+start_add
                    rch_ind_temp[divs[d]+1:divs[d+1]+1] = new_ind
                start_add = np.max(new_ind)
                
            new_rch_ind[rch] = rch_ind_temp    
            mn = np.where(rch_ind_temp == np.min(rch_ind_temp))[0]
            mx = np.where(rch_ind_temp == np.max(rch_ind_temp))[0]
            new_rch_eps[rch[mn]] = 1 
            new_rch_eps[rch[mx]] = 1  

            check = np.where(np.diff(rch_ind_temp) < 1)[0]
            if len(check) > 0:
                print(ind, uniq_rch[ind], 'problem with indexes')

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
    # level4 = np.array([int(str(point)[0:4]) for point in subcls.basins])
    # uniq_basins = np.unique(level4) #np.unique(subcls.basins)
    
    # on 1/24/2025 I changed the scale for aggregation to the segment level from 
    # the basin level to try and cut down on index issues. This is only for 
    # mhv based reaches.
    uniq_basins = np.unique(subcls.seg)
    for ind in list(range(len(uniq_basins))):
        # print(ind, 'BASIN = ', uniq_basins[ind])
        # basin = np.where(level4 == uniq_basins[ind])[0]
        basin = np.where(subcls.seg == uniq_basins[ind])[0]
        sort_ids = np.argsort(subcls.ind[basin])
        basin_l6 =  subcls.basins[basin[sort_ids]]
        basin_rch = subcls.rch_id1[basin[sort_ids]]
        basin_dist = subcls.rch_len1[basin[sort_ids]]
        basin_flag = subcls.type1[basin[sort_ids]]
        basin_acc = subcls.facc[basin[sort_ids]]
        basin_wse = subcls.elv[basin[sort_ids]]
        basin_lon = subcls.lon[basin[sort_ids]]
        basin_lat = subcls.lat[basin[sort_ids]]
        basin_ind = subcls.ind[basin[sort_ids]]
        # basin_x, basin_y, __, __ = reproject_utm(basin_lat, basin_lon)

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
                rch_lat = basin_lat[rch]
                rch_lon = basin_lon[rch]
                # rch_x = basin_x[rch]
                # rch_y = basin_y[rch]
                rch_ind = basin_ind[rch]
                end1, end2 = find_neighbors(basin_rch, basin_dist, basin_flag,
                                            basin_acc, basin_wse, basin_lon, basin_lat, 
                                            rch_lon, rch_lat, rch_ind, rch_id, rch)

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
        new_rch_id[basin[sort_ids]] = basin_rch
        new_rch_dist[basin[sort_ids]] = basin_dist
        new_flag[basin[sort_ids]] = basin_flag

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
    # level4 = np.array([int(str(point)[0:4]) for point in subcls.basins])
    # uniq_basins = np.unique(level4) #np.unique(subcls.basins)
    
    # on 1/24/2025 I changed the scale for aggregation to the segment level from 
    # the basin level to try and cut down on index issues. This is only for 
    # mhv based reaches.
    uniq_basins = np.unique(subcls.seg)
    for ind in list(range(len(uniq_basins))):
        #print(ind)
        # basin = np.where(level4 == uniq_basins[ind])[0]
        basin = np.where(subcls.seg == uniq_basins[ind])[0]
        sort_ids = np.argsort(subcls.ind[basin])
        basin_l6 =  subcls.basins[basin[sort_ids]]
        basin_rch = subcls.rch_id2[basin[sort_ids]]
        basin_dist = subcls.rch_len2[basin[sort_ids]]
        basin_flag = subcls.type2[basin[sort_ids]]
        basin_acc = subcls.facc[basin[sort_ids]]
        basin_wse = subcls.elv[basin[sort_ids]]
        basin_lon = subcls.lon[basin[sort_ids]]
        basin_lat = subcls.lat[basin[sort_ids]]
        basin_ind = subcls.ind[basin[sort_ids]]
        # basin_x, basin_y, __, __ = reproject_utm(basin_lat, basin_lon)

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
                rch_lat = basin_lat[rch]
                rch_lon = basin_lon[rch]
                # rch_x = basin_x[rch]
                # rch_y = basin_y[rch]
                rch_ind = basin_ind[rch]
                end1, end2 = find_neighbors(basin_rch, basin_dist, basin_flag,
                                            basin_acc, basin_wse, basin_lon, basin_lat, 
                                            rch_lon, rch_lat, rch_ind, rch_id, rch)

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
        new_rch_id[basin[sort_ids]] = basin_rch
        new_rch_dist[basin[sort_ids]] = basin_dist
        new_flag[basin[sort_ids]] = basin_flag

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
    # level4 = np.array([int(str(point)[0:4]) for point in subcls.basins])
    # uniq_basins = np.unique(level4) #np.unique(subcls.basins)
    
    # on 1/24/2025 I changed the scale for aggregation to the segment level from 
    # the basin level to try and cut down on index issues. This is only for 
    # mhv based reaches.
    uniq_basins = np.unique(subcls.seg)
    for ind in list(range(len(uniq_basins))):
        #print(ind)
        # basin = np.where(level4 == uniq_basins[ind])[0]
        basin = np.where(subcls.seg == uniq_basins[ind])[0]
        sort_ids = np.argsort(subcls.ind[basin])
        basin_l6 =  subcls.basins[basin[sort_ids]]
        basin_rch = subcls.rch_id3[basin[sort_ids]]
        basin_dist = subcls.rch_len3[basin[sort_ids]]
        basin_flag = subcls.type3[basin[sort_ids]]
        basin_acc = subcls.facc[basin[sort_ids]]
        basin_wse = subcls.elv[basin[sort_ids]]
        basin_lon = subcls.lon[basin[sort_ids]]
        basin_lat = subcls.lat[basin[sort_ids]]
        basin_ind = subcls.ind[basin[sort_ids]]
        # basin_x, basin_y, __, __ = reproject_utm(basin_lat, basin_lon)

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
                rch_lat = basin_lat[rch]
                rch_lon = basin_lon[rch]
                # rch_x = basin_x[rch]
                # rch_y = basin_y[rch]
                rch_ind = basin_ind[rch]
                end1, end2 = find_neighbors(basin_rch, basin_dist, basin_flag,
                                            basin_acc, basin_wse, basin_lon, basin_lat, 
                                            rch_lon, rch_lat, rch_ind, rch_id, rch)

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
        new_rch_id[basin[sort_ids]] = basin_rch
        new_rch_dist[basin[sort_ids]] = basin_dist
        new_flag[basin[sort_ids]] = basin_flag

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
        rch_lon = basin_lon[rch]
        rch_lat = basin_lat[rch]
        rch_len = np.unique(basin_dist[rch])
        rch_acc = np.max(basin_acc[rch])
        rch_wse = np.min(basin_wse[rch])

        # Find upstream neighboring reaches.
        ngh1, ngh2 = find_neighbors(basin_rch, basin_dist, basin_flag, basin_acc,
                          basin_wse, basin_lon, basin_lat, rch_lon, rch_lat, rch_ind,
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
    reach_id = np.zeros(len(rch_topo), dtype = int)
    for ind in list(range(len(rch_topo))):
        if len(str(int(rch_topo[ind]))) == 1:
            fill = '000'
            reach_id[ind] = int(str(subcls.basins[ind])+fill+str(int(rch_topo[ind]))+str(int(subcls.type5[ind])))
        if len(str(int(rch_topo[ind]))) == 2:
            fill = '00'
            reach_id[ind] = int(str(subcls.basins[ind])+fill+str(int(rch_topo[ind]))+str(int(subcls.type5[ind])))
        if len(str(int(rch_topo[ind]))) == 3:
            fill = '0'
            reach_id[ind] = int(str(subcls.basins[ind])+fill+str(int(rch_topo[ind]))+str(int(subcls.type5[ind])))
        if len(str(int(rch_topo[ind]))) == 4:
            reach_id[ind] = int(str(subcls.basins[ind])+str(int(rch_topo[ind]))+str(int(subcls.type5[ind])))

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
    node_id = np.zeros(len(subcls.rch_id5), dtype = int)
    uniq_rch = np.unique(subcls.rch_id5)

    # Loop through each reach and divide it up based on the specified node
    # length, then number the nodes in order of flow accumulation.
    for ind in list(range(len(uniq_rch))):

        # Current reach information.
        rch = np.where(subcls.rch_id5 == uniq_rch[ind])[0]
        sort_ids = np.argsort(subcls.rch_ind5[rch])
        distance = subcls.rch_dist5[rch[sort_ids]]

        ID = subcls.rch_ind5[rch[sort_ids]]

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
            node_num[rch[sort_ids]] = abs(temp_node_id - np.max(temp_node_id))+1
            node_dist[rch[sort_ids]] = temp_node_dist
        else:
            node_num[rch[sort_ids]] = temp_node_id
            node_dist[rch[sort_ids]] = temp_node_dist

        #if np.max(node_dist[rch])>node_len*2:
            #print(ind, 'max distance too long - likely an index problem')

        # Create formal Node ID.
        for inz in list(range(len(rch))):
            #if len(str(int(node_num[rch[inz]]))) > 3:
                #print(ind)
            if len(str(int(node_num[rch[sort_ids[inz]]]))) == 1:
                fill = '00'
                node_id[rch[sort_ids[inz]]] = int(str(subcls.reach_id[rch[sort_ids[inz]]])[:-1]+fill+str(int(node_num[rch[sort_ids[inz]]]))+str(subcls.reach_id[rch[sort_ids[inz]]])[10:11])
            if len(str(int(node_num[rch[sort_ids[inz]]]))) == 2:
                fill = '0'
                node_id[rch[sort_ids[inz]]] = int(str(subcls.reach_id[rch[sort_ids[inz]]])[:-1]+fill+str(int(node_num[rch[sort_ids[inz]]]))+str(subcls.reach_id[rch[sort_ids[inz]]])[10:11])
            if len(str(int(node_num[rch[sort_ids[inz]]]))) == 3:
                node_id[rch[sort_ids[inz]]] = int(str(subcls.reach_id[rch[sort_ids[inz]]])[:-1]+str(int(node_num[rch[sort_ids[inz]]]))+str(subcls.reach_id[rch[sort_ids[inz]]])[10:11])

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
    # x, y, __, __ = reproject_utm(subcls.lat, subcls.lon)
    x = np.copy(subcls.lon)
    y = np.copy(subcls.lat)
    rch_eps = np.where(subcls.rch_eps5 == 1)[0]
    all_pts = np.vstack((x, y)).T
    eps_pts = np.vstack((x[rch_eps], y[rch_eps])).T
    kdt = sp.cKDTree(all_pts)
    eps_dist, eps_ind = kdt.query(eps_pts, k = 5, distance_upper_bound = 0.005)
    #actual ghost node identification.
    ghost_dist = np.copy(eps_dist)
    #replacing duplicate points with neighbor distance
    ghost_dist[np.where(ghost_dist[:,1] == 0),2] = ghost_dist[np.where(ghost_dist[:,1] == 0),3]
    ghost_ids = eps_ind[np.where(ghost_dist[:,2] >= 0.0013)[0],0] #changed to 180 on 11/15/2023 for mhv. 
    #added to attempt to filter out unnecessary ghost nodes.
    ghost_pts = np.vstack((x[ghost_ids], y[ghost_ids])).T
    gst_dist, gst_ind = kdt.query(ghost_pts, k = 5, distance_upper_bound = 0.01)
    gst_dist[np.where(gst_dist[:,1] == 0),1] = gst_dist[np.where(gst_dist[:,1] == 0),2]
    rmv_ids = gst_ind[np.where(gst_dist[:,4] < 0.0013)[0]]
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
                #new_nodes[ghost[idx]] = int(str(new_nodes[ghost[idx]])[:-1]+str(6))
                #new_reaches[ghost[idx]] = int(str(new_reaches[ghost[idx]])[:-1]+str(6))

                #new reach id
                if len(str(rch_num)) == 1:
                    fill = '000'
                    new_reaches[ghost[idx]] = int(str(new_reaches[ghost[idx]])[0:6]+fill+str(rch_num)+str(6))
                if len(str(rch_num)) == 2:
                    fill = '00'
                    new_reaches[ghost[idx]] = int(str(new_reaches[ghost[idx]])[0:6]+fill+str(rch_num)+str(6))
                if len(str(rch_num)) == 3:
                    fill = '0'
                    new_reaches[ghost[idx]] = int(str(new_reaches[ghost[idx]])[0:6]+fill+str(rch_num)+str(6))
                if len(str(rch_num)) == 4:
                    new_reaches[ghost[idx]] = int(str(new_reaches[ghost[idx]])[0:6]+str(rch_num)+str(6))

                #new node id
                if len(str(nd_num)) == 1:
                    fill = '00'
                    new_nodes[ghost[idx]] = int(str(new_reaches[ghost[idx]])[:-1]+fill+str(nd_num)+str(6))
                if len(str(nd_num)) == 2:
                    fill = '0'
                    new_nodes[ghost[idx]] = int(str(new_reaches[ghost[idx]])[:-1]+fill+str(nd_num)+str(6))
                if len(str(nd_num)) == 3:
                    new_nodes[ghost[idx]] = int(str(new_reaches[ghost[idx]])[:-1]+str(nd_num)+str(6))

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

def update_netcdf(nc_file, centerlines):
    data = nc.Dataset(nc_file, 'r+')
    # check to see if variables have been created already. If not create them.
    if 'reach_id' in data.groups['centerlines'].variables:
        data.groups['centerlines'].variables['reach_id'][:] = centerlines.reach_id
        data.groups['centerlines'].variables['rch_len'][:] = centerlines.rch_len
        data.groups['centerlines'].variables['node_num'][:] = centerlines.node_num
        data.groups['centerlines'].variables['rch_eps'][:] = centerlines.rch_eps
        data.groups['centerlines'].variables['type'][:] = centerlines.type
        data.groups['centerlines'].variables['rch_ind'][:] = centerlines.rch_ind
        data.groups['centerlines'].variables['rch_num'][:] = centerlines.rch_num
        data.groups['centerlines'].variables['node_id'][:] = centerlines.node_id
        data.groups['centerlines'].variables['rch_dist'][:] = centerlines.rch_dist
        data.groups['centerlines'].variables['node_len'][:] = centerlines.node_len
        data.groups['centerlines'].variables['rch_issue_flag'][:] = centerlines.rch_flag
        data.close()
    else:
        # create variables. 
        data.groups['centerlines'].createVariable('reach_id', 'i8', ('num_points',))
        data.groups['centerlines'].createVariable('rch_len', 'f8', ('num_points',))
        data.groups['centerlines'].createVariable('node_num', 'i8', ('num_points',))
        data.groups['centerlines'].createVariable('rch_eps', 'i4', ('num_points',))
        data.groups['centerlines'].createVariable('type', 'i4', ('num_points',))
        data.groups['centerlines'].createVariable('rch_ind', 'i8', ('num_points',))
        data.groups['centerlines'].createVariable('rch_num', 'i8', ('num_points',))
        data.groups['centerlines'].createVariable('node_id', 'i8', ('num_points',))
        data.groups['centerlines'].createVariable('rch_dist', 'f8', ('num_points',))
        data.groups['centerlines'].createVariable('node_len', 'f8', ('num_points',))
        data.groups['centerlines'].createVariable('rch_issue_flag', 'i4', ('num_points',))
        # populate variables. 
        data.groups['centerlines'].variables['reach_id'][:] = centerlines.reach_id
        data.groups['centerlines'].variables['rch_len'][:] = centerlines.rch_len
        data.groups['centerlines'].variables['node_num'][:] = centerlines.node_num
        data.groups['centerlines'].variables['rch_eps'][:] = centerlines.rch_eps
        data.groups['centerlines'].variables['type'][:] = centerlines.type
        data.groups['centerlines'].variables['rch_ind'][:] = centerlines.rch_ind
        data.groups['centerlines'].variables['rch_num'][:] = centerlines.rch_num
        data.groups['centerlines'].variables['node_id'][:] = centerlines.node_id
        data.groups['centerlines'].variables['rch_dist'][:] = centerlines.rch_dist
        data.groups['centerlines'].variables['node_len'][:] = centerlines.node_len
        data.groups['centerlines'].variables['rch_issue_flag'][:] = centerlines.rch_flag
        data.close()

###############################################################################

def check_rchs(rch_id, dist, indexes):
    issues = []
    unq_rch = np.unique(rch_id)
    for ind in list(range(len(unq_rch))):
        rch = np.where(rch_id == unq_rch[ind])[0]
        sort_ids = rch[np.argsort(indexes[rch])]
        if len(rch) == 1:
            continue
        diff = np.abs(np.diff(dist[sort_ids]))
        if np.max(diff) > 1000:
            issues.append(unq_rch[ind])
    return issues

###############################################################################

def correct_rchs(subcls, issues):
    start_id = np.max(subcls.rch_id5) + 1
    for ind in list(range(len(issues))):
        rch = np.where(subcls.rch_id5 == issues[ind])[0]
        unq_segs = np.unique(subcls.seg[rch])
        for s in list(range(len(unq_segs))):
            seg = np.where(subcls.seg[rch] == unq_segs[s])[0]
            subcls.rch_id5[rch[seg]] = start_id
            subcls.rch_dist5[rch[seg]] = subcls.rch_dist5[rch[seg]]-np.min(subcls.rch_dist5[rch[seg]])
            min_dist = np.min(subcls.rch_dist5[rch[seg]])
            max_dist = np.max(subcls.rch_dist5[rch[seg]])
            if max_dist-min_dist < 0:
                print(ind)
            subcls.rch_len5[rch[seg]] = max_dist-min_dist
            start_id = start_id+1

    update = np.where(subcls.rch_len5 == 0)[0]
    subcls.rch_len5[update] = 90
            
###############################################################################