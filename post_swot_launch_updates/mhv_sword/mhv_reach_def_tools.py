from __future__ import division
import utm
import numpy as np
from scipy import spatial as sp
import netCDF4 as nc
from pyproj import Proj

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