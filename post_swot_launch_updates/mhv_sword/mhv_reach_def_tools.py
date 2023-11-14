from __future__ import division
import utm
import numpy as np
from scipy import spatial as sp
import netCDF4 as nc
from pyproj import Proj

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
    data.tile = new.groups['centerlines'].variables['mh_tile'][:]
    data.eps = new.groups['centerlines'].variables['endpoints'][:]
    data.lake_id = new.groups['centerlines'].variables['lake_id'][:]
    data.strorder = new.groups['centerlines'].variables['strmorder'][:]
    new.close()

    return data

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
        bound1 = int(np.where(dist == lower_dist)[0][0])
        bound2 = int(np.where(dist == upper_dist)[0][0])
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
        # segs = np.unique(subcls.seg[rch])
        new_ind = np.zeros(len(rch))
        eps = np.zeros(len(rch))
        rch_dist = calc_segDist(rch_lon, rch_lat, new_rch_id[rch], subcls.facc[rch], subcls.ind[rch])

        ### added on 11/14/2023 to try and fix issue with short segments that have overlapping points. 
        if len(np.unique(rch_dist)) != len(rch_dist):
            unq_ind = np.unique(rch_dist, return_index=True)[1]
            segs = np.unique(subcls.seg[rch][unq_ind])
        else:
            segs = np.unique(subcls.seg[rch])

        # Reformat indexes if multiple segments are within a reach.
        if len(segs) > 1:
            # print(ind)
            # break
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
            if len(rch_segs) < 4: #use to be 5.
                pt_dist, pt_ind = kdt.query(ep_pts, k = len(rch_segs)) 
            else:
                pt_dist, pt_ind = kdt.query(ep_pts, k = 4) 

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
                print(uniq_rch[ind], 'ind =', ind, len(final_eps), 'problem with indexes')
                # break

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