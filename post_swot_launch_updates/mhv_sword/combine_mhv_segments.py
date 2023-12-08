from __future__ import division
import time
import numpy as np
import argparse
from scipy import spatial as sp
import netCDF4 as nc
import utm
from pyproj import Proj

###############################################################################
################################# FUNCTIONS ###################################
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
    data.lon = np.array(new.groups['centerlines'].variables['x'][:])
    data.lat = np.array(new.groups['centerlines'].variables['y'][:])
    data.x = np.array(new.groups['centerlines'].variables['easting'][:])
    data.y = np.array(new.groups['centerlines'].variables['northing'][:])
    data.seg = np.array(new.groups['centerlines'].variables['segID'][:])
    data.ind = np.array(new.groups['centerlines'].variables['segInd'][:])
    data.id = np.array(new.groups['centerlines'].variables['cl_id'][:])
    data.segDist = np.array(new.groups['centerlines'].variables['segDist'][:])
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

def update_rch_indexes(subcls, new_rch_id):
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
        rch_segs = np.unique(subcls.seg[rch])
        rch_eps_all = np.zeros(len(rch))
        if len(rch_segs) == 1:
            new_rch_ind[rch] = subcls.ind[rch]
            ep1 = np.where(subcls.ind[rch] == np.min(subcls.ind[rch]))[0]
            ep2 = np.where(subcls.ind[rch] == np.max(subcls.ind[rch]))[0]
            new_rch_eps[rch[ep1]] = 1
            new_rch_eps[rch[ep2]] = 1
            #reverse index order to have indexes increasing in the upstream direction.
            if subcls.facc[rch[ep1]] < subcls.facc[rch[ep2]]:
                new_rch_ind[rch] = abs(new_rch_ind[rch] - np.max(new_rch_ind[rch]))
            
        else:
            for r in list(range(len(rch_segs))):
                seg = np.where(subcls.seg[rch] == rch_segs[r])[0]
                mn = np.where(subcls.ind[rch[seg]] == np.min(subcls.ind[rch[seg]]))[0]
                mx = np.where(subcls.ind[rch[seg]] == np.max(subcls.ind[rch[seg]]))[0]
                rch_eps_all[seg[mn]] = 1
                rch_eps_all[seg[mx]] = 1

            eps_ind = np.where(rch_eps_all>0)[0]
            ep_pts = np.vstack((rch_x[eps_ind], rch_y[eps_ind])).T
            kdt = sp.cKDTree(rch_pts)
            if len(rch) < 4: #use to be 5.
                pt_dist, pt_ind = kdt.query(ep_pts, k = len(rch_segs)) 
            else:
                pt_dist, pt_ind = kdt.query(ep_pts, k = 4)
            row_sums = np.sum(rch_eps_all[pt_ind], axis = 1)
            final_eps = np.where(row_sums == 1)[0]
            if len(final_eps) == 0:
                print(ind, uniq_rch[ind], len(rch), 'index issue - short reach')
                # final_eps = np.where(rch_eps_all == 1)[0]
                final_eps = np.array([0,len(rch)-1])

            elif len(final_eps) > 2:
                print(ind, uniq_rch[ind], len(rch), 'index issue - possible tributary')
                # break

            # Re-ordering points based on updated endpoints.
            new_ind = np.zeros(len(rch))
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

    return new_rch_ind, new_rch_eps

###############################################################################

def update_netcdf(nc_file, centerlines):
    data = nc.Dataset(nc_file, 'r+')
    # check to see if variables have been created already. If not create them.
    if 'new_segs' in data.groups['centerlines'].variables:
        data.groups['centerlines'].variables['new_segs'][:] = centerlines.new_segs
        data.groups['centerlines'].variables['new_segs_ind'][:] = centerlines.new_ind
        data.groups['centerlines'].variables['new_segs_dist'][:] = centerlines.new_dist
        data.groups['centerlines'].variables['new_segs_eps'][:] = centerlines.new_eps
        data.close()
    else:
        # create variables. 
        data.groups['centerlines'].createVariable('new_segs', 'i8', ('num_points',))
        data.groups['centerlines'].createVariable('new_segs_ind', 'i8', ('num_points',))
        data.groups['centerlines'].createVariable('new_segs_dist', 'f8', ('num_points',))
        data.groups['centerlines'].createVariable('new_segs_eps', 'i4', ('num_points',))
        # populate variables. 
        data.groups['centerlines'].variables['new_segs'][:] = centerlines.new_segs
        data.groups['centerlines'].variables['new_segs_ind'][:] = centerlines.new_ind
        data.groups['centerlines'].variables['new_segs_dist'][:] = centerlines.new_dist
        data.groups['centerlines'].variables['new_segs_eps'][:] = centerlines.new_eps
        data.close()

###############################################################################
##################### Defining Reach and Node Locations #######################
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Region", type = str)
parser.add_argument("local_processing", help="'True' for local machine, 'False' for server", type = str)
args = parser.parse_args()

start_all = time.time()
region = args.region

# Input file(s).
if args.local_processing == 'True':
    main_dir = '/Users/ealteanau/Documents/SWORD_Dev/inputs/'
else:
    main_dir = '/afs/cas.unc.edu/depts/geological_sciences/pavelsky/students/ealtenau/SWORD_dev/inputs/'
nc_file = main_dir+'MHV_SWORD/'+region+'_mhv_sword.nc'

# Reading in data.
data = read_merge_netcdf(nc_file)

# Making sure flow accumulation minimum isn't zero.
data.facc[np.where(data.facc == 0)[0]] = 0.001
# Cutting basins to 6 digits.
data.basins = np.array([int(str(ind)[0:6]) for ind in data.basins])

# Creating empty objects to fill with attributes.
centerlines = Object()
centerlines.new_segs = np.zeros(len(data.lon))
centerlines.new_ind = np.zeros(len(data.lon))
centerlines.new_eps = np.zeros(len(data.lon))
centerlines.new_dist = np.zeros(len(data.lon))

# Loop through each level 2 basin. Subsetting per level 2 basin speeds up the script.
level2_basins = np.array([int(str(ind)[0:2]) for ind in data.basins])
uniq_level2 = np.unique(level2_basins)
uniq_level2 = np.delete(uniq_level2, 0)
for ind in list(range(len(uniq_level2))):

    print('STARTING BASIN: ' + str(uniq_level2[ind]))

    start = time.time()

    # Define objects to assign attributes.
    subcls = Object()

    # Subset data.
    level2 = np.where(level2_basins == uniq_level2[ind])[0]
    subcls.id = data.id[level2]
    subcls.lon = data.lon[level2]
    subcls.lat = data.lat[level2]
    subcls.seg = data.seg[level2]
    subcls.ind = data.ind[level2]
    subcls.dist = data.segDist[level2]
    subcls.wth = data.wth[level2]
    subcls.elv = data.elv[level2]
    subcls.facc = data.facc[level2]
    subcls.lake = data.lake[level2]
    subcls.delta = data.delta[level2]
    subcls.nchan = data.nchan[level2]
    subcls.grand = data.grand[level2]
    subcls.grod = data.grod[level2]
    subcls.grod_fid = data.grod_fid[level2]
    subcls.hfalls_fid = data.hfalls_fid[level2]
    subcls.basins = data.basins[level2]
    subcls.manual = data.manual[level2]
    subcls.num_obs = data.num_obs[level2]
    subcls.orbits = data.orbits[level2]
    subcls.lake_id = data.lake_id[level2]
    subcls.strorder = data.strorder[level2]
    subcls.eps = data.eps[level2]
    subcls.lon[np.where(subcls.lon < -180)] = -180.0
    subcls.lon[np.where(subcls.lon > 180)] = 180.0
    subcls.x = np.copy(subcls.lon)
    subcls.y = np.copy(subcls.lat)

    all_pts = np.vstack((subcls.lon, subcls.lat)).T
    kdt = sp.cKDTree(all_pts)
    pt_dist, pt_ind = kdt.query(all_pts, k = 6)
    pt_dist = pt_dist[:,1::]
    pt_ind = pt_ind[:,1::]

    new_segs = np.zeros(len(subcls.seg))
    start_seg = np.unique(subcls.seg[np.where(subcls.facc == np.max(subcls.facc))[0]])[0]
    new_segs[np.where(subcls.seg == start_seg)] = 1
    loop = 1
    cnt = 1
    while np.min(new_segs) == 0:
        # print(loop, start_seg, len(np.unique(subcls.seg)))
        seg = np.where(subcls.seg == start_seg)[0]
        new_segs[seg] = cnt

        seg_eps = np.where(subcls.eps[seg] > 0)[0]
        if len(seg_eps) <= 1:
            pt1 = np.where(subcls.ind[seg] == np.min(subcls.ind[seg]))[0]
            pt2 = np.where(subcls.ind[seg] == np.max(subcls.ind[seg]))[0]
            seg_eps = np.array([pt1,pt2])
        
        ngh1_segs = np.unique(subcls.seg[pt_ind[seg[seg_eps[0]],:]])
        ngh2_segs = np.unique(subcls.seg[pt_ind[seg[seg_eps[1]],:]])
        ngh1_segs = np.delete(ngh1_segs, np.where(ngh1_segs == start_seg)[0])
        ngh2_segs = np.delete(ngh2_segs, np.where(ngh2_segs == start_seg)[0])

        if len(ngh1_segs) == 1 and len(ngh2_segs) == 0:
            # print('cond1')
            if np.max(new_segs[np.where(subcls.seg == ngh1_segs)]) == 0:
                start_seg = ngh1_segs[0]
                loop = loop+1
            else:
                new_segs[seg] = np.max(new_segs[np.where(subcls.seg == ngh1_segs)]) #recently added may cause problems 
                zeros = np.where(new_segs == 0)[0]
                if len(zeros) == 0: 
                    print('Segment Aggregation Done')
                else:
                    start_seg = np.unique(subcls.seg[zeros[np.where(subcls.facc[zeros] == np.max(subcls.facc[zeros]))[0]]])[0]
                    cnt = np.max(new_segs)+1
                    loop = loop+1

        elif len(ngh1_segs) == 0 and len(ngh2_segs) == 1:
            # print('cond2')
            if np.max(new_segs[np.where(subcls.seg == ngh2_segs)]) == 0:
                start_seg = ngh2_segs[0]
                loop = loop+1
            else:
                new_segs[seg] = np.max(new_segs[np.where(subcls.seg == ngh2_segs)]) #recently added may cause problems
                zeros = np.where(new_segs == 0)[0]
                if len(zeros) == 0: 
                    print('Segment Aggregation Done')
                else:
                    start_seg = np.unique(subcls.seg[zeros[np.where(subcls.facc[zeros] == np.max(subcls.facc[zeros]))[0]]])[0]
                    cnt = np.max(new_segs)+1
                    loop = loop+1

        elif len(ngh1_segs) == 1 and len(ngh2_segs) == 1:
            # print('cond3')
            if np.max(new_segs[np.where(subcls.seg == ngh1_segs)]) == 0 and np.max(new_segs[np.where(subcls.seg == ngh2_segs)]) == 0:             
                if np.max(subcls.facc[np.where(subcls.seg == ngh1_segs)])  > np.max(subcls.facc[np.where(subcls.seg == ngh2_segs)]):
                    start_seg = ngh1_segs[0] # currently have next neighbor going to downstream end. 
                else:
                    start_seg = ngh2_segs[0]
                loop = loop+1
            elif np.max(new_segs[np.where(subcls.seg == ngh1_segs)]) > 0 and np.max(new_segs[np.where(subcls.seg == ngh2_segs)]) == 0:
                new_segs[seg] = np.max(new_segs[np.where(subcls.seg == ngh1_segs)]) 
                start_seg = ngh2_segs[0]
                loop = loop+1
            elif np.max(new_segs[np.where(subcls.seg == ngh1_segs)]) == 0 and np.max(new_segs[np.where(subcls.seg == ngh2_segs)]) > 0:
                new_segs[seg] = np.max(new_segs[np.where(subcls.seg == ngh2_segs)]) 
                start_seg = ngh1_segs[0]
                loop = loop+1
            else:
                zeros = np.where(new_segs == 0)[0]
                if len(zeros) == 0: 
                    print('Segment Aggregation Done')
                else:
                    start_seg = np.unique(subcls.seg[zeros[np.where(subcls.facc[zeros] == np.max(subcls.facc[zeros]))[0]]])[0]
                    cnt = np.max(new_segs)+1
                    loop = loop+1
        
        elif len(ngh1_segs) > 1 and len(ngh2_segs) == 1:
            # print('cond4')
            if np.max(new_segs[np.where(subcls.seg == ngh2_segs)]) == 0:
                start_seg = ngh2_segs[0]
                loop=loop+1
            else:
                max_vals = np.array([np.max(new_segs[np.where(subcls.seg == ns)]) for ns in ngh1_segs])
                if np.min(max_vals) == 0:
                    start_seg = ngh1_segs[np.where(max_vals == 0)[0][0]]
                    cnt = np.max(new_segs)+1
                    loop = loop+1
                else:
                    zeros = np.where(new_segs == 0)[0]
                    if len(zeros) == 0: 
                        print('Segment Aggregation Done')
                    else:
                        start_seg = np.unique(subcls.seg[zeros[np.where(subcls.facc[zeros] == np.max(subcls.facc[zeros]))[0]]])[0]
                        cnt = np.max(new_segs)+1
                        loop = loop+1

        elif len(ngh1_segs) == 1 and len(ngh2_segs) > 1:
            # print('cond5')
            if np.max(new_segs[np.where(subcls.seg == ngh1_segs)]) == 0:
                start_seg = ngh1_segs[0]
                loop=loop+1
            else:
                max_vals = np.array([np.max(new_segs[np.where(subcls.seg == ns)]) for ns in ngh2_segs])
                if np.min(max_vals) == 0:
                    start_seg = ngh2_segs[np.where(max_vals == 0)[0][0]]
                    cnt = np.max(new_segs)+1
                    loop = loop+1
                else:
                    zeros = np.where(new_segs == 0)[0]
                    if len(zeros) == 0: 
                        print('Segment Aggregation Done')
                    else:
                        start_seg = np.unique(subcls.seg[zeros[np.where(subcls.facc[zeros] == np.max(subcls.facc[zeros]))[0]]])[0]
                        cnt = np.max(new_segs)+1
                        loop = loop+1

        elif len(ngh1_segs) > 1 and len(ngh2_segs) == 0:
            # print('cond6')
            max_vals = np.array([np.max(new_segs[np.where(subcls.seg == ns)]) for ns in ngh1_segs])
            if np.min(max_vals) == 0:
                start_seg = ngh1_segs[np.where(max_vals == 0)[0][0]]
                cnt = np.max(new_segs)+1
                loop = loop+1
            else:
                zeros = np.where(new_segs == 0)[0]
                if len(zeros) == 0: 
                    print('Segment Aggregation Done')
                else:
                    start_seg = np.unique(subcls.seg[zeros[np.where(subcls.facc[zeros] == np.max(subcls.facc[zeros]))[0]]])[0]
                    cnt = np.max(new_segs)+1
                    loop = loop+1

        elif len(ngh1_segs) == 0 and len(ngh2_segs) > 1:
            # print('cond7')
            max_vals = np.array([np.max(new_segs[np.where(subcls.seg == ns)]) for ns in ngh2_segs])
            if np.min(max_vals) == 0:
                start_seg = ngh2_segs[np.where(max_vals == 0)[0][0]]
                cnt = np.max(new_segs)+1
                loop = loop+1
            else:
                zeros = np.where(new_segs == 0)[0]
                if len(zeros) == 0: 
                    print('Segment Aggregation Done')
                else:
                    start_seg = np.unique(subcls.seg[zeros[np.where(subcls.facc[zeros] == np.max(subcls.facc[zeros]))[0]]])[0]
                    cnt = np.max(new_segs)+1
                    loop = loop+1

        elif len(ngh1_segs) > 1 and len(ngh2_segs) > 1:
            # print('cond8')
            max_vals1 = np.array([np.max(new_segs[np.where(subcls.seg == ns)]) for ns in ngh1_segs])
            max_vals2 = np.array([np.max(new_segs[np.where(subcls.seg == ns)]) for ns in ngh2_segs])
            if np.min(max_vals1) == 0:
                start_seg = ngh1_segs[np.where(max_vals1 == 0)[0][0]]
                cnt = np.max(new_segs)+1
                loop = loop+1
            elif np.min(max_vals2) == 0:
                start_seg = ngh2_segs[np.where(max_vals2 == 0)[0][0]]
                cnt = np.max(new_segs)+1
                loop = loop+1
            else:
                zeros = np.where(new_segs == 0)[0]
                if len(zeros) == 0: 
                    print('Segment Aggregation Done')
                else:
                    start_seg = np.unique(subcls.seg[zeros[np.where(subcls.facc[zeros] == np.max(subcls.facc[zeros]))[0]]])[0]
                    cnt = np.max(new_segs)+1
                    loop = loop+1

        else: 
            # print('cond9')
            zeros = np.where(new_segs == 0)[0]
            if len(zeros) == 0: 
                print('Segment Aggregation Done')
            else:
                start_seg = np.unique(subcls.seg[zeros[np.where(subcls.facc[zeros] == np.max(subcls.facc[zeros]))[0]]])[0]
                cnt = np.max(new_segs)+1
                loop = loop+1

        if loop > len(np.unique(subcls.seg))+10:
            print('LOOP STUCK!!!')

    subcls.new_segs = new_segs
    subcls.new_ind, subcls.new_eps = update_rch_indexes(subcls, subcls.new_segs)
    subcls.new_dist = calc_segDist(subcls.lon, subcls.lat, subcls.new_segs,
                                   subcls.facc, subcls.new_ind)
    
    centerlines.new_segs[level2] = subcls.new_segs
    centerlines.new_ind[level2] = subcls.new_ind
    centerlines.new_eps[level2] = subcls.new_eps
    centerlines.new_dist[level2] = subcls.new_dist

    end=time.time()
    print('Time to Finish Basin: ' + str(np.round((end-start)/60, 2)) + ' min')

print('Updating NetCDF')
update_netcdf(nc_file, centerlines)

end_all=time.time()
print('Time to Finish All Basins: ' + str(np.round((end_all-start_all)/60, 2)) + ' min')            





