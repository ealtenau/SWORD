"""
Shortest Path Variable Fromatting (path_variables_nc.py)
==============================================================

Script for ingesting and formatting shortest path geopackage 
files calculated along the SWORD v17 network in QGIS. Shortest
paths are derived from every outlet to every connected 
headwater point in SWORD.

The script is run at a Pfafstetter Level 2 basin scale.
Command line arguments required are the two-letter
region identifier (i.e. NA), SWORD version (i.e. v17),
and Pfafstetter Level 2 basin (i.e. 74).

Execution example (terminal):
    python path_variables_nc.py NA v17 74

"""

import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import netCDF4 as nc
import geopandas as gp
import time
import argparse
from scipy import spatial as sp
from scipy import stats
from scipy import interpolate
from geopy import distance
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

################################################################################################
################################################################################################
################################################################################################

def get_distances(lon,lat):
    """
    Calculates geodesic distance along a set of points given input 
    latitude and longitude values. 

    Parameters
    ----------
    lon: numpy.array() 
        Longitude
    lat: numpy.array()
        Latitude

    Returns
    -------
    distances: numpy.array()
        Numpy array of cumulative distances in meters along the 
        input coordinates. 
    
    """

    traces = len(lon) -1
    distances = np.zeros(traces)
    for i in range(traces):
        start = (lat[i], lon[i])
        finish = (lat[i+1], lon[i+1])
        distances[i] = distance.geodesic(start, finish).m
    distances = np.append(0,distances)
    return distances

################################################################################################

def side_chan_filt(cl_rchs, main_side, cl_lon, cl_lat, rch_paths_dist):
    """
    Calculates distance from outlet in the side channel SWORD reaches
    that did not have an associated shortest path. 

    Parameters
    ----------
    cl_rchs: numpy.array() 
        SWORD Reach IDs at the centerline scale. 
    main_side: numpy.array()
        Flag indicating whether a SWORD reach is on the main or side
        channel network.
    cl_lon: numpy.array()
        SWORD longitude values at the centerline scale. 
    cl_lat: numpy.array()
        SWORD latitude values at the centerline scale. 
    rch_paths_dist: numpy.array()
        Distance from outlet calculated along the main network. 

    Returns
    -------
    side_dist: numpy.array()
        Numpy array of distance from outlet in meters for both the 
        SWORD main and side network reaches. 
    
    """
     
    ngh_matrix = np.zeros(cl_rchs.shape)
    side_dist = np.copy(rch_paths_dist)
    count=1
    side_inds = np.where(main_side == 1)[0]
    side_chans = np.unique(cl_rchs[0,side_inds])
    if len(side_chans) > 0:
        #finding where the side channels are listed as neighbors. 
        for ind in list(range(len(side_chans))):
            r1 = np.where(cl_rchs[1,:] == side_chans[ind])[0]
            r2 = np.where(cl_rchs[2,:] == side_chans[ind])[0]
            r3 = np.where(cl_rchs[3,:] == side_chans[ind])[0]
            ngh_matrix[1,r1] = 1
            ngh_matrix[1,r2] = 1
            ngh_matrix[1,r3] = 1

        #finding the reaches on the main channel network with side neighbors. 
        row_sum = np.sum(ngh_matrix, axis = 0)
        ngh_pts = np.where((row_sum > 0)&(main_side==0))[0]
        
        #looping through the side channels starting with the side channel that 
        #has a neighbor with the lowest distance from outlet.
        flag = np.zeros(len(cl_rchs[0,:]))
        flag[np.where(main_side == 0)] = 1
        
        ### Added the lines to remove nan values on 3/21/2024. To fix issues with Europe. 
        rmv = np.where(np.isnan(side_dist[ngh_pts]) == True)[0]
        ngh_pts = np.delete(ngh_pts,rmv)
        #####
        
        start_pt = ngh_pts[np.where(side_dist[ngh_pts] == np.nanmin(side_dist[ngh_pts]))[0]][0]
        loop = 1
        check = len(ngh_pts)+5000 #was 500 
        while len(ngh_pts) > 0:
            # print(loop, cl_rchs[0::,start_pt])
            nghs = cl_rchs[1::,start_pt]
            nghs = nghs[nghs>0]
            ngh_basins = np.array([str(n)[0:2] for n in nghs])
            nghs = nghs[ngh_basins == basin[2:4]]
            ngh_chan = np.array([np.max(main_side[np.where(cl_rchs[0,:]==n)]) for n in nghs])
            nghs = nghs[ngh_chan==1]
            ngh_flag = np.array([np.max(flag[np.where(cl_rchs[0,:]==n)]) for n in nghs])
            nghs = nghs[ngh_flag==0]

            if len(nghs) > 0:
                rch = np.where(cl_rchs[0,:] == nghs[0])[0] #if multiple choose first.
                sort_ind = rch[np.argsort(cl_ind[rch])] 
                dnstrm_pt = np.where(cl_rchs[:,sort_ind] == cl_rchs[0,start_pt])[1]
                # print(dnstrm_pt)
                if len(dnstrm_pt) > 1 or len(dnstrm_pt) == 0:
                    # in an odd case where the reach hasn't broken at a tributary and is at both ends. 
                    dnstrm_pt = 0
                # if dnstrm_pt == 0:
                    x_coords = cl_lon[sort_ind]
                    y_coords = cl_lat[sort_ind]
                    diff = get_distances(x_coords,y_coords)
                    rch_dist = np.cumsum(diff)
                    rch_dist_out = rch_dist+side_dist[start_pt]+30
                    side_dist[sort_ind] = rch_dist_out
                    flag[sort_ind] = 1
                    if len(nghs) == 1:
                        if start_pt in ngh_pts:
                            ngh_pts = np.delete(ngh_pts, np.where(ngh_pts == start_pt)[0])
                    if len(nghs)>1 and np.max(main_side[sort_ind[dnstrm_pt]]) == 1:
                        ngh_pts = np.append(ngh_pts,sort_ind[dnstrm_pt]) 
                    start_pt = sort_ind[-1]
                    loop = loop+1
                else:
                    x_coords = cl_lon[sort_ind[::-1]]
                    y_coords = cl_lat[sort_ind[::-1]]
                    diff = get_distances(x_coords,y_coords)
                    rch_dist = np.cumsum(diff)
                    rch_dist_out = rch_dist+side_dist[start_pt]+30
                    side_dist[sort_ind[::-1]] = rch_dist_out
                    flag[sort_ind] = 1
                    if len(nghs) == 1:
                        if start_pt in ngh_pts:
                            ngh_pts = np.delete(ngh_pts, np.where(ngh_pts == start_pt)[0])
                    if len(nghs)>1 and np.max(main_side[sort_ind[dnstrm_pt]]) == 1:
                        ngh_pts = np.append(ngh_pts,sort_ind[dnstrm_pt])
                    start_pt = sort_ind[0]
                    loop = loop+1  
            
            else:
                ngh_pts = np.delete(ngh_pts, np.where(ngh_pts == start_pt)[0])
                if len(ngh_pts) == 0:
                    loop = loop+1
                    continue
                else:
                    start_pt = ngh_pts[np.where(side_dist[ngh_pts] == np.nanmin(side_dist[ngh_pts]))[0]][0]
                    loop = loop+1

            if loop > check:
                print('LOOP1 STUCK', cl_rchs[0::,start_pt])
                break

        #have to fill in weird scenerios. 
        if np.min(flag) == 0:
            missed_rchs = np.unique(cl_rchs[0,np.where(flag == 0)])
            start_rch = missed_rchs[0]
            loop = 1
            check = len(missed_rchs)+ 100 #was 100 
            while len(missed_rchs) > 0:
                # print(loop, start_rch)
                rch = np.where(cl_rchs[0,:] == start_rch)[0] #if multiple choose first.
                sort_ind = rch[np.argsort(cl_ind[rch])]
                eps = np.array([sort_ind[0],sort_ind[-1]])
                end_pts = np.vstack((cl_lon[eps], cl_lat[eps])).T
                basin_pts = np.vstack((cl_lon, cl_lat)).T
                kdt = sp.cKDTree(basin_pts)
                pt_dist, pt_ind = kdt.query(end_pts, k = 15)
                end1_dist = np.array(side_dist[pt_ind[0,np.where(side_dist[pt_ind[0,:]]>0)]][0])
                end2_dist = np.array(side_dist[pt_ind[1,np.where(side_dist[pt_ind[1,:]]>0)]][0])
                if len(end1_dist) > 0:
                    end1_dist = np.array([np.min(end1_dist)])
                if len(end2_dist) > 0:
                    end2_dist = np.array([np.min(end2_dist)])

                if len(end1_dist) == 0 and len(end2_dist) > 0:
                    x_coords = cl_lon[sort_ind[::-1]]
                    y_coords = cl_lat[sort_ind[::-1]]
                    diff = get_distances(x_coords,y_coords)
                    rch_dist = np.cumsum(diff)
                    rch_dist_out = rch_dist+end2_dist+30
                    side_dist[sort_ind[::-1]] = rch_dist_out
                    flag[sort_ind[::-1]] = 1
                    missed_rchs = np.delete(missed_rchs, np.where(missed_rchs == start_rch)[0])
                    if len(missed_rchs) == 0:
                        continue
                    start_rch = missed_rchs[0]
                    loop = loop+1

                elif len(end1_dist) > 0 and len(end2_dist) == 0:
                    x_coords = cl_lon[sort_ind]
                    y_coords = cl_lat[sort_ind]
                    diff = get_distances(x_coords,y_coords)
                    rch_dist = np.cumsum(diff)
                    rch_dist_out = rch_dist+end1_dist+30
                    side_dist[sort_ind] = rch_dist_out
                    flag[sort_ind] = 1
                    missed_rchs = np.delete(missed_rchs, np.where(missed_rchs == start_rch)[0])
                    if len(missed_rchs) == 0:
                        continue
                    start_rch = missed_rchs[0]
                    loop = loop+1

                elif len(end1_dist) > 0 and len(end2_dist) > 0:
                    dnstrm_pt = end1_dist < end2_dist
                    if dnstrm_pt == True:
                        x_coords = cl_lon[sort_ind]
                        y_coords = cl_lat[sort_ind]
                        diff = get_distances(x_coords,y_coords)
                        rch_dist = np.cumsum(diff)
                        rch_dist_out = rch_dist+end1_dist+30
                        side_dist[sort_ind] = rch_dist_out
                        flag[sort_ind] = 1
                        missed_rchs = np.delete(missed_rchs, np.where(missed_rchs == start_rch)[0])
                        if len(missed_rchs) == 0:
                            continue
                        start_rch = missed_rchs[0]
                        loop = loop+1
                    else:
                        x_coords = cl_lon[sort_ind[::-1]]
                        y_coords = cl_lat[sort_ind[::-1]]
                        diff = get_distances(x_coords,y_coords)
                        rch_dist = np.cumsum(diff)
                        rch_dist_out = rch_dist+end2_dist+30
                        side_dist[sort_ind[::-1]] = rch_dist_out
                        flag[sort_ind[::-1]] = 1
                        missed_rchs = np.delete(missed_rchs, np.where(missed_rchs == start_rch)[0])
                        if len(missed_rchs) == 0:
                            continue
                        start_rch = missed_rchs[0]
                        loop = loop+1
                
                else:
                    #need to see if there are actually neighbors or not... if so, just don't delete the reach from the que
                    any_nghs = np.where(cl_rchs[1::,rch] > 0)[0]
                    # any_nghs = cl_rchs[1::,rch[np.where(cl_rchs[1::,rch] > 0)[1]]]
                    # any_nghs = np.unique(any_nghs)
                    # any_nghs = any_nghs[any_nghs>0]
                    # if no nghs are found 
                    if len(any_nghs) == 0:
                        x_coords = cl_lon[sort_ind]
                        y_coords = cl_lat[sort_ind]
                        diff = get_distances(x_coords,y_coords)
                        rch_dist = np.cumsum(diff)
                        rch_dist_out = np.array(rch_dist)
                        side_dist[sort_ind] = rch_dist_out
                        flag[sort_ind] = 1
                        missed_rchs = np.delete(missed_rchs, np.where(missed_rchs == start_rch)[0])
                        if len(missed_rchs) == 0:
                            continue
                        start_rch = missed_rchs[0]
                        loop = loop+1
                    else:
                        #try another start reach. 
                        if len(missed_rchs) == 0:
                            continue
                        start_rch = missed_rchs[np.where(missed_rchs != start_rch)[0][0]]
                        # all_rchs = np.append(any_nghs, start_rch)
                        # start_rch = missed_rchs[np.where(np.in1d(missed_rchs, all_rchs) == False)[0][0]]
                        loop = loop+1

                if loop > check:
                    print('LOOP2 STUCK', start_rch)
                    break

    return side_dist

################################################################################################
                
def write_path_netcdf(outfile,region,cl_ind,cl_lon,cl_lat,
                      cl_rchs,cl_nodes,rch_paths,rch_paths_order,
                      rch_paths_dist,rch_paths_dist2,main_side):
    """
    Outputs a netCDF file containing shortest path variables and 
    distance from outlet calculated along a Pfafstetter Level 2 basin
    in the SWORD database. 

    Parameters
    ----------
    outfile: str 
        The directory location to output the netCDF file.
    region: str
        Two-letter SWORD region/continent identifier (i.e. NA).
    cl_ind: numpy.array()
        SWORD centerline IDs.
    cl_lon: numpy.array()
        SWORD centerline longitude (WGS 84, EPSG:4326). 
    cl_lat: numpy.array()
        SWORD centerline latitude (WGS 84, EPSG:4326).
    cl_rchs: numpy.array()
        SWORD centerline reach IDs. 
    cl_nodes: numpy.array()
        SWORD centerline node IDs. 
    rch_paths: numpy.array()
        Path frequency: The number of times a reach is traveled along get to any 
        given headwater point.
    rch_paths_order: numpy.array()
        Path order: Unique values representing continuous paths from the
        river outlet to the headwaters ordered from the longest path (1) 
        to the shortest path (N). 
    rch_paths_dist: numpy.array()
        Distance from outlet on main network only.
    rch_paths_dist2: numpy.array()
        Distance from outlet on main and side networks.
    main_side: numpy.array()
        Main-Side network flag. 
        
    Returns
    -------
    None. 
    
    """
    
    # global attributes
    root_grp = nc.Dataset(outfile, 'w', format='NETCDF4')
    root_grp.x_min = np.min(cl_lon)
    root_grp.x_max = np.max(cl_lon)
    root_grp.y_min = np.min(cl_lat)
    root_grp.y_max = np.max(cl_lat)
    root_grp.Name = 'hb'+region
    root_grp.production_date = time.strftime("%d-%b-%Y %H:%M:%S", time.gmtime()) #utc time
    #root_grp.history = 'Created ' + time.ctime(time.time())

    # groups
    cl_grp = root_grp.createGroup('centerlines')

    # dimensions
    #root_grp.createDimension('d1', 2)
    cl_grp.createDimension('num_points', len(cl_ind))
    cl_grp.createDimension('num_domains', 4)
    # cl_grp.createDimension('num_paths', num_paths)

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
    path_order = cl_grp.createVariable(
        'path_order_by_length', 'i8', ('num_points',), fill_value=-9999.)
    path_freq = cl_grp.createVariable(
        'path_travel_frequency', 'i8', ('num_points',), fill_value=-9999.)
    path_dist = cl_grp.createVariable(
        'dist_out', 'f8', ('num_points',), fill_value=-9999.)
    path_dist_all = cl_grp.createVariable(
        'dist_out_all', 'f8', ('num_points',), fill_value=-9999.)
    main_net = cl_grp.createVariable(
        'main_side_chan', 'i4', ('num_points',), fill_value=-9999.)

    # saving data
    print("saving nc")

    # centerline data
    cl_id[:] = cl_ind[:]
    cl_x[:] = cl_lon[:]
    cl_y[:] = cl_lat[:]
    reach_id[:,:] = cl_rchs[:]
    node_id[:,:] = cl_nodes[:]
    path_order[:] = rch_paths_order[:]
    path_dist[:] = rch_paths_dist[:]
    path_dist_all[:] = rch_paths_dist2[:]
    path_freq[:] = rch_paths[:]
    main_net[:] = main_side[:]

    root_grp.close()

################################################################################################

def calc_path_variables(path_files, kdt):
    """
    Calculates path frequency, path order, and distance from outlet variables
    based on the shortest path files.  

    Parameters
    ----------
    path_files: list 
        List containing all the related shortest path files for a Pfafstetter
        Level 2 basin.
    kdt: scipy.spatial.cKDTree object 
        KD-Tree for nearest neighbors in SWORD.

    Returns
    -------
    rch_paths: numpy.array()
        Path frequency: The number of times a reach is traveled along get to any 
        given headwater point.
    rch_paths_order: numpy.array()
        Path order: Unique values representing continuous paths from the
        river outlet to the headwaters ordered from the longest path (1) 
        to the shortest path (N).
    rch_paths_dist: numpy.array()
        Distance from outlet (meters).

    """

    count = 1
    for ind in list(range(len(path_files))):
        print(ind, len(path_files)-1)
        path = gp.read_file(path_dir+path_files[ind])
        path_lens = [len(g.coords.xy[0]) for g in path['geometry']]
        sort = np.argsort(path_lens)[::-1]
        for p in list(range(len(sort))):
            # print(p, len(path)-1)
            lon = np.array(path['geometry'][sort[p]].coords.xy[0])
            lat = np.array(path['geometry'][sort[p]].coords.xy[1])
            
            if len(lon) == 0:
                continue
        
            # dist = calc_path_dist(lon,lat) #takes longer...
            diff = get_distances(lon,lat)
            dist = np.cumsum(diff)

            #spatial query with sword
            path_pts = np.vstack((lon, lat)).T
            pt_dist, pt_ind = kdt.query(path_pts, k = 1)
            rch_paths[pt_ind] = rch_paths[pt_ind]+1 
            
            #assigns the paths a number in order by length.
            add = np.where(rch_paths_order[pt_ind] == 0)[0]
            rch_paths_dist[pt_ind[add]] = dist[add]
            rch_paths_order[pt_ind[add]] = count
            count = count+1
            # if len(add) > 0:
            #     rch_paths[pt_ind] = rch_paths[pt_ind]+1 # added condition for areas where paths are incomplete for efficiency (i.e. Amazon).

            del(dist);del(lon);del(lat)
        del(path)
    return rch_paths, rch_paths_dist, rch_paths_order

################################################################################################

def filter_zero_pts(rch_paths, rch_paths_dist, rch_paths_order, cl_rchs, cl_ind):
    """
    Fills in path frequency, path order, and distance from outlet variables
    for skipped coordinates along the shortest paths.  

    Parameters
    ----------
    rch_paths: numpy.array()
        Path frequency: The number of times a reach is traveled along get to any 
        given headwater point.
    rch_paths_order: numpy.array()
        Path order: Unique values representing continuous paths from the
        river outlet to the headwaters ordered from the longest path (1) 
        to the shortest path (N).
    rch_paths_dist: numpy.array()
        Distance from outlet (meters).
    cl_rchs: numpy.array()
        SWORD centerline reach IDs.
    cl_ind: numpy.array()
        SWORD centerline IDs.
    Returns
    -------
    None.

    """

    #find SWORD reaches that contain a centerline with a path frequency value 
    #equal to zero. Loop through and correct the zero point values.  
    zero = np.where(rch_paths == 0)[0]
    zero_rchs = np.unique(cl_rchs[0,zero])
    for z in list(range(len(zero_rchs))):
        #find all centerline points associated with the reach ID. 
        pts = np.where(cl_rchs[0,:] == zero_rchs[z])[0]
        #calculate percentage of reach with zero values for path frequency. 
        rch_zeros = np.where(rch_paths[pts] == 0)[0]
        perc = (len(rch_zeros)/len(pts))*100
        # print([z, perc])
        #if the reach is longer than 3 points and the percentage of zero path 
        #frequency values is less than 70%, replace zero values based on non-zero 
        #reach values. 
        if perc < 70 and len(pts) > 3:
            nz = np.where(rch_paths[pts] > 0)[0]
            rch_paths[pts[rch_zeros]] = stats.mode(rch_paths[pts[nz]])[0]
            rch_paths_order[pts[rch_zeros]] = stats.mode(rch_paths_order[pts[nz]])[0]
            #updating distance from outlet. 
            for zp in list(range(len(rch_zeros))):
                nz2 = np.where(rch_paths_dist[pts] > 0)[0]
                id_diff = np.abs(cl_ind[pts[rch_zeros[zp]]] - cl_ind[pts[nz2]])
                nghs = np.where(id_diff == 1)[0]
                if len(nghs) == 1:
                    if cl_ind[pts[nghs]] < cl_ind[pts[rch_zeros[zp]]]:
                        rch_paths_dist[pts[rch_zeros[zp]]] = rch_paths_dist[pts[nghs]]+30
                    else:
                        rch_paths_dist[pts[rch_zeros[zp]]] = rch_paths_dist[pts[nghs]]-30
                else:
                    nghs2 = np.argsort(id_diff)[0:5]
                    good_vals = np.where(rch_paths_dist[pts[nghs2]] > 0)[0]
                    rch_paths_dist[pts[rch_zeros[zp]]] = np.mean(rch_paths_dist[pts[nghs2[good_vals]]]) 
        #short reach condition. 
        elif perc < 100 and len(pts) <= 3:
            nz = np.where(rch_paths[pts] > 0)[0]
            rch_paths[pts[rch_zeros]] = stats.mode(rch_paths[pts[nz]])[0]
            rch_paths_order[pts[rch_zeros]] = stats.mode(rch_paths_order[pts[nz]])[0]
            rch_paths_dist[pts[rch_zeros]] = np.max(rch_paths_dist[pts[nz]])
        #if other conditions fail, leave values zero. 
        else:
            rch_paths[pts] = 0
            rch_paths_order[pts] = 0
            rch_paths_dist[pts] = 0

################################################################################################

def filter_short_side_channels(cl_rchs, main_side, rch_paths, rch_paths_order):
    """
    Identifies and corrects short reaches on the main channel network that 
    have been flagged as a side channel reach.  

    Parameters
    ----------
    cl_rchs: numpy.array()
        SWORD centerline reach IDs.
    main_side: numpy.array()
        Main-Side network flag.
    rch_paths: numpy.array()
        Path frequency: The number of times a reach is traveled along get to any 
        given headwater point.
    rch_paths_order: numpy.array()
        Path order: Unique values representing continuous paths from the
        river outlet to the headwaters ordered from the longest path (1) 
        to the shortest path (N).

    Returns
    -------
    None.

    """

    #loop through side channel reaches and correct to a main network 
    #reach if there is only one reach neighbor on each end of the 
    #side channel reach. 
    side_chan = np.unique(cl_rchs[0,np.where(main_side == 1)[0]])
    for s in list(range(len(side_chan))):
        # print(s)
        pts = np.where(cl_rchs[0,:] == side_chan[s])[0]
        if len(pts) <= 6:
            try:
                print(side_chan[s])
                nghs = np.unique(cl_rchs[1::,pts])
                nghs = nghs[nghs > 0]
                p = np.array([np.max(rch_paths[np.where(cl_rchs[0,:] == n)[0]]) for n in nghs])
                po = np.array([np.max(rch_paths_order[np.where(cl_rchs[0,:] == n)[0]]) for n in nghs])
                if len(np.unique(p)) == 1 and len(np.unique(po)) == 1:
                    main_side[pts] = 0
                    rch_paths[pts] = np.unique(p)
                    rch_paths_order[pts] = np.unique(po)
            except:
                continue

################################################################################################
################################################################################################
################################################################################################

start_all = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version (i.e. v17)", type = str)
parser.add_argument("basin", help="Pfafstetter Level 2 Basin Number (i.e. 74)", type = str)
args = parser.parse_args()

region = args.region
version = args.version
basin = args.basin

print('Starting Basin: ', basin)
sword_dir = main_dir+'/data/outputs/Reaches_Nodes/'+version+\
    '/reach_geometry/'+region.lower()+'_sword_'+version+'_connectivity.nc'
path_dir = main_dir+'/data/outputs/Reaches_Nodes/'+version+\
    '/network_building/'+region+'/hb'+basin+'_paths/'
path_nc = main_dir+'/data/outputs/Reaches_Nodes/'+version+\
    '/network_building/pathway_netcdfs/'+region+'/hb'+basin+'_path_vars.nc'
path_files = os.listdir(path_dir)
path_files = np.array([f for f in path_files if '.gpkg' in f])
path_files = np.array([f for f in path_files if 'path_' in f])

#reading in sword data.
sword = nc.Dataset(sword_dir)
cl_id_all = sword.groups['centerlines'].variables['cl_id'][:]
cl_x_all = sword.groups['centerlines'].variables['x'][:]
cl_y_all = sword.groups['centerlines'].variables['y'][:]
cl_rchs_all = sword.groups['centerlines'].variables['reach_id'][:]
cl_nodes_all = sword.groups['centerlines'].variables['node_id'][:]
sword.close()

#subset sword to basin. 
l2_basins = np.array([int(str(ind)[0:2]) for ind in cl_rchs_all[0,:]])
l2 = np.where(l2_basins == int(basin))[0]
cl_ind = cl_id_all[l2]
cl_lon = cl_x_all[l2]
cl_lat = cl_y_all[l2]
cl_rchs = cl_rchs_all[:,l2]
cl_nodes = cl_nodes_all[l2]
del(cl_id_all);del(cl_x_all);del(cl_y_all);del(cl_rchs_all)#;del(cl_nodes_all)

#if more than one file, sort by longest path in each file (descending).
if len(path_files) > 1:
    start = time.time()
    print('Finding Start File')
    length = []
    for idx in list(range(len(path_files))):
        # print(idx)
        path = gp.read_file(path_dir+path_files[idx])
        path_lens = max([len(g.coords.xy[0]) for g in path['geometry']])
        length.append(path_lens)
        del(path)
    file_order = np.argsort(length)[::-1]
    path_files = path_files[file_order]
    end = time.time()
    print('Finished Ordering Paths in: '+str(np.round((end-start)/60,2))+' mins')

#defining filler variables. 
rch_paths = np.zeros(len(cl_ind))
rch_paths_dist = np.zeros(len(cl_ind))
rch_paths_order = np.zeros(len(cl_ind))

#initialzing spatial query. 
sword_pts = np.vstack((cl_lon, cl_lat)).T
kdt = sp.cKDTree(sword_pts)

#calculating path variables. 
print('Starting Path Calculations')
start = time.time()
rch_paths, rch_paths_dist, rch_paths_order = calc_path_variables(path_files, kdt)
end = time.time()
print('Finished Path Calculations in: '+str(np.round((end-start)/60,2))+' mins')

print('Filtering Missed Points')
start = time.time()
filter_zero_pts(rch_paths, rch_paths_dist, rch_paths_order, cl_rchs, cl_ind)
#define side channels
main_side = np.zeros(len(cl_ind))
main_side[np.where(rch_paths == 0)[0]] = 1
end = time.time()
print('Finished Filter in: '+str(np.round((end-start)/60,2))+' mins')

print('Filling in Side Channels')
start = time.time()
rch_paths_dist2 = side_chan_filt(cl_rchs, main_side, cl_lon, cl_lat, rch_paths_dist)
#filling last few point gaps where paths are mostly covered. 
inds = np.arange(rch_paths_dist2.shape[0])
good = np.where(np.isfinite(rch_paths_dist2))[0]
interp = interpolate.interp1d(inds[good], rch_paths_dist2[good],bounds_error=False)
rch_paths_dist2 = np.where(np.isfinite(rch_paths_dist2),rch_paths_dist2,interp(inds))
end = time.time()
print('Finished Filling Side Channels in: '+str(np.round((end-start)/60,2))+' mins')

print('Filtering Short Side Channels')
filter_short_side_channels(cl_rchs, main_side, rch_paths, rch_paths_order)

print('Normalizing Path Frequency')
unq_paths = np.unique(rch_paths_order)
norm_freq = np.zeros(len(rch_paths))
for ord in list(range(len(unq_paths))):
    pts = np.where(rch_paths_order == unq_paths[ord])[0]
    if min(rch_paths[pts]) > 1:
        norm_freq[pts] = rch_paths[pts]/min(rch_paths[pts])
    else:
        norm_freq[pts] = rch_paths[pts]

print('Saving NetCDF')
start = time.time()
write_path_netcdf(path_nc,basin,cl_ind,cl_lon,cl_lat,
                  cl_rchs,cl_nodes,norm_freq,rch_paths_order,
                  rch_paths_dist,rch_paths_dist2,main_side)

end_all = time.time()
print('Finished Basin '+basin+' in: '+str(np.round((end_all-start_all)/60,2))+' mins')



### PLOTS
# import matplotlib.pyplot as plt

# plt.scatter(cl_lon, cl_lat, c=np.round(np.log(rch_paths)), s = 5, cmap='rainbow')
# plt.show()

# plt.scatter(cl_lon, cl_lat, c=rch_paths, s = 5, cmap='rainbow')
# plt.show()

# plt.scatter(cl_lon, cl_lat, c=rch_paths_dist, s = 2, cmap='rainbow')
# plt.show()

# plt.scatter(cl_lon, cl_lat, c=rch_paths_dist2, s = 2, cmap='rainbow')
# plt.show()

# plt.scatter(cl_lon, cl_lat, c=np.log(rch_paths_order), s = 5, cmap='rainbow')
# plt.show()

# side = np.where(main_side == 1)[0]
# plt.scatter(cl_lon, cl_lat, c='blue', s = 5)
# plt.scatter(cl_lon[side], cl_lat[side], c='red', s = 5)
# plt.show()


# zero = np.where(np.isnan(rch_paths_dist2)==True)[0]
# plt.scatter(cl_lon, cl_lat, c=rch_paths_dist2, s = 5)
# plt.scatter(cl_lon[zero], cl_lat[zero], c='red', s = 5)
# plt.show()


# inds = np.arange(rch_paths_dist2.shape[0])
# good = np.where(np.isfinite(rch_paths_dist2))[0]
# f = interpolate.interp1d(inds[good], rch_paths_dist2[good],bounds_error=False)
# B = np.where(np.isfinite(rch_paths_dist2),rch_paths_dist2,f(inds))
# rch_paths_dist2[zero]

# plt.scatter(cl_lon, cl_lat, c=B, s = 5)
# plt.show()


# rch = np.where(cl_rchs[0,:] == 83300200021)[0]
# rch_paths_dist[rch]
# rch_paths[rch]
# rch_paths_order[rch]

# plt.scatter(cl_lon[rch], cl_lat[rch], c=rch_paths_dist[rch], s = 5, cmap='rainbow')
# plt.show()

# plt.scatter(cl_lon, cl_lat, c=rch_paths_dist, s = 5, cmap='rainbow')
# plt.show()


# ### SAVE CSV
# path_csv = main_dir+'/data/outputs/Reaches_Nodes/'+version+'/network_testing/'+basin+'_paths/'+basin+'_path_vars.csv'
# df=pd.DataFrame(np.array([cl_lon, cl_lat, rch_paths, rch_paths_dist, rch_paths_order, main_side, cl_rchs[0,:], cl_nodes, cl_ind, side_segs]).T)
# df=df.rename(columns={0: "x", 1: "y", 2: "cumulative_path", 3: "dist_out", 4: "path_order_by_length", 5: "main_side", 6: "reach_id", 7: "node_id", 8: "cl_id", 9: "side_seg"})
# df.to_csv(path_csv)


# path_csv = main_dir+'/data/outputs/Reaches_Nodes/'+version+'/network_testing/'+basin+'_paths/'+basin+'_side_channels.csv'
# df=pd.DataFrame(np.array([cl_lon[side], cl_lat[side]]).T)
# df=df.rename(columns={0: "x", 1: "y"})
# df.to_csv(path_csv)


# #read in vars from nc
# path_nc = main_dir+'/data/outputs/Reaches_Nodes/'+version+'/pathways/'+region+'/'+basin+'_path_vars.nc'

# data = nc.Dataset(path_nc)

# rch_paths = data.groups['centerlines'].variables['path_travel_frequency'][:]
# rch_paths_order = data.groups['centerlines'].variables['path_order_by_length'][:]
# rch_paths_dist = data.groups['centerlines'].variables['distance_from_outlet'][:]
# main_side = data.groups['centerlines'].variables['main_side_chan'][:]


### Think about straheler stream order?
# strm_order = np.copy(rch_paths)
# normalize = np.where(strm_order >= 5)[0] # Mississippi has 5 outlets so 5 is the first order streams in the connected river system...
# strm_order[normalize] = (np.round(np.log(rch_paths[normalize]))-(np.min(np.round(np.log(rch_paths[normalize])))))+1
# strm_order[np.where(strm_order == 0)] = 1




