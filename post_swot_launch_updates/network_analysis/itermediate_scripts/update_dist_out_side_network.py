import numpy as np
import netCDF4 as nc
import geopandas as gp
from shapely.geometry import LineString, Point
from geopandas import GeoSeries
import pandas as pd
import time
import argparse
import os
from scipy import spatial as sp
import matplotlib.pyplot as plt
from scipy import stats
from scipy import interpolate
from geopy import distance
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

################################################################################################
################################################################################################
################################################################################################

def get_distances(lon,lat):
    traces = len(lon) -1
    distances = np.zeros(traces)
    for i in range(traces):
        start = (lat[i], lon[i])
        finish = (lat[i+1], lon[i+1])
        distances[i] = distance.geodesic(start, finish).m
    distances = np.append(0,distances)
    return distances

################################################################################################

def side_chan_filt(cl_rchs, main_side, cl_lon, cl_lat, rch_paths_dist, cl_ind, basin):
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
            nghs = nghs[ngh_basins == str(basin)]
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
                        loop = loop+1

                if loop > check:
                    print('LOOP2 STUCK', start_rch)
                    break

    return side_dist

################################################################################################
################################################################################################
################################################################################################

region = 'EU'
version = 'v17'

nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'\
    +region.lower()+'_sword_'+version+'.nc'
con_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'\
    +version+'/reach_geometry/'+region.lower()+'_sword_'+version+'_connectivity.nc'

sword = nc.Dataset(nc_fn)
con = nc.Dataset(con_dir)

cl_rchs_all = con['/centerlines/reach_id'][:]
main_side_all = sword['/reaches/main_side'][:]
cl_lon_all = con['/centerlines/x'][:]
cl_lat_all = con['/centerlines/y'][:]
cl_ind_all = con['/centerlines/cl_id'][:]
rch_dist_all = sword['/reaches/dist_out'][:]
rch_id_all = sword['/reaches/reach_id'][:]
rch_lon_all = sword['/reaches/x'][:]
rch_lat_all = sword['/reaches/y'][:]

rmv = np.where(main_side_all == 1)[0]
rch_paths_dist_all = np.copy(rch_dist_all)
rch_paths_dist_all[rmv] = 0

l2_basins = np.array([int(str(ind)[0:2]) for ind in cl_rchs_all[0,:]])
rch_l2_basins = np.array([int(str(ind)[0:2]) for ind in rch_id_all])
unq_l2 = np.unique(l2_basins)

new_dist = np.zeros(len(cl_lon_all))

### loop
for ind in list(range(len(unq_l2))):
    print(ind, len(unq_l2)-1)
    basin = unq_l2[ind]
    l2 = np.where(l2_basins == basin)[0]
    rch_l2 = np.where(rch_l2_basins == basin)[0]

    cl_rchs = cl_rchs_all[:,l2]
    cl_lon = cl_lon_all[l2]
    cl_lat = cl_lat_all[l2]
    cl_ind = cl_ind_all[l2]
    rch_id = rch_id_all[rch_l2]

    main_side = np.zeros(len(cl_lon))
    rch_paths_dist = np.zeros(len(cl_lon))
    unq_rchs = np.unique(cl_rchs[0,:])
    for r in list(range(len(unq_rchs))):
        cl_inds = np.where(cl_rchs[0,:] == unq_rchs[r])
        rch_ind = np.where(rch_id_all == unq_rchs[r])
        main_side[cl_inds] = main_side_all[rch_ind]
        rch_paths_dist[cl_inds] = rch_paths_dist_all[rch_ind]

    new_dist[l2] = side_chan_filt(cl_rchs, main_side, cl_lon, cl_lat, rch_paths_dist, cl_ind, basin)


new_dist_out = np.copy(rch_dist_all)
rchs = rch_id_all[np.where(main_side_all == 1)[0]]
for r2 in list(range(len(rchs))):
    idx = np.where(cl_rchs_all[0,:] == rchs[r2])[0]
    ridx = np.where(rch_id_all == rchs[r2])[0]
    new_dist_out[ridx] = max(new_dist[idx])






# zero = np.where(np.isnan(new_dist)==True)[0]
# plt.scatter(cl_lon, cl_lat, c=new_dist, s = 5)
# plt.scatter(cl_lon[zero], cl_lat[zero], c='red', s = 5)
# plt.show()


### Mid-loop edits 
# zero = np.where(side_dist==0)[0]
# plt.scatter(cl_lon, cl_lat, c=side_dist, s = 5, cmap = 'rainbow')
# plt.scatter(cl_lon[zero], cl_lat[zero], c='black', s = 5)
# plt.show()


### BEFORE EDITS
zero = np.where(new_dist_out == 0)[0]
plt.scatter(rch_lon_all, rch_lat_all, c=new_dist_out, s = 5, cmap = 'rainbow')
plt.scatter(rch_lon_all[zero], rch_lat_all[zero], c='black', s = 5)
plt.show()

# df2 = pd.DataFrame(np.array([rch_lon_all, rch_lat_all, new_dist_out]).T)
# df2.to_csv('/Users/ealtenau/Desktop/eu_side_dist_out_test.csv', index=False)


'''
maybe try using the path segs for filling in dist_out?

'''