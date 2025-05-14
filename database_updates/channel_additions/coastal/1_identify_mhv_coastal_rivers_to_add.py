from __future__ import division
import os
import time
import numpy as np
import geopandas as gp
from shapely.geometry import Point
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from statistics import mode
import netCDF4 as nc
from scipy import spatial as sp
from geopy import distance
import glob

###############################################################################
###############################################################################
###############################################################################

def define_network_regions(mhv_segs, pt_ind):
    unq_paths = np.unique(mhv_segs)
    start_path = np.array([unq_paths[0]])
    flag = np.zeros(len(mhv_segs))
    network = np.zeros(len(mhv_segs))
    check = len(unq_paths) + 5000
    loop = 1
    cnt = 1
    while len(start_path) > 0:
        # print(loop, start_path, cnt)
            
        # if 2424 in start_path:
        #     print('segment check')
        #     break
            
        nghs = []
        for n in list(range(len(start_path))):
            # print(n)
            pts = np.where(mhv_segs == start_path[n])[0]
            network[pts] = cnt
            flag[pts] = 1
            good = np.where(pt_ind[pts] < len(mhv_segs))
            ngh_segs = np.unique(mhv_segs[pt_ind[pts[good[0]],good[1]]])
            nghs.append(ngh_segs)
        nghs = np.array([item for sublist in nghs for item in sublist])
        nghs = nghs[np.where(np.in1d(nghs,start_path)==False)[0]]
        nghs = np.unique(nghs)
        if len(nghs) > 0:
            ngh_flag = np.array([np.max(flag[np.where(mhv_segs == s)[0]]) for s in nghs])
            nf_nghs = nghs[ngh_flag == 0]
            if len(nf_nghs) > 0:
                network[np.in1d(mhv_segs, nf_nghs)] = cnt 
                findex = np.where(np.in1d(mhv_segs, nf_nghs) == True)[0]
                flag[findex] = 1
                start_path = nf_nghs
                loop = loop+1
            else:
                if min(flag) == 0:
                    start_path = np.array([np.unique(mhv_segs[np.where(flag == 0)])[0]])
                    cnt = cnt+1
                    loop = loop+1
                else:
                    start_path = []
        else:
            if len(nghs) == 0:
                if min(flag) == 0:
                    start_path = np.array([np.unique(mhv_segs[np.where(flag == 0)])[0]])
                    cnt = cnt+1
                    loop = loop+1
                else:
                    start_path = []
            else:
                start_path = []
                continue

        if loop > check:
                print('LOOP STUCK')
                break
        
    return network

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
###############################################################################
###############################################################################

start_all = time.time()
region = 'OC'
version='v18'

# Input file(s).
mhv_dir = '/Users/ealtenau/Documents/SWORD_Dev/inputs/MHV_SWORD/netcdf/' + region +'/'
mhv_files = np.sort(glob.glob(os.path.join(mhv_dir, '*.nc')))
outpath = '/Users/ealtenau/Documents/SWORD_Dev/inputs/MHV_SWORD/gpkg/'+region+'/coast_additions/'

if os.path.exists(outpath) == False:
    os.makedirs(outpath)

for ind in list(range(len(mhv_files))):
    start = time.time()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('~~~~ STARTING BASIN', mhv_files[ind][-13:-11], '~~~~')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #mhv variables
    mhv = nc.Dataset(mhv_files[ind], 'r+')
    mhv_x = np.array(mhv['/centerlines/x'][:])
    mhv_y = np.array(mhv['/centerlines/y'][:])
    mhv_flag = np.array(mhv['/centerlines/swordflag_filt'][:])
    mhv_segs = np.array(mhv['/centerlines/new_segs'][:])
    mhv_eps = np.array(mhv['/centerlines/endpoints'][:])
    mhv_basins = np.array(mhv['/centerlines/basin_code'][:])
    mhv_strm = np.array(mhv['/centerlines/strmorder'][:])
    mhv_facc = np.array(mhv['/centerlines/flowacc'][:])
    mhv_dist = np.array(mhv['/centerlines/new_segDist'][:])
    mhv_ind = np.array(mhv['/centerlines/new_segs_ind'][:])
    mhv_wth = np.array(mhv['/centerlines/p_width'][:])
    mhv_rch_flag = np.array(mhv['/centerlines/rch_issue_flag'][:])
    mhv_coast = np.array(mhv['/centerlines/coastflag'][:])

    print('Identify coastal MHV reaches to add')
    mhv_pts = np.vstack((mhv_x, mhv_y)).T
    kdt = sp.cKDTree(mhv_pts)
    pt_dist, pt_ind = kdt.query(mhv_pts, k = 10, distance_upper_bound=0.005)
    network = define_network_regions(mhv_segs, pt_ind)
    add_flag = np.zeros(len(mhv_segs))
    coast = np.where(mhv_coast > 0)[0]
    unq_nets = np.unique(network[coast])
    for net in list(range(len(unq_nets))):
        pts = np.where(network == unq_nets[net])[0]
        if max(mhv_flag[pts]) == 0:
            add_flag[pts] = 1

    print('Filtering additions by length')
    add_nets = np.unique(network[np.where(add_flag>0)[0]])
    for n in list(range(len(add_nets))):
        pts2 = np.where(network == add_nets[n])[0]
        if len(pts2) < 50:
            add_flag[pts2] = 0

    print('Removing any flagged reaches')
    #removing any reaches that are flagged from reach creation. 
    plus = np.where(add_flag > 0)[0]
    rmv_flag = np.unique(network[np.where(mhv_rch_flag[plus] == 1)[0]])
    flag_erse = np.where(np.in1d(network, rmv_flag) == True)[0]
    if len(flag_erse) > 0:
        add_flag[flag_erse] = 0

    print('Writing GPKG')
    add_idx = np.where(add_flag > 0)[0]
    #write the gpkg files. 
    points = gp.GeoDataFrame([
        np.array(mhv['/centerlines/x'][add_idx]),
        np.array(mhv['/centerlines/y'][add_idx]),
        np.array(mhv['/centerlines/segID'][add_idx]),
        # np.array(mhv['/centerlines/segID_old'][add_idx]),
        np.array(mhv['/centerlines/strmorder'][add_idx]),
        np.array(mhv['/centerlines/swordflag'][add_idx]),
        np.array(mhv['/centerlines/basin'][add_idx]),
        np.array(mhv['/centerlines/cl_id'][add_idx]),
        np.array(mhv['/centerlines/easting'][add_idx]),
        np.array(mhv['/centerlines/northing'][add_idx]),
        np.array(mhv['/centerlines/segInd'][add_idx]),
        # np.array(mhv['/centerlines/segDist'][add_idx]),
        np.array(mhv['/centerlines/p_width'][add_idx]),
        np.array(mhv['/centerlines/p_height'][add_idx]),
        np.array(mhv['/centerlines/flowacc'][add_idx]),
        np.array(mhv['/centerlines/nchan'][add_idx]),
        np.array(mhv['/centerlines/manual_add'][add_idx]),
        np.array(mhv['/centerlines/endpoints'][add_idx]),
        # np.array(mhv['/centerlines/mh_tile'][add_idx]),
        np.array(mhv['/centerlines/lakeflag'][add_idx]),
        np.array(mhv['/centerlines/deltaflag'][add_idx]),
        np.array(mhv['/centerlines/grand_id'][add_idx]),
        np.array(mhv['/centerlines/grod_id'][add_idx]),
        np.array(mhv['/centerlines/grod_fid'][add_idx]),
        np.array(mhv['/centerlines/hfalls_fid'][add_idx]),
        np.array(mhv['/centerlines/basin_code'][add_idx]),
        np.array(mhv['/centerlines/number_obs'][add_idx]),
        # np.array(mhv['/centerlines/orbits'][add_idx]),
        np.array(mhv['/centerlines/lake_id'][add_idx]),
        np.array(mhv['/centerlines/swordflag_filt'][add_idx]),
        np.array(mhv['/centerlines/reach_id'][add_idx]),
        np.array(mhv['/centerlines/rch_len'][add_idx]),
        np.array(mhv['/centerlines/node_num'][add_idx]),
        np.array(mhv['/centerlines/rch_eps'][add_idx]),
        np.array(mhv['/centerlines/type'][add_idx]),
        np.array(mhv['/centerlines/rch_ind'][add_idx]),
        np.array(mhv['/centerlines/rch_num'][add_idx]),
        np.array(mhv['/centerlines/node_id'][add_idx]),
        np.array(mhv['/centerlines/rch_dist'][add_idx]),
        np.array(mhv['/centerlines/node_len'][add_idx]),
        np.array(mhv['/centerlines/new_segs'][add_idx]),
        np.array(mhv['/centerlines/new_segs_ind'][add_idx]),
        np.array(mhv['/centerlines/new_segDist'][add_idx]),
        # np.array(mhv['/centerlines/new_segs_eps'][add_idx]),
        add_flag[add_idx],
        network[add_idx],        
    ]).T

    #rename columns.
    points.rename(
        columns={
            0:"x",
            1:"y",
            2:"segID",
            # 3:"segID_old",
            3:"strmorder",
            4:"swordflag",
            5:"basin",
            6:"cl_id",
            7:"easting",
            8:"northing",
            9:"segInd",
            # 11:"segDist",
            10:"p_width",
            11:"p_height",
            12:"flowacc",
            13:"nchan",
            14:"manual_add",
            15:"endpoints",
            # 16:"mh_tile",
            16:"lakeflag",
            17:"deltaflag",
            18:"grand_id",
            19:"grod_id",
            20:"grod_fid",
            21:"hfalls_fid",
            22:"basin_code",
            23:"number_obs",
            # 26:"orbits",
            24:"lake_id",
            25:"swordflag_filt",
            26:"reach_id",
            27:"rch_len",
            28:"node_num",
            29:"rch_eps",
            30:"type",
            31:"rch_ind",
            32:"rch_num",
            33:"node_id",
            34:"rch_dist",
            35:"node_len",
            36:"new_segs",
            37:"new_segs_ind",
            38:"new_segs_dist",
            # 42:"new_segs_eps",
            39:"add_flag",
            40:"network",
            },inplace=True)
    
    points = points.apply(pd.to_numeric, errors='ignore') # points.dtypes
    geom = gp.GeoSeries(map(Point, zip(mhv['/centerlines/x'][add_idx], mhv['/centerlines/y'][add_idx])))
    points['geometry'] = geom
    points = gp.GeoDataFrame(points)
    points.set_geometry(col='geometry')
    points = points.set_crs(4326, allow_override=True)
    outgpkg = outpath + '/mhv_sword_hb'+str(mhv_files[ind][-13:-11])+'_pts_'+version+'_coastal_additions.gpkg'
    if points.shape[0] == 0:
        continue
    else:
        points.to_file(outgpkg, driver='GPKG', layer='points')

    print('Updating NetCDF')
    ## update netcdf with new flag.
    if 'add_coast' in mhv.groups['centerlines'].variables.keys():
        mhv.groups['centerlines'].variables['add_coast'][:] = add_flag
        mhv.groups['centerlines'].variables['network_all'][:] = network
    else:
        mhv.groups['centerlines'].createVariable('add_coast', 'i4', ('num_points',), fill_value=-9999.)
        mhv.groups['centerlines'].createVariable('network_all', 'i8', ('num_points',), fill_value=-9999.)
        #populate new variables.
        mhv.groups['centerlines'].variables['add_coast'][:] = add_flag
        mhv.groups['centerlines'].variables['network_all'][:] = network
    mhv.close()
    
    end = time.time()
    print('Finished Basin', str(mhv_files[ind][-13:-11]), 'in', str(np.round((end-start)/60,2)), 'min.')

end_all = time.time()
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('FINISHED ',region, 'IN:', str(np.round((end_all-start_all)/60,2)), 'min')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')


############################################################################################
############################################################################################
############################################################################################

### can be used to check gpkg formats
# feat1 = next(points.iterfeatures())
# for prop in feat1['properties']:
#     print(prop, type(feat1['properties'][prop]))

# p = np.where(network == 0)[0]
# plt.scatter(mhv_x, mhv_y, s = 3, c = 'black')
# plt.scatter(mhv_x[p], mhv_y[p], s = 10, c = 'red')
# plt.show()

# plt.scatter(mhv_x, mhv_y, c = 'blue', s = 5)
# plt.scatter(mhv_x[pts], mhv_y[pts], c = 'red', s = 5)
# plt.show()


# plt.scatter(mhv_x, mhv_y, c = network, cmap = 'rainbow', s = 5)
# plt.show()

# p = np.where(add_flag == 1)[0]
# plt.scatter(mhv_x, mhv_y, s = 3, c = 'blue')
# plt.scatter(mhv_x[p], mhv_y[p], s = 10, c = 'red')
# plt.show()