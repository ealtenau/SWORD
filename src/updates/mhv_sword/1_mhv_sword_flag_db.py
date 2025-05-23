from __future__ import division
import os
main_dir = os.getcwd()
import numpy as np
from scipy import spatial as sp
import netCDF4 as nc
import time
import geopandas as gp
import pandas as pd
from shapely.geometry import Point
import argparse

import warnings
warnings.filterwarnings("ignore") #if code stops working may need to comment out to check warnings. 

###############################################################################

def getListOfFiles(dirName):

    """
    FUNCTION:
        For the given path, gets a recursive list of all files in the directory tree.

    INPUTS
        dirName -- Input directory

    OUTPUTS
        allFiles -- list of files under directory
    """

    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

###############################################################################

def sword_flag(mhv,swd_x,swd_y):
    mhv['points'] = mhv.apply(lambda x: [y for y in x['geometry'].coords], axis=1)
    flag = np.zeros(len(mhv))
    sublinks = np.array(mhv['LINKNO'][np.where(mhv['strmOrder']>=3)[0]])
    for ind in list(range(len(sublinks))):
        # print(ind, len(sublinks))
        link = np.where(mhv['LINKNO'] == sublinks[ind])[0]
        subpts = mhv['points'][int(link)]
        x = np.array([line[0] for line in subpts])
        y = np.array([line[1] for line in subpts])
        seg_ind = np.array(range(1,len(x)+1))
        seg = np.repeat(mhv['LINKNO'][int(link)], len(x))
        so = np.repeat(mhv['strmOrder'][int(link)], len(x))
        up = np.repeat(mhv['fromnode'][int(link)], len(x))
        down = np.repeat(mhv['tonode'][int(link)], len(x))

        sword_pts = np.vstack((swd_x, swd_y)).T
        mhv_pts = np.vstack((x, y)).T
        kdt = sp.cKDTree(sword_pts)
        pt_dist, pt_ind = kdt.query(mhv_pts, k = 20)
        
        good_vals = np.where(pt_dist[:,0] < 0.01)[0]
        perc = (len(good_vals)/len(x))*100

        if perc <= 25:
            flag[int(link)] = 0
            fg = np.repeat(0, len(x))
        else:
            flag[int(link)] = 1
            fg = np.repeat(1, len(x))

        if ind == 0:
            x_all = x
            y_all = y
            seg_all = seg
            so_all = so
            flag_all = fg
            up_all = up
            down_all = down
            seg_ind_all = seg_ind
        else:
            x_all = np.append(x_all,x, axis=0)
            y_all = np.append(y_all,y, axis=0)
            seg_all = np.append(seg_all,seg, axis=0)
            so_all = np.append(so_all,so, axis=0)
            flag_all = np.append(flag_all,fg, axis=0)
            up_all = np.append(up_all,up, axis=0)
            down_all = np.append(down_all,down, axis=0)
            seg_ind_all = np.append(seg_ind_all,seg_ind, axis=0)

    return x_all, y_all, seg_all, seg_ind_all, so_all, up_all, down_all, flag_all, flag

###############################################################################

def identify_hw_junc(seg_pts, up_pts, down_pts):
    seg_hwout_pts = np.zeros(len(seg_pts))
    seg_junc_pts = np.zeros(len(seg_pts))
    unq_nodes = np.unique([up_pts, down_pts])
    for n in list(range(len(unq_nodes))):
        up_idx = np.where(up_pts == unq_nodes[n])[0]
        down_idx = np.where(down_pts == unq_nodes[n])[0]
        in_up = np.unique(seg_pts[up_idx])
        in_down = np.unique(seg_pts[down_idx])
        if len(in_up) > 1 or len(in_down) > 1:
            #label junctions (1 = down junction, 2 = up junction)
            seg_junc_pts[np.where(np.in1d(seg_pts, in_up)==True)[0]] = 2 #downstream junc links have the upstream nodes.
            seg_junc_pts[np.where(np.in1d(seg_pts, in_down)==True)[0]] = 1 #upstream junc links have the downstream nodes.  
        
        if len(in_up) == 0:
            #label as outlet
            seg_hwout_pts[np.where(np.in1d(seg_pts, in_down)==True)[0]] = 2
        if len(in_down) == 0:
            #label as headwater
            seg_hwout_pts[np.where(np.in1d(seg_pts, in_up)==True)[0]] = 1

    return seg_hwout_pts, seg_junc_pts

###############################################################################

def aggregate_segs(seg_pts, seg_hwout_pts, seg_junc_pts, up_pts, down_pts):
    outlets = np.unique(seg_pts[np.where(seg_hwout_pts == 2)[0]])
    network = np.zeros(len(seg_pts))
    flag = np.zeros(len(seg_pts))
    start_seg = np.array([outlets[0]])
    cnt = 1
    loop = 1
    while min(flag) == 0:
        # print(loop, start_seg)
        pts = np.where(seg_pts == start_seg)[0]
        #upstream segment 
        up_nodes = np.unique(up_pts[pts]) 
        up_segs = np.unique(seg_pts[np.where(down_pts == up_nodes)[0]])

        if len(up_segs) == 0: #headwater
            network[pts] = cnt
            flag[pts] = 1
            start_pts = np.unique(seg_pts[np.where((seg_hwout_pts == 2)&(flag == 0))[0]]) #outlets 
            if len(start_pts) > 0:
                start_seg = np.array([start_pts[0]])
                cnt = cnt+1
                loop = loop+1
            elif len(start_pts) == 0:
                start_pts = np.unique(seg_pts[np.where((seg_junc_pts == 1)&(flag == 0))[0]]) #junctions
                if len(start_pts) > 0:
                    start_seg = np.array([start_pts[0]])
                    cnt = cnt+1
                    loop = loop+1
                else:
                    start_pts = np.unique(seg_pts[np.where(flag == 0)[0]]) #junctions
                    if len(start_pts) > 0:
                        start_seg = np.array([start_pts[0]])
                        cnt = cnt+1
                        loop = loop+1
                    else:
                        loop = loop+1
                        continue
        elif len(up_segs) == 1: #normal
            network[pts] = cnt
            flag[pts] = 1
            start_seg = up_segs
            loop = loop+1
        else: #junction
            network[pts] = cnt
            flag[pts] = 1
            start_pts = np.unique(seg_pts[np.where((seg_junc_pts == 1)&(flag == 0))[0]]) #junctions 
            if len(start_pts) > 0:
                start_seg = np.array([start_pts[0]])
                cnt = cnt+1
                loop = loop+1
            elif len(start_pts) == 0:
                start_pts = np.unique(seg_pts[np.where((seg_hwout_pts == 2)&(flag == 0))[0]]) #junctions
                if len(start_pts) > 0:
                    start_seg = np.array([start_pts[0]])
                    cnt = cnt+1
                    loop = loop+1
                else:
                    start_pts = np.unique(seg_pts[np.where(flag == 0)[0]]) #junctions
                    if len(start_pts) > 0:
                        start_seg = np.array([start_pts[0]])
                        cnt = cnt+1
                        loop = loop+1
                    else:
                        loop = loop+1
                        continue

        if loop > len(np.unique(seg_pts))+500:
            print('LOOP STUCK')
            break
    
    return network

###############################################################################

def update_indexes(seg_pts, seg_ind_pts, up_pts, down_pts,
                   seg_hwout_pts, seg_junc_pts, new_segs):
    unq_segs = np.unique(new_segs)
    new_indexes = np.copy(seg_ind_pts)
    for s in list(range(len(unq_segs))):
        # print(s, len(unq_segs)-1)
        pts = np.where(new_segs == unq_segs[s])[0]
        subsegs = np.unique(seg_pts[pts])
        if len(subsegs) == 1:
            continue
        else:
            #order subsegs
            ds_out = np.unique(seg_pts[pts[np.where(seg_hwout_pts[pts] == 2)[0]]])
            ds_juncs = np.unique(seg_pts[pts[np.where(seg_junc_pts[pts] == 1)[0]]])
            ds = np.unique(np.append(ds_out,ds_juncs))
            us_out = np.unique(seg_pts[pts[np.where(seg_hwout_pts[pts] == 1)[0]]])
            us_juncs = np.unique(seg_pts[pts[np.where(seg_junc_pts[pts] == 2)[0]]])
            us = np.unique(np.append(us_out,us_juncs))
            if len(ds) > 0:
                next_seg = ds
                order = np.zeros(len(subsegs))
                cnt = 1
                while min(order) == 0:
                    order[np.where(subsegs == next_seg)[0]] = cnt
                    subpts = np.where(seg_pts == next_seg)[0]
                    up_nodes = np.unique(up_pts[subpts])
                    up_segs = np.unique(seg_pts[np.where(down_pts == up_nodes)[0]])
                    if len(up_segs) > 0:
                        if True in np.unique(np.in1d(up_segs,subsegs)):
                            next_seg = up_segs
                            cnt = cnt+1
                        else:
                            continue
                    else:
                        continue
                    if cnt > len(subsegs)+5:
                        print('LOOP STUCK')
                        break
            else:
                next_seg = us
                order = np.zeros(len(subsegs))
                cnt = 1
                while min(order) == 0:
                    order[np.where(subsegs == next_seg)[0]] = cnt
                    subpts = np.where(seg_pts == next_seg)[0]
                    dn_nodes = np.unique(down_pts[subpts])
                    dn_segs = np.unique(seg_pts[np.where(up_pts == dn_nodes)[0]])
                    if len(dn_segs) > 0:
                        if True in np.unique(np.in1d(dn_segs,subsegs)):
                            next_seg = dn_segs
                            cnt = cnt+1
                        else:
                            continue
                    else:
                        continue
                    if cnt > len(subsegs)+5:
                        print('LOOP STUCK')
                        break
                
                order = order[::-1]
            
            #update the indexes 
            segs_ordered = subsegs[np.argsort(order)]
            for idx in list(range(len(subsegs))):
                if idx == 0:
                    new_indexes[np.where(seg_pts == segs_ordered[idx])[0]] = seg_ind_pts[np.where(seg_pts == segs_ordered[idx])[0]]
                else:
                    add_val = max(new_indexes[np.where(seg_pts == segs_ordered[idx-1])[0]])
                    current_pts = np.where(seg_pts[pts] == segs_ordered[idx])[0]
                    add_ind = seg_ind_pts[pts[current_pts]]+add_val
                    new_indexes[pts[current_pts]] = add_ind

    return new_indexes

###############################################################################

def find_neighbors(basin_rch, basin_flag, basin_x, basin_y, 
                   rch_x, rch_y, rch_ind, rch_id, rch):

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
    if len(rch) <= 4:
        pt_dist, pt_ind = kdt.query(ep_pts, k = 4, distance_upper_bound = 0.003) #distance upper bound = 300.0 for meters 
    else:#elif rch_len > 600:
        pt_dist, pt_ind = kdt.query(ep_pts, k = 10, distance_upper_bound = 0.003) #distance upper bound = 300.0 for meters 

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
    ep1_flg = np.zeros(len(ep1_ngb))
    for idy in list(range(len(ep1_ngb))):
        ep1_flg[idy] = np.max(basin_flag[np.where(basin_rch == ep1_ngb[idy])])

    ep2_flg = np.zeros(len(ep2_ngb))
    for idy in list(range(len(ep2_ngb))):
        ep2_flg[idy] = np.max(basin_flag[np.where(basin_rch == ep2_ngb[idy])])

    # Creating final arrays.
    ep1 = np.array([ep1_ngb, ep1_flg]).T
    ep2 = np.array([ep2_ngb, ep2_flg]).T

    return ep1, ep2

###############################################################################

def filter_sword_flag(seg_pts, seg_ind_pts,flag_pts, x_pts, y_pts):
    cnt=[]
    flag = np.copy(flag_pts)
    check = np.unique(seg_pts[np.where(flag == 0)[0]])
    for s in list(range(len(check))):
        # print(s, len(check)-1)
        line = np.where(seg_pts == check[s])[0]
        # seg_x = x[line]
        # seg_y = y[line]
        seg_lon = x_pts[line]
        seg_lat = y_pts[line]
        seg_ind = seg_ind_pts[line]
        end1, end2 = find_neighbors(seg_pts, flag, x_pts, y_pts, seg_lon, 
                                    seg_lat, seg_ind, check[s], line)
        if len(end1) == 0:
            continue
        elif len(end2) == 0:
            continue
        else:
            # Cond. 1: end 1 has SWORD flag, but end 2 does not. 
            if np.max(end1[:,1]) == 1 and np.max(end2[:,1]) == 0:
                for n in list(range(len(end2))):
                    line2 = np.where(seg_pts == end2[0,0])[0]
                    seg_lon2 = x_pts[line2]
                    seg_lat2 = y_pts[line2]
                    seg_ind2 = seg_ind_pts[line2]
                    ngh_end1, ngh_end2 = find_neighbors(seg_pts, flag, x_pts, y_pts, seg_lon2, 
                                        seg_lat2, seg_ind2, check[s], line2)
                    if n == 0:
                        ngh_end1_all = np.copy(ngh_end1)
                        ngh_end2_all = np.copy(ngh_end2)
                    else:
                        ngh_end1_all = np.concatenate((ngh_end1_all, ngh_end1), axis = 0)
                        ngh_end2_all = np.concatenate((ngh_end2_all, ngh_end2), axis = 0)
                if np.max(ngh_end1_all[:,1]) == 1 or np.max(ngh_end2_all[:,1]) == 1:
                    # print(s, check[s], 'cond.1')
                    flag[line] = 1
                    # flag[line] = 1
                    cnt.append(check[s])
                else:
                    continue
            # Cond. 2: end 2 has SWORD flag, but end 1 does not.
            elif np.max(end1[:,1]) == 0 and np.max(end2[:,1]) == 1:
                for n in list(range(len(end1))):
                    line2 = np.where(seg_pts == end1[0,0])[0]
                    seg_lon2 = x_pts[line2]
                    seg_lat2 = y_pts[line2]
                    seg_ind2 = seg_ind_pts[line2]
                    ngh_end1, ngh_end2 = find_neighbors(seg_pts, flag, x_pts, y_pts, seg_lon2, 
                                        seg_lat2, seg_ind2, check[s], line2)
                    if n == 0:
                        ngh_end1_all = np.copy(ngh_end1)
                        ngh_end2_all = np.copy(ngh_end2)
                    else:
                        ngh_end1_all = np.concatenate((ngh_end1_all, ngh_end1), axis = 0)
                        ngh_end2_all = np.concatenate((ngh_end2_all, ngh_end2), axis = 0)
                if np.max(ngh_end1_all[:,1]) == 1 or np.max(ngh_end2_all[:,1]) == 1:
                    # print(s, check[s], 'cond.2')
                    # flag_all[subset[line]] = 1
                    flag[line] = 1
                    cnt.append(check[s])
                else:
                    continue
            # Cond. 3: Both ends have SWORD flag. 
            elif np.max(end1[:,1]) == 1 and np.max(end2[:,1]) == 1:
                # print(s, check[s], 'cond.3')
                # flag_all[subset[line]] = 1
                flag[line] = 1
                cnt.append(check[s])

            else:
                continue

    return flag, cnt

###############################################################################

def save_mhv_nc(x_pts, y_pts, seg_pts, seg_ind_pts, so_pts, flag_pts, up_pts, down_pts, 
                seg_hwout_pts, seg_junc_pts, new_segs, new_indexes, flag_filt, region, outfile):

    """
    FUNCTION:
        Writes filtered merged NetCDF. Datasets combined include: GRWL,
        MERIT Hydro, GROD, GRanD, HydroBASINS, Global Deltas, SWOT Track
        information, and Prior Lake Database locations.

    INPUTS
        merged -- Object containing merged attributes for the GRWL centerline.
        outfile -- Outpath directory to write the NetCDF file.

    OUTPUTS
        Merged NetCDF file.
    """

    # global attributes
    root_grp = nc.Dataset(outfile, 'w', format='NETCDF4')
    root_grp.x_min = np.min(x_pts)
    root_grp.x_max = np.max(x_pts)
    root_grp.y_min = np.min(y_pts)
    root_grp.y_max = np.max(y_pts)
    root_grp.production_date = time.strftime("%d-%b-%Y %H:%M:%S", time.gmtime()) #utc time

    # subgroups
    cl_grp = root_grp.createGroup('centerlines')

    # dimensions
    root_grp.createDimension('ID', 2)
    cl_grp.createDimension('num_points', len(x_pts))

    ### variables and units

    # root group variables
    Name = root_grp.createVariable('Name', 'S1', ('ID'))
    Name._Encoding = 'ascii'

    # centerline variables
    x = cl_grp.createVariable(
        'x', 'f8', ('num_points',), fill_value=-9999.)
    x.units = 'degrees east'
    y = cl_grp.createVariable(
        'y', 'f8', ('num_points',), fill_value=-9999.)
    y.units = 'degrees north'
    segID = cl_grp.createVariable(
        'segID', 'i8', ('num_points',), fill_value=-9999.)
    segInd = cl_grp.createVariable(
        'segInd', 'i8', ('num_points',), fill_value=-9999.)
    strmorder= cl_grp.createVariable(
        'strmorder', 'i4', ('num_points',), fill_value=-9999.)
    swordflag= cl_grp.createVariable(
        'swordflag', 'i4', ('num_points',), fill_value=-9999.)
    basin= cl_grp.createVariable(
        'basin', 'i8', ('num_points',), fill_value=-9999.)
    upLink = cl_grp.createVariable(
        'up_link', 'i8', ('num_points',), fill_value=-9999.)
    downLink = cl_grp.createVariable(
        'down_link', 'i8', ('num_points',), fill_value=-9999.)
    new_segID = cl_grp.createVariable(
        'new_segs', 'i8', ('num_points',), fill_value=-9999.)
    new_segInd = cl_grp.createVariable(
        'new_segs_ind', 'i8', ('num_points',), fill_value=-9999.)
    swordflag_filt = cl_grp.createVariable(
        'swordflag_filt', 'i4', ('num_points',), fill_value=-9999.)
    head_out = cl_grp.createVariable(
        'headwaters_outlets', 'i4', ('num_points',), fill_value=-9999.)
    junctions = cl_grp.createVariable(
        'junctions', 'i4', ('num_points',), fill_value=-9999.)
        
    # data
    print("saving nc")

    # root group data
    cont_str = nc.stringtochar(np.array([region], 'S2'))
    Name[:] = cont_str

    # centerline data
    x[:] = np.array(x_pts)
    y[:] = np.array(y_pts)
    segID[:] = np.array(seg_pts)
    segInd[:] = np.array(seg_ind_pts)
    strmorder[:] = np.array(so_pts)
    swordflag[:] = np.array(flag_pts)
    basin[:] = np.array(basin)
    upLink[:] = np.array(up_pts)
    downLink[:] = np.array(down_pts)
    new_segID[:] = np.array(new_segs)
    new_segInd[:] = np.array(new_indexes)
    swordflag_filt[:] = np.array(flag_filt)
    head_out[:] = np.array(seg_hwout_pts)
    junctions[:] = np.array(seg_junc_pts)

    root_grp.close()

###############################################################################
###############################################################################
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Region", type = str)
parser.add_argument("version", help="<Required> Version", type = str)
args = parser.parse_args()

region = args.region
version = args.version
# region = 'NA'
# version = 'v18'

outdir = main_dir+'/data/inputs/MHV_SWORD/'
sword_dir = main_dir+'/data/outputs/Reaches_Nodes/'+version+'/netcdf/'
mhv_dir = main_dir+'/data/inputs/MeritHydroVector/'
mhv_files = np.sort(np.array(np.array([file for file in getListOfFiles(mhv_dir) if '.shp' in file])))
mhv_basins = np.array([int(f[-6:-4]) for f in mhv_files])

sword = nc.Dataset(sword_dir+region.lower()+'_sword_'+version+'.nc')
sword_x = sword.groups['centerlines'].variables['x'][:]
sword_y = sword.groups['centerlines'].variables['y'][:]
sword_id = sword.groups['centerlines'].variables['reach_id'][0,:]
sword_l2 = np.array([int(str(ind)[0:2]) for ind in sword_id])
unq_l2 = np.unique(sword_l2)

start_all = time.time()
for ind in list(range(4,len(unq_l2))):
    start_basin = time.time()
    print('Starting Basin ' + str(unq_l2[ind]))
    pts = np.where(sword_l2 == unq_l2[ind])[0]
    swd_x = sword_x[pts]
    swd_y = sword_y[pts]
    swd_id = sword_id[pts]
    f = np.where(mhv_basins == unq_l2[ind])[0]
    mhv = gp.read_file(mhv_files[int(f)])
    
    print('Converting Lines to Points & SWORD Flag')
    start = time.time()
    x_pts, y_pts, seg_pts, seg_ind_pts, so_pts, up_pts, down_pts, flag_pts, flag = sword_flag(mhv,swd_x,swd_y)
    end = time.time()
    print(str((end-start)/60) + ' min')
    
    print('Identifying Headwaters, Outlets, andh Junctions')
    start = time.time()
    seg_hwout_pts, seg_junc_pts = identify_hw_junc(seg_pts, up_pts, down_pts)
    end = time.time()
    print(str((end-start)/60) + ' min')
    
    print('Aggregating MHV Segments')
    start = time.time()
    new_segs = aggregate_segs(seg_pts, seg_hwout_pts, seg_junc_pts, up_pts, down_pts) 
    end = time.time()
    print(str((end-start)/60) + ' min')

    print('Updating Indexes for New Segments')
    start = time.time()
    new_indexes = update_indexes(seg_pts, seg_ind_pts, up_pts, down_pts, 
                                 seg_hwout_pts, seg_junc_pts, new_segs)
    end = time.time()
    print(str((end-start)/60) + ' min')

    print('Filtering SWORD Flag')
    start = time.time()
    flag_filt, count = filter_sword_flag(seg_pts, seg_ind_pts,flag_pts, x_pts, y_pts)
    end = time.time()
    print(str((end-start)/60) + ' min, Segments corrected: ' + str(len(count)))

    print('Creating and Writing GPKG')
    start = time.time()
    mhv_pts = gp.GeoDataFrame([
        x_pts,
        y_pts,
        seg_pts,
        seg_ind_pts,
        so_pts,
        flag_pts,
        up_pts,
        down_pts,
        seg_hwout_pts,
        seg_junc_pts,
        new_segs,
        new_indexes,
        flag_filt,]).T

    mhv_pts.rename(
        columns={
            0:"x",
            1:"y",
            2:"segment",
            3:"segInd",
            4:"strmorder",
            5:"sword_flag",
            6:"upNode",
            7:"downNode",
            8:"hw_out",
            9:"junc",
            10:"new_segID",
            11:"new_segInd",
            12:"sflag_filt"},inplace=True)

    mhv_pts = mhv_pts.apply(pd.to_numeric, errors='ignore') # nodes.dtypes
    geom = gp.GeoSeries(map(Point, zip(x_pts, y_pts)))
    mhv_pts['geometry'] = geom
    mhv_pts = gp.GeoDataFrame(mhv_pts)
    mhv_pts.set_geometry(col='geometry')
    mhv_pts = mhv_pts.set_crs(4326, allow_override=True)
    outgpkg = outdir+'gpkg/'+region+'/mhv_sword_hb'+str(unq_l2[ind])+'_pts_'+version+'.gpkg'
    mhv_pts.to_file(outgpkg, driver='GPKG', layer='mhv_pts')
    end = time.time()
    print(str((end-start)/60) + ' min')

    print('Writing NetCDF')
    outnet = outdir+'netcdf/'+region+'/mhv_sword_hb'+str(unq_l2[ind])+'_pts_'+version+'.nc'
    l2_basin = np.repeat(unq_l2[ind], len(seg_pts))
    save_mhv_nc(x_pts, y_pts, seg_pts, seg_ind_pts, so_pts, flag_pts, up_pts, down_pts, 
                seg_hwout_pts, seg_junc_pts, new_segs, new_indexes, flag_filt, region, outnet)
    end_basin = time.time()
    print('Finished Basin '+ str(unq_l2[ind]) + ' in: ' + str(np.round((end_basin-start_basin)/3600, 2)) + ' hrs')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

end_all=time.time()
print('*** Finished ALL Basins in: ' + str(np.round((end_all-start_all)/3600, 2)) + ' hrs ***')


### Export the Lines (Optional)
    # mhv['sword_flag'] = flag
    # mhv = mhv.drop(columns=['points'])
    # mhv.set_geometry(col='geometry') #removed "inplace=True" option on leopold. 
    # mhv = mhv.set_crs(4326, allow_override=True)
    # mhv.to_file(outdir+region.lower()+'/mhv_sword_hb'+str(unq_l2[ind])+'.gpkg', driver='GPKG', layer='mhv')
