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

def define_network_regions(add_all, mhv_segs, mhv_facc, pt_ind3):
    unq_paths = np.unique(add_all)
    start_path = np.array([unq_paths[0]])
    flag = np.zeros(len(add_all))
    network = np.zeros(len(mhv_segs))
    check = len(unq_paths) + 5000
    up_nghs = []
    loop = 1
    cnt = 1
    while len(start_path) > 0:
        # print(loop, start_path)
            
        # if 2424 in start_path:
        #     print('segment check')
        #     break
            
        nghs = []
        for n in list(range(len(start_path))):
            pts = np.where(mhv_segs == start_path[n])[0]
            # ends = np.where(mhv_eps[pts] > 0)[0]
            network[pts] = cnt
            ngh_segs = np.unique(mhv_segs[pt_ind3[pts]])
            ngh_facc = np.array([np.median(mhv_facc[np.where(mhv_segs == s)[0]]) for s in ngh_segs])
            ngh_segs = ngh_segs[ngh_facc < np.median(mhv_facc[pts])]
            ngh_flag = np.array([max(mhv_flag[np.where(mhv_segs == s)[0]]) for s in ngh_segs])
            ngh_segs = ngh_segs[ngh_flag == 0]
            nghs.append(ngh_segs)
        nghs = np.array([item for sublist in nghs for item in sublist])
        nghs = nghs[np.where(np.in1d(nghs,start_path)==False)[0]]
        if len(nghs) > 0:
            up_nghs.append(nghs)
            network[np.in1d(mhv_segs, nghs)] = cnt 
            findex = np.where(np.in1d(unq_paths, start_path) == True)[0]
            flag[findex] = 1
            start_path = nghs
            loop = loop+1
        else:
            if len(nghs) == 0:
                findex = np.where(np.in1d(unq_paths, start_path) == True)[0]
                flag[findex] = 1
                if len(unq_paths[flag < 1]) > 0:
                    start_path = np.array([unq_paths[flag < 1][0]])
                    # network[np.in1d(mhv_segs, nghs)] = cnt  
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
        
    return up_nghs, network

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
region = 'NA'
version = 'v18'

# Input file(s).
mhv_dir = '/Users/ealtenau/Documents/SWORD_Dev/inputs/MHV_SWORD/netcdf/' + region +'/'
mhv_files = glob.glob(os.path.join(mhv_dir, '*.nc'))
sword_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
outpath = '/Users/ealtenau/Documents/SWORD_Dev/inputs/MHV_SWORD/gpkg/'+region+'/additions/'

if os.path.exists(outpath) == False:
    os.makedirs(outpath)

sword = nc.Dataset(sword_fn)
clx_all = np.array(sword['/centerlines/x'][:])
cly_all = np.array(sword['/centerlines/y'][:])
cl_id_all = np.array(sword['/centerlines/cl_id'][:])
cl_rchs_all = np.array(sword['/centerlines/reach_id'][0,:])
cl_type_all = np.array([int(str(ind)[-1]) for ind in cl_rchs_all])
rch_hw_all = np.array(sword['/reaches/end_reach'][:])
rchs_all = np.array(sword['/reaches/reach_id'][:])

### need to filter by level 2... 
sword_l2 = np.array([int(str(ind)[0:2]) for ind in cl_rchs_all])
rch_l2 = np.array([int(str(ind)[0:2]) for ind in rchs_all])
mhv_l2 = np.array([int(ind[-13:-11]) for ind in mhv_files])
uniq_level2 = np.unique(sword_l2)
for ind in list(range(13,len(uniq_level2))):
    start = time.time()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('~~~~ STARTING BASIN', uniq_level2[ind], '~~~~')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    sl2 = np.where(sword_l2 == uniq_level2[ind])[0]
    rl2 = np.where(rch_l2 == uniq_level2[ind])[0]
    ml2 = np.where(mhv_l2 == uniq_level2[ind])[0]
    #sword variables
    clx = clx_all[sl2]
    cly = cly_all[sl2]
    cl_rchs = cl_rchs_all[sl2]
    cl_type = cl_type_all[sl2]
    cl_id = cl_id_all[sl2]
    rchs = rchs_all[rl2]
    rch_hw = rch_hw_all[rl2]
    #mhv variables
    mhv = nc.Dataset(mhv_files[ml2[0]], 'r+')
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

    print('Filtering current MHV-SWORD flag')
    mhv_pts = np.vstack((mhv_x, mhv_y)).T
    cl_pts = np.vstack((clx, cly)).T
    kdt = sp.cKDTree(cl_pts)
    pt_dist, pt_ind = kdt.query(mhv_pts, k = 4) #sword points close to mhv. 
    ### filtering current mhv-sword flag. Primarily at tributary junctions. 
    check = np.where(mhv_flag == 1)[0]
    unq_segs = np.unique(mhv_segs[check])
    erase = []
    for seg in list(range(len(unq_segs))):
        pts = np.where(mhv_segs == unq_segs[seg])[0]
        # ends = np.where(mhv_eps[pts] > 0)[0]
        seg_match = np.max(pt_dist[pts], axis = 1)
        coverage = len(np.where(seg_match < 0.005)[0])/len(pts)*100
        if coverage < 20:
            erase.append(unq_segs[seg])
    #zero out flagged mhv reaches with little sword coverage. 
    flag_fix = np.where(np.in1d(mhv_segs, erase) == True)[0]
    mhv_flag[flag_fix] = 0

    print('Identify MHV tributries to add')
    ### identifing mhv reaches that should be added.
    zero = np.where(mhv_flag == 0)[0]
    unq_segs = np.unique(mhv_segs[zero])
    trib_add = []
    for s in list(range(len(unq_segs))):
        pts = np.where(mhv_segs == unq_segs[s])[0]
        # ends = np.where(mhv_eps[pts] > 0)[0]
        seg_match = np.max(pt_dist[pts], axis = 1)
        coverage = len(np.where(seg_match < 0.002)[0])/len(pts)*100 #radius is currently ~200 m 
        if 30 > coverage > 0:
            stypes = np.unique(cl_type[pt_ind[pts]])
            if 5 in stypes:
                continue
            else:
                trib_add.append(unq_segs[s])
    
    print('Identify MHV headwaters to add')
    #add in headwater segments not included in trib search.
    kdt2 = sp.cKDTree(mhv_pts)
    pt_dist2, pt_ind2 = kdt2.query(cl_pts, k = 4) #mhv points close to sword.
    hw = np.where(rch_hw == 1)[0]
    cl_hw = np.where(np.in1d(cl_rchs, rchs[hw])==True)[0]
    # pt_dist2[cl_hw,:]
    max_dist = np.max(pt_dist2[cl_hw], axis = 1)
    add = np.where(max_dist < 0.002)[0] #radius is currently ~200 m 
    hw_segs = np.unique(mhv_segs[pt_ind2[cl_hw[add]]])
    add_segs = hw_segs[np.where(np.in1d(hw_segs, trib_add)== False)[0]]
    #add new segments to tributary segments. 
    trib_add = np.array(trib_add)
    add_segs = np.array(add_segs)
    add_all = np.append(trib_add,add_segs)

    print('Identify upstream MHV reaches')
    #trace upstream mhv reaches. 
    pt_dist3, pt_ind3 = kdt2.query(mhv_pts, k = 10)
    up_nghs, network = define_network_regions(add_all, mhv_segs, mhv_facc, pt_ind3)
    up_nghs = np.unique(np.array([item for sublist in up_nghs for item in sublist]))
    ngh_add = up_nghs[np.where(np.in1d(up_nghs, add_all)== False)[0]]
    # additions = np.append(add_all, ngh_add)

    print('Filter short MHV reach additions')
    #narrow down areas that only have a total network length > 50 km?
    unq_net = np.unique(network)
    unq_net = unq_net[unq_net>0]
    rmv_segs = []
    for net in list(range(len(unq_net))):
        pts = np.where(network == unq_net[net])[0]
        net_segs = np.unique(mhv_segs[pts])
        seg_dist = np.array([np.max(mhv_dist[np.where(mhv_segs == s)[0]]) for s in net_segs])
        net_dist = sum(seg_dist)
        if net_dist < 50000:
            rmv_segs.append(net_segs)
    rmv_segs = np.unique(np.array([item for sublist in rmv_segs for item in sublist]))
    # len(rmv_segs)/len(additions)*100

    ngh_add = ngh_add[np.where(np.in1d(ngh_add, rmv_segs)== False)[0]]
    add_all = add_all[np.where(np.in1d(add_all, rmv_segs)== False)[0]]
    # additions = additions[np.where(np.in1d(additions, rmv_segs)== False)[0]]
    add_flag = np.zeros(len(mhv_segs))
    add_flag[np.where(np.in1d(mhv_segs, add_all)==True)[0]] = 1
    add_flag[np.where(np.in1d(mhv_segs, ngh_add)==True)[0]] = 2

    print('Identify joining points')
    ### isolate the "joining points" of the connecting segments. Label them with a value
    rmv_net = []
    for a in list(range(len(add_all))):
        pts = np.where(mhv_segs == add_all[a])[0]
        #if condition for headwater vs tributary conditions.
        end_dist = np.where(pt_dist[pts,0] < 0.002)[0]
        types = cl_type[pt_ind[pts[end_dist],0]]
        if 5 in types:
            add_flag[pts] = 0
            continue 
        if 6 in types:
            #for headwater joins... 
            end_rch = np.where(types == 6)[0] #find the headwater ghost reach.
            hw_rch = np.unique(cl_rchs[pt_ind[pts[end_dist[end_rch]],0]])
            if len(hw_rch) > 1:
                rmv_net.append(add_all[a])
                continue
            else:
                hw_pts = np.where(cl_rchs == hw_rch)[0] #identify the sword centerline points for the headwater reach.
                hw_max = np.where(cl_id[hw_pts] == np.max(cl_id[hw_pts]))[0] #find the maxium centerline index for the headwater reach. 
                hw_min_pts = np.where(pt_dist2[hw_pts[hw_max],:] < 0.0015)[1] #find the closest mhv point to the maxium centerline index for the headwater reach.
                if len(hw_min_pts) == 0:
                    sind = np.where(mhv_segs[pt_ind2[hw_pts[hw_max],:]] == add_all[a])[1]
                    if len(sind) == 0:
                        rmv_net.append(add_all[a])
                        continue
                    else:
                        hw_min_pts = np.where(pt_dist2[hw_pts[hw_max],sind] == min(pt_dist2[hw_pts[hw_max],sind]))[0]
                        add_flag[pt_ind2[hw_pts[hw_max],sind[hw_min_pts]]] = 3
                else:
                    sind = np.where(mhv_segs[pt_ind2[hw_pts[hw_max],hw_min_pts]] == add_all[a])[0]
                    if len(sind) == 0:
                        rmv_net.append(add_all[a])
                        continue
                    else:
                        max_ind = np.where(mhv_ind[pts] == max(mhv_ind[pt_ind2[hw_pts[hw_max],hw_min_pts[sind]]]))[0] 
                        add_flag[pts[max_ind]] = 3
        else:
            #for tributary additions just find closest mhv point to sword. 
            min_pt = np.where(pt_dist[pts,0] < 0.0015)[0]
            if len(min_pt) == 0:
                min_pt = np.where(pt_dist[pts,0] == min(pt_dist[pts,0]))[0]
                add_flag[pts[min_pt]] = 3
            else:
                max_ind = np.where(mhv_ind[pts] == max(mhv_ind[pts[min_pt]]))[0]
                add_flag[pts[max_ind]] = 3
    erse_net = np.unique(network[np.where(np.in1d(mhv_segs, rmv_net)== True)[0]])
    erse_net_pts = np.where(np.in1d(network, erse_net) == True)[0]
    if len(erse_net_pts) > 0:
        add_flag[erse_net_pts] = 0
    
    ### log the closest sword points (cl_id...). Try to do the most grunt work here. 
    ### recalculate seg dist?
    print('Filter narrow MHV reach additions')
    junc = np.where(add_flag == 3)[0]
    unq_segs = np.unique(mhv_segs[junc])
    junc_wth = np.array([int(np.median(mhv_wth[np.where(mhv_segs == s)])) for s in unq_segs])
    seg_keep = unq_segs[np.where(junc_wth >= 30)[0]]
    net_keep = np.unique(network[np.where(np.in1d(mhv_segs, seg_keep)==True)[0]])
    add_flag_cp = np.copy(add_flag)
    add_flag[np.where(np.in1d(network, net_keep)==False)[0]] = 0

    print('Erasing unecessary MHV points')
    unq_segs = np.unique(mhv_segs[np.where(add_flag == 3)[0]])
    for s in list(range(len(unq_segs))):
        pts = np.where(mhv_segs == unq_segs[s])[0]
        break_ind = mhv_ind[pts[np.where(add_flag[pts] == 3)[0]]]
        erse_pts = np.where(mhv_ind[pts] < break_ind)[0]
        add_flag[pts[erse_pts]] = 0

    print('Identifying potential index problems')
    unq_segs = np.unique(mhv_segs[np.where(add_flag > 0)[0]])
    idx_problems = []
    for s in list(range(len(unq_segs))):
        pts = np.where(mhv_segs == unq_segs[s])[0]
        sort_inds = np.argsort(mhv_ind[pts])
        dist = mhv_dist[pts[sort_inds]]
        diff_dist = np.diff(dist)
        if max(diff_dist) > 500:
            idx_problems.append(np.unique(network[pts]))
    idx_erse = np.where(np.in1d(network, idx_problems) == True)[0]
    if len(idx_erse) > 0:
        add_flag[idx_erse] = 0
    
    print('Identifying centerline mismatches')
    unq_segs = np.unique(mhv_segs[np.where(add_flag == 3)[0]])
    overlap_problem = []
    one_pt_segs = []
    for s in list(range(len(unq_segs))):
        pts = np.where(mhv_segs == unq_segs[s])[0]
        keep_idx = np.where(add_flag[pts] > 0)[0]
        #if keep_idx == 1 - remove the segment...
        if len(keep_idx) == 1:
            one_pt_segs.append(unq_segs[s])
        near = np.where(pt_dist[pts[keep_idx],0] < 0.001)[0]
        if len(near) > 10:
            # record problem segments. 
            overlap_problem.append(unq_segs[s])
            max_id = max(mhv_ind[pts[keep_idx[near]]]) #put max index as new joining point...
            add_flag[pts[np.where(add_flag[pts] == 3)]] = max(add_flag[pts[add_flag[pts] != 3]])
            add_flag[pts[np.where(mhv_ind[pts] == max_id)]] = 3
    #erasing one point segments - typically at junctions. 
    one_pt_erse = np.where(np.in1d(mhv_segs, one_pt_segs) == True)[0]
    if len(one_pt_erse) > 0:
        add_flag[one_pt_erse] = 0

    print('Removing any flagged reaches')
    #removing any reaches that are flagged from reach creation. 
    plus = np.where(add_flag > 0)[0]
    rmv_flag = np.unique(network[np.where(mhv_rch_flag[plus] == 1)[0]])
    flag_erse = np.where(np.in1d(network, rmv_flag) == True)[0]
    if len(flag_erse) > 0:
        add_flag[flag_erse] = 0

    #write csv file of segments to check.
    check_csv = {'new_seg_id': np.array(overlap_problem).astype('int64')}
    check_csv = pd.DataFrame(check_csv)
    csv_outdir = outpath + '/hb'+str(uniq_level2[ind])+'_'+version+'_mhv_additions_check.csv'
    check_csv.to_csv(csv_outdir, index=False)

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
            },inplace=True)
    
    points = points.apply(pd.to_numeric, errors='ignore') # points.dtypes
    geom = gp.GeoSeries(map(Point, zip(mhv['/centerlines/x'][add_idx], mhv['/centerlines/y'][add_idx])))
    points['geometry'] = geom
    points = gp.GeoDataFrame(points)
    points.set_geometry(col='geometry')
    points = points.set_crs(4326, allow_override=True)
    outgpkg = outpath + '/mhv_sword_hb'+str(uniq_level2[ind])+'_pts_'+version+'_additions.gpkg'
    if points.shape[0] == 0:
        continue
    else:
        points.to_file(outgpkg, driver='GPKG', layer='points')

    print('Updating NetCDF')
    ### update netcdf with new flag.
    if 'add_flag' in mhv.groups['centerlines'].variables.keys():
        mhv.groups['centerlines'].variables['add_flag'][:] = add_flag
    else:
        mhv.groups['centerlines'].createVariable('add_flag', 'i4', ('num_points',), fill_value=-9999.)
        #populate new variables.
        mhv.groups['centerlines'].variables['add_flag'][:] = add_flag

    end = time.time()
    print('Finished Basin', str(uniq_level2[ind]), 'in', str(np.round((end-start)/60,2)), 'min. Segment additions to check:', str(len(overlap_problem)))

sword.close()
mhv.close()
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

# plt.scatter(mhv_x, mhv_y, s = 3, c = 'black')
# plt.scatter(clx, cly, s = 3, c = 'blue')
# plt.scatter(mhv_x[pts], mhv_y[pts], s = 10, c = 'red')
# plt.scatter(mhv_x[pts[ends]], mhv_y[pts[ends]], s = 10, c = 'gold')
# plt.show()

# p = np.where(np.in1d(mhv_segs, nghs)==True)[0]
# plt.scatter(mhv_x, mhv_y, s = 3, c = 'black')
# plt.scatter(clx, cly, s = 3, c = 'blue')
# plt.scatter(mhv_x[p], mhv_y[p], s = 10, c = 'red')
# plt.show()


# plt.scatter(mhv_x, mhv_y, s = 3, c = 'black')
# plt.scatter(clx, cly, s = 3, c = 'blue')
# plt.scatter(mhv_x[seg[sort_inds[1::]]], mhv_y[seg[sort_inds[1::]]], s = 10, c = test, cmap = 'rainbow')
# plt.show()

# ######

# p = np.where(np.in1d(mhv_segs, add_all)==True)[0]
# plt.scatter(mhv_x, mhv_y, s = 3, c = 'black')
# plt.scatter(clx, cly, s = 3, c = 'blue')
# plt.scatter(mhv_x[p], mhv_y[p], s = 10, c = 'red')
# plt.show()


# p = np.where(np.in1d(mhv_segs, add_all)==True)[0]
# p2 = np.where(np.in1d(mhv_segs, ngh_add)==True)[0]
# plt.scatter(mhv_x, mhv_y, s = 3, c = 'black')
# plt.scatter(clx, cly, s = 3, c = 'blue')
# plt.scatter(mhv_x[p], mhv_y[p], s = 10, c = 'red')
# plt.scatter(mhv_x[p2], mhv_y[p2], s = 5, c = 'gold')
# plt.show()

# p = np.where(add_flag > 0)[0]
# plt.scatter(mhv_x, mhv_y, s = 3, c = 'black')
# plt.scatter(mhv_x[p], mhv_y[p], s = 10, c = 'cyan')
# plt.show()

# plt.scatter(mhv_x, mhv_y, s = 3, c = network, cmap = 'rainbow')


# t0 = np.where(mhv_old_segs_all == 164203)[0]
# m = np.where(mhv_old_segs_all == 164143)[0]
# t1 = np.where(mhv_old_segs_all == 157423)[0]
# t2 = np.where(mhv_old_segs_all == 164083)[0]

# np.median(mhv_facc_all[t0])
# np.median(mhv_facc_all[m])
# np.median(mhv_facc_all[t1])
# np.median(mhv_facc_all[t2])

# max(mhv_facc_all[t0])
# max(mhv_facc_all[m])
# max(mhv_facc_all[t1])
# max(mhv_facc_all[t2])

# mhv_segs_all[m]
# mhv_segs_all[t1]
# mhv_segs_all[t2]


# mhv_segs[pt_ind2[hw_pts[hw_max],hw_min]]
# mhv_ind[pt_ind2[hw_pts[hw_max],hw_min]]


# plt.scatter(clx, cly, s = 3, c = 'blue')
# plt.scatter(mhv_x[pts], mhv_y[pts], s = 10, c = 'red')
# plt.scatter(mhv_x[pts[keep_idx]], mhv_y[pts[keep_idx]], s = 10, c = 'gold')
# plt.show()


# plt.scatter(mhv_x[pts], mhv_y[pts], s = 10, c = mhv_ind[pts], cmap = 'rainbow')
# plt.show()