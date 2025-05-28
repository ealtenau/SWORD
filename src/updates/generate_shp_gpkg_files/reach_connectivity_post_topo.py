# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import time
from scipy import spatial as sp
import argparse
import geopy.distance
import src.updates.sword_utils as swd 

###############################################################################

def find_connections(centerlines, reaches):
    #loop through break reaches
    unq_rch = np.unique(centerlines.reach_id[0,:])
    neighbors = np.zeros((4, len(centerlines.x)))
    for ind in list(range(len(unq_rch))):
        # print(ind, len(unq_rch)-1)    
        #rebuild the topology from the centerlines
        rch = np.where(centerlines.reach_id[0,:] == unq_rch[ind])[0]
        mn_id = rch[np.where(centerlines.cl_id[rch] == np.min(centerlines.cl_id[rch]))[0]]
        mx_id = rch[np.where(centerlines.cl_id[rch] == np.max(centerlines.cl_id[rch]))[0]]

        topo_rch = np.where(reaches.id == unq_rch[ind])[0]
        topo_up = reaches.rch_id_up[:,topo_rch]
        topo_dn = reaches.rch_id_down[:,topo_rch]
        topo = np.unique(np.array([topo_up,topo_dn]))
        topo = topo[topo>0]

        if len(rch) < 10:
            # print(ind, len(rch))
            num_pts = len(rch)+1
            good_pts1 = np.where(pt_ind[mn_id,0:num_pts] != len(centerlines.x))[1]
            good_pts2 = np.where(pt_ind[mx_id,0:num_pts] != len(centerlines.x))[1]
            pt1 = np.where(np.unique(centerlines.reach_id[0,pt_ind[mn_id,good_pts1]]) != unq_rch[ind])[0]
            pt2 = np.where(np.unique(centerlines.reach_id[0,pt_ind[mx_id,good_pts2]]) != unq_rch[ind])[0]
            n1 = np.unique(centerlines.reach_id[0,pt_ind[mn_id,good_pts1]])[pt1]
            n2 = np.unique(centerlines.reach_id[0,pt_ind[mx_id,good_pts2]])[pt2]
            
            #accounting for topology.
            n1 = n1[np.in1d(n1,topo)]
            n2 = n2[np.in1d(n2,topo)]

            #seeing if any topology reaches are not in the nghs
            all_nghs = np.append(n1,n2)
            missed_topo = topo[np.where(np.in1d(topo, all_nghs)==False)[0]]
            if len(missed_topo) > 0:
                if missed_topo in topo_up:
                    check1 = np.where(np.in1d(topo_dn, n1)==True)[0]
                    check2 = np.where(np.in1d(topo_dn, n2)==True)[0]
                    if len(check1) > 0:
                        n2 = np.append(n2,missed_topo)
                    if len(check2) > 0:
                        n1 = np.append(n1,missed_topo)
                if missed_topo in topo_dn:
                    check1 = np.where(np.in1d(topo_up, n1)==True)[0]
                    check2 = np.where(np.in1d(topo_up, n2)==True)[0]
                    if len(check1) > 0:
                        n2 = np.append(n2,missed_topo)
                    if len(check2) > 0:
                        n1 = np.append(n1,missed_topo)

            if len(n1) >= 4:
                n1 = n1[0:3]

            if len(n2) >= 4:
                n2 = n2[0:3]
            
            n1 = np.reshape(n1,(len(n1),1))
            n2 = np.reshape(n2,(len(n2),1))
            neighbors[1:len(n1)+1, mn_id] = n1
            neighbors[1:len(n2)+1, mx_id] = n2 

        else:
            good_pts1 = np.where(pt_ind[mn_id,:] != len(centerlines.x))[1]
            good_pts2 = np.where(pt_ind[mx_id,:] != len(centerlines.x))[1]
            pt1 = np.where(np.unique(centerlines.reach_id[0,pt_ind[mn_id,good_pts1]]) != unq_rch[ind])[0]
            pt2 = np.where(np.unique(centerlines.reach_id[0,pt_ind[mx_id,good_pts2]]) != unq_rch[ind])[0]
            n1 = np.unique(centerlines.reach_id[0,pt_ind[mn_id,good_pts1]])[pt1]
            n2 = np.unique(centerlines.reach_id[0,pt_ind[mx_id,good_pts2]])[pt2]
            n1 = np.reshape(n1,(len(n1),1))
            n2 = np.reshape(n2,(len(n2),1))
            #accounting for topology.
            n1 = n1[np.in1d(n1,topo)]
            n2 = n2[np.in1d(n2,topo)]

            #trying to account for short reaches. not needed with topology defined.
            # t1 = np.array([str(n)[11] for n in n1])
            # t2 = np.array([str(n)[11] for n in n2])
            
            # if len(n1) > 1:
            #     # if min(len1) < 8:
            #     if '4' in t1:
            #         dist_pts1 = np.where(centerlines.reach_id[0,pt_ind[mn_id,good_pts1]] != unq_rch[ind])[0]
            #         nghs1 = centerlines.reach_id[0,pt_ind[mn_id,good_pts1[dist_pts1]]]
            #         dist1 = pt_dist[mn_id,good_pts1[dist_pts1]]
            #         min_dist1 = np.array([np.min(dist1[np.where(nghs1 == n)]) for n in n1])
            #         # dam_ind1 = np.where(len1 <= 8)[0]
            #         # other_rchs1 = n1[np.where(len1 > 8)]
            #         dam_ind1 = np.where(t1 == '4')[0]
            #         other_rchs1 = n1[np.where(t1 != '4')]
            #         for idx in list(range(len(other_rchs1))):
            #             pt1 = np.where(n1 == other_rchs1[idx])[0]
            #             if min(min_dist1[dam_ind1]) < 0.0005 and min_dist1[pt1] > 0.0009:
            #                 n1 = np.delete(n1, pt1)
            # if len(n2) > 1:
            #     # if min(len2) < 8:
            #     if '4' in t2:
            #         dist_pts2 = np.where(centerlines.reach_id[0,pt_ind[mx_id,good_pts2]] != unq_rch[ind])[0]
            #         nghs2 = centerlines.reach_id[0,pt_ind[mx_id,good_pts2[dist_pts2]]]
            #         dist2 = pt_dist[mx_id,good_pts2[dist_pts2]]
            #         min_dist2 = np.array([np.min(dist2[np.where(nghs2 == n)]) for n in n2])
            #         # dam_ind2 = np.where(len2 <= 8)[0]
            #         # other_rchs2 = n2[np.where(len2 > 8)]
            #         dam_ind2 = np.where(t2 == '4')[0]
            #         other_rchs2 = n2[np.where(t2 != '4')]
            #         for idx in list(range(len(other_rchs2))):
            #             pt2 = np.where(n2 == other_rchs2[idx])[0]
            #             if min(min_dist2[dam_ind2]) < 0.0005 and min_dist2[pt2] > 0.0009: #added min() to first condition in EU. 
            #                 n2 = np.delete(n2, pt2, axis = 0)
                
            n1 = np.reshape(n1,(len(n1),1))
            n2 = np.reshape(n2,(len(n2),1))
            if len(n1) == 4 or len(n1) > 4:
                neighbors[0:4, mn_id] = n1[0:4]
            else:
                neighbors[1:len(n1)+1, mn_id] = n1
            if len(n2) == 4 or len(n2) > 4:
                neighbors[0:4, mn_id] = n2[0:4]
            else:
                neighbors[1:len(n2)+1, mx_id] = n2
    
    neighbors[0,:] = centerlines.reach_id[0,:]
    
    return neighbors 

###############################################################################

def find_common_points(centerlines):
    # function: find_common_points
    multi_pts = np.where(centerlines.multi_flag == 2)[0]
    common = np.zeros(len(centerlines.x), dtype=int)
    for ind in list(range(len(multi_pts))):
        # print(ind, len(multi_pts)-1)
        if common[multi_pts[ind]] == 1:
            continue
        
        #find all neighbors
        nghs = centerlines.neighbors[np.where(centerlines.neighbors[:,multi_pts[ind]] > 0)[0],multi_pts[ind]]
        nghs = nghs[np.in1d(nghs,np.unique(centerlines.neighbors[0,:]))] #added on 9/9/2024 to account for deleted reaches that were still in neighbors...

        #need to loop through and see if any neighbor pts are already common and continue.
        flag=[]
        for n in list(range(0,len(nghs))):
            # print(n)
            if n == 0:
                flag.append(common[multi_pts[ind]])
            else:
                r = np.where(centerlines.neighbors[0,:] == nghs[n])[0]
                mn = r[np.where(centerlines.cl_id[r] == np.min(centerlines.cl_id[r]))[0]]
                mx = r[np.where(centerlines.cl_id[r] == np.max(centerlines.cl_id[r]))[0]]

                coords_1 = (centerlines.y[multi_pts[ind]], centerlines.x[multi_pts[ind]])
                coords_2 = (centerlines.y[mn], centerlines.x[mn])
                coords_3 = (centerlines.y[mx], centerlines.x[mx])
                d1 = geopy.distance.geodesic(coords_1, coords_2).m
                d2 = geopy.distance.geodesic(coords_1, coords_3).m

                if d1 < d2:
                    flag.append(common[mn][0])
                else:
                    flag.append(common[mx][0])
        
        if np.max(flag) == 1:
            continue

        # if no neighbors are common attach topology variables.
        facc = np.zeros(len(nghs))
        wse = np.zeros(len(nghs))
        wth = np.zeros(len(nghs))
        for n in list(range(len(nghs))):
            r = np.where(reaches.id == nghs[n])
            facc[n] =  reaches.facc[r]
            wse[n] =  reaches.wse[r]
            wth[n] = reaches.wth[r]

        f = np.where(facc == np.max(facc))[0]
        h = np.where(wse == np.min(wse))[0]
        w = np.where(wth == np.max(wth))[0]

        if len(f) == 1:
            # print('cond.1')
            if f == 0:
                common[multi_pts[ind]] = 1
        elif len(h) == 1:
            # print('cond.2')
            if h == 0:
                common[multi_pts[ind]] = 1
        elif len(w) == 1:
            # print('cond.3')
            if w == 0:
                common[multi_pts[ind]] = 1
        else:
            # print('cond.4')
            common[multi_pts[ind]] = 1   

    return common

###############################################################################
###############################################################################
###############################################################################
    
start_all = time.time()

#read in netcdf data. 
parser = argparse.ArgumentParser()
parser.add_argument("region", help="continental region", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

paths = swd.prepare_paths(main_dir, region, version)
sword_fn = paths['nc_dir']+paths['nc_fn']
outfile = paths['geom_dir']+paths['geom_fn']
centerlines, nodes, reaches = swd.read_nc(sword_fn)

#spatial query with all centerline points...
sword_pts = np.vstack((centerlines.x, centerlines.y)).T
kdt = sp.cKDTree(sword_pts)
pt_dist, pt_ind = kdt.query(sword_pts, k = 10, distance_upper_bound=0.005)

#find all neighbors.
# centerlines.neighbors = find_connections(centerlines, reaches) #re-building topology based on spatial query. 
centerlines.neighbors = np.copy(centerlines.reach_id)

#flag points with multiple neighbors.
reach_id_binary = np.copy(centerlines.neighbors)
reach_id_binary[np.where(reach_id_binary > 0)] = 1
row_sums = np.sum(reach_id_binary, axis = 0)
multi = np.zeros(len(row_sums))
multi[np.where(row_sums == 2)] = 1
multi[np.where(row_sums > 2)] = 2
centerlines.multi_flag = multi

#find common points. 
centerlines.common = find_common_points(centerlines)

#write separate netcdf.
print('Writing NetCDF')
swd.write_con_nc(centerlines, outfile)
end_all = time.time()

print('DONE', len(np.where(centerlines.multi_flag == 2)[0]), len(np.where(centerlines.common == 1)[0]))
print(str(np.round((end_all-start_all)/60,2))+' mins')
