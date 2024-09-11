from __future__ import division
import numpy as np
import time
import netCDF4 as nc
import pandas as pd
from scipy import spatial as sp
import utm 
import argparse
from pyproj import Proj
import geopy.distance

###############################################################################
###############################################################################
###############################################################################

class Object(object):
    """
    FUNCTION:
        Creates class object to assign attributes to.
    """
    pass 

###############################################################################

def read_data(filename):

    centerlines = Object()
    nodes = Object()
    reaches = Object ()
    
    data = nc.Dataset(filename)
    
    centerlines.cl_id = np.array(data.groups['centerlines'].variables['cl_id'][:])
    centerlines.x = np.array(data.groups['centerlines'].variables['x'][:])
    centerlines.y = np.array(data.groups['centerlines'].variables['y'][:])
    centerlines.reach_id = np.array(data.groups['centerlines'].variables['reach_id'][:])
    centerlines.node_id = np.array(data.groups['centerlines'].variables['node_id'][:])
    
    nodes.id = np.array(data.groups['nodes'].variables['node_id'][:])
    nodes.cl_id = np.array(data.groups['nodes'].variables['cl_ids'][:])
    nodes.x = np.array(data.groups['nodes'].variables['x'][:])
    nodes.y = np.array(data.groups['nodes'].variables['y'][:])
    nodes.len = np.array(data.groups['nodes'].variables['node_length'][:])
    nodes.wse = np.array(data.groups['nodes'].variables['wse'][:])
    nodes.wse_var = np.array(data.groups['nodes'].variables['wse_var'][:])
    nodes.wth = np.array(data.groups['nodes'].variables['width'][:])
    nodes.wth_var = np.array(data.groups['nodes'].variables['width_var'][:])
    nodes.grod = np.array(data.groups['nodes'].variables['obstr_type'][:])
    nodes.grod_fid = np.array(data.groups['nodes'].variables['grod_id'][:])
    nodes.hfalls_fid = np.array(data.groups['nodes'].variables['hfalls_id'][:])
    nodes.nchan_max = np.array(data.groups['nodes'].variables['n_chan_max'][:])
    nodes.nchan_mod = np.array(data.groups['nodes'].variables['n_chan_mod'][:])
    nodes.dist_out = np.array(data.groups['nodes'].variables['dist_out'][:])
    nodes.reach_id = np.array(data.groups['nodes'].variables['reach_id'][:])
    nodes.facc = np.array(data.groups['nodes'].variables['facc'][:])
    nodes.lakeflag = np.array(data.groups['nodes'].variables['lakeflag'][:])
    nodes.wth_coef = np.array(data.groups['nodes'].variables['wth_coef'][:])
    nodes.ext_dist_coef = np.array(data.groups['nodes'].variables['ext_dist_coef'][:])
    nodes.max_wth = np.array(data.groups['nodes'].variables['max_width'][:])
    nodes.meand_len = np.array(data.groups['nodes'].variables['meander_length'][:])
    nodes.river_name = np.array(data.groups['nodes'].variables['river_name'][:])
    nodes.manual_add = np.array(data.groups['nodes'].variables['manual_add'][:])
    nodes.sinuosity = np.array(data.groups['nodes'].variables['sinuosity'][:])
    nodes.edit_flag = np.array(data.groups['nodes'].variables['edit_flag'][:])
    nodes.trib_flag = np.array(data.groups['nodes'].variables['trib_flag'][:])

    reaches.id = np.array(data.groups['reaches'].variables['reach_id'][:])
    reaches.cl_id = np.array(data.groups['reaches'].variables['cl_ids'][:])
    reaches.x = np.array(data.groups['reaches'].variables['x'][:])
    reaches.x_min = np.array(data.groups['reaches'].variables['x_min'][:])
    reaches.x_max = np.array(data.groups['reaches'].variables['x_max'][:])
    reaches.y = np.array(data.groups['reaches'].variables['y'][:])
    reaches.y_min = np.array(data.groups['reaches'].variables['y_min'][:])
    reaches.y_max = np.array(data.groups['reaches'].variables['y_max'][:])
    reaches.len = np.array(data.groups['reaches'].variables['reach_length'][:])
    reaches.wse = np.array(data.groups['reaches'].variables['wse'][:])
    reaches.wse_var = np.array(data.groups['reaches'].variables['wse_var'][:])
    reaches.wth = np.array(data.groups['reaches'].variables['width'][:])
    reaches.wth_var = np.array(data.groups['reaches'].variables['width_var'][:])
    reaches.slope = np.array(data.groups['reaches'].variables['slope'][:])
    reaches.rch_n_nodes = np.array(data.groups['reaches'].variables['n_nodes'][:])
    reaches.grod = np.array(data.groups['reaches'].variables['obstr_type'][:])
    reaches.grod_fid = np.array(data.groups['reaches'].variables['grod_id'][:])
    reaches.hfalls_fid = np.array(data.groups['reaches'].variables['hfalls_id'][:])
    reaches.lakeflag = np.array(data.groups['reaches'].variables['lakeflag'][:])
    reaches.nchan_max = np.array(data.groups['reaches'].variables['n_chan_max'][:])
    reaches.nchan_mod = np.array(data.groups['reaches'].variables['n_chan_mod'][:])
    reaches.dist_out = np.array(data.groups['reaches'].variables['dist_out'][:])
    reaches.n_rch_up = np.array(data.groups['reaches'].variables['n_rch_up'][:])
    reaches.n_rch_down = np.array(data.groups['reaches'].variables['n_rch_down'][:])
    reaches.rch_id_up = np.array(data.groups['reaches'].variables['rch_id_up'][:])
    reaches.rch_id_down = np.array(data.groups['reaches'].variables['rch_id_dn'][:])
    reaches.max_obs = np.array(data.groups['reaches'].variables['swot_obs'][:])
    reaches.orbits = np.array(data.groups['reaches'].variables['swot_orbits'][:])
    reaches.facc = np.array(data.groups['reaches'].variables['facc'][:])
    reaches.iceflag = np.array(data.groups['reaches'].variables['iceflag'][:])
    reaches.max_wth = np.array(data.groups['reaches'].variables['max_width'][:])
    reaches.river_name = np.array(data.groups['reaches'].variables['river_name'][:])
    reaches.low_slope = np.array(data.groups['reaches'].variables['low_slope_flag'][:])
    reaches.edit_flag= np.array(data.groups['reaches'].variables['edit_flag'][:])
    reaches.trib_flag = np.array(data.groups['reaches'].variables['trib_flag'][:])

    data.close()    

    return centerlines, nodes, reaches

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

def write_nc(centerlines, outfile):

    start = time.time()

    # global attributes
    root_grp = nc.Dataset(outfile, 'w', format='NETCDF4')
    root_grp.x_min = np.min(centerlines.x)
    root_grp.x_max = np.max(centerlines.x)
    root_grp.y_min = np.min(centerlines.y)
    root_grp.y_max = np.max(centerlines.y)
    root_grp.Name = region
    root_grp.production_date = time.strftime("%d-%b-%Y %H:%M:%S", time.gmtime()) #utc time
    #root_grp.history = 'Created ' + time.ctime(time.time())

    # groups
    cl_grp = root_grp.createGroup('centerlines')

    # dimensions
    #root_grp.createDimension('d1', 2)
    cl_grp.createDimension('num_points', len(centerlines.cl_id))
    cl_grp.createDimension('num_domains', 4)

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
    common = cl_grp.createVariable(
        'common', 'i4', ('num_points',), fill_value=-9999.)

    # saving data
    print("saving nc")

    # centerline data
    cl_id[:] = centerlines.cl_id
    cl_x[:] = centerlines.x
    cl_y[:] = centerlines.y
    reach_id[:,:] = centerlines.neighbors
    common[:] = centerlines.common

    root_grp.close()

    end = time.time()

    print("Ended Saving NetCDF in: ", str(np.round((end-start)/60, 2)), " min")

    return outfile

###############################################################################
###############################################################################
###############################################################################
    
start_all = time.time()

#read in netcdf data. 
parser = argparse.ArgumentParser()
parser.add_argument("region", help="continental region", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("local_processing", help="'True' for local machine, 'False' for server", type = str)
args = parser.parse_args()

region = args.region
version = args.version

if args.local_processing == 'True':
    outdir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'
else:
    outdir = '/afs/cas.unc.edu/depts/geological_sciences/pavelsky/students/ealtenau/SWORD_dev/outputs/Reaches_Nodes/'

outpath = outdir+version+'/'
fn = outpath+'netcdf/'+region.lower()+'_sword_'+version+'.nc'
centerlines, nodes, reaches = read_data(fn)

#spatial query with all centerline points...
sword_pts = np.vstack((centerlines.x, centerlines.y)).T
kdt = sp.cKDTree(sword_pts)
pt_dist, pt_ind = kdt.query(sword_pts, k = 10, distance_upper_bound=0.005)

#find all neighbors.
centerlines.neighbors = find_connections(centerlines, reaches) #re-building topology based on spatial query. 
# centerlines.neighbors = np.copy(centerlines.reach_id)

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
outfile = outpath+'reach_geometry/'+region.lower()+'_sword_'+version+'_connectivity.nc'
write_nc(centerlines, outfile)
print('DONE', len(np.where(centerlines.multi_flag == 2)[0]), len(np.where(centerlines.common == 1)[0]))

























'''
for ind in list(range(len(multi_rchs))):
    rch = np.where(centerlines.neighbors[0,:] == multi_rchs[ind])[0]
    mn_id = rch[np.where(centerlines.cl_id[rch] == np.min(centerlines.cl_id[rch]))[0]]
    mx_id = rch[np.where(centerlines.cl_id[rch] == np.max(centerlines.cl_id[rch]))[0]]
    n1 = centerlines.neighbors[np.where(centerlines.neighbors[:,mn_id] > 0)[0],mn_id]
    n2 = centerlines.neighbors[np.where(centerlines.neighbors[:,mx_id] > 0)[0],mx_id]

    end1=[]
    for e1 in list(range(0,len(n1))):
        if e1 == 0:
            end1.append(n1)
        else:
            r1 = np.where(centerlines.neighbors[0,:] == n1[e1])[0]
            mn1 = r1[np.where(centerlines.cl_id[r1] == np.min(centerlines.cl_id[r1]))[0]]
            mx1 = r1[np.where(centerlines.cl_id[r1] == np.max(centerlines.cl_id[r1]))[0]]

            coords_1 = (centerlines.y[mn_id], centerlines.x[mx_id])
            coords_2 = (centerlines.y[mn1], centerlines.x[mn1])
            coords_3 = (centerlines.y[mx1], centerlines.x[mx1])
            d1 = geopy.distance.geodesic(coords_1, coords_2).m
            d2 = geopy.distance.geodesic(coords_1, coords_3).m

            if d1 < d2:
                end1.append(centerlines.neighbors[np.where(centerlines.neighbors[:,mn1] > 0)[0],mn1])
            else:
                end1.append(centerlines.neighbors[np.where(centerlines.neighbors[:,mx1] > 0)[0],mx1])

    #unlist list 
    end1_rchs = np.array([item for sublist in end1 for item in sublist])
    e1_rch, count = np.unique(end1_rchs, return_counts=True)
        
 '''       

    