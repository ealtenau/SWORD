# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import geopandas as gp
import pandas as pd
import argparse
import glob
import argparse
import src.updates.sword_utils as swd 
import src.updates.calc_utils as ct 
# import matplotlib.pyplot as plt

###############################################################################   

def aggregate_files(trib_files):
    
    for f in list(range(len(trib_files))):
        gdf = gp.read_file(trib_files[f])
        if f == 0:
            gdf_all = gdf.copy()
        else:
            gdf_all = pd.concat([gdf_all, gdf], ignore_index=True)

    return gdf_all
    
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version
multi_file = 'True'

paths = swd.prepare_paths(main_dir, region, version)
nc_fn = paths['nc_dir']+paths['nc_fn']
trib_dir = paths['updates_dir']+'/tribs'
trib_files = np.sort(glob.glob(os.path.join(trib_dir, '*.gpkg')))

centerlines, nodes, reaches = swd.read_nc(nc_fn)

cl_level6 = np.array([str(ind)[0:6] for ind in centerlines.node_id[0,:]])
cl_node_num_int = np.array([int(str(ind)[10:13]) for ind in centerlines.node_id[0,:]])
cl_rch_type = np.array([str(ind)[-1] for ind in centerlines.node_id[0,:]])

#automatic
if multi_file == 'True':
    tribs = aggregate_files(trib_files)
else:
    tribs = gp.read_file(trib_files[0]) 
reach = np.array(tribs['reach_id']) 
break_id = np.array(tribs['cl_id']) 

#manual
# reach = np.array([13265000111])
# break_id = np.array([14147935])

unq_rchs = np.unique(reach)
for r in list(range(len(unq_rchs))):
    print(r, unq_rchs[r], len(unq_rchs)-1)
    cl_r = np.where(centerlines.reach_id[0,:] == unq_rchs[r])[0]
    order_ids = np.argsort(centerlines.cl_id[cl_r])
    old_dist = reaches.dist_out[np.where(reaches.id == unq_rchs[r])[0]]
    old_len = reaches.len[np.where(reaches.id == unq_rchs[r])[0]]
    base_val = old_dist - old_len

    breaks = break_id[np.where(reach == unq_rchs[r])[0]]
    break_pts = np.array([np.where(centerlines.cl_id[cl_r[order_ids]] == b)[0][0] for b in breaks])

    #append start and end points. 
    bounds = np.append(0,break_pts)
    bounds = np.append(bounds, len(cl_r))
    bounds = np.sort(bounds) #added 4/26/24
    bounds = np.unique(bounds) #added 6/7/24

    new_divs = np.zeros(len(cl_r))
    count = 1
    for b in list(range(len(bounds)-1)):
        update_nds = cl_r[order_ids[bounds[b]:bounds[b+1]]]
        nds = np.unique(centerlines.node_id[0,update_nds])
        fill = np.where(np.in1d(centerlines.node_id[0,cl_r[order_ids]], nds) == True)[0]
        if np.max(new_divs[fill])==0:
            new_divs[fill] = count 
            count = count+1
        else:
            z = np.where(new_divs[fill] == 0)[0]
            new_divs[fill[z]] = count
            count = count+1

    #loop through bounds-1... 
    unq_divs = np.unique(new_divs)
    if len(unq_divs) == 1:
        continue
    else:
        for d in list(range(len(unq_divs))):
            # print('b', b)
            if d == 0:
                # print('1')
                update_ids = cl_r[order_ids[np.where(new_divs == unq_divs[d])]]
                new_cl_rch_id = centerlines.reach_id[0,update_ids]
                new_cl_node_ids = centerlines.node_id[0,update_ids]
                new_rch_id = np.unique(centerlines.reach_id[0,update_ids])[0]
            else:
                # print('2')
                #Create New Reach ID
                update_ids = cl_r[order_ids[np.where(new_divs == unq_divs[d])]]
                old_nodes = np.unique(centerlines.node_id[0,update_ids])
                old_rch = np.unique(centerlines.reach_id[0,update_ids])[0]
                l6_basin = np.where(cl_level6 == np.unique(cl_level6[update_ids]))[0]
                cl_rch_num_int = np.array([int(str(ind)[6:10]) for ind in centerlines.node_id[0,l6_basin]])
                new_rch_num = np.max(cl_rch_num_int)+1
                if len(str(new_rch_num)) == 1:
                    fill = '000'
                    new_rch_id = int(str(np.unique(cl_level6[update_ids])[0])+fill+str(new_rch_num)+str(np.unique(cl_rch_type[update_ids])[0]))
                if len(str(new_rch_num)) == 2:
                    fill = '00'
                    new_rch_id = int(str(np.unique(cl_level6[update_ids])[0])+fill+str(new_rch_num)+str(np.unique(cl_rch_type[update_ids])[0]))
                if len(str(new_rch_num)) == 3:
                    fill = '0'
                    new_rch_id = int(str(np.unique(cl_level6[update_ids])[0])+fill+str(new_rch_num)+str(np.unique(cl_rch_type[update_ids])[0]))
                if len(str(new_rch_num)) == 4:
                    new_rch_id = int(str(np.unique(cl_level6[update_ids])[0])+str(new_rch_num)+str(np.unique(cl_rch_type[update_ids])[0]))
                new_cl_rch_id = np.repeat(new_rch_id, len(update_ids))

                # print('3')
                #Create New Node IDs 
                new_cl_node_ids = np.zeros(len(update_ids),dtype=int)
                new_cl_node_nums = cl_node_num_int[update_ids] - np.min(cl_node_num_int[update_ids]) + 1
                for n in list(range(len(new_cl_node_nums))):
                    if len(str(new_cl_node_nums[n])) == 1:
                        fill = '00'
                        new_cl_node_ids[n] = int(str(new_rch_id)[0:-1]+fill+str(new_cl_node_nums[n])+str(new_rch_id)[-1])
                    if len(str(new_cl_node_nums[n])) == 2:
                        fill = '0'
                        new_cl_node_ids[n] = int(str(new_rch_id)[0:-1]+fill+str(new_cl_node_nums[n])+str(new_rch_id)[-1])
                    if len(str(new_cl_node_nums[n])) == 3:
                        new_cl_node_ids[n] = int(str(new_rch_id)[0:-1]+fill+str(new_cl_node_nums[n])+str(new_rch_id)[-1])

            x_coords = centerlines.x[update_ids]
            y_coords = centerlines.y[update_ids]
            diff = ct.get_distances(x_coords,y_coords)
            dist = np.cumsum(diff)
            
            # print('5')
            new_rch_len = np.max(dist)
            new_rch_x = np.median(centerlines.x[update_ids])
            new_rch_y = np.median(centerlines.y[update_ids])
            new_rch_x_max = np.max(centerlines.x[update_ids])
            new_rch_x_min = np.min(centerlines.x[update_ids])
            new_rch_y_max = np.max(centerlines.y[update_ids])
            new_rch_y_min = np.min(centerlines.y[update_ids])

            # print('6')
            unq_nodes = np.unique(new_cl_node_ids)
            new_node_len = np.zeros(len(unq_nodes))
            new_node_x = np.zeros(len(unq_nodes))
            new_node_y = np.zeros(len(unq_nodes))
            new_node_id = np.zeros(len(unq_nodes))
            new_node_cl_ids = np.zeros((2,len(unq_nodes)))
            for n2 in list(range(len(unq_nodes))):
                pts = np.where(new_cl_node_ids == unq_nodes[n2])[0]
                new_node_x[n2] = np.median(centerlines.x[update_ids[pts]])
                new_node_y[n2] = np.median(centerlines.y[update_ids[pts]])
                new_node_len[n2] = max(np.cumsum(diff[pts]))
                new_node_id[n2] = unq_nodes[n2]
                new_node_cl_ids[0,n2] = np.min(centerlines.cl_id[update_ids[pts]])
                new_node_cl_ids[1,n2] = np.max(centerlines.cl_id[update_ids[pts]])
                if len(pts) == 1:
                    new_node_len[n2] = 30
            
            if new_rch_id in reaches.id:
                # print('7')
                node_ind = np.where(np.in1d(nodes.id, new_node_id)==True)[0]
                nodes.len[node_ind] = new_node_len
                nodes.cl_id[:,node_ind] = new_node_cl_ids
                nodes.x[node_ind] = new_node_x
                nodes.y[node_ind] = new_node_y
                
                rch = np.where(reaches.id == new_rch_id)[0]
                reaches.cl_id[0,rch] = np.min(centerlines.cl_id[update_ids])
                reaches.cl_id[1,rch] = np.max(centerlines.cl_id[update_ids])
                reaches.x[rch] = new_rch_x
                reaches.x_min[rch] = new_rch_x_min
                reaches.x_max[rch] = new_rch_x_max
                reaches.y[rch] = new_rch_y
                reaches.y_min[rch] = new_rch_y_min
                reaches.y_max[rch] = new_rch_y_max
                reaches.len[rch] = new_rch_len
                reaches.rch_n_nodes[rch] = len(new_node_id)
                if reaches.edit_flag[rch] == 'NaN':
                    edit_val = '6'
                elif '6' not in reaches.edit_flag[rch][0].split(','):
                    edit_val = reaches.edit_flag[rch] + ',6'
                else:
                    edit_val = reaches.edit_flag[rch]
                reaches.edit_flag[rch] = edit_val
                nodes.edit_flag[node_ind] = edit_val

            else:
                # print('8')
                centerlines.reach_id[0,update_ids] = new_cl_rch_id
                centerlines.node_id[0,update_ids] = new_cl_node_ids
                
                old_ind = np.where(np.in1d(nodes.id, old_nodes) == True)[0]
                nodes.id[old_ind] = new_node_id
                nodes.len[old_ind] = new_node_len
                nodes.cl_id[:,old_ind] = new_node_cl_ids
                nodes.x[old_ind] = new_node_x
                nodes.y[old_ind] = new_node_y
                nodes.reach_id[old_ind] = np.repeat(new_rch_id, len(new_node_id))
                
                rch = np.where(reaches.id == old_rch)[0]
                reaches.id = np.append(reaches.id, new_rch_id)
                new_cl_ids = np.array([np.min(centerlines.cl_id[update_ids]), np.max(centerlines.cl_id[update_ids])]).reshape(2,1)
                reaches.cl_id = np.append(reaches.cl_id, new_cl_ids, axis=1)
                reaches.x = np.append(reaches.x, new_rch_x)
                reaches.x_min = np.append(reaches.x_min, new_rch_x_min)
                reaches.x_max = np.append(reaches.x_max, new_rch_x_max)
                reaches.y = np.append(reaches.y, new_rch_y)
                reaches.y_min = np.append(reaches.y_min, new_rch_y_min)
                reaches.y_max = np.append(reaches.y_max, new_rch_y_max)
                reaches.len = np.append(reaches.len, new_rch_len)
                reaches.rch_n_nodes = np.append(reaches.rch_n_nodes, len(new_node_id))
                #fill attribute with current values. 
                reaches.wse = np.append(reaches.wse, reaches.wse[rch])
                reaches.wse_var = np.append(reaches.wse_var, reaches.wse_var[rch])
                reaches.wth = np.append(reaches.wth, reaches.wth[rch])
                reaches.wth_var = np.append(reaches.wth_var, reaches.wth_var[rch])
                reaches.slope = np.append(reaches.slope, reaches.slope[rch])
                reaches.grod = np.append(reaches.grod, reaches.grod[rch])
                reaches.grod_fid = np.append(reaches.grod_fid, reaches.grod_fid[rch])
                reaches.hfalls_fid = np.append(reaches.hfalls_fid, reaches.hfalls_fid[rch])
                reaches.lakeflag = np.append(reaches.lakeflag, reaches.lakeflag[rch])
                reaches.nchan_max = np.append(reaches.nchan_max, reaches.nchan_max[rch])
                reaches.nchan_mod = np.append(reaches.nchan_mod, reaches.nchan_mod[rch])
                reaches.dist_out = np.append(reaches.dist_out, reaches.dist_out[rch])
                reaches.n_rch_up = np.append(reaches.n_rch_up, reaches.n_rch_up[rch])
                reaches.n_rch_down = np.append(reaches.n_rch_down, reaches.n_rch_down[rch])
                reaches.rch_id_up = np.append(reaches.rch_id_up, reaches.rch_id_up[:,rch], axis=1)
                reaches.rch_id_down = np.append(reaches.rch_id_down, reaches.rch_id_down[:,rch], axis=1)
                reaches.max_obs = np.append(reaches.max_obs, reaches.max_obs[rch])
                reaches.orbits = np.append(reaches.orbits, reaches.orbits[:,rch], axis=1)
                reaches.facc = np.append(reaches.facc, reaches.facc[rch])
                reaches.iceflag = np.append(reaches.iceflag, reaches.iceflag[:,rch], axis=1)
                reaches.max_wth = np.append(reaches.max_wth, reaches.max_wth[rch])
                reaches.river_name = np.append(reaches.river_name, reaches.river_name[rch])
                reaches.low_slope = np.append(reaches.low_slope, reaches.low_slope[rch])
                reaches.trib_flag = np.append(reaches.trib_flag, reaches.trib_flag[rch])
                reaches.path_freq = np.append(reaches.path_freq, reaches.path_freq[rch])
                reaches.path_order = np.append(reaches.path_order, reaches.path_order[rch])
                reaches.main_side = np.append(reaches.main_side, reaches.main_side[rch])
                reaches.path_segs = np.append(reaches.path_segs, reaches.path_segs[rch])
                reaches.strm_order = np.append(reaches.strm_order, reaches.strm_order[rch])
                reaches.end_rch = np.append(reaches.end_rch, reaches.end_rch[rch])
                reaches.network = np.append(reaches.network, reaches.network[rch])
                if reaches.edit_flag[rch] == 'NaN':
                    edit_val = '6'
                elif '6' not in reaches.edit_flag[rch][0].split(','):
                    edit_val = reaches.edit_flag[rch] + ',6'
                else:
                    edit_val = reaches.edit_flag[rch]
                reaches.edit_flag = np.append(reaches.edit_flag, edit_val)
                nodes.edit_flag[old_ind] = edit_val
        
        ### TOPOLOGY Updates 
        nrchs = np.unique(centerlines.reach_id[0,cl_r[order_ids]])
        max_id = [max(centerlines.cl_id[cl_r[order_ids[np.where(centerlines.reach_id[0,cl_r[order_ids]] == n)[0]]]]) for n in nrchs]
        id_sort = np.argsort(max_id)
        nrchs = nrchs[id_sort]
        #need to order nrchs in terms of indexes can update dist out easier? 
        for idx in list(range(len(nrchs))):
            pts = np.where(centerlines.reach_id[0,cl_r[order_ids]] == nrchs[idx])[0]
            binary = np.copy(centerlines.reach_id[1:,cl_r[order_ids[pts]]])
            binary[np.where(binary > 0)] = 1
            binary_sum = np.sum(binary, axis = 0)
            existing_nghs = np.where(binary_sum > 0)[0]
            if len(existing_nghs) > 0:
                mn = np.where(centerlines.cl_id[cl_r[order_ids[pts]]] == min(centerlines.cl_id[cl_r[order_ids[pts]]]))[0]
                mx = np.where(centerlines.cl_id[cl_r[order_ids[pts]]] == max(centerlines.cl_id[cl_r[order_ids[pts]]]))[0]
                if mn in existing_nghs and mx not in existing_nghs:
                    #updating new neighbors at the centerline level. 
                    centerlines.reach_id[1:,cl_r[order_ids[pts[mx]]]] = 0
                    centerlines.reach_id[1:,cl_r[order_ids[pts[mx]+1]]] = 0 
                    centerlines.reach_id[1,cl_r[order_ids[pts[mx]]]] = centerlines.reach_id[0,cl_r[order_ids[pts[mx]+1]]][0] #centerlines.reach_id[:,cl_r[order_ids[pts[mx]]]]
                    centerlines.reach_id[1,cl_r[order_ids[pts[mx]+1]]] = centerlines.reach_id[0,cl_r[order_ids[pts[mx]]]][0] #centerlines.reach_id[:,cl_r[order_ids[pts[mx]+1]]]
                    #updating new neighbors at the reach level.
                    ridx = np.where(reaches.id == centerlines.reach_id[0,cl_r[order_ids[pts[mx]]]][0])[0]
                    reaches.n_rch_up[ridx] = 1
                    reaches.rch_id_up[:,ridx] = 0
                    reaches.rch_id_up[0,ridx] = centerlines.reach_id[0,cl_r[order_ids[pts[mx]+1]]][0]
                    if idx > 0:
                        #upstream neighor
                        ridx2 = np.where(reaches.id == centerlines.reach_id[0,cl_r[order_ids[pts[mx]+1]]][0])[0]
                        reaches.n_rch_down[ridx2] = 1
                        reaches.rch_id_down[:,ridx2] = 0
                        reaches.rch_id_down[0,ridx2] = centerlines.reach_id[0,cl_r[order_ids[pts[mx]]]][0]
                        #current reach 
                        reaches.n_rch_down[ridx] = 1
                        reaches.rch_id_down[:,ridx] = 0
                        reaches.rch_id_down[0,ridx] = centerlines.reach_id[0,cl_r[order_ids[pts[mn]-1]]][0]

                elif mx in existing_nghs and mn not in existing_nghs:
                    centerlines.reach_id[1:,cl_r[order_ids[pts[mn]]]] = 0
                    centerlines.reach_id[1:,cl_r[order_ids[pts[mn]-1]]] = 0
                    centerlines.reach_id[1,cl_r[order_ids[pts[mn]]]] = centerlines.reach_id[0,cl_r[order_ids[pts[mn]-1]]][0] #centerlines.reach_id[:,cl_r[order_ids[pts[mx]]]]
                    centerlines.reach_id[1,cl_r[order_ids[pts[mn]-1]]] = centerlines.reach_id[0,cl_r[order_ids[pts[mn]]]][0] #centerlines.reach_id[:,cl_r[order_ids[pts[mx]+1]]]
                    #updating new neighbors at the reach level.
                    ridx = np.where(reaches.id == centerlines.reach_id[0,cl_r[order_ids[pts[mn]]]][0])[0]
                    reaches.n_rch_down[ridx] = 1
                    reaches.rch_id_down[:,ridx] = 0
                    reaches.rch_id_down[0,ridx] = centerlines.reach_id[0,cl_r[order_ids[pts[mn]-1]]][0]
                    if idx > 0:
                        #upstream neighbor
                        ridx2 = np.where(reaches.id == centerlines.reach_id[0,cl_r[order_ids[pts[mn]-1]]][0])[0]
                        reaches.n_rch_up[ridx2] = 1
                        reaches.rch_id_up[:,ridx2] = 0
                        reaches.rch_id_up[0,ridx2] = centerlines.reach_id[0,cl_r[order_ids[pts[mn]]]][0]
                        #current reach 
                        reaches.n_rch_up[ridx] = 1
                        reaches.rch_id_up[:,ridx] = 0
                        reaches.rch_id_up[0,ridx] = centerlines.reach_id[0,cl_r[order_ids[pts[mx]+1]]][0]
                
                else:
                    #update downstream end for reach level. 
                    ridx = np.where(reaches.id == centerlines.reach_id[0,cl_r[order_ids[pts[mn]]]][0])[0] 
                    reaches.n_rch_down[ridx] = 1
                    reaches.rch_id_down[:,ridx] = 0
                    reaches.rch_id_down[0,ridx] = centerlines.reach_id[0,cl_r[order_ids[pts[mn]-1]]][0]
                    #find the max id and change that reaches values to current reach...
                    up_nghs = np.unique(centerlines.reach_id[1:,cl_r[order_ids[pts[mx]]]])
                    up_nghs = up_nghs[up_nghs>0]
                    for up in list(range(len(up_nghs))):
                        #updating upstream most neighbor of original reach's neighbors at the centerline level.
                        ngh_rch = np.where(centerlines.reach_id[0,:] == up_nghs[up])[0]
                        vals = np.where(centerlines.reach_id[1:,ngh_rch] == nrchs[0])
                        centerlines.reach_id[vals[0]+1,ngh_rch[vals[1]]] = centerlines.reach_id[0,cl_r[order_ids[pts[mx]]]][0]
                        #updating upstream most neighbor of original reach's neighbors at the reach level. 
                        ridx = np.where(reaches.id == up_nghs[up])[0]
                        nridx = np.where(reaches.rch_id_down[:,ridx] == nrchs[0])[0]
                        reaches.rch_id_down[nridx,ridx] = centerlines.reach_id[0,cl_r[order_ids[pts[mx]]]][0]
        #Distance from Outlet 
        rch_indx = np.where(np.in1d(reaches.id,nrchs)==True)[0]
        rch_cs = np.cumsum(reaches.len[rch_indx])
        reaches.dist_out[rch_indx] = rch_cs+base_val

### Filler variables.
swd.discharge_attr_nc(reaches)

### Write Data. 
swd.write_nc(centerlines, reaches, nodes, region, nc_fn)

print('Cl Dimensions:', len(np.unique(centerlines.cl_id)), len(centerlines.cl_id))
print('Rch Dimensions:', len(np.unique(centerlines.reach_id[0,:])), len(np.unique(nodes.reach_id)), len(np.unique(reaches.id)),len(reaches.id))
print('Node Dimensions:', len(np.unique(centerlines.node_id[0,:])), len(np.unique(nodes.id)), len(nodes.id))


# plt.scatter(centerlines.x[cl_r[order_ids]], centerlines.y[cl_r[order_ids]], c=centerlines.cl_id[cl_r[order_ids]], s=5)
# plt.show()

# plt.scatter(centerlines.x[cl_r[order_ids]], centerlines.y[cl_r[order_ids]], c=centerlines.reach_id[0,cl_r[order_ids]], s=5)
# plt.show()

# plt.scatter(centerlines.x[cl_r[order_ids]], centerlines.y[cl_r[order_ids]], c=centerlines.node_id[0,cl_r[order_ids]], s=5)
# plt.show()

# plt.scatter(centerlines.x[cl_r[order_ids]], centerlines.y[cl_r[order_ids]], c=old_nums, s=5)
# plt.show()

# check = np.where(centerlines.cl_id == breaks)[0]
# plt.plot(centerlines.x[cl_r[order_ids]], centerlines.y[cl_r[order_ids]], c='black')
# plt.plot(centerlines.x[update_ids], centerlines.y[update_ids], c='blue')
# plt.plot(centerlines.x[update_ids2], centerlines.y[update_ids2], c='green')
# plt.scatter(centerlines.x[cl_r[order_ids[bounds]]], centerlines.y[cl_r[order_ids[bounds]]], c='red', s = 10)
# plt.scatter(centerlines.x[check], centerlines.y[check], c='grey', s = 10)
# plt.show()

# plt.plot(centerlines.x[cl_r[order_ids]], centerlines.y[cl_r[order_ids]], c='white')
# plt.scatter(centerlines.x[cl_r[order_ids]], centerlines.y[cl_r[order_ids]], c=new_divs)
# plt.scatter(centerlines.x[update_ids], centerlines.y[update_ids], c='white', s = 2)
# plt.show()

# ridx = np.where(reaches.id == 81153800093)[0] 
# reaches.n_rch_up[ridx] 
# reaches.rch_id_up[:,ridx] 
# reaches.n_rch_down[ridx] 
# reaches.rch_id_down[:,ridx] 

