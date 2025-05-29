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
from src.updates.sword import SWORD
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

sword = SWORD(main_dir, region, version)
trib_dir = sword.paths['updates_dir']+'/tribs'
trib_files = np.sort(glob.glob(os.path.join(trib_dir, '*.gpkg')))

cl_level6 = np.array([str(ind)[0:6] for ind in sword.centerlines.node_id[0,:]])
cl_node_num_int = np.array([int(str(ind)[10:13]) for ind in sword.centerlines.node_id[0,:]])
cl_rch_type = np.array([str(ind)[-1] for ind in sword.centerlines.node_id[0,:]])

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
    cl_r = np.where(sword.centerlines.reach_id[0,:] == unq_rchs[r])[0]
    order_ids = np.argsort(sword.centerlines.cl_id[cl_r])
    old_dist = sword.reaches.dist_out[np.where(sword.reaches.id == unq_rchs[r])[0]]
    old_len = sword.reaches.len[np.where(sword.reaches.id == unq_rchs[r])[0]]
    base_val = old_dist - old_len

    breaks = break_id[np.where(reach == unq_rchs[r])[0]]
    break_pts = np.array([np.where(sword.centerlines.cl_id[cl_r[order_ids]] == b)[0][0] for b in breaks])

    #append start and end points. 
    bounds = np.append(0,break_pts)
    bounds = np.append(bounds, len(cl_r))
    bounds = np.sort(bounds) #added 4/26/24
    bounds = np.unique(bounds) #added 6/7/24

    new_divs = np.zeros(len(cl_r))
    count = 1
    for b in list(range(len(bounds)-1)):
        update_nds = cl_r[order_ids[bounds[b]:bounds[b+1]]]
        nds = np.unique(sword.centerlines.node_id[0,update_nds])
        fill = np.where(np.in1d(sword.centerlines.node_id[0,cl_r[order_ids]], nds) == True)[0]
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
                new_cl_rch_id = sword.centerlines.reach_id[0,update_ids]
                new_cl_node_ids = sword.centerlines.node_id[0,update_ids]
                new_rch_id = np.unique(sword.centerlines.reach_id[0,update_ids])[0]
            else:
                # print('2')
                #Create New Reach ID
                update_ids = cl_r[order_ids[np.where(new_divs == unq_divs[d])]]
                old_nodes = np.unique(sword.centerlines.node_id[0,update_ids])
                old_rch = np.unique(sword.centerlines.reach_id[0,update_ids])[0]
                l6_basin = np.where(cl_level6 == np.unique(cl_level6[update_ids]))[0]
                cl_rch_num_int = np.array([int(str(ind)[6:10]) for ind in sword.centerlines.node_id[0,l6_basin]])
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

            x_coords = sword.centerlines.x[update_ids]
            y_coords = sword.centerlines.y[update_ids]
            diff = ct.get_distances(x_coords,y_coords)
            dist = np.cumsum(diff)
            
            # print('5')
            new_rch_len = np.max(dist)
            new_rch_x = np.median(sword.centerlines.x[update_ids])
            new_rch_y = np.median(sword.centerlines.y[update_ids])
            new_rch_x_max = np.max(sword.centerlines.x[update_ids])
            new_rch_x_min = np.min(sword.centerlines.x[update_ids])
            new_rch_y_max = np.max(sword.centerlines.y[update_ids])
            new_rch_y_min = np.min(sword.centerlines.y[update_ids])

            # print('6')
            unq_nodes = np.unique(new_cl_node_ids)
            new_node_len = np.zeros(len(unq_nodes))
            new_node_x = np.zeros(len(unq_nodes))
            new_node_y = np.zeros(len(unq_nodes))
            new_node_id = np.zeros(len(unq_nodes))
            new_node_cl_ids = np.zeros((2,len(unq_nodes)))
            for n2 in list(range(len(unq_nodes))):
                pts = np.where(new_cl_node_ids == unq_nodes[n2])[0]
                new_node_x[n2] = np.median(sword.centerlines.x[update_ids[pts]])
                new_node_y[n2] = np.median(sword.centerlines.y[update_ids[pts]])
                new_node_len[n2] = max(np.cumsum(diff[pts]))
                new_node_id[n2] = unq_nodes[n2]
                new_node_cl_ids[0,n2] = np.min(sword.centerlines.cl_id[update_ids[pts]])
                new_node_cl_ids[1,n2] = np.max(sword.centerlines.cl_id[update_ids[pts]])
                if len(pts) == 1:
                    new_node_len[n2] = 30
            
            if new_rch_id in sword.reaches.id:
                # print('7')
                node_ind = np.where(np.in1d(sword.nodes.id, new_node_id)==True)[0]
                sword.nodes.len[node_ind] = new_node_len
                sword.nodes.cl_id[:,node_ind] = new_node_cl_ids
                sword.nodes.x[node_ind] = new_node_x
                sword.nodes.y[node_ind] = new_node_y
                
                rch = np.where(sword.reaches.id == new_rch_id)[0]
                sword.reaches.cl_id[0,rch] = np.min(sword.centerlines.cl_id[update_ids])
                sword.reaches.cl_id[1,rch] = np.max(sword.centerlines.cl_id[update_ids])
                sword.reaches.x[rch] = new_rch_x
                sword.reaches.x_min[rch] = new_rch_x_min
                sword.reaches.x_max[rch] = new_rch_x_max
                sword.reaches.y[rch] = new_rch_y
                sword.reaches.y_min[rch] = new_rch_y_min
                sword.reaches.y_max[rch] = new_rch_y_max
                sword.reaches.len[rch] = new_rch_len
                sword.reaches.rch_n_nodes[rch] = len(new_node_id)
                if sword.reaches.edit_flag[rch] == 'NaN':
                    edit_val = '6'
                elif '6' not in sword.reaches.edit_flag[rch][0].split(','):
                    edit_val = sword.reaches.edit_flag[rch] + ',6'
                else:
                    edit_val = sword.reaches.edit_flag[rch]
                sword.reaches.edit_flag[rch] = edit_val
                sword.nodes.edit_flag[node_ind] = edit_val

            else:
                # print('8')
                sword.centerlines.reach_id[0,update_ids] = new_cl_rch_id
                sword.centerlines.node_id[0,update_ids] = new_cl_node_ids
                
                old_ind = np.where(np.in1d(sword.nodes.id, old_nodes) == True)[0]
                sword.nodes.id[old_ind] = new_node_id
                sword.nodes.len[old_ind] = new_node_len
                sword.nodes.cl_id[:,old_ind] = new_node_cl_ids
                sword.nodes.x[old_ind] = new_node_x
                sword.nodes.y[old_ind] = new_node_y
                sword.nodes.reach_id[old_ind] = np.repeat(new_rch_id, len(new_node_id))
                
                rch = np.where(sword.reaches.id == old_rch)[0]
                sword.reaches.id = np.append(sword.reaches.id, new_rch_id)
                new_cl_ids = np.array([np.min(sword.centerlines.cl_id[update_ids]), np.max(sword.centerlines.cl_id[update_ids])]).reshape(2,1)
                sword.reaches.cl_id = np.append(sword.reaches.cl_id, new_cl_ids, axis=1)
                sword.reaches.x = np.append(sword.reaches.x, new_rch_x)
                sword.reaches.x_min = np.append(sword.reaches.x_min, new_rch_x_min)
                sword.reaches.x_max = np.append(sword.reaches.x_max, new_rch_x_max)
                sword.reaches.y = np.append(sword.reaches.y, new_rch_y)
                sword.reaches.y_min = np.append(sword.reaches.y_min, new_rch_y_min)
                sword.reaches.y_max = np.append(sword.reaches.y_max, new_rch_y_max)
                sword.reaches.len = np.append(sword.reaches.len, new_rch_len)
                sword.reaches.rch_n_nodes = np.append(sword.reaches.rch_n_nodes, len(new_node_id))
                #fill attribute with current values. 
                sword.reaches.wse = np.append(sword.reaches.wse, sword.reaches.wse[rch])
                sword.reaches.wse_var = np.append(sword.reaches.wse_var, sword.reaches.wse_var[rch])
                sword.reaches.wth = np.append(sword.reaches.wth, sword.reaches.wth[rch])
                sword.reaches.wth_var = np.append(sword.reaches.wth_var, sword.reaches.wth_var[rch])
                sword.reaches.slope = np.append(sword.reaches.slope, sword.reaches.slope[rch])
                sword.reaches.grod = np.append(sword.reaches.grod, sword.reaches.grod[rch])
                sword.reaches.grod_fid = np.append(sword.reaches.grod_fid, sword.reaches.grod_fid[rch])
                sword.reaches.hfalls_fid = np.append(sword.reaches.hfalls_fid, sword.reaches.hfalls_fid[rch])
                sword.reaches.lakeflag = np.append(sword.reaches.lakeflag, sword.reaches.lakeflag[rch])
                sword.reaches.nchan_max = np.append(sword.reaches.nchan_max, sword.reaches.nchan_max[rch])
                sword.reaches.nchan_mod = np.append(sword.reaches.nchan_mod, sword.reaches.nchan_mod[rch])
                sword.reaches.dist_out = np.append(sword.reaches.dist_out, sword.reaches.dist_out[rch])
                sword.reaches.n_rch_up = np.append(sword.reaches.n_rch_up, sword.reaches.n_rch_up[rch])
                sword.reaches.n_rch_down = np.append(sword.reaches.n_rch_down, sword.reaches.n_rch_down[rch])
                sword.reaches.rch_id_up = np.append(sword.reaches.rch_id_up, sword.reaches.rch_id_up[:,rch], axis=1)
                sword.reaches.rch_id_down = np.append(sword.reaches.rch_id_down, sword.reaches.rch_id_down[:,rch], axis=1)
                sword.reaches.max_obs = np.append(sword.reaches.max_obs, sword.reaches.max_obs[rch])
                sword.reaches.orbits = np.append(sword.reaches.orbits, sword.reaches.orbits[:,rch], axis=1)
                sword.reaches.facc = np.append(sword.reaches.facc, sword.reaches.facc[rch])
                sword.reaches.iceflag = np.append(sword.reaches.iceflag, sword.reaches.iceflag[:,rch], axis=1)
                sword.reaches.max_wth = np.append(sword.reaches.max_wth, sword.reaches.max_wth[rch])
                sword.reaches.river_name = np.append(sword.reaches.river_name, sword.reaches.river_name[rch])
                sword.reaches.low_slope = np.append(sword.reaches.low_slope, sword.reaches.low_slope[rch])
                sword.reaches.trib_flag = np.append(sword.reaches.trib_flag, sword.reaches.trib_flag[rch])
                sword.reaches.path_freq = np.append(sword.reaches.path_freq, sword.reaches.path_freq[rch])
                sword.reaches.path_order = np.append(sword.reaches.path_order, sword.reaches.path_order[rch])
                sword.reaches.main_side = np.append(sword.reaches.main_side, sword.reaches.main_side[rch])
                sword.reaches.path_segs = np.append(sword.reaches.path_segs, sword.reaches.path_segs[rch])
                sword.reaches.strm_order = np.append(sword.reaches.strm_order, sword.reaches.strm_order[rch])
                sword.reaches.end_rch = np.append(sword.reaches.end_rch, sword.reaches.end_rch[rch])
                sword.reaches.network = np.append(sword.reaches.network, sword.reaches.network[rch])
                if sword.reaches.edit_flag[rch] == 'NaN':
                    edit_val = '6'
                elif '6' not in sword.reaches.edit_flag[rch][0].split(','):
                    edit_val = sword.reaches.edit_flag[rch] + ',6'
                else:
                    edit_val = sword.reaches.edit_flag[rch]
                sword.reaches.edit_flag = np.append(sword.reaches.edit_flag, edit_val)
                sword.nodes.edit_flag[old_ind] = edit_val
        
        ### TOPOLOGY Updates 
        nrchs = np.unique(sword.centerlines.reach_id[0,cl_r[order_ids]])
        max_id = [max(sword.centerlines.cl_id[cl_r[order_ids[np.where(sword.centerlines.reach_id[0,cl_r[order_ids]] == n)[0]]]]) for n in nrchs]
        id_sort = np.argsort(max_id)
        nrchs = nrchs[id_sort]
        #need to order nrchs in terms of indexes can update dist out easier? 
        for idx in list(range(len(nrchs))):
            pts = np.where(sword.centerlines.reach_id[0,cl_r[order_ids]] == nrchs[idx])[0]
            binary = np.copy(sword.centerlines.reach_id[1:,cl_r[order_ids[pts]]])
            binary[np.where(binary > 0)] = 1
            binary_sum = np.sum(binary, axis = 0)
            existing_nghs = np.where(binary_sum > 0)[0]
            if len(existing_nghs) > 0:
                mn = np.where(sword.centerlines.cl_id[cl_r[order_ids[pts]]] == min(sword.centerlines.cl_id[cl_r[order_ids[pts]]]))[0]
                mx = np.where(sword.centerlines.cl_id[cl_r[order_ids[pts]]] == max(sword.centerlines.cl_id[cl_r[order_ids[pts]]]))[0]
                if mn in existing_nghs and mx not in existing_nghs:
                    #updating new neighbors at the centerline level. 
                    sword.centerlines.reach_id[1:,cl_r[order_ids[pts[mx]]]] = 0
                    sword.centerlines.reach_id[1:,cl_r[order_ids[pts[mx]+1]]] = 0 
                    sword.centerlines.reach_id[1,cl_r[order_ids[pts[mx]]]] = sword.centerlines.reach_id[0,cl_r[order_ids[pts[mx]+1]]][0] #sword.centerlines.reach_id[:,cl_r[order_ids[pts[mx]]]]
                    sword.centerlines.reach_id[1,cl_r[order_ids[pts[mx]+1]]] = sword.centerlines.reach_id[0,cl_r[order_ids[pts[mx]]]][0] #sword.centerlines.reach_id[:,cl_r[order_ids[pts[mx]+1]]]
                    #updating new neighbors at the reach level.
                    ridx = np.where(sword.reaches.id == sword.centerlines.reach_id[0,cl_r[order_ids[pts[mx]]]][0])[0]
                    sword.reaches.n_rch_up[ridx] = 1
                    sword.reaches.rch_id_up[:,ridx] = 0
                    sword.reaches.rch_id_up[0,ridx] = sword.centerlines.reach_id[0,cl_r[order_ids[pts[mx]+1]]][0]
                    if idx > 0:
                        #upstream neighor
                        ridx2 = np.where(sword.reaches.id == sword.centerlines.reach_id[0,cl_r[order_ids[pts[mx]+1]]][0])[0]
                        sword.reaches.n_rch_down[ridx2] = 1
                        sword.reaches.rch_id_down[:,ridx2] = 0
                        sword.reaches.rch_id_down[0,ridx2] = sword.centerlines.reach_id[0,cl_r[order_ids[pts[mx]]]][0]
                        #current reach 
                        sword.reaches.n_rch_down[ridx] = 1
                        sword.reaches.rch_id_down[:,ridx] = 0
                        sword.reaches.rch_id_down[0,ridx] = sword.centerlines.reach_id[0,cl_r[order_ids[pts[mn]-1]]][0]

                elif mx in existing_nghs and mn not in existing_nghs:
                    sword.centerlines.reach_id[1:,cl_r[order_ids[pts[mn]]]] = 0
                    sword.centerlines.reach_id[1:,cl_r[order_ids[pts[mn]-1]]] = 0
                    sword.centerlines.reach_id[1,cl_r[order_ids[pts[mn]]]] = sword.centerlines.reach_id[0,cl_r[order_ids[pts[mn]-1]]][0] #sword.centerlines.reach_id[:,cl_r[order_ids[pts[mx]]]]
                    sword.centerlines.reach_id[1,cl_r[order_ids[pts[mn]-1]]] = sword.centerlines.reach_id[0,cl_r[order_ids[pts[mn]]]][0] #sword.centerlines.reach_id[:,cl_r[order_ids[pts[mx]+1]]]
                    #updating new neighbors at the reach level.
                    ridx = np.where(sword.reaches.id == sword.centerlines.reach_id[0,cl_r[order_ids[pts[mn]]]][0])[0]
                    sword.reaches.n_rch_down[ridx] = 1
                    sword.reaches.rch_id_down[:,ridx] = 0
                    sword.reaches.rch_id_down[0,ridx] = sword.centerlines.reach_id[0,cl_r[order_ids[pts[mn]-1]]][0]
                    if idx > 0:
                        #upstream neighbor
                        ridx2 = np.where(sword.reaches.id == sword.centerlines.reach_id[0,cl_r[order_ids[pts[mn]-1]]][0])[0]
                        sword.reaches.n_rch_up[ridx2] = 1
                        sword.reaches.rch_id_up[:,ridx2] = 0
                        sword.reaches.rch_id_up[0,ridx2] = sword.centerlines.reach_id[0,cl_r[order_ids[pts[mn]]]][0]
                        #current reach 
                        sword.reaches.n_rch_up[ridx] = 1
                        sword.reaches.rch_id_up[:,ridx] = 0
                        sword.reaches.rch_id_up[0,ridx] = sword.centerlines.reach_id[0,cl_r[order_ids[pts[mx]+1]]][0]
                
                else:
                    #update downstream end for reach level. 
                    ridx = np.where(sword.reaches.id == sword.centerlines.reach_id[0,cl_r[order_ids[pts[mn]]]][0])[0] 
                    sword.reaches.n_rch_down[ridx] = 1
                    sword.reaches.rch_id_down[:,ridx] = 0
                    sword.reaches.rch_id_down[0,ridx] = sword.centerlines.reach_id[0,cl_r[order_ids[pts[mn]-1]]][0]
                    #find the max id and change that reaches values to current reach...
                    up_nghs = np.unique(sword.centerlines.reach_id[1:,cl_r[order_ids[pts[mx]]]])
                    up_nghs = up_nghs[up_nghs>0]
                    for up in list(range(len(up_nghs))):
                        #updating upstream most neighbor of original reach's neighbors at the centerline level.
                        ngh_rch = np.where(sword.centerlines.reach_id[0,:] == up_nghs[up])[0]
                        vals = np.where(sword.centerlines.reach_id[1:,ngh_rch] == nrchs[0])
                        sword.centerlines.reach_id[vals[0]+1,ngh_rch[vals[1]]] = sword.centerlines.reach_id[0,cl_r[order_ids[pts[mx]]]][0]
                        #updating upstream most neighbor of original reach's neighbors at the reach level. 
                        ridx = np.where(sword.reaches.id == up_nghs[up])[0]
                        nridx = np.where(sword.reaches.rch_id_down[:,ridx] == nrchs[0])[0]
                        sword.reaches.rch_id_down[nridx,ridx] = sword.centerlines.reach_id[0,cl_r[order_ids[pts[mx]]]][0]
        #Distance from Outlet 
        rch_indx = np.where(np.in1d(sword.reaches.id,nrchs)==True)[0]
        rch_cs = np.cumsum(sword.reaches.len[rch_indx])
        sword.reaches.dist_out[rch_indx] = rch_cs+base_val

### Write Data. 
sword.save_nc()

print('Cl Dimensions:', len(np.unique(sword.centerlines.cl_id)), len(sword.centerlines.cl_id))
print('Rch Dimensions:', len(np.unique(sword.centerlines.reach_id[0,:])), len(np.unique(sword.nodes.reach_id)), len(np.unique(sword.reaches.id)),len(sword.reaches.id))
print('Node Dimensions:', len(np.unique(sword.centerlines.node_id[0,:])), len(np.unique(sword.nodes.id)), len(sword.nodes.id))


# plt.scatter(sword.centerlines.x[cl_r[order_ids]], sword.centerlines.y[cl_r[order_ids]], c=sword.centerlines.cl_id[cl_r[order_ids]], s=5)
# plt.show()

# plt.scatter(sword.centerlines.x[cl_r[order_ids]], sword.centerlines.y[cl_r[order_ids]], c=sword.centerlines.reach_id[0,cl_r[order_ids]], s=5)
# plt.show()

# plt.scatter(sword.centerlines.x[cl_r[order_ids]], sword.centerlines.y[cl_r[order_ids]], c=sword.centerlines.node_id[0,cl_r[order_ids]], s=5)
# plt.show()

# plt.scatter(sword.centerlines.x[cl_r[order_ids]], sword.centerlines.y[cl_r[order_ids]], c=old_nums, s=5)
# plt.show()

# check = np.where(sword.centerlines.cl_id == breaks)[0]
# plt.plot(sword.centerlines.x[cl_r[order_ids]], sword.centerlines.y[cl_r[order_ids]], c='black')
# plt.plot(sword.centerlines.x[update_ids], sword.centerlines.y[update_ids], c='blue')
# plt.plot(sword.centerlines.x[update_ids2], sword.centerlines.y[update_ids2], c='green')
# plt.scatter(sword.centerlines.x[cl_r[order_ids[bounds]]], sword.centerlines.y[cl_r[order_ids[bounds]]], c='red', s = 10)
# plt.scatter(sword.centerlines.x[check], sword.centerlines.y[check], c='grey', s = 10)
# plt.show()

# plt.plot(sword.centerlines.x[cl_r[order_ids]], sword.centerlines.y[cl_r[order_ids]], c='white')
# plt.scatter(sword.centerlines.x[cl_r[order_ids]], sword.centerlines.y[cl_r[order_ids]], c=new_divs)
# plt.scatter(sword.centerlines.x[update_ids], sword.centerlines.y[update_ids], c='white', s = 2)
# plt.show()

# ridx = np.where(sword.reaches.id == 81153800093)[0] 
# sword.reaches.n_rch_up[ridx] 
# sword.reaches.rch_id_up[:,ridx] 
# sword.reaches.n_rch_down[ridx] 
# sword.reaches.rch_id_down[:,ridx] 

