# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import pandas as pd
import numpy as np
import argparse
import src.updates.sword_utils as swd 

parser = argparse.ArgumentParser()
parser.add_argument("region", help="continental region", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

# region = 'OC'
# version = 'v18'

paths = swd.prepare_paths(main_dir, region, version)
sword_fn = paths['nc_dir']+paths['nc_fn']
csv_dir = paths['update_dir']+region.lower()+'_missing_ghost_reaches.csv'
check_dir = paths['topo_dir']+'order_problems.csv'
out_dir = paths['update_dir']

centerlines, nodes, reaches = swd.read_nc(sword_fn)
reaches.type = np.array([int(str(rch)[-1]) for rch in reaches.id])
old_num_rchs = len(reaches.id)

rch_nums = np.array([int(str(rch)[6:10]) for rch in centerlines.reach_id[0,:]])
node_nums = np.array([int(str(rch)[10:13]) for rch in centerlines.node_id[0,:]])
cl_level6 = np.array([int(str(rch)[0:6]) for rch in centerlines.node_id[0,:]])

rch_csv = pd.read_csv(csv_dir)
try:
    check = pd.read_csv(check_dir)
except:
    check = []

if len(check) > 0:
    rmv = np.where(np.in1d(rch_csv['reach_id'],check['reach_id']))[0]
    if len(rmv) > 0:
        rch_csv = rch_csv.drop(rmv)
        print('Reaches with node ordering issues droped: '+str(len(rmv)))

subreaches = np.array(rch_csv['reach_id'])
hw_out_orig = np.array(rch_csv['hw_out'])

#find the node(s) for each reach that should be the ghost reach... 
all_new_ghost_nodes = []
all_new_ghost_nums = []
all_new_ghost_rchs = []
hw_out = []
for r in list(range(len(subreaches))):
    nind = np.where(nodes.reach_id == subreaches[r])[0]
    nodes_ordered = np.sort(nodes.id[nind])
    # print(r, len(nind))
    if hw_out_orig[r] == 1:
        change_node = max(nodes_ordered)
        len_check = np.where(centerlines.node_id[0,:] == change_node)[0]
        if len(len_check) == 1 or len(nind) == 2:
            # print(r, subreaches[r])
            all_new_ghost_nodes.append(nodes_ordered[-1])
            all_new_ghost_nums.append(2)
            all_new_ghost_rchs.append(subreaches[r])
            hw_out.append(hw_out_orig[r])
            all_new_ghost_nodes.append(nodes_ordered[-2])
            all_new_ghost_nums.append(1)
            all_new_ghost_rchs.append(subreaches[r])
            hw_out.append(hw_out_orig[r])
        else:
            all_new_ghost_nodes.append(nodes_ordered[-1])
            all_new_ghost_nums.append(1)
            all_new_ghost_rchs.append(subreaches[r])
            hw_out.append(hw_out_orig[r])
    else:
        change_node = max(nodes_ordered)
        len_check = np.where(centerlines.node_id[0,:] == change_node)[0]
        if len(len_check) == 1 or len(nind) == 2:
            all_new_ghost_nodes.append(nodes_ordered[0])
            all_new_ghost_nums.append(1)
            all_new_ghost_rchs.append(subreaches[r])
            hw_out.append(hw_out_orig[r])
            all_new_ghost_nodes.append(nodes_ordered[1])
            all_new_ghost_nums.append(2)
            all_new_ghost_rchs.append(subreaches[r])
            hw_out.append(hw_out_orig[r])
        else:
            all_new_ghost_nodes.append(nodes_ordered[0])
            all_new_ghost_nums.append(1)
            all_new_ghost_rchs.append(subreaches[r])
            hw_out.append(hw_out_orig[r])

all_new_ghost_nodes = np.array(all_new_ghost_nodes)
all_new_ghost_nums = np.array(all_new_ghost_nums)
all_new_ghost_rchs = np.array(all_new_ghost_rchs)
hw_out = np.array(hw_out)

issues = []
for ind in list(range(len(all_new_ghost_nodes))):
    print(ind, all_new_ghost_nodes[ind], all_new_ghost_rchs[ind], len(all_new_ghost_nodes)-1)
    # if ind == 23:
    #     break
    
    update_ids = np.where(centerlines.node_id[0,:] == all_new_ghost_nodes[ind])[0]
    if len(update_ids) == 0: #added for odd reach cases. usually one node reaches with zero up and down neighbors.
        issues.append(all_new_ghost_nodes[ind])
        continue

    nds = np.where(nodes.id == all_new_ghost_nodes[ind])[0]
    old_rch = np.unique(centerlines.reach_id[0,update_ids])
    num_nodes = len(np.where(nodes.reach_id == old_rch)[0])
    check2 = np.where(all_new_ghost_rchs == old_rch)[0]

    bsn6 = np.where(cl_level6 == np.unique(cl_level6[update_ids]))[0]
    if num_nodes == 1:
        new_node_num = all_new_ghost_nums[ind] #1
        node_nums[update_ids] = new_node_num
        new_rch_num = max(rch_nums[bsn6])+1
        # new_node_num = np.unique(node_nums[update_ids])[0]
        # node_nums[update_ids] = node_nums[update_ids]
        # new_rch_num = np.unique(rch_nums[update_ids])[0] #don't need to update reach id if only one node. Became problematic when using two nodes. 
        if len(check2) == 1:
            rch_nums[update_ids] = new_rch_num
        else:
            if max(check2) == ind:
                upind = np.where(np.in1d(centerlines.node_id[0,:], all_new_ghost_nodes[check2])==True)[0]
                rch_nums[upind] = new_rch_num
               
    else:
        new_node_num = all_new_ghost_nums[ind] #1
        node_nums[update_ids] = new_node_num
        #use to separate how headwaters and outlets were handled
        #outlets use to be given a value of zero but was an issue if two 
        #reaches needed updated in the same level 6 basin.
        #cause duplicate reach ids. Decided to use max value for outlets too
        #to prevent having to renumber all reaches in a level six basin 
        #and require topology updates for all. 
        new_rch_num = max(rch_nums[bsn6])+1  
        if len(check2) == 1:
            rch_nums[update_ids] = new_rch_num
        else:
            if max(check2) == ind:
                upind = np.where(np.in1d(centerlines.node_id[0,:], all_new_ghost_nodes[check2])==True)[0]
                rch_nums[upind] = new_rch_num
            
    if len(str(new_rch_num)) == 1:
        fill = '000'
        new_rch_id = int(str(np.unique(cl_level6[update_ids])[0])+fill+str(new_rch_num)+'6')
    if len(str(new_rch_num)) == 2:
        fill = '00'
        new_rch_id = int(str(np.unique(cl_level6[update_ids])[0])+fill+str(new_rch_num)+'6')
    if len(str(new_rch_num)) == 3:
        fill = '0'
        new_rch_id = int(str(np.unique(cl_level6[update_ids])[0])+fill+str(new_rch_num)+'6')
    if len(str(new_rch_num)) == 4:
        new_rch_id = int(str(np.unique(cl_level6[update_ids])[0])+str(new_rch_num)+'6')

    if len(str(new_node_num)) == 1:
        fill = '00'
        new_node_id = int(str(new_rch_id)[0:-1]+fill+str(new_node_num)+str(new_rch_id)[-1])
    if len(str(new_node_num)) == 2:
        fill = '0'
        new_node_id = int(str(new_rch_id)[0:-1]+fill+str(new_node_num)+str(new_rch_id)[-1])
    if len(str(new_node_num)) == 3:
        new_node_id = int(str(new_rch_id)[0:-1]+str(new_node_num)+str(new_rch_id)[-1])
    
    # if new_node_id == 75371508000016:
    #     print('first index of duplicated node id')
    #     break

    ### update centerline and node points
    centerlines.reach_id[0,update_ids] = new_rch_id #c = np.where(centerlines.reach_id[0,:] == old_rch)[0]; 
    if num_nodes == 1: #only update neighbor reach ids if the reach was one node to begin with or all nodes are being changed. 
        if len(check2) > 1:
            if max(check2) == ind:
                centerlines.reach_id[1,np.where(centerlines.reach_id[1,:]== old_rch)[0]] = new_rch_id
                centerlines.reach_id[2,np.where(centerlines.reach_id[2,:]== old_rch)[0]] = new_rch_id
                centerlines.reach_id[3,np.where(centerlines.reach_id[3,:]== old_rch)[0]] = new_rch_id
        else:
            centerlines.reach_id[1,np.where(centerlines.reach_id[1,:]== old_rch)[0]] = new_rch_id
            centerlines.reach_id[2,np.where(centerlines.reach_id[2,:]== old_rch)[0]] = new_rch_id
            centerlines.reach_id[3,np.where(centerlines.reach_id[3,:]== old_rch)[0]] = new_rch_id  
    centerlines.node_id[0,update_ids] = new_node_id
    centerlines.node_id[1,np.where(centerlines.node_id[1,:]== all_new_ghost_nodes[ind])[0]] = new_node_id
    centerlines.node_id[2,np.where(centerlines.node_id[2,:]== all_new_ghost_nodes[ind])[0]] = new_node_id
    centerlines.node_id[3,np.where(centerlines.node_id[3,:]== all_new_ghost_nodes[ind])[0]] = new_node_id
    
    # if new_node_id in nodes.id:
    #     print('!!! new node id already exists !!!')
    #     print(new_node_id)
    #     break
    
    nodes.id[nds] = new_node_id
    nodes.reach_id[nds] = new_rch_id
   
    #update key variables for original reach if it was only one node to start with. 
    cl_rch = np.where(centerlines.reach_id[0,:] == old_rch)[0]
    rch = np.where(reaches.id == old_rch)[0]
    if len(cl_rch) == 0:
        if len(check2) == 1:
            reaches.id[rch] = new_rch_id
            new_cl_ids = np.array([np.min(centerlines.cl_id[update_ids]), np.max(centerlines.cl_id[update_ids])]).reshape(2,1)
            reaches.cl_id[:,rch] = new_cl_ids
            reaches.x[rch] = np.median(centerlines.x[update_ids])
            reaches.x_min[rch] = np.min(centerlines.x[update_ids])
            reaches.x_max[rch] = np.max(centerlines.x[update_ids])
            reaches.y[rch] = np.median(centerlines.y[update_ids])
            reaches.y_min[rch] = np.min(centerlines.y[update_ids])
            reaches.y_max[rch] = np.max(centerlines.y[update_ids])
            reaches.len[rch] = nodes.len[nds]
            reaches.rch_n_nodes[rch] = 1
            #fill some attributes with node values. 
            reaches.wse[rch] = nodes.wse[nds]
            reaches.wse_var[rch] = nodes.wse_var[nds]
            reaches.wth[rch] = nodes.wth[nds]
            reaches.wth_var[rch] = nodes.wth_var[nds]
            reaches.grod[rch] = nodes.grod[nds]
            reaches.grod_fid[rch] = nodes.grod_fid[nds]
            reaches.hfalls_fid[rch] = nodes.hfalls_fid[nds]
            reaches.lakeflag[rch] = nodes.lakeflag[nds]
            reaches.dist_out[rch] = nodes.dist_out[nds]
            reaches.facc[rch] = nodes.facc[nds]
            reaches.max_wth[rch] = nodes.max_wth[nds]
            reaches.river_name[rch] = nodes.river_name[nds]
            # reaches.edit_flag[rch] = nodes.edit_flag[nds]
            reaches.trib_flag[rch] = nodes.trib_flag[nds]
            reaches.nchan_max[rch] = nodes.nchan_max[nds]
            reaches.nchan_mod[rch] = nodes.nchan_mod[nds]
            reaches.path_freq[rch] = nodes.path_freq[nds]
            reaches.path_order[rch] = nodes.path_order[nds]
            reaches.main_side[rch] = nodes.main_side[nds]
            reaches.path_segs[rch] = nodes.path_segs[nds]
            reaches.strm_order[rch] = nodes.strm_order[nds]
            reaches.network[rch] = nodes.network[nds]
            
            ### update edit flag for boundary change. 
            if reaches.edit_flag[rch] == 'NaN':
                edit_val = '6'
            elif '6' not in reaches.edit_flag[rch][0].split(','):
                edit_val = reaches.edit_flag[rch] + ',6'
            else:
                edit_val = reaches.edit_flag[rch]
            reaches.edit_flag[rch] = edit_val
            nodes.edit_flag[nds] = edit_val

            ### topology and end reach variables....
            if hw_out[ind] == 1: #headwater
                reaches.end_rch[rch] = 1
                dn_nghs = reaches.rch_id_down[:,rch]
                dn_nghs = dn_nghs[dn_nghs>0]
                for dn in list(range(len(dn_nghs))):    
                    ngh_rch = np.where(reaches.id == dn_nghs[dn])[0]
                    nr_ind = np.where(reaches.rch_id_up[:,ngh_rch] == old_rch)[0]
                    reaches.rch_id_up[nr_ind,ngh_rch] = new_rch_id
                    ngh_cl = np.where(centerlines.reach_id[0,:] == dn_nghs[dn])[0]
                    centerlines.reach_id[1,ngh_cl[np.where(centerlines.reach_id[1,ngh_cl] == old_rch)[0]]] = new_rch_id
                    centerlines.reach_id[2,ngh_cl[np.where(centerlines.reach_id[2,ngh_cl] == old_rch)[0]]]= new_rch_id
                    centerlines.reach_id[3,ngh_cl[np.where(centerlines.reach_id[3,ngh_cl] == old_rch)[0]]] = new_rch_id
                    
            else: #outlet
                reaches.end_rch[rch] = 2
                up_nghs = reaches.rch_id_up[:,rch]
                up_nghs = up_nghs[up_nghs>0]
                for up in list(range(len(up_nghs))):    
                    ngh_rch = np.where(reaches.id == up_nghs[up])[0]
                    nr_ind = np.where(reaches.rch_id_down[:,ngh_rch] == old_rch)[0]
                    reaches.rch_id_down[nr_ind,ngh_rch] = new_rch_id
                    ngh_cl = np.where(centerlines.reach_id[0,:] == up_nghs[up])[0]
                    centerlines.reach_id[1,ngh_cl[np.where(centerlines.reach_id[1,ngh_cl] == old_rch)[0]]] = new_rch_id
                    centerlines.reach_id[2,ngh_cl[np.where(centerlines.reach_id[2,ngh_cl] == old_rch)[0]]] = new_rch_id
                    centerlines.reach_id[3,ngh_cl[np.where(centerlines.reach_id[3,ngh_cl] == old_rch)[0]]] = new_rch_id
        else:
            #add new reach to reaches object if not already included.
            rch_new = np.where(reaches.id == new_rch_id)[0]
            if len(rch_new) == 0:
                if len(check2) == num_nodes:
                    reaches.id[rch] = new_rch_id
                else:
                    reaches.id = np.append(reaches.id, new_rch_id)
                    new_cl_ids = np.array([np.min(centerlines.cl_id[update_ids]), np.max(centerlines.cl_id[update_ids])]).reshape(2,1)
                    reaches.cl_id = np.append(reaches.cl_id, new_cl_ids, axis=1)
                    reaches.x = np.append(reaches.x, np.median(centerlines.x[update_ids]))
                    reaches.x_min = np.append(reaches.x_min, np.min(centerlines.x[update_ids]))
                    reaches.x_max = np.append(reaches.x_max, np.max(centerlines.x[update_ids]))
                    reaches.y = np.append(reaches.y, np.median(centerlines.y[update_ids]))
                    reaches.y_min = np.append(reaches.y_min, np.min(centerlines.y[update_ids]))
                    reaches.y_max = np.append(reaches.y_max, np.max(centerlines.y[update_ids]))
                    reaches.len = np.append(reaches.len, nodes.len[nds])
                    reaches.rch_n_nodes = np.append(reaches.rch_n_nodes, 1)
                    #fill some attributes with node values. 
                    reaches.wse = np.append(reaches.wse, nodes.wse[nds])
                    reaches.wse_var = np.append(reaches.wse_var, nodes.wse_var[nds])
                    reaches.wth = np.append(reaches.wth, nodes.wth[nds])
                    reaches.wth_var = np.append(reaches.wth_var, nodes.wth_var[nds])
                    reaches.grod = np.append(reaches.grod, nodes.grod[nds])
                    reaches.grod_fid = np.append(reaches.grod_fid, nodes.grod_fid[nds])
                    reaches.hfalls_fid = np.append(reaches.hfalls_fid, nodes.hfalls_fid[nds])
                    reaches.lakeflag = np.append(reaches.lakeflag, nodes.lakeflag[nds])
                    reaches.facc = np.append(reaches.facc, nodes.facc[nds])
                    reaches.max_wth = np.append(reaches.max_wth, nodes.max_wth[nds])
                    reaches.river_name = np.append(reaches.river_name, nodes.river_name[nds])
                    reaches.edit_flag = np.append(reaches.edit_flag, edit_val)
                    reaches.trib_flag = np.append(reaches.trib_flag, nodes.trib_flag[nds])
                    reaches.nchan_max = np.append(reaches.nchan_max, nodes.nchan_max[nds])
                    reaches.nchan_mod = np.append(reaches.nchan_mod, nodes.nchan_mod[nds])
                    reaches.path_freq = np.append(reaches.path_freq, nodes.path_freq[nds])
                    reaches.path_order = np.append(reaches.path_order, nodes.path_order[nds])
                    reaches.main_side = np.append(reaches.main_side, nodes.main_side[nds])
                    reaches.path_segs = np.append(reaches.path_segs, nodes.path_segs[nds])
                    reaches.strm_order = np.append(reaches.strm_order, nodes.strm_order[nds])
                    reaches.network = np.append(reaches.network, nodes.network[nds])
                    #fill other attrubutes with current reach values. 
                    reaches.slope = np.append(reaches.slope, reaches.slope[rch])
                    reaches.low_slope = np.append(reaches.low_slope, reaches.low_slope[rch])
                    reaches.iceflag = np.append(reaches.iceflag, reaches.iceflag[:,rch], axis=1)
                    reaches.max_obs = np.append(reaches.max_obs, reaches.max_obs[rch])
                    reaches.orbits = np.append(reaches.orbits, reaches.orbits[:,rch], axis=1)
                    #fill topology attributes based on if the new reach is a headwater or outlet.
                    if hw_out[ind] == 1: #headwater
                        end_rch = 1
                        n_rch_up = 0
                        rch_id_up = np.array([0,0,0,0]); rch_id_up = rch_id_up.reshape((4,1))
                        n_rch_down = 1
                        rch_id_down = np.array([old_rch[0],0,0,0]); rch_id_down = rch_id_down.reshape((4,1))
                        rch_dist_out = nodes.dist_out[nds]
                        #updating current reach topology
                        reaches.n_rch_up[rch] = 1
                        reaches.rch_id_up[0,rch] = new_rch_id
                        reaches.end_rch[rch] = 0
                        reaches.dist_out[rch] = reaches.dist_out[rch] - nodes.len[nds]
                        centerlines.reach_id[1,cl_rch[np.where(cl_rch == max(cl_rch))]] = new_rch_id
                        cl_rch2 = np.where(centerlines.reach_id[0,:] == new_rch_id)[0]
                        centerlines.reach_id[1,cl_rch2[np.where(cl_rch2 == min(cl_rch2))]] = old_rch
                    else: #outlet
                        end_rch = 2
                        n_rch_down = 0
                        rch_id_down = np.array([0,0,0,0]); rch_id_down = rch_id_down.reshape((4,1))
                        n_rch_up = 1
                        rch_id_up = np.array([old_rch[0],0,0,0]); rch_id_up = rch_id_up.reshape((4,1))
                        rch_dist_out = nodes.len[nds]
                        #updating current reach topology
                        reaches.n_rch_down[rch] = 1
                        reaches.rch_id_down[0,rch] = new_rch_id
                        reaches.end_rch[rch] = 0
                        #no need to update reach dist out for outlets. 
                        centerlines.reach_id[1,cl_rch[np.where(cl_rch == min(cl_rch))]] = new_rch_id
                        cl_rch2 = np.where(centerlines.reach_id[0,:] == new_rch_id)[0]
                        centerlines.reach_id[1,cl_rch2[np.where(cl_rch2 == max(cl_rch2))]] = old_rch
                    #appending the new reach topology.
                    reaches.n_rch_up = np.append(reaches.n_rch_up, n_rch_up)
                    reaches.n_rch_down = np.append(reaches.n_rch_down, n_rch_down)
                    reaches.rch_id_up = np.append(reaches.rch_id_up, rch_id_up, axis = 1)
                    reaches.rch_id_down = np.append(reaches.rch_id_down, rch_id_down, axis = 1)
                    reaches.end_rch = np.append(reaches.end_rch, end_rch)
                    reaches.dist_out = np.append(reaches.dist_out, rch_dist_out)
            
            #there are more than one nodes in a new reach...
            else:
                update_ids2 = np.where(centerlines.reach_id[0,:] == new_rch_id)[0]
                reaches.cl_id[0,rch_new] = np.min(centerlines.cl_id[update_ids2])
                reaches.cl_id[1,rch_new] = np.max(centerlines.cl_id[update_ids2])
                reaches.x[rch_new] = np.median(centerlines.x[update_ids2])
                reaches.x_min[rch_new] = np.min(centerlines.x[update_ids2])
                reaches.x_max[rch_new] = np.max(centerlines.x[update_ids2])
                reaches.y[rch_new] = np.median(centerlines.y[update_ids2])
                reaches.y_min[rch_new] = np.min(centerlines.y[update_ids2])
                reaches.y_max[rch_new] = np.max(centerlines.y[update_ids2])
                #fill some attributes with node values.
                nds_all = np.where(nodes.reach_id == new_rch_id)[0] 
                reaches.wse[rch_new] = np.mean(nodes.wse[nds_all])
                reaches.wse_var[rch_new] = np.max(nodes.wse_var[nds_all])
                reaches.wth[rch_new] = np.mean(nodes.wth[nds_all])
                reaches.wth_var[rch_new] = np.max(nodes.wth_var[nds_all])
                reaches.grod[rch_new] = np.max(nodes.grod[nds_all])
                reaches.grod_fid[rch_new] = np.max(nodes.grod_fid[nds_all])
                reaches.hfalls_fid[rch_new] = np.max(nodes.hfalls_fid[nds_all])
                reaches.lakeflag[rch_new] = np.max(nodes.lakeflag[nds_all])
                reaches.dist_out[rch_new] = np.max(nodes.dist_out[nds_all])
                reaches.facc[rch_new] = np.max(nodes.facc[nds_all])
                reaches.max_wth[rch_new] = np.max(nodes.max_wth[nds_all])
                reaches.river_name[rch_new] = nodes.river_name[nds_all][0]
                # reaches.edit_flag[rch_new] = nodes.edit_flag[nds]
                reaches.trib_flag[rch_new] = np.max(nodes.trib_flag[nds_all])
                reaches.nchan_max[rch_new] = np.max(nodes.nchan_max[nds_all])
                reaches.nchan_mod[rch_new] = np.max(nodes.nchan_mod[nds_all])
                reaches.path_freq[rch_new] = np.max(nodes.path_freq[nds_all])
                reaches.path_order[rch_new] = np.max(nodes.path_order[nds_all])
                reaches.main_side[rch_new] = np.max(nodes.main_side[nds_all])
                reaches.path_segs[rch_new] = np.max(nodes.path_segs[nds_all])
                reaches.strm_order[rch_new] = np.max(nodes.strm_order[nds_all])
                reaches.network[rch_new] = np.max(nodes.network[nds_all])
                #node based updates
                nnodes = np.unique(centerlines.node_id[0,update_ids2])
                reaches.len[rch_new] = np.sum(nodes.len[np.where(np.in1d(nodes.id, nnodes)==True)[0]])
                reaches.rch_n_nodes[rch_new] = len(nnodes)
                #centerline updates (3/6/2025)
                # centerlines.reach_id[1,cl_rch[np.where(cl_rch == max(cl_rch))]] = new_rch_id #centerlines.reach_id[1,cl_rch]
                # cl_rch2 = np.where(centerlines.reach_id[0,:] == new_rch_id)[0]
                # centerlines.reach_id[1,cl_rch2] = 0 #zero out from first node edits. 
                # centerlines.reach_id[1,cl_rch2[np.where(cl_rch2 == min(cl_rch2))]] = old_rch #centerlines.reach_id[1,cl_rch2]
                # print(ind, new_rch_id, old_rch, all_new_ghost_rchs[ind])
                #edit flag
                reaches.edit_flag[rch_new] = edit_val
                nodes.edit_flag[np.where(np.in1d(nodes.id, nnodes)==True)[0]] = edit_val  
                ### topology and end reach variables....
                if hw_out[ind] == 1: #headwater
                    reaches.end_rch[rch_new] = 1
                    dn_nghs = reaches.rch_id_down[:,rch]
                    dn_nghs = dn_nghs[dn_nghs>0]
                    for dn in list(range(len(dn_nghs))):    
                        ngh_rch = np.where(reaches.id == dn_nghs[dn])[0]
                        nr_ind = np.where(reaches.rch_id_up[:,ngh_rch] == old_rch)[0]
                        reaches.rch_id_up[nr_ind,ngh_rch] = new_rch_id
                        ngh_cl = np.where(centerlines.reach_id[0,:] == dn_nghs[dn])[0]
                        centerlines.reach_id[1,ngh_cl[np.where(centerlines.reach_id[1,ngh_cl] == old_rch)[0]]] = new_rch_id
                        centerlines.reach_id[2,ngh_cl[np.where(centerlines.reach_id[2,ngh_cl] == old_rch)[0]]] = new_rch_id
                        centerlines.reach_id[3,ngh_cl[np.where(centerlines.reach_id[3,ngh_cl] == old_rch)[0]]] = new_rch_id
                        
                else: #outlet
                    reaches.end_rch[rch_new] = 2
                    up_nghs = reaches.rch_id_up[:,rch]
                    up_nghs = up_nghs[up_nghs>0]
                    for up in list(range(len(up_nghs))):    
                        ngh_rch = np.where(reaches.id == up_nghs[up])[0]
                        nr_ind = np.where(reaches.rch_id_down[:,ngh_rch] == old_rch)[0]
                        reaches.rch_id_down[nr_ind,ngh_rch] = new_rch_id
                        ngh_cl = np.where(centerlines.reach_id[0,:] == up_nghs[up])[0]
                        centerlines.reach_id[1,ngh_cl[np.where(centerlines.reach_id[1,ngh_cl] == old_rch)[0]]] = new_rch_id
                        centerlines.reach_id[2,ngh_cl[np.where(centerlines.reach_id[2,ngh_cl] == old_rch)[0]]] = new_rch_id
                        centerlines.reach_id[3,ngh_cl[np.where(centerlines.reach_id[3,ngh_cl] == old_rch)[0]]] = new_rch_id  
    else:
        # update current reach attributes and append new ones.
        reaches.x[rch] = np.median(centerlines.x[cl_rch])
        reaches.x_min[rch] = np.min(centerlines.x[cl_rch])
        reaches.x_max[rch] = np.max(centerlines.x[cl_rch])
        reaches.y[rch] = np.median(centerlines.y[cl_rch])
        reaches.y_min[rch] = np.min(centerlines.y[cl_rch])
        reaches.y_max[rch] = np.max(centerlines.y[cl_rch])
        reaches.cl_id[0,rch] = np.min(centerlines.cl_id[cl_rch])
        reaches.cl_id[1,rch] = np.max(centerlines.cl_id[cl_rch])
        reaches.len[rch] = reaches.len[rch] - nodes.len[nds]
        reaches.rch_n_nodes[rch] = reaches.rch_n_nodes[rch] - 1
        ### update edit flag for boundary change. 
        if reaches.edit_flag[rch] == 'NaN':
            edit_val = '6'
        elif '6' not in reaches.edit_flag[rch][0].split(','):
            edit_val = reaches.edit_flag[rch] + ',6'
        else:
            edit_val = reaches.edit_flag[rch]
        reaches.edit_flag[rch] = edit_val
        nodes.edit_flag[nds] = edit_val
        #add new reach to reaches object if not already included.
        rch_new = np.where(reaches.id == new_rch_id)[0]
        if len(rch_new) == 0:
            if len(check2) == num_nodes:
                reaches.id[rch] = new_rch_id
                new_cl_ids = np.array([np.min(centerlines.cl_id[update_ids]), np.max(centerlines.cl_id[update_ids])]).reshape(2,1)
                reaches.cl_id[:,rch] = new_cl_ids
                reaches.x[rch] = np.median(centerlines.x[update_ids])
                reaches.x_min[rch] = np.min(centerlines.x[update_ids])
                reaches.x_max[rch] = np.max(centerlines.x[update_ids])
                reaches.y[rch] = np.median(centerlines.y[update_ids])
                reaches.y_min[rch] = np.min(centerlines.y[update_ids])
                reaches.y_max[rch] = np.max(centerlines.y[update_ids])
                reaches.len[rch] = nodes.len[nds]
                reaches.rch_n_nodes[rch] = 1
                #fill some attributes with node values. 
                reaches.wse[rch] = nodes.wse[nds]
                reaches.wse_var[rch] = nodes.wse_var[nds]
                reaches.wth[rch] = nodes.wth[nds]
                reaches.wth_var[rch] = nodes.wth_var[nds]
                reaches.grod[rch] = nodes.grod[nds]
                reaches.grod_fid[rch] = nodes.grod_fid[nds]
                reaches.hfalls_fid[rch] = nodes.hfalls_fid[nds]
                reaches.lakeflag[rch] = nodes.lakeflag[nds]
                reaches.dist_out[rch] = nodes.dist_out[nds]
                reaches.facc[rch] = nodes.facc[nds]
                reaches.max_wth[rch] = nodes.max_wth[nds]
                reaches.river_name[rch] = nodes.river_name[nds]
                # reaches.edit_flag[rch] = nodes.edit_flag[nds]
                reaches.trib_flag[rch] = nodes.trib_flag[nds]
                reaches.nchan_max[rch] = nodes.nchan_max[nds]
                reaches.nchan_mod[rch] = nodes.nchan_mod[nds]
                reaches.path_freq[rch] = nodes.path_freq[nds]
                reaches.path_order[rch] = nodes.path_order[nds]
                reaches.main_side[rch] = nodes.main_side[nds]
                reaches.path_segs[rch] = nodes.path_segs[nds]
                reaches.strm_order[rch] = nodes.strm_order[nds]
                reaches.network[rch] = nodes.network[nds]
                
                ### update edit flag for boundary change. 
                if reaches.edit_flag[rch] == 'NaN':
                    edit_val = '6'
                elif '6' not in reaches.edit_flag[rch][0].split(','):
                    edit_val = reaches.edit_flag[rch] + ',6'
                else:
                    edit_val = reaches.edit_flag[rch]
                reaches.edit_flag[rch] = edit_val
                nodes.edit_flag[nds] = edit_val

                ### topology and end reach variables....
                if hw_out[ind] == 1: #headwater
                    reaches.end_rch[rch] = 1
                    dn_nghs = reaches.rch_id_down[:,rch]
                    dn_nghs = dn_nghs[dn_nghs>0]
                    for dn in list(range(len(dn_nghs))):    
                        ngh_rch = np.where(reaches.id == dn_nghs[dn])[0]
                        nr_ind = np.where(reaches.rch_id_up[:,ngh_rch] == old_rch)[0]
                        reaches.rch_id_up[nr_ind,ngh_rch] = new_rch_id
                        ngh_cl = np.where(centerlines.reach_id[0,:] == dn_nghs[dn])[0]
                        centerlines.reach_id[1,ngh_cl[np.where(centerlines.reach_id[1,ngh_cl] == old_rch)[0]]] = new_rch_id
                        centerlines.reach_id[2,ngh_cl[np.where(centerlines.reach_id[2,ngh_cl] == old_rch)[0]]] = new_rch_id
                        centerlines.reach_id[3,ngh_cl[np.where(centerlines.reach_id[3,ngh_cl] == old_rch)[0]]] = new_rch_id
                        
                else: #outlet
                    reaches.end_rch[rch] = 2
                    up_nghs = reaches.rch_id_up[:,rch]
                    up_nghs = up_nghs[up_nghs>0]
                    for up in list(range(len(up_nghs))):    
                        ngh_rch = np.where(reaches.id == up_nghs[up])[0]
                        nr_ind = np.where(reaches.rch_id_down[:,ngh_rch] == old_rch)[0]
                        reaches.rch_id_down[nr_ind,ngh_rch] = new_rch_id
                        ngh_cl = np.where(centerlines.reach_id[0,:] == up_nghs[up])[0]
                        centerlines.reach_id[1,ngh_cl[np.where(centerlines.reach_id[1,ngh_cl] == old_rch)[0]]] = new_rch_id
                        centerlines.reach_id[2,ngh_cl[np.where(centerlines.reach_id[2,ngh_cl] == old_rch)[0]]] = new_rch_id
                        centerlines.reach_id[3,ngh_cl[np.where(centerlines.reach_id[3,ngh_cl] == old_rch)[0]]] = new_rch_id
            else:
                reaches.id = np.append(reaches.id, new_rch_id)
                new_cl_ids = np.array([np.min(centerlines.cl_id[update_ids]), np.max(centerlines.cl_id[update_ids])]).reshape(2,1)
                reaches.cl_id = np.append(reaches.cl_id, new_cl_ids, axis=1)
                reaches.x = np.append(reaches.x, np.median(centerlines.x[update_ids]))
                reaches.x_min = np.append(reaches.x_min, np.min(centerlines.x[update_ids]))
                reaches.x_max = np.append(reaches.x_max, np.max(centerlines.x[update_ids]))
                reaches.y = np.append(reaches.y, np.median(centerlines.y[update_ids]))
                reaches.y_min = np.append(reaches.y_min, np.min(centerlines.y[update_ids]))
                reaches.y_max = np.append(reaches.y_max, np.max(centerlines.y[update_ids]))
                reaches.len = np.append(reaches.len, nodes.len[nds])
                reaches.rch_n_nodes = np.append(reaches.rch_n_nodes, 1)
                #fill some attributes with node values. 
                reaches.wse = np.append(reaches.wse, nodes.wse[nds])
                reaches.wse_var = np.append(reaches.wse_var, nodes.wse_var[nds])
                reaches.wth = np.append(reaches.wth, nodes.wth[nds])
                reaches.wth_var = np.append(reaches.wth_var, nodes.wth_var[nds])
                reaches.grod = np.append(reaches.grod, nodes.grod[nds])
                reaches.grod_fid = np.append(reaches.grod_fid, nodes.grod_fid[nds])
                reaches.hfalls_fid = np.append(reaches.hfalls_fid, nodes.hfalls_fid[nds])
                reaches.lakeflag = np.append(reaches.lakeflag, nodes.lakeflag[nds])
                reaches.facc = np.append(reaches.facc, nodes.facc[nds])
                reaches.max_wth = np.append(reaches.max_wth, nodes.max_wth[nds])
                reaches.river_name = np.append(reaches.river_name, nodes.river_name[nds])
                reaches.edit_flag = np.append(reaches.edit_flag, edit_val)
                reaches.trib_flag = np.append(reaches.trib_flag, nodes.trib_flag[nds])
                reaches.nchan_max = np.append(reaches.nchan_max, nodes.nchan_max[nds])
                reaches.nchan_mod = np.append(reaches.nchan_mod, nodes.nchan_mod[nds])
                reaches.path_freq = np.append(reaches.path_freq, nodes.path_freq[nds])
                reaches.path_order = np.append(reaches.path_order, nodes.path_order[nds])
                reaches.main_side = np.append(reaches.main_side, nodes.main_side[nds])
                reaches.path_segs = np.append(reaches.path_segs, nodes.path_segs[nds])
                reaches.strm_order = np.append(reaches.strm_order, nodes.strm_order[nds])
                reaches.network = np.append(reaches.network, nodes.network[nds])
                #fill other attrubutes with current reach values. 
                reaches.slope = np.append(reaches.slope, reaches.slope[rch])
                reaches.low_slope = np.append(reaches.low_slope, reaches.low_slope[rch])
                reaches.iceflag = np.append(reaches.iceflag, reaches.iceflag[:,rch], axis=1)
                reaches.max_obs = np.append(reaches.max_obs, reaches.max_obs[rch])
                reaches.orbits = np.append(reaches.orbits, reaches.orbits[:,rch], axis=1)
                #fill topology attributes based on if the new reach is a headwater or outlet.
                if hw_out[ind] == 1: #headwater
                    end_rch = 1
                    n_rch_up = 0
                    rch_id_up = np.array([0,0,0,0]); rch_id_up = rch_id_up.reshape((4,1))
                    n_rch_down = 1
                    rch_id_down = np.array([old_rch[0],0,0,0]); rch_id_down = rch_id_down.reshape((4,1))
                    rch_dist_out = nodes.dist_out[nds]
                    #updating current reach topology
                    reaches.n_rch_up[rch] = 1
                    reaches.rch_id_up[0,rch] = new_rch_id
                    reaches.end_rch[rch] = 0
                    reaches.dist_out[rch] = reaches.dist_out[rch] - nodes.len[nds]
                    centerlines.reach_id[1,cl_rch[np.where(cl_rch == max(cl_rch))]] = new_rch_id
                    cl_rch2 = np.where(centerlines.reach_id[0,:] == new_rch_id)[0]
                    centerlines.reach_id[1,cl_rch2[np.where(cl_rch2 == min(cl_rch2))]] = old_rch
                else: #outlet
                    end_rch = 2
                    n_rch_down = 0
                    rch_id_down = np.array([0,0,0,0]); rch_id_down = rch_id_down.reshape((4,1))
                    n_rch_up = 1
                    rch_id_up = np.array([old_rch[0],0,0,0]); rch_id_up = rch_id_up.reshape((4,1))
                    rch_dist_out = nodes.len[nds]
                    #updating current reach topology
                    reaches.n_rch_down[rch] = 1
                    reaches.rch_id_down[0,rch] = new_rch_id
                    reaches.end_rch[rch] = 0
                    #no need to update reach dist out for outlets. 
                    centerlines.reach_id[1,cl_rch[np.where(cl_rch == min(cl_rch))]] = new_rch_id
                    cl_rch2 = np.where(centerlines.reach_id[0,:] == new_rch_id)[0]
                    centerlines.reach_id[1,cl_rch2[np.where(cl_rch2 == max(cl_rch2))]] = old_rch
                #appending the new reach topology.
                reaches.n_rch_up = np.append(reaches.n_rch_up, n_rch_up)
                reaches.n_rch_down = np.append(reaches.n_rch_down, n_rch_down)
                reaches.rch_id_up = np.append(reaches.rch_id_up, rch_id_up, axis = 1)
                reaches.rch_id_down = np.append(reaches.rch_id_down, rch_id_down, axis = 1)
                reaches.end_rch = np.append(reaches.end_rch, end_rch)
                reaches.dist_out = np.append(reaches.dist_out, rch_dist_out)
        
        #there are more than one nodes in a new reach...
        else:
            update_ids2 = np.where(centerlines.reach_id[0,:] == new_rch_id)[0]
            reaches.cl_id[0,rch_new] = np.min(centerlines.cl_id[update_ids2])
            reaches.cl_id[1,rch_new] = np.max(centerlines.cl_id[update_ids2])
            reaches.x[rch_new] = np.median(centerlines.x[update_ids2])
            reaches.x_min[rch_new] = np.min(centerlines.x[update_ids2])
            reaches.x_max[rch_new] = np.max(centerlines.x[update_ids2])
            reaches.y[rch_new] = np.median(centerlines.y[update_ids2])
            reaches.y_min[rch_new] = np.min(centerlines.y[update_ids2])
            reaches.y_max[rch_new] = np.max(centerlines.y[update_ids2])
            #node based updates
            nnodes = np.unique(centerlines.node_id[0,update_ids2])
            reaches.len[rch_new] = np.sum(nodes.len[np.where(np.in1d(nodes.id, nnodes)==True)[0]])
            reaches.rch_n_nodes[rch_new] = len(nnodes)
            #centerline updates (3/6/2025)
            centerlines.reach_id[1,cl_rch[np.where(cl_rch == max(cl_rch))]] = new_rch_id #centerlines.reach_id[1,cl_rch]
            cl_rch2 = np.where(centerlines.reach_id[0,:] == new_rch_id)[0]
            centerlines.reach_id[1,cl_rch2] = 0 #zero out from first node edits. 
            centerlines.reach_id[1,cl_rch2[np.where(cl_rch2 == min(cl_rch2))]] = old_rch #centerlines.reach_id[1,cl_rch2]
            # print(ind, new_rch_id, old_rch, all_new_ghost_rchs[ind])
            #edit flag
            reaches.edit_flag[rch_new] = edit_val
            nodes.edit_flag[np.where(np.in1d(nodes.id, nnodes)==True)[0]] = edit_val

#writing flagged reaches.
issue_csv = {'reach_id': np.array(issues).astype('int64')}
issue_csv = pd.DataFrame(issue_csv)
issue_csv.to_csv(out_dir+region.lower()+'_check_ghost_reaches.csv', index=False)

#writing data. 
print('Writing New NetCDF')
swd.discharge_attr_nc(reaches)
swd.write_nc(centerlines, reaches, nodes, region, sword_fn)

#checking dimensions
print('Cl Dimensions:', len(np.unique(centerlines.cl_id)), len(centerlines.cl_id))
print('Node Dimensions:', len(np.unique(centerlines.node_id[0,:])), len(np.unique(nodes.id)), len(nodes.id))
print('Rch Dimensions:', len(np.unique(centerlines.reach_id[0,:])), len(np.unique(nodes.reach_id)), len(np.unique(reaches.id)), len(reaches.id))
print('Previous Number of Reaches: '+str(old_num_rchs))
print('Issues: '+str(issues))
print(np.unique(reaches.edit_flag))


#### checks 
#single node
# old_ind = np.where(centerlines.reach_id[0,:] == 71120000621)[0]
# new_ind = np.where(centerlines.reach_id[0,:] == 71120001246)[0]
# ngh_ind = np.where(centerlines.reach_id[0,:] == 71120000371)[0]
# centerlines.reach_id[1,old_ind]
# centerlines.reach_id[1,ngh_ind]
# centerlines.reach_id[1,new_ind]

# old_ind = np.where(reaches.id == 71120000621)[0]
# new_ind = np.where(reaches.id == 71120001246)[0]
# ngh_ind = np.where(reaches.id == 71120000371)[0]
# reaches.rch_id_up[:,old_ind]; reaches.n_rch_up[old_ind]
# reaches.rch_id_up[:,ngh_ind]; reaches.n_rch_up[ngh_ind]
# reaches.rch_id_up[:,new_ind]; reaches.n_rch_up[new_ind]
# reaches.rch_id_down[:,old_ind]; reaches.n_rch_down[old_ind]
# reaches.rch_id_down[:,ngh_ind]; reaches.n_rch_down[ngh_ind]
# reaches.rch_id_down[:,new_ind]; reaches.n_rch_down[new_ind]

### double node
# old_ind = np.where(centerlines.reach_id[0,:] == 71182702613)[0]
# new_ind = np.where(centerlines.reach_id[0,:] == 71182702856)[0]
# ngh_ind = np.where(centerlines.reach_id[0,:] == 71182702593)[0]
# centerlines.reach_id[1,old_ind]
# centerlines.reach_id[1,ngh_ind] #problematic 
# centerlines.reach_id[1,new_ind] 

# old_ind = np.where(reaches.id == 71182702613)[0]
# new_ind = np.where(reaches.id == 71182702856)[0]
# ngh_ind = np.where(reaches.id == 71182702593)[0]
# reaches.rch_id_up[:,old_ind]; reaches.n_rch_up[old_ind]
# reaches.rch_id_up[:,ngh_ind]; reaches.n_rch_up[ngh_ind] 
# reaches.rch_id_up[:,new_ind]; reaches.n_rch_up[new_ind]
# reaches.rch_id_down[:,old_ind]; reaches.n_rch_down[old_ind]
# reaches.rch_id_down[:,ngh_ind]; reaches.n_rch_down[ngh_ind]
# reaches.rch_id_down[:,new_ind]; reaches.n_rch_down[new_ind]

# vals, cnt = np.unique(nodes.id, return_counts=True)


