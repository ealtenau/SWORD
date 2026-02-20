# -*- coding: utf-8 -*-
"""
Create new ghost reaches and nodes 
(create_missing_ghost_reach.py)
===============================================

This script identifies and creates missing ghost
nodes and reaches in the SWOT River Database (SWORD).
The preprocessing script 'find_incorrect_ghost_reaches.py'
must be run to produce input csv files for this script. 

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python path/to/create_missing_ghost_reach.py NA v17

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import pandas as pd
import numpy as np
import argparse
from src.updates.sword_duckdb import SWORD

parser = argparse.ArgumentParser()
parser.add_argument("region", help="continental region", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

#read data
db_path = os.path.join(main_dir, f'data/duckdb/sword_{version}.duckdb')
sword = SWORD(db_path, region, version)
sword.copy() #copies original file for version control.

csv_dir = sword.paths['update_dir']+region.lower()+'_missing_ghost_reaches.csv'
check_dir = sword.paths['topo_dir']+'order_problems.csv'
out_dir = sword.paths['update_dir']

sword.reaches.type = np.array([int(str(rch)[-1]) for rch in sword.reaches.id])
old_num_rchs = len(sword.reaches.id)

#isolate basin, reach, and node numbers at centerline spatial scale. 
rch_nums = np.array([int(str(rch)[6:10]) for rch in sword.centerlines.reach_id[0,:]])
node_nums = np.array([int(str(rch)[10:13]) for rch in sword.centerlines.node_id[0,:]])
cl_level6 = np.array([int(str(rch)[0:6]) for rch in sword.centerlines.node_id[0,:]])

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
    nind = np.where(sword.nodes.reach_id == subreaches[r])[0]
    nodes_ordered = np.sort(sword.nodes.id[nind])
    # print(r, len(nind))
    if hw_out_orig[r] == 1:
        change_node = max(nodes_ordered)
        len_check = np.where(sword.centerlines.node_id[0,:] == change_node)[0]
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
        len_check = np.where(sword.centerlines.node_id[0,:] == change_node)[0]
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
    
    update_ids = np.where(sword.centerlines.node_id[0,:] == all_new_ghost_nodes[ind])[0]
    if len(update_ids) == 0: #added for odd reach cases. usually one node reaches with zero up and down neighbors.
        issues.append(all_new_ghost_nodes[ind])
        continue

    nds = np.where(sword.nodes.id == all_new_ghost_nodes[ind])[0]
    old_rch = np.unique(sword.centerlines.reach_id[0,update_ids])
    num_nodes = len(np.where(sword.nodes.reach_id == old_rch)[0])
    check2 = np.where(all_new_ghost_rchs == old_rch)[0]

    bsn6 = np.where(cl_level6 == np.unique(cl_level6[update_ids]))[0]
    if num_nodes == 1:
        new_node_num = all_new_ghost_nums[ind] #1
        node_nums[update_ids] = new_node_num
        new_rch_num = max(rch_nums[bsn6])+1
        # new_node_num = np.unique(node_nums[update_ids])[0]
        # node_nums[update_ids] = node_nums[update_ids]
        # new_rch_num = np.unique(rch_nums[update_ids])[0] #don't need to update reach id if only one node. Became problematic when using two sword.nodes. 
        if len(check2) == 1:
            rch_nums[update_ids] = new_rch_num
        else:
            if max(check2) == ind:
                upind = np.where(np.in1d(sword.centerlines.node_id[0,:], all_new_ghost_nodes[check2])==True)[0]
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
                upind = np.where(np.in1d(sword.centerlines.node_id[0,:], all_new_ghost_nodes[check2])==True)[0]
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
    sword.centerlines.reach_id[0,update_ids] = new_rch_id #c = np.where(sword.centerlines.reach_id[0,:] == old_rch)[0]; 
    if num_nodes == 1: #only update neighbor reach ids if the reach was one node to begin with or all nodes are being changed. 
        if len(check2) > 1:
            if max(check2) == ind:
                sword.centerlines.reach_id[1,np.where(sword.centerlines.reach_id[1,:]== old_rch)[0]] = new_rch_id
                sword.centerlines.reach_id[2,np.where(sword.centerlines.reach_id[2,:]== old_rch)[0]] = new_rch_id
                sword.centerlines.reach_id[3,np.where(sword.centerlines.reach_id[3,:]== old_rch)[0]] = new_rch_id
        else:
            sword.centerlines.reach_id[1,np.where(sword.centerlines.reach_id[1,:]== old_rch)[0]] = new_rch_id
            sword.centerlines.reach_id[2,np.where(sword.centerlines.reach_id[2,:]== old_rch)[0]] = new_rch_id
            sword.centerlines.reach_id[3,np.where(sword.centerlines.reach_id[3,:]== old_rch)[0]] = new_rch_id  
    sword.centerlines.node_id[0,update_ids] = new_node_id
    sword.centerlines.node_id[1,np.where(sword.centerlines.node_id[1,:]== all_new_ghost_nodes[ind])[0]] = new_node_id
    sword.centerlines.node_id[2,np.where(sword.centerlines.node_id[2,:]== all_new_ghost_nodes[ind])[0]] = new_node_id
    sword.centerlines.node_id[3,np.where(sword.centerlines.node_id[3,:]== all_new_ghost_nodes[ind])[0]] = new_node_id
    
    # if new_node_id in sword.nodes.id:
    #     print('!!! new node id already exists !!!')
    #     print(new_node_id)
    #     break
    
    sword.nodes.id[nds] = new_node_id
    sword.nodes.reach_id[nds] = new_rch_id
   
    #update key variables for original reach if it was only one node to start with. 
    cl_rch = np.where(sword.centerlines.reach_id[0,:] == old_rch)[0]
    rch = np.where(sword.reaches.id == old_rch)[0]
    if len(cl_rch) == 0:
        if len(check2) == 1:
            sword.reaches.id[rch] = new_rch_id
            new_cl_ids = np.array([np.min(sword.centerlines.cl_id[update_ids]), np.max(sword.centerlines.cl_id[update_ids])]).reshape(2,1)
            sword.reaches.cl_id[:,rch] = new_cl_ids
            sword.reaches.x[rch] = np.median(sword.centerlines.x[update_ids])
            sword.reaches.x_min[rch] = np.min(sword.centerlines.x[update_ids])
            sword.reaches.x_max[rch] = np.max(sword.centerlines.x[update_ids])
            sword.reaches.y[rch] = np.median(sword.centerlines.y[update_ids])
            sword.reaches.y_min[rch] = np.min(sword.centerlines.y[update_ids])
            sword.reaches.y_max[rch] = np.max(sword.centerlines.y[update_ids])
            sword.reaches.len[rch] = sword.nodes.len[nds]
            sword.reaches.rch_n_nodes[rch] = 1
            #fill some attributes with node values. 
            sword.reaches.wse[rch] = sword.nodes.wse[nds]
            sword.reaches.wse_var[rch] = sword.nodes.wse_var[nds]
            sword.reaches.wth[rch] = sword.nodes.wth[nds]
            sword.reaches.wth_var[rch] = sword.nodes.wth_var[nds]
            sword.reaches.grod[rch] = sword.nodes.grod[nds]
            sword.reaches.grod_fid[rch] = sword.nodes.grod_fid[nds]
            sword.reaches.hfalls_fid[rch] = sword.nodes.hfalls_fid[nds]
            sword.reaches.lakeflag[rch] = sword.nodes.lakeflag[nds]
            sword.reaches.dist_out[rch] = sword.nodes.dist_out[nds]
            sword.reaches.facc[rch] = sword.nodes.facc[nds]
            sword.reaches.max_wth[rch] = sword.nodes.max_wth[nds]
            sword.reaches.river_name[rch] = sword.nodes.river_name[nds]
            # sword.reaches.edit_flag[rch] = sword.nodes.edit_flag[nds]
            sword.reaches.trib_flag[rch] = sword.nodes.trib_flag[nds]
            sword.reaches.nchan_max[rch] = sword.nodes.nchan_max[nds]
            sword.reaches.nchan_mod[rch] = sword.nodes.nchan_mod[nds]
            sword.reaches.path_freq[rch] = sword.nodes.path_freq[nds]
            sword.reaches.path_order[rch] = sword.nodes.path_order[nds]
            sword.reaches.main_side[rch] = sword.nodes.main_side[nds]
            sword.reaches.path_segs[rch] = sword.nodes.path_segs[nds]
            sword.reaches.strm_order[rch] = sword.nodes.strm_order[nds]
            sword.reaches.network[rch] = sword.nodes.network[nds]
            sword.reaches.add_flag[rch] = sword.nodes.add_flag[nds]
            
            ### update edit flag for boundary change. 
            if sword.reaches.edit_flag[rch] == 'NaN':
                edit_val = '6'
            elif '6' not in sword.reaches.edit_flag[rch][0].split(','):
                edit_val = sword.reaches.edit_flag[rch] + ',6'
            else:
                edit_val = sword.reaches.edit_flag[rch]
            sword.reaches.edit_flag[rch] = edit_val
            sword.nodes.edit_flag[nds] = edit_val

            ### topology and end reach variables....
            if hw_out[ind] == 1: #headwater
                sword.reaches.end_rch[rch] = 1
                dn_nghs = sword.reaches.rch_id_down[:,rch]
                dn_nghs = dn_nghs[dn_nghs>0]
                for dn in list(range(len(dn_nghs))):    
                    ngh_rch = np.where(sword.reaches.id == dn_nghs[dn])[0]
                    nr_ind = np.where(sword.reaches.rch_id_up[:,ngh_rch] == old_rch)[0]
                    sword.reaches.rch_id_up[nr_ind,ngh_rch] = new_rch_id
                    ngh_cl = np.where(sword.centerlines.reach_id[0,:] == dn_nghs[dn])[0]
                    sword.centerlines.reach_id[1,ngh_cl[np.where(sword.centerlines.reach_id[1,ngh_cl] == old_rch)[0]]] = new_rch_id
                    sword.centerlines.reach_id[2,ngh_cl[np.where(sword.centerlines.reach_id[2,ngh_cl] == old_rch)[0]]]= new_rch_id
                    sword.centerlines.reach_id[3,ngh_cl[np.where(sword.centerlines.reach_id[3,ngh_cl] == old_rch)[0]]] = new_rch_id
                    
            else: #outlet
                sword.reaches.end_rch[rch] = 2
                up_nghs = sword.reaches.rch_id_up[:,rch]
                up_nghs = up_nghs[up_nghs>0]
                for up in list(range(len(up_nghs))):    
                    ngh_rch = np.where(sword.reaches.id == up_nghs[up])[0]
                    nr_ind = np.where(sword.reaches.rch_id_down[:,ngh_rch] == old_rch)[0]
                    sword.reaches.rch_id_down[nr_ind,ngh_rch] = new_rch_id
                    ngh_cl = np.where(sword.centerlines.reach_id[0,:] == up_nghs[up])[0]
                    sword.centerlines.reach_id[1,ngh_cl[np.where(sword.centerlines.reach_id[1,ngh_cl] == old_rch)[0]]] = new_rch_id
                    sword.centerlines.reach_id[2,ngh_cl[np.where(sword.centerlines.reach_id[2,ngh_cl] == old_rch)[0]]] = new_rch_id
                    sword.centerlines.reach_id[3,ngh_cl[np.where(sword.centerlines.reach_id[3,ngh_cl] == old_rch)[0]]] = new_rch_id
        else:
            #add new reach to reaches object if not already included.
            rch_new = np.where(sword.reaches.id == new_rch_id)[0]
            if len(rch_new) == 0:
                if len(check2) == num_nodes:
                    sword.reaches.id[rch] = new_rch_id
                else:
                    sword.reaches.id = np.append(sword.reaches.id, new_rch_id)
                    new_cl_ids = np.array([np.min(sword.centerlines.cl_id[update_ids]), np.max(sword.centerlines.cl_id[update_ids])]).reshape(2,1)
                    sword.reaches.cl_id = np.append(sword.reaches.cl_id, new_cl_ids, axis=1)
                    sword.reaches.x = np.append(sword.reaches.x, np.median(sword.centerlines.x[update_ids]))
                    sword.reaches.x_min = np.append(sword.reaches.x_min, np.min(sword.centerlines.x[update_ids]))
                    sword.reaches.x_max = np.append(sword.reaches.x_max, np.max(sword.centerlines.x[update_ids]))
                    sword.reaches.y = np.append(sword.reaches.y, np.median(sword.centerlines.y[update_ids]))
                    sword.reaches.y_min = np.append(sword.reaches.y_min, np.min(sword.centerlines.y[update_ids]))
                    sword.reaches.y_max = np.append(sword.reaches.y_max, np.max(sword.centerlines.y[update_ids]))
                    sword.reaches.len = np.append(sword.reaches.len, sword.nodes.len[nds])
                    sword.reaches.rch_n_nodes = np.append(sword.reaches.rch_n_nodes, 1)
                    #fill some attributes with node values. 
                    sword.reaches.wse = np.append(sword.reaches.wse, sword.nodes.wse[nds])
                    sword.reaches.wse_var = np.append(sword.reaches.wse_var, sword.nodes.wse_var[nds])
                    sword.reaches.wth = np.append(sword.reaches.wth, sword.nodes.wth[nds])
                    sword.reaches.wth_var = np.append(sword.reaches.wth_var, sword.nodes.wth_var[nds])
                    sword.reaches.grod = np.append(sword.reaches.grod, sword.nodes.grod[nds])
                    sword.reaches.grod_fid = np.append(sword.reaches.grod_fid, sword.nodes.grod_fid[nds])
                    sword.reaches.hfalls_fid = np.append(sword.reaches.hfalls_fid, sword.nodes.hfalls_fid[nds])
                    sword.reaches.lakeflag = np.append(sword.reaches.lakeflag, sword.nodes.lakeflag[nds])
                    sword.reaches.facc = np.append(sword.reaches.facc, sword.nodes.facc[nds])
                    sword.reaches.max_wth = np.append(sword.reaches.max_wth, sword.nodes.max_wth[nds])
                    sword.reaches.river_name = np.append(sword.reaches.river_name, sword.nodes.river_name[nds])
                    sword.reaches.edit_flag = np.append(sword.reaches.edit_flag, edit_val)
                    sword.reaches.trib_flag = np.append(sword.reaches.trib_flag, sword.nodes.trib_flag[nds])
                    sword.reaches.nchan_max = np.append(sword.reaches.nchan_max, sword.nodes.nchan_max[nds])
                    sword.reaches.nchan_mod = np.append(sword.reaches.nchan_mod, sword.nodes.nchan_mod[nds])
                    sword.reaches.path_freq = np.append(sword.reaches.path_freq, sword.nodes.path_freq[nds])
                    sword.reaches.path_order = np.append(sword.reaches.path_order, sword.nodes.path_order[nds])
                    sword.reaches.main_side = np.append(sword.reaches.main_side, sword.nodes.main_side[nds])
                    sword.reaches.path_segs = np.append(sword.reaches.path_segs, sword.nodes.path_segs[nds])
                    sword.reaches.strm_order = np.append(sword.reaches.strm_order, sword.nodes.strm_order[nds])
                    sword.reaches.network = np.append(sword.reaches.network, sword.nodes.network[nds])
                    sword.reaches.add_flag = np.append(sword.reaches.add_flag, sword.nodes.add_flag[nds])
                    #fill other attrubutes with current reach values. 
                    sword.reaches.slope = np.append(sword.reaches.slope, sword.reaches.slope[rch])
                    sword.reaches.low_slope = np.append(sword.reaches.low_slope, sword.reaches.low_slope[rch])
                    sword.reaches.iceflag = np.append(sword.reaches.iceflag, sword.reaches.iceflag[:,rch], axis=1)
                    sword.reaches.max_obs = np.append(sword.reaches.max_obs, sword.reaches.max_obs[rch])
                    sword.reaches.orbits = np.append(sword.reaches.orbits, sword.reaches.orbits[:,rch], axis=1)
                    #fill topology attributes based on if the new reach is a headwater or outlet.
                    if hw_out[ind] == 1: #headwater
                        end_rch = 1
                        n_rch_up = 0
                        rch_id_up = np.array([0,0,0,0]); rch_id_up = rch_id_up.reshape((4,1))
                        n_rch_down = 1
                        rch_id_down = np.array([old_rch[0],0,0,0]); rch_id_down = rch_id_down.reshape((4,1))
                        rch_dist_out = sword.nodes.dist_out[nds]
                        #updating current reach topology
                        sword.reaches.n_rch_up[rch] = 1
                        sword.reaches.rch_id_up[0,rch] = new_rch_id
                        sword.reaches.end_rch[rch] = 0
                        sword.reaches.dist_out[rch] = sword.reaches.dist_out[rch] - sword.nodes.len[nds]
                        sword.centerlines.reach_id[1,cl_rch[np.where(cl_rch == max(cl_rch))]] = new_rch_id
                        cl_rch2 = np.where(sword.centerlines.reach_id[0,:] == new_rch_id)[0]
                        sword.centerlines.reach_id[1,cl_rch2[np.where(cl_rch2 == min(cl_rch2))]] = old_rch
                    else: #outlet
                        end_rch = 2
                        n_rch_down = 0
                        rch_id_down = np.array([0,0,0,0]); rch_id_down = rch_id_down.reshape((4,1))
                        n_rch_up = 1
                        rch_id_up = np.array([old_rch[0],0,0,0]); rch_id_up = rch_id_up.reshape((4,1))
                        rch_dist_out = sword.nodes.len[nds]
                        #updating current reach topology
                        sword.reaches.n_rch_down[rch] = 1
                        sword.reaches.rch_id_down[0,rch] = new_rch_id
                        sword.reaches.end_rch[rch] = 0
                        #no need to update reach dist out for outlets. 
                        sword.centerlines.reach_id[1,cl_rch[np.where(cl_rch == min(cl_rch))]] = new_rch_id
                        cl_rch2 = np.where(sword.centerlines.reach_id[0,:] == new_rch_id)[0]
                        sword.centerlines.reach_id[1,cl_rch2[np.where(cl_rch2 == max(cl_rch2))]] = old_rch
                    #appending the new reach topology.
                    sword.reaches.n_rch_up = np.append(sword.reaches.n_rch_up, n_rch_up)
                    sword.reaches.n_rch_down = np.append(sword.reaches.n_rch_down, n_rch_down)
                    sword.reaches.rch_id_up = np.append(sword.reaches.rch_id_up, rch_id_up, axis = 1)
                    sword.reaches.rch_id_down = np.append(sword.reaches.rch_id_down, rch_id_down, axis = 1)
                    sword.reaches.end_rch = np.append(sword.reaches.end_rch, end_rch)
                    sword.reaches.dist_out = np.append(sword.reaches.dist_out, rch_dist_out)
            
            #there are more than one nodes in a new reach...
            else:
                update_ids2 = np.where(sword.centerlines.reach_id[0,:] == new_rch_id)[0]
                sword.reaches.cl_id[0,rch_new] = np.min(sword.centerlines.cl_id[update_ids2])
                sword.reaches.cl_id[1,rch_new] = np.max(sword.centerlines.cl_id[update_ids2])
                sword.reaches.x[rch_new] = np.median(sword.centerlines.x[update_ids2])
                sword.reaches.x_min[rch_new] = np.min(sword.centerlines.x[update_ids2])
                sword.reaches.x_max[rch_new] = np.max(sword.centerlines.x[update_ids2])
                sword.reaches.y[rch_new] = np.median(sword.centerlines.y[update_ids2])
                sword.reaches.y_min[rch_new] = np.min(sword.centerlines.y[update_ids2])
                sword.reaches.y_max[rch_new] = np.max(sword.centerlines.y[update_ids2])
                #fill some attributes with node values.
                nds_all = np.where(sword.nodes.reach_id == new_rch_id)[0] 
                sword.reaches.wse[rch_new] = np.mean(sword.nodes.wse[nds_all])
                sword.reaches.wse_var[rch_new] = np.max(sword.nodes.wse_var[nds_all])
                sword.reaches.wth[rch_new] = np.mean(sword.nodes.wth[nds_all])
                sword.reaches.wth_var[rch_new] = np.max(sword.nodes.wth_var[nds_all])
                sword.reaches.grod[rch_new] = np.max(sword.nodes.grod[nds_all])
                sword.reaches.grod_fid[rch_new] = np.max(sword.nodes.grod_fid[nds_all])
                sword.reaches.hfalls_fid[rch_new] = np.max(sword.nodes.hfalls_fid[nds_all])
                sword.reaches.lakeflag[rch_new] = np.max(sword.nodes.lakeflag[nds_all])
                sword.reaches.dist_out[rch_new] = np.max(sword.nodes.dist_out[nds_all])
                sword.reaches.facc[rch_new] = np.max(sword.nodes.facc[nds_all])
                sword.reaches.max_wth[rch_new] = np.max(sword.nodes.max_wth[nds_all])
                sword.reaches.river_name[rch_new] = sword.nodes.river_name[nds_all][0]
                # sword.reaches.edit_flag[rch_new] = sword.nodes.edit_flag[nds]
                sword.reaches.trib_flag[rch_new] = np.max(sword.nodes.trib_flag[nds_all])
                sword.reaches.nchan_max[rch_new] = np.max(sword.nodes.nchan_max[nds_all])
                sword.reaches.nchan_mod[rch_new] = np.max(sword.nodes.nchan_mod[nds_all])
                sword.reaches.path_freq[rch_new] = np.max(sword.nodes.path_freq[nds_all])
                sword.reaches.path_order[rch_new] = np.max(sword.nodes.path_order[nds_all])
                sword.reaches.main_side[rch_new] = np.max(sword.nodes.main_side[nds_all])
                sword.reaches.path_segs[rch_new] = np.max(sword.nodes.path_segs[nds_all])
                sword.reaches.strm_order[rch_new] = np.max(sword.nodes.strm_order[nds_all])
                sword.reaches.network[rch_new] = np.max(sword.nodes.network[nds_all])
                sword.reaches.add_flag[rch_new] = np.max(sword.nodes.add_flag[nds_all])
                #node based updates
                nnodes = np.unique(sword.centerlines.node_id[0,update_ids2])
                sword.reaches.len[rch_new] = np.sum(sword.nodes.len[np.where(np.in1d(sword.nodes.id, nnodes)==True)[0]])
                sword.reaches.rch_n_nodes[rch_new] = len(nnodes)
                #centerline updates (3/6/2025)
                # sword.centerlines.reach_id[1,cl_rch[np.where(cl_rch == max(cl_rch))]] = new_rch_id #sword.centerlines.reach_id[1,cl_rch]
                # cl_rch2 = np.where(sword.centerlines.reach_id[0,:] == new_rch_id)[0]
                # sword.centerlines.reach_id[1,cl_rch2] = 0 #zero out from first node edits. 
                # sword.centerlines.reach_id[1,cl_rch2[np.where(cl_rch2 == min(cl_rch2))]] = old_rch #sword.centerlines.reach_id[1,cl_rch2]
                # print(ind, new_rch_id, old_rch, all_new_ghost_rchs[ind])
                #edit flag
                sword.reaches.edit_flag[rch_new] = edit_val
                sword.nodes.edit_flag[np.where(np.in1d(sword.nodes.id, nnodes)==True)[0]] = edit_val  
                ### topology and end reach variables....
                if hw_out[ind] == 1: #headwater
                    sword.reaches.end_rch[rch_new] = 1
                    dn_nghs = sword.reaches.rch_id_down[:,rch]
                    dn_nghs = dn_nghs[dn_nghs>0]
                    for dn in list(range(len(dn_nghs))):    
                        ngh_rch = np.where(sword.reaches.id == dn_nghs[dn])[0]
                        nr_ind = np.where(sword.reaches.rch_id_up[:,ngh_rch] == old_rch)[0]
                        sword.reaches.rch_id_up[nr_ind,ngh_rch] = new_rch_id
                        ngh_cl = np.where(sword.centerlines.reach_id[0,:] == dn_nghs[dn])[0]
                        sword.centerlines.reach_id[1,ngh_cl[np.where(sword.centerlines.reach_id[1,ngh_cl] == old_rch)[0]]] = new_rch_id
                        sword.centerlines.reach_id[2,ngh_cl[np.where(sword.centerlines.reach_id[2,ngh_cl] == old_rch)[0]]] = new_rch_id
                        sword.centerlines.reach_id[3,ngh_cl[np.where(sword.centerlines.reach_id[3,ngh_cl] == old_rch)[0]]] = new_rch_id
                        
                else: #outlet
                    sword.reaches.end_rch[rch_new] = 2
                    up_nghs = sword.reaches.rch_id_up[:,rch]
                    up_nghs = up_nghs[up_nghs>0]
                    for up in list(range(len(up_nghs))):    
                        ngh_rch = np.where(sword.reaches.id == up_nghs[up])[0]
                        nr_ind = np.where(sword.reaches.rch_id_down[:,ngh_rch] == old_rch)[0]
                        sword.reaches.rch_id_down[nr_ind,ngh_rch] = new_rch_id
                        ngh_cl = np.where(sword.centerlines.reach_id[0,:] == up_nghs[up])[0]
                        sword.centerlines.reach_id[1,ngh_cl[np.where(sword.centerlines.reach_id[1,ngh_cl] == old_rch)[0]]] = new_rch_id
                        sword.centerlines.reach_id[2,ngh_cl[np.where(sword.centerlines.reach_id[2,ngh_cl] == old_rch)[0]]] = new_rch_id
                        sword.centerlines.reach_id[3,ngh_cl[np.where(sword.centerlines.reach_id[3,ngh_cl] == old_rch)[0]]] = new_rch_id  
    else:
        # update current reach attributes and append new ones.
        sword.reaches.x[rch] = np.median(sword.centerlines.x[cl_rch])
        sword.reaches.x_min[rch] = np.min(sword.centerlines.x[cl_rch])
        sword.reaches.x_max[rch] = np.max(sword.centerlines.x[cl_rch])
        sword.reaches.y[rch] = np.median(sword.centerlines.y[cl_rch])
        sword.reaches.y_min[rch] = np.min(sword.centerlines.y[cl_rch])
        sword.reaches.y_max[rch] = np.max(sword.centerlines.y[cl_rch])
        sword.reaches.cl_id[0,rch] = np.min(sword.centerlines.cl_id[cl_rch])
        sword.reaches.cl_id[1,rch] = np.max(sword.centerlines.cl_id[cl_rch])
        sword.reaches.len[rch] = sword.reaches.len[rch] - sword.nodes.len[nds]
        sword.reaches.rch_n_nodes[rch] = sword.reaches.rch_n_nodes[rch] - 1
        ### update edit flag for boundary change. 
        if sword.reaches.edit_flag[rch] == 'NaN':
            edit_val = '6'
        elif '6' not in sword.reaches.edit_flag[rch][0].split(','):
            edit_val = sword.reaches.edit_flag[rch] + ',6'
        else:
            edit_val = sword.reaches.edit_flag[rch]
        sword.reaches.edit_flag[rch] = edit_val
        sword.nodes.edit_flag[nds] = edit_val
        #add new reach to reaches object if not already included.
        rch_new = np.where(sword.reaches.id == new_rch_id)[0]
        if len(rch_new) == 0:
            if len(check2) == num_nodes:
                sword.reaches.id[rch] = new_rch_id
                new_cl_ids = np.array([np.min(sword.centerlines.cl_id[update_ids]), np.max(sword.centerlines.cl_id[update_ids])]).reshape(2,1)
                sword.reaches.cl_id[:,rch] = new_cl_ids
                sword.reaches.x[rch] = np.median(sword.centerlines.x[update_ids])
                sword.reaches.x_min[rch] = np.min(sword.centerlines.x[update_ids])
                sword.reaches.x_max[rch] = np.max(sword.centerlines.x[update_ids])
                sword.reaches.y[rch] = np.median(sword.centerlines.y[update_ids])
                sword.reaches.y_min[rch] = np.min(sword.centerlines.y[update_ids])
                sword.reaches.y_max[rch] = np.max(sword.centerlines.y[update_ids])
                sword.reaches.len[rch] = sword.nodes.len[nds]
                sword.reaches.rch_n_nodes[rch] = 1
                #fill some attributes with node values. 
                sword.reaches.wse[rch] = sword.nodes.wse[nds]
                sword.reaches.wse_var[rch] = sword.nodes.wse_var[nds]
                sword.reaches.wth[rch] = sword.nodes.wth[nds]
                sword.reaches.wth_var[rch] = sword.nodes.wth_var[nds]
                sword.reaches.grod[rch] = sword.nodes.grod[nds]
                sword.reaches.grod_fid[rch] = sword.nodes.grod_fid[nds]
                sword.reaches.hfalls_fid[rch] = sword.nodes.hfalls_fid[nds]
                sword.reaches.lakeflag[rch] = sword.nodes.lakeflag[nds]
                sword.reaches.dist_out[rch] = sword.nodes.dist_out[nds]
                sword.reaches.facc[rch] = sword.nodes.facc[nds]
                sword.reaches.max_wth[rch] = sword.nodes.max_wth[nds]
                sword.reaches.river_name[rch] = sword.nodes.river_name[nds]
                # sword.reaches.edit_flag[rch] = sword.nodes.edit_flag[nds]
                sword.reaches.trib_flag[rch] = sword.nodes.trib_flag[nds]
                sword.reaches.nchan_max[rch] = sword.nodes.nchan_max[nds]
                sword.reaches.nchan_mod[rch] = sword.nodes.nchan_mod[nds]
                sword.reaches.path_freq[rch] = sword.nodes.path_freq[nds]
                sword.reaches.path_order[rch] = sword.nodes.path_order[nds]
                sword.reaches.main_side[rch] = sword.nodes.main_side[nds]
                sword.reaches.path_segs[rch] = sword.nodes.path_segs[nds]
                sword.reaches.strm_order[rch] = sword.nodes.strm_order[nds]
                sword.reaches.network[rch] = sword.nodes.network[nds]
                sword.reaches.add_flag[rch] = sword.nodes.add_flag[nds]
                
                ### update edit flag for boundary change. 
                if sword.reaches.edit_flag[rch] == 'NaN':
                    edit_val = '6'
                elif '6' not in sword.reaches.edit_flag[rch][0].split(','):
                    edit_val = sword.reaches.edit_flag[rch] + ',6'
                else:
                    edit_val = sword.reaches.edit_flag[rch]
                sword.reaches.edit_flag[rch] = edit_val
                sword.nodes.edit_flag[nds] = edit_val

                ### topology and end reach variables....
                if hw_out[ind] == 1: #headwater
                    sword.reaches.end_rch[rch] = 1
                    dn_nghs = sword.reaches.rch_id_down[:,rch]
                    dn_nghs = dn_nghs[dn_nghs>0]
                    for dn in list(range(len(dn_nghs))):    
                        ngh_rch = np.where(sword.reaches.id == dn_nghs[dn])[0]
                        nr_ind = np.where(sword.reaches.rch_id_up[:,ngh_rch] == old_rch)[0]
                        sword.reaches.rch_id_up[nr_ind,ngh_rch] = new_rch_id
                        ngh_cl = np.where(sword.centerlines.reach_id[0,:] == dn_nghs[dn])[0]
                        sword.centerlines.reach_id[1,ngh_cl[np.where(sword.centerlines.reach_id[1,ngh_cl] == old_rch)[0]]] = new_rch_id
                        sword.centerlines.reach_id[2,ngh_cl[np.where(sword.centerlines.reach_id[2,ngh_cl] == old_rch)[0]]] = new_rch_id
                        sword.centerlines.reach_id[3,ngh_cl[np.where(sword.centerlines.reach_id[3,ngh_cl] == old_rch)[0]]] = new_rch_id
                        
                else: #outlet
                    sword.reaches.end_rch[rch] = 2
                    up_nghs = sword.reaches.rch_id_up[:,rch]
                    up_nghs = up_nghs[up_nghs>0]
                    for up in list(range(len(up_nghs))):    
                        ngh_rch = np.where(sword.reaches.id == up_nghs[up])[0]
                        nr_ind = np.where(sword.reaches.rch_id_down[:,ngh_rch] == old_rch)[0]
                        sword.reaches.rch_id_down[nr_ind,ngh_rch] = new_rch_id
                        ngh_cl = np.where(sword.centerlines.reach_id[0,:] == up_nghs[up])[0]
                        sword.centerlines.reach_id[1,ngh_cl[np.where(sword.centerlines.reach_id[1,ngh_cl] == old_rch)[0]]] = new_rch_id
                        sword.centerlines.reach_id[2,ngh_cl[np.where(sword.centerlines.reach_id[2,ngh_cl] == old_rch)[0]]] = new_rch_id
                        sword.centerlines.reach_id[3,ngh_cl[np.where(sword.centerlines.reach_id[3,ngh_cl] == old_rch)[0]]] = new_rch_id
            else:
                sword.reaches.id = np.append(sword.reaches.id, new_rch_id)
                new_cl_ids = np.array([np.min(sword.centerlines.cl_id[update_ids]), np.max(sword.centerlines.cl_id[update_ids])]).reshape(2,1)
                sword.reaches.cl_id = np.append(sword.reaches.cl_id, new_cl_ids, axis=1)
                sword.reaches.x = np.append(sword.reaches.x, np.median(sword.centerlines.x[update_ids]))
                sword.reaches.x_min = np.append(sword.reaches.x_min, np.min(sword.centerlines.x[update_ids]))
                sword.reaches.x_max = np.append(sword.reaches.x_max, np.max(sword.centerlines.x[update_ids]))
                sword.reaches.y = np.append(sword.reaches.y, np.median(sword.centerlines.y[update_ids]))
                sword.reaches.y_min = np.append(sword.reaches.y_min, np.min(sword.centerlines.y[update_ids]))
                sword.reaches.y_max = np.append(sword.reaches.y_max, np.max(sword.centerlines.y[update_ids]))
                sword.reaches.len = np.append(sword.reaches.len, sword.nodes.len[nds])
                sword.reaches.rch_n_nodes = np.append(sword.reaches.rch_n_nodes, 1)
                #fill some attributes with node values. 
                sword.reaches.wse = np.append(sword.reaches.wse, sword.nodes.wse[nds])
                sword.reaches.wse_var = np.append(sword.reaches.wse_var, sword.nodes.wse_var[nds])
                sword.reaches.wth = np.append(sword.reaches.wth, sword.nodes.wth[nds])
                sword.reaches.wth_var = np.append(sword.reaches.wth_var, sword.nodes.wth_var[nds])
                sword.reaches.grod = np.append(sword.reaches.grod, sword.nodes.grod[nds])
                sword.reaches.grod_fid = np.append(sword.reaches.grod_fid, sword.nodes.grod_fid[nds])
                sword.reaches.hfalls_fid = np.append(sword.reaches.hfalls_fid, sword.nodes.hfalls_fid[nds])
                sword.reaches.lakeflag = np.append(sword.reaches.lakeflag, sword.nodes.lakeflag[nds])
                sword.reaches.facc = np.append(sword.reaches.facc, sword.nodes.facc[nds])
                sword.reaches.max_wth = np.append(sword.reaches.max_wth, sword.nodes.max_wth[nds])
                sword.reaches.river_name = np.append(sword.reaches.river_name, sword.nodes.river_name[nds])
                sword.reaches.edit_flag = np.append(sword.reaches.edit_flag, edit_val)
                sword.reaches.trib_flag = np.append(sword.reaches.trib_flag, sword.nodes.trib_flag[nds])
                sword.reaches.nchan_max = np.append(sword.reaches.nchan_max, sword.nodes.nchan_max[nds])
                sword.reaches.nchan_mod = np.append(sword.reaches.nchan_mod, sword.nodes.nchan_mod[nds])
                sword.reaches.path_freq = np.append(sword.reaches.path_freq, sword.nodes.path_freq[nds])
                sword.reaches.path_order = np.append(sword.reaches.path_order, sword.nodes.path_order[nds])
                sword.reaches.main_side = np.append(sword.reaches.main_side, sword.nodes.main_side[nds])
                sword.reaches.path_segs = np.append(sword.reaches.path_segs, sword.nodes.path_segs[nds])
                sword.reaches.strm_order = np.append(sword.reaches.strm_order, sword.nodes.strm_order[nds])
                sword.reaches.network = np.append(sword.reaches.network, sword.nodes.network[nds])
                sword.reaches.add_flag = np.append(sword.reaches.add_flag, sword.nodes.add_flag[nds])
                #fill other attrubutes with current reach values. 
                sword.reaches.slope = np.append(sword.reaches.slope, sword.reaches.slope[rch])
                sword.reaches.low_slope = np.append(sword.reaches.low_slope, sword.reaches.low_slope[rch])
                sword.reaches.iceflag = np.append(sword.reaches.iceflag, sword.reaches.iceflag[:,rch], axis=1)
                sword.reaches.max_obs = np.append(sword.reaches.max_obs, sword.reaches.max_obs[rch])
                sword.reaches.orbits = np.append(sword.reaches.orbits, sword.reaches.orbits[:,rch], axis=1)
                #fill topology attributes based on if the new reach is a headwater or outlet.
                if hw_out[ind] == 1: #headwater
                    end_rch = 1
                    n_rch_up = 0
                    rch_id_up = np.array([0,0,0,0]); rch_id_up = rch_id_up.reshape((4,1))
                    n_rch_down = 1
                    rch_id_down = np.array([old_rch[0],0,0,0]); rch_id_down = rch_id_down.reshape((4,1))
                    rch_dist_out = sword.nodes.dist_out[nds]
                    #updating current reach topology
                    sword.reaches.n_rch_up[rch] = 1
                    sword.reaches.rch_id_up[0,rch] = new_rch_id
                    sword.reaches.end_rch[rch] = 0
                    sword.reaches.dist_out[rch] = sword.reaches.dist_out[rch] - sword.nodes.len[nds]
                    sword.centerlines.reach_id[1,cl_rch[np.where(cl_rch == max(cl_rch))]] = new_rch_id
                    cl_rch2 = np.where(sword.centerlines.reach_id[0,:] == new_rch_id)[0]
                    sword.centerlines.reach_id[1,cl_rch2[np.where(cl_rch2 == min(cl_rch2))]] = old_rch
                else: #outlet
                    end_rch = 2
                    n_rch_down = 0
                    rch_id_down = np.array([0,0,0,0]); rch_id_down = rch_id_down.reshape((4,1))
                    n_rch_up = 1
                    rch_id_up = np.array([old_rch[0],0,0,0]); rch_id_up = rch_id_up.reshape((4,1))
                    rch_dist_out = sword.nodes.len[nds]
                    #updating current reach topology
                    sword.reaches.n_rch_down[rch] = 1
                    sword.reaches.rch_id_down[0,rch] = new_rch_id
                    sword.reaches.end_rch[rch] = 0
                    #no need to update reach dist out for outlets. 
                    sword.centerlines.reach_id[1,cl_rch[np.where(cl_rch == min(cl_rch))]] = new_rch_id
                    cl_rch2 = np.where(sword.centerlines.reach_id[0,:] == new_rch_id)[0]
                    sword.centerlines.reach_id[1,cl_rch2[np.where(cl_rch2 == max(cl_rch2))]] = old_rch
                #appending the new reach topology.
                sword.reaches.n_rch_up = np.append(sword.reaches.n_rch_up, n_rch_up)
                sword.reaches.n_rch_down = np.append(sword.reaches.n_rch_down, n_rch_down)
                sword.reaches.rch_id_up = np.append(sword.reaches.rch_id_up, rch_id_up, axis = 1)
                sword.reaches.rch_id_down = np.append(sword.reaches.rch_id_down, rch_id_down, axis = 1)
                sword.reaches.end_rch = np.append(sword.reaches.end_rch, end_rch)
                sword.reaches.dist_out = np.append(sword.reaches.dist_out, rch_dist_out)
        
        #there are more than one nodes in a new reach...
        else:
            update_ids2 = np.where(sword.centerlines.reach_id[0,:] == new_rch_id)[0]
            sword.reaches.cl_id[0,rch_new] = np.min(sword.centerlines.cl_id[update_ids2])
            sword.reaches.cl_id[1,rch_new] = np.max(sword.centerlines.cl_id[update_ids2])
            sword.reaches.x[rch_new] = np.median(sword.centerlines.x[update_ids2])
            sword.reaches.x_min[rch_new] = np.min(sword.centerlines.x[update_ids2])
            sword.reaches.x_max[rch_new] = np.max(sword.centerlines.x[update_ids2])
            sword.reaches.y[rch_new] = np.median(sword.centerlines.y[update_ids2])
            sword.reaches.y_min[rch_new] = np.min(sword.centerlines.y[update_ids2])
            sword.reaches.y_max[rch_new] = np.max(sword.centerlines.y[update_ids2])
            #node based updates
            nnodes = np.unique(sword.centerlines.node_id[0,update_ids2])
            sword.reaches.len[rch_new] = np.sum(sword.nodes.len[np.where(np.in1d(sword.nodes.id, nnodes)==True)[0]])
            sword.reaches.rch_n_nodes[rch_new] = len(nnodes)
            #centerline updates (3/6/2025)
            sword.centerlines.reach_id[1,cl_rch[np.where(cl_rch == max(cl_rch))]] = new_rch_id #sword.centerlines.reach_id[1,cl_rch]
            cl_rch2 = np.where(sword.centerlines.reach_id[0,:] == new_rch_id)[0]
            sword.centerlines.reach_id[1,cl_rch2] = 0 #zero out from first node edits. 
            sword.centerlines.reach_id[1,cl_rch2[np.where(cl_rch2 == min(cl_rch2))]] = old_rch #sword.centerlines.reach_id[1,cl_rch2]
            # print(ind, new_rch_id, old_rch, all_new_ghost_rchs[ind])
            #edit flag
            sword.reaches.edit_flag[rch_new] = edit_val
            sword.nodes.edit_flag[np.where(np.in1d(sword.nodes.id, nnodes)==True)[0]] = edit_val

#writing flagged sword.reaches.
issue_csv = {'reach_id': np.array(issues).astype('int64')}
issue_csv = pd.DataFrame(issue_csv)
issue_csv.to_csv(out_dir+region.lower()+'_check_ghost_reaches.csv', index=False)

#writing data. 
print('Writing New NetCDF')
sword.save_nc()

#checking dimensions
print('Cl Dimensions:', len(np.unique(sword.centerlines.cl_id)), len(sword.centerlines.cl_id))
print('Node Dimensions:', len(np.unique(sword.centerlines.node_id[0,:])), len(np.unique(sword.nodes.id)), len(sword.nodes.id))
print('Rch Dimensions:', len(np.unique(sword.centerlines.reach_id[0,:])), len(np.unique(sword.nodes.reach_id)), len(np.unique(sword.reaches.id)), len(sword.reaches.id))
print('Previous Number of Reaches: '+str(old_num_rchs))
print('Issues: '+str(issues))
print(np.unique(sword.reaches.edit_flag))


#### checks 
#single node
# old_ind = np.where(sword.centerlines.reach_id[0,:] == 71120000621)[0]
# new_ind = np.where(sword.centerlines.reach_id[0,:] == 71120001246)[0]
# ngh_ind = np.where(sword.centerlines.reach_id[0,:] == 71120000371)[0]
# sword.centerlines.reach_id[1,old_ind]
# sword.centerlines.reach_id[1,ngh_ind]
# sword.centerlines.reach_id[1,new_ind]

# old_ind = np.where(sword.reaches.id == 71120000621)[0]
# new_ind = np.where(sword.reaches.id == 71120001246)[0]
# ngh_ind = np.where(sword.reaches.id == 71120000371)[0]
# sword.reaches.rch_id_up[:,old_ind]; sword.reaches.n_rch_up[old_ind]
# sword.reaches.rch_id_up[:,ngh_ind]; sword.reaches.n_rch_up[ngh_ind]
# sword.reaches.rch_id_up[:,new_ind]; sword.reaches.n_rch_up[new_ind]
# sword.reaches.rch_id_down[:,old_ind]; sword.reaches.n_rch_down[old_ind]
# sword.reaches.rch_id_down[:,ngh_ind]; sword.reaches.n_rch_down[ngh_ind]
# sword.reaches.rch_id_down[:,new_ind]; sword.reaches.n_rch_down[new_ind]

### double node
# old_ind = np.where(sword.centerlines.reach_id[0,:] == 71182702613)[0]
# new_ind = np.where(sword.centerlines.reach_id[0,:] == 71182702856)[0]
# ngh_ind = np.where(sword.centerlines.reach_id[0,:] == 71182702593)[0]
# sword.centerlines.reach_id[1,old_ind]
# sword.centerlines.reach_id[1,ngh_ind] #problematic 
# sword.centerlines.reach_id[1,new_ind] 

# old_ind = np.where(sword.reaches.id == 71182702613)[0]
# new_ind = np.where(sword.reaches.id == 71182702856)[0]
# ngh_ind = np.where(sword.reaches.id == 71182702593)[0]
# sword.reaches.rch_id_up[:,old_ind]; sword.reaches.n_rch_up[old_ind]
# sword.reaches.rch_id_up[:,ngh_ind]; sword.reaches.n_rch_up[ngh_ind] 
# sword.reaches.rch_id_up[:,new_ind]; sword.reaches.n_rch_up[new_ind]
# sword.reaches.rch_id_down[:,old_ind]; sword.reaches.n_rch_down[old_ind]
# sword.reaches.rch_id_down[:,ngh_ind]; sword.reaches.n_rch_down[ngh_ind]
# sword.reaches.rch_id_down[:,new_ind]; sword.reaches.n_rch_down[new_ind]

# vals, cnt = np.unique(sword.nodes.id, return_counts=True)


