# -*- coding: utf-8 -*-
"""
Aggregating One-Node Reaches (aggregate_1node_rchs.py)
======================================================

This script identifies and aggregates 1-node length 
reaches in the SWOT River Database (SWORD) with neighboring 
reaches.

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python aggregate_1node_rchs.py NA v17 

"""


from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import time
from statistics import mode
import pandas as pd
import argparse
from src.updates.sword import SWORD

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

# File paths. 
sword = SWORD(main_dir, region, version)
out_dir = sword.paths['update_dir']

# Find single node reaches that are not ghost reaches. 
group = np.array([str(ind)[-1] for ind in sword.reaches.id])
agg_ind = np.where((sword.reaches.rch_n_nodes == 1)&(group != '6'))[0]
rmv_dams = np.where(group[agg_ind] == '4')[0] #removing any single node dam reaches from aggregation.
agg_ind = np.delete(agg_ind, rmv_dams)
agg = sword.reaches.id[agg_ind]

# Combine with neighbor reach - ideally downstream neighbor.
print('Aggregating 1-Node Reaches')
rmv_agg = []
multi_dn = []
multi_both = []
max_id = np.max(sword.centerlines.cl_id)
for r in list(range(len(agg))):
    # print(r, len(agg)-1)
    rch = np.where(sword.reaches.id == agg[r])[0]
    nds = np.where(sword.nodes.reach_id == agg[r])[0]
    
    ### if more than one node are identified update reach n nodes and move on. 
    if len(nds) > 1:
        print(agg[r], '*** reach has more than 1 node ***')
        rmv_agg.append(r)
        sword.reaches.rch_n_nodes[rch] = len(nds)
        continue 

    ### aggregate 
    #######################
    #   1 Downstream 
    #######################
    n_rch_dn = sword.reaches.n_rch_down[rch]
    if n_rch_dn == 1:
        downstream = sword.reaches.rch_id_down[np.where(sword.reaches.rch_id_down[:,rch]>0)[0],rch][0]
        ngh_dn_rchs_up = sword.reaches.n_rch_up[np.where(sword.reaches.id == downstream)[0]]
        if ngh_dn_rchs_up > 1:
            ### need to make upstream condition. 
            n_rch_up = sword.reaches.n_rch_up[rch]
            if n_rch_up == 1:
                upstream = sword.reaches.rch_id_up[np.where(sword.reaches.rch_id_up[:,rch]>0)[0],rch][0]
                ngh_up_rchs_dn = sword.reaches.n_rch_down[np.where(sword.reaches.id == upstream)[0]]
                if ngh_up_rchs_dn == 1:
                    print('multiple reaches downstream:', agg[r])
                    multi_dn.append(agg[r])
                    
                    new_rch_id = sword.reaches.rch_id_up[np.where(sword.reaches.rch_id_up[:,rch]>0)[0],rch][0]            
                    new_num = 1
                    new_node_id = int(str(new_rch_id)[0:-1]+'00'+str(new_num)+str(new_rch_id)[-1])
                    #update neighboring node ids to increase by one. 
                    ngh_nds = np.where(sword.nodes.reach_id == new_rch_id)[0]
                    cl_ngh_nds = np.where(sword.centerlines.reach_id[0,:] == new_rch_id)[0]
                    sword.nodes.id[ngh_nds] = sword.nodes.id[ngh_nds]+10
                    sword.centerlines.node_id[0,cl_ngh_nds] = sword.centerlines.node_id[0,cl_ngh_nds]+10
                    ### update centerline variables with new ids. 
                    cind = np.where(sword.centerlines.node_id[0,:]== sword.nodes.id[nds])[0]
                    old_cind = np.where(sword.centerlines.reach_id[0,:]== new_rch_id)[0]
                    #nodes
                    sword.centerlines.node_id[0,cind] = new_node_id
                    sword.centerlines.node_id[1,cind] = new_node_id
                    sword.centerlines.node_id[2,cind] = new_node_id
                    sword.centerlines.node_id[3,cind] = new_node_id
                    #reaches
                    mn_ind = np.where(sword.centerlines.cl_id[old_cind] == np.min(sword.centerlines.cl_id[old_cind]))[0]
                    mx_ind = np.where(sword.centerlines.cl_id[cind] == np.max(sword.centerlines.cl_id[cind]))[0]
                    sword.centerlines.reach_id[0,cind] = new_rch_id
                    sword.centerlines.reach_id[1,cind[mx_ind]] = 0
                    sword.centerlines.reach_id[2,cind[mx_ind]] = 0
                    sword.centerlines.reach_id[3,cind[mx_ind]] = 0
                    sword.centerlines.reach_id[1,old_cind[mn_ind]] = 0
                    sword.centerlines.reach_id[2,old_cind[mn_ind]] = 0
                    sword.centerlines.reach_id[3,old_cind[mn_ind]] = 0
                    #centerline ids
                    old_cl_ids = sword.centerlines.cl_id[old_cind] - np.min(sword.centerlines.cl_id[old_cind]) + 1
                    agg_cl_ids = sword.centerlines.cl_id[cind] - np.min(sword.centerlines.cl_id[cind]) + 1
                    agg_cl_ids = agg_cl_ids+max_id
                    old_cl_ids = old_cl_ids+np.max(agg_cl_ids)
                    sword.centerlines.cl_id[old_cind] = old_cl_ids
                    sword.centerlines.cl_id[cind] = agg_cl_ids
                    max_id = np.max(sword.centerlines.cl_id)
                    
                    ### update node variables 
                    sword.nodes.id[nds] = new_node_id
                    sword.nodes.reach_id[nds] = new_rch_id
                    new_node_cl_ids = np.array([np.min(sword.centerlines.cl_id[cind]), np.max(sword.centerlines.cl_id[cind])]).reshape(2,1)
                    sword.nodes.cl_id[:,nds] = new_node_cl_ids

                    ### update reach variables
                    #important ids
                    update_ids = np.where(sword.centerlines.reach_id[0,:]== new_rch_id)[0]
                    new_rch = np.where(sword.reaches.id == new_rch_id)[0]
                    new_rch_nodes = np.where(sword.nodes.reach_id == new_rch_id)[0]
                    #updating values
                    new_cl_ids = np.array([np.min(sword.centerlines.cl_id[update_ids]), np.max(sword.centerlines.cl_id[update_ids])]).reshape(2,1)
                    sword.reaches.cl_id[:,new_rch] = new_cl_ids
                    sword.reaches.x[new_rch] = np.median(sword.centerlines.x[update_ids])
                    sword.reaches.x_min[new_rch] = np.min(sword.centerlines.x[update_ids])
                    sword.reaches.x_max[new_rch] = np.max(sword.centerlines.x[update_ids])
                    sword.reaches.y[new_rch] = np.median(sword.centerlines.y[update_ids])
                    sword.reaches.y_min[new_rch] = np.min(sword.centerlines.y[update_ids])
                    sword.reaches.y_max[new_rch] = np.max(sword.centerlines.y[update_ids])
                    sword.reaches.rch_n_nodes[new_rch] = sword.reaches.rch_n_nodes[new_rch]+1
                    sword.reaches.len[new_rch] = sword.reaches.len[new_rch]+sword.nodes.len[nds] #add node length to reach length
                    # sword.reaches.dist_out[new_rch] = sword.nodes.dist_out[nds] #new dist out shouldn't change with upstream neighbor
                    #node based values
                    sword.reaches.wse[new_rch] = np.median(sword.nodes.wse[new_rch_nodes])
                    sword.reaches.wse_var[new_rch] = max(sword.nodes.wse_var[new_rch_nodes])
                    sword.reaches.wth[new_rch] = np.median(sword.nodes.wth[new_rch_nodes])
                    sword.reaches.wth_var[new_rch] = max(sword.nodes.wth_var[new_rch_nodes])
                    sword.reaches.nchan_max[new_rch] = max(sword.nodes.nchan_max[new_rch_nodes])
                    sword.reaches.nchan_mod[new_rch] = mode(sword.nodes.nchan_mod[new_rch_nodes])
                    sword.reaches.grod[new_rch] = max(sword.nodes.grod[new_rch_nodes])
                    sword.reaches.grod_fid[new_rch] = max(sword.nodes.grod_fid[new_rch_nodes])
                    sword.reaches.hfalls_fid[new_rch] = max(sword.nodes.hfalls_fid[new_rch_nodes])
                    sword.reaches.lakeflag[new_rch] = mode(sword.nodes.lakeflag[new_rch_nodes])
                    sword.reaches.max_wth[new_rch] = max(sword.nodes.max_wth[new_rch_nodes])
                    #slope 
                    order_ids = np.argsort(sword.nodes.id[new_rch_nodes])
                    slope_pts = np.vstack([sword.nodes.dist_out[new_rch_nodes[order_ids]]/1000, np.ones(len(new_rch_nodes))]).T
                    slope, intercept = np.linalg.lstsq(slope_pts, sword.nodes.wse[new_rch_nodes[order_ids]], rcond=None)[0]
                    sword.reaches.slope[new_rch] = abs(slope) # m/km
                    ### update edit flag for boundary change. 
                    if sword.reaches.edit_flag[new_rch] == 'NaN':
                        edit_val = '6'
                    elif '6' not in sword.reaches.edit_flag[new_rch][0].split(','):
                        edit_val = sword.reaches.edit_flag[new_rch] + ',6'
                    else:
                        edit_val = sword.reaches.edit_flag[new_rch]
                    sword.reaches.edit_flag[new_rch] = edit_val
                    sword.nodes.edit_flag[np.where(sword.nodes.reach_id == new_rch_id)[0]] = edit_val
                    ### topology 
                    #replace new reach upstream neighbors with current reach upstream neighbors. 
                    sword.reaches.n_rch_down[new_rch] = sword.reaches.n_rch_down[rch]
                    sword.reaches.rch_id_down[:,new_rch] = sword.reaches.rch_id_down[:,rch]
                    dn_ngh_rchs = sword.reaches.rch_id_down[:,rch]; dn_ngh_rchs = dn_ngh_rchs[dn_ngh_rchs>0]
                    #replace all new upstream neighbors downstream neighbors with new reach id. 
                    for dr in list(range(len(dn_ngh_rchs))):
                        dn_rch = np.where(sword.reaches.id == dn_ngh_rchs[dr])[0]
                        val = np.where(sword.reaches.rch_id_up[:,dn_rch] == agg[r])[0]
                        if len(val) == 0:
                            print('!!! TOPOLOGY ISSUE !!!', agg[r])
                            break
                        sword.reaches.rch_id_up[val,dn_rch] = new_rch_id
                else:
                    print('multiple upstream and downstream reaches:', agg[r])
                    multi_both.append(agg[r])
                    rmv_agg.append(r)
                    continue
            else:
                print('multiple upstream and downstream reaches:', agg[r])
                multi_both.append(agg[r])
                rmv_agg.append(r)
                continue
                
        else:
            new_rch_id = sword.reaches.rch_id_down[np.where(sword.reaches.rch_id_down[:,rch]>0)[0],rch][0]
            #get new node number. for downstream reach it should be the max node number plus one. 
            ngh_nds = np.where(sword.nodes.reach_id == new_rch_id)[0]
            rch_node_nums = np.array([int(str(ind)[10:13]) for ind in sword.nodes.id[ngh_nds]])
            new_num = max(rch_node_nums)+1
            if len(str(new_num)) == 1:
                fill = '00'
                new_node_id = int(str(new_rch_id)[0:-1]+fill+str(new_num)+str(new_rch_id)[-1])
            if len(str(new_num)) == 2:
                fill = '0'
                new_node_id = int(str(new_rch_id)[0:-1]+fill+str(new_num)+str(new_rch_id)[-1])
            if len(str(new_num)) == 3:
                new_node_id = int(str(new_rch_id)[0:-1]+str(new_num)+str(new_rch_id)[-1])

            # if new_rch_id == 71181300371:
            #     break

            ### update centerline variables
            cind = np.where(sword.centerlines.node_id[0,:]== sword.nodes.id[nds])[0]
            old_cind = np.where(sword.centerlines.reach_id[0,:]== new_rch_id)[0]
            #nodes
            sword.centerlines.node_id[0,cind] = new_node_id
            sword.centerlines.node_id[1,cind] = new_node_id
            sword.centerlines.node_id[2,cind] = new_node_id
            sword.centerlines.node_id[3,cind] = new_node_id
            #reaches
            mn_ind = np.where(sword.centerlines.cl_id[cind] == np.min(sword.centerlines.cl_id[cind]))[0]
            mx_ind = np.where(sword.centerlines.cl_id[old_cind] == np.max(sword.centerlines.cl_id[old_cind]))[0]
            sword.centerlines.reach_id[0,cind] = new_rch_id
            sword.centerlines.reach_id[1,cind[mn_ind]] = 0
            sword.centerlines.reach_id[2,cind[mn_ind]] = 0
            sword.centerlines.reach_id[3,cind[mn_ind]] = 0
            sword.centerlines.reach_id[1,old_cind[mx_ind]] = 0
            sword.centerlines.reach_id[2,old_cind[mx_ind]] = 0
            sword.centerlines.reach_id[3,old_cind[mx_ind]] = 0
            #centerline ids
            old_cl_ids = sword.centerlines.cl_id[old_cind] - np.min(sword.centerlines.cl_id[old_cind]) + 1
            agg_cl_ids = sword.centerlines.cl_id[cind] - np.min(sword.centerlines.cl_id[cind]) + 1
            old_cl_ids = old_cl_ids+max_id
            agg_cl_ids = agg_cl_ids+np.max(old_cl_ids)
            sword.centerlines.cl_id[old_cind] = old_cl_ids
            sword.centerlines.cl_id[cind] = agg_cl_ids
            max_id = np.max(sword.centerlines.cl_id)

            ### update node variables 
            sword.nodes.id[nds] = new_node_id
            sword.nodes.reach_id[nds] = new_rch_id
            new_node_cl_ids = np.array([np.min(sword.centerlines.cl_id[cind]), np.max(sword.centerlines.cl_id[cind])]).reshape(2,1)
            sword.nodes.cl_id[:,nds] = new_node_cl_ids

            ### update reach variables
            #important ids
            update_ids = np.where(sword.centerlines.reach_id[0,:]== new_rch_id)[0]
            new_rch = np.where(sword.reaches.id == new_rch_id)[0]
            new_rch_nodes = np.where(sword.nodes.reach_id == new_rch_id)[0]
            #updating values
            new_cl_ids = np.array([np.min(sword.centerlines.cl_id[update_ids]), np.max(sword.centerlines.cl_id[update_ids])]).reshape(2,1)
            sword.reaches.cl_id[:,new_rch] = new_cl_ids
            sword.reaches.x[new_rch] = np.median(sword.centerlines.x[update_ids])
            sword.reaches.x_min[new_rch] = np.min(sword.centerlines.x[update_ids])
            sword.reaches.x_max[new_rch] = np.max(sword.centerlines.x[update_ids])
            sword.reaches.y[new_rch] = np.median(sword.centerlines.y[update_ids])
            sword.reaches.y_min[new_rch] = np.min(sword.centerlines.y[update_ids])
            sword.reaches.y_max[new_rch] = np.max(sword.centerlines.y[update_ids])
            sword.reaches.rch_n_nodes[new_rch] = sword.reaches.rch_n_nodes[new_rch]+1
            sword.reaches.len[new_rch] = sword.reaches.len[new_rch]+sword.nodes.len[nds] #add node length to reach length
            sword.reaches.dist_out[new_rch] = sword.nodes.dist_out[nds] #new dist out should be the node dist out given it is the new max node. 
            #node based values
            sword.reaches.wse[new_rch] = np.median(sword.nodes.wse[new_rch_nodes])
            sword.reaches.wse_var[new_rch] = max(sword.nodes.wse_var[new_rch_nodes])
            sword.reaches.wth[new_rch] = np.median(sword.nodes.wth[new_rch_nodes])
            sword.reaches.wth_var[new_rch] = max(sword.nodes.wth_var[new_rch_nodes])
            sword.reaches.nchan_max[new_rch] = max(sword.nodes.nchan_max[new_rch_nodes])
            sword.reaches.nchan_mod[new_rch] = mode(sword.nodes.nchan_mod[new_rch_nodes])
            sword.reaches.grod[new_rch] = max(sword.nodes.grod[new_rch_nodes])
            sword.reaches.grod_fid[new_rch] = max(sword.nodes.grod_fid[new_rch_nodes])
            sword.reaches.hfalls_fid[new_rch] = max(sword.nodes.hfalls_fid[new_rch_nodes])
            sword.reaches.lakeflag[new_rch] = mode(sword.nodes.lakeflag[new_rch_nodes])
            sword.reaches.max_wth[new_rch] = max(sword.nodes.max_wth[new_rch_nodes])
            #slope 
            order_ids = np.argsort(sword.nodes.id[new_rch_nodes])
            slope_pts = np.vstack([sword.nodes.dist_out[new_rch_nodes[order_ids]]/1000, np.ones(len(new_rch_nodes))]).T
            slope, intercept = np.linalg.lstsq(slope_pts, sword.nodes.wse[new_rch_nodes[order_ids]], rcond=None)[0]
            sword.reaches.slope[new_rch] = abs(slope) # m/km
            ### update edit flag for boundary change. 
            if sword.reaches.edit_flag[new_rch] == 'NaN':
                edit_val = '6'
            elif '6' not in sword.reaches.edit_flag[new_rch][0].split(','):
                edit_val = sword.reaches.edit_flag[new_rch] + ',6'
            else:
                edit_val = sword.reaches.edit_flag[new_rch]
            sword.reaches.edit_flag[new_rch] = edit_val
            sword.nodes.edit_flag[np.where(sword.nodes.reach_id == new_rch_id)[0]] = edit_val
            ### topology 
            #replace new reach upstream neighbors with current reach upstream neighbors. 
            sword.reaches.n_rch_up[new_rch] = sword.reaches.n_rch_up[rch]
            sword.reaches.rch_id_up[:,new_rch] = sword.reaches.rch_id_up[:,rch]
            up_ngh_rchs = sword.reaches.rch_id_up[:,rch]; up_ngh_rchs = up_ngh_rchs[up_ngh_rchs>0]
            #replace all new upstream neighbors downstream neighbors with new reach id. 
            for ur in list(range(len(up_ngh_rchs))):
                up_rch = np.where(sword.reaches.id == up_ngh_rchs[ur])[0]
                val = np.where(sword.reaches.rch_id_down[:,up_rch] == agg[r])[0]
                if len(val) == 0:
                    print('!!! TOPOLOGY ISSUE !!!', agg[r])
                    break
                sword.reaches.rch_id_down[val,up_rch] = new_rch_id
    #######################
    #   Multi Downstream 
    #######################
    else:
        n_rch_up = sword.reaches.n_rch_up[rch]
        if n_rch_up == 1:
            upstream = sword.reaches.rch_id_up[np.where(sword.reaches.rch_id_up[:,rch]>0)[0],rch][0]
            ngh_up_rchs_dn = sword.reaches.n_rch_down[np.where(sword.reaches.id == upstream)[0]]
            if ngh_up_rchs_dn == 1:
                print('multiple reaches downstream:', agg[r])
                multi_dn.append(agg[r])
                    
                new_rch_id = sword.reaches.rch_id_up[np.where(sword.reaches.rch_id_up[:,rch]>0)[0],rch][0]            
                new_num = 1
                new_node_id = int(str(new_rch_id)[0:-1]+'00'+str(new_num)+str(new_rch_id)[-1])
                #update neighboring node ids to increase by one. 
                ngh_nds = np.where(sword.nodes.reach_id == new_rch_id)[0]
                cl_ngh_nds = np.where(sword.centerlines.reach_id[0,:] == new_rch_id)[0]
                sword.nodes.id[ngh_nds] = sword.nodes.id[ngh_nds]+10
                sword.centerlines.node_id[0,cl_ngh_nds] = sword.centerlines.node_id[0,cl_ngh_nds]+10
                ### update centerline variables with new ids. 
                cind = np.where(sword.centerlines.node_id[0,:]== sword.nodes.id[nds])[0]
                old_cind = np.where(sword.centerlines.reach_id[0,:]== new_rch_id)[0]
                #nodes
                sword.centerlines.node_id[0,cind] = new_node_id
                sword.centerlines.node_id[1,cind] = new_node_id
                sword.centerlines.node_id[2,cind] = new_node_id
                sword.centerlines.node_id[3,cind] = new_node_id
                #reaches
                mn_ind = np.where(sword.centerlines.cl_id[old_cind] == np.min(sword.centerlines.cl_id[old_cind]))[0]
                mx_ind = np.where(sword.centerlines.cl_id[cind] == np.max(sword.centerlines.cl_id[cind]))[0]
                sword.centerlines.reach_id[0,cind] = new_rch_id
                sword.centerlines.reach_id[1,cind[mx_ind]] = 0
                sword.centerlines.reach_id[2,cind[mx_ind]] = 0
                sword.centerlines.reach_id[3,cind[mx_ind]] = 0
                sword.centerlines.reach_id[1,old_cind[mn_ind]] = 0
                sword.centerlines.reach_id[2,old_cind[mn_ind]] = 0
                sword.centerlines.reach_id[3,old_cind[mn_ind]] = 0
                #centerline ids
                old_cl_ids = sword.centerlines.cl_id[old_cind] - np.min(sword.centerlines.cl_id[old_cind]) + 1
                agg_cl_ids = sword.centerlines.cl_id[cind] - np.min(sword.centerlines.cl_id[cind]) + 1
                agg_cl_ids = agg_cl_ids+max_id
                old_cl_ids = old_cl_ids+np.max(agg_cl_ids)
                sword.centerlines.cl_id[old_cind] = old_cl_ids
                sword.centerlines.cl_id[cind] = agg_cl_ids
                max_id = np.max(sword.centerlines.cl_id)
                    
                ### update node variables 
                sword.nodes.id[nds] = new_node_id
                sword.nodes.reach_id[nds] = new_rch_id
                new_node_cl_ids = np.array([np.min(sword.centerlines.cl_id[cind]), np.max(sword.centerlines.cl_id[cind])]).reshape(2,1)
                sword.nodes.cl_id[:,nds] = new_node_cl_ids

                ### update reach variables
                #important ids
                update_ids = np.where(sword.centerlines.reach_id[0,:]== new_rch_id)[0]
                new_rch = np.where(sword.reaches.id == new_rch_id)[0]
                new_rch_nodes = np.where(sword.nodes.reach_id == new_rch_id)[0]
                #updating values
                new_cl_ids = np.array([np.min(sword.centerlines.cl_id[update_ids]), np.max(sword.centerlines.cl_id[update_ids])]).reshape(2,1)
                sword.reaches.cl_id[:,new_rch] = new_cl_ids
                sword.reaches.x[new_rch] = np.median(sword.centerlines.x[update_ids])
                sword.reaches.x_min[new_rch] = np.min(sword.centerlines.x[update_ids])
                sword.reaches.x_max[new_rch] = np.max(sword.centerlines.x[update_ids])
                sword.reaches.y[new_rch] = np.median(sword.centerlines.y[update_ids])
                sword.reaches.y_min[new_rch] = np.min(sword.centerlines.y[update_ids])
                sword.reaches.y_max[new_rch] = np.max(sword.centerlines.y[update_ids])
                sword.reaches.rch_n_nodes[new_rch] = sword.reaches.rch_n_nodes[new_rch]+1
                sword.reaches.len[new_rch] = sword.reaches.len[new_rch]+sword.nodes.len[nds] #add node length to reach length
                # sword.reaches.dist_out[new_rch] = sword.nodes.dist_out[nds] #new dist out shouldn't change with upstream neighbor
                #node based values
                sword.reaches.wse[new_rch] = np.median(sword.nodes.wse[new_rch_nodes])
                sword.reaches.wse_var[new_rch] = max(sword.nodes.wse_var[new_rch_nodes])
                sword.reaches.wth[new_rch] = np.median(sword.nodes.wth[new_rch_nodes])
                sword.reaches.wth_var[new_rch] = max(sword.nodes.wth_var[new_rch_nodes])
                sword.reaches.nchan_max[new_rch] = max(sword.nodes.nchan_max[new_rch_nodes])
                sword.reaches.nchan_mod[new_rch] = mode(sword.nodes.nchan_mod[new_rch_nodes])
                sword.reaches.grod[new_rch] = max(sword.nodes.grod[new_rch_nodes])
                sword.reaches.grod_fid[new_rch] = max(sword.nodes.grod_fid[new_rch_nodes])
                sword.reaches.hfalls_fid[new_rch] = max(sword.nodes.hfalls_fid[new_rch_nodes])
                sword.reaches.lakeflag[new_rch] = mode(sword.nodes.lakeflag[new_rch_nodes])
                sword.reaches.max_wth[new_rch] = max(sword.nodes.max_wth[new_rch_nodes])
                #slope 
                order_ids = np.argsort(sword.nodes.id[new_rch_nodes])
                slope_pts = np.vstack([sword.nodes.dist_out[new_rch_nodes[order_ids]]/1000, np.ones(len(new_rch_nodes))]).T
                slope, intercept = np.linalg.lstsq(slope_pts, sword.nodes.wse[new_rch_nodes[order_ids]], rcond=None)[0]
                sword.reaches.slope[new_rch] = abs(slope) # m/km
                ### update edit flag for boundary change. 
                if sword.reaches.edit_flag[new_rch] == 'NaN':
                    edit_val = '6'
                elif '6' not in sword.reaches.edit_flag[new_rch][0].split(','):
                    edit_val = sword.reaches.edit_flag[new_rch] + ',6'
                else:
                    edit_val = sword.reaches.edit_flag[new_rch]
                sword.reaches.edit_flag[new_rch] = edit_val
                sword.nodes.edit_flag[np.where(sword.nodes.reach_id == new_rch_id)[0]] = edit_val
                ### topology 
                #replace new reach upstream neighbors with current reach upstream neighbors. 
                sword.reaches.n_rch_down[new_rch] = sword.reaches.n_rch_down[rch]
                sword.reaches.rch_id_down[:,new_rch] = sword.reaches.rch_id_down[:,rch]
                dn_ngh_rchs = sword.reaches.rch_id_down[:,rch]; dn_ngh_rchs = dn_ngh_rchs[dn_ngh_rchs>0]
                #replace all new upstream neighbors downstream neighbors with new reach id. 
                for dr in list(range(len(dn_ngh_rchs))):
                    dn_rch = np.where(sword.reaches.id == dn_ngh_rchs[dr])[0]
                    val = np.where(sword.reaches.rch_id_up[:,dn_rch] == agg[r])[0]
                    if len(val) == 0:
                        print('!!! TOPOLOGY ISSUE !!!', agg[r])
                        break
                    sword.reaches.rch_id_up[val,dn_rch] = new_rch_id
            else:
                print('multiple upstream and downstream reaches:', agg[r])
                multi_both.append(agg[r])
                rmv_agg.append(r)
                continue
        else:
            print('multiple upstream and downstream reaches:', agg[r])
            multi_both.append(agg[r])
            rmv_agg.append(r)
            continue

# Delete aggregated sword.reaches. 
print('Deleting 1-Node Reaches')
agg_final = np.delete(agg, rmv_agg)
sword.delete_rchs(agg_final)

# Write csv files of aggregated reach ids. 
multi_dn_csv = {'reach_id': np.array(multi_dn).astype('int64')}
multi_dn_csv = pd.DataFrame(multi_dn_csv)
multi_dn_csv.to_csv(out_dir+region.lower()+'_multi_ds_1node_rchs.csv', index=False)

multi_both_csv = {'reach_id': np.array(multi_both).astype('int64')}
multi_both_csv = pd.DataFrame(multi_both_csv)
multi_both_csv.to_csv(out_dir+region.lower()+'_multi_us_ds_1node_rchs.csv', index=False)

all_agg_csv = {'reach_id': np.array(agg).astype('int64')}
all_agg_csv = pd.DataFrame(all_agg_csv)
all_agg_csv.to_csv(out_dir+region.lower()+'_1node_rchs.csv', index=False)

# Write Data.
print('Writing New NetCDF')
sword.save_nc()

#checking dimensions.
end = time.time()
print('Cl Dimensions:', len(np.unique(sword.centerlines.cl_id)), len(sword.centerlines.cl_id))
print('Node Dimensions:', len(np.unique(sword.centerlines.node_id[0,:])), len(np.unique(sword.nodes.id)), len(sword.nodes.id))
print('Rch Dimensions:', len(np.unique(sword.centerlines.reach_id[0,:])), len(np.unique(sword.nodes.reach_id)), len(np.unique(sword.reaches.id)), len(sword.reaches.id))
print('Reaches with multiple downstream neighbors:', len(multi_dn))
print('Reaches with multiple upstream and downstream neighbors:', len(multi_both))
print('Edit flag values:', np.unique(sword.reaches.edit_flag))
print('DONE IN:', str(np.round((end-start)/60, 2)), 'mins')
