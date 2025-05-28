# -*- coding: utf-8 -*-
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
import src.updates.sword_utils as swd 

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

# Line-by-line degugging. 
# region = 'OC'
# version = 'v18'

# File paths. 
paths = swd.prepare_paths(main_dir, region, version)
sword_fn = paths['nc_dir']+paths['nc_fn']
out_dir = paths['update_dir']

# Read data. 
centerlines, nodes, reaches = swd.read_nc(sword_fn)

# Find single node reaches that are not ghost reaches. 
group = np.array([str(ind)[-1] for ind in reaches.id])
agg_ind = np.where((reaches.rch_n_nodes == 1)&(group != '6'))[0]
rmv_dams = np.where(group[agg_ind] == '4')[0] #removing any single node dam reaches from aggregation.
agg_ind = np.delete(agg_ind, rmv_dams)
agg = reaches.id[agg_ind]

# Combine with neighbor reach - ideally downstream neighbor.
print('Aggregating 1-Node Reaches')
rmv_agg = []
multi_dn = []
multi_both = []
max_id = np.max(centerlines.cl_id)
for r in list(range(len(agg))):
    # print(r, len(agg)-1)
    rch = np.where(reaches.id == agg[r])[0]
    nds = np.where(nodes.reach_id == agg[r])[0]
    
    ### if more than one node are identified update reach n nodes and move on. 
    if len(nds) > 1:
        print(agg[r], '*** reach has more than 1 node ***')
        rmv_agg.append(r)
        reaches.rch_n_nodes[rch] = len(nds)
        continue 

    ### aggregate 
    #######################
    #   1 Downstream 
    #######################
    n_rch_dn = reaches.n_rch_down[rch]
    if n_rch_dn == 1:
        downstream = reaches.rch_id_down[np.where(reaches.rch_id_down[:,rch]>0)[0],rch][0]
        ngh_dn_rchs_up = reaches.n_rch_up[np.where(reaches.id == downstream)[0]]
        if ngh_dn_rchs_up > 1:
            ### need to make upstream condition. 
            n_rch_up = reaches.n_rch_up[rch]
            if n_rch_up == 1:
                upstream = reaches.rch_id_up[np.where(reaches.rch_id_up[:,rch]>0)[0],rch][0]
                ngh_up_rchs_dn = reaches.n_rch_down[np.where(reaches.id == upstream)[0]]
                if ngh_up_rchs_dn == 1:
                    print('multiple reaches downstream:', agg[r])
                    multi_dn.append(agg[r])
                    
                    new_rch_id = reaches.rch_id_up[np.where(reaches.rch_id_up[:,rch]>0)[0],rch][0]            
                    new_num = 1
                    new_node_id = int(str(new_rch_id)[0:-1]+'00'+str(new_num)+str(new_rch_id)[-1])
                    #update neighboring node ids to increase by one. 
                    ngh_nds = np.where(nodes.reach_id == new_rch_id)[0]
                    cl_ngh_nds = np.where(centerlines.reach_id[0,:] == new_rch_id)[0]
                    nodes.id[ngh_nds] = nodes.id[ngh_nds]+10
                    centerlines.node_id[0,cl_ngh_nds] = centerlines.node_id[0,cl_ngh_nds]+10
                    ### update centerline variables with new ids. 
                    cind = np.where(centerlines.node_id[0,:]== nodes.id[nds])[0]
                    old_cind = np.where(centerlines.reach_id[0,:]== new_rch_id)[0]
                    #nodes
                    centerlines.node_id[0,cind] = new_node_id
                    centerlines.node_id[1,cind] = new_node_id
                    centerlines.node_id[2,cind] = new_node_id
                    centerlines.node_id[3,cind] = new_node_id
                    #reaches
                    mn_ind = np.where(centerlines.cl_id[old_cind] == np.min(centerlines.cl_id[old_cind]))[0]
                    mx_ind = np.where(centerlines.cl_id[cind] == np.max(centerlines.cl_id[cind]))[0]
                    centerlines.reach_id[0,cind] = new_rch_id
                    centerlines.reach_id[1,cind[mx_ind]] = 0
                    centerlines.reach_id[2,cind[mx_ind]] = 0
                    centerlines.reach_id[3,cind[mx_ind]] = 0
                    centerlines.reach_id[1,old_cind[mn_ind]] = 0
                    centerlines.reach_id[2,old_cind[mn_ind]] = 0
                    centerlines.reach_id[3,old_cind[mn_ind]] = 0
                    #centerline ids
                    old_cl_ids = centerlines.cl_id[old_cind] - np.min(centerlines.cl_id[old_cind]) + 1
                    agg_cl_ids = centerlines.cl_id[cind] - np.min(centerlines.cl_id[cind]) + 1
                    agg_cl_ids = agg_cl_ids+max_id
                    old_cl_ids = old_cl_ids+np.max(agg_cl_ids)
                    centerlines.cl_id[old_cind] = old_cl_ids
                    centerlines.cl_id[cind] = agg_cl_ids
                    max_id = np.max(centerlines.cl_id)
                    
                    ### update node variables 
                    nodes.id[nds] = new_node_id
                    nodes.reach_id[nds] = new_rch_id
                    new_node_cl_ids = np.array([np.min(centerlines.cl_id[cind]), np.max(centerlines.cl_id[cind])]).reshape(2,1)
                    nodes.cl_id[:,nds] = new_node_cl_ids

                    ### update reach variables
                    #important ids
                    update_ids = np.where(centerlines.reach_id[0,:]== new_rch_id)[0]
                    new_rch = np.where(reaches.id == new_rch_id)[0]
                    new_rch_nodes = np.where(nodes.reach_id == new_rch_id)[0]
                    #updating values
                    new_cl_ids = np.array([np.min(centerlines.cl_id[update_ids]), np.max(centerlines.cl_id[update_ids])]).reshape(2,1)
                    reaches.cl_id[:,new_rch] = new_cl_ids
                    reaches.x[new_rch] = np.median(centerlines.x[update_ids])
                    reaches.x_min[new_rch] = np.min(centerlines.x[update_ids])
                    reaches.x_max[new_rch] = np.max(centerlines.x[update_ids])
                    reaches.y[new_rch] = np.median(centerlines.y[update_ids])
                    reaches.y_min[new_rch] = np.min(centerlines.y[update_ids])
                    reaches.y_max[new_rch] = np.max(centerlines.y[update_ids])
                    reaches.rch_n_nodes[new_rch] = reaches.rch_n_nodes[new_rch]+1
                    reaches.len[new_rch] = reaches.len[new_rch]+nodes.len[nds] #add node length to reach length
                    # reaches.dist_out[new_rch] = nodes.dist_out[nds] #new dist out shouldn't change with upstream neighbor
                    #node based values
                    reaches.wse[new_rch] = np.median(nodes.wse[new_rch_nodes])
                    reaches.wse_var[new_rch] = max(nodes.wse_var[new_rch_nodes])
                    reaches.wth[new_rch] = np.median(nodes.wth[new_rch_nodes])
                    reaches.wth_var[new_rch] = max(nodes.wth_var[new_rch_nodes])
                    reaches.nchan_max[new_rch] = max(nodes.nchan_max[new_rch_nodes])
                    reaches.nchan_mod[new_rch] = mode(nodes.nchan_mod[new_rch_nodes])
                    reaches.grod[new_rch] = max(nodes.grod[new_rch_nodes])
                    reaches.grod_fid[new_rch] = max(nodes.grod_fid[new_rch_nodes])
                    reaches.hfalls_fid[new_rch] = max(nodes.hfalls_fid[new_rch_nodes])
                    reaches.lakeflag[new_rch] = mode(nodes.lakeflag[new_rch_nodes])
                    reaches.max_wth[new_rch] = max(nodes.max_wth[new_rch_nodes])
                    #slope 
                    order_ids = np.argsort(nodes.id[new_rch_nodes])
                    slope_pts = np.vstack([nodes.dist_out[new_rch_nodes[order_ids]]/1000, np.ones(len(new_rch_nodes))]).T
                    slope, intercept = np.linalg.lstsq(slope_pts, nodes.wse[new_rch_nodes[order_ids]], rcond=None)[0]
                    reaches.slope[new_rch] = abs(slope) # m/km
                    ### update edit flag for boundary change. 
                    if reaches.edit_flag[new_rch] == 'NaN':
                        edit_val = '6'
                    elif '6' not in reaches.edit_flag[new_rch][0].split(','):
                        edit_val = reaches.edit_flag[new_rch] + ',6'
                    else:
                        edit_val = reaches.edit_flag[new_rch]
                    reaches.edit_flag[new_rch] = edit_val
                    nodes.edit_flag[np.where(nodes.reach_id == new_rch_id)[0]] = edit_val
                    ### topology 
                    #replace new reach upstream neighbors with current reach upstream neighbors. 
                    reaches.n_rch_down[new_rch] = reaches.n_rch_down[rch]
                    reaches.rch_id_down[:,new_rch] = reaches.rch_id_down[:,rch]
                    dn_ngh_rchs = reaches.rch_id_down[:,rch]; dn_ngh_rchs = dn_ngh_rchs[dn_ngh_rchs>0]
                    #replace all new upstream neighbors downstream neighbors with new reach id. 
                    for dr in list(range(len(dn_ngh_rchs))):
                        dn_rch = np.where(reaches.id == dn_ngh_rchs[dr])[0]
                        val = np.where(reaches.rch_id_up[:,dn_rch] == agg[r])[0]
                        if len(val) == 0:
                            print('!!! TOPOLOGY ISSUE !!!', agg[r])
                            break
                        reaches.rch_id_up[val,dn_rch] = new_rch_id
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
            new_rch_id = reaches.rch_id_down[np.where(reaches.rch_id_down[:,rch]>0)[0],rch][0]
            #get new node number. for downstream reach it should be the max node number plus one. 
            ngh_nds = np.where(nodes.reach_id == new_rch_id)[0]
            rch_node_nums = np.array([int(str(ind)[10:13]) for ind in nodes.id[ngh_nds]])
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
            cind = np.where(centerlines.node_id[0,:]== nodes.id[nds])[0]
            old_cind = np.where(centerlines.reach_id[0,:]== new_rch_id)[0]
            #nodes
            centerlines.node_id[0,cind] = new_node_id
            centerlines.node_id[1,cind] = new_node_id
            centerlines.node_id[2,cind] = new_node_id
            centerlines.node_id[3,cind] = new_node_id
            #reaches
            mn_ind = np.where(centerlines.cl_id[cind] == np.min(centerlines.cl_id[cind]))[0]
            mx_ind = np.where(centerlines.cl_id[old_cind] == np.max(centerlines.cl_id[old_cind]))[0]
            centerlines.reach_id[0,cind] = new_rch_id
            centerlines.reach_id[1,cind[mn_ind]] = 0
            centerlines.reach_id[2,cind[mn_ind]] = 0
            centerlines.reach_id[3,cind[mn_ind]] = 0
            centerlines.reach_id[1,old_cind[mx_ind]] = 0
            centerlines.reach_id[2,old_cind[mx_ind]] = 0
            centerlines.reach_id[3,old_cind[mx_ind]] = 0
            #centerline ids
            old_cl_ids = centerlines.cl_id[old_cind] - np.min(centerlines.cl_id[old_cind]) + 1
            agg_cl_ids = centerlines.cl_id[cind] - np.min(centerlines.cl_id[cind]) + 1
            old_cl_ids = old_cl_ids+max_id
            agg_cl_ids = agg_cl_ids+np.max(old_cl_ids)
            centerlines.cl_id[old_cind] = old_cl_ids
            centerlines.cl_id[cind] = agg_cl_ids
            max_id = np.max(centerlines.cl_id)

            ### update node variables 
            nodes.id[nds] = new_node_id
            nodes.reach_id[nds] = new_rch_id
            new_node_cl_ids = np.array([np.min(centerlines.cl_id[cind]), np.max(centerlines.cl_id[cind])]).reshape(2,1)
            nodes.cl_id[:,nds] = new_node_cl_ids

            ### update reach variables
            #important ids
            update_ids = np.where(centerlines.reach_id[0,:]== new_rch_id)[0]
            new_rch = np.where(reaches.id == new_rch_id)[0]
            new_rch_nodes = np.where(nodes.reach_id == new_rch_id)[0]
            #updating values
            new_cl_ids = np.array([np.min(centerlines.cl_id[update_ids]), np.max(centerlines.cl_id[update_ids])]).reshape(2,1)
            reaches.cl_id[:,new_rch] = new_cl_ids
            reaches.x[new_rch] = np.median(centerlines.x[update_ids])
            reaches.x_min[new_rch] = np.min(centerlines.x[update_ids])
            reaches.x_max[new_rch] = np.max(centerlines.x[update_ids])
            reaches.y[new_rch] = np.median(centerlines.y[update_ids])
            reaches.y_min[new_rch] = np.min(centerlines.y[update_ids])
            reaches.y_max[new_rch] = np.max(centerlines.y[update_ids])
            reaches.rch_n_nodes[new_rch] = reaches.rch_n_nodes[new_rch]+1
            reaches.len[new_rch] = reaches.len[new_rch]+nodes.len[nds] #add node length to reach length
            reaches.dist_out[new_rch] = nodes.dist_out[nds] #new dist out should be the node dist out given it is the new max node. 
            #node based values
            reaches.wse[new_rch] = np.median(nodes.wse[new_rch_nodes])
            reaches.wse_var[new_rch] = max(nodes.wse_var[new_rch_nodes])
            reaches.wth[new_rch] = np.median(nodes.wth[new_rch_nodes])
            reaches.wth_var[new_rch] = max(nodes.wth_var[new_rch_nodes])
            reaches.nchan_max[new_rch] = max(nodes.nchan_max[new_rch_nodes])
            reaches.nchan_mod[new_rch] = mode(nodes.nchan_mod[new_rch_nodes])
            reaches.grod[new_rch] = max(nodes.grod[new_rch_nodes])
            reaches.grod_fid[new_rch] = max(nodes.grod_fid[new_rch_nodes])
            reaches.hfalls_fid[new_rch] = max(nodes.hfalls_fid[new_rch_nodes])
            reaches.lakeflag[new_rch] = mode(nodes.lakeflag[new_rch_nodes])
            reaches.max_wth[new_rch] = max(nodes.max_wth[new_rch_nodes])
            #slope 
            order_ids = np.argsort(nodes.id[new_rch_nodes])
            slope_pts = np.vstack([nodes.dist_out[new_rch_nodes[order_ids]]/1000, np.ones(len(new_rch_nodes))]).T
            slope, intercept = np.linalg.lstsq(slope_pts, nodes.wse[new_rch_nodes[order_ids]], rcond=None)[0]
            reaches.slope[new_rch] = abs(slope) # m/km
            ### update edit flag for boundary change. 
            if reaches.edit_flag[new_rch] == 'NaN':
                edit_val = '6'
            elif '6' not in reaches.edit_flag[new_rch][0].split(','):
                edit_val = reaches.edit_flag[new_rch] + ',6'
            else:
                edit_val = reaches.edit_flag[new_rch]
            reaches.edit_flag[new_rch] = edit_val
            nodes.edit_flag[np.where(nodes.reach_id == new_rch_id)[0]] = edit_val
            ### topology 
            #replace new reach upstream neighbors with current reach upstream neighbors. 
            reaches.n_rch_up[new_rch] = reaches.n_rch_up[rch]
            reaches.rch_id_up[:,new_rch] = reaches.rch_id_up[:,rch]
            up_ngh_rchs = reaches.rch_id_up[:,rch]; up_ngh_rchs = up_ngh_rchs[up_ngh_rchs>0]
            #replace all new upstream neighbors downstream neighbors with new reach id. 
            for ur in list(range(len(up_ngh_rchs))):
                up_rch = np.where(reaches.id == up_ngh_rchs[ur])[0]
                val = np.where(reaches.rch_id_down[:,up_rch] == agg[r])[0]
                if len(val) == 0:
                    print('!!! TOPOLOGY ISSUE !!!', agg[r])
                    break
                reaches.rch_id_down[val,up_rch] = new_rch_id
    #######################
    #   Multi Downstream 
    #######################
    else:
        n_rch_up = reaches.n_rch_up[rch]
        if n_rch_up == 1:
            upstream = reaches.rch_id_up[np.where(reaches.rch_id_up[:,rch]>0)[0],rch][0]
            ngh_up_rchs_dn = reaches.n_rch_down[np.where(reaches.id == upstream)[0]]
            if ngh_up_rchs_dn == 1:
                print('multiple reaches downstream:', agg[r])
                multi_dn.append(agg[r])
                    
                new_rch_id = reaches.rch_id_up[np.where(reaches.rch_id_up[:,rch]>0)[0],rch][0]            
                new_num = 1
                new_node_id = int(str(new_rch_id)[0:-1]+'00'+str(new_num)+str(new_rch_id)[-1])
                #update neighboring node ids to increase by one. 
                ngh_nds = np.where(nodes.reach_id == new_rch_id)[0]
                cl_ngh_nds = np.where(centerlines.reach_id[0,:] == new_rch_id)[0]
                nodes.id[ngh_nds] = nodes.id[ngh_nds]+10
                centerlines.node_id[0,cl_ngh_nds] = centerlines.node_id[0,cl_ngh_nds]+10
                ### update centerline variables with new ids. 
                cind = np.where(centerlines.node_id[0,:]== nodes.id[nds])[0]
                old_cind = np.where(centerlines.reach_id[0,:]== new_rch_id)[0]
                #nodes
                centerlines.node_id[0,cind] = new_node_id
                centerlines.node_id[1,cind] = new_node_id
                centerlines.node_id[2,cind] = new_node_id
                centerlines.node_id[3,cind] = new_node_id
                #reaches
                mn_ind = np.where(centerlines.cl_id[old_cind] == np.min(centerlines.cl_id[old_cind]))[0]
                mx_ind = np.where(centerlines.cl_id[cind] == np.max(centerlines.cl_id[cind]))[0]
                centerlines.reach_id[0,cind] = new_rch_id
                centerlines.reach_id[1,cind[mx_ind]] = 0
                centerlines.reach_id[2,cind[mx_ind]] = 0
                centerlines.reach_id[3,cind[mx_ind]] = 0
                centerlines.reach_id[1,old_cind[mn_ind]] = 0
                centerlines.reach_id[2,old_cind[mn_ind]] = 0
                centerlines.reach_id[3,old_cind[mn_ind]] = 0
                #centerline ids
                old_cl_ids = centerlines.cl_id[old_cind] - np.min(centerlines.cl_id[old_cind]) + 1
                agg_cl_ids = centerlines.cl_id[cind] - np.min(centerlines.cl_id[cind]) + 1
                agg_cl_ids = agg_cl_ids+max_id
                old_cl_ids = old_cl_ids+np.max(agg_cl_ids)
                centerlines.cl_id[old_cind] = old_cl_ids
                centerlines.cl_id[cind] = agg_cl_ids
                max_id = np.max(centerlines.cl_id)
                    
                ### update node variables 
                nodes.id[nds] = new_node_id
                nodes.reach_id[nds] = new_rch_id
                new_node_cl_ids = np.array([np.min(centerlines.cl_id[cind]), np.max(centerlines.cl_id[cind])]).reshape(2,1)
                nodes.cl_id[:,nds] = new_node_cl_ids

                ### update reach variables
                #important ids
                update_ids = np.where(centerlines.reach_id[0,:]== new_rch_id)[0]
                new_rch = np.where(reaches.id == new_rch_id)[0]
                new_rch_nodes = np.where(nodes.reach_id == new_rch_id)[0]
                #updating values
                new_cl_ids = np.array([np.min(centerlines.cl_id[update_ids]), np.max(centerlines.cl_id[update_ids])]).reshape(2,1)
                reaches.cl_id[:,new_rch] = new_cl_ids
                reaches.x[new_rch] = np.median(centerlines.x[update_ids])
                reaches.x_min[new_rch] = np.min(centerlines.x[update_ids])
                reaches.x_max[new_rch] = np.max(centerlines.x[update_ids])
                reaches.y[new_rch] = np.median(centerlines.y[update_ids])
                reaches.y_min[new_rch] = np.min(centerlines.y[update_ids])
                reaches.y_max[new_rch] = np.max(centerlines.y[update_ids])
                reaches.rch_n_nodes[new_rch] = reaches.rch_n_nodes[new_rch]+1
                reaches.len[new_rch] = reaches.len[new_rch]+nodes.len[nds] #add node length to reach length
                # reaches.dist_out[new_rch] = nodes.dist_out[nds] #new dist out shouldn't change with upstream neighbor
                #node based values
                reaches.wse[new_rch] = np.median(nodes.wse[new_rch_nodes])
                reaches.wse_var[new_rch] = max(nodes.wse_var[new_rch_nodes])
                reaches.wth[new_rch] = np.median(nodes.wth[new_rch_nodes])
                reaches.wth_var[new_rch] = max(nodes.wth_var[new_rch_nodes])
                reaches.nchan_max[new_rch] = max(nodes.nchan_max[new_rch_nodes])
                reaches.nchan_mod[new_rch] = mode(nodes.nchan_mod[new_rch_nodes])
                reaches.grod[new_rch] = max(nodes.grod[new_rch_nodes])
                reaches.grod_fid[new_rch] = max(nodes.grod_fid[new_rch_nodes])
                reaches.hfalls_fid[new_rch] = max(nodes.hfalls_fid[new_rch_nodes])
                reaches.lakeflag[new_rch] = mode(nodes.lakeflag[new_rch_nodes])
                reaches.max_wth[new_rch] = max(nodes.max_wth[new_rch_nodes])
                #slope 
                order_ids = np.argsort(nodes.id[new_rch_nodes])
                slope_pts = np.vstack([nodes.dist_out[new_rch_nodes[order_ids]]/1000, np.ones(len(new_rch_nodes))]).T
                slope, intercept = np.linalg.lstsq(slope_pts, nodes.wse[new_rch_nodes[order_ids]], rcond=None)[0]
                reaches.slope[new_rch] = abs(slope) # m/km
                ### update edit flag for boundary change. 
                if reaches.edit_flag[new_rch] == 'NaN':
                    edit_val = '6'
                elif '6' not in reaches.edit_flag[new_rch][0].split(','):
                    edit_val = reaches.edit_flag[new_rch] + ',6'
                else:
                    edit_val = reaches.edit_flag[new_rch]
                reaches.edit_flag[new_rch] = edit_val
                nodes.edit_flag[np.where(nodes.reach_id == new_rch_id)[0]] = edit_val
                ### topology 
                #replace new reach upstream neighbors with current reach upstream neighbors. 
                reaches.n_rch_down[new_rch] = reaches.n_rch_down[rch]
                reaches.rch_id_down[:,new_rch] = reaches.rch_id_down[:,rch]
                dn_ngh_rchs = reaches.rch_id_down[:,rch]; dn_ngh_rchs = dn_ngh_rchs[dn_ngh_rchs>0]
                #replace all new upstream neighbors downstream neighbors with new reach id. 
                for dr in list(range(len(dn_ngh_rchs))):
                    dn_rch = np.where(reaches.id == dn_ngh_rchs[dr])[0]
                    val = np.where(reaches.rch_id_up[:,dn_rch] == agg[r])[0]
                    if len(val) == 0:
                        print('!!! TOPOLOGY ISSUE !!!', agg[r])
                        break
                    reaches.rch_id_up[val,dn_rch] = new_rch_id
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

# Delete aggregated reaches. 
print('Deleting 1-Node Reaches')
agg_final = np.delete(agg, rmv_agg)
swd.delete_rchs(reaches, agg_final)

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

# Filler variables.
swd.discharge_attr_nc(reaches)

# Write Data.
print('Writing New NetCDF')
swd.write_nc(centerlines, reaches, nodes, region, sword_fn)

#checking dimensions.
end = time.time()
print('Cl Dimensions:', len(np.unique(centerlines.cl_id)), len(centerlines.cl_id))
print('Node Dimensions:', len(np.unique(centerlines.node_id[0,:])), len(np.unique(nodes.id)), len(nodes.id))
print('Rch Dimensions:', len(np.unique(centerlines.reach_id[0,:])), len(np.unique(nodes.reach_id)), len(np.unique(reaches.id)), len(reaches.id))
print('Reaches with multiple downstream neighbors:', len(multi_dn))
print('Reaches with multiple upstream and downstream neighbors:', len(multi_both))
print('Edit flag values:', np.unique(reaches.edit_flag))
print('DONE IN:', str(np.round((end-start)/60, 2)), 'mins')
