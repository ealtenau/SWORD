# -*- coding: utf-8 -*-
'''
This script goes through and re-creates the nodes for an indentified reach. 
Used mostly for incorrect node ordering or large/small node lengths
(i.e. node length = 0 or node length > 1000).

(c) E. Altenau 4/22/2025.
'''

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import pandas as pd
import time
import argparse
from scipy import stats as st
import src.updates.sword_utils as swd
import src.updates.calc_utils as ct 

start_all = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="continental region", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("csv", help="csv file of reaches to delete", type = str)
args = parser.parse_args()

region = args.region
version = args.version
 
paths = swd.prepare_paths(main_dir, region, version)
sword_fn = paths['geom_dir']+paths['geom_fn']
csv_dir1 = paths['update_dir']+region.lower()+'_node_order_problems.csv'
csv_dir2 = paths['update_dir']+region.lower()+'_node_length_probems.csv'

#read sword
centerlines, nodes, reaches = swd.read_nc(sword_fn)
cl_node_num_int = np.array([int(str(ind)[10:13]) for ind in centerlines.node_id[0,:]])

#read csv 
redo_order_df = pd.read_csv(csv_dir1) 
redo_len_df = pd.read_csv(csv_dir2) 

redo_order = np.array(redo_order_df['reach_id']) 
redo_len = np.array(redo_len_df['reach_id']) 
redo_rch = np.append(redo_order, redo_len)

unq_rchs = np.unique(redo_rch)
for r in list(range(len(unq_rchs))):
    print(r, unq_rchs[r], len(unq_rchs)-1)
    cl_r = np.where(centerlines.reach_id[0,:] == unq_rchs[r])[0]
    order_ids = np.argsort(centerlines.cl_id[cl_r])
    nodes_rch =  cl_node_num_int[cl_r[order_ids]]
    ## redo nodes for reach. 
    subnodes = swd.Object()
    old_nums = centerlines.node_id[0,cl_r[order_ids]]
    num_nodes = len(np.unique(old_nums))
    cl_ids = centerlines.cl_id[cl_r[order_ids]]
    break_int = np.ceil(len(cl_ids)/num_nodes)
    breaks = np.arange(0,len(cl_ids),int(break_int))
    breaks = np.append(breaks, len(cl_ids))
    new_nums = np.zeros(len(cl_ids),dtype=int)
    cnt = 1
    for b in list(range(len(breaks)-1)):
        rng = breaks[b+1]-breaks[b]
        if rng <= 3:
            count = cnt-1
        else:
            count = cnt
        #create actual id
        if len(str(count)) == 1:
            fill = '00'
            new_nums[breaks[b]:breaks[b+1]] = int(str(int(unq_rchs[r]))[:-1]+fill+str(count)+str(int(unq_rchs[r]))[-1])
        if len(str(count)) == 2:
            fill = '0'
            new_nums[breaks[b]:breaks[b+1]] = int(str(int(unq_rchs[r]))[:-1]+fill+str(count)+str(int(unq_rchs[r]))[-1])
        if len(str(count)) == 3:
            new_nums[breaks[b]:breaks[b+1]] = int(str(int(unq_rchs[r]))[:-1]+str(count)+str(int(unq_rchs[r]))[-1])
        cnt = cnt+1

    #update centerline level
    centerlines.node_id[0,cl_r[order_ids]] = new_nums

    #update n_nodes for reach level...
    current = np.where(reaches.id == unq_rchs[r])[0]
    reaches.rch_n_nodes[current] = len(np.unique(new_nums))
    x_coords = centerlines.x[cl_r[order_ids]]
    y_coords = centerlines.y[cl_r[order_ids]]
    rdiff = ct.get_distances(x_coords,y_coords)

    #create fill variables
    unq_nodes = np.unique(new_nums)
    subnodes.id = np.zeros(len(unq_nodes), dtype=int)
    subnodes.cl_id = np.zeros((2,len(unq_nodes)))
    subnodes.x = np.zeros(len(unq_nodes))
    subnodes.y = np.zeros(len(unq_nodes))
    subnodes.len = np.zeros(len(unq_nodes))
    subnodes.wse = np.zeros(len(unq_nodes))
    subnodes.wse_var = np.zeros(len(unq_nodes))
    subnodes.wth = np.zeros(len(unq_nodes))
    subnodes.wth_var = np.zeros(len(unq_nodes))
    subnodes.grod = np.zeros(len(unq_nodes))
    subnodes.grod_fid = np.zeros(len(unq_nodes))
    subnodes.hfalls_fid = np.zeros(len(unq_nodes))
    subnodes.nchan_max = np.zeros(len(unq_nodes))
    subnodes.nchan_mod = np.zeros(len(unq_nodes))
    subnodes.dist_out = np.zeros(len(unq_nodes))
    subnodes.reach_id = np.zeros(len(unq_nodes))
    subnodes.facc = np.zeros(len(unq_nodes))
    subnodes.lakeflag = np.zeros(len(unq_nodes))
    subnodes.wth_coef = np.zeros(len(unq_nodes))
    subnodes.ext_dist_coef = np.zeros(len(unq_nodes))
    subnodes.max_wth = np.zeros(len(unq_nodes))
    subnodes.meand_len = np.zeros(len(unq_nodes))
    subnodes.river_name = np.repeat('NODATA', len(unq_nodes))
    subnodes.manual_add = np.zeros(len(unq_nodes))
    subnodes.sinuosity = np.zeros(len(unq_nodes))
    subnodes.edit_flag = np.repeat('NaN', len(unq_nodes))
    subnodes.trib_flag = np.zeros(len(unq_nodes))
    subnodes.path_freq = np.zeros(len(unq_nodes))
    subnodes.path_order = np.zeros(len(unq_nodes))
    subnodes.path_segs = np.zeros(len(unq_nodes))
    subnodes.main_side = np.zeros(len(unq_nodes))
    subnodes.strm_order = np.zeros(len(unq_nodes))
    subnodes.end_rch = np.zeros(len(unq_nodes))
    subnodes.network = np.zeros(len(unq_nodes))

    #loop through them and add attributes 
    for n in list(range(len(unq_nodes))):
        pts = np.where(new_nums == unq_nodes[n])[0]
        old_node = np.where(nodes.id == st.mode(old_nums[pts])[0])[0]
            
        subnodes.id[n] = unq_nodes[n]
        subnodes.cl_id[0,n] = min(cl_ids[pts])
        subnodes.cl_id[1,n] = max(cl_ids[pts])
        subnodes.x[n] = np.median(centerlines.x[cl_r[order_ids[pts]]])
        subnodes.y[n] = np.median(centerlines.y[cl_r[order_ids[pts]]])
        subnodes.wse[n] = nodes.wse[old_node][0]
        subnodes.wse_var[n] = nodes.wse_var[old_node][0]
        subnodes.wth[n] = nodes.wth[old_node][0]
        subnodes.wth_var[n] = nodes.wth_var[old_node][0]
        subnodes.grod[n] = nodes.grod[old_node][0]
        subnodes.grod_fid[n] = nodes.grod_fid[old_node][0]
        subnodes.hfalls_fid[n] = nodes.hfalls_fid[old_node][0]
        subnodes.nchan_max[n] = nodes.nchan_max[old_node][0]
        subnodes.nchan_mod[n] = nodes.nchan_mod[old_node][0]
        subnodes.reach_id[n] = nodes.reach_id[old_node][0]
        subnodes.facc[n] = nodes.facc[old_node][0]
        subnodes.lakeflag[n] = nodes.lakeflag[old_node][0]
        subnodes.wth_coef[n] = nodes.wth_coef[old_node][0]
        subnodes.ext_dist_coef[n] = nodes.ext_dist_coef[old_node][0]
        subnodes.max_wth[n] = nodes.max_wth[old_node][0]
        subnodes.meand_len[n] = nodes.meand_len[old_node][0]
        subnodes.river_name[n] = nodes.river_name[old_node][0]
        subnodes.manual_add[n] = nodes.manual_add[old_node][0]
        subnodes.sinuosity[n] = nodes.sinuosity[old_node][0]
        subnodes.edit_flag[n] = nodes.edit_flag[old_node][0]
        subnodes.trib_flag[n] = nodes.trib_flag[old_node][0]
        subnodes.path_freq[n] = nodes.path_freq[old_node][0]
        subnodes.path_order[n] = nodes.path_order[old_node][0]
        subnodes.path_segs[n] = nodes.path_segs[old_node][0]
        subnodes.main_side[n] = nodes.main_side[old_node][0]
        subnodes.strm_order[n] = nodes.strm_order[old_node][0]
        subnodes.end_rch[n] = nodes.end_rch[old_node][0]
        subnodes.network[n] = nodes.network[old_node][0]
        subnodes.len[n] = max(np.cumsum(rdiff[pts]))
        
    sort_nodes = np.argsort(subnodes.id)
    base_val = reaches.dist_out[current] - reaches.len[current] 
    node_cs = np.cumsum(subnodes.len[sort_nodes])
    subnodes.dist_out[sort_nodes] = node_cs+base_val 

    #delete old nodes
    node_ind = np.where(nodes.reach_id == unq_rchs[r])[0]
    swd.delete_nodes(nodes, node_ind)

    #append new nodes
    swd.append_nodes(nodes, subnodes)

#write the new data.
swd.discharge_attr_nc(reaches)
swd.write_nc(centerlines, reaches, nodes, region, sword_fn)

end_all = time.time()
print('Cl Dimensions:', len(np.unique(centerlines.cl_id)), len(centerlines.cl_id))
print('Rch Dimensions:', len(np.unique(centerlines.reach_id[0,:])), len(np.unique(nodes.reach_id)), len(np.unique(reaches.id)),len(reaches.id))
print('Node Dimensions:', len(np.unique(centerlines.node_id[0,:])), len(np.unique(nodes.id)), len(nodes.id))
print('zero node lengths:', len(np.where(nodes.len == 0)[0]), ', long node lengths:', len(np.where(nodes.len > 1000)[0]))
print('min node char len:', len(str(np.min(nodes.id))))
print('max node char len:', len(str(np.max(nodes.id))))
print('min reach char len:', len(str(np.min(reaches.id))))
print('max reach char len:', len(str(np.max(reaches.id))))
print('Edit flag values:', np.unique(reaches.edit_flag))

# '''

# plt.scatter(centerlines.x[cl_r[order_ids]], centerlines.y[cl_r[order_ids]], c=centerlines.node_id[0,cl_r[order_ids]], s=5)
# plt.title('new ids')
# plt.show()

# plt.scatter(centerlines.x[cl_r[order_ids]], centerlines.y[cl_r[order_ids]], c=old_nums, s=5)
# plt.title('old ids')
# plt.show()

# '''