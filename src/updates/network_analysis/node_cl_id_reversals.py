"""
Reversing Node and Centerline IDs based on topology
(node_cl_id_reversals.py).
===========================================================

This script reverses the node and centerline IDs if they 
are not ordered downstream to upstream based on topology. 

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter region 
identifier (i.e. NA), SWORD version (i.e. v17), and a 
True/False statement for whether or not to update the 
SWORD netCDF.

CSV file outputs are created and located in 
sword.paths['update_dir'].

Execution example (terminal):
    python node_cl_id_reversals.py NA v17 True

"""

import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import netCDF4 as nc
import geopandas as gp
import pandas as pd
import argparse
import sys
import time
import matplotlib.pyplot as plt
from scipy import spatial as sp
from src.updates.sword import SWORD

start_all = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("update", help="update netcdf or not (True or False)", type = str)
args = parser.parse_args()

region = args.region
version = args.version
update_nc = args.update

#read data. 
sword = SWORD(main_dir, region, version)
sword.copy() #copies original file for version control.
outpath = sword.paths['update_dir']

#copies of pre-edit centerline and node ids. 
old_cl_ids = np.copy(sword.centerlines.cl_id)
old_cl_nodes = np.copy(sword.centerlines.node_id)
old_nodes = np.copy(sword.nodes.id)
old_node_cl_ids = np.copy(sword.nodes.cl_id)
old_cl_rchs = np.copy(sword.centerlines.reach_id)

start = time.time()
print('Reversing Centerline IDs')
cl_pts = np.vstack((sword.centerlines.x, sword.centerlines.y)).T
kdt = sp.cKDTree(cl_pts)
pt_dist, pt_ind = kdt.query(cl_pts, k = 4)
cl_rev = []
for r in list(range(len(sword.reaches.id))):
    # print(r)
    cl_ind = np.where(sword.centerlines.reach_id[0,:] == sword.reaches.id[r])[0]
    min_id = min(sword.centerlines.cl_id[cl_ind])
    max_id = max(sword.centerlines.cl_id[cl_ind])
    min_ind = np.where(sword.centerlines.cl_id[cl_ind] == min_id)[0] #sword.centerlines.cl_id[cl_ind[min_ind]]
    max_ind = np.where(sword.centerlines.cl_id[cl_ind] == max_id)[0] #sword.centerlines.cl_id[cl_ind[max_ind]]
    # up_nghs = sword.centerlines.reach_id[1:,cl_ind[max_ind]] # sword.centerlines.reach_id[0:,cl_ind[max_ind]]
    # up_nghs = up_nghs[up_nghs>0]
    # dn_nghs = sword.centerlines.reach_id[1:,cl_ind[min_ind]] # sword.centerlines.reach_id[0:,cl_ind[min_ind]]
    # dn_nghs = dn_nghs[dn_nghs>0]
    up_nghs = sword.reaches.rch_id_up[:,r]
    up_nghs = up_nghs[up_nghs>0]
    dn_nghs = sword.reaches.rch_id_down[:,r]
    dn_nghs = dn_nghs[dn_nghs>0]
    #comparing to spatial query neighbors. 
    max_nghs = np.unique(sword.centerlines.reach_id[0,pt_ind[cl_ind[max_ind],:]])
    max_nghs = max_nghs[max_nghs != sword.reaches.id[r]]
    min_nghs = np.unique(sword.centerlines.reach_id[0,pt_ind[cl_ind[min_ind],:]])
    min_nghs = min_nghs[min_nghs != sword.reaches.id[r]]
    if len(min_nghs) == 0:
        min_nghs = np.array([0])
    if len(max_nghs) == 0:
        max_nghs = np.array([0])
    
    #condition for very short reaches 
    common = np.intersect1d(min_nghs,max_nghs)
    if len(common) == 1:
        if common in up_nghs:
            in_max = np.where(np.in1d(dn_nghs, max_nghs)==True)[0]
            in_min = np.where(np.in1d(dn_nghs, min_nghs)==True)[0]
            if len(in_min) == True:
                cind = np.where(min_nghs == common)[0]
                min_nghs = np.delete(min_nghs, cind)
            if len(in_max) == True:
                cind = np.where(max_nghs == common)[0]
                max_nghs = np.delete(max_nghs, cind)
        if common in dn_nghs:
            in_max = np.where(np.in1d(up_nghs, max_nghs)==True)[0]
            in_min = np.where(np.in1d(up_nghs, min_nghs)==True)[0]
            if len(in_min) == True:
                cind = np.where(min_nghs == common)[0]
                min_nghs = np.delete(min_nghs, cind)
            if len(in_max) == True:
                cind = np.where(max_nghs == common)[0]
                max_nghs = np.delete(max_nghs, cind)
        
    #conditions to determine whether or not to reverse ids. 
    if len(up_nghs) == 0 and len(dn_nghs) > 0:
        # if dn_nghs[0] in sword.reaches.rch_id_up[:,r]:
        if len(np.where(np.in1d(dn_nghs,max_nghs) == True)[0]) > 0:
            cl_rev.append(sword.reaches.id[r])
            sort_inds = np.argsort(sword.centerlines.cl_id[cl_ind])
            sword.centerlines.cl_id[cl_ind[sort_inds]] = sword.centerlines.cl_id[cl_ind[sort_inds]][::-1] #sword.centerlines.cl_id[cl_ind[min_ind]]; sword.centerlines.cl_id[cl_ind[max_ind]]
            sword.centerlines.reach_id[1:,cl_ind[min_ind]] = 0
            sword.centerlines.reach_id[1:,cl_ind[max_ind]] = 0
            dn_nghs = dn_nghs.reshape(len(dn_nghs),1)
            # sword.centerlines.reach_id[1:len(up_nghs),cl_ind[min_ind]] = up_nghs
            sword.centerlines.reach_id[1:len(dn_nghs)+1,cl_ind[max_ind]] = dn_nghs

    if len(dn_nghs) == 0 and len(up_nghs) > 0:
        # if up_nghs[0] in sword.reaches.rch_id_down[:,r]:
        if len(np.where(np.in1d(up_nghs,min_nghs)== True)[0]) > 0:
            cl_rev.append(sword.reaches.id[r])
            sort_inds = np.argsort(sword.centerlines.cl_id[cl_ind])
            sword.centerlines.cl_id[cl_ind[sort_inds]] = sword.centerlines.cl_id[cl_ind[sort_inds]][::-1] #sword.centerlines.cl_id[cl_ind[min_ind]]; sword.centerlines.cl_id[cl_ind[max_ind]]
            sword.centerlines.reach_id[1:,cl_ind[min_ind]] = 0
            sword.centerlines.reach_id[1:,cl_ind[max_ind]] = 0
            up_nghs = up_nghs.reshape(len(up_nghs),1)
            sword.centerlines.reach_id[1:len(up_nghs)+1,cl_ind[min_ind]] = up_nghs
            # sword.centerlines.reach_id[1:len(dn_nghs),cl_ind[max_ind]] = dn_nghs

    if len(dn_nghs) > 0 and len(up_nghs) > 0:
        if len(np.where(np.in1d(up_nghs,min_nghs)== True)[0]) > 0 or len(np.where(np.in1d(dn_nghs,max_nghs) == True)[0]) > 0: #use to be and 
            cl_rev.append(sword.reaches.id[r])
            sort_inds = np.argsort(sword.centerlines.cl_id[cl_ind])
            sword.centerlines.cl_id[cl_ind[sort_inds]] = sword.centerlines.cl_id[cl_ind[sort_inds]][::-1] #sword.centerlines.cl_id[cl_ind[min_ind]]; sword.centerlines.cl_id[cl_ind[max_ind]]
            sword.centerlines.reach_id[1:,cl_ind[min_ind]] = 0
            sword.centerlines.reach_id[1:,cl_ind[max_ind]] = 0
            up_nghs = up_nghs.reshape(len(up_nghs),1)
            dn_nghs = dn_nghs.reshape(len(dn_nghs),1)
            sword.centerlines.reach_id[1:len(up_nghs)+1,cl_ind[min_ind]] = up_nghs
            sword.centerlines.reach_id[1:len(dn_nghs)+1,cl_ind[max_ind]] = dn_nghs
end = time.time()
print(str(np.round((end-start)/60,2))+' mins')

start = time.time()
print('Reversing Node IDs')   
node_rev = []
order_issues = []
for r in list(range(len(sword.reaches.id))):
    #getting centerline and node dimension indexes for a reach. 
    cl_ind = np.where(sword.centerlines.reach_id[0,:] == sword.reaches.id[r])[0]
    node_ind = np.where(sword.nodes.reach_id == sword.reaches.id[r])[0]
    #finding the min and max centerline ids 
    min_id = min(sword.centerlines.cl_id[cl_ind])
    max_id = max(sword.centerlines.cl_id[cl_ind])
    min_ind = np.where(sword.centerlines.cl_id[cl_ind] == min_id)[0] 
    max_ind = np.where(sword.centerlines.cl_id[cl_ind] == max_id)[0]
    #seeing if the node id listed for each min and max is in reverse order. 
    if sword.centerlines.node_id[0,cl_ind[min_ind]] > sword.centerlines.node_id[0,cl_ind[max_ind]]:
        #finding if there is an existing index ordering issue and skipping. recorded in order_issues. 
        up_int = int(str(sword.centerlines.node_id[0,cl_ind[min_ind]])[-4:-2])
        dn_int = int(str(sword.centerlines.node_id[0,cl_ind[max_ind]])[-4:-2])
        if abs(dn_int-up_int) < len(node_ind)/2:
            order_issues.append(sword.reaches.id[r])
            continue
        #if no existing issues, reverse the node ids. 
        node_rev.append(sword.reaches.id[r])
        sort_inds2 = np.argsort(sword.nodes.id[node_ind])
        sword.nodes.id[node_ind[sort_inds2]] = sword.nodes.id[node_ind[sort_inds2]][::-1] #reversing node dimension
        sword.nodes.dist_out[node_ind[sort_inds2]] = sword.nodes.dist_out[node_ind[sort_inds2]][::-1] #reversing node dist out
        #looping through the nodes and reversing the ids in the centerline dimension. 
        #and updating the cl_ids in the the node dimension. 
        subnodes = old_nodes[node_ind]
        for n in list(range(len(subnodes))):
            nind = np.where(old_cl_nodes[0,:] == subnodes[n])[0]
            sword.centerlines.node_id[0,nind] = sword.nodes.id[node_ind[n]]
end = time.time()
print(str(np.round((end-start)/60,2))+' mins')

start = time.time()
print('Updating Node Centerline ID Ranges') 
subreaches = np.array(node_rev)  
for r in list(range(len(node_rev))):
    node_ind = np.where(sword.nodes.reach_id == subreaches[r])[0]
    subnodes = sword.nodes.id[node_ind]
    for n in list(range(len(subnodes))):
            nind = np.where(sword.centerlines.node_id[0,:] == subnodes[n])[0]
            mn = min(sword.centerlines.cl_id[nind]) 
            mx = max(sword.centerlines.cl_id[nind]) 
            sword.nodes.cl_id[0,node_ind[n]] = mn
            sword.nodes.cl_id[1,node_ind[n]] = mx
end = time.time()
print(str(np.round((end-start)/60,2))+' mins')

print('Updating Centerline Node ID Neighbors')
id_arr = sword.centerlines.node_id[0,pt_ind]
row_sum = np.sum(abs(np.diff(id_arr, axis=1)), axis = 1)
# (len(np.where(row_sum > 0)[0])/len(row_sum))*100
update = np.where(row_sum > 0)[0]
sword.centerlines.node_id[1:4,update] = id_arr[update,1:4].T

#writing csv files for reversed nodes and centerline ids.             
print('Writing CSV Files')
cl_csv = pd.DataFrame({"reach_id": cl_rev})
cl_csv.to_csv(outpath+'centerline_reversals.csv', index = False)
rev_flag = sword.reaches.edit_flag[np.where(np.in1d(sword.reaches.id, node_rev)==True)[0]]
node_csv = pd.DataFrame({"reach_id": node_rev, "edit_flag": rev_flag})
node_csv.to_csv(outpath+'node_reversals.csv', index = False)
issue_csv = pd.DataFrame({"reach_id": order_issues})
issue_csv.to_csv(outpath+'order_problems.csv', index = False)

#updating the netcdf. 
if update_nc == 'True':
    print('Updating the NetCDF')
    sword.save_nc()

end_all = time.time()
print('*** '+region + ' Done in: '+str(np.round((end_all-start_all)/60,2))+' mins ***')
