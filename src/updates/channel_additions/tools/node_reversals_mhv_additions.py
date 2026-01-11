"""
Reversing Node IDs based on topology
(node_reversals_mhv_additions.py).
===========================================================

This script reverses the node IDs if they 
are not ordered downstream to upstream based on topology.
It is ideally run after adding the MHV rivers to SWORD. 

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter region 
identifier (i.e. NA), SWORD version (i.e. v17), and a 
True/False statement for whether or not to update the 
SWORD netCDF. Choosing False will export the csv file 
of identified reaches with nodes to reverse without 
updating the master file. 

CSV file outputs are created and located in 
'/data/update_requests/'+version+'/'+region+'/'.

Execution example (terminal):
    python path/to/node_reversals_mhv_additions.py NA v17 True

"""

import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import pandas as pd
import argparse
import sys
import time
from scipy import spatial as sp
from src.updates.sword_duckdb import SWORD

start_all = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("update", help="update netcdf or not (True or False)", type = str)
args = parser.parse_args()

region = args.region
version = args.version
update_nc = args.update

nc_fn = main_dir+'/data/outputs/Reaches_Nodes/'\
    +version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
outpath = main_dir+'/data/update_requests/'+version+'/'+region+'/'

#read sword
db_path = os.path.join(main_dir, f'data/duckdb/sword_{version}.duckdb')
sword = SWORD(db_path, region, version)
if update_nc == 'True':
     sword.copy() #copy file for version control. 

#copies of pre-edit centerline and node ids. 
old_cl_ids = np.copy(sword.centerlines.cl_id)
old_cl_nodes = np.copy(sword.centerlines.node_id)
old_nodes = np.copy(sword.nodes.id)
old_node_cl_ids = np.copy(sword.nodes.cl_id)
old_cl_rchs = np.copy(sword.centerlines.reach_id)

start = time.time()
cl_pts = np.vstack((sword.centerlines.x, sword.centerlines.y)).T
kdt = sp.cKDTree(cl_pts)
pt_dist, pt_ind = kdt.query(cl_pts, k = 4)
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
rev_flag = sword.reaches.edit_flag[np.where(np.in1d(sword.reaches.id, node_rev)==True)[0]]
node_csv = pd.DataFrame({"reach_id": node_rev, "edit_flag": rev_flag})
node_csv.to_csv(outpath+'node_reversals.csv', index = False)
issue_csv = pd.DataFrame({"reach_id": order_issues})
issue_csv.to_csv(outpath+'order_problems.csv', index = False)

#updating the netcdf. 
if update_nc == 'True':
    print('Saving the NetCDF')
    sword.save_nc()

sword.close()
end_all = time.time()
print('*** '+region + ' Done in: '+str(np.round((end_all-start_all)/60,2))+' mins ***')
