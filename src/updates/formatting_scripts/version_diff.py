# -*- coding: utf-8 -*-
"""
SWORD Version Translation (version_diff.py).
===============================================================

This script creates translation files for the Reach and Node IDs
between two SWORD versions.  

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA), SWORD version for the 
translation (i.e. v17), and SWORD version to compare to
(i.e. v16).

Translation files are written as csv files and located 
at sword.paths['version_dir'].

Execution example (terminal):
    python path/to/version_diff.py NA v17 v16

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import pandas as pd
from scipy import spatial as sp
import pandas as pd
import argparse
from src.updates.sword import SWORD

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("new_version", help="latest version", type = str)
parser.add_argument("old_version", help="older version to compare to", type = str)

args = parser.parse_args()

region = args.region
new_v = args.new_version
old_v = args.old_version

#manual
# region = 'OC'
# new_v = 'v17b'
# old_v = 'v16'

#read data and create outpaths.
new_sword = SWORD(main_dir, region, new_v)
old_sword = SWORD(main_dir, region, old_v)
rch_outpath = new_sword.paths['version_dir']+region+'_ReachIDs_'+new_v+'_vs_'+old_v+'.csv'
node_outpath = new_sword.paths['version_dir']+region+'_NodeIDs_'+new_v+'_vs_'+old_v+'.csv'

#find closest points between the two versions.     
old_pts = np.vstack((old_sword.nodes.x, old_sword.nodes.y)).T
new_pts = np.vstack((new_sword.nodes.x, new_sword.nodes.y)).T
kdt = sp.cKDTree(old_pts)
eps_dist, eps_ind = kdt.query(new_pts, k = 2) 

#translation indexes. 
indexes = eps_ind[:,0]
dist = eps_dist[:,0]
old_node_ids = old_sword.nodes.id[indexes]
old_reach_ids = old_sword.nodes.reach_id[indexes]

#Flag any nodes with a previous version node greater than 500 m away 
#or with edit_flag = '7' (i.e. new centerline channel).
new_cls = np.where(dist > 0.005)[0]
old_node_ids[new_cls] = 0
old_reach_ids[new_cls] = 0
shift_flag = np.zeros(len(old_reach_ids))
shift_flag[np.where(old_node_ids == 0)[0]] = 1
#len(np.where(shift_flag == 1)[0])/len(shift_flag)*100 #percentage check. 

#creating flags to identify if reach boundaries changed
#at the node level. 
print('calculating node dimension')
boundary_flag = np.zeros(len(old_reach_ids),dtype=int) #binary flag: 0 - no change, 1 - boundary change. 
boundary_perc = np.zeros(len(old_reach_ids),dtype=int) #percentage of overlapping reach for boundary change. 
dominant_rch = np.zeros(len(old_reach_ids),dtype=int) #dominant Reach ID for shifted reach. 
number_of_rchs = np.zeros(len(old_reach_ids),dtype=int) #number of reaches from the older version in the new version reach boundary. 
for r in list(range(len(new_sword.reaches.id))):
    pts = np.where(new_sword.nodes.reach_id == new_sword.reaches.id[r])[0]
    unique_elements, counts = np.unique(old_reach_ids[pts], return_counts=True)
    keep = np.where(unique_elements>0)[0]
    unq_elements = unique_elements[keep]
    counts = counts[keep]
    if len(unique_elements) > 1:
        boundary_flag[pts] = 1
        number_of_rchs[pts] = len(unq_elements)
        max_cnt = np.where(counts == max(counts))[0]
        if len(max_cnt) > 1:
            max_cnt = max_cnt[0]
        boundary_perc[pts] = counts[max_cnt]/len(pts)*100
        dominant_rch[pts] = unq_elements[max_cnt]

#creating flags to identify if reach boundaries changed
#at the reach level. 
print('calculating reach dimension')
rch_bnd_flag = np.zeros(len(new_sword.reaches.id),dtype=int) #binary flag: 0 - no change, 1 - boundary change.
rch_bnd_perc = np.zeros(len(new_sword.reaches.id),dtype=int) #percentage of overlapping reach for boundary change. 
rch_dom = np.zeros(len(new_sword.reaches.id),dtype=int) #dominant Reach ID for shifted reach. 
num_rch = np.zeros(len(new_sword.reaches.id),dtype=int) #number of reaches from the older version in the new version reach boundary. 
old_rch_id = np.zeros(len(new_sword.reaches.id),dtype=int) #Reach ID of older version. 
for ind in list(range(len(new_sword.reaches.id))):
    # print(ind)
    pts2 = np.where(new_sword.nodes.reach_id == new_sword.reaches.id[ind])[0]
    rch_bnd_flag[ind] = np.unique(boundary_flag[pts2])[0]
    rch_bnd_perc[ind] = np.unique(boundary_perc[pts2])[0]
    rch_dom[ind] = np.unique(dominant_rch[pts2])[0]
    num_rch[ind] = np.unique(number_of_rchs[pts2])[0]
    if np.unique(boundary_flag[pts2]) == 1:
        old_rch_id[ind] = np.unique(dominant_rch[pts2])[0]
    else:
        old_rch_id[ind] = np.unique(old_reach_ids[pts2])[0]
    
# output reach differences in csv format. 
data = pd.DataFrame(np.array([new_sword.reaches.x, 
                              new_sword.reaches.y, 
                              new_sword.reaches.id, 
                              old_rch_id, 
                              rch_bnd_flag, 
                              rch_bnd_perc, 
                              rch_dom, 
                              num_rch])).T

data.columns = ['lon', 
                'lat', 
                new_v+'_reach_id', 
                old_v+'_reach_id', 
                'boundary_flag', 
                'boundary_percent', 
                'dominant reach', 
                'v16 number of reaches']
for i in data.columns[2::]:
    try:
        data[[i]] = data[[i]].astype(float).astype(int)
    except:
        pass
data.to_csv(rch_outpath,index=False)

# output node differences in csv format.
data2 = pd.DataFrame(np.array([new_sword.nodes.x, 
                               new_sword.nodes.y, 
                               new_sword.nodes.id, 
                               new_sword.nodes.reach_id, 
                               old_node_ids, 
                               old_reach_ids, 
                               shift_flag, 
                               boundary_flag, 
                               boundary_perc, 
                               dominant_rch, 
                               number_of_rchs])).T

data2.columns = ['lon', 
                 'lat', 
                 new_v+'_node_id', 
                 new_v+'_reach_id', 
                 old_v+'_node_id', 
                 old_v+'_reach_id', 
                 'shift_flag', 
                 'boundary_flag', 
                 'boundary_percent', 
                 'dominant_reach', 
                 old_v+'_num_reaches']
for j in data2.columns[2::]:
    try:
        data2[[j]] = data2[[j]].astype(float).astype(int)
    except:
        pass
data2.to_csv(node_outpath, index=False)

print('DONE')
print('percent of nodes shifted:',np.round(len(np.where(shift_flag == 1)[0])/len(shift_flag)*100,2))
print('percent of reaches with different boundaries:',np.round(len(np.where(rch_bnd_flag == 1)[0])/len(rch_bnd_flag)*100,2))

