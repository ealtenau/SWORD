# -*- coding: utf-8 -*-
"""
Checking Topology of Short Reaches
(check_short_rch_topo.py)
===================================================

This script checks for incorrect topology in 
short SWORD reaches.

Flagged reaches are saved as a csv file located at:
'/data/update_requests/'+version+'/'+region+'/'

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python path/to/check_short_rch_topo.py NA v17

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import pandas as pd
import numpy as np
import netCDF4 as nc
import argparse
import src.updates.geo_utils as geo

##################################################################

def find_character_in_array(arr, char):
    """
    Finds the indices of a character within an array of strings.

    Args:
        arr: A list of strings.
        char: The character to search for.

    Returns:
        A list of tuples, where each tuple contains:
            - The index of the string in the array.
            - The index of the character within that string.
        Returns an empty list if the character is not found.
    """
    results = []
    for i, string in enumerate(arr):
        for j, c in enumerate(string):
            if c == char:
                results.append(i)
    return results

##################################################################

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

nc_fn = main_dir+'/data/outputs/Reaches_Nodes/'+version+'/netcdf/'\
    +region.lower()+'_sword_'+version+'.nc'
outdir = main_dir+'/data/update_requests/'+version+'/'+region+'/'

#read data. 
sword = nc.Dataset(nc_fn)
edit_flag = np.array(sword['/reaches/edit_flag'][:])
reaches = np.array(sword['/reaches/reach_id'][:])
rch_id_up = np.array(sword['/reaches/rch_id_up'][:])
rch_id_dn = np.array(sword['/reaches/rch_id_dn'][:])
cl_rchs = np.array(sword['/centerlines/reach_id'][:])
cl_ids = np.array(sword['/centerlines/cl_id'][:])
x = np.array(sword['/centerlines/x'][:])
y = np.array(sword['/centerlines/y'][:])

#narrow down reaches to recent additions. 
subset = find_character_in_array(edit_flag, '7')
subset = np.array(subset) #np.unique(edit_flag[subset])

#loop through reaches and see if there are inconsistencies
#between topology and actual distance between neighboring
#reaches.
new_reaches = reaches[subset]
check = []
for r in list(range(len(new_reaches))):
    print(r, len(new_reaches)-1)
    rch = np.where(cl_rchs[0,:] == new_reaches[r])[0]
    if len(rch) <=5:
        mn_id = np.where(cl_ids[rch] == min(cl_ids[rch]))
        mx_id = np.where(cl_ids[rch] == max(cl_ids[rch]))
        mn_x = x[rch[mn_id]]
        mn_y = y[rch[mn_id]]
        mx_x = x[rch[mx_id]]
        mx_y = y[rch[mx_id]]
        dn_nghs = cl_rchs[1::,rch[mn_id]]; dn_nghs = dn_nghs[dn_nghs>0]
        up_nghs = cl_rchs[1::,rch[mx_id]]; up_nghs = up_nghs[up_nghs>0]
        #get downstream neighbor coordinate at the maximum index.
        dn_x = []
        dn_y = []
        for d in list(range(len(dn_nghs))):
            dnr = np.where(cl_rchs[0,:] == dn_nghs[d])[0]
            dnr_mx_id = np.where(cl_ids[dnr] == max(cl_ids[dnr]))[0]
            dn_x.append(x[dnr[dnr_mx_id]])
            dn_y.append(y[dnr[dnr_mx_id]])
        dn_x = np.array(dn_x)
        dn_y = np.array(dn_y)
        #get upstream neighbor coordinate at the minimum index. 
        up_x = []
        up_y = []
        for u in list(range(len(up_nghs))):
            unr = np.where(cl_rchs[0,:] == up_nghs[u])[0]
            unr_mn_id = np.where(cl_ids[unr] == min(cl_ids[unr]))[0]
            up_x.append(x[unr[unr_mn_id]])
            up_y.append(y[unr[unr_mn_id]])
        up_x = np.array(up_x)
        up_y = np.array(up_y)
        #downstream neighbor distance difference.
        x_coords1 = np.append(mn_x,dn_x)
        y_coords1 = np.append(mn_y,dn_y) 
        diff1 = geo.get_distances(x_coords1,y_coords1)
        #upstream neighbor distance difference. 
        x_coords2 = np.append(mx_x,up_x)
        y_coords2 = np.append(mx_y,up_y)
        diff2 = geo.get_distances(x_coords2,y_coords2)
        if max(diff1) > 500 or max(diff2) > 500:
            check.append(new_reaches[r])

#save flagged reaches.
rch_csv = pd.DataFrame({"reach_id": check})
rch_csv.to_csv(outdir+'short_reach_topo_check.csv', index = False)