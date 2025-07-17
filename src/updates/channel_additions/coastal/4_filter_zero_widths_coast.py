"""
Adding Coastal MHV Additions to SWORD 
(4_filter_zero_widths_coast.py)
===================================================

This script flags MHV coastal addition reaches in sword
that have zero width values for deletion.

Output is a csv file of reach IDs to be deleted 
located at:
'/data/update_requests/'+version+'/'+region+'/'

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python path/to/4_filter_zero_widths_coast.py NA v17

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import netCDF4 as nc
import pandas as pd
import argparse

##############################################################################

def find_no_width_reaches(reaches, n_rch_down, rch_id_up):    
    """
    Identifies added MHV reaches with zero width values 
    and all associated upstream reaches. 

    Parameters
    ----------
    reaches: numpy.array()
        SWORD reach ID. 
    n_rch_down: numpy.array()
        Number of downstream reaches. 
    rch_id_up: numpy.array()
        Upstream reach IDs. 

    Returns
    -------
    flag: numpy.array()
        Flag indicating if a reach should be deleted
        based on widths. 

    """
    
    flag = np.zeros(len(reaches))
    start_rch = np.array([reaches[np.where(n_rch_down == 0)[0]][0]])
    loop = 1
    while len(start_rch) > 0:
        # print(loop, start_rch)
        rch = np.where(reaches == start_rch)[0]
        #flagged reach is tagged for deletion and need to tag upstream reaches. 
        if flag[rch] == 2:
            flag[np.where(flag == 0)] = 2
            if min(flag) <= 0:
                print(loop, start_rch, 'zeros left with zero downstream')
                start_rch = []
            else:
                start_rch = []
            loop = loop+1

        #flagged reach is 0 or 1. 
        else:
            if flag[rch] <= 0:
                flag[rch] = 1
            #find upstream reaches.
            up_rchs = np.unique(rch_id_up[:,rch]); up_rchs = up_rchs[up_rchs>0]
            if len(up_rchs) > 0:
                up_wth = np.array([width[np.where(reaches == r)] for r in up_rchs])
                up_edit = np.array([edit_flag[np.where(reaches == r)] for r in up_rchs])
                rmv = np.where((up_wth <= 0) & (up_edit == '7'))[0]
                #upstream reaches to remove.
                if len(rmv) > 0:
                    rmv_idx = np.where(np.in1d(reaches, up_rchs[rmv]) == True)[0]
                    flag[rmv_idx] = 2
                    #find next start reach. 
                    up_flag = np.array([flag[np.where(reaches == r)] for r in up_rchs])
                    up_idx = np.where(up_flag == 0)[0]
                    if len(up_idx) > 0:
                        if len(up_idx) > 1:
                            start_rch = np.array([up_rchs[up_idx[0]]])
                            flag[np.where(np.in1d(reaches, up_rchs[1::]) == True)[0]] = -1
                        else:
                            start_rch = np.array([up_rchs[up_idx[0]]])
                    else: #len(start_rch) == 0:
                        if min(flag) <= 0:
                            new_idx = np.where((flag == 0) & (n_rch_down == 0))[0]
                            if len(new_idx) > 0:
                                start_rch = np.array([reaches[new_idx[0]]])
                            else:# len(start_rch) == 0:
                                new_idx2 = np.where(flag == -1)[0]
                                if len(new_idx2) > 0:
                                    start_rch = np.array([reaches[new_idx2[0]]])
                                else:# len(start_rch) == 0:
                                    new_idx3 = np.where(flag == 2)[0]
                                    if len(new_idx3) > 0:
                                        start_rch = np.array([reaches[new_idx3[0]]])
                                    else:
                                        print('zeros left with zero downstream')
                                        start_rch = []
                        else:
                            start_rch = []
                    loop = loop+1
                #no upstream reaches to remove.
                else:
                    #find next start reach.
                    up_flag = np.array([flag[np.where(reaches == r)] for r in up_rchs])
                    up_idx = np.where(up_flag == 0)[0]
                    if len(up_idx) > 0:
                        if len(up_idx) > 1:
                            start_rch = np.array([up_rchs[up_idx[0]]])
                            flag[np.where(np.in1d(reaches, up_rchs[1::]) == True)[0]] = -1
                        else:
                            start_rch = np.array([up_rchs[up_idx[0]]])
                    else: #len(start_rch) == 0:
                        if min(flag) <= 0:
                            new_idx = np.where((flag == 0) & (n_rch_down == 0))[0]
                            if len(new_idx) > 0:
                                start_rch = np.array([reaches[new_idx[0]]])
                            else:# len(start_rch) == 0:
                                new_idx2 = np.where(flag == -1)[0]
                                if len(new_idx2) > 0:
                                    start_rch = np.array([reaches[new_idx2[0]]])
                                else:# len(start_rch) == 0:
                                    new_idx3 = np.where(flag == 2)[0]
                                    if len(new_idx3) > 0:
                                        start_rch = np.array([reaches[new_idx3[0]]])
                                    else:
                                        print('zeros left with zero downstream')
                                        start_rch = []
                        else:
                            start_rch = []
                    loop = loop+1
            
            #no upstream reaches for current reach. 
            else:
                #find next start reach.
                if min(flag) <= 0:
                    new_idx = np.where((flag == 0) & (n_rch_down == 0))[0]
                    if len(new_idx) > 0:
                        start_rch = np.array([reaches[new_idx[0]]])
                    else:# len(start_rch) == 0:
                        new_idx2 = np.where(flag == -1)[0]
                        if len(new_idx2) > 0:
                            start_rch = np.array([reaches[new_idx2[0]]])
                        else:# len(start_rch) == 0:
                            new_idx3 = np.where(flag == 2)[0]
                            if len(new_idx3) > 0:
                                start_rch = np.array([reaches[new_idx3[0]]])
                            else:
                                print('zeros left with zero downstream')
                                start_rch = []
                else:
                    start_rch = []
                loop = loop+1 

        if loop > len(reaches) + 500:
            print('LOOP STUCK')
            break

    return flag

##############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

#reading in sword data.
sword_fn = main_dir+'/data/outputs/Reaches_Nodes/'+version+'/netcdf/'\
    +region.lower()+'_sword_'+version+'.nc'
outdir = main_dir+'/data/update_requests/'+version+'/'+region+'/'

sword = nc.Dataset(sword_fn)
reaches = np.array(sword['/reaches/reach_id'][:])
n_rch_up = np.array(sword['/reaches/n_rch_up'][:])
n_rch_down = np.array(sword['/reaches/n_rch_down'][:])
rch_id_up = np.array(sword['/reaches/rch_id_up'][:])
rch_id_down = np.array(sword['/reaches/rch_id_dn'][:])
width = np.array(sword['/reaches/width'][:])
edit_flag = np.array(sword['/reaches/edit_flag'][:])
x = np.array(sword['/reaches/x'][:])
y = np.array(sword['/reaches/y'][:])

flag = find_no_width_reaches(reaches, n_rch_down, rch_id_up)
if min(flag) < 1:
    print('!!! Problem with Flag Function !!!')
else:
    rch_del = reaches[np.where(flag == 2)[0]]
    rch_csv = pd.DataFrame({"reach_id": rch_del})
    rch_csv.to_csv(outdir+'width_based_deletions_coast.csv', index = False)
    print('Done. Number of deletions:', len(rch_del))