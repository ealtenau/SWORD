from __future__ import division
import numpy as np
import time
import netCDF4 as nc
import pandas as pd
from scipy import spatial as sp
from shapely.geometry import Point
import geopandas as gp
import glob
import os
import matplotlib.pyplot as plt

##############################################################################

def find_no_width_reaches(reaches, n_rch_down, rch_id_up):    
    flag = np.zeros(len(reaches))
    start_rch = np.array([reaches[np.where(n_rch_down == 0)[0]][0]])
    loop = 1
    while len(start_rch) > 0:
        # print(loop, start_rch)
        rch = np.where(reaches == start_rch)[0]
        #flagged reach is tagged for deletion and need to tag upstream reaches. 
        if flag[rch] == 2:
            flag[np.where(flag == 0)] = 2
            #need to do stuff to fill in above flagged reaches... 
            # up_rchs = np.unique(rch_id_up[:,rch]); up_rchs = up_rchs[up_rchs>0]
            # while len(up_rchs) > 0:
            #     fill = np.where(np.in1d(reaches, up_rchs) == True)[0]
            #     flag[fill] = 2
            #     up_rchs = np.unique(rch_id_up[:,fill]); up_rchs = up_rchs[up_rchs>0]
            #find next start reach. 
            # if min(flag) <= 0:
            #     new_idx = np.where((flag == 0) & (n_rch_down == 0))[0]
            #     if len(new_idx) > 0:
            #         start_rch = np.array([reaches[new_idx[0]]])
            #     else:# len(start_rch) == 0:
            #         new_idx2 = np.where(flag == -1)[0]
            #         if len(new_idx2) > 0:
            #             start_rch = np.array([reaches[new_idx2[0]]])
            #         else:# len(start_rch) == 0:
            #             new_idx3 = np.where(flag == 2)[0]
            #             if len(new_idx3) > 0:
            #                 start_rch = np.array([reaches[new_idx3[0]]])
            #             else:
            #                 print(loop, start_rch, 'zeros left with zero downstream')
            #                 start_rch = []
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
##############################################################################
##############################################################################

region = 'SA'
version = 'v18'

#reading in sword data.
sword_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
outdir = '/Users/ealtenau/Documents/SWORD_Dev/update_requests/'+version+'/'+region+'/'

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
### few easy edits 
# edit_flag[np.where(edit_flag == '6,6')[0]] = '6' #np.unique(edit_flag)
# width[np.where(width < 0)[0]] = 0
# sword['/reaches/edit_flag'][:] = edit_flag
# sword['/reaches/width'][:] = width
# sword.close()

flag = find_no_width_reaches(reaches, n_rch_down, rch_id_up)
if min(flag) < 1:
    print('!!! Problem with Flag Function !!!')
else:
    rch_del = reaches[np.where(flag == 2)[0]]
    rch_csv = pd.DataFrame({"reach_id": rch_del})
    rch_csv.to_csv(outdir+'width_based_deletions_coast.csv', index = False)
    print('Done. Number of deletions:', len(rch_del))


# p_1 = np.where(flag == -1)[0]
# p0 = np.where(flag == 0)[0]
# p1 = np.where(flag == 1)[0]
# p2 = np.where(flag == 2)[0]

# plt.scatter(x[p_1],y[p_1],c='black', s = 10)
# plt.scatter(x[p0],y[p0],c='grey', s = 5)
# plt.scatter(x[p1],y[p1],c='cyan', s = 5)
# plt.scatter(x[p2],y[p2],c='red', s = 5)
# plt.show()