"""
Shifting SWORD reaches using MERIT Hydro Vector
(channel_shifting_mhv.py)
=========================================================

This script shifts pre-identified SWORD reaches based on 
proximity to a MERIT Hydro Vector-SWORD translation database 
(built using scripts in 'src/updates/mhv_sword/').

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA), SWORD version (i.e. v17), 
and whether or not to save plots (True or False).

Execution example (terminal):
    python path/to/channel_shifting_mhv.py NA v17 True

"""

import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import itertools
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import glob 
from scipy import spatial as sp
import time
import pandas as pd
import argparse
import src.updates.geo_utils as geo 
from src.updates.sword import SWORD

start_all = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Region", type = str)
parser.add_argument("version", help="<Required> Version", type = str)
parser.add_argument("save_plots", help="<Required> Whether or not to save plots (True or False)", type = str)
args = parser.parse_args()

region = args.region
version = args.version
save_plots = args.save_plots

#mhv-sword files. 
mhv_nc_dir = main_dir+'/data/inputs/MHV_SWORD/netcdf/' + region +'/'
mhv_nc_files = np.sort(glob.glob(os.path.join(mhv_nc_dir, '*.nc')))

### read sword data. 
sword = SWORD(main_dir, region, version)
sword.copy() #make a copy for version control. 
old_x = np.copy(sword.centerlines.x)
old_y = np.copy(sword.centerlines.y)

#getting identified reach ID files for shifting.
sword.paths['update_dir']
rch_dir = sword.paths['update_dir']+'shifting/'
rch_files = np.sort(glob.glob(os.path.join(rch_dir, '*.csv')))
plotdir = rch_dir+'plots/'
if os.path.exists(plotdir) == False:
    os.makedirs(plotdir)

#getting associated mhv-sword and reach shifting files. 
#for each Pfafstetter level 2 basin. 
rch_file_l2 = np.array([int(rf[-17:-15]) for rf in rch_files]) #file name structure: hb71_rch_shifts.csv
mhv_file_l2 = np.array([int(mf[-13:-11]) for mf in mhv_nc_files]) #file name structure: mhv_sword_hb71_pts_v18.nc
check_rch = []
for f in list(range(len(rch_files))):
    start = time.time()
    
    print('Starting Basin:', rch_file_l2[f])
    #reading in csv file of identified reaches to shift. 
    shift_df = pd.read_csv(rch_files[f])
    shift_rchs = np.array(shift_df['reach_id'])

    #reading in the mhv-sword translation database. 
    mhv_read = np.where(mhv_file_l2 == rch_file_l2[f])[0]
    mhv = nc.Dataset(mhv_nc_files[mhv_read[0]])
    mhv_x = np.array(mhv['/centerlines/x'][:])
    mhv_y = np.array(mhv['/centerlines/y'][:])
    mhv_flag = np.array(mhv['/centerlines/swordflag_filt'][:])
    keep = np.where(mhv_flag > 0)[0]
    mhv_x = mhv_x[keep]
    mhv_y = mhv_y[keep]
    pts = np.array([(mhv_x[i], mhv_y[i]) for i in range(len(mhv_x))])

    #setting up permutations of coordinates to test. 
    sx = np.arange(-0.01, 0.01, 0.0003)
    sy = np.arange(-0.01, 0.01, 0.0003)
    shift_coords = list(itertools.product(sx, sy)) #4489 permutations

    #loop through and find ideal shifting x-y for each reach. 
    for r in list(range(len(shift_rchs))):
        print(r, len(shift_rchs)-1)
        rch = np.where(sword.centerlines.reach_id[0,:] == shift_rchs[r])[0]
        #find mhv points within 2 km bounding box of sword reach. 
        mn_x = np.min(sword.centerlines.x[rch])-0.02
        mx_x = np.max(sword.centerlines.x[rch])+0.02
        mn_y = np.min(sword.centerlines.y[rch])-0.02
        mx_y = np.max(sword.centerlines.y[rch])+0.02
        ll = np.array([mn_x, mn_y])  # lower-left
        ur = np.array([mx_x, mx_y])  # upper-right

        #sliping mhv to sword reach extent. 
        idx = np.all(np.logical_and(ll <= pts, pts <= ur), axis=1)
        mhv_x_clip = mhv_x[idx]
        mhv_y_clip = mhv_y[idx]
        if len(mhv_x_clip) == 0:
            check_rch.append(shift_rchs[r])
            print('No MHV points within 2 km of reach:', shift_rchs[r])
            continue 
        else:
            mhv_pts = np.vstack((mhv_x_clip, mhv_y_clip)).T

        #find ideal shifting x-y parameters. 
        offset = np.zeros(len(shift_coords))
        for ind in list(range(len(shift_coords))):
            nx = sword.centerlines.x[rch]+shift_coords[ind][0]
            ny = sword.centerlines.y[rch]+shift_coords[ind][1]
            #spatial query between mhv and shifted x-y. 
            shift_pts = np.vstack((nx, ny)).T
            kdt = sp.cKDTree(mhv_pts)
            pt_dist, pt_ind = kdt.query(shift_pts, k = 1)
            #median difference between spatial query coords. 
            x_diff = np.abs(nx-mhv_x_clip[pt_ind])
            y_diff = np.abs(ny-mhv_y_clip[pt_ind])
            x_diff_med = np.median(x_diff)
            y_diff_med = np.median(y_diff)
            add = abs(x_diff_med)+abs(y_diff_med) 
            offset[ind] = add
        
        #use minimum offest difference as ideal shifting x-y parameter.     
        min_ind = np.where(offset == min(offset))[0]
        new_x = sword.centerlines.x[rch]+shift_coords[min_ind[0]][0]
        new_y = sword.centerlines.y[rch]+shift_coords[min_ind[0]][1]

        #update sword attributes with new shifted coordinates. 
        #centerline dimension updates.
        sword.centerlines.x[rch] = new_x
        sword.centerlines.y[rch] = new_y 
        #reach dimension updates.
        rind = np.where(sword.reaches.id == shift_rchs[r])[0]
        sword.reaches.x[rind] = np.median(new_x)
        sword.reaches.y[rind] = np.median(new_y)
        sword.reaches.x_min[rind] = np.min(new_x)
        sword.reaches.x_max[rind] = np.max(new_x)
        sword.reaches.y_min[rind] = np.min(new_y)
        sword.reaches.y_max[rind] = np.max(new_y)
        #re-calculating reach length. 
        order_ids = np.argsort(sword.centerlines.cl_id[rch])
        x_coords = sword.centerlines.x[rch[order_ids]]
        y_coords = sword.centerlines.y[rch[order_ids]]
        diff = geo.get_distances(x_coords,y_coords)
        dist = np.cumsum(diff)
        sword.reaches.len[rind] = np.max(dist)
        #node dimension updates.
        unq_nodes = np.unique(sword.centerlines.node_id[0,rch])
        node_len = np.zeros(len(unq_nodes))
        node_x = np.zeros(len(unq_nodes))
        node_y = np.zeros(len(unq_nodes))
        for n in list(range(len(unq_nodes))):
            nind = np.where(sword.nodes.id == unq_nodes[n])[0]
            npts = np.where(sword.centerlines.node_id[0,rch[order_ids]] == unq_nodes[n])[0]
            sword.nodes.x[nind] = np.median(sword.centerlines.x[rch[order_ids[npts]]])
            sword.nodes.y[nind] = np.median(sword.centerlines.y[rch[order_ids[npts]]])
            sword.nodes.len[nind] = max(np.cumsum(diff[npts]))
            if len(npts) == 1:
                sword.nodes.len[nind] = 30
        #updaging distance from outlet. 
        base_val = sword.reaches.dist_out[rind] - sword.reaches.len[rind]
        sword.reaches.dist_out[rind] = sword.reaches.len[rind]+base_val
        nr = np.where(sword.nodes.reach_id == shift_rchs[r])
        sword.nodes.dist_out[nr] = np.cumsum(sword.nodes.len[nr])+base_val

        #output figures for testing/checking.
        if save_plots == 'True':
            plt.scatter(old_x[rch], old_y[rch], c = 'red', s=3)
            plt.scatter(sword.centerlines.x[rch], sword.centerlines.y[rch], c = 'cyan', s=3)
            plt.scatter(mhv_x_clip, mhv_y_clip, c = 'black', s=1)
            plt.title('Reach: '+str(shift_rchs[r]))
            plt.xlabel('lon')
            plt.ylabel('lat')
            plt.xlim(mn_x+0.001, mx_x+0.001)
            plt.ylim(mn_y+0.001, mx_y+0.001)
            # plt.show()
            plt.savefig(plotdir+'rch_'+str(shift_rchs[r]))
            plt.close()
    
    end = time.time()
    print('Finished Basin', rch_file_l2[f], 'in', str(np.round((end-start)/60, 2)) + ' min')

print('Writing New NetCDF')
sword.save_nc()

#saving mismatch csv files. These reaches don't have a mhv centerline within 2 km. 
issue_csv = {'reach_id': np.array(check_rch).astype('int64')}
issue_csv = pd.DataFrame(issue_csv)
issue_csv.to_csv(rch_dir+region.lower()+'_mhv_mismatches.csv', index=False)

end_all = time.time()
print('Finished', region, 'in', str(np.round((end_all-start_all)/60, 2)) + ' min')
