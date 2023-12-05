from __future__ import division
import os
if os.path.exists('/Users/ealteanau/Documents/SWORD_Dev/src/SWORD/post_swot_launch_updates/mhv_sword/'):
    os.chdir('/Users/ealteanau/Documents/SWORD_Dev/src/SWORD/post_swot_launch_updates/mhv_sword/')
else:
    os.chdir('/afs/cas.unc.edu/users/e/a/ealtenau/SWORD/post_swot_launch_updates/mhv_sword/')
import mhv_reach_def_tools as rdt
# import Write_Database_Files as wf
import time
import numpy as np
import geopandas as gp
from shapely.geometry import Point
import pandas as pd
import argparse
from scipy import spatial as sp

###############################################################################
##################### Defining Reach and Node Locations #######################
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Region", type = str)
parser.add_argument("local_processing", help="'True' for local machine, 'False' for server", type = str)
args = parser.parse_args()

start_all = time.time()
region = args.region

# Input file(s).
if args.local_processing == 'True':
    main_dir = '/Users/ealteanau/Documents/SWORD_Dev/inputs/'
else:
    main_dir = '/afs/cas.unc.edu/depts/geological_sciences/pavelsky/students/ealtenau/SWORD_dev/inputs/'
nc_file = main_dir+'MHV_SWORD/'+region+'_mhv_sword.nc'

# Reading in data.
data = rdt.read_merge_netcdf(nc_file)

# Making sure flow accumulation minimum isn't zero.
data.facc[np.where(data.facc == 0)[0]] = 0.001
# Cutting basins to 6 digits.
data.basins = np.array([int(str(ind)[0:6]) for ind in data.basins])

# Creating empty objects to fill with attributes.
centerlines = rdt.Object()

# Loop through each level 2 basin. Subsetting per level 2 basin speeds up the script.
level2_basins = np.array([int(str(ind)[0:2]) for ind in data.basins])
uniq_level2 = np.unique(level2_basins)
uniq_level2 = np.delete(uniq_level2, 0)
cnt = 0
start_id = 0
for ind in list(range(0,1)):#len(uniq_level2))):

    print('STARTING BASIN: ' + str(uniq_level2[ind]))

    start = time.time()

    # Define objects to assign attributes.
    subcls = rdt.Object()

    # Subset data.
    level2 = np.where(level2_basins == uniq_level2[ind])[0]
    subcls.id = data.id[level2]
    subcls.lon = data.lon[level2]
    subcls.lat = data.lat[level2]
    subcls.seg = data.seg[level2]
    subcls.ind = data.ind[level2]
    subcls.dist = data.segDist[level2]
    subcls.wth = data.wth[level2]
    subcls.elv = data.elv[level2]
    subcls.facc = data.facc[level2]
    subcls.lake = data.lake[level2]
    subcls.delta = data.delta[level2]
    subcls.nchan = data.nchan[level2]
    subcls.grand = data.grand[level2]
    subcls.grod = data.grod[level2]
    subcls.grod_fid = data.grod_fid[level2]
    subcls.hfalls_fid = data.hfalls_fid[level2]
    subcls.basins = data.basins[level2]
    subcls.manual = data.manual[level2]
    subcls.num_obs = data.num_obs[level2]
    subcls.orbits = data.orbits[level2]
    subcls.lake_id = data.lake_id[level2]
    subcls.strorder = data.strorder[level2]
    subcls.eps = data.eps[level2]
    subcls.lon[np.where(subcls.lon < -180)] = -180.0
    subcls.lon[np.where(subcls.lon > 180)] = 180.0
    subcls.x = np.copy(subcls.lon)
    subcls.y = np.copy(subcls.lat)

    all_pts = np.vstack((subcls.lon, subcls.lat)).T
    kdt = sp.cKDTree(all_pts)
    pt_dist, pt_ind = kdt.query(all_pts, k = 6)
    pt_dist = pt_dist[:,1::]
    pt_ind = pt_ind[:,1::]

    new_segs = np.zeros(len(subcls.seg))
    start_seg = np.unique(subcls.seg[np.where(subcls.facc == np.max(subcls.facc))[0]])[0]
    new_segs[np.where(subcls.seg == start_seg)] = 1
    loop = 1
    cnt = 1
    while np.min(new_segs) == 0:
        print(loop, start_seg)
        seg = np.where(subcls.seg == start_seg)[0]
        seg_eps = np.where(subcls.eps[seg] > 0)[0]
        new_segs[seg] = cnt

        ngh1_segs = np.unique(subcls.seg[pt_ind[seg[seg_eps[0]],:]])
        ngh2_segs = np.unique(subcls.seg[pt_ind[seg[seg_eps[1]],:]])
        ngh1_segs =np.delete(ngh1_segs, np.where(ngh1_segs == start_seg)[0])
        ngh2_segs =np.delete(ngh2_segs, np.where(ngh2_segs == start_seg)[0])

        if len(ngh1_segs) == 1 and len(ngh2_segs) == 0:
            if np.max(new_segs[np.where(subcls.seg == ngh1_segs)]) == 0:
                # new_segs[np.where(subcls.seg == ngh1_segs)] = cnt
                start_seg = ngh1_segs[0]
                loop = loop+1
            else:
                zeros = np.where(new_segs == 0)[0]
                start_seg = np.unique(subcls.seg[zeros[np.where(subcls.facc[zeros] == np.max(subcls.facc[zeros]))[0]]])[0]
                cnt = cnt+1
                loop = loop+1

        elif len(ngh1_segs) == 0 and len(ngh2_segs) == 1:
            if np.max(new_segs[np.where(subcls.seg == ngh2_segs)]) == 0:
                # new_segs[np.where(subcls.seg == ngh2_segs)] = cnt
                start_seg = ngh2_segs[0]
                loop = loop+1
            else:
                zeros = np.where(new_segs == 0)[0]
                start_seg = np.unique(subcls.seg[zeros[np.where(subcls.facc[zeros] == np.max(subcls.facc[zeros]))[0]]])[0]
                cnt = cnt+1
                loop = loop+1

        elif len(ngh1_segs) == 1 and len(ngh2_segs) == 1:
            if np.max(new_segs[np.where(subcls.seg == ngh1_segs)]) == 0 and np.max(new_segs[np.where(subcls.seg == ngh2_segs)]) == 0:
                # new_segs[np.where(subcls.seg == ngh1_segs)] = cnt
                # new_segs[np.where(subcls.seg == ngh2_segs)] = cnt                
                if np.max(subcls.facc[np.where(subcls.seg == ngh1_segs)])  > np.max(subcls.facc[np.where(subcls.seg == ngh2_segs)]):
                    start_seg = ngh2_segs[0]
                else:
                    start_seg = ngh1_segs[0]
                loop = loop+1
            elif np.max(new_segs[np.where(subcls.seg == ngh1_segs)]) > 0 and np.max(new_segs[np.where(subcls.seg == ngh2_segs)]) == 0:
                # new_segs[np.where(subcls.seg == ngh2_segs)] = cnt 
                start_seg = ngh2_segs[0]
                loop = loop+1
            elif np.max(new_segs[np.where(subcls.seg == ngh1_segs)]) == 0 and np.max(new_segs[np.where(subcls.seg == ngh2_segs)]) > 0:
                # new_segs[np.where(subcls.seg == ngh1_segs)] = cnt 
                start_seg = ngh1_segs[0]
                loop = loop+1
            else:
                zeros = np.where(new_segs == 0)[0]
                start_seg = np.unique(subcls.seg[zeros[np.where(subcls.facc[zeros] == np.max(subcls.facc[zeros]))[0]]])[0]
                cnt = cnt+1
                loop = loop+1
        
        elif len(ngh1_segs) > 1 and len(ngh2_segs) == 1:
            if np.max(new_segs[np.where(subcls.seg == ngh2_segs)]) == 0:
                new_segs[np.where(subcls.seg == ngh2_segs)] = cnt
            if np.max(new_segs[np.where(subcls.seg == ngh1_segs[0])]) == 0:
                start_seg = ngh1_segs[0]
                cnt = cnt+1
                # new_segs[np.where(subcls.seg == start_seg)] = cnt
                loop = loop+1
            else:
                zeros = np.where(new_segs == 0)[0]
                start_seg = np.unique(subcls.seg[zeros[np.where(subcls.facc[zeros] == np.max(subcls.facc[zeros]))[0]]])[0]
                cnt = cnt+1
                loop = loop+1

        elif len(ngh1_segs) == 1 and len(ngh2_segs) > 1:
            if np.max(new_segs[np.where(subcls.seg == ngh1_segs)]) == 0:
                new_segs[np.where(subcls.seg == ngh1_segs)] = cnt
            if np.max(new_segs[np.where(subcls.seg == ngh2_segs[0])]) == 0:
                start_seg = ngh2_segs[0]
                cnt = cnt+1
                # new_segs[np.where(subcls.seg == start_seg)] = cnt
                loop = loop+1
            else:
                zeros = np.where(new_segs == 0)[0]
                start_seg = np.unique(subcls.seg[zeros[np.where(subcls.facc[zeros] == np.max(subcls.facc[zeros]))[0]]])[0]
                cnt = cnt+1
                loop = loop+1

        elif len(ngh1_segs) > 1 and len(ngh2_segs) == 0:
            if np.max(new_segs[np.where(subcls.seg == ngh1_segs[0])]) == 0:
                start_seg = ngh1_segs[0]
                cnt = cnt+1
                # new_segs[np.where(subcls.seg == start_seg)] = cnt
                loop = loop+1
            else:
                zeros = np.where(new_segs == 0)[0]
                start_seg = np.unique(subcls.seg[zeros[np.where(subcls.facc[zeros] == np.max(subcls.facc[zeros]))[0]]])[0]
                cnt = cnt+1
                loop = loop+1

        elif len(ngh1_segs) == 0 and len(ngh2_segs) > 1:
            if np.max(new_segs[np.where(subcls.seg == ngh2_segs[0])]) == 0:
                start_seg = ngh2_segs[0]
                cnt = cnt+1
                # new_segs[np.where(subcls.seg == start_seg)] = cnt
                loop = loop+1
            else:
                zeros = np.where(new_segs == 0)[0]
                start_seg = np.unique(subcls.seg[zeros[np.where(subcls.facc[zeros] == np.max(subcls.facc[zeros]))[0]]])[0]
                cnt = cnt+1
                loop = loop+1

        elif len(ngh1_segs) > 1 and len(ngh2_segs) > 1:
            if np.max(new_segs[np.where(subcls.seg == ngh1_segs[0])]) == 0:
                start_seg = ngh1_segs[0]
                cnt = cnt+1
                # new_segs[np.where(subcls.seg == start_seg)] = cnt
                loop = loop+1
            elif np.max(new_segs[np.where(subcls.seg == ngh2_segs[0])]) == 0:
                start_seg = ngh2_segs[0]
                cnt = cnt+1
                # new_segs[np.where(subcls.seg == start_seg)] = cnt
                loop = loop+1
            else:
                zeros = np.where(new_segs == 0)[0]
                start_seg = np.unique(subcls.seg[zeros[np.where(subcls.facc[zeros] == np.max(subcls.facc[zeros]))[0]]])[0]
                cnt = cnt+1
                loop = loop+1

        else: #if len(ngh1_segs) == 0 and len(ngh2_segs) == 0:
            zeros = np.where(new_segs == 0)[0]
            start_seg = np.unique(subcls.seg[zeros[np.where(subcls.facc[zeros] == np.max(subcls.facc[zeros]))[0]]])[0]
            cnt = cnt+1
            loop = loop+1

        if loop > len(np.unique(subcls.seg))+10:
            print('LOOP STUCK!!!')

        
            





