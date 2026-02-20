"""
Flagging Offset SWORD reaches using MERIT Hydro Vector
(channel_shifting_flag_mhv.py)
=========================================================

This script uses SWORD's proximity to a MERIT Hydro Vector-
SWORD translation database (built using scripts in 
'src/updates/mhv_sword/') to flag SWORD reaches that have 
a consistent offset from the river center location. It is 
limited to single channel rivers. 

Outputs are geopackage files of SWORD centerline points
with a shifting flag attribute for each Pfafstetter level 2
basin. 

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python path/to/channel_shifting_flag_mhv.py NA v17

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
from scipy import spatial as sp
import geopandas as gp
import time
import netCDF4 as nc
import argparse
import glob
import src.updates.geo_utils as geo 
from src.updates.sword_duckdb import SWORD

start_all = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Region", type = str)
parser.add_argument("version", help="<Required> Version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

#getting mhv files.
mhv_dir = main_dir+'/data/inputs/MHV_SWORD/netcdf/' + region +'/'
mhv_files = glob.glob(os.path.join(mhv_dir, '*.nc'))

#reading sword data
db_path = os.path.join(main_dir, f'data/duckdb/sword_{version}.duckdb')
sword = SWORD(db_path, region, version)
gpkg_fn = sword.paths['gpkg_dir']+sword.paths['gpkg_rch_fn']
gpkg = gp.read_file(gpkg_fn)
#creating outdir. 
outdir = main_dir+'/data/shift_testing/'
if os.path.exists(outdir) == False:
    os.makedirs(outdir)

#creating reach type and shift attribute arrays. 
rchs_type_all = np.array([int(str(ind)[-1]) for ind in sword.reaches.id])
shift_flag_all = np.zeros(len(sword.reaches.id))

#filtering data by Pfafstetter level 2 basin. 
sword_l2 = np.array([int(str(ind)[0:2]) for ind in sword.centerlines.reach_id[0,:]])
rch_l2 = np.array([int(str(ind)[0:2]) for ind in sword.reaches.id])
mhv_l2 = np.array([int(ind[-13:-11]) for ind in mhv_files])
uniq_level2 = np.unique(sword_l2)
#loop through level 2 basins and flag offset reaches. 
for ind in list(range(len(uniq_level2))):
    start = time.time()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('~~~~ STARTING BASIN', uniq_level2[ind], '~~~~')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    sl2 = np.where(sword_l2 == uniq_level2[ind])[0]
    rl2 = np.where(rch_l2 == uniq_level2[ind])[0]
    ml2 = np.where(mhv_l2 == uniq_level2[ind])[0]
    #sword variables
    clx = sword.centerlines.x[sl2]
    cly = sword.centerlines.y[sl2]
    cl_rchs = sword.centerlines.reach_id[0,sl2]
    cl_id = sword.centerlines.cl_id[sl2]
    rchs = sword.reaches.id[rl2]
    rch_type = rchs_type_all[rl2]
    rch_wth = sword.reaches.wth[rl2]
    rch_nchan = sword.reaches.nchan_mod[rl2]
    rch_x = sword.reaches.x[rl2]
    rch_y = sword.reaches.y[rl2]
    rch_max_wth = sword.reaches.max_width[rl2]
    #mhv variables
    mhv = nc.Dataset(mhv_files[ml2[0]])
    mhv_x = np.array(mhv['/centerlines/x'][:])
    mhv_y = np.array(mhv['/centerlines/y'][:])
    mhv_flag = np.array(mhv['/centerlines/swordflag_filt'][:])
    #fiter to sword flag only 
    flt = np.where(mhv_flag == 1)[0]
    mhv_x_filt = mhv_x[flt]
    mhv_y_filt = mhv_y[flt]

    mhv_pts = np.vstack((mhv_x_filt, mhv_y_filt)).T
    sword_pts = np.vstack((clx, cly)).T
    kdt = sp.cKDTree(mhv_pts)
    pt_dist, pt_ind = kdt.query(sword_pts, k = 6) #mhv points close to sword. 
    cl_med_dist = np.median(pt_dist, axis=1) #median distance to mhv 

    print('Starting SWORD Filter')
    start = time.time()
    shift_flag = np.zeros(len(rchs))
    for r in list(range(len(rchs))):
        cind = np.where(cl_rchs == rchs[r])[0]
        dist = cl_med_dist[cind]
        chan = rch_nchan[r]
        width = rch_wth[r]
        max_width =  np.max(rch_max_wth[r])
        
        # default radius/threshold is 200 for reaches with no width information.
        if width <= 30 and max_width <= 30:
            radius = 100
        # if width value for reach is bad, use 2x max_width value if it's good. 
        elif width <= 30 and max_width > 30:
            radius = max_width*2
        # use 2x reach width value for reach with width information. 
        else:
            radius = width*2

        threshold = geo.meters_to_degrees(radius, rch_y[r])
        flag = np.where(dist>threshold)[0] 
        if len(flag) == 0:
            continue 
        else:
            perc = (len(flag)/len(cind))*100

        if perc > 25 and chan == 1: #was perc > 50.
            if width >= 100:
                shift_flag[r] = 3
            elif 50 <= width < 100: 
                shift_flag[r] = 2
            else:
                shift_flag[r] = 1
            
    end = time.time()
    print(str(np.round((end-start),2))+' sec')

    #erase lakes, ghost, and dam reaches from flag. 
    shift_flag[np.where(rch_type == 3)[0]]=0
    shift_flag[np.where(rch_type == 4)[0]]=0
    shift_flag[np.where(rch_type == 6)[0]]=0
    ## filling in the continental scale
    shift_flag_all[rl2] = shift_flag

#save data. 
gpkg['shift_flag'] = shift_flag_all
gpkg.to_file(outdir+region.lower()+'_shift_flag_'+version+
             '.gpkg', driver='GPKG', layer='reaches')

end_all = time.time()
print('FINISHED ALL BASINS IN: '+str(np.round((end-start)/60,2))+' mins')

### PLOTS
# import matplotlib.pyplot as plt
# f = np.where(shift_flag>0)[0]
# plt.scatter(mhv_x_filt, mhv_y_filt, c='lightgrey',s=2)
# plt.scatter(clx, cly, c='black',s=2)
# plt.scatter(rch_x[f], rch_y[f], c='red')
# plt.show()

# geo.meters_to_degrees(1000, np.median(sword.lat[rch]))
# plt.scatter(sword.lon, sword.lat, c='blue', s = 3)
# plt.scatter(new_lon, new_lat, c='magenta', s = 3)
# plt.scatter(new_lon_smooth, new_lat_smooth, c='cyan', s = 3)
# plt.show()
