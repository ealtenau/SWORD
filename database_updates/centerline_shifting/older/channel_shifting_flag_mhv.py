from __future__ import division
import os
import math
from shapely.geometry import Point
import numpy as np
from osgeo import gdal
import rasterio
from scipy import spatial as sp
import geopandas as gp
import pandas as pd
import time
import netCDF4 as nc
import argparse
import glob
import matplotlib.pyplot as plt

###############################################################################
######################## Reading and Writing Functions ########################
###############################################################################

class Object(object):
    """
    FUNCTION:
        Creates class object to assign attributes to.
    """
    pass

###############################################################################

def meters_to_degrees(meters, latitude):
    deg = np.round(meters/(111.32 * 1000 * math.cos(latitude * (math.pi / 180))),5)
    return deg

###############################################################################
###############################################################################
###############################################################################

start_all = time.time()

# parser = argparse.ArgumentParser()
# parser.add_argument("region", help="<Required> Region", type = str)
# parser.add_argument("version", help="<Required> Version", type = str)
# args = parser.parse_args()

# region = args.region
# version = args.version

region = 'NA'
version = 'v17b'

# Input file(s).
mhv_dir = '/Users/ealtenau/Documents/SWORD_Dev/inputs/MHV_SWORD/netcdf/' + region +'/'
mhv_files = glob.glob(os.path.join(mhv_dir, '*.nc'))
sword_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
gpkg_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/gpkg/'+region.lower()+'_sword_reaches_'+version+'.gpkg'
outdir = '/Users/ealtenau/Documents/SWORD_Dev/shift_testing/'

if os.path.exists(outdir) == False:
    os.makedirs(outdir)

gpkg = gp.read_file(gpkg_fn)
sword = nc.Dataset(sword_fn)
clx_all = np.array(sword['/centerlines/x'][:])
cly_all = np.array(sword['/centerlines/y'][:])
cl_id_all = np.array(sword['/centerlines/cl_id'][:])
cl_rchs_all = np.array(sword['/centerlines/reach_id'][0,:])
rchs_all = np.array(sword['/reaches/reach_id'][:])
rchs_type_all = np.array([int(str(ind)[-1]) for ind in rchs_all])
rchs_wth_all = np.array(sword['/reaches/width'][:])
rchs_nchan_all = np.array(sword['/reaches/n_chan_mod'][:])
rchs_x_all = np.array(sword['/reaches/x'][:])
rchs_y_all = np.array(sword['/reaches/y'][:])
rchs_max_wth_all = np.array(sword['/reaches/max_width'][:])
shift_flag_all = np.zeros(len(rchs_all))

### need to filter by level 2... 
sword_l2 = np.array([int(str(ind)[0:2]) for ind in cl_rchs_all])
rch_l2 = np.array([int(str(ind)[0:2]) for ind in rchs_all])
mhv_l2 = np.array([int(ind[-13:-11]) for ind in mhv_files])
uniq_level2 = np.unique(sword_l2)

for ind in list(range(len(uniq_level2))):
    start = time.time()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('~~~~ STARTING BASIN', uniq_level2[ind], '~~~~')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    sl2 = np.where(sword_l2 == uniq_level2[ind])[0]
    rl2 = np.where(rch_l2 == uniq_level2[ind])[0]
    ml2 = np.where(mhv_l2 == uniq_level2[ind])[0]
    #sword variables
    clx = clx_all[sl2]
    cly = cly_all[sl2]
    cl_rchs = cl_rchs_all[sl2]
    cl_id = cl_id_all[sl2]
    rchs = rchs_all[rl2]
    rch_type = rchs_type_all[rl2]
    rch_wth = rchs_wth_all[rl2]
    rch_nchan = rchs_nchan_all[rl2]
    rch_x = rchs_x_all[rl2]
    rch_y = rchs_y_all[rl2]
    rch_max_wth = rchs_max_wth_all[rl2]
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

        threshold = meters_to_degrees(radius, rch_y[r])
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


gpkg['shift_flag'] = shift_flag_all
gpkg.to_file(outdir+region.lower()+'_shift_flag_'+version+'.gpkg', driver='GPKG', layer='reaches')

end_all = time.time()
print('FINISHED ALL BASINS IN: '+str(np.round((end-start)/60,2))+' mins')

# f = np.where(shift_flag>0)[0]
# plt.scatter(mhv_x_filt, mhv_y_filt, c='lightgrey',s=2)
# plt.scatter(clx, cly, c='black',s=2)
# plt.scatter(rch_x[f], rch_y[f], c='red')
# plt.show()

# meters_to_degrees(1000, np.median(sword.lat[rch]))
# plt.scatter(sword.lon, sword.lat, c='blue', s = 3)
# plt.scatter(new_lon, new_lat, c='magenta', s = 3)
# plt.scatter(new_lon_smooth, new_lat_smooth, c='cyan', s = 3)
# plt.show()
