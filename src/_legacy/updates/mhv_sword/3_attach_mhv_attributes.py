"""
Attaching Regional Auxillary Attributes to MERIT Hydro
Vector-SWORD (MHV-SWORD) translation dataset.
(3_attach_mhv_attributes.py)
===================================================

This script attaches auxially dataset attributes
from MERIT Hydro rasters to the MHV-SWORD database. 
These attributes are needed to add MHV centerlines 
to SWORD.

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA).

Execution example (terminal):
    python path/to/3_attach_mhv_attributes NA

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import netCDF4 as nc 
from scipy import spatial as sp
import time
import glob
import argparse
import src.updates.geo_utils as geo 
import src.updates.auxillary_utils as aux 
import warnings
warnings.filterwarnings("ignore") #if code stops working may need to comment out to check warnings. 

start_all = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Region", type = str)
args = parser.parse_args()

region = args.region

mh_elv_dir = main_dir+'/data/inputs/MERIT_Hydro/'+region+'/elv/'
mh_facc_dir = main_dir+'/data/inputs/MERIT_Hydro/'+region+'/upa/'
mh_wth_dir = main_dir+'/data/inputs/MERIT_Hydro/'+region+'/wth/'
facc_paths = np.sort(np.array([file for file in geo.getListOfFiles(mh_facc_dir) if '.tif' in file]))
elv_paths = np.sort(np.array([file for file in geo.getListOfFiles(mh_elv_dir) if '.tif' in file]))
wth_paths = np.sort(np.array([file for file in geo.getListOfFiles(mh_wth_dir) if '.tif' in file]))
mhv_fn = main_dir+'/data/inputs/MHV_SWORD/netcdf/'+region+'/'
mhv_files = glob.glob(os.path.join(mhv_fn, '*.nc'))

for f in list(range(len(mhv_files))): #having trouble with ind = 5 for AS (basin 35)
    print('Starting File:', mhv_files[f][-25:])
    mhv = nc.Dataset(mhv_files[f], 'r+')
    mhv_lon = mhv.groups['centerlines'].variables['x'][:].data
    mhv_lat = mhv.groups['centerlines'].variables['y'][:].data
    mhv_id = mhv.groups['centerlines'].variables['new_segs'][:].data
    mhv_ind = mhv.groups['centerlines'].variables['new_segs_ind'][:].data
    #convert degrees over 180. 
    convert = np.where(mhv_lon > 180)[0]
    if len(convert) > 0:
        mhv_lon[convert] = mhv_lon[convert]-360

    mhv_points = [(mhv_lon[i], mhv_lat[i]) for i in range(len(mhv_lon))]
    mhv_pts = np.array(mhv_points)

    #filler variables
    mhv_nchan = np.repeat(1, len(mhv_lon))
    mhv_manual_add = np.repeat(0, len(mhv_lon))
    mhv_tile = np.repeat('NaNtile', len(mhv_lon))
    mhv_elv = np.repeat(0, len(mhv_lon))
    mhv_wth = np.repeat(0, len(mhv_lon))
    mhv_facc = np.repeat(0, len(mhv_lon))
    mhv_endpts = np.repeat(0, len(mhv_lon))
    mhv_dist = np.repeat(0, len(mhv_lon))
    mhv_x = np.repeat(0, len(mhv_lon))
    mhv_y = np.repeat(0, len(mhv_lon))
    mhv_cl_id = np.repeat(0, len(mhv_lon))

    #get overlapping mhv tiles with basin. 
    elv_basin_paths = geo.pt_raster_overlap(mhv_lon, mhv_lat, elv_paths)

    start = time.time()
    print('======== Starting MH-MHV Merge ========')
    for ind in list(range(len(elv_basin_paths))):
        # print(ind, len(elv_basin_paths))
        ind2 = np.where(elv_paths == elv_basin_paths[ind])[0][0]
        tile = elv_paths[ind][-15:-8]
        mh_lon, mh_lat, mh_elv, mh_wth, mh_facc = aux.mh_vals(elv_paths[ind2], 
                                                              wth_paths[ind2], 
                                                              facc_paths[ind2])
        
        if len(mh_lon) == 0:
            continue

        else:
            # create bounding box to clip mhv by.
            xmin = np.min(mh_lon)
            xmax = np.max(mh_lon)
            ymin = np.min(mh_lat)
            ymax = np.max(mh_lat)
            ll = np.array([xmin, ymin])  # lower-left
            ur = np.array([xmax, ymax])  # upper-right
            
            mhv_idx = np.all(np.logical_and(ll <= mhv_pts, mhv_pts <= ur), axis=1)
            if len(mhv_idx) == 0:
                continue
            else:
                mhv_lon_crop = mhv_lon[mhv_idx]
                mhv_lat_crop = mhv_lat[mhv_idx]

                mh_pts = np.vstack((mh_lon, mh_lat)).T
                mhv_pts_crop = np.vstack((mhv_lon_crop,mhv_lat_crop)).T
                kdt = sp.cKDTree(mh_pts)
                pt_dist, pt_ind = kdt.query(mhv_pts_crop, k = 5)

                elv = mh_elv[pt_ind[:,0]]
                wth = mh_wth[pt_ind[:,0]]
                facc = mh_facc[pt_ind[:,0]]

                mhv_elv[mhv_idx] = elv
                mhv_wth[mhv_idx] = wth
                mhv_facc[mhv_idx] = facc
                mhv_tile[mhv_idx] = tile

    end = time.time()
    print('Finished ' + region + ' Merge in: ' + str(np.round((end-start)/60, 2)) + ' min')

    start2 = time.time()
    print('======== Starting Segment Calculations ========')
    unq_seg = np.unique(mhv_id)
    cnt = 0
    for idx in list(range(len(unq_seg))):
        # print(idx, len(unq_seg))
        pts = np.where(mhv_id == unq_seg[idx])[0]
        seg_ind = mhv_ind[pts]
        seg_lon = mhv_lon[pts]
        seg_lat = mhv_lat[pts]
        seg_elv = mhv_elv[pts]
        seg_facc = mhv_facc[pts]
        seg_epts = mhv_endpts[pts]
        
        #assigning endpoint values
        mx = np.where(seg_ind == max(seg_ind))[0]
        mn = np.where(seg_ind == min(seg_ind))[0]
        seg_epts[mn] = 1
        seg_epts[mx] = 2

        #creating unqiue cl_id for basin. 
        seg_cl_id = seg_ind+cnt
        cnt = cnt+1

        # getting utm coordinates
        seg_x, seg_y, __, __ = geo.reproject_utm(seg_lat, seg_lon)

        #segment distance
        sort_ind = np.argsort(seg_ind)
        x_coords = seg_lon[sort_ind]
        y_coords = seg_lat[sort_ind]
        diff = geo.get_distances(x_coords,y_coords)
        seg_dist = np.cumsum(diff)

        #filling in arrays.
        mhv_cl_id[pts] = seg_cl_id
        mhv_endpts[pts] = seg_epts
        mhv_dist[pts[sort_ind]] = seg_dist
        mhv_x[pts] = seg_x
        mhv_y[pts] = seg_y
        
    end2 = time.time()
    print('Finished Segments in: ' + str(np.round((end2-start2)/60, 2)) + ' min')

    print('======== Adding New Variables to NetCDF ========')
    if 'flowacc' in mhv.groups['centerlines'].variables:
        mhv.groups['centerlines'].variables['cl_id'][:] = mhv_cl_id
        mhv.groups['centerlines'].variables['easting'][:] = mhv_x
        mhv.groups['centerlines'].variables['northing'][:] = mhv_y
        # mhv.groups['centerlines'].variables['segInd'][:] = mhv_ind
        mhv.groups['centerlines'].variables['new_segDist'][:] = mhv_dist
        mhv.groups['centerlines'].variables['p_width'][:] = mhv_wth
        mhv.groups['centerlines'].variables['p_height'][:] = mhv_elv
        mhv.groups['centerlines'].variables['flowacc'][:] = mhv_facc
        mhv.groups['centerlines'].variables['nchan'][:] = mhv_nchan
        mhv.groups['centerlines'].variables['manual_add'][:] = mhv_manual_add
        mhv.groups['centerlines'].variables['endpoints'][:] = mhv_endpts
        mhv.groups['centerlines'].variables['mh_tile'][:] = mhv_tile
        mhv.close()
    else:
        mhv.groups['centerlines'].createVariable('cl_id', 'i8', ('num_points',))
        mhv.groups['centerlines'].createVariable('easting', 'f8', ('num_points',))
        mhv.groups['centerlines'].createVariable('northing', 'f8', ('num_points',))
        # mhv.groups['centerlines'].createVariable('segInd', 'f8', ('num_points',))
        mhv.groups['centerlines'].createVariable('new_segDist', 'f8', ('num_points',))
        mhv.groups['centerlines'].createVariable('p_width', 'f8', ('num_points',))
        mhv.groups['centerlines'].createVariable('p_height', 'f8', ('num_points',))
        mhv.groups['centerlines'].createVariable('flowacc', 'f8', ('num_points',))
        mhv.groups['centerlines'].createVariable('nchan', 'i4', ('num_points',))
        mhv.groups['centerlines'].createVariable('manual_add', 'i4', ('num_points',))
        mhv.groups['centerlines'].createVariable('endpoints', 'i4', ('num_points',))
        mhv.groups['centerlines'].createVariable('mh_tile', 'S7', ('num_points',))
        mhv.groups['centerlines'].variables['mh_tile']._Encoding = 'ascii'
        # populating new variables.
        mhv.groups['centerlines'].variables['cl_id'][:] = mhv_cl_id
        mhv.groups['centerlines'].variables['easting'][:] = mhv_x
        mhv.groups['centerlines'].variables['northing'][:] = mhv_y
        # mhv.groups['centerlines'].variables['segInd'][:] = mhv_ind
        mhv.groups['centerlines'].variables['new_segDist'][:] = mhv_dist
        mhv.groups['centerlines'].variables['p_width'][:] = mhv_wth
        mhv.groups['centerlines'].variables['p_height'][:] = mhv_elv
        mhv.groups['centerlines'].variables['flowacc'][:] = mhv_facc
        mhv.groups['centerlines'].variables['nchan'][:] = mhv_nchan
        mhv.groups['centerlines'].variables['manual_add'][:] = mhv_manual_add
        mhv.groups['centerlines'].variables['endpoints'][:] = mhv_endpts
        mhv.groups['centerlines'].variables['mh_tile'][:] = mhv_tile
        mhv.close()

end_all = time.time()
print('Finished '+region+' in: ' + str(np.round((end_all-start_all)/60, 2)) + ' min')



### PLOTS
# import matplotlib.pyplot as plt

# z = np.where(mhv_wth == 0)[0]
# plt.scatter(mhv_lon, mhv_lat, s = 3, c=np.log(mhv_wth), cmap='rainbow')
# # plt.scatter(mhv_lon[z], mhv_lat[z], s = 3, c='lightgrey')
# plt.show()

# z = np.where(mhv_facc == 0)[0]
# plt.scatter(mhv_lon, mhv_lat, s = 3, c=np.log(mhv_facc), cmap='rainbow')
# # plt.scatter(mhv_lon[z], mhv_lat[z], s = 3, c='lightgrey')
# plt.show()

# z = np.where(mhv_elv == 0)[0]
# plt.scatter(mhv_lon, mhv_lat, s = 3, c=mhv_elv, cmap='rainbow')
# plt.scatter(mhv_lon[z], mhv_lat[z], s = 3, c='lightgrey')
# plt.show()

# plt.scatter(mhv_lon, mhv_lat, s = 3, c='blue')
# plt.scatter(mh_lon, mh_lat, s = 1, c='red')
# plt.show()

# plt.scatter(mh_lon, mh_lat, s = 1, c=np.log(mh_wth), cmap='rainbow')
# plt.show()

# plt.scatter(mh_lon, mh_lat, s = 1, c=np.log(mh_elv), cmap='rainbow')
# plt.show()


# plt.scatter(mhv_lon[pts], mhv_lat[pts], s = 5, c=mhv_dist[pts], cmap='rainbow')
# plt.show()
