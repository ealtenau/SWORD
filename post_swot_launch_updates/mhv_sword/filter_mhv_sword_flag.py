from __future__ import division
import numpy as np
from scipy import spatial as sp
import netCDF4 as nc
import time
import os 
import geopandas as gp
import pandas as pd
from shapely.geometry import Point
import argparse 

###############################################################################

def find_neighbors(basin_rch, basin_flag, basin_x, basin_y, 
                   rch_x, rch_y, rch_ind, rch_id, rch):

    # Formatting all basin coordinate values.
    basin_pts = np.vstack((basin_x, basin_y)).T
    # Formatting the current reach's endpoint coordinates.
    if len(rch) == 1:
        eps = np.array([0,0])
    else:
        pt1 = np.where(rch_ind == np.min(rch_ind))[0][0]
        pt2 = np.where(rch_ind == np.max(rch_ind))[0][0]
        eps = np.array([pt1,pt2]).T

    # Performing a spatial query to get the closest points within the basin
    # to the current reach's endpoints.
    ep_pts = np.vstack((rch_x[eps], rch_y[eps])).T
    kdt = sp.cKDTree(basin_pts)

    #for grwl the values were 100 and 200 
    if len(rch) <= 4:
        pt_dist, pt_ind = kdt.query(ep_pts, k = 4, distance_upper_bound = 0.003) #distance upper bound = 300.0 for meters 
    else:#elif rch_len > 600:
        pt_dist, pt_ind = kdt.query(ep_pts, k = 10, distance_upper_bound = 0.003) #distance upper bound = 300.0 for meters 

    # Identifying endpoint neighbors.
    ep1_ind = pt_ind[0,:]
    ep1_dist = pt_dist[0,:]
    na1 = np.where(ep1_ind == len(basin_rch))
    ep1_dist = np.delete(ep1_dist, na1)
    ep1_ind = np.delete(ep1_ind, na1)
    s1 = np.where(basin_rch[ep1_ind] == rch_id)
    ep1_dist = np.delete(ep1_dist, s1)
    ep1_ind = np.delete(ep1_ind, s1)
    ep1_ngb = np.unique(basin_rch[ep1_ind])

    ep2_ind = pt_ind[1,:]
    ep2_dist = pt_dist[1,:]
    na2 = np.where(ep2_ind == len(basin_rch))
    ep2_dist = np.delete(ep2_dist, na2)
    ep2_ind = np.delete(ep2_ind, na2)
    s2 = np.where(basin_rch[ep2_ind] == rch_id)
    ep2_dist = np.delete(ep2_dist, s2)
    ep2_ind = np.delete(ep2_ind, s2)
    ep2_ngb = np.unique(basin_rch[ep2_ind])

    # Pulling attribute information for the endpoint neighbors.
    ep1_flg = np.zeros(len(ep1_ngb))
    for idy in list(range(len(ep1_ngb))):
        ep1_flg[idy] = np.max(basin_flag[np.where(basin_rch == ep1_ngb[idy])])

    ep2_flg = np.zeros(len(ep2_ngb))
    for idy in list(range(len(ep2_ngb))):
        ep2_flg[idy] = np.max(basin_flag[np.where(basin_rch == ep2_ngb[idy])])

    # Creating final arrays.
    ep1 = np.array([ep1_ngb, ep1_flg]).T
    ep2 = np.array([ep2_ngb, ep2_flg]).T

    return ep1, ep2

###############################################################################
###############################################################################
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Region", type = str)
parser.add_argument("local_processing", help="'True' for local machine, 'False' for server", type = str)
args = parser.parse_args()

start_all = time.time()
region = args.region

if args.local_processing == 'True':
    main_dir = '/Users/ealteanau/Documents/SWORD_Dev/inputs/'
else:
    main_dir = '/afs/cas.unc.edu/depts/geological_sciences/pavelsky/students/ealtenau/SWORD_dev/inputs/'
mhv_file = main_dir+'MHV_SWORD/'+region+'_mhv_sword.nc'

mhv = nc.Dataset(mhv_file, 'r+')
flag_all = np.array(mhv.groups['centerlines'].variables['swordflag'][:])
lon_all = np.array(mhv.groups['centerlines'].variables['x'][:])
lat_all = np.array(mhv.groups['centerlines'].variables['y'][:])
x_all = np.array(mhv.groups['centerlines'].variables['easting'][:])
y_all = np.array(mhv.groups['centerlines'].variables['northing'][:])
seg_all = np.array(mhv.groups['centerlines'].variables['segID'][:]) 
index_all = np.array(mhv.groups['centerlines'].variables['segInd'][:]) 
basins = np.array(mhv.groups['centerlines'].variables['basin_code'][:])

l2 = np.array([int(str(ind)[0:2]) for ind in basins])
unq_l2 = np.unique(l2)
unq_l2 = np.delete(unq_l2, 0)

start_all = time.time()
for ind in list(range(len(unq_l2))):
    print('STARTING BASIN: '+ str(unq_l2[ind]))
    subset = np.where(l2 == unq_l2[ind])[0]
    flag = flag_all[subset]
    x = x_all[subset]
    y = y_all[subset]
    lon = lon_all[subset]
    lat = lat_all[subset]
    seg = seg_all[subset]
    index = index_all[subset]

    cnt=[]
    check = np.unique(seg[np.where(flag == 0)[0]])
    for s in list(range(len(check))):
        # print(s, len(check)-1)
        line = np.where(seg == check[s])[0]
        seg_x = x[line]
        seg_y = y[line]
        seg_lon = lon[line]
        seg_lat = lat[line]
        seg_ind = index[line]
        end1, end2 = find_neighbors(seg, flag, lon, lat, seg_lon, 
                                    seg_lat, seg_ind, check[s], line)
        if len(end1) == 0:
            continue
        elif len(end2) == 0:
            continue
        else:
            # Cond. 1: end 1 has SWORD flag, but end 2 does not. 
            if np.max(end1[:,1]) == 1 and np.max(end2[:,1]) == 0:
                for n in list(range(len(end2))):
                    line2 = np.where(seg == end2[0,0])[0]
                    seg_lon2 = lon[line2]
                    seg_lat2 = lat[line2]
                    seg_ind2 = index[line2]
                    ngh_end1, ngh_end2 = find_neighbors(seg, flag, lon, lat, seg_lon2, 
                                        seg_lat2, seg_ind2, check[s], line2)
                    if n == 0:
                        ngh_end1_all = np.copy(ngh_end1)
                        ngh_end2_all = np.copy(ngh_end2)
                    else:
                        ngh_end1_all = np.concatenate((ngh_end1_all, ngh_end1), axis = 0)
                        ngh_end2_all = np.concatenate((ngh_end2_all, ngh_end2), axis = 0)
                if np.max(ngh_end1_all[:,1]) == 1 or np.max(ngh_end2_all[:,1]) == 1:
                    print(s, check[s], 'cond.1')
                    flag_all[subset[line]] = 1
                    flag[line] = 1
                    cnt.append(check[s])
                else:
                    continue
            # Cond. 2: end 2 has SWORD flag, but end 1 does not.
            elif np.max(end1[:,1]) == 0 and np.max(end2[:,1]) == 1:
                for n in list(range(len(end1))):
                    line2 = np.where(seg == end1[0,0])[0]
                    seg_lon2 = lon[line2]
                    seg_lat2 = lat[line2]
                    seg_ind2 = index[line2]
                    ngh_end1, ngh_end2 = find_neighbors(seg, flag, lon, lat, seg_lon2, 
                                        seg_lat2, seg_ind2, check[s], line2)
                    if n == 0:
                        ngh_end1_all = np.copy(ngh_end1)
                        ngh_end2_all = np.copy(ngh_end2)
                    else:
                        ngh_end1_all = np.concatenate((ngh_end1_all, ngh_end1), axis = 0)
                        ngh_end2_all = np.concatenate((ngh_end2_all, ngh_end2), axis = 0)
                if np.max(ngh_end1_all[:,1]) == 1 or np.max(ngh_end2_all[:,1]) == 1:
                    print(s, check[s], 'cond.2')
                    flag_all[subset[line]] = 1
                    flag[line] = 1
                    cnt.append(check[s])
                else:
                    continue
            # Cond. 3: Both ends have SWORD flag. 
            elif np.max(end1[:,1]) == 1 and np.max(end2[:,1]) == 1:
                print(s, check[s], 'cond.3')
                flag_all[subset[line]] = 1
                # flag[line] = 1
                cnt.append(check[s])

            else:
                continue

### update netcdf. 
mhv.groups['centerlines'].createVariable('swordflag_filt', 'i4', ('num_points',))
mhv.groups['centerlines'].variables['swordflag_filt'][:] = flag_all
mhv.close()

end_all=time.time()
print('Time to Finish: ' +str(np.round((end_all-start_all)/60, 2)) + 
      ' min. Segments corrected: ' + str(len(cnt)))

'''
pts = gp.GeoDataFrame([
    lon,
    lat,
    seg,
    flag,
]).T

pts.rename(
    columns={
        0:"x",
        1:"y",
        2:"segment",
        3:"swordflag",
        },inplace=True)

pts = pts.apply(pd.to_numeric, errors='ignore') # pts.dtypes
geom = gp.GeoSeries(map(Point, zip(lon, lat)))
pts['geometry'] = geom
pts = gp.GeoDataFrame(pts)
pts.set_geometry(col='geometry')
pts = pts.set_crs(4326, allow_override=True)

outgpkg = '/Users/ealteanau/Documents/SWORD_Dev/inputs/MHV_SWORD/hb81_mhv_swordflag_filt_test3.gpkg'
pts.to_file(outgpkg, driver='GPKG', layer='pts')
'''



