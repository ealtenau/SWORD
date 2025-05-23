
from __future__ import division
import os
main_dir = os.getcwd()
import numpy as np
import time
import netCDF4 as nc
import pandas as pd
from scipy import spatial as sp
from shapely.geometry import Point
import geopandas as gp
import glob

###############################################################################

class Object(object):
    """
    FUNCTION:
        Creates class object to assign attributes to.
    """
    pass

###############################################################################

def find_tributary_junctions(centerlines):

    sword_pts = np.vstack((centerlines.x, centerlines.y)).T
    kdt = sp.cKDTree(sword_pts)
    pt_dist, pt_ind = kdt.query(sword_pts, k = 10, distance_upper_bound=0.005)

    tribs = np.zeros(len(centerlines.reach_id[0,:]))
    uniq_segs = np.unique(centerlines.reach_id[0,:])
    for ind in list(range(len(uniq_segs))):
        print(ind, len(uniq_segs)-1)

        # Isolate endpoints for the edited segment.
        seg = np.where(centerlines.reach_id[0,:] == uniq_segs[ind])[0]
        pt1 = seg[np.where(centerlines.id[seg] == np.min(centerlines.id[seg]))[0]]
        pt2 = seg[np.where(centerlines.id[seg] == np.max(centerlines.id[seg]))[0]]
                                
        ep1_ind = pt_ind[pt1,:]
        ep1_dist = pt_dist[pt1,:]
        na1 = np.where(ep1_ind == len(centerlines.reach_id[0,:]))
        ep1_dist = np.delete(ep1_dist, na1)
        ep1_ind = np.delete(ep1_ind, na1)
        na1_1 = np.where(centerlines.reach_id[0,ep1_ind] == uniq_segs[ind])[0]
        ep1_dist = np.delete(ep1_dist, na1_1)
        ep1_ind = np.delete(ep1_ind, na1_1)

        ep2_ind = pt_ind[pt2,:]
        ep2_dist = pt_dist[pt2,:]
        na2 = np.where(ep2_ind == len(centerlines.reach_id[0,:]))
        ep2_dist = np.delete(ep2_dist, na2)
        ep2_ind = np.delete(ep2_ind, na2)
        na2_1 = np.where(centerlines.reach_id[0,ep2_ind] == uniq_segs[ind])[0]
        ep2_dist = np.delete(ep2_dist, na2_1)
        ep2_ind = np.delete(ep2_ind, na2_1)

        ep1_segs = np.unique(centerlines.reach_id[0,[ep1_ind]])
        ep2_segs = np.unique(centerlines.reach_id[0,[ep2_ind]])
        
        if len(ep1_segs) > 0:
            for e1 in list(range(len(ep1_segs))):
                #finding min/max reach cl_ids.
                s1 = np.where(centerlines.reach_id[0,:] == ep1_segs[e1])[0]
                ep1_min = np.min(centerlines.id[s1])
                ep1_max = np.max(centerlines.id[s1])
                #finding the junction point cl_id. 
                con1_ind = np.where(centerlines.reach_id[0,ep1_ind] == ep1_segs[e1])[0]
                con1_pt = ep1_ind[np.where(ep1_dist[con1_ind] == np.min(ep1_dist[con1_ind]))[0]][0]
                ep1_junct = centerlines.id[con1_pt]
                if ep1_junct > ep1_min+5 and ep1_junct < ep1_max-5:
                    if len(seg) >= 15: 
                        tribs[con1_pt] = 1
        
        if len(ep2_segs) > 0:
            for e2 in list(range(len(ep2_segs))):
                #finding min/max reach cl_ids. 
                s2 = np.where(centerlines.reach_id[0,:] == ep2_segs[e2])[0]
                ep2_min = np.min(centerlines.id[s2])
                ep2_max = np.max(centerlines.id[s2])
                #finding the junction point cl_id. 
                con2_ind = np.where(centerlines.reach_id[0,ep2_ind] == ep2_segs[e2])[0]
                con2_pt = ep2_ind[np.where(ep2_dist[con2_ind] == np.min(ep2_dist[con2_ind]))[0]][0]
                ep2_junct = centerlines.id[con2_pt]
                if ep2_junct > ep2_min+5 and ep2_junct < ep2_max-5:
                    if len(seg) >= 15:
                        tribs[con2_pt] = 1

    return tribs

#####################################################################################################
#####################################################################################################
#####################################################################################################

start_all = time.time()
region = 'OC'
version = 'v18'

sword_fn = main_dir+'/data/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
mhv_dir = main_dir+'/data/inputs/MHV_SWORD/netcdf/' + region +'/'
mhv_files = np.sort(glob.glob(os.path.join(mhv_dir, '*.nc')))

#reading in sword data.
sword = nc.Dataset(sword_fn)
centerlines = Object()
centerlines.id= sword.groups['centerlines'].variables['cl_id'][:]
centerlines.x = sword.groups['centerlines'].variables['x'][:]
centerlines.y = sword.groups['centerlines'].variables['y'][:]
centerlines.reach_id = sword.groups['centerlines'].variables['reach_id'][:]
centerlines.node_id = sword.groups['centerlines'].variables['node_id'][:]
sword.close()

for ind in list(range(len(mhv_files))):
    mhv_data = nc.Dataset(mhv_files[ind])
    if 'add_flag' in mhv_data.groups['centerlines'].variables.keys():
        mhv = Object()
        mhv.x = np.array(mhv_data['/centerlines/x'][:])
        mhv.y = np.array(mhv_data['/centerlines/y'][:])
        mhv.junc = np.array(mhv_data['/centerlines/add_flag'][:])
        join_pts = np.where(mhv.junc == 3)[0]

        ### new trib function. need to consider headwaters vs middle of the river additions. 
        sword_pts = np.vstack((centerlines.x, centerlines.y)).T
        mhv_pts = np.vstack((mhv.x[join_pts], mhv.y[join_pts])).T
        kdt = sp.cKDTree(sword_pts)
        pt_dist, pt_ind = kdt.query(mhv_pts, k = 5)

        join_ids = np.zeros(len(join_pts))
        trib = np.zeros(len(join_pts))
        for j in list(range(len(join_pts))):
            join_rch = centerlines.reach_id[0,pt_ind[j,0]]
            join_dist = pt_dist[j,0]
            if join_dist > 0.003:
                continue
            else:
                if str(join_rch)[-1] == '6':
                    rch = np.where(centerlines.reach_id[0,:] == join_rch)[0]
                    max_id = max(centerlines.id[rch])
                    join_ids[j] = max_id
                else:
                    join_ids[j] = centerlines.id[pt_ind[j,0]]
                    trib[j] = 1

        t = join_ids[np.where(trib == 1)[0]]
        tributaries = np.where(np.in1d(centerlines.id, t) == True)[0]
        df = pd.DataFrame(np.array([centerlines.x[tributaries], 
                                    centerlines.y[tributaries], 
                                    centerlines.reach_id[0,tributaries],
                                    centerlines.id[tributaries]]).T)
        df.rename(
            columns={
                0:"x",
                1:"y",
                2:"reach_id",
                3:"cl_id",
                },inplace=True)

        geom = gp.GeoSeries(map(Point, zip(centerlines.x[tributaries], centerlines.y[tributaries])))
        df['geometry'] = geom
        df = gp.GeoDataFrame(df)
        df.set_geometry(col='geometry')
        df = df.set_crs(4326, allow_override=True)

        if df.shape[0] == 0:
            continue
        else:
            outgpkg=main_dir+'/data/update_requests/'+version+'/'+region+'/tribs'+\
                '/hb'+mhv_files[ind][-13:-11]+'_sword_tributary_breaks_'+version+'.gpkg'
            df.to_file(outgpkg, driver='GPKG', layer='tribs')
            print('DONE', 'breaks', len(tributaries), 'reaches', len(np.unique(centerlines.reach_id[0,:])))
    
    else:
        continue

