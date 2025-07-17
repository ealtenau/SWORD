# -*- coding: utf-8 -*-
"""
Creating and formatting MERIT Hydro Vector (MHV) - 
SWORD translation dataset (MHV-SWORD).  
(1_mhv_sword_flag_db.py)
===================================================

This script converts the MHV polyline vector files 
into a point-based dataset with a flag indicating if 
MHV is closely associated with SWORD. 

Data is output in geopackage and netCDF file formats
at Pfafstetter level 2 basin scale. Output files are
located at:
'/data/inputs/MHV_SWORD/gpkg/'
'/data/inputs/MHV_SWORD/netcdf'

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e v17).

Execution example (terminal):
    python path/to/1_mhv_sword_flag_db.py NA v17

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import time
import geopandas as gp
import pandas as pd
from shapely.geometry import Point
import argparse
import src.updates.geo_utils as geo 
import src.updates.mhv_sword.mhv_reach_def_tools as rdt
from src.updates.sword import SWORD
import warnings
warnings.filterwarnings("ignore") #if code stops working may need to comment out to check warnings. 

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Region", type = str)
parser.add_argument("version", help="<Required> Version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

#define file paths. 
outdir = main_dir+'/data/inputs/MHV_SWORD/'
mhv_dir = main_dir+'/data/inputs/MeritHydroVector/'
mhv_files = np.sort(np.array(np.array([file for file in geo.getListOfFiles(mhv_dir) if '.shp' in file])))
mhv_basins = np.array([int(f[-6:-4]) for f in mhv_files])

#read data. 
sword = SWORD(main_dir, region, version)
sword_l2 = np.array([int(str(ind)[0:2]) for ind in sword.centerlines.reach_id[0,:]])
unq_l2 = np.unique(sword_l2)

start_all = time.time()
for ind in list(range(4,len(unq_l2))):
    start_basin = time.time()
    print('Starting Basin ' + str(unq_l2[ind]))
    pts = np.where(sword_l2 == unq_l2[ind])[0]
    swd_x = sword.centerlines.x[pts]
    swd_y = sword.centerlines.y[pts]
    swd_id = sword.centerlines.reach_id[0,pts]
    f = np.where(mhv_basins == unq_l2[ind])[0]
    mhv = gp.read_file(mhv_files[int(f)])
    
    print('Converting Lines to Points & SWORD Flag')
    start = time.time()
    x_pts, y_pts, seg_pts, \
        seg_ind_pts, so_pts, \
            up_pts, down_pts, \
                flag_pts, flag = rdt.sword_flag(mhv,swd_x,swd_y)
    end = time.time()
    print(str((end-start)/60) + ' min')
    
    print('Identifying Headwaters, Outlets, andh Junctions')
    start = time.time()
    seg_hwout_pts, seg_junc_pts = rdt.identify_hw_junc(seg_pts, up_pts, down_pts)
    end = time.time()
    print(str((end-start)/60) + ' min')
    
    print('Aggregating MHV Segments')
    start = time.time()
    new_segs = rdt.aggregate_segs(seg_pts, seg_hwout_pts, seg_junc_pts, up_pts, down_pts) 
    end = time.time()
    print(str((end-start)/60) + ' min')

    print('Updating Indexes for New Segments')
    start = time.time()
    new_indexes = rdt.update_indexes(seg_pts, seg_ind_pts, up_pts, down_pts, 
                                     seg_hwout_pts, seg_junc_pts, new_segs)
    end = time.time()
    print(str((end-start)/60) + ' min')

    print('Filtering SWORD Flag')
    start = time.time()
    flag_filt, count = rdt.filter_sword_flag(seg_pts, seg_ind_pts,flag_pts, x_pts, y_pts)
    end = time.time()
    print(str((end-start)/60) + ' min, Segments corrected: ' + str(len(count)))

    print('Creating and Writing GPKG')
    start = time.time()
    mhv_pts = gp.GeoDataFrame([
        x_pts,
        y_pts,
        seg_pts,
        seg_ind_pts,
        so_pts,
        flag_pts,
        up_pts,
        down_pts,
        seg_hwout_pts,
        seg_junc_pts,
        new_segs,
        new_indexes,
        flag_filt,]).T

    mhv_pts.rename(
        columns={
            0:"x",
            1:"y",
            2:"segment",
            3:"segInd",
            4:"strmorder",
            5:"sword_flag",
            6:"upNode",
            7:"downNode",
            8:"hw_out",
            9:"junc",
            10:"new_segID",
            11:"new_segInd",
            12:"sflag_filt"},inplace=True)

    mhv_pts = mhv_pts.apply(pd.to_numeric, errors='ignore') # nodes.dtypes
    geom = gp.GeoSeries(map(Point, zip(x_pts, y_pts)))
    mhv_pts['geometry'] = geom
    mhv_pts = gp.GeoDataFrame(mhv_pts)
    mhv_pts.set_geometry(col='geometry')
    mhv_pts = mhv_pts.set_crs(4326, allow_override=True)
    outgpkg = outdir+'gpkg/'+region+'/mhv_sword_hb'+str(unq_l2[ind])+'_pts_'+version+'.gpkg'
    mhv_pts.to_file(outgpkg, driver='GPKG', layer='mhv_pts')
    end = time.time()
    print(str((end-start)/60) + ' min')

    print('Writing NetCDF')
    outnet = outdir+'netcdf/'+region+'/mhv_sword_hb'+str(unq_l2[ind])+'_pts_'+version+'.nc'
    l2_basin = np.repeat(unq_l2[ind], len(seg_pts))
    rdt.save_mhv_nc(x_pts, y_pts, seg_pts, seg_ind_pts, so_pts, flag_pts, up_pts, down_pts, 
                    seg_hwout_pts, seg_junc_pts, new_segs, new_indexes, flag_filt, region, outnet)
    end_basin = time.time()
    print('Finished Basin '+ str(unq_l2[ind]) + ' in: ' + str(np.round((end_basin-start_basin)/3600, 2)) + ' hrs')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

end_all=time.time()
print('*** Finished ALL Basins in: ' + str(np.round((end_all-start_all)/3600, 2)) + ' hrs ***')


### Export the Lines (Optional)
    # mhv['sword_flag'] = flag
    # mhv = mhv.drop(columns=['points'])
    # mhv.set_geometry(col='geometry') #removed "inplace=True" option on leopold. 
    # mhv = mhv.set_crs(4326, allow_override=True)
    # mhv.to_file(outdir+region.lower()+'/mhv_sword_hb'+str(unq_l2[ind])+'.gpkg', driver='GPKG', layer='mhv')
