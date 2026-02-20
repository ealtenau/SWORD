"""
Attaching Global Auxillary Attributes to MERIT Hydro
Vector-SWORD (MHV-SWORD) translation dataset.
(2_attach_global_attributes.py)
===================================================

This script attaches global-scale auxially dataset 
attributesto the MHV-SWORD database. These attributes 
are needed to add MHV centerlines to SWORD.

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA).

Execution example (terminal):
    python path/to/2_attach_global_attributes NA

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import time
import numpy as np
import glob
import geopandas as gp
import netCDF4 as nc
from shapely.geometry import Point
import pandas as pd
import argparse
import src.updates.geo_utils as geo 
import src.updates.auxillary_utils as aux 

start_all = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Region", type = str)
args = parser.parse_args()

region = args.region

#Define input directories and filenames. Will need to be changed based on user needs.
data_dir = main_dir+'/data/inputs/'

# Global Paths.
fn_grand = data_dir + 'GRAND/GRanD_dams_v1_1.shp'
fn_deltas = data_dir + 'Deltas/global_map.shp'
track_dir = data_dir + 'SWOT_Tracks/2020_orbits/'
track_list = glob.glob(os.path.join(track_dir, 'ID_PASS*.shp'))

# Regional Paths.
mhv_dir = data_dir + 'MHV_SWORD/netcdf/' + region + '/' #+ region.lower() + '_mhv_sword.nc'
mhv_files = glob.glob(os.path.join(mhv_dir, '*.nc'))
fn_grod = data_dir + 'GROD/GROD_'+region+'.csv'
fn_basins = data_dir + 'HydroBASINS/' + region + '/' + region + '_hb_lev08.shp'
lake_dir = data_dir + 'LakeDatabase/20200702_PLD/For_Merge/' + region + '/'
lake_path = np.array(np.array([file for file in geo.getListOfFiles(lake_dir) if '.shp' in file]))

# Open global shapefiles for spatial intersections. 
lake_db = gp.GeoDataFrame.from_file(lake_path[0])
delta_db = gp.GeoDataFrame.from_file(fn_deltas)

# Merging each level two basin file. 
for ind in list(range(len(mhv_files))):
    
    start = time.time()
    
    # Reading in data.
    mhv = nc.Dataset(mhv_files[ind], 'r+') 
    mhv_lon = mhv.groups['centerlines'].variables['x'][:]
    mhv_lat = mhv.groups['centerlines'].variables['y'][:]
    mhv_l2 = mhv.groups['centerlines'].variables['basin'][:]
    mhv_seg = mhv.groups['centerlines'].variables['segID'][:]

    # Creating fill variables.
    mhv_lake_id = np.zeros(len(mhv_lon))
    mhv_lakeflag = np.zeros(len(mhv_lon))
    mhv_deltaflag = np.zeros(len(mhv_lon))
    mhv_grand_id = np.zeros(len(mhv_lon))
    mhv_grod_id = np.zeros(len(mhv_lon))
    mhv_grod_fid = np.zeros(len(mhv_lon))
    mhv_hfalls_fid = np.zeros(len(mhv_lon))
    mhv_basin_code = np.zeros(len(mhv_lon))
    mhv_number_obs = np.zeros(len(mhv_lon))
    mhv_orbits = np.zeros([len(mhv_lon), 200])

    # Defining mhv extent.
    mhv_ext = [np.min(mhv_lon), np.min(mhv_lat),
                np.max(mhv_lon), np.max(mhv_lat)]

    # Creating geodataframe for spatial joins. 
    mhv_df = gp.GeoDataFrame([mhv_lon, mhv_lat]).T
    mhv_df.rename(columns={0:"x",1:"y"},inplace=True)
    mhv_df = mhv_df.apply(pd.to_numeric, errors='ignore')
    geom = gp.GeoSeries(map(Point, zip(mhv_lon, mhv_lat)))
    mhv_df['geometry'] = geom
    mhv_df = gp.GeoDataFrame(mhv_df)
    mhv_df.set_geometry(col='geometry')
    mhv_df = mhv_df.set_crs(4326, allow_override=True)

    # Subset Lake db to mhv extent.
    lake_db_clip = lake_db.cx[mhv_ext[0]:mhv_ext[2], mhv_ext[1]:mhv_ext[3]]

    # Attach Prior Lake Database (PLD) IDs.
    mhv_lakeid = geo.vector_to_vector_intersect(mhv_df, lake_db_clip, 'lake_id')
    mhv_lakes = np.zeros(len(mhv_lakeid))
    mhv_lakes[np.where(mhv_lakeid > 0)] = 1

    # Adding dam, basin, delta, and SWOT track information.
    mhv_grand, mhv_grod, \
        mhv_grodfid, mhv_hfallsfid = aux.add_dams(mhv_lon, mhv_lat, 
                                                  fn_grand, fn_grod)
    mhv_basins = geo.vector_to_vector_join_nearest(mhv_df, fn_basins,'PFAF_ID')
    mhv_deltas_raw = geo.vector_to_vector_intersect(mhv_df, delta_db, 'DeltaID')
    mhv_deltas = aux.filter_deltas(mhv_seg, mhv_deltas_raw)
    track_files = geo.pt_vector_overlap(mhv_lon, mhv_lat, track_list)
    mhv_numobs, mhv_orbs = aux.add_swot_tracks(mhv_df, track_files)

    # Filling in segment gaps with no basin values.
    mhv_new_basins = aux.filter_basin_codes(mhv_seg, mhv_basins)

    # Fill in local values. 
    mhv_lake_id[:] = mhv_lakeid
    mhv_lakeflag[:] = mhv_lakes
    mhv_deltaflag[:] = mhv_deltas
    mhv_grand_id[:] = mhv_grand
    mhv_grod_id[:] = mhv_grod
    mhv_grod_fid[:] = mhv_grodfid
    mhv_hfalls_fid[:] = mhv_hfallsfid
    mhv_basin_code[:] = mhv_new_basins
    mhv_number_obs[:] = mhv_numobs
    mhv_orbits[:,:] = mhv_orbs

    # Add attributes to NetCDF
    mhv.groups['centerlines'].createDimension('orbit', 200)
    mhv.groups['centerlines'].createVariable('lakeflag', 'i4', ('num_points',))
    mhv.groups['centerlines'].createVariable('deltaflag', 'i4', ('num_points',))
    mhv.groups['centerlines'].createVariable('grand_id', 'i4', ('num_points',))
    mhv.groups['centerlines'].createVariable('grod_id', 'i4', ('num_points',))
    mhv.groups['centerlines'].createVariable('grod_fid', 'i8', ('num_points',))
    mhv.groups['centerlines'].createVariable('hfalls_fid', 'i8', ('num_points',))
    mhv.groups['centerlines'].createVariable('basin_code', 'i4', ('num_points',))
    mhv.groups['centerlines'].createVariable('number_obs', 'i4', ('num_points',))
    mhv.groups['centerlines'].createVariable('orbits', 'i4', ('num_points','orbit'))
    mhv.groups['centerlines'].createVariable('lake_id', 'i8', ('num_points',))

    mhv.groups['centerlines'].variables['lakeflag'][:] = mhv_lakeflag
    mhv.groups['centerlines'].variables['deltaflag'][:] = mhv_deltaflag
    mhv.groups['centerlines'].variables['grand_id'][:] = mhv_grand_id
    mhv.groups['centerlines'].variables['grod_id'][:] = mhv_grod_id
    mhv.groups['centerlines'].variables['grod_fid'][:] = mhv_grod_fid
    mhv.groups['centerlines'].variables['hfalls_fid'][:] = mhv_hfalls_fid
    mhv.groups['centerlines'].variables['basin_code'][:] = mhv_basin_code 
    mhv.groups['centerlines'].variables['number_obs'][:] = mhv_number_obs
    mhv.groups['centerlines'].variables['orbits'][:,:] = mhv_orbits
    mhv.groups['centerlines'].variables['lake_id'][:] = mhv_lake_id
    mhv.close()

    end = time.time()
    print('Finished Basin ' + str(mhv_files[ind]) + ' in: ' + str(np.round((end-start)/60, 2)) + ' min')

end_all = time.time()
print('Finished '+region+' in: ' + str(np.round((end_all-start_all)/60, 2)) + ' min')



