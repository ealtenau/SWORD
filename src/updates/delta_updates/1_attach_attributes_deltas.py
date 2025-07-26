# -*- coding: utf-8 -*-
"""
Attaching Auxillary Attributes to Delta Files
(1_attach_attributes_deltas.py)
===================================================

This script attaches auxially dataset attributes, 
needed to form the SWOT River Database (SWORD), to 
provided delta shapefiles.

Output is a netCDF file with all necessary geospatial 
and hydrologic attributes for the provided delta. 
NetCDF files are located at:
'/data/inputs/Deltas/delta_updates/netcdf/'

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and the directory path 
containing the delta shapefile.

Execution example (terminal):
    python path/to/1_attach_attributes_deltas.py NA path/to/delta_files

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import time
import numpy as np
from scipy import spatial as sp
import glob
import geopandas as gp
from shapely.geometry import Point
import pandas as pd
import argparse
import src.updates.delta_updates.delta_utils as dlt
import src.updates.auxillary_utils as aux 
import src.updates.geo_utils as geo 

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("dir", help="directory to delta files", type = str)
args = parser.parse_args()

region = args.region
delta_dir = args.dir

#pull shapefiles. 
delta_files = np.sort(np.array([file for file in geo.getListOfFiles(delta_dir) if '.shp' in file]))
rch_file = delta_files[1]
node_file = delta_files[0]
#generate outfile.
outfile = main_dir + '/data/inputs/Deltas/delta_updates/netcdf/' + os.path.basename(rch_file)[0:-4] + '.nc' 
outfile = outfile.replace("reaches","delta")

#global paths.
fn_grand = main_dir + '/data/inputs/GRAND/GRanD_dams_v1_1.shp'
fn_grod = main_dir + '/data/inputs/GROD/v1.1/GROD_v1.1_coledit.csv'
track_dir = main_dir + '/data/inputs/SWOT_Tracks/2020_orbits/'
track_list = glob.glob(os.path.join(track_dir, 'ID_PASS*.shp'))

#regional paths.
fn_basins = main_dir + '/data/inputs/HydroBASINS/' + region + '/' + region + '_hb_lev08.shp'
lake_dir = main_dir + '/data/inputs/LakeDatabase/20200702_PLD/For_Merge/' + region + '/' #need new PLD!!!
lake_path = np.array(np.array([file for file in geo.getListOfFiles(lake_dir) if '.shp' in file]))[0]
mh_elv_dir = main_dir+'/data/inputs/MERIT_Hydro/'+region+'/elv/'
mh_facc_dir = main_dir+'/data/inputs/MERIT_Hydro/'+region+'/upa/'
mh_wth_dir = main_dir+'/data/inputs/MERIT_Hydro/'+region+'/wth/'
facc_paths = np.sort(np.array([file for file in geo.getListOfFiles(mh_facc_dir) if '.tif' in file]))
elv_paths = np.sort(np.array([file for file in geo.getListOfFiles(mh_elv_dir) if '.tif' in file]))
wth_paths = np.sort(np.array([file for file in geo.getListOfFiles(mh_wth_dir) if '.tif' in file]))

#read auxillary vector files. 
lake_df = gp.read_file(lake_path)
basin_df = gp.read_file(fn_basins)

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('STARTING DELTA:', os.path.basename(rch_file))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

#reading original delta vector files. 
print('Reading and Formatting Delta Files')
delta_rchs = gp.read_file(rch_file)
delta_nodes = gp.read_file(node_file)

#converting polylines to points. 
delta_pts = geo.lines_to_pts(delta_rchs) #delta_pts.__dict__.keys()
#adding file name to object.
delta_pts.file = os.path.basename(rch_file)  
#reverse indexes (downstream to upstream).
delta_pts.index = dlt.reverse_indexes(delta_pts.reach_id_R, 
                                      delta_pts.index)
#defining delta extent.
delta_ext = [np.min(delta_pts.x), np.min(delta_pts.y),
           np.max(delta_pts.x), np.max(delta_pts.y)]
#creating points geodataframe for spatial joins. 
delta_pts_df = geo.pts_to_geodf(delta_pts.x, delta_pts.y)

#vector intersections. 
print('Performing Vector Intersections')
#subset Lake db to mhv extent.
lake_df_clip = lake_df.cx[delta_ext[0]:delta_ext[2], 
                          delta_ext[1]:delta_ext[3]]

#attach prior lake database (PLD) IDs.
delta_pts.lakeid = geo.vector_to_vector_intersect(delta_pts_df, 
                                                  lake_df_clip, 
                                                  'lake_id') #need to update with new PLD.
#turn into binary flag. 
delta_pts.lakeid[np.where(delta_pts.lakeid > 0)] = 1

#adding dam, basin, and SWOT track information.
#grod and grand. 
delta_pts.grand,\
    delta_pts.obstr_type,\
        delta_pts.grod_id,\
            delta_pts.hfalls_id = aux.add_dams(delta_pts.x, 
                                               delta_pts.y, 
                                               fn_grand, 
                                               fn_grod)
#get list of overlapping swot passes.
track_files = geo.pt_vector_overlap(delta_pts.x, 
                                    delta_pts.y, 
                                    track_list)
#swot orbit-pass information. 
delta_pts.swot_obs, \
    delta_pts.swot_orbit = aux.add_swot_tracks(delta_pts_df,
                                         track_files)
#intersecting hydrobasin pfafstetter codes. 
delta_pts.basins = geo.vector_to_vector_join_nearest(delta_pts_df, 
                                                     basin_df,
                                                     True, 
                                                     'PFAF_ID')
#filtering basin codes. 
delta_pts.basins_filt = aux.filter_basin_codes(delta_pts.reach_id_R, 
                                               delta_pts.basins)
#stopping process if there are zero basin values. 
if min(delta_pts.basins_filt) == 0:
    print('!!! Zero Basin Codes Exist !!!')
    sys.exit()

#raster intersections. 
print('Performing MERIT Hydro Intersections')
#getting overlapping MERIT Hydro tiles. 
elv_subset = geo.pt_raster_overlap(delta_pts.x, delta_pts.y, elv_paths)
#attaching MERIT Hydro data to delta. 
delta_pts.wse, \
    delta_pts.mh_wth, \
        delta_pts.facc, \
            delta_pts.mh_tile = aux.attach_mh(elv_paths, 
                                              wth_paths, 
                                              facc_paths, 
                                              elv_subset, 
                                              delta_pts.x, 
                                              delta_pts.y)

#calculate other necessary attributes for reach definition. 
print('Calulating Remaining Attributes')
#utm coordinates
delta_pts.east, delta_pts.north, __, __ = geo.reproject_utm(delta_pts.y, 
                                                            delta_pts.x)
#distance along reaches. 
delta_pts.dist, delta_pts.len = aux.calc_geodesic_dist(delta_pts.x, 
                                                       delta_pts.y, 
                                                       delta_pts.reach_id_R, 
                                                       delta_pts.index)

#attaching width from node file. 
node_arr = np.vstack((np.array(delta_nodes.x), np.array(delta_nodes.y))).T
pt_arr = np.vstack((delta_pts.x,delta_pts.y)).T
kdt = sp.cKDTree(node_arr)
pt_dist, pt_ind = kdt.query(pt_arr, k = 1)
node_wth = np.array(delta_nodes.width)
node_wth_var = np.array(delta_nodes.width_var)
node_wth_max = np.array(delta_nodes.max_width)
node_id = np.array(delta_nodes.node_id_rg)
node_sinu = np.array(delta_nodes.sinuosity)
delta_pts.width = node_wth[pt_ind]
delta_pts.wth_var = node_wth_var[pt_ind]
delta_pts.max_width = node_wth_max[pt_ind]
delta_pts.node_id = node_id[pt_ind]
delta_pts.sinuosity = node_sinu[pt_ind]

#unique centerline ids. 
delta_pts.cl_id = aux.unique_cl_id(delta_pts.reach_id_R, delta_pts.index)

#endpoints. 
delta_pts.ends = aux.assign_endpoints(delta_pts.reach_id_R, delta_pts.index)

#filler variables
delta_pts.lakeflag = np.repeat(0, len(delta_pts.x))
delta_pts.deltaflag = np.repeat(1, len(delta_pts.x))
delta_pts.nchan = np.repeat(1, len(delta_pts.x))
delta_pts.manual_add = np.repeat(1, len(delta_pts.x))

#format topology variables
delta_pts.fmt_rch_id_up, \
    delta_pts.fmt_rch_id_dn = dlt.format_topo_attributes(delta_pts.rch_id_up, 
                                                         delta_pts.rch_id_dn, 
                                                         delta_pts.n_rch_up, 
                                                         delta_pts.n_rch_down)

#write data. 
print('Writing NetCDF File')
dlt.write_data_nc(delta_pts, outfile)

end = time.time()
print('Finished Delta in:', str(np.round((end-start)/60, 2)) + ' min')

##################################################################
##################################################################
##################################################################
###PLOTS
# import matplotlib.pyplot as plt

# plt.scatter(delta_pts.x, delta_pts.y, c=delta_pts.index, s=3, cmap='rainbow')
# plt.show()

# plt.scatter(delta_pts.x, delta_pts.y, c=delta_pts.dist, s=3, cmap='rainbow')
# plt.show()

# plt.scatter(delta_pts.x, delta_pts.y, c=delta_pts.basins_filt, s=3, cmap='rainbow')
# plt.show()

# plt.scatter(delta_pts.x, delta_pts.y, c=delta_pts.wse, s=3, cmap='rainbow')
# plt.show()

# plt.scatter(delta_pts.x, delta_pts.y, c=delta_pts.facc, s=3, cmap='rainbow')
# plt.show()

# rch = np.where(delta_pts==27)[0]
# plt.plot(delta_pts.x[rch], delta_pts.y[rch])
# plt.show()