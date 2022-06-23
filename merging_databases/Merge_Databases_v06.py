# -*- coding: utf-8 -*-

"""
-------------------------------------------------------------------------------
                    Merging River & Water Body Datasets
-------------------------------------------------------------------------------
Copyright (c) 2018-2021 UNC Chapel Hill. All rights reserved.
Created by E. Altenau. Some functions were taken or modified from C. Lion's
"Tools.py" script (c) 2016.

DESCRIPTON:
    This is the main script used to merge information from multiple datasets
    with the GRWL river centerlines. Merged outputs will be used as inputs to
    define reaches and nodes in the SWOT River Database (SWORD). The
    various datasets merged together are as follows:

    GRWL (Global River Widths from Landsat): Contains river centerline
        locations, width, number of channels, and water body type information
        for rivers 30 m wide and greater [Allen and Pavelsky, 2018].
    MERIT Hydro: Provides hyrologically-corrected elevation values and flow
        accumulation for rivers derived from the MERIT DEM
        [Yamazaki et al., 2019].
    HydroBASINS: Contains basin boundary locations and Pfafstetter basin codes
        for each continent [Lehner and Grill, 2013].
    GRanD (Global Reservior and Dam Database): Provides locations of the worlds
        largest dams [Lehner et al., 2011].
    GROD (Global River Obstruction Database): Provides locations and structure
        type for all dams along the GRWL river network [under development, UNC].
    Global Deltas: Provides the spatial extent of 48 of the world’s largest
        deltas [Tessler et al., 2015].
    SWOT Tracks: Provides SWOT orbit track numbers and days of overpass.
    Prior Lake Database: Provides lake extents derived from Landsat.

INPUTS:
    fn_basins -- HydroBASINS file directory(.shp)
    fn_grand -- GRanD file directory(.shp)
    fn_grod -- GROD file directory (.csv)
    fn_deltas -- Global Deltas directory (.shp)
    track_dir -- Directory to all SWOT Track polygon files (.shp)
    grwl_dir -- Directory to all GRWL files (.shp)
    mh_facc_dir -- Directory to all MERIT Hydro flow accumulation files (.tif)
    mh_elv_dir -- Directory to all MERIT Hydro elevation files (.tif)
    out_dir -- Directory to put the outputs.
    region -- Continent directory name (ex: "NA")
    fn_merge -- Filename of the mosaicked output file.
    lake_dir -- Prior Lake Database file directory (.shp)

OUTPUTS:
    Individual Shapefiles -- Shapefile containing all the merged
        attributes along the GRWL centerline for each GRWL tile extent. The
        filename will contain the GRWL tile location and modifier "_merge"
        (ex: n60w150_merge.shp).
    Mosaicked Shapefile -- Shapefile containing all the merged attributes
        along the GRWL centerlines for an entire continent. The finalname will
        contain the "_version" modifier (ex: NA_Merge_v01.shp).
    Mosaicked NetCDF -- NetCDF file containing all the merged attributes
        along the GRWL centerlines for an entire continent. The final filename
        will contain the "_version" modifier (ex: NA_Merge_v01.nc).

VERSION UPDATES since v05:
    - Relocated smoothing the centerlines back to GRWL pre-processing script:
        "GRWL_Updates_v04.py"
-------------------------------------------------------------------------------
"""

from __future__ import division
import os
os.chdir('C:/Users/ealtenau/Documents/Research/SWAG/For_Server/scripts/merging_databases/') #path to scripts will need updating. 
import Merge_Tools_v06 as mgt
import time
import numpy as np
from scipy import spatial as sp
import glob
#import utm
import geopandas as gp

###############################################################################
###############################    Inputs    ##################################
###############################################################################

start_all = time.time()

#Define input directories and filenames. Will need to be changed based on user needs.
region = 'EU'
fn_merge = region + '_Merge_v10.shp'
out_dir = '../../outputs/Merged_Data/'+ region + '/'

# Global Paths.
fn_grand = '../../inputs/GRAND/GRanD_dams_v1_1.shp'
fn_deltas = '../../inputs/Deltas/global_map.shp'
track_dir = '../../inputs/SWOT_Tracks/2020_orbits/'
track_list = glob.glob(os.path.join(track_dir, 'ID_PASS*.shp'))

# Regional Paths.
fn_grod = '../../inputs/GROD/GROD_'+region+'.csv'
fn_basins = '../../inputs/HydroBASINS/' + region + '/hybas_eu_lev08_v1c.shp'
grwl_dir = '../../inputs/GRWL/Updates/' + region + '/'
mh_facc_dir = '../../inputs/MERIT_Hydro/' + region + '/upa/'
mh_elv_dir = '../../inputs/MERIT_Hydro/' + region + '/elv/'
grwl_paths = [file for file in mgt.getListOfFiles(grwl_dir) if '.shp' in file]
facc_paths = [file for file in mgt.getListOfFiles(mh_facc_dir) if '.tif' in file]
elv_paths = [file for file in mgt.getListOfFiles(mh_elv_dir) if '.tif' in file]
lake_dir = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/inputs/LakeDatabase/20200702_PLD/For_Merge/PLD_AS_SI_EU_AU.shp'

# open global shapefiles for spatial intersections. 
lake_db = gp.GeoDataFrame.from_file(lake_dir)
delta_db = gp.GeoDataFrame.from_file(fn_deltas)

###############################################################################
############################    Merging Data    ###############################
###############################################################################

#Lena, Ob, and Amazon tiles that require a different flow accumulation threshold
#value (line 173).
flagged_tiles = np.array(['n56e078', 'n56e084', 'n60e060', 'n60e066', 'n60e072', 'n60e126',
                          'n60e132', 'n64e060', 'n64e066', 'n64e072', 'n64e078', 'n64e120',
                          'n64e126', 'n68e120', 'n68e126', 'n72e120', 'n72e126', 's04w054',
                          's04w060', 's04w066', 's04w072', 's08w060', 's08w066'])

merged = mgt.Object()
cnt = 0
for ind in list(range(len(grwl_paths))):

    start = time.time()

    # Read in and format GRWL. "mgt.xxx" functions called from "Merge_Tools_vXX.py".
    fn_grwl = grwl_paths[ind]
    outpath =  out_dir + fn_grwl[-16:-9] + '_merge.shp'
    grwl = mgt.open_grwl(fn_grwl)
    grwl.lon[np.where(grwl.lon < -180)] = -180.0
    grwl.lon[np.where(grwl.lon > 180)] = 180.0
            
    # Condition to skip file if it does not contain any data.
    if len(grwl.seg) == 0:
        print(fn_grwl[-16:-9] + ": No GRWL Data - Skipped")
        continue
    
    # Creating GRWL tile name array.
    tile =  fn_grwl[-16:-9]
    grwl.tile = np.repeat(tile, len(grwl.seg))

    # Defining GRWL tile extent.
    grwl_ext = [np.min(grwl.lon), np.min(grwl.lat),
                np.max(grwl.lon), np.max(grwl.lat)]
    
    # Finding overlapping MERIT Hydro (mh) tiles with GRWL.
    overlap_ids = mgt.find_MH_tiles(grwl, facc_paths)
    fn_mh_facc = map(lambda i: facc_paths[i], overlap_ids)
    fn_mh_elv = map(lambda i: elv_paths[i], overlap_ids)

    # Reading in mh data.
    mhydro = mgt.MH_coords(fn_mh_facc, grwl_ext)
    mhydro.facc = mgt.MH_vals(fn_mh_facc, grwl_ext)
    mhydro.elv = mgt.MH_vals(fn_mh_elv, grwl_ext)

    # Threshold determines extent of mh rivers.
    if tile in flagged_tiles: #used a slightly larger thresh for large anabranching rivers (Amazon, Ob, Lena).
        thresh = 100
    else:
        thresh = 10 #reduced from 500 originally.

    idx = np.where(mhydro.facc > thresh)

    # Pulling only the mh river locations.
    mh = mgt.Object()
    mh.lon = mhydro.lon[idx]
    mh.lat = mhydro.lat[idx]
    mh.elv = mhydro.elv[idx]
    mh.facc = mhydro.facc[idx]

    # Re-project mh coordinates into UTM.
    mh.x, mh.y,__,__ = mgt.find_projection(mh.lat, mh.lon)

    ### Finding closest mh points to GRWL and extracting the mh values.
    grwl_pts = np.vstack((grwl.x, grwl.y)).T
    mh_pts = np.vstack((mh.x, mh.y)).T
    kdt = sp.cKDTree(mh_pts)
    pt_dist, pt_ind = kdt.query(grwl_pts, k = 20)
    grwl.elv = np.median(mh.elv[pt_ind[:]], axis = 1)
    grwl.facc = np.median(mh.facc[pt_ind[:]], axis = 1)
    nn_dist = pt_dist[:,1] #only taking shortest distance.

    # Condition that pulls the larget flow accumulation value for minimum
    # distances greater than 100. This is useful in cases where grwl is
    # in between two channels due to the higher density of mh.
    dist_id = np.where(nn_dist > 100)
    facc_max = np.max(mh.facc[pt_ind[:]], axis = 1)
    grwl.facc[dist_id] = facc_max[dist_id]

    # Filtering the flow accumulation values.
    grwl.facc_filt = mgt.filter_facc(grwl)

    # Calculating segment flow distance.
    grwl.dist = mgt.calc_segDist(grwl)

    # Attach Prior Lake Database (PLD) IDs.
    grwl.lake_id = mgt.add_lakedb(grwl, fn_grwl, lake_db)
    grwl.old_lakes = np.copy(grwl.lake)

    if np.max(np.unique(grwl.lake_id)) == 0:
        grwl.new_lakes = np.copy(grwl.lake)
    else:
    # Create new Lake Flag based on PLD information.
        grwl.new_lakes = np.zeros(len(grwl.seg))
        lakes = np.where(grwl.lake_id > 0)[0]
        tidal = np.where(grwl.lake == 3)[0]
        canal = np.where(grwl.lake == 2)[0]
        grwl.new_lakes[lakes] = 1
        grwl.new_lakes[tidal] = 3
        grwl.new_lakes[canal] = 2

    # Adding dam, basin, delta, and SWOT track information.
    grwl.grand, grwl.grod, grwl.grod_fid, grwl.hfalls_fid = mgt.add_dams(grwl, fn_grand, fn_grod)
    grwl.basins = mgt.add_basins(grwl, fn_grwl, fn_basins)
    grwl.delta = mgt.add_deltas(grwl, fn_grwl, delta_db)
    grwl.old_deltas = np.copy(grwl.delta)
    track_files = mgt.overlapping_tracks(grwl, track_list)
    grwl.num_obs, grwl.orbits = mgt.add_tracks(grwl, fn_grwl, track_files)

    # Filling in segment gaps with no basin values.
    mgt.fill_zero_basins(grwl)

    # Combining current data with previous data.
    mgt.combine_vals(merged, grwl, cnt)
    cnt = cnt+1

    # Writing individual shapefiles.
    mgt.save_merge_shp(grwl, outpath)
    end = time.time()
    print(ind, len(track_files), 'Runtime: ' + str((end-start)/60) + ' min: ' + outpath)

###############################################################################
###################### Combining Individual Tiles #############################
###############################################################################

merge_outfile = out_dir + fn_merge
shp_file = merge_outfile
nc_file = shp_file[:-4] + '.nc'

# Filter Data.
start = time.time()
mgt.format_data(merged)
#merged.lake_id = merged.lake_id.astype(long)
end = time.time()
print('Time to Filter Combined Data: ' + str((end-start)/60) + ' min')

# Save filtered data as netcdf file.
start = time.time()
mgt.save_merged_nc(merged, nc_file)
end = time.time()
print('Time to Write NetCDF: ' + str((end-start)/60) + ' min')

'''
# Save continental scale data as shp file. This section is commented out
# because it takes a long time. If a user thinks it would be helpful they
# can uncomment this section. 
start = time.time()
mgt.save_filtered_shp(merged, shp_file)
end = time.time()
print('Time to Write Shp: ' + str((end-start)/60) + ' min: ' + merge_outfile)
'''

end_all = time.time()
print('Total Runtime: ' + str((end_all-start_all)/60) + ' min: ' + nc_file)
