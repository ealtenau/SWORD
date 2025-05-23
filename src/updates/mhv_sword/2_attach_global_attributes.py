from __future__ import division
import os
main_dir = os.getcwd()
import time
import numpy as np
from scipy import spatial as sp
import glob
import geopandas as gp
import netCDF4 as nc
from shapely.geometry import Point
import pandas as pd
from osgeo import ogr
import argparse
# import matplotlib.pyplot as plt

###############################################################################

def getListOfFiles(dirName):

    """
    FUNCTION:
        For the given path, gets a recursive list of all files in the directory tree.

    INPUTS
        dirName -- Input directory

    OUTPUTS
        allFiles -- list of files under directory
    """

    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

###############################################################################

def add_lakedb(mhv_df, lake_db):

    # Attaching PLD IDs
    points = mhv_df
    intersect = gp.sjoin(lake_db, points, how="inner")
    intersect = pd.DataFrame(intersect)
    intersect = intersect.drop_duplicates(subset='index_right', keep='first')

    # Creating arrays.
    ids = np.array(intersect.index_right)
    lb_code = np.zeros(len(mhv_df))
    lb_code[ids] = np.array(intersect.lake_id)

    # Assigning basin locations to GRWL object attributes.
    return lb_code

###############################################################################

def open_grand_shpfile(filename):

    driver = ogr.GetDriverByName('ESRI Shapefile')

    shape = driver.Open(filename)
    layer = shape.GetLayer()
    numfeat = layer.GetFeatureCount()

    lon = np.zeros(numfeat)
    lat = np.zeros(numfeat)
    grand_id = np.zeros(numfeat)
    catch_skm = np.zeros(numfeat)

    cnt = 0
    for feature in layer:
        geom = feature.GetGeometryRef()
        lon[cnt] = geom.GetX()
        lat[cnt] = geom.GetY()
        grand_id[cnt] = feature.GetField('GRAND_ID')
        catch_skm[cnt] = feature.GetField('CATCH_SKM')
        cnt += 1
        feature.Destroy()

    del driver, shape, numfeat, layer

    return lat, lon, grand_id, catch_skm

###############################################################################

def add_dams(mhv_lon, mhv_lat, fn_grand, fn_grod):

    # Read in GRaND and GROD data.
    grand_lat, grand_lon, grand_id, grand_skm = open_grand_shpfile(fn_grand)
    grand_lat = abs(grand_lat)
    grod_info = pd.read_csv(fn_grod)
    grod_lat =  np.array(grod_info.lat)
    grod_lon =  np.array(grod_info.lon)
    grod_names = np.array(grod_info.name) #.flatten() #grod_names = np.array(grod_info[[0]]).flatten()
    grod_id = np.zeros(len(grod_names))
    grod_fid = np.array(grod_info.fid)#.flatten()

    # Remove NaN values from arrays.
    remove = np.isnan(grod_lat)
    delete = np.where(remove == True)[0]
    grod_lat = np.delete(grod_lat, delete)
    grod_lon = np.delete(grod_lon, delete)
    grod_names = np.delete(grod_names, delete)
    grod_fid = np.delete(grod_fid, delete)
    grod_id = np.delete(grod_id, delete)

    # Assign numbers to GROD dam types.
    grod_id[np.where(grod_names == 'Dam')] = 1
    grod_id[np.where(grod_names == 'Locks')] = 2
    grod_id[np.where(grod_names == 'Low_Permeable_Dams')] = 3
    grod_id[np.where(grod_names == 'Waterfall')] = 4
    grod_id[np.where(grod_names == 'Partial_Dams_gte50')] = 5
    grod_id[np.where(grod_names == 'Partial_Dams_lt50')] = 6
    grod_id[np.where(grod_names == 'Channel_Dams')] = 7 #was 2 before excluding.

    # narrowing down points in GRWL bounding box.
    pts = np.array([grod_lon, grod_lat]).T
    ll = np.array([np.min(mhv_lon), np.min(mhv_lat)])  # lower-left
    ur = np.array([np.max(mhv_lon), np.max(mhv_lat)])  # upper-right

    inidx = np.all(np.logical_and(ll <= pts, pts <= ur), axis=1)
    inbox = pts[inidx]
    grod_id_clip = grod_id[inidx]
    grod_fid_clip = grod_fid[inidx]
    grod_lon_clip = inbox[:,0]
    grod_lat_clip = inbox[:,1]

    # Attach dam locations to GRWL.
    mhv_pts = np.vstack((mhv_lon, mhv_lat)).T
    grand_pts = np.vstack((grand_lon, grand_lat)).T
    kdt = sp.cKDTree(mhv_pts)
    grand_dist, grand_idx = kdt.query(grand_pts, k = 1)
    # Narrowing dam locations by distance threshold.
    grand_dist_thresh = np.where(grand_dist <= 0.01)[0]
    grand_locs, grand_locs_idx = np.unique(grand_idx[grand_dist_thresh], return_index=True)
    # Creating arrays.
    grand_ID = np.zeros(len(mhv_lon))
    # Assigning values.
    grand_ID[grand_locs] = grand_id[grand_dist_thresh[grand_locs_idx]]


    if len(grod_lon_clip) == 0:
        # Creating arrays.
        grod_ID = np.zeros(len(mhv_lon))
        grod_FID = np.zeros(len(mhv_lon))
        hfalls_FID = np.zeros(len(mhv_lon))
    else:
        # Attach dam locations to GRWL.
        grod_pts = np.vstack((grod_lon_clip, grod_lat_clip)).T
        kdt = sp.cKDTree(mhv_pts)
        grod_dist, grod_idx = kdt.query(grod_pts, k = 1)

        # Narrowing dam locations by distance threshold.
        grod_dist_thresh = np.where(grod_dist <= 0.02)[0] #was 100m
        grod_locs, grod_locs_idx = np.unique(grod_idx[grod_dist_thresh], return_index=True)

        # Creating arrays.
        grod_ID = np.zeros(len(mhv_lon))
        grod_FID = np.zeros(len(mhv_lon))
        hfalls_FID = np.zeros(len(mhv_lon))
        # Assigning values.
        grod_ID[grod_locs] = grod_id_clip[grod_dist_thresh[grod_locs_idx]]
        grod_FID[grod_locs] = grod_fid_clip[grod_dist_thresh[grod_locs_idx]]
        hfalls_FID[grod_locs] = grod_fid_clip[grod_dist_thresh[grod_locs_idx]]
        grod_FID[np.where(grod_ID == 4)] = 0
        hfalls_FID[np.where(grod_ID != 4)] = 0

    return grand_ID, grod_ID, grod_FID, hfalls_FID

###############################################################################

def add_basins(mhv_df, fn_basins):

    # Attaching basin codes
    points = mhv_df
    poly = gp.GeoDataFrame.from_file(fn_basins)
    intersect = gp.sjoin(poly, points, how="inner")
    intersect = pd.DataFrame(intersect)
    intersect = intersect.drop_duplicates(subset='index_right', keep='first')

    # Creating arrays.
    ids = np.array(intersect.index_right)
    basin_code = np.zeros(len(mhv_df)) #see if len(points) works...
    basin_code[ids] = np.array(intersect.PFAF_ID)

    # Assigning basin locations to GRWL object attributes.
    return basin_code

###############################################################################

def add_deltas(mhv_df, mhv_seg, delta_db):

    # Finding where delta shapefiles intersect the GRWL shapefile.
    points = mhv_df
    poly = delta_db
    intersect = gp.sjoin(poly, points, how="inner")
    intersect = pd.DataFrame(intersect)
    intersect = intersect.drop_duplicates(subset='index_right', keep='first')

    # Identifying the delta ID.
    ids = np.array(intersect.index_right)
    delta_flag = np.zeros(len(mhv_df))
    delta_flag[ids] = np.array(intersect.DeltaID)

    # Filtering GRWL lake flag and new delta flag information to create final
    # coastal flag.
    uniq_segs = np.unique(mhv_seg)
    coastal_flag = np.zeros(len(mhv_seg))
    for ind in list(range(len(uniq_segs))):
        seg = np.where(mhv_seg == uniq_segs[ind])[0]
        flag = (len(np.where(delta_flag[seg] > 0)[0])/len(seg))*100
        if flag > 25:
            coastal_flag[seg] = 1

    return coastal_flag

###############################################################################

def overlapping_tracks(mhv_lon, mhv_lat, track_list):

    #define grwl extent as ogr geometry format.
    poly1 = ogr.Geometry(ogr.wkbLinearRing)
    poly1.AddPoint(min(mhv_lon), max(mhv_lat))
    poly1.AddPoint(min(mhv_lon), min(mhv_lat))
    poly1.AddPoint(max(mhv_lon), min(mhv_lat))
    poly1.AddPoint(max(mhv_lon), max(mhv_lat))
    poly1.AddPoint(min(mhv_lon), max(mhv_lat))
    mhvGeometry = ogr.Geometry(ogr.wkbPolygon)
    mhvGeometry.AddGeometry(poly1)

    #find overlapping SWOT tracks.
    track_files = []
    for fn in track_list:
        driver = ogr.GetDriverByName('ESRI Shapefile')
        shape = driver.Open(fn)
        inLayer = shape.GetLayer()
        #inLayer is always of size one because polygon is a unique value.
        for feature in inLayer:
            track=feature.GetGeometryRef()
            answer = track.Intersects(mhvGeometry)
            if answer == True:
                track_files.append(fn)
    track_files = np.unique(track_files)

    return(track_files)

###############################################################################

def add_tracks(mhv_df, list_files):

    # Reading in GRWL point information.
    points = mhv_df

    # Creating empty arrays to fill.
    orbit_array = np.zeros((len(points), 200), dtype=int)

    row = 0
    for ind in list(range(len(list_files))):
        #print ind
        poly = gp.GeoDataFrame.from_file(list_files[ind])
        intersect = gp.sjoin(poly, points, how="inner")
        intersect = pd.DataFrame(intersect)
        intersect = intersect.drop_duplicates(subset='index_right', keep='first')
        ids = np.array(intersect.index_right)
        if len(ids) == 0:
            continue 
        else:
            orbit_array[ids, row] = intersect.ID_PASS
            row = row+1

    # Finding number of SWOT observations.
    orbit_binary = np.copy(orbit_array)
    orbit_binary[np.where(orbit_binary > 0)] = 1
    num_obs = np.sum(orbit_binary, axis = 1)

    # Assigning SWOT track information to new object attributes.
    return num_obs, orbit_array

###############################################################################

def fill_zero_basins(mhv_lon_clip, mhv_lat_clip, mhv_seg_clip, mhv_basins):

    # Loop though each GRWL segment and fill in points where the basin code = 0.
    new_basins = np.copy(mhv_basins)
    zero_pts = np.where(mhv_basins == 0)[0]
    uniq_segs = np.unique(mhv_seg_clip[zero_pts])
    for ind in list(range(len(uniq_segs))):
        seg = np.where(mhv_seg_clip == uniq_segs[ind])[0]
        zpts = np.where(mhv_basins[seg] == 0)[0]
        vpts = np.where(mhv_basins[seg] > 0)[0]

        if len(zpts) == 0:
            continue

        if len(vpts) == 0:
            # if there are no points within the segment greater than 0;
            # use all grwl tile points with basin values > 0.
            vpts = np.where(mhv_basins > 0)[0]

            if len(vpts) == 0:
                continue

            #find closest neighbor for all points.
            z_pts = np.vstack((mhv_lon_clip[seg[zpts]], mhv_lat_clip[seg[zpts]])).T
            v_pts = np.vstack((mhv_lon_clip[vpts], mhv_lat_clip[vpts])).T
            kdt = sp.cKDTree(v_pts)
            eps_dist, eps_ind = kdt.query(z_pts, k = 25) #was 25 when using grwl. 

            min_dist = np.min(eps_dist)
            if min_dist > 0.1: #1000 when using meters. 
                continue

            #calculate mode of closest basin values.
            close_basins = mhv_basins[vpts[eps_ind]].flatten()
            basin_mode = max(set(list(close_basins)), key=list(close_basins).count)
            #assign zero basin values the mode value.
            new_basins[seg[zpts]] = np.repeat(basin_mode, len(zpts))

        else:
            #find closest neighbor for all points.
            z_pts = np.vstack((mhv_lon_clip[seg[zpts]], mhv_lat_clip[seg[zpts]])).T
            v_pts = np.vstack((mhv_lon_clip[seg[vpts]], mhv_lat_clip[seg[vpts]])).T
            kdt = sp.cKDTree(v_pts)
            __, eps_ind = kdt.query(z_pts, k = 25) #was 25 when using grwl. 
            eps_ind = np.unique(eps_ind)
            rmv = np.where(eps_ind == len(vpts))[0] 
            eps_ind = np.delete(eps_ind, rmv)
            
            #calculate mode of closest basin values.
            if len(vpts) < len(zpts):
                close_basins = mhv_basins[seg[vpts]].flatten()
            else:
                close_basins = mhv_basins[seg[vpts[eps_ind]]].flatten()
            basin_mode = max(set(list(close_basins)), key=list(close_basins).count)
            #basin_mode = max(grwl.basins[seg])
            #assign zero basin values the mode value.
            new_basins[seg[zpts]] = np.repeat(basin_mode, len(zpts))
    
    return new_basins

###############################################################################
##############################    MAIN CODE    ################################
###############################################################################

start_all = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Region", type = str)
args = parser.parse_args()

region = args.region
# region = 'NA'

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
lake_path = np.array(np.array([file for file in getListOfFiles(lake_dir) if '.shp' in file]))

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
    mhv_lakeid = add_lakedb(mhv_df, lake_db_clip)
    mhv_lakes = np.zeros(len(mhv_lakeid))
    mhv_lakes[np.where(mhv_lakeid > 0)] = 1

    # Adding dam, basin, delta, and SWOT track information.
    mhv_grand, mhv_grod, mhv_grodfid, mhv_hfallsfid = add_dams(mhv_lon, mhv_lat, fn_grand, fn_grod)
    mhv_basins = add_basins(mhv_df, fn_basins)
    mhv_deltas = add_deltas(mhv_df, mhv_seg, delta_db)
    track_files = overlapping_tracks(mhv_lon, mhv_lat, track_list)
    mhv_numobs, mhv_orbs = add_tracks(mhv_df, track_files)

    # Filling in segment gaps with no basin values.
    mhv_new_basins = fill_zero_basins(mhv_lon, mhv_lat, mhv_seg, mhv_basins)

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

# Add attributes to NetCDF
# mhv.groups['centerlines'].createDimension('orbit', 200)
# mhv.groups['centerlines'].createVariable('lakeflag', 'i4', ('num_points',))
# mhv.groups['centerlines'].createVariable('deltaflag', 'i4', ('num_points',))
# mhv.groups['centerlines'].createVariable('grand_id', 'i4', ('num_points',))
# mhv.groups['centerlines'].createVariable('grod_id', 'i4', ('num_points',))
# mhv.groups['centerlines'].createVariable('grod_fid', 'i8', ('num_points',))
# mhv.groups['centerlines'].createVariable('hfalls_fid', 'i8', ('num_points',))
# mhv.groups['centerlines'].createVariable('basin_code', 'i4', ('num_points',))
# mhv.groups['centerlines'].createVariable('number_obs', 'i4', ('num_points',))
# mhv.groups['centerlines'].createVariable('orbits', 'i4', ('num_points','orbit'))
# mhv.groups['centerlines'].createVariable('lake_id', 'i8', ('num_points',))

# mhv.groups['centerlines'].variables['lakeflag'][:] = mhv_lakeflag
# mhv.groups['centerlines'].variables['deltaflag'][:] = mhv_deltaflag
# mhv.groups['centerlines'].variables['grand_id'][:] = mhv_grand_id
# mhv.groups['centerlines'].variables['grod_id'][:] = mhv_grod_id
# mhv.groups['centerlines'].variables['grod_fid'][:] = mhv_grod_fid
# mhv.groups['centerlines'].variables['hfalls_fid'][:] = mhv_hfalls_fid
# mhv.groups['centerlines'].variables['basin_code'][:] = mhv_basin_code 
# mhv.groups['centerlines'].variables['number_obs'][:] = mhv_number_obs
# mhv.groups['centerlines'].variables['orbits'][:,:] = mhv_orbits
# mhv.groups['centerlines'].variables['lake_id'][:] = mhv_lake_id
# mhv.close()

'''
vals = np.where(mhv_new_basins == 0)[0]
plt.scatter(mhv_lon_clip, mhv_lat_clip, s = 3, c='black')
plt.scatter(mhv_lon_clip[vals], mhv_lat_clip[vals], s = 3, c='deepskyblue')
plt.show()

df = pd.DataFrame(np.array([mhv_lon_clip, mhv_lat_clip, mhv_lakes]).T)

'''

