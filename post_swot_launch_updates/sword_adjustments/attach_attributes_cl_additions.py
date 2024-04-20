from __future__ import division
import os
import time
import numpy as np
from scipy import spatial as sp
import glob
import geopandas as gp
import netCDF4 as nc
from shapely.geometry import Point
import pandas as pd
from osgeo import ogr
import matplotlib.pyplot as plt

###############################################################################

class Object(object):
    """
    FUNCTION:
        Creates class object to assign attributes to.
    """
    pass

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
    grod_names = np.array(grod_info.type) #.flatten() #grod_names = np.array(grod_info[[0]]).flatten()
    grod_id = np.zeros(len(grod_names))
    grod_fid = np.array(grod_info.grod_id)#.flatten()

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
    grod_id[np.where(grod_names == 'Lock')] = 2
    grod_id[np.where(grod_names == 'Low head dam')] = 3
    grod_id[np.where(grod_names == 'Waterfall')] = 4
    grod_id[np.where(grod_names == 'Partial dam 2')] = 5
    grod_id[np.where(grod_names == 'Partial dam 1')] = 6
    grod_id[np.where(grod_names == 'Channel dam')] = 7 #was 2 before excluding.

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

def read_cl_data(cl_dir):
    cl = gp.read_file(cl_dir)
    cl_x = np.array([gm.x for gm in cl.geometry])
    cl_y = np.array([gm.y for gm in cl.geometry])
    cl_seg = np.array(cl['segment'])
    cl_ind = np.array(cl.index)
    # cl_eps = np.array(cl['endpoints'])
    return cl_x, cl_y, cl_seg, cl_ind #, cl_eps

###############################################################################

# def update_segs(cl):
#     unq_segs = np.unique(cl.seg)
#     new_segs = np.copy(cl.seg)
#     count = np.max(unq_segs) + 1
#     for s in list(range(len(unq_segs))):
#         seg = np.where(cl.seg == unq_segs[s])[0]
#         gdf = gp.GeoDataFrame(geometry=gp.points_from_xy(cl.x[seg], cl.y[seg]),crs="EPSG:4326").to_crs("EPSG:3857")
#         diff = gdf.distance(gdf.shift(1)); diff[0] = 0
#         jumps = np.where(diff > 1000)[0]
#         breaks = np.abs(np.diff(cl.ind[seg]))
#         # print(s, np.unique(breaks))
#         if np.max(np.unique(breaks)) > 1:
#             print(s, 'index')
#             brks = np.where(breaks != 1)[0]+1 
#             brks = np.append(0,brks)
#             brks = np.append(brks,len(seg))
#             for b in list(range(len(brks)-1)):              
#                 end_pt = brks[b+1]
#                 new_segs[seg[brks[b]:end_pt]] = count 
#                 count = count+1
#         elif len(jumps) > 0:
#             print(s, 'jump') 
#             jumps = np.append(0,jumps)
#             jumps = np.append(jumps,len(seg))
#             for j in list(range(len(jumps)-1)):              
#                 end_pt = jumps[j+1]
#                 new_segs[seg[jumps[j]:end_pt]] = count 
#                 count = count+1
#         else:
#             continue
#     return new_segs

###############################################################################

def update_segs(cl):
    unq_segs = np.unique(cl.seg)
    new_segs = np.copy(cl.seg)
    count = np.max(unq_segs) + 1
    for s in list(range(len(unq_segs))):
        seg = np.where(cl.seg == unq_segs[s])[0]
        order_ids = seg[np.argsort(cl.ind[seg])]
        gdf = gp.GeoDataFrame(geometry=gp.points_from_xy(cl.x[order_ids], cl.y[order_ids]),crs="EPSG:4326").to_crs("EPSG:3857")
        diff = gdf.distance(gdf.shift(1)); diff[0] = 0
        jumps = np.where(diff > 1000)[0]
        if len(jumps) > 0:
            print(s, 'jump') 
            jumps = np.append(0,jumps)
            jumps = np.append(jumps,len(seg))
            for j in list(range(len(jumps)-1)):              
                end_pt = jumps[j+1]
                new_segs[order_ids[jumps[j]:end_pt]] = count 
                count = count+1
        else:
            continue
    return new_segs

###############################################################################

def order_edits(cl):

    """
    FUNCTION:
        Creates a new 1-D array that contains ordered point indexes
        for each edited centerline segment.

    INPUTS
         edits -- Object containing attributes for the edited centerlines.

    OUTPUTS
        edits_segInd -- Ordered point index for each edited segment.
    """

    # Loop through edited segments.
    cl_Ind = np.zeros(len(cl.seg))
    uniq_segs = np.unique(cl.seg)
    for ind in list(range(len(uniq_segs))):
        seg = np.where(cl.seg == uniq_segs[ind])[0]
        seg_x = cl.x[seg]
        seg_y = cl.y[seg]
        eps = np.where(cl.eps[seg] == 1)[0]

        # If no endpoints are found default to the first index value to start.
        if len(eps) == 0: # condition added on 9/19/19.
            eps = np.array([0])

        cl_Ind[seg[eps[0]]]=1
        idx = eps[0]

        # Order points in a segment starting from the first endpoint.
        count = 2
        while np.min(cl_Ind[seg]) == 0:
            d = np.sqrt((seg_x[idx]-seg_x)**2 + (seg_y[idx]-seg_y)**2)
            dzero = np.where(cl_Ind[seg] == 0)[0]
            next_pt = dzero[np.where(d[dzero] == np.min(d[dzero]))[0]][0]
            cl_Ind[seg[next_pt]] = count
            count = count+1
            idx = next_pt

    return cl_Ind

###############################################################################

def save_mhv_nc(cl, region, outfile):

    # global attributes
    root_grp = nc.Dataset(outfile, 'w', format='NETCDF4')
    root_grp.x_min = np.min(cl.x)
    root_grp.x_max = np.max(cl.x)
    root_grp.y_min = np.min(cl.x)
    root_grp.y_max = np.max(cl.x)
    root_grp.production_date = time.strftime("%d-%b-%Y %H:%M:%S", time.gmtime()) #utc time

    # subgroups
    cl_grp = root_grp.createGroup('centerlines')

    # dimensions
    root_grp.createDimension('ID', 2)
    cl_grp.createDimension('num_points', len(cl.x))
    cl_grp.createDimension('orbit', 200)

    ### variables and units

    # root group variables
    Name = root_grp.createVariable('Name', 'S1', ('ID'))
    Name._Encoding = 'ascii'

    # centerline variables
    x = cl_grp.createVariable('x', 'f8', ('num_points',), fill_value=-9999.)
    x.units = 'degrees east'
    y = cl_grp.createVariable('y', 'f8', ('num_points',), fill_value=-9999.)
    y.units = 'degrees north'
    segID = cl_grp.createVariable('segID', 'i8', ('num_points',), fill_value=-9999.)
    segInd = cl_grp.createVariable('segInd', 'i8', ('num_points',), fill_value=-9999.)
    segID_old = cl_grp.createVariable('segID_old', 'i8', ('num_points',), fill_value=-9999.)
    basin = cl_grp.createVariable('basin_code', 'i8', ('num_points',), fill_value=-9999.)
    lakes = cl_grp.createVariable('lakeflag', 'i4', ('num_points',), fill_value=-9999.)
    deltas = cl_grp.createVariable('deltaflag', 'i4', ('num_points',), fill_value=-9999.)
    grand = cl_grp.createVariable('grand_id', 'i4', ('num_points',), fill_value=-9999.)
    grod = cl_grp.createVariable('grod_id', 'i4', ('num_points',), fill_value=-9999.)
    grod_fid = cl_grp.createVariable('grod_fid', 'i8', ('num_points',), fill_value=-9999.)
    hfall_fid = cl_grp.createVariable('hfalls_fid', 'i8', ('num_points',), fill_value=-9999.)
    swot_obs = cl_grp.createVariable('number_obs', 'i4', ('num_points',), fill_value=-9999.)
    swot_orbits = cl_grp.createVariable('orbits', 'i4', ('num_points','orbit'), fill_value=-9999.)
    lake_id = cl_grp.createVariable('lake_id', 'i8', ('num_points',), fill_value=-9999.)
    seg_dist = cl_grp.createVariable('segDist', 'f8', ('num_points',), fill_value=-9999.)
    endpoints = cl_grp.createVariable('endpoints', 'i4', ('num_points',), fill_value=-9999.)
    wse = cl_grp.createVariable('p_height', 'f8', ('num_points',), fill_value=-9999.)
    wth = cl_grp.createVariable('p_width', 'f8', ('num_points',), fill_value=-9999.)
    facc = cl_grp.createVariable('flowacc', 'f8', ('num_points',), fill_value=-9999.)
    manual = cl_grp.createVariable('manual_add', 'i4', ('num_points',))
    nchan = cl_grp.createVariable('nchan', 'i4', ('num_points',))

    # data
    print("saving nc")

    # root group data
    cont_str = nc.stringtochar(np.array([region], 'S2'))
    Name[:] = cont_str

    # centerline data
    x[:] = np.array(cl.x)
    y[:] = np.array(cl.y)
    segID[:] = np.array(cl.new_seg)
    segInd[:] = np.array(cl.ind)
    segID_old[:] = np.array(cl.seg)
    lakes[:] = np.array(cl.lakes)
    basin[:] = np.array(cl.new_basins)
    deltas[:] = np.array(cl.deltas)
    grand[:] = np.array(cl.grand)
    grod[:] = np.array(cl.grod)
    grod_fid[:] = np.array(cl.grodfid)
    hfall_fid[:] = np.array(cl.hfallsfid)
    swot_obs[:] = np.array(cl.numobs)
    swot_orbits[:,:] = np.array(cl.orbs)
    lake_id[:] = np.array(cl.lakeid)
    endpoints[:] = np.array(cl.eps)
    seg_dist[:] = np.array(cl.dist)
    wse[:] = np.array(cl.wse)
    wth[:] = np.array(cl.wth)
    facc[:] = np.array(cl.facc)
    manual[:] = np.array(cl.manual)
    nchan[:] = np.array(cl.nchan)

    root_grp.close()

###############################################################################
##############################    MAIN CODE    ################################
###############################################################################

start_all = time.time()

#Define input directories and filenames. Will need to be changed based on user needs.
region = 'SA'
data_dir = '/Users/ealtenau/Documents/SWORD_Dev/inputs/'
cl_dir = '/Users/ealtenau/Documents/SWORD_Dev/update_requests/v17/'+region+'/channel_additions/'+region.lower()+'_mhv_point_additions_ALL.gpkg'
outfile = '/Users/ealtenau/Documents/SWORD_Dev/update_requests/v17/'+region+'/channel_additions/'+region.lower()+'_channel_additions.nc'

# Global Paths.
fn_grand = data_dir + 'GRAND/GRanD_dams_v1_1.shp'
fn_deltas = data_dir + 'Deltas/global_map.shp'
track_dir = data_dir + 'SWOT_Tracks/2020_orbits/'
track_list = glob.glob(os.path.join(track_dir, 'ID_PASS*.shp'))
fn_grod = data_dir + 'GROD/v1.1/GROD_v1.1.csv'

# Regional Paths.
mhv_dir = data_dir + 'MHV_SWORD/' + region.lower() + '_mhv_sword.nc'
fn_basins = data_dir + 'HydroBASINS/' + region + '/' + region + '_hb_lev08.shp'
lake_dir = data_dir + 'LakeDatabase/20200702_PLD/For_Merge/' + region + '/'
lake_path = np.array(np.array([file for file in getListOfFiles(lake_dir) if '.shp' in file]))

# Read in new centerline data. 
cl = Object()
cl.x, cl.y, cl.seg, cl.ind = read_cl_data(cl_dir)
# Re-number segments with non-sequential indexes.
cl.new_seg = update_segs(cl)
# cl.ind = order_edits(cl)
# cl.new_seg = np.copy(cl.seg)

# Create distance and endpoint attributes. 
cl.eps = np.zeros(len(cl.x))
cl.dist = np.zeros(len(cl.x))
unq_seg = np.unique(cl.new_seg)
for s in list(range(len(unq_seg))):
    seg = np.where(cl.new_seg == unq_seg[s])[0]
    mn = np.where(cl.ind[seg] == np.min(cl.ind[seg]))[0]
    mx = np.where(cl.ind[seg] == np.max(cl.ind[seg]))[0]
    gdf = gp.GeoDataFrame(geometry=gp.points_from_xy(cl.x[seg], cl.y[seg]),crs="EPSG:4326").to_crs("EPSG:3857")
    diff = gdf.distance(gdf.shift(1)); diff[0] = 0
    dist = np.cumsum(diff)
    cl.dist[seg] = dist
    cl.eps[seg[mn]] = 1
    cl.eps[seg[mx]] = 2
    
# Open global shapefiles for spatial intersections. 
lake_db = gp.GeoDataFrame.from_file(lake_path[0])
delta_db = gp.GeoDataFrame.from_file(fn_deltas)

# Reading in mhv data.
mhv = nc.Dataset(mhv_dir) 
mhv_lon = mhv.groups['centerlines'].variables['x'][:]
mhv_lat = mhv.groups['centerlines'].variables['y'][:]
mhv_wth = mhv.groups['centerlines'].variables['p_width'][:]
mhv_wse = mhv.groups['centerlines'].variables['p_height'][:]
mhv_facc = mhv.groups['centerlines'].variables['flowacc'][:]
mhv.close()

# Merge MHV attributes onto new centerlines.
mhv_pts = np.vstack((mhv_lon, mhv_lat)).T
cl_pts = np.vstack((cl.x, cl.y)).T
kdt = sp.cKDTree(mhv_pts)
pt_dist, pt_ind = kdt.query(cl_pts, k = 10)
cl.wse = np.median(mhv_wse[pt_ind], axis=1)
cl.wth = np.median(mhv_wth[pt_ind], axis=1)
cl.facc = np.median(mhv_facc[pt_ind], axis=1)

# Defining new centerlines extent.
cl_ext = [np.min(cl.x), np.min(cl.y),
            np.max(cl.x), np.max(cl.y)]

# Creating geodataframe for spatial joins. 
cl_df = gp.read_file(cl_dir)

# Subset Lake db to mhv extent.
lake_db_clip = lake_db.cx[cl_ext[0]:cl_ext[2], cl_ext[1]:cl_ext[3]]

# Attach Prior Lake Database (PLD) IDs.
cl.lakeid = add_lakedb(cl_df, lake_db_clip)
cl.lakes = np.zeros(len(cl.lakeid))
cl.lakes[np.where(cl.lakeid > 0)] = 1

# Adding dam, basin, delta, and SWOT track information.
cl.grand, cl.grod, cl.grodfid, cl.hfallsfid = add_dams(cl.x, cl.y, fn_grand, fn_grod)
cl.basins = add_basins(cl_df, fn_basins)
cl.deltas = add_deltas(cl_df, cl.new_seg, delta_db)
track_files = overlapping_tracks(cl.x, cl.y, track_list)
cl.numobs, cl.orbs = add_tracks(cl_df, track_files)

# Filling in segment gaps with no basin values.
cl.new_basins = fill_zero_basins(cl.x, cl.y, cl.new_seg, cl.basins)

# Create filler variables.
cl.nchan = np.repeat(1,len(cl.x))
cl.manual = np.repeat(1,len(cl.x))

# Write netcdf...
save_mhv_nc(cl, region, outfile)

end_all = time.time()
print('Finished Merge in: ' + str(np.round((end_all-start_all)/60, 2)) + ' min')


'''

plt.scatter(cl.x, cl.y, s = 3, c=cl.ind, cmap='rainbow')
plt.show()

seg = np.where(cl.new_seg == 30)[0]
sort_ids = np.argsort(cl.ind[seg])
plt.plot(cl.x[seg[sort_ids]], cl.y[seg[sort_ids]])
plt.show()

plt.scatter(cl.x[seg], cl.y[seg], s = 3, c=cl.ind[seg], cmap='rainbow')
plt.scatter(cl.x[seg[test]], cl.y[seg[test]], s = 3, c='red')
plt.show()

plt.scatter(cl_x[seg], cl_y[seg], s = 3, c=cl_dist[seg], cmap='rainbow')
plt.show()



([ 2,  4,  7,  9, 10, 13, 14, 15, 16, 17, 18, 22, 25, 31, 32, 34, 35,
       36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
       53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
       70, 71, 72, 73, 74])

       
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33])

seg = np.where(cl.seg == 19)[0]
plt.scatter(cl.x[seg], cl.y[seg], s = 3, c=cl.ind[seg], cmap='rainbow')
plt.show()

seg = np.where(cl.seg == 19)[0]
order_ids = np.argsort(cl.ind[seg])
plt.plot(cl.x[seg[order_ids]], cl.y[seg[order_ids]])
plt.show()      



unq_segs = np.unique(cl.seg)
for s in list(range(len(unq_segs))):
    seg = np.where(cl.seg == unq_segs[s])[0]
    diff = np.diff(cl.ind[seg])
    pts = np.where(abs(diff) != 1)[0]


plt.scatter(cl.x[seg], cl.y[seg], s = 3, c=cl.ind[seg], cmap='rainbow')
plt.scatter(cl.x[seg[pts]], cl.y[seg[pts]], s = 5, c='black')
plt.show()
    







seg = np.where(cl.new_seg == 56)[0]
eps = np.where(cl.eps[seg]>0)[0]
plt.scatter(cl.x[seg], cl.y[seg], s = 3, c=cl_Ind[seg], cmap='rainbow')
plt.scatter(cl.x[seg[eps]], cl.y[seg[eps]], s = 5, c='black')
plt.show()

unq_segs = np.unique(cl.new_seg)
for s in list(range(len(unq_segs))):
    seg = np.where(cl.new_seg == unq_segs[s])[0]
    gdf = gp.GeoDataFrame(geometry=gp.points_from_xy(cl.x[seg], cl.y[seg]),crs="EPSG:4326").to_crs("EPSG:3857")
    diff = gdf.distance(gdf.shift(1)); diff[0] = 0
    jumps = np.where(diff > 1000)[0]
    breaks = np.diff(cl.ind[seg])
    if np.max(np.unique(breaks)) > 1:
        print(s, 'index')
    elif len(jumps) > 0:
        print(s, 'jump')

'''

