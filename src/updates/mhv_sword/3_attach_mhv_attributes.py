import os
main_dir = os.getcwd()
import numpy as np
import netCDF4 as nc 
from scipy import spatial as sp
from osgeo import gdal, ogr
import time
from pyproj import Proj
import utm
import glob
from geopy import distance
import matplotlib.pyplot as plt
import argparse
import warnings
warnings.filterwarnings("ignore") #if code stops working may need to comment out to check warnings. 

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

def get_distances(lon,lat):
    traces = len(lon) -1
    distances = np.zeros(traces)
    for i in range(traces):
        start = (lat[i], lon[i])
        finish = (lat[i+1], lon[i+1])
        distances[i] = distance.geodesic(start, finish).m
    distances = np.append(0,distances)
    return distances

###############################################################################

def MH_vals(elv_fn, wth_fn, facc_fn):

    """
    FUNCTION:
        Reads in and formats MERIT Hydro raster values as arrays.

    INPUTS
        filepaths -- List of MERIT Hydro raster paths that overlap a GRWL shapefile.
        grwl_ext -- Latitude/Longitude extent of GRWL shapefile.

    OUTPUTS
        vals -- raster values in 1-D array. These values will coincide with
        coordinate values returned from "MH_coords" function.
    """

    # Mosaicking all MERIT Hydro rasters that overlap the GRWL shapefile.
    
    elv_raster = gdal.Open(elv_fn)
    wth_raster = gdal.Open(wth_fn)
    facc_raster = gdal.Open(facc_fn)

    # Pulls and flattens raster vlues.
    xul = elv_raster.GetGeoTransform()[0]
    xres = elv_raster.GetGeoTransform()[1]
    yul = elv_raster.GetGeoTransform()[3]
    yres = elv_raster.GetGeoTransform()[5]
    
    lon=np.array([xul+xres*c+xres/2. for r in range(elv_raster.RasterXSize) for c in range(elv_raster.RasterYSize)])
    lat=np.array([yul+yres*r+yres/2. for r in range(elv_raster.RasterXSize) for c in range(elv_raster.RasterYSize)])
    elv = np.array(elv_raster.GetRasterBand(1).ReadAsArray()).flatten()
    wth = np.array(wth_raster.GetRasterBand(1).ReadAsArray()).flatten()
    facc = np.array(facc_raster.GetRasterBand(1).ReadAsArray()).flatten()

    keep = np.where(facc >= 10)[0]
    elv = elv[keep]
    wth = wth[keep]
    facc = facc[keep]
    lon = lon[keep]
    lat = lat[keep]

    return lon, lat, elv, wth, facc

###############################################################################

def reproject_utm(latitude, longitude):

    """
    Modified from C. Lion's function by E. Altenau
    Copyright (c) 2018 UNC Chapel Hill. All rights reserved.

    FUNCTION:
        Projects all points in UTM.

    INPUTS
        latitude -- latitude in degrees (1-D array)
        longitude -- longitude in degrees (1-D array)

    OUTPUTS
        east -- easting in UTM (1-D array)
        north -- northing in UTM (1-D array)
        utm_num -- UTM zone number (1-D array of utm zone numbers for each point)
        utm_let -- UTM zone letter (1-D array of utm zone letters for each point)
    """

    east = np.zeros(len(latitude))
    north = np.zeros(len(latitude))
    east_int = np.zeros(len(latitude))
    north_int = np.zeros(len(latitude))
    zone_num = np.zeros(len(latitude))
    zone_let = []

	# Finds UTM letter and zone for each lat/lon pair.

    for ind in list(range(len(latitude))):
        (east_int[ind], north_int[ind],
	 zone_num[ind], zone_let_int) = utm.from_latlon(latitude[ind],
	                                                longitude[ind])
        zone_let.append(zone_let_int)

    # Finds the unique UTM zones and converts the lat/lon pairs to UTM.
    unq_zones = np.unique(zone_num)
    utm_let = np.unique(zone_let)[0]

    for idx in list(range(len(unq_zones))):
        pt_len = len(np.where(zone_num == unq_zones[idx])[0])

    idx = np.where(pt_len == np.max(pt_len))

    # Set the projection

    if np.sum(latitude) > 0:
        myproj = Proj(
            "+proj=utm +zone=" + str(int(unq_zones[idx])) +
            " +ellips=WGS84 +datum=WGS84 +units=m")
    else:
        myproj = Proj(
            "+proj=utm +south +zone=" + str(int(unq_zones[idx])) +
           " +ellips=WGS84 +datum=WGS84 +units=m")

    # Convert all the lon/lat to the main UTM zone
    (east, north) = myproj(longitude, latitude)

    return east, north, zone_num, zone_let

###############################################################################

def overlapping_files(mhv_lon, mhv_lat, elv_paths):

    #define grwl extent as ogr geometry format.
    poly1 = ogr.Geometry(ogr.wkbLinearRing)
    poly1.AddPoint(min(mhv_lon), max(mhv_lat))
    poly1.AddPoint(min(mhv_lon), min(mhv_lat))
    poly1.AddPoint(max(mhv_lon), min(mhv_lat))
    poly1.AddPoint(max(mhv_lon), max(mhv_lat))
    poly1.AddPoint(min(mhv_lon), max(mhv_lat))
    mhvGeometry = ogr.Geometry(ogr.wkbPolygon)
    mhvGeometry.AddGeometry(poly1)
    poly_box = mhvGeometry.GetEnvelope()        

    #find overlapping SWOT tracks.
    track_files = []
    for fn in elv_paths:
        # Read raster extent
        # Open the raster file
        try:
            raster_ds = gdal.Open(fn)
            raster_geotransform = raster_ds.GetGeoTransform()
            raster_extent = (
                raster_geotransform[0],
                raster_geotransform[0] + raster_geotransform[1] * raster_ds.RasterXSize,
                raster_geotransform[3] + raster_geotransform[5] * raster_ds.RasterYSize,
                raster_geotransform[3]
            )

            # Check for overlap
            overlap = (
                poly_box[0] < raster_extent[1] and
                poly_box[1] > raster_extent[0] and
                poly_box[2] < raster_extent[3] and
                poly_box[3] > raster_extent[2]
            )

            if overlap == True:
                track_files.append(fn)

        except:
            print('!!Read Error!!', fn)
            continue
        
    track_files = np.unique(track_files)

    return(track_files)

###############################################################################
###############################################################################
###############################################################################

start_all = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Region", type = str)
args = parser.parse_args()

region = args.region
# region = 'AS'
mh_elv_dir = main_dir+'/data/inputs/MERIT_Hydro/'+region+'/elv/'
mh_facc_dir = main_dir+'/data/inputs/MERIT_Hydro/'+region+'/upa/'
mh_wth_dir = main_dir+'/data/inputs/MERIT_Hydro/'+region+'/wth/'
facc_paths = np.sort(np.array([file for file in getListOfFiles(mh_facc_dir) if '.tif' in file]))
elv_paths = np.sort(np.array([file for file in getListOfFiles(mh_elv_dir) if '.tif' in file]))
wth_paths = np.sort(np.array([file for file in getListOfFiles(mh_wth_dir) if '.tif' in file]))
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
    elv_basin_paths = overlapping_files(mhv_lon, mhv_lat, elv_paths)

    start = time.time()
    print('======== Starting MH-MHV Merge ========')
    for ind in list(range(len(elv_basin_paths))):
        # print(ind, len(elv_basin_paths))
        ind2 = np.where(elv_paths == elv_basin_paths[ind])[0][0]
        tile = elv_paths[ind][-15:-8]
        mh_lon, mh_lat, mh_elv, mh_wth, mh_facc = MH_vals(elv_paths[ind2], wth_paths[ind2], facc_paths[ind2])
        
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
        seg_x, seg_y, __, __ = reproject_utm(seg_lat, seg_lon)

        #segment distance
        sort_ind = np.argsort(seg_ind)
        x_coords = seg_lon[sort_ind]
        y_coords = seg_lat[sort_ind]
        diff = get_distances(x_coords,y_coords)
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



'''
z = np.where(mhv_wth == 0)[0]
plt.scatter(mhv_lon, mhv_lat, s = 3, c=np.log(mhv_wth), cmap='rainbow')
# plt.scatter(mhv_lon[z], mhv_lat[z], s = 3, c='lightgrey')
plt.show()

z = np.where(mhv_facc == 0)[0]
plt.scatter(mhv_lon, mhv_lat, s = 3, c=np.log(mhv_facc), cmap='rainbow')
# plt.scatter(mhv_lon[z], mhv_lat[z], s = 3, c='lightgrey')
plt.show()

z = np.where(mhv_elv == 0)[0]
plt.scatter(mhv_lon, mhv_lat, s = 3, c=mhv_elv, cmap='rainbow')
plt.scatter(mhv_lon[z], mhv_lat[z], s = 3, c='lightgrey')
plt.show()

plt.scatter(mhv_lon, mhv_lat, s = 3, c='blue')
plt.scatter(mh_lon, mh_lat, s = 1, c='red')
plt.show()

plt.scatter(mh_lon, mh_lat, s = 1, c=np.log(mh_wth), cmap='rainbow')
plt.show()

plt.scatter(mh_lon, mh_lat, s = 1, c=np.log(mh_elv), cmap='rainbow')
plt.show()


plt.scatter(mhv_lon[pts], mhv_lat[pts], s = 5, c=mhv_dist[pts], cmap='rainbow')
plt.show()

'''