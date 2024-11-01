import numpy as np
import netCDF4 as nc 
from scipy import spatial as sp
import os 
from osgeo import gdal
import time
from pyproj import Proj
import utm
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

    keep = np.where(facc >= 25)[0]
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
		"+proj=utm +zone=" + str(int(unq_zones[idx])) + utm_let +
		" +ellips=WGS84 +datum=WGS84 +units=m")
    else:
        myproj = Proj(
		"+proj=utm +south +zone=" + str(int(unq_zones[idx])) + utm_let +
		" +ellips=WGS84 +datum=WGS84 +units=m")

    # Convert all the lon/lat to the main UTM zone
    (east, north) = myproj(longitude, latitude)

    return east, north, zone_num, zone_let

###############################################################################
###############################################################################
###############################################################################

start_all = time.time()
region = 'AS'
mh_elv_dir = '/Users/ealteanau/Documents/SWORD_Dev/inputs/MERIT_Hydro/'+region+'/elv/'
mh_facc_dir = '/Users/ealteanau/Documents/SWORD_Dev/inputs/MERIT_Hydro/'+region+'/upa/'
mh_wth_dir = '/Users/ealteanau/Documents/SWORD_Dev/inputs/MERIT_Hydro/'+region+'/wth/'
facc_paths = np.sort(np.array([file for file in getListOfFiles(mh_facc_dir) if '.tif' in file]))
elv_paths = np.sort(np.array([file for file in getListOfFiles(mh_elv_dir) if '.tif' in file]))
wth_paths = np.sort(np.array([file for file in getListOfFiles(mh_wth_dir) if '.tif' in file]))
mhv_fn = '/Users/ealteanau/Documents/SWORD_Dev/inputs/MHV_SWORD/'+region.lower()+'_mhv_sword.nc'

mhv = nc.Dataset(mhv_fn, 'r+')
mhv_lon = mhv.groups['centerlines'].variables['x'][:].data
mhv_lat = mhv.groups['centerlines'].variables['y'][:].data
mhv_id = mhv.groups['centerlines'].variables['segID'][:].data
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
mhv_ind = np.repeat(0, len(mhv_lon))
mhv_dist = np.repeat(0, len(mhv_lon))
mhv_x = np.repeat(0, len(mhv_lon))
mhv_y = np.repeat(0, len(mhv_lon))
mhv_cl_id = np.repeat(0, len(mhv_lon))

start = time.time()
print('======== Starting MH-MHV Merge ========')
for ind in list(range(len(elv_paths))):
    # print(ind, len(elv_paths))
    tile = elv_paths[ind][-15:-8]
    mh_lon, mh_lat, mh_elv, mh_wth, mh_facc = MH_vals(elv_paths[ind], wth_paths[ind], facc_paths[ind])
    
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
for idx in list(range(len(unq_seg))):
    print(idx, len(unq_seg))
    
    pts = np.where(mhv_id == unq_seg[idx])[0]
    seg_ind = pts
    seg_lon = mhv_lon[pts]
    seg_lat = mhv_lat[pts]
    seg_elv = mhv_elv[pts]
    seg_facc = mhv_facc[pts]
    seg_epts = mhv_endpts[pts]
    seg_epts[0] = 1
    seg_epts[-1] = 2

    seg_x, seg_y, __, __ = reproject_utm(seg_lat, seg_lon)

    #order the reach points based on index values, then calculate the
    #eculdean distance bewteen each ordered point.
    order_ids = np.argsort(seg_ind)
    dist = np.zeros(len(seg_lon))
    dist[order_ids[0]] = 0
    for idx in list(range(len(order_ids)-1)):
        d = np.sqrt((seg_x[order_ids[idx]]-seg_x[order_ids[idx+1]])**2 +
                    (seg_y[order_ids[idx]]-seg_y[order_ids[idx+1]])**2)
        dist[order_ids[idx+1]] = d + dist[order_ids[idx]]

    #format flow distance as array and determine flow direction by flowacc.
    dist = np.array(dist)
    start = seg_facc[np.where(dist == np.min(dist))[0][0]]
    end = seg_facc[np.where(dist == np.max(dist))[0][0]]

    if end > start:
        seg_dist = abs(dist-np.max(dist))

    else:
        seg_dist = dist

    mhv_ind[pts] = seg_ind
    mhv_cl_id[pts] = seg_ind
    mhv_endpts[pts] = seg_epts
    mhv_dist[pts] = seg_dist
    mhv_x[pts] = seg_x
    mhv_y[pts] = seg_y
    
end2 = time.time()
print('Finished Segments in: ' + str(np.round((end2-start2)/60, 2)) + ' min')

print('======== Adding New Variables to NetCDF ========')
mhv.groups['centerlines'].createVariable('cl_id', 'i8', ('num_points',))
mhv.groups['centerlines'].createVariable('easting', 'f8', ('num_points',))
mhv.groups['centerlines'].createVariable('northing', 'f8', ('num_points',))
mhv.groups['centerlines'].createVariable('segInd', 'f8', ('num_points',))
mhv.groups['centerlines'].createVariable('segDist', 'f8', ('num_points',))
mhv.groups['centerlines'].createVariable('p_width', 'f8', ('num_points',))
mhv.groups['centerlines'].createVariable('p_height', 'f8', ('num_points',))
mhv.groups['centerlines'].createVariable('flowacc', 'f8', ('num_points',))
mhv.groups['centerlines'].createVariable('nchan', 'i4', ('num_points',))
mhv.groups['centerlines'].createVariable('manual_add', 'i4', ('num_points',))
mhv.groups['centerlines'].createVariable('endpoints', 'i4', ('num_points',))
mhv.groups['centerlines'].createVariable('mh_tile', 'S7', ('num_points',))
mhv.groups['centerlines'].variables['mh_tile']._Encoding = 'ascii'

mhv.groups['centerlines'].variables['cl_id'][:] = mhv_cl_id
mhv.groups['centerlines'].variables['easting'][:] = mhv_x
mhv.groups['centerlines'].variables['northing'][:] = mhv_y
mhv.groups['centerlines'].variables['segInd'][:] = mhv_ind
mhv.groups['centerlines'].variables['segDist'][:] = mhv_dist
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



# test = np.where(mhv_lon < 0)[0]
# plt.scatter(mhv_lon[test], mhv_lat[test], s = 1, c=mhv_dist[test])
# plt.scatter(mhv_lon[test], mhv_lat[test], s = 1)
# plt.show()

# df = pd.DataFrame(np.array([mhv_lon[test], mhv_lat[test]]).T)
# df.to_csv('/Users/ealteanau/Desktop/as_lat_lon_test.csv')

