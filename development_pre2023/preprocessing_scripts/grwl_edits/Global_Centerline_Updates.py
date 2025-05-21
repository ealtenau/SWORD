#### this file will be to preprocess global centerline edits. 


from __future__ import division
import os
import time
import utm
from osgeo import ogr
from osgeo import osr
from osgeo import gdal
import numpy as np
import pandas as pd
from scipy import spatial as sp
from shapely.geometry import Point
import sys
import geopandas as gp
from pyproj import Proj
import rasterio
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
import matplotlib.pyplot as plt

###############################################################################
################################# Functions ###################################
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
        allFiles -- List of files under directory
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

def find_edit_endpoints(edits, segments):

    """
    FUNCTION:
        Creates a new 1-D array that contains the endpoints for each
        edited centerline segment. 0 = not an endpoint, 1 = first endpoint,
        2 = second endpoint.

    INPUTS
        edits -- Object containing attributes for the edited centerlines.

    OUTPUTS
        endpoints -- Endpoint locations for all edit segments.
    """

    # Loop through segments.
    endpoints = np.zeros(len(segments))
    uniq_segs = np.unique(segments)
    for ind in list(range(len(uniq_segs))):
        seg = np.where(segments == uniq_segs[ind])[0]
        seg_x = edits.x[seg]
        seg_y = edits.y[seg]

        # Removing duplicate coordinates.
        coords_df = pd.DataFrame(np.array([seg_x, seg_y]).T)
        duplicates = np.where(pd.DataFrame.duplicated(coords_df))
        if len(duplicates) > 0:
            seg_x = np.delete(seg_x, duplicates)
            seg_y = np.delete(seg_y, duplicates)
            new_seg = np.delete(seg, duplicates)
        else:
            new_seg = np.copy(seg)

        # For each segment calculate distance between points and identify the
        # points with only two neighbors < 60 m away.
        count = 1
        for point in list(range(len(new_seg))):
            dist = np.sqrt((seg_x[point]-seg_x)**2 + (seg_y[point]-seg_y)**2)
            if len(np.where(dist < 60)[0]) < 3:
                endpoints[new_seg[point]] = count
                count = count+1

        # Default to the min and max indexes if no endpoints are found.
        eps = np.where(endpoints[seg] > 0)[0]
        if len(eps) < 2:
            mx = np.where(seg == np.max(seg))
            mn = np.where(seg == np.min(seg))
            endpoints[seg[eps]] = 0
            endpoints[seg[mn]] = 1
            endpoints[seg[mx]] = 2

    return endpoints
###############################################################################

def find_new_eps(edits, segments):

    # Loop through segments.
    endpoints = np.zeros(len(segments))
    uniq_segs = np.unique(segments)
    for ind in list(range(len(uniq_segs))):
        seg = np.where(segments == uniq_segs[ind])[0]
        mx = np.where(edits.ind[seg] == np.max(edits.ind[seg]))
        mn = np.where(edits.ind[seg] == np.min(edits.ind[seg]))
        endpoints[seg] = 0
        endpoints[seg[mn]] = 1
        endpoints[seg[mx]] = 2

    return endpoints

###############################################################################

def order_edits(edits, segments, endpoints):

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
    edits_segInd = np.zeros(len(segments))
    edits_segDist = np.zeros(len(segments))
    uniq_segs = np.unique(segments)
    for ind in list(range(len(uniq_segs))):
        seg = np.where(segments == uniq_segs[ind])[0]
        seg_x = edits.x[seg]
        seg_y = edits.y[seg]
        eps = np.where(endpoints[seg] > 0)[0]

        # If no endpoints are found default to the first index value to start.
        if len(eps) == 0: # condition added on 9/19/19.
            eps = np.array([0])

        edits_segInd[seg[eps[0]]]=1
        edits_segDist[seg[eps[0]]]=0
        idx = eps[0]

        # Order points in a segment starting from the first endpoint.
        count = 2
        while np.min(edits_segInd[seg]) == 0:
            d = np.sqrt((seg_x[idx]-seg_x)**2 + (seg_y[idx]-seg_y)**2)
            dzero = np.where(edits_segInd[seg] == 0)[0]
            next_pt = dzero[np.where(d[dzero] == np.min(d[dzero]))[0]][0]
            edits_segInd[seg[next_pt]] = count
            edits_segDist[seg[next_pt]] = d[next_pt]
            count = count+1
            idx = next_pt

    return edits_segInd

###############################################################################

def find_edit_tributary_junctions(edits):

    """
    FUNCTION:
        Creates a new 1-D that contains the locations of and edit
        segments that need to be cut at a new tributary junction.

    INPUTS
        edits -- Object containing attributes for the edited centerlines.

    OUTPUTS
        tribs -- Locations along an edit segment where the segment should
            be cut: 0 - no tributary, 1 - tributary.
    """

    # Loop through each edited segment and calculate closet GRWL points to the
    # edited segment endpoints.
    tribs = np.zeros(len(edits.seg))
    uniq_segs = np.unique(edits.seg)
    for ind in list(range(len(uniq_segs))):

        # Don't include current segment in points to search. 
        keep = np.where(edits.seg != uniq_segs[ind])[0]
        edit_pts = np.vstack((edits.x[keep], edits.y[keep])).T

        # Isolate endpoints for the edited segment.
        seg = np.where(edits.seg == uniq_segs[ind])[0]
        if len(seg) == 1:
            eps = np.array([0,0])
        else:
            pt1 = np.where(edits.ind[seg] == np.min(edits.ind[seg]))[0]
            pt2 = np.where(edits.ind[seg] == np.max(edits.ind[seg]))[0]
            eps = np.array([pt1,pt2]).T

        # Perform spatial query of closest GRWL points to the edited segment
        # endpoints.
        ep_pts = np.vstack((edits.x[seg[eps]], edits.y[seg[eps]])).T
        kdt = sp.cKDTree(edit_pts)

        if len(seg) < 3:
            pt_dist, pt_ind = kdt.query(ep_pts, k = 4,
                                        distance_upper_bound = 45.0)
        elif 3 <= len(seg) and len(seg) <= 6:
            pt_dist, pt_ind = kdt.query(ep_pts, k = 10,
                                        distance_upper_bound = 100.0)
        elif len(seg) > 6:
            pt_dist, pt_ind = kdt.query(ep_pts, k = 10,
                                        distance_upper_bound = 200.0)

        ep1_ind = pt_ind[0,:]
        ep1_dist = pt_dist[0,:]
        na1 = np.where(ep1_ind == len(edit_pts))
        ep1_dist = np.delete(ep1_dist, na1)
        ep1_ind = np.delete(ep1_ind, na1)

        ep2_ind = pt_ind[1,:]
        ep2_dist = pt_dist[1,:]
        na2 = np.where(ep2_ind == len(edit_pts))
        ep2_dist = np.delete(ep2_dist, na2)
        ep2_ind = np.delete(ep2_ind, na2)

        ep1_segs = np.unique(edits.seg[ep1_ind])
        ep2_segs = np.unique(edits.seg[ep2_ind])

        # If there is only one neighboring GRWL segment, designate it as a
        # tributary junction if the edited segment endpoint falls in the middle
        # of the segment.
        if len(ep1_segs) == 1:
            ep1_min = np.min(edits.ind[np.where(edits.seg == ep1_segs[0])[0]])
            ep1_max = np.max(edits.ind[np.where(edits.seg == ep1_segs[0])[0]])
            if np.min(edits.ind[ep1_ind]) > ep1_min+5 and np.max(edits.ind[ep1_ind]) < ep1_max-5:
                tribs[ep1_ind[0]] = 1

        if len(ep2_segs) == 1:
            ep2_min = np.min(edits.ind[np.where(edits.seg == ep2_segs[0])[0]])
            ep2_max = np.max(edits.ind[np.where(edits.seg == ep2_segs[0])[0]])
            if np.min(edits.ind[ep2_ind]) > ep2_min+5 and np.max(edits.ind[ep2_ind]) < ep2_max-5:
                tribs[ep2_ind[0]] = 1

    return tribs

###############################################################################

def cut_edit_segments(edits, start_seg):

    """
    FUNCTION:
        Creates a new 1-D array that contains unique segment IDs for the GRWL
        segments that need to be cut at tributary junctions.

    INPUTS
        grwl --  Object containing attributes for the GRWL centerlines.
        start_seg -- Starting ID value to assign to the new cut segments. This
            is typically assigned the max_seg value + 1.

    OUTPUTS
        new_segs -- Updated Segment IDs.
    """

    new_segs = np.copy(edits.seg)
    cut = np.where(edits.tribs == 1)[0]
    cut_segs = np.unique(edits.seg[cut])
    seg_id = start_seg

    # Loop through segments that contain tributary junctions and identify
    # the new boundaries of the segment to cut and re-number.
    for ind in list(range(len(cut_segs))):
        seg = np.where(edits.seg == cut_segs[ind])[0]
        num_tribs = np.where(edits.tribs[seg] == 1)[0]
        max_ind = np.where(edits.ind[seg] == np.max(edits.ind[seg]))[0]
        min_ind = np.where(edits.ind[seg] == np.min(edits.ind[seg]))[0]
        bounds = np.insert(num_tribs, 0, min_ind)
        bounds = np.insert(bounds, len(bounds), max_ind)
        for idx in list(range(len(bounds)-1)):
            id1 = bounds[idx]
            id2 = bounds[idx+1]
            new_segs[seg[id1:id2]] = seg_id
            seg_id = seg_id+1

    return new_segs

###############################################################################
###############################  MAIN CODE  ###################################
###############################################################################

outdir = '/Users/ealteanau/Documents/SWORD_Dev/inputs/GRWL/EDITS/'
grwl_dir = '/Users/ealteanau/Documents/SWORD_Dev/inputs/GRWL/GRWL_Masks_V01.01_LatLonNames/'
grwl_paths = np.sort([file for file in getListOfFiles(grwl_dir) if '.tif' in file])
fn_cl = '/Users/ealteanau/Documents/SWORD_Dev/inputs/GRWL_temp/EDITS/gpkg/global_cl_updates_Nov2022.gpkg'

# Read centerline points.
cls = gp.read_file(fn_cl)

# Loop through and find which points are associated with a tile. 
for ind in list(range(len(grwl_paths))): 

    # Find tile name. 
    pattern = grwl_paths[ind][-11:-4]

    # Remove after tile runs. 
    if pattern in ['n56w114','n56e078','n60e066','n60e072']:
        print(ind, pattern, '- Skipping')
        continue 

    # Create outpath. 
    if os.path.exists(outdir): 
        outpath =  outdir+ pattern + '_edit.gpkg'
    else:
        os.makedirs(outdir)
        outpath = outdir + pattern + '_edit.gpkg'

    # Read in grwl tile.     
    raster = gdal.Open(grwl_paths[ind])
    prj=raster.GetProjection()
    srs=osr.SpatialReference(wkt=prj)
    utm_val = srs.GetAttrValue('projcs')
    if len(utm_val) < 12:
        zone = utm_val[-2:-1]
    else:
        zone = utm_val[-3:-1]
    let = utm_val[-1]

    # Defining GRWL tile extent and convert to lat lon.
    left,xres,__,top,__,yres  = raster.GetGeoTransform()
    right = left + (raster.RasterXSize * xres)
    bottom = top + (raster.RasterYSize * yres)
    if top > 10000000:
        top = 10000000.0

    # Setting hemisphere to converto to lat lon. 
    if pattern[0] == 's':
        hemi = False
    else:
        hemi = True
    ymin,xmin = utm.to_latlon(left, bottom, int(zone), northern=hemi)  
    ymax,xmax = utm.to_latlon(right, top, int(zone), northern=hemi)
    xmin = round(xmin)
    xmax = round(xmax)
    ymin = round(ymin)
    ymax = round(ymax)
    if xmax < 0 and xmin > 0:
        xmax = xmax*-1
    
    # Delete raster.
    raster = None

    # Subset global points to tile extent. 
    cls_tile = cls.cx[xmin:xmax, ymin:ymax]

    # Skip if there are no points. 
    if cls_tile.shape[0] == 0:
        print(ind, pattern, '- No GRWL Edits')
        continue

    else:
        edits = Object()
        edits.lat = np.array([np.array(cls_tile.geometry[i])[1] for i in cls_tile.index])
        edits.lon = np.array([np.array(cls_tile.geometry[i])[0] for i in cls_tile.index])
        east = []
        north = []
        for idx in list(range(len(edits.lat))):
            east.append(utm.from_latlon(edits.lat[idx], edits.lon[idx])[0])
            north.append(utm.from_latlon(edits.lat[idx], edits.lon[idx])[1])    
        edits.x = np.array(east)
        edits.y = np.array(north)
        edits.seg = np.array(cls_tile['fid'])
        edits.segInd = cls_tile.index

        # Finding endpoints and indexes. 
        edits.eps = find_edit_endpoints(edits, edits.seg)
        edits.ind = order_edits(edits, edits.seg, edits.eps)

        # Cutting existing segments at tributary junctions.
        edits.tribs = find_edit_tributary_junctions(edits)
        start_seg = np.max(edits.seg)+1
        edits.new_seg = cut_edit_segments(edits, start_seg)
        edits.new_eps = find_edit_endpoints(edits, edits.new_seg)
        edits.new_ind = order_edits(edits, edits.new_seg, edits.new_eps)

        # Create filler variable for lake flag. 
        edits.lake = np.repeat(1,len(edits.seg))

        # Create geodatabase to write.
        new_gdb = gp.GeoDataFrame([
            edits.lat,
            edits.lon,
            edits.x,
            edits.y,
            edits.new_seg,
            edits.lake
            ]).T

        # Rename columns.
        new_gdb.rename(
            columns={
                0:"lat",
                1:"lon",
                2:"x",
                3:"y",
                4:"seg",
                5:"lakeflag",
                },inplace=True)

        # Create geometry column.
        new_gdb = new_gdb.apply(pd.to_numeric, errors='ignore') # nodes.dtypes
        geom = gp.GeoSeries(map(Point, zip(new_gdb['lon'], new_gdb['lat'])))
        new_gdb['geometry'] = geom
        new_gdb = gp.GeoDataFrame(new_gdb)
        new_gdb.set_geometry(col='geometry')
        new_gdb = new_gdb.set_crs(4326, allow_override=True)

        # Save edits. 
        new_gdb.to_file(outpath, driver='GPKG')
        print('***', ind, pattern, '- EDITS SAVED ***')

        # Delete subset of global centelrine edits. 
        cls_tile = None

###############################################################################

'''
---
Figures for checking output.
---

unq_id = np.unique(edits.seg)
number_of_colors = len(unq_id)+5
color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
plt.figure(1, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Edit Segments',  fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
for i in list(range(len(unq_id))):
    seg = np.where(edits.seg == unq_id[i])
    plt.scatter(edits.lon[seg], edits.lat[seg], c=color[i], s = 5, edgecolors = 'None')
w = np.where(edits.eps > 0)[0]
plt.scatter(edits.lon[w], edits.lat[w], c='black', s= 20, edgecolors = None)
plt.show()


unq_id2 = np.unique(edits.new_seg)
number_of_colors2 = len(unq_id2)+5
color2 = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors2)]
plt.figure(2, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Edit New Segments',  fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
for i in list(range(len(unq_id2))):
    seg = np.where(edits.new_seg == unq_id2[i])
    plt.scatter(edits.lon[seg], edits.lat[seg], c=color2[i], s = 5, edgecolors = 'None')
w2 = np.where(edits.new_eps > 0)[0]
plt.scatter(edits.lon[w2], edits.lat[w2], c='black', s= 20, edgecolors = None)
plt.show()

###

plt.figure(3, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Segments', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(edits.lon, edits.lat, c=edits.segInd, edgecolors='none', s = 5)
plt.colorbar()
plt.show()

plt.figure(4, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Segments', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(edits.lon, edits.lat, c=edits.ind, edgecolors='none', s = 5)
plt.colorbar()
plt.show()

eps = np.where(edits.eps > 0)[0]
plt.figure(5, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Segments', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(edits.lon, edits.lat, c=edits.seg, edgecolors='none', s = 5)
plt.scatter(edits.lon[eps], edits.lat[eps], c='black', edgecolors='none', s = 8)
plt.colorbar()
plt.show()

'''