from __future__ import division
import numpy as np
from scipy import spatial as sp
import netCDF4 as nc
import time
import os 
import geopandas as gp
import pandas as pd
from shapely.geometry import Point
import argparse

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

def sword_flag(mhv,swd_x,swd_y):
    mhv['points'] = mhv.apply(lambda x: [y for y in x['geometry'].coords], axis=1)
    flag = np.zeros(len(mhv))
    sublinks = np.array(mhv['LINKNO'][np.where(mhv['strmOrder']>=3)[0]])
    for ind in list(range(len(sublinks))):
        # print(ind, len(sublinks))
        link = np.where(mhv['LINKNO'] == sublinks[ind])[0]
        subpts = mhv['points'][int(link)]
        x = np.array([line[0] for line in subpts])
        y = np.array([line[1] for line in subpts])
        seg = np.repeat(mhv['LINKNO'][int(link)], len(x))
        so = np.repeat(mhv['strmOrder'][int(link)], len(x))

        sword_pts = np.vstack((swd_x, swd_y)).T
        mhv_pts = np.vstack((x, y)).T
        kdt = sp.cKDTree(sword_pts)
        pt_dist, pt_ind = kdt.query(mhv_pts, k = 20)
        
        good_vals = np.where(pt_dist[:,0] < 0.01)[0]
        perc = (len(good_vals)/len(x))*100

        if perc <= 25:
            flag[int(link)] = 0
            fg = np.repeat(0, len(x))
        else:
            flag[int(link)] = 1
            fg = np.repeat(1, len(x))

        if ind == 0:
            x_all = x
            y_all = y
            seg_all = seg
            so_all = so
            flag_all = fg
        else:
            x_all = np.append(x_all,x, axis=0)
            y_all = np.append(y_all,y, axis=0)
            seg_all = np.append(seg_all,seg, axis=0)
            so_all = np.append(so_all,so, axis=0)
            flag_all = np.append(flag_all,fg, axis=0)

    return x_all, y_all, seg_all, so_all, flag_all, flag

###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Region", type = str)
parser.add_argument("version", help="<Required> Version", type = str)
args = parser.parse_args()

region = args.region
version = args.version
# outdir = '/Users/ealteanau/Documents/SWORD_Dev/inputs/MHV_SWORD/'
# sword_dir = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'
# mhv_dir = '/Users/ealteanau/Documents/SWORD_Dev/inputs/MeritHydroVector/'
outdir = '- /afs/cas.unc.edu/depts/geological_sciences/pavelsky/students/ealtenau/SWORD_dev/inputs/MHV_SWORD/'
sword_dir = '/afs/cas.unc.edu/depts/geological_sciences/pavelsky/students/ealtenau/SWORD_dev/outputs/Reaches_Nodes/'+version+'/netcdf/'
mhv_dir = '- /afs/cas.unc.edu/depts/geological_sciences/pavelsky/students/ealtenau/SWORD_dev/inputs/MeritHydroVector/'
mhv_files = np.sort(np.array(np.array([file for file in getListOfFiles(mhv_dir) if '.shp' in file])))
mhv_basins = np.array([int(f[-6:-4]) for f in mhv_files])

sword = nc.Dataset(sword_dir+region.lower()+'_sword_'+version+'.nc', 'r+')
sword_x = sword.groups['centerlines'].variables['x'][:]
sword_y = sword.groups['centerlines'].variables['y'][:]
sword_id = sword.groups['centerlines'].variables['reach_id'][0,:]
sword_l2 = np.array([int(str(ind)[0:2]) for ind in sword_id])
unq_l2 = np.unique(sword_l2)

start_all = time.time()
for ind in list(range(len(unq_l2))):
    start = time.time()
    print('Starting Basin ' + str(unq_l2[ind]))
    pts = np.where(sword_l2 == unq_l2[ind])[0]
    swd_x = sword_x[pts]
    swd_y = sword_y[pts]
    swd_id = sword_id[pts]
    f = np.where(mhv_basins == unq_l2[ind])[0]
    mhv = gp.read_file(mhv_files[int(f)])
    
    # start = time.time()
    x_pts, y_pts, seg_pts, so_pts, flag_pts, flag = sword_flag(mhv,swd_x,swd_y)
    mhv['sword_flag'] = flag
    # end = time.time()
    # print('Done in: ' + str((end-start)/60) + ' min: ')
    
    mhv = mhv.drop(columns=['points'])
    mhv.set_geometry(col='geometry') #removed "inplace=True" option on leopold. 
    mhv = mhv.set_crs(4326, allow_override=True)
    mhv.to_file(outdir+'mhv_sword_hb'+str(unq_l2[ind])+'.gpkg', driver='GPKG', layer='mhv')
    
    #create pts dataframe 
    mhv_pts = gp.GeoDataFrame([
        x_pts,
        y_pts,
        seg_pts,
        so_pts,
        flag_pts,]).T

    mhv_pts.rename(
        columns={
            0:"x",
            1:"y",
            2:"segment",
            3:"strmorder",
            4:"sword_flag",},inplace=True)

    mhv_pts = mhv_pts.apply(pd.to_numeric, errors='ignore') # nodes.dtypes
    geom = gp.GeoSeries(map(Point, zip(x_pts, y_pts)))
    mhv_pts['geometry'] = geom
    mhv_pts = gp.GeoDataFrame(mhv_pts)
    mhv_pts.set_geometry(col='geometry')
    mhv_pts = mhv_pts.set_crs(4326, allow_override=True)
    mhv_pts.to_file(outdir+'mhv_sword_hb'+str(unq_l2[ind])+'_pts.gpkg', driver='GPKG', layer='mhv_pts')

    end = time.time()
    print('Finished in: ' + str(np.round((end-start)/60, 2)) + ' min')

end_all=time.time()
print('Finished all Basins in: ' + str(np.round((end_all-start_all)/60, 2)) + ' min')
