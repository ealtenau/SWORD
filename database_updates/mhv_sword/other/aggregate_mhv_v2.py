import geopandas as gp
import netCDF4 as nc
import numpy as np
import os
import pandas as pd
import time 

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

def save_mhv_nc(mhv_seg_all, mhv_seg_old_all, mhv_x_all, mhv_y_all, mhv_order_all, 
                mhv_flag_all, mhv_basins_all, region, outfile):

    """
    FUNCTION:
        Writes filtered merged NetCDF. Datasets combined include: GRWL,
        MERIT Hydro, GROD, GRanD, HydroBASINS, Global Deltas, SWOT Track
        information, and Prior Lake Database locations.

    INPUTS
        merged -- Object containing merged attributes for the GRWL centerline.
        outfile -- Outpath directory to write the NetCDF file.

    OUTPUTS
        Merged NetCDF file.
    """

    # global attributes
    root_grp = nc.Dataset(outfile, 'w', format='NETCDF4')
    root_grp.x_min = np.min(mhv_x_all)
    root_grp.x_max = np.max(mhv_x_all)
    root_grp.y_min = np.min(mhv_y_all)
    root_grp.y_max = np.max(mhv_y_all)
    root_grp.production_date = time.strftime("%d-%b-%Y %H:%M:%S", time.gmtime()) #utc time

    # subgroups
    cl_grp = root_grp.createGroup('centerlines')

    # dimensions
    root_grp.createDimension('ID', 2)
    cl_grp.createDimension('num_points', len(mhv_x_all))

    ### variables and units

    # root group variables
    Name = root_grp.createVariable('Name', 'S1', ('ID'))
    Name._Encoding = 'ascii'

    # centerline variables
    x = cl_grp.createVariable(
        'x', 'f8', ('num_points',), fill_value=-9999.)
    x.units = 'degrees east'
    y = cl_grp.createVariable(
        'y', 'f8', ('num_points',), fill_value=-9999.)
    y.units = 'degrees north'
    segID = cl_grp.createVariable(
        'segID', 'i8', ('num_points',), fill_value=-9999.)
    segID_old = cl_grp.createVariable(
        'segID_old', 'i8', ('num_points',), fill_value=-9999.)
    strmorder= cl_grp.createVariable(
        'strmorder', 'i4', ('num_points',), fill_value=-9999.)
    swordflag= cl_grp.createVariable(
        'swordflag', 'i4', ('num_points',), fill_value=-9999.)
    basin= cl_grp.createVariable(
        'basin', 'i8', ('num_points',), fill_value=-9999.)
    
    # data
    print("saving nc")

    # root group data
    cont_str = nc.stringtochar(np.array([region], 'S2'))
    Name[:] = cont_str

    # centerline data
    x[:] = np.array(mhv_x_all)
    y[:] = np.array(mhv_y_all)
    segID[:] = np.array(mhv_seg_all)
    segID_old[:] = np.array(mhv_seg_old_all)
    strmorder[:] = np.array(mhv_order_all)
    swordflag[:] = np.array(mhv_flag_all)
    basin[:] = np.array(mhv_basins_all)

    root_grp.close()

###############################################################################

mhv_dir = '/Users/ealteanau/Documents/SWORD_Dev/inputs/MHV_SWORD/'
mhv_files = np.sort(np.array(np.array([file for file in getListOfFiles(mhv_dir) if 'pts' in file])))
mhv_basins = np.array([int(f[-11:-10]) for f in mhv_files])
regions = ['AS', 'NA', 'SA', 'EU', 'AF', 'OC']

for ind in list(range(len(regions))):
    start = time.time()
    if regions[ind] == 'AS':
        subset = np.where((mhv_basins >= 3) & (mhv_basins < 5))[0]
    if regions[ind] == 'NA':
        subset = np.where(mhv_basins >= 7)[0]
    if regions[ind] == 'SA':
        subset = np.where(mhv_basins == 6)[0]
    if regions[ind] == 'EU':
        subset = np.where(mhv_basins == 2)[0]
    if regions[ind] == 'AF':
        subset = np.where(mhv_basins == 1)[0]
    if regions[ind] == 'OC':
        subset = np.where(mhv_basins == 5)[0]

    outfile = mhv_dir + regions[ind].lower() + '_mhv_sword_v2.nc'
    combine_files = mhv_files[subset]
    #loop through combine_files and aggregate
    cnt=1
    for idx in list(range(len(combine_files))):
        mhv = gp.read_file(combine_files[idx])

        mhv_x = np.array(mhv['x'])
        mhv_y = np.array(mhv['y'])
        mhv_seg = np.array(mhv['segment'])
        mhv_order = np.array(mhv['strmorder'])
        mhv_flag = np.array(mhv['sword_flag'])

        new_segs = np.zeros(len(mhv_seg))
        basins = np.zeros(len(mhv_seg))
        unq_segs = np.unique(mhv_seg) 
        for s in list(range(len(unq_segs))):
            pts = np.where(mhv_seg == unq_segs[s])[0]
            new_segs[pts] = cnt
            basins[pts] = int(combine_files[idx][-11:-9])
            cnt=cnt+1
        if idx == 0:
            mhv_seg_all = np.copy(new_segs)
            mhv_seg_old_all = np.copy(mhv_seg)
            mhv_x_all = np.copy(mhv_x)
            mhv_y_all = np.copy(mhv_y)
            mhv_order_all = np.copy(mhv_order)
            mhv_flag_all = np.copy(mhv_flag)
            mhv_basins_all = np.copy(basins)
        else:
            mhv_seg_all = np.insert(mhv_seg_all, len(mhv_seg_all), np.copy(new_segs))
            mhv_seg_old_all = np.insert(mhv_seg_old_all, len(mhv_seg_old_all), np.copy(mhv_seg))
            mhv_x_all = np.insert(mhv_x_all, len(mhv_x_all), np.copy(mhv_x))
            mhv_y_all = np.insert(mhv_y_all, len(mhv_y_all), np.copy(mhv_y))
            mhv_order_all = np.insert(mhv_order_all, len(mhv_order_all), np.copy(mhv_order))
            mhv_flag_all = np.insert(mhv_flag_all, len(mhv_flag_all), np.copy(mhv_flag))
            mhv_basins_all = np.insert(mhv_basins_all, len(mhv_basins_all), np.copy(basins))

    save_mhv_nc(mhv_seg_all, mhv_seg_old_all, mhv_x_all, mhv_y_all, mhv_order_all, 
                mhv_flag_all, mhv_basins_all, regions[ind], outfile)
    end = time.time()
    print('Finished ' + regions[ind] + ' in: ' + str(np.round((end-start)/60, 2)) + ' min')
    print(len(np.unique(mhv_seg_old_all)), len(np.unique(mhv_seg_all)), cnt)

