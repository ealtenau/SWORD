from __future__ import division
import numpy as np
from scipy import spatial as sp
import netCDF4 as nc
from pyproj import Proj
import time
import geopy.distance
import pandas as pd
import argparse
import re
import os 

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

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Region", type = str)
parser.add_argument("version", help="<Required> Version", type = str)
parser.add_argument('-tiles', nargs='+', help='<Optional> List of tiles to run.')
args = parser.parse_args()

region = args.region
version = args.version
fn_merge = region + '_Merge_'+version+'.nc'
data_dir = '/Users/ealteanau/Documents/SWORD_Dev/inputs/'

grwl_dir = data_dir + 'GRWL_temp/New_Updates/' + region + '/'
grwl_paths = np.array([file for file in getListOfFiles(grwl_dir) if '.shp' in file])

if args.tiles:
    matches = list(args.tiles)
    grwl_paths = [file for file in grwl_paths if re.search('|'.join(matches), file)]
    print(grwl_paths)
else:
    print(grwl_paths)

#########################################################################################
# def write_cl_iceflag_nc(centerlines, outfile):

#     start = time.time()

#     # global attributes
#     root_grp = nc.Dataset(outfile, 'w', format='NETCDF4')
#     root_grp.production_date = time.strftime("%d-%b-%Y %H:%M:%S", time.gmtime()) #utc time
#     #root_grp.history = 'Created ' + time.ctime(time.time())

#     # groups
#     cl_grp = root_grp.createGroup('centerlines')

#     # dimensions
#     #root_grp.createDimension('d1', 2)
#     cl_grp.createDimension('num_points', len(centerlines.cl_id))
#     cl_grp.createDimension('num_domains', 4)
#     cl_grp.createDimension('julian_day', 366)

#     # centerline variables
#     cl_id = cl_grp.createVariable(
#         'cl_id', 'i8', ('num_points',), fill_value=-9999.)
#     cl_x = cl_grp.createVariable(
#         'x', 'f8', ('num_points',), fill_value=-9999.)
#     cl_x.units = 'degrees east'
#     cl_y = cl_grp.createVariable(
#         'y', 'f8', ('num_points',), fill_value=-9999.)
#     cl_y.units = 'degrees north'
#     reach_id = cl_grp.createVariable(
#         'reach_id', 'i8', ('num_domains','num_points'), fill_value=-9999.)
#     reach_id.format = 'CBBBBBRRRRT'
#     node_id = cl_grp.createVariable(
#         'node_id', 'i8', ('num_domains','num_points'), fill_value=-9999.)
#     node_id.format = 'CBBBBBRRRRNNNT'
#     cl_iceflag = cl_grp.createVariable(
#         'iceflag', 'i4', ('julian_day','num_points'), fill_value=-9999.)

#     # saving data
#     print("saving nc")
#     # centerline data
#     cl_id[:] = centerlines.cl_id
#     cl_x[:] = centerlines.x
#     cl_y[:] = centerlines.y
#     reach_id[:,:] = centerlines.reach_id
#     node_id[:,:] = centerlines.node_id
#     cl_iceflag[:,:] = centerlines.ice_flag

#     root_grp.close()
#     end = time.time()
#     print("Ended Saving Main NetCDF in: ", str(np.round((end-start)/60, 2)), " min")

#     return outfile

# outfile = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'\
#     'SWOT_Coverage_Ice/v14/netcdf/na_centerline_iceflag.nc'
# write_cl_iceflag_nc(centerlines, outfile)



