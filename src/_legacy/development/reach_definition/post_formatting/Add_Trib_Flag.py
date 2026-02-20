from __future__ import division
import os
main_dir = os.getcwd()
import numpy as np
from scipy import spatial as sp
import netCDF4 as nc
import time
import geopandas as gp
import argparse
import matplotlib.pyplot as plt

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
args = parser.parse_args()

region = args.region
version = args.version

sword_dir = main_dir+'/data/outputs/Reaches_Nodes/'+version+'/netcdf/'
mhv_dir = main_dir+'/data/inputs/MHV_SWORD/'
mhv_files = np.sort(np.array(np.array([file for file in getListOfFiles(mhv_dir) if 'pts' in file])))
mhv_basins = np.array([int(f[-11:-9]) for f in mhv_files])

sword = nc.Dataset(sword_dir+region.lower()+'_sword_'+version+'.nc', 'r+')
sword_x = sword.groups['nodes'].variables['x'][:]
sword_y = sword.groups['nodes'].variables['y'][:]
sword_nid = sword.groups['nodes'].variables['node_id'][:]
sword_nrid = sword.groups['nodes'].variables['reach_id'][:]
sword_rid = sword.groups['reaches'].variables['reach_id'][:]
sword_l2 = np.array([int(str(ind)[0:2]) for ind in sword_nid])
unq_l2 = np.unique(sword_l2)

rch_tribs = np.zeros(len(sword_rid))
node_tribs = np.zeros(len(sword_nid))
start_all = time.time()
for ind in list(range(len(unq_l2))):
    print('Starting Basin ' + str(unq_l2[ind]))
    pts = np.where(sword_l2 == unq_l2[ind])[0]
    swd_x = sword_x[pts]
    swd_y = sword_y[pts]
    swd_nid = sword_nid[pts]
    swd_nrid = sword_nrid[pts]
    f = np.where(mhv_basins == unq_l2[ind])[0]
    mhv = gp.read_file(mhv_files[int(f)])

    subset = np.where((mhv.sword_flag == 0) & (mhv.strmorder >= 3))[0]
    mhv_x = np.array(mhv.x[subset])
    mhv_y = np.array(mhv.y[subset])

    # find closest points.     
    mhv_pts = np.vstack((mhv_x, mhv_y)).T
    node_pts = np.vstack((swd_x, swd_y)).T
    kdt = sp.cKDTree(mhv_pts)
    eps_dist, eps_ind = kdt.query(node_pts, k = 10)
    
    flag = np.where(eps_dist[:,0] <= 0.003)[0]
    fg_nodes = np.unique(swd_nid[flag])
    fg_rchs = np.unique(swd_nrid[flag])

    ind_nodes = np.intersect1d(sword_nid, fg_nodes, return_indices=True)[1]
    ind_rchs = np.intersect1d(sword_rid, fg_rchs, return_indices=True)[1]
    node_tribs[ind_nodes] = 1
    rch_tribs[ind_rchs] = 1

#add flag to database.
sword.groups['nodes'].createVariable('trib_flag', 'i4', ('num_nodes',), fill_value=-9999.)
sword.groups['reaches'].createVariable('trib_flag', 'i4', ('num_reaches',), fill_value=-9999.)
sword.groups['nodes'].variables['trib_flag'][:] = node_tribs 
sword.groups['reaches'].variables['trib_flag'][:] = rch_tribs 
sword.close()

end_all = time.time()
print('Finished All: ' + str(np.round((end_all-start_all)/60, 2)) + ' min')


# t = np.where(node_tribs == 1)[0]
# plt.scatter(mhv_x, mhv_y, c='grey',s=3)
# plt.scatter(swd_x, swd_y, c='black',s=3)
# plt.scatter(sword_x[t], sword_y[t], c='red', s=5)
# plt.show()