"""
Calculating Stream Order (stream_order.py).
===============================================================

This scripts calculates unique segment values and an initial 
stream order for based on the path variables and adds them 
to the path variable netCDF.  

The script is run at a Pfafstetter Level 2 basin scale.
Command line arguments required are the two-letter
region identifier (i.e. NA), SWORD version (i.e. v17),
and Pfafstetter Level 2 basin (i.e. 74).

Execution example (terminal):
    python stream_order.py NA v17

""" 

import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import netCDF4 as nc
import argparse

###############################################################################

def find_path_segs(order, paths):
    """
    Calculates unique segment values between river network 
    junctions based on path variables.

    Parameters
    ----------
    order: numpy.array()
        Path order: Unique values representing continuous paths from the
        river outlet to the headwaters ordered from the longest path (1) 
        to the shortest path (N).
    paths: numpy.array()
        Path frequency: The number of times a point is traveled along get to any 
        given headwater point.
        
    Returns
    -------
    path_segs: numpy.array()
        Unique IDs given to river segments between junctions. 
        
    """
    
    unq_paths = np.unique(order)
    unq_paths = unq_paths[unq_paths>0]
    cnt = 1
    path_segs = np.zeros(len(order))
    for p in list(range(len(unq_paths))):
        pth = np.where(order == unq_paths[p])[0]
        sections = np.unique(paths[pth])
        for s in list(range(len(sections))):
            sec = np.where(paths[pth] == sections[s])[0]
            path_segs[pth[sec]] = cnt
            cnt = cnt+1
    return path_segs

###############################################################################
###############################################################################
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version (i.e. v17)", type = str)
parser.add_argument("basin", help="Pfafstetter Level 2 Basin Number (i.e. 74)", type = str)
args = parser.parse_args()

region = args.region
version = args.version
basin = args.basin

path_nc = main_dir+'/data/outputs/Reaches_Nodes/'+version+\
    '/network_building/pathway_netcdfs/'+region+'/hb'+basin+'_path_vars.nc'
con_dir = main_dir+'/data/outputs/Reaches_Nodes/'+version+\
    '/reach_geometry/'+region.lower()+'_sword_'+version+'_connectivity.nc'

#read in data. 
data = nc.Dataset(path_nc,'r+')
con = nc.Dataset(con_dir)

#assign relevant data to arrays. 
paths = data.groups['centerlines'].variables['path_travel_frequency'][:]
order = data.groups['centerlines'].variables['path_order_by_length'][:]
main_side = data.groups['centerlines'].variables['main_side_chan'][:]
rchs = data.groups['centerlines'].variables['reach_id'][:]
x = data.groups['centerlines'].variables['x'][:]
y = data.groups['centerlines'].variables['y'][:]
cl_ids = data.groups['centerlines'].variables['cl_id'][:]
dist = data.groups['centerlines'].variables['dist_out_all'][:]
ends = con.groups['centerlines'].variables['end_reach'][:]
con_rchs = con.groups['centerlines'].variables['reach_id'][:]
con_cl_ids = con.groups['centerlines'].variables['cl_id'][:]

#calculate starting stream order. 
strm_order = np.zeros(len(paths))
normalize = np.where(paths > 0)[0] 
strm_order[normalize] = (np.round(np.log(paths[normalize])))+1
strm_order[np.where(paths == 0)] = -9999

#find path segments. 
path_segs = find_path_segs(order,paths)

if 'stream_order' in data.groups['centerlines'].variables.keys():
    data.groups['centerlines'].variables['stream_order'][:] = strm_order
    data.groups['centerlines'].variables['path_segments'][:] = path_segs
    data.close()
else:
    stream_order = data.groups['centerlines'].createVariable(
        'stream_order', 'i4', ('num_points',), fill_value=-9999.)
    data.groups['centerlines'].variables['stream_order'][:] = strm_order
    path_segments = data.groups['centerlines'].createVariable(
        'path_segments', 'i8', ('num_points',), fill_value=-9999.)
    data.groups['centerlines'].variables['path_segments'][:] = path_segs
    data.close()


print('Done')

###############################################################################
###PLOTS
# import matplotlib.pyplot as plt

# good = np.where(strm_order>0)[0]
# plt.figure(1)
# plt.scatter(x[good],y[good],c=strm_order[good],cmap = 'rainbow', s = 5)
# plt.show()

# plt.figure(2)
# plt.scatter(x,y,c='blue', s = 5)
# plt.scatter(x[pth],y[pth],c='gold', s = 5)
# plt.scatter(x[pth[junc[1]:junc[2]+1]],y[pth[junc[1]:junc[2]+1]], c='magenta', s=5)
# plt.scatter(x[pth[junc[3]:junc[4]+1]],y[pth[junc[3]:junc[4]+1]], c='green', s=5)
# # plt.scatter(x[pth[junc]],y[pth[junc]],c='black', s = 5)
# plt.show()

# plt.figure(3)
# junc=np.where(ends == 3)[0]
# plt.scatter(x,y,c=path_segs, cmap = 'rainbow', s = 5)
# plt.scatter(x[junc],y[junc],c='black', s = 5)
# plt.show()

# one = np.where(order == 1)[0]
# np.unique(paths[one])

# plt.figure(4)
# junc=np.where(ends == 3)[0]
# plt.scatter(x,y,c=path_segs, cmap = 'rainbow', s = 5)
# plt.scatter(x[seg],y[seg],c='black', s = 5)
# plt.show()

# df = pd.DataFrame(np.array([x,y,path_segs,new_strm_order,rchs[0,:]]).T)
