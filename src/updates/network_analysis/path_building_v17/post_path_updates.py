"""
Filtering SWORD path variables (post_path_updates.py).
===============================================================

This scripts applies filters to the new path variables added
to the SWORD v17 database.  

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python post_path_updates.py NA v17

""" 

import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import netCDF4 as nc
import argparse
from scipy import stats

###############################################################################
###########################  FUNCTIONS  #######################################
###############################################################################

def side_chan_segs(cl_rchs, main_side, path_segs, reaches, dist_out):
    """
    Adds unique segment values to the side channel network.

    Parameters
    ----------
    cl_rchs: numpy.array()
        SWORD centerline reach IDs.
    main_side: numpy.array()
        Main-Side network flag.
    path_segs: numpy.array()
        Unique IDs given to river segments between junctions.
    reaches: numpy.array()
        SWORD reach IDs. 
    dist_out: numpy.array()
        SWORD distance from outlet at reach scale. 
        
    Returns
    -------
    new_segs: numpy.array()
        Updated segment values with side channels included. 
        
    """
    
    # ngh_matrix = np.zeros(cl_rchs.shape)
    # count=1
    new_segs = np.copy(path_segs)
    seg_cnt = np.max(path_segs)
    side_chans = reaches[np.where(main_side == 1)[0]]
    if len(side_chans) > 0:
        
        #getting distance from outlet for side channels. 
        side_dist = np.array([dist_out[np.where(reaches==r)] for r in side_chans])
        
        #creating flag to keep track of covered reaches. 
        flag = np.zeros(len(reaches))
        flag[np.where(main_side == 0)] = 1
        
        #determining side channel segment values. 
        start_rch = side_chans[np.where(side_dist == np.min(side_dist))[0]][0]
        start_pt = np.where(cl_rchs[0,:] == start_rch)[0]
        loop = 1
        check = len(side_chans)+500
        while len(side_chans) > 0:
            # print(loop, start_rch)
            side_chans = np.delete(side_chans, np.where(side_chans == start_rch)[0])
            flag[np.where(reaches == start_rch)] = 1
            new_segs[np.where(reaches == start_rch)] = seg_cnt
            
            nghs = cl_rchs[1::,start_pt]
            nghs = nghs[nghs>0]
            ngh_flag = np.array([np.max(flag[np.where(reaches==n)]) for n in nghs])
            nghs = nghs[ngh_flag==0]

            if len(nghs) == 1:
                side_dist = np.array([dist_out[np.where(reaches==r)] for r in side_chans])
                start_rch = nghs[0]
                start_pt = np.where(cl_rchs[0,:] == start_rch)[0]
                loop=loop+1
            else:
                if len(side_chans) == 0:
                    loop = loop+1
                    continue
                else:
                    side_dist = np.array([dist_out[np.where(reaches==r)] for r in side_chans])
                    start_rch = side_chans[np.where(side_dist == np.min(side_dist))[0]][0]
                    start_pt = np.where(cl_rchs[0,:] == start_rch)[0]
                    seg_cnt = seg_cnt+1
                    loop=loop+1
            if loop > check:
                print('LOOP STUCK', start_rch)
                break

    return new_segs

###############################################################################

def define_network_regions(subpaths, subreaches, cl_rchs, basin):
    """
    Calculates and assigns a unique number to each connected 
    river network in the SWORD database.

    Parameters
    ----------
    subpaths: numpy.array()
        Path frequency array subset to a level 2 basin. 
    subreaches: numpy.array()
        Reach ID array subset to a level 2 basin. 
    cl_rchs: numpy.array()
        SWORD centerline reach IDs. 
    basin: int
         Pfafstetter level 2 basin. 
        
    Returns
    -------
    network: numpy.array()
        Unique values associated with connected river networks.  
        
    """

    unq_paths = np.unique(subpaths)
    unq_paths = unq_paths[unq_paths>0]
    start_path = np.min(unq_paths)
    cnt = 1
    flag = np.zeros(len(subreaches))
    flag[np.where(subpaths == 0)] = 1
    network = np.zeros(len(subreaches))
    check = len(unq_paths) + 500
    loop = 1
    #find all neighbors associated with the start reach
    #until no more neighbors are found. 
    while min(flag) == 0:
        # print(loop, start_path)
        path_rchs = subreaches[np.in1d(subpaths,start_path)]
        cl_inds = np.in1d(cl_rchs[0,:], path_rchs)
        nghs = cl_rchs[1::,cl_inds]
        nghs = np.unique(nghs[nghs>0])
        nghs_l2 = np.array([int(str(n)[0:2]) for n in nghs])
        nghs = nghs[nghs_l2 == basin]
        ngh_flag = np.array([flag[np.where(subreaches == n)] for n in nghs])
        nghs = nghs[np.where(ngh_flag==0)[0]]
        if len(nghs) > 0:
            ngh_paths = np.unique(subpaths[np.in1d(subreaches, nghs)])
            ngh_paths = ngh_paths[ngh_paths>0]
            network[np.in1d(subpaths, ngh_paths)] = cnt 
            flag[np.in1d(subpaths, ngh_paths)] = 1
            start_path = ngh_paths 
            loop = loop+1
        else:
            if min(flag) == 0:
                start_path = np.min(subpaths[np.where(flag == 0)])
                cnt = cnt+1
                network[np.in1d(subpaths, start_path)] = cnt 
                flag[np.in1d(subpaths, start_path)] = 1
                loop = loop+1
            else:
                continue

        if loop > check:
                print('LOOP STUCK')
                break
    
    return network

###############################################################################
################################  MAIN  #######################################
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("region", help="continental region", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

sword_dir = main_dir+'/data/outputs/Reaches_Nodes/'\
    +version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
con_dir = main_dir+'/data/outputs/Reaches_Nodes/'+version+\
    '/reach_geometry/'+region.lower()+'_sword_'+version+'_connectivity.nc'

#read data. 
conn = nc.Dataset(con_dir)
sword = nc.Dataset(sword_dir,'r+')

#assign relevant data to arrays. 
###centerline attributes
cl_rchs = conn.groups['centerlines'].variables['reach_id'][:]
cl_nodes = conn.groups['centerlines'].variables['node_id'][:]
cl_lon = conn.groups['centerlines'].variables['x'][:]
cl_lat = conn.groups['centerlines'].variables['y'][:]
cl_index = conn.groups['centerlines'].variables['cl_id'][:]
###reach attributes
main_side = sword.groups['reaches'].variables['main_side'][:]
path_segs = sword.groups['reaches'].variables['path_segs'][:]
reaches = sword.groups['reaches'].variables['reach_id'][:]
dist_out =  sword.groups['reaches'].variables['dist_out'][:]
path_order = sword.groups['reaches'].variables['path_order'][:]
path_freq = sword.groups['reaches'].variables['path_freq'][:]
x = sword.groups['reaches'].variables['x'][:]
y = sword.groups['reaches'].variables['y'][:]
Type = np.array([int(str(r)[-1]) for r in reaches])
end_rch = sword.groups['reaches'].variables['end_reach'][:]
# strm_order = sword.groups['reaches'].variables['stream_order'][:]
# main_side[np.where(main_side == 2)] = 0
###node attributes.
nodes_main_side = sword.groups['nodes'].variables['main_side'][:]
nodes_strm_order = sword.groups['nodes'].variables['stream_order'][:]
nodes_path_segs = sword.groups['nodes'].variables['path_segs'][:]
node_rchs = sword.groups['nodes'].variables['reach_id'][:]
nx = sword.groups['nodes'].variables['x'][:]
ny = sword.groups['nodes'].variables['y'][:]

print('Updating Side Channel Path Segments')
side_segs = side_chan_segs(cl_rchs, main_side, path_segs, reaches, dist_out)

print('Updating Main-Side and Stream Order')
basin_networks = np.zeros(len(reaches))
new_main_side = np.copy(main_side)
strm_order_all = np.zeros(len(reaches))
level2 = np.array([int(str(r)[0:2]) for r in reaches])
unq_l2 = np.unique(level2)
for ind in list(range(len(unq_l2))):
    print(unq_l2[ind])
    l2 = np.where(level2 == unq_l2[ind])[0]
    
    #subsetting data to a level 2 basin scale. 
    subpaths = path_order[l2]
    subreaches = reaches[l2]
    submain_side = main_side[l2]
    subtype = Type[l2]
    subx = x[l2]
    suby = y[l2]
    subpath_freq = path_freq[l2]
    subdist = dist_out[l2]
    subsegs = side_segs[l2]
    subends = end_rch[l2]

    #defining connected networks. 
    network = define_network_regions(subpaths, subreaches, cl_rchs, unq_l2[ind])
    basin_networks[l2] = network

    #updating main-side channel attribute. 
    unq_net = np.unique(network)
    unq_net = unq_net[unq_net>0]
    deltas = np.copy(submain_side)
    for net in list(range(len(unq_net))):
        ntwk = np.where(network == unq_net[net])[0]
        outlets = len(np.where(subends[ntwk] == 2)[0])
        if outlets > 1:
            index = np.where((subtype[ntwk] == 5) & (submain_side[ntwk] == 0))[0]
            unq_paths = np.unique(subpaths[ntwk[index]]) #was subpaths
            unq_paths = unq_paths[1::]
            segind = np.where(np.in1d(subpaths[ntwk], unq_paths)==True)[0]
            unq_segs = np.unique(subsegs[ntwk[segind]])
            for s in list(range(len(unq_segs))):
                seg = np.where(subsegs == unq_segs[s])[0]
                seg_type = stats.mode(subtype[seg])
                if seg_type[0] >= 5:
                    deltas[seg] = 2

    new_main_side[l2] = deltas
    
    #recalculating stream order. 
    strm_order = np.zeros(len(subpath_freq))
    normalize = np.where(deltas == 0)[0] 
    strm_order[normalize] = (np.round(np.log(subpath_freq[normalize])))+1
    strm_order[np.where(deltas > 0)] = -9999

    #filter stream order. 
    unq_pths = np.unique(subpaths[np.where(deltas == 0)[0]])
    unq_pths = unq_pths[::-1]
    unq_pths = unq_pths[unq_pths>0]
    for p in list(range(len(unq_pths))):
        pth = np.where(subpaths == unq_pths[p])[0]
        sort_inds = np.argsort(subdist[pth])
        diff = np.diff(strm_order[pth[sort_inds]])
        wrong = np.where(diff > 0)[0] 
        if len(wrong) > 0:
            # print(p, unq_pths[p])
            max_break = wrong[-1]+1
            
            if max_break == max(sort_inds):
                max_break = max_break-1
            
            new_val = strm_order[pth[sort_inds[max_break+1]]]
            strm_order[pth[sort_inds[0:max_break]]] = new_val
    
    strm_order[np.where(deltas > 0)] = -9999
    strm_order_all[l2] = strm_order
    
#########################################
#########################################    

print('Updating Node Attributes')
nodes_strm_order_all = np.copy(nodes_strm_order)
nodes_side_segs = np.copy(nodes_path_segs)
nodes_new_main_side = np.copy(nodes_main_side)
nodes_networks = np.zeros(len(nodes_main_side))
for r in list(range(len(reaches))):
    nds = np.where(node_rchs == reaches[r])[0]
    nodes_strm_order_all[nds] = strm_order_all[r]
    nodes_side_segs[nds] = side_segs[r]
    nodes_new_main_side[nds] = new_main_side[r]
    nodes_networks[nds] = basin_networks[r]

rch_nan = np.where(new_main_side == 1)[0]
node_nan = np.where(nodes_new_main_side == 1)[0]

print('Updating NetCDF')
sword.groups['reaches'].variables['stream_order'][:] = strm_order_all
sword.groups['reaches'].variables['path_segs'][:] = side_segs
sword.groups['reaches'].variables['main_side'][:] = new_main_side
sword.groups['reaches'].variables['path_order'][rch_nan] = -9999
sword.groups['reaches'].variables['path_freq'][rch_nan] = -9999
sword.groups['nodes'].variables['stream_order'][:] = nodes_strm_order_all
sword.groups['nodes'].variables['path_segs'][:] = nodes_side_segs
sword.groups['nodes'].variables['main_side'][:] = nodes_new_main_side
sword.groups['nodes'].variables['path_order'][node_nan] = -9999
sword.groups['nodes'].variables['path_freq'][node_nan] = -9999

if 'network' in sword.groups['reaches'].variables.keys():
    sword.groups['reaches'].variables['network'][:] = basin_networks
    sword.groups['nodes'].variables['network'][:] = nodes_networks
else:
    sword.groups['reaches'].createVariable(
        'network', 'i4', ('num_reaches',), fill_value=-9999.)
    sword.groups['nodes'].createVariable(
        'network', 'i4', ('num_nodes',), fill_value=-9999.)
    #populate new variables.
    sword.groups['reaches'].variables['network'][:] = basin_networks
    sword.groups['nodes'].variables['network'][:] = nodes_networks

sword.close()
conn.close()

###############################################################################
### PLOTS
# import matplotlib.pyplot as plt

# plt.scatter(x[l2], y[l2], c=network, cmap='rainbow', s = 3)
# plt.show()

# plt.scatter(subx[normalize], suby[normalize], c=strm_order[normalize], cmap='rainbow', s = 3)
# plt.show()

# other = np.where(nodes_new_main_side == 2)[0]
# side = np.where(nodes_new_main_side == 1)[0]
# plt.scatter(nx, ny, c = 'blue', s = 5)
# plt.scatter(nx[other], ny[other], c = 'gold', s = 5)
# plt.scatter(nx[side], ny[side], c = 'magenta', s = 5)
# plt.show()

# other = np.where(deltas == 2)[0]
# side = np.where(deltas == 1)[0]
# plt.scatter(subx, suby, c = 'blue', s = 5)
# plt.scatter(subx[other], suby[other], c = 'gold', s = 5)
# plt.scatter(subx[side], suby[side], c = 'magenta', s = 5)
# plt.show()

# plt.scatter(x, y, c=basin_networks, cmap='rainbow', s = 3)
# plt.show()