import numpy as np
import netCDF4 as nc
import geopandas as gp
import pandas as pd
import argparse
import sys
import time
import matplotlib.pyplot as plt
from scipy import spatial as sp

start_all = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

# region = 'OC'
# version = 'v17'

nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'\
    +version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
outpath = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Topology/'+version+'/'+region+'/'

# sword = nc.Dataset(nc_fn,'r+')
sword = nc.Dataset(nc_fn)

reaches = np.array(sword.groups['reaches'].variables['reach_id'][:])
rch_lon = np.array(sword.groups['reaches'].variables['x'][:])
rch_lat = np.array(sword.groups['reaches'].variables['y'][:])
rch_id_up = np.array(sword.groups['reaches'].variables['rch_id_up'][:])
rch_id_dn = np.array(sword.groups['reaches'].variables['rch_id_dn'][:])
n_rch_up = np.array(sword.groups['reaches'].variables['n_rch_up'][:])
n_rch_dn = np.array(sword.groups['reaches'].variables['n_rch_down'][:])
rch_dist = np.array(sword.groups['reaches'].variables['dist_out'][:])
rch_ms = np.array(sword.groups['reaches'].variables['main_side'][:])
nodes = np.array(sword.groups['nodes'].variables['node_id'][:])
node_rchs = np.array(sword.groups['nodes'].variables['reach_id'][:])
nx = np.array(sword.groups['nodes'].variables['x'][:])
ny = np.array(sword.groups['nodes'].variables['y'][:])
node_dist = np.array(sword.groups['nodes'].variables['dist_out'][:])

#copies of pre-edit centerline and node ids. 
old_rch_dist = np.copy(rch_dist)
old_node_dist = np.copy(node_dist)

#only looking at side channel network. 
side_rchs = np.where(rch_ms > 0)[0]
unq_rchs = np.unique(reaches[side_rchs])

start = time.time()
print('Finding Reversed Side Reaches')   
incorrect_rchs = []
main_nghs = []
dist_nghs = []
for r in list(range(len(unq_rchs))):
    # print(r)
    #getting centerline and node dimension indexes for a reach. 
    rch_ind = np.where(reaches == unq_rchs[r])[0]
    node_ind = np.where(node_rchs == unq_rchs[r])[0]
    #finding the min and max centerline ids 
    nghs_dn = rch_id_dn[:,rch_ind]; nghs_dn = nghs_dn[nghs_dn>0]
    nghs_ms_dn = np.array([rch_ms[np.where(reaches == r)[0]] for r in nghs_dn])
    nghs_dist_dn = np.array([rch_dist[np.where(reaches == r)[0]] for r in nghs_dn])
    nghs_up = rch_id_up[:,rch_ind]; nghs_up = nghs_up[nghs_up>0]
    nghs_ms_up = np.array([rch_ms[np.where(reaches == r)[0]] for r in nghs_up])
    nghs_dist_up = np.array([rch_dist[np.where(reaches == r)[0]] for r in nghs_up])

    side_dn = np.where(nghs_ms_dn == 1)[0]
    side_up = np.where(nghs_ms_up == 1)[0]
    # two side channel neighbors
    if len(side_dn) > 0 and len(side_up) > 0:
        if max(nghs_dist_dn[side_dn]) > max(nghs_dist_up[side_up]):
            # print(r, 'cond 1: two side neighbors')
            #record reach if above conditions are met. 
            incorrect_rchs.append(unq_rchs[r])
            main_dn = np.where(nghs_ms_dn == 0)[0]
            if len(main_dn) > 0:
                if len(main_dn) > 1:
                    max_ind = np.where(nghs_dist_dn[main_dn] == max(nghs_dist_dn[main_dn]))[0]
                    main_nghs.append(nghs_dn[main_dn[max_ind]][0])
                    dist_nghs.append(nghs_dist_dn[main_dn[max_ind]][0][0])
                else:
                    main_nghs.append(nghs_dn[main_dn][0])
                    dist_nghs.append(nghs_dist_dn[main_dn][0][0])
            else:
                main_nghs.append(0)
                dist_nghs.append(0)
    # one side channel neighbor upstream
    elif len(side_dn) == 0 and len(side_up) > 0:
        if rch_dist[rch_ind] > max(nghs_dist_up[side_up]):
            # print(r, 'cond 2: one side neighbor upstream')
            #record reach if above conditions are met. 
            incorrect_rchs.append(unq_rchs[r])
            main_dn = np.where(nghs_ms_dn == 0)[0]
            if len(main_dn) > 0:
                if len(main_dn) > 1:
                    max_ind = np.where(nghs_dist_dn[main_dn] == max(nghs_dist_dn[main_dn]))[0]
                    main_nghs.append(nghs_dn[main_dn[max_ind]][0])
                    dist_nghs.append(nghs_dist_dn[main_dn[max_ind]][0][0])
                else:
                    main_nghs.append(nghs_dn[main_dn][0])
                    dist_nghs.append(nghs_dist_dn[main_dn][0][0])
            else:
                main_nghs.append(0)
                dist_nghs.append(0)
    # one side channel neighbor downstream
    elif len(side_dn) > 0 and len(side_up) == 0:
        if max(nghs_dist_dn[side_dn]) > rch_dist[rch_ind]:
            # print(r, 'cond 3: one side neighbor downstream')
            #record reach if above conditions are met. 
            incorrect_rchs.append(unq_rchs[r])
            main_dn = np.where(nghs_ms_dn == 0)[0]
            if len(main_dn) > 0:
                if len(main_dn) > 1:
                    max_ind = np.where(nghs_dist_dn[main_dn] == max(nghs_dist_dn[main_dn]))[0]
                    main_nghs.append(nghs_dn[main_dn[max_ind]][0])
                    dist_nghs.append(nghs_dist_dn[main_dn[max_ind]][0][0])
                else:
                    main_nghs.append(nghs_dn[main_dn][0])
                    dist_nghs.append(nghs_dist_dn[main_dn][0][0])
            else:
                main_nghs.append(0)
                dist_nghs.append(0)
    else:
        continue


incorrect_rchs = np.array(incorrect_rchs)
main_nghs = np.array(main_nghs)
dist_nghs = np.array(dist_nghs)

end = time.time()
print(str(np.round((end-start)/60,2))+' mins')

#writing csv files for reversed nodes and centerline ids.             
print('Writing CSV Files')
category = rch_ms[np.in1d(reaches, incorrect_rchs)]
rch_x = rch_lon[np.in1d(reaches, incorrect_rchs)]
rch_y = rch_lat[np.in1d(reaches, incorrect_rchs)]
df = pd.DataFrame({"reach_id": incorrect_rchs, 
                   "x": rch_x, 
                   "y": rch_y,
                   "end_reach": category, 
                   "main_nghs": main_nghs, 
                   "dist_nghs": dist_nghs})
df.to_csv(outpath+'distout_reversals.csv', index = False)

#updating the netcdf. 
# print('Updating the NetCDF')
# sword.groups['nodes'].variables['dist_out'][:] = xxx
# sword.groups['reaches'].variables['dist_out'][:] = xxx
# sword.close()

end_all = time.time()
print('*** '+region + ' Done in: '+str(np.round((end_all-start_all)/60,2))+' mins ***')
print('number of incorrect reaches:', len(incorrect_rchs))








# np.array([old_nodes[node_ind], nodes[node_ind]]).T
# np.array([old_cl_nodes[0,cl_ind], cl_nodes[0,cl_ind]]).T
# np.array([cl_ids[cl_ind], cl_nodes[0,cl_ind]]).T
# np.array([old_cl_ids[cl_ind], old_cl_nodes[0,cl_ind]]).T


# plt.scatter(cl_x[cl_ind[sort_cl_inds]], cl_y[cl_ind[sort_cl_inds]], c=old_cl_nodes[0,cl_ind[sort_cl_inds]], s = 5)
# plt.show()

# plt.scatter(cl_x[cl_ind[sort_cl_inds]], cl_y[cl_ind[sort_cl_inds]], c=cl_nodes[0,cl_ind[sort_cl_inds]], s = 5)
# plt.show()

# plt.scatter(nx[node_ind[sort_inds2]], ny[node_ind[sort_inds2]], c=old_nodes[node_ind[sort_inds2]], s = 5)
# plt.show()

# plt.scatter(nx[node_ind[sort_inds2]], ny[node_ind[sort_inds2]], c=nodes[node_ind[sort_inds2]], s = 5)
# plt.show()


# np.diff(old_cl_nodes[0,cl_ind[sort_cl_inds]])
# np.diff(cl_nodes[0,cl_ind[sort_cl_inds]])