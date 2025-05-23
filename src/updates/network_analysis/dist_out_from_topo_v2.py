import os
main_dir = os.getcwd()
import numpy as np
import netCDF4 as nc
import geopandas as gp
import pandas as pd
import argparse
import sys
import time
from itertools import chain
import matplotlib.pyplot as plt

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
# parser.add_argument("basin", help="<Required> Level Two Pfafstetter Basin (i.e. 74 or 'All' for the whole region)", type = str)
args = parser.parse_args()

region = args.region
version = args.version
# basin = args.basin

### debugging 
# region = 'NA'
# basin = 'All'
# version = 'v18'

nc_fn = main_dir+'/data/outputs/Reaches_Nodes/'\
    +version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
netcdf = nc.Dataset(nc_fn, 'r+') #not sure yet if I will use this script to update netcdf or not... 

reaches = np.array(netcdf['/reaches/reach_id'][:])
n_rch_up = np.array(netcdf['/reaches/n_rch_up'][:])
n_rch_dn = np.array(netcdf['/reaches/n_rch_down'][:])
rch_len = np.array(netcdf['/reaches/reach_length'][:])
rch_x = np.array(netcdf['/reaches/x'][:])
rch_y = np.array(netcdf['/reaches/y'][:])
rch_dist = np.array(netcdf['/reaches/dist_out'][:])
rch_id_up = np.array(netcdf['/reaches/rch_id_up'][:])
rch_id_dn = np.array(netcdf['/reaches/rch_id_dn'][:])
nodes = np.array(netcdf['/nodes/node_id'][:])
node_rchs = np.array(netcdf['/nodes/reach_id'][:])
node_len = np.array(netcdf['/nodes/node_length'][:])
node_dist = np.array(netcdf['/nodes/dist_out'][:])

# if len(reaches) != len(nc_reach_id):
#     print('!!! Reaches in NetCDF not equal to GPKG !!!')
#     sys.exit()

print('Calculating DistOut from Topology')
dist_out = np.repeat(-9999, len(reaches)).astype(np.float64) #filler array for new outlet distance. 
flag = np.zeros(len(reaches))
outlets = reaches[np.where(n_rch_dn == 0)[0]]
start_rchs = np.array([outlets[0]]) #start with any outlet first. 
loop = 1
### While loop 
while len(start_rchs) > 0:
    #for loop to go through all start_rchs, which are the upstream neighbors of 
    #the previously updated reaches. The first reach is any outlet. 
    # print('LOOP:',loop, start_rchs)
    up_ngh_list = []
    for r in list(range(len(start_rchs))):
        rch = np.where(reaches == start_rchs[r])[0]
        if n_rch_dn[rch] == 0:
            dist_out[rch] = rch_len[rch]
            up_nghs = rch_id_up[:,rch]; up_nghs = up_nghs[up_nghs>0]
            up_ngh_list.append(up_nghs)
            # loop=loop+1
        else:
            dn_nghs = rch_id_dn[:,rch]; dn_nghs = dn_nghs[dn_nghs>0]
            dn_dist = np.array([dist_out[np.where(reaches == n)[0]][0] for n in dn_nghs])
            if min(dn_dist) == -9999:
                #set condition to start at next outlet. multichannel cases. 
                flag[rch] = 1
                # outlets = reaches[np.where((n_rch_dn == 0) & (dist_out == -9999))[0]]
                # start_rchs = np.array([outlets[0]])
            else:
                add_val = max(dn_dist)
                dist_out[rch] = rch_len[rch]+add_val
                up_nghs = rch_id_up[:,rch]; up_nghs = up_nghs[up_nghs>0]
                up_ngh_list.append(up_nghs) 
    #formatting next start reaches.         
    up_ngh_arr = np.array(list(chain.from_iterable(up_ngh_list)))
    start_rchs = np.unique(up_ngh_arr)
    #if no more upstream neighbors move to next outlet. 
    if len(start_rchs) == 0:
        outlets = reaches[np.where((n_rch_dn == 0) & (dist_out == -9999))[0]]
        #a case where all downstream reaches have filled but not all upstream.
        if len(outlets) == 0 and min(dist_out) > -9999:
            start_rchs = np.array([])
        elif len(outlets) == 0 and min(dist_out) == -9999:
            #find reach with downstream distances filled but a value of -9999
            print('!!! PROBLEM !!! --> No more upstream reaches, but still -9999 values in outlet distance')
            break
        else:
            start_rchs = np.array([outlets[0]])
    loop = loop+1
    if loop > 5*len(reaches):
        print('!!! LOOP STUCK !!!')
        break

# update node levels based on new dist out. 
print('Updating Nodes')
node_dist_out = np.copy(node_dist)
for r in list(range(len(reaches))):
    nds = np.where(node_rchs == reaches[r])[0]
    sort_nodes = np.argsort(nodes[nds])
    base_val = dist_out[r] - rch_len[r]
    node_cs = np.cumsum(node_len[nds[sort_nodes]])
    node_dist_out[nds[sort_nodes]] = node_cs+base_val

print('Updating NetCDF')
netcdf['/reaches/dist_out'][:] = dist_out
netcdf['/nodes/dist_out'][:] = node_dist_out
netcdf.close()
print('Done')
