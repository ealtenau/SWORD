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
parser.add_argument("basin", help="<Required> Level Two Pfafstetter Basin (i.e. 74)", type = str)

args = parser.parse_args()

region = args.region
version = args.version
basin = args.basin

# region = 'SA'
# basin = 'All'
# version = 'v17'

# testing paths:
# rch_shp_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/gpkg copy/na_sword_reaches_v17_orig.gpkg'
# rch_shp_fn ='/Users/ealtenau/Documents/SWORD_Dev/update_requests/v17/SA/topo_fixes/sa_sword_reaches_hb62_v17_FG1_LSFix_MS_TopoFix_Manual.shp'
# csv_fn ='/Users/ealtenau/Documents/SWORD_Dev/update_requests/v17/NA/na_sword_reaches_hb82_v17_rev_LS_FI.csv'

nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'\
    +version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'

if basin == 'All':
    rch_shp_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'\
        +version+'/gpkg/'+region.lower()+'_sword_reaches_'+version+'.gpkg'
    outdir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Topology/'+version+'/'+region+\
        '/dist_out_updates/'+region.lower()+'_sword_reaches_'+version+'_distout_update.gpkg'
else:
    rch_shp_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'\
        +version+'/shp/'+region+'/'+region.lower()+'_sword_reaches_hb'+basin+'_'+version+'.shp'
    outdir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Topology/'+version+'/'+region+\
        '/dist_out_updates/'+region.lower()+'_sword_reaches_hb'+basin+'_'+version+'_distout_update.gpkg'

shp = gp.read_file(rch_shp_fn)
netcdf = nc.Dataset(nc_fn) #not sure yet if I will use this script to update netcdf or not... 

reaches = np.array(shp['reach_id'])
n_rch_up = np.array(shp['n_rch_up'])
n_rch_dn = np.array(shp['n_rch_dn'])
rch_len = np.array(shp['reach_len'])
rch_x = np.array(shp['x'])
rch_y = np.array(shp['y'])

# reaches = np.array(netcdf['/reaches/reach_id'][:])
# n_rch_up = np.array(netcdf['/reaches/n_rch_up'][:])
# n_rch_dn = np.array(netcdf['/reaches/n_rch_down'][:])
# rch_len = np.array(netcdf['/reaches/reach_length'][:])
# rch_x = np.array(netcdf['/reaches/x'][:])
# rch_y = np.array(netcdf['/reaches/y'][:])
nc_rch_id_up = np.array(netcdf['/reaches/rch_id_up'][:])
nc_rch_id_dn = np.array(netcdf['/reaches/rch_id_dn'][:])
nc_reach_id = np.array(netcdf['/reaches/reach_id'][:])
shp_indexes = np.where(np.in1d(nc_reach_id, reaches) == True)[0]
rch_id_up = nc_rch_id_up[:,shp_indexes]
rch_id_dn = nc_rch_id_dn[:,shp_indexes]
netcdf.close()

if len(reaches) != len(nc_reach_id):
    print('!!! Reaches in NetCDF not equal to GPKG !!!')
    sys.exit()

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


# print('Updating NetCDF')

print('Done')
shp['dist_out2'] = dist_out
shp.to_file(outdir, driver='GPKG', layer='reaches')
print('Geopackage File:', outdir)

# good = np.where(dist_out > -9999)[0]
# plt.scatter(rch_x, rch_y, c='black', s=3)
# plt.scatter(rch_x[good], rch_y[good], c=dist_out[good], cmap='rainbow', s=3)
# plt.show()

