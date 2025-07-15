"""
Updating Reach Topology (attach_topology_nc.py).
===============================================================

This script updates the SWORD netCDF topological attributes
based on manually updated topology in the SWORD reaches geopackage 
or shapefiles.  

The script can be run at a regional/continental scale
or a level two Pfafstetter basin scale. Command line arguments 
required are the two-letter region identifier (i.e. NA), 
SWORD version (i.e. v17), and the basin scale at which topology 
is to be updated (i.e. 74 or All).

Execution example (terminal):
    python path/to/attach_topology_nc.py NA v17 All

"""

import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import netCDF4 as nc
import geopandas as gp
import pandas as pd
import argparse
import sys
import time
from src.updates.sword import SWORD

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("basin", help="<Required> Level Two Pfafstetter Basin (i.e. 74 or All)", type = str)
args = parser.parse_args()

region = args.region
version = args.version
basin = args.basin

#read sword netcdf. 
sword = SWORD(main_dir, region, version)
sword.copy() #copies original file for version control. 

if basin == 'All':
    rch_shp_fn = sword.paths['gpkg_dir'] + sword.paths['gpkg_rch_fn']
else:
    rch_shp_fn = sword.paths['shp_dir'] + sword.paths['shp_rch_fn']
    rch_shp_fn = rch_shp_fn.replace("XX", basin)

#read vector data assign arrays. 
rch_shp = gp.read_file(rch_shp_fn)
reaches = np.array(rch_shp['reach_id'])
rch_id_up = np.array(rch_shp['rch_id_up'])
rch_id_dn = np.array(rch_shp['rch_id_dn'])
n_rch_up = np.array(rch_shp['n_rch_up'])
n_rch_dn = np.array(rch_shp['n_rch_dn'])
dist_out = np.array(rch_shp['dist_out'])
rch_len = np.array(rch_shp['reach_len'])
main_side = np.array(rch_shp['main_side'])
end_rch = np.array(rch_shp['end_reach'])

#make sure the netcdf and vector files are same dimensions. 
zero_out = np.where(np.in1d(sword.reaches.id, reaches) == True)[0]
cl_zero_out = np.where(np.in1d(sword.centerlines.reach_id[0,:], reaches) == True)[0]
if len(zero_out) < len(reaches):
    diff = len(reaches)-len(zero_out)
    diff_rchs = reaches[np.where(np.in1d(reaches, sword.reaches.id) == False)[0]]
    print('!WARNING!: netcdf has ' + str(diff) + ' less reaches than shapefile')
    print(diff_rchs)
    sys.exit()

#reset netcdf topology attributes to zero. 
sword.reaches.rch_id_up[:,zero_out] = 0
sword.reaches.rch_id_down[:,zero_out] = 0
sword.reaches.n_rch_up[zero_out] = 0
sword.reaches.n_rch_down[zero_out] = 0
sword.centerlines.reach_id[1::,cl_zero_out] = 0 

#update topology per reach. 
print('updating topology')
for ind in list(range(len(reaches))):
    # print(ind, reaches[ind], len(reaches)-1)    
    nc_ind = np.where(sword.reaches.id == reaches[ind])[0]
    cl_ind = np.where(sword.centerlines.reach_id[0,:] == reaches[ind])[0]
    cl_id_up = cl_ind[np.where(sword.centerlines.cl_id[cl_ind] == np.max(sword.centerlines.cl_id[cl_ind]))]
    cl_id_dn = cl_ind[np.where(sword.centerlines.cl_id[cl_ind] == np.min(sword.centerlines.cl_id[cl_ind]))]
    
    ###dist_out / main_side / end_reach
    #reaches
    sword.reaches.dist_out[nc_ind] = dist_out[ind]
    #nodes
    nds = np.where(sword.nodes.reach_id == reaches[ind])[0]
    sort_nodes = np.argsort(sword.nodes.id[nds])
    base_val = dist_out[ind] - rch_len[ind]
    node_cs = np.cumsum(sword.nodes.len[nds[sort_nodes]])
    sword.nodes.dist_out[nds[sort_nodes]] = node_cs+base_val 

    ###upstream
    if n_rch_up[ind] == 1:
        sword.reaches.rch_id_up[0,nc_ind] = int(rch_id_up[ind])
        sword.reaches.n_rch_up[nc_ind] = n_rch_up[ind]
        sword.centerlines.reach_id[1,cl_id_up] = int(rch_id_up[ind])
    if n_rch_up[ind] > 1:
        rup = np.array(rch_id_up[ind].split(),dtype=int)
        rup = rup.reshape(len(rup),1)
        sword.reaches.rch_id_up[0:len(rup),nc_ind] = rup
        sword.reaches.n_rch_up[nc_ind] = n_rch_up[ind]
        if n_rch_up[ind] > 3:
            sword.centerlines.reach_id[1:4,cl_id_up] = rup[0:3]
        else:
            sword.centerlines.reach_id[1:len(rup)+1,cl_id_up] = rup #sword.centerlines.reach_id[:,cl_id_up]
    ###downstream
    if n_rch_dn[ind] == 1:
        sword.reaches.rch_id_down[0,nc_ind] = int(rch_id_dn[ind])
        sword.reaches.n_rch_down[nc_ind] = n_rch_dn[ind]
        sword.centerlines.reach_id[1,cl_id_dn] = int(rch_id_dn[ind])
    if n_rch_dn[ind] > 1:
        rdn = np.array(rch_id_dn[ind].split(),dtype=int)
        rdn = rdn.reshape(len(rdn),1)
        sword.reaches.rch_id_down[0:len(rdn),nc_ind] = rdn
        sword.reaches.n_rch_down[nc_ind] = n_rch_dn[ind]
        if n_rch_dn[ind] > 3:
            sword.centerlines.reach_id[1:4,cl_id_dn] = rdn[0:3]
        else:
            sword.centerlines.reach_id[1:len(rdn)+1,cl_id_dn] = rdn #sword.centerlines.reach_id[:,cl_id_dn]
    
### save netcdf
sword.save_nc()

end = time.time()
print('DONE')
print(str(np.round((end-start)/60,2))+' mins')
