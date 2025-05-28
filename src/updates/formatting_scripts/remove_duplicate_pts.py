# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import time
import pandas as pd
import argparse
import geopandas as gp
import src.updates.sword_utils as swd

###############################################################################

def calc_rch_facc(nodes, centerlines, cl_rch):
    unq_nodes = np.unique(centerlines.node_id[0,cl_rch])
    facc = np.zeros(len(cl_rch))
    for n in list(range(len(unq_nodes))):
        cl_ind = np.where(centerlines.node_id[0,cl_rch] == unq_nodes[n])[0]
        nd_ind = np.where(nodes.id == unq_nodes[n])[0]
        facc[cl_ind] = nodes.facc[nd_ind]
    return facc

###############################################################################

start_all = time.time()

#read in netcdf data. 
parser = argparse.ArgumentParser()
parser.add_argument("region", help="continental region", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

paths = swd.prepare_paths(main_dir, region, version)
sword_fn = paths['geom_dir']+paths['geom_fn']

#read data.
centerlines, nodes, reaches = swd.read_nc(sword_fn)

df = pd.DataFrame(np.array([centerlines.x, centerlines.y]).T)
dups = df.duplicated()
rmv = np.where(dups == True)[0]

rch_updates = np.unique(centerlines.reach_id[0,rmv]) #len(rch_updates)
node_updates = np.unique(centerlines.node_id[0,rmv]) #len(node_updates)

#remove duplicated centerline points
print('Removing Duplicated Centerline Points')
start = time.time()
centerlines.cl_id = np.delete(centerlines.cl_id, rmv, axis=0)
centerlines.x = np.delete(centerlines.x, rmv, axis=0)
centerlines.y = np.delete(centerlines.y, rmv, axis=0)
centerlines.reach_id = np.delete(centerlines.reach_id, rmv, axis=1)
centerlines.node_id = np.delete(centerlines.node_id, rmv, axis=1)
end = time.time()
print('Finished Removing Duplicated Points in: '+str(np.round((end-start),2))+' sec')

#loop through and update attributes for affected reaches and nodes. 
#reach attributes to update: 'cl_ids', 'x', 'x_min', 'x_max', 'y', 'y_min', 'y_max', 'reach_length', 'dist_out'
#node attributes to update: 'cl_ids', 'x', 'y', 'node_length', 'dist_out'
print('Starting Reach and Nodes Updates')
start = time.time()
for ind in list(range(len(rch_updates))):
    print(ind, len(rch_updates)-1)
    cl_rch = np.where(centerlines.reach_id[0,:] == rch_updates[ind])[0]
    rch_ind = np.where(reaches.id == rch_updates[ind])[0]

    if len(cl_rch) == 0:
        print('reach removed completely', rch_updates[ind])
        node_ind = np.where(nodes.reach_id == rch_updates[ind])[0]
        nodes.id = np.delete(nodes.id, node_ind, axis = 0)
        nodes.cl_id = np.delete(nodes.cl_id, node_ind, axis = 1)
        nodes.x = np.delete(nodes.x, node_ind, axis = 0)
        nodes.y = np.delete(nodes.y, node_ind, axis = 0)
        nodes.len = np.delete(nodes.len, node_ind, axis = 0)
        nodes.wse = np.delete(nodes.wse, node_ind, axis = 0)
        nodes.wse_var = np.delete(nodes.wse_var, node_ind, axis = 0)
        nodes.wth = np.delete(nodes.wth, node_ind, axis = 0)
        nodes.wth_var = np.delete(nodes.wth_var, node_ind, axis = 0)
        nodes.grod = np.delete(nodes.grod, node_ind, axis = 0)
        nodes.grod_fid = np.delete(nodes.grod_fid, node_ind, axis = 0)
        nodes.hfalls_fid = np.delete(nodes.hfalls_fid, node_ind, axis = 0)
        nodes.nchan_max = np.delete(nodes.nchan_max, node_ind, axis = 0)
        nodes.nchan_mod = np.delete(nodes.nchan_mod, node_ind, axis = 0)
        nodes.dist_out = np.delete(nodes.dist_out, node_ind, axis = 0)
        nodes.reach_id = np.delete(nodes.reach_id, node_ind, axis = 0)
        nodes.facc = np.delete(nodes.facc, node_ind, axis = 0)
        nodes.lakeflag = np.delete(nodes.lakeflag, node_ind, axis = 0)
        nodes.wth_coef = np.delete(nodes.wth_coef, node_ind, axis = 0)
        nodes.ext_dist_coef = np.delete(nodes.ext_dist_coef, node_ind, axis = 0)
        nodes.max_wth = np.delete(nodes.max_wth, node_ind, axis = 0)
        nodes.meand_len = np.delete(nodes.meand_len, node_ind, axis = 0)
        nodes.river_name = np.delete(nodes.river_name, node_ind, axis = 0)
        nodes.manual_add = np.delete(nodes.manual_add, node_ind, axis = 0)
        nodes.sinuosity = np.delete(nodes.sinuosity, node_ind, axis = 0)
        nodes.edit_flag = np.delete(nodes.edit_flag, node_ind, axis = 0)
        nodes.trib_flag = np.delete(nodes.trib_flag, node_ind, axis = 0)

        reaches.id = np.delete(reaches.id, rch_ind, axis = 0)
        reaches.cl_id = np.delete(reaches.cl_id, rch_ind, axis = 1)
        reaches.x = np.delete(reaches.x, rch_ind, axis = 0)
        reaches.x_min = np.delete(reaches.x_min, rch_ind, axis = 0)
        reaches.x_max = np.delete(reaches.x_max, rch_ind, axis = 0)
        reaches.y = np.delete(reaches.y, rch_ind, axis = 0)
        reaches.y_min = np.delete(reaches.y_min, rch_ind, axis = 0)
        reaches.y_max = np.delete(reaches.y_max, rch_ind, axis = 0)
        reaches.len = np.delete(reaches.len, rch_ind, axis = 0)
        reaches.wse = np.delete(reaches.wse, rch_ind, axis = 0)
        reaches.wse_var = np.delete(reaches.wse_var, rch_ind, axis = 0)
        reaches.wth = np.delete(reaches.wth, rch_ind, axis = 0)
        reaches.wth_var = np.delete(reaches.wth_var, rch_ind, axis = 0)
        reaches.slope = np.delete(reaches.slope, rch_ind, axis = 0)
        reaches.rch_n_nodes = np.delete(reaches.rch_n_nodes, rch_ind, axis = 0)
        reaches.grod = np.delete(reaches.grod, rch_ind, axis = 0)
        reaches.grod_fid = np.delete(reaches.grod_fid, rch_ind, axis = 0)
        reaches.hfalls_fid = np.delete(reaches.hfalls_fid, rch_ind, axis = 0)
        reaches.lakeflag = np.delete(reaches.lakeflag, rch_ind, axis = 0)
        reaches.nchan_max = np.delete(reaches.nchan_max, rch_ind, axis = 0)
        reaches.nchan_mod = np.delete(reaches.nchan_mod, rch_ind, axis = 0)
        reaches.dist_out = np.delete(reaches.dist_out, rch_ind, axis = 0)
        reaches.n_rch_up = np.delete(reaches.n_rch_up, rch_ind, axis = 0)
        reaches.n_rch_down = np.delete(reaches.n_rch_down, rch_ind, axis = 0)
        reaches.rch_id_up = np.delete(reaches.rch_id_up, rch_ind, axis = 1)
        reaches.rch_id_down = np.delete(reaches.rch_id_down, rch_ind, axis = 1)
        reaches.max_obs = np.delete(reaches.max_obs, rch_ind, axis = 0)
        reaches.orbits = np.delete(reaches.orbits, rch_ind, axis = 1)
        reaches.facc = np.delete(reaches.facc, rch_ind, axis = 0)
        reaches.iceflag = np.delete(reaches.iceflag, rch_ind, axis = 1)
        reaches.max_wth = np.delete(reaches.max_wth, rch_ind, axis = 0)
        reaches.river_name = np.delete(reaches.river_name, rch_ind, axis = 0)
        reaches.low_slope = np.delete(reaches.low_slope, rch_ind, axis = 0)
        reaches.edit_flag = np.delete(reaches.edit_flag, rch_ind, axis = 0)
        reaches.trib_flag = np.delete(reaches.trib_flag, rch_ind, axis = 0)

        #removing residual neighbors with deleted reach id in centerline and reach groups. 
        cl_ind1 = np.where(centerlines.reach_id[0,:] == rch_updates[ind])[0]
        cl_ind2 = np.where(centerlines.reach_id[1,:] == rch_updates[ind])[0]
        cl_ind3 = np.where(centerlines.reach_id[2,:] == rch_updates[ind])[0]
        cl_ind4 = np.where(centerlines.reach_id[3,:] == rch_updates[ind])[0]
        if len(cl_ind1) > 0:
            centerlines.reach_id[0,cl_ind1] = 0
        if len(cl_ind2) > 0:
            centerlines.reach_id[1,cl_ind2] = 0
        if len(cl_ind3) > 0:
            centerlines.reach_id[2,cl_ind3] = 0
        if len(cl_ind4) > 0:
            centerlines.reach_id[3,cl_ind4] = 0

        rch_up_ind1 = np.where(reaches.rch_id_up[0,:] == rch_updates[ind])[0]
        rch_up_ind2 = np.where(reaches.rch_id_up[1,:] == rch_updates[ind])[0]
        rch_up_ind3 = np.where(reaches.rch_id_up[2,:] == rch_updates[ind])[0]
        rch_up_ind4 = np.where(reaches.rch_id_up[3,:] == rch_updates[ind])[0]
        if len(rch_up_ind1) > 0:
            reaches.rch_id_up[0,rch_up_ind1] = 0
        if len(rch_up_ind2) > 0:
            reaches.rch_id_up[1,rch_up_ind2] = 0
        if len(rch_up_ind3) > 0:
            reaches.rch_id_up[2,rch_up_ind3] = 0
        if len(rch_up_ind4) > 0:
            reaches.rch_id_up[3,rch_up_ind4] = 0

        rch_dn_ind1 = np.where(reaches.rch_id_down[0,:] == rch_updates[ind])[0]
        rch_dn_ind2 = np.where(reaches.rch_id_down[1,:] == rch_updates[ind])[0]
        rch_dn_ind3 = np.where(reaches.rch_id_down[2,:] == rch_updates[ind])[0]
        rch_dn_ind4 = np.where(reaches.rch_id_down[3,:] == rch_updates[ind])[0]
        if len(rch_dn_ind1) > 0:
            reaches.rch_id_down[0,rch_dn_ind1] = 0
        if len(rch_dn_ind2) > 0:
            reaches.rch_id_down[1,rch_dn_ind2] = 0
        if len(rch_dn_ind3) > 0:
            reaches.rch_id_down[2,rch_dn_ind3] = 0
        if len(rch_dn_ind4) > 0:
            reaches.rch_id_down[3,rch_dn_ind4] = 0
        # skip rest of the loop. 
        continue

    #calculate distance variables for centerline points. 
    facc = calc_rch_facc(nodes, centerlines, cl_rch)
    sort_ids = np.argsort(centerlines.cl_id[cl_rch])
    gdf = gp.GeoDataFrame(geometry=gp.points_from_xy(centerlines.x[cl_rch[sort_ids]], 
                                                     centerlines.y[cl_rch[sort_ids]]),crs="EPSG:4326").to_crs("EPSG:3857")
    diff = gdf.distance(gdf.shift(1)); diff[0] = 0
    dist = np.array(np.cumsum(diff))
    
    #update reach variables:
    diff_len = reaches.len[rch_ind] - np.max(dist)
    reaches.x[rch_ind] = np.median(centerlines.x[cl_rch])
    reaches.y[rch_ind] = np.median(centerlines.y[cl_rch])
    reaches.x_max[rch_ind] = np.max(centerlines.x[cl_rch])
    reaches.x_min[rch_ind] = np.min(centerlines.x[cl_rch])
    reaches.y_max[rch_ind] = np.max(centerlines.y[cl_rch])
    reaches.y_min[rch_ind] = np.min(centerlines.y[cl_rch])
    reaches.cl_id[0,rch_ind] = np.min(centerlines.cl_id[cl_rch])
    reaches.cl_id[1,rch_ind] = np.max(centerlines.cl_id[cl_rch])
    reaches.rch_n_nodes[rch_ind] = len(np.unique(centerlines.node_id[0,cl_rch]))
    reaches.len[rch_ind] = np.max(dist)
    reaches.dist_out[rch_ind] = reaches.dist_out[rch_ind]-diff_len

    nds = node_updates[np.in1d(node_updates, centerlines.node_id[0,cl_rch])]
    for idx in list(range(len(nds))):
        cl_nds = np.where(centerlines.node_id[0,cl_rch] == nds[idx])[0]
        ns = np.where(nodes.id == nds[idx])[0]

        #update reach variables:
        node_len = np.max(dist[cl_nds])-np.min(dist[cl_nds])
        node_diff_len = nodes.len[ns] - node_len
        nodes.x[ns] = np.median(centerlines.x[cl_rch[cl_nds]])
        nodes.y[ns] = np.median(centerlines.y[cl_rch[cl_nds]])
        nodes.cl_id[0,ns] = np.min(centerlines.cl_id[cl_rch[cl_nds]])
        nodes.cl_id[1,ns] = np.max(centerlines.cl_id[cl_rch[cl_nds]])
        nodes.len[ns] = node_len
        nodes.dist_out[ns] = nodes.dist_out[ns]-node_diff_len 

diff = np.in1d(nodes.id, np.unique(centerlines.node_id[0,:]))
rmv2 = np.where(diff == False)[0]
nodes.id = np.delete(nodes.id, rmv2, axis = 0)
nodes.cl_id = np.delete(nodes.cl_id, rmv2, axis = 1)
nodes.x = np.delete(nodes.x, rmv2, axis = 0)
nodes.y = np.delete(nodes.y, rmv2, axis = 0)
nodes.len = np.delete(nodes.len, rmv2, axis = 0)
nodes.wse = np.delete(nodes.wse, rmv2, axis = 0)
nodes.wse_var = np.delete(nodes.wse_var, rmv2, axis = 0)
nodes.wth = np.delete(nodes.wth, rmv2, axis = 0)
nodes.wth_var = np.delete(nodes.wth_var, rmv2, axis = 0)
nodes.grod = np.delete(nodes.grod, rmv2, axis = 0)
nodes.grod_fid = np.delete(nodes.grod_fid, rmv2, axis = 0)
nodes.hfalls_fid = np.delete(nodes.hfalls_fid, rmv2, axis = 0)
nodes.nchan_max = np.delete(nodes.nchan_max, rmv2, axis = 0)
nodes.nchan_mod = np.delete(nodes.nchan_mod, rmv2, axis = 0)
nodes.dist_out = np.delete(nodes.dist_out, rmv2, axis = 0)
nodes.reach_id = np.delete(nodes.reach_id, rmv2, axis = 0)
nodes.facc = np.delete(nodes.facc, rmv2, axis = 0)
nodes.lakeflag = np.delete(nodes.lakeflag, rmv2, axis = 0)
nodes.wth_coef = np.delete(nodes.wth_coef, rmv2, axis = 0)
nodes.ext_dist_coef = np.delete(nodes.ext_dist_coef, rmv2, axis = 0)
nodes.max_wth = np.delete(nodes.max_wth, rmv2, axis = 0)
nodes.meand_len = np.delete(nodes.meand_len, rmv2, axis = 0)
nodes.river_name = np.delete(nodes.river_name, rmv2, axis = 0)
nodes.manual_add = np.delete(nodes.manual_add, rmv2, axis = 0)
nodes.sinuosity = np.delete(nodes.sinuosity, rmv2, axis = 0)
nodes.edit_flag = np.delete(nodes.edit_flag, rmv2, axis = 0)
nodes.trib_flag = np.delete(nodes.trib_flag, rmv2, axis = 0)

#####################################################################

print('Writing NetCDF')
swd.discharge_attr_nc(reaches)
swd.write_nc(centerlines, reaches, nodes, region, sword_fn)

end_all = time.time()
print('Finished '+region+' Updates in: '+str(np.round((end_all-start_all)/3600,2))+' hrs')

end = time.time()
print('Finished Reach and Node Updates in: '+str(np.round((end-start)/60,2))+' min')
print('Cl Dimensions:', len(np.unique(centerlines.cl_id)), len(centerlines.cl_id))
print('Rch Dimensions:', len(np.unique(centerlines.reach_id[0,:])), len(np.unique(nodes.reach_id)), len(reaches.id))
print('Node Dimensions:', len(np.unique(centerlines.node_id[0,:])), len(np.unique(nodes.id)), len(nodes.id))
