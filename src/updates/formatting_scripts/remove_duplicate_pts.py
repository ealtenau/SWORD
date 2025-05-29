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
from src.updates.sword import SWORD

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

sword = SWORD(main_dir, region, version)

df = pd.DataFrame(np.array([sword.centerlines.x, sword.centerlines.y]).T)
dups = df.duplicated()
rmv = np.where(dups == True)[0]

rch_updates = np.unique(sword.centerlines.reach_id[0,rmv]) #len(rch_updates)
node_updates = np.unique(sword.centerlines.node_id[0,rmv]) #len(node_updates)

#remove duplicated centerline points
print('Removing Duplicated Centerline Points')
start = time.time()
sword.centerlines.cl_id = np.delete(sword.centerlines.cl_id, rmv, axis=0)
sword.centerlines.x = np.delete(sword.centerlines.x, rmv, axis=0)
sword.centerlines.y = np.delete(sword.centerlines.y, rmv, axis=0)
sword.centerlines.reach_id = np.delete(sword.centerlines.reach_id, rmv, axis=1)
sword.centerlines.node_id = np.delete(sword.centerlines.node_id, rmv, axis=1)
end = time.time()
print('Finished Removing Duplicated Points in: '+str(np.round((end-start),2))+' sec')

#loop through and update attributes for affected reaches and sword.nodes. 
#reach attributes to update: 'cl_ids', 'x', 'x_min', 'x_max', 'y', 'y_min', 'y_max', 'reach_length', 'dist_out'
#node attributes to update: 'cl_ids', 'x', 'y', 'node_length', 'dist_out'
print('Starting Reach and Nodes Updates')
start = time.time()
for ind in list(range(len(rch_updates))):
    print(ind, len(rch_updates)-1)
    cl_rch = np.where(sword.centerlines.reach_id[0,:] == rch_updates[ind])[0]
    rch_ind = np.where(sword.reaches.id == rch_updates[ind])[0]

    if len(cl_rch) == 0:
        print('reach removed completely', rch_updates[ind])
        node_ind = np.where(sword.nodes.reach_id == rch_updates[ind])[0]
        sword.nodes.id = np.delete(sword.nodes.id, node_ind, axis = 0)
        sword.nodes.cl_id = np.delete(sword.nodes.cl_id, node_ind, axis = 1)
        sword.nodes.x = np.delete(sword.nodes.x, node_ind, axis = 0)
        sword.nodes.y = np.delete(sword.nodes.y, node_ind, axis = 0)
        sword.nodes.len = np.delete(sword.nodes.len, node_ind, axis = 0)
        sword.nodes.wse = np.delete(sword.nodes.wse, node_ind, axis = 0)
        sword.nodes.wse_var = np.delete(sword.nodes.wse_var, node_ind, axis = 0)
        sword.nodes.wth = np.delete(sword.nodes.wth, node_ind, axis = 0)
        sword.nodes.wth_var = np.delete(sword.nodes.wth_var, node_ind, axis = 0)
        sword.nodes.grod = np.delete(sword.nodes.grod, node_ind, axis = 0)
        sword.nodes.grod_fid = np.delete(sword.nodes.grod_fid, node_ind, axis = 0)
        sword.nodes.hfalls_fid = np.delete(sword.nodes.hfalls_fid, node_ind, axis = 0)
        sword.nodes.nchan_max = np.delete(sword.nodes.nchan_max, node_ind, axis = 0)
        sword.nodes.nchan_mod = np.delete(sword.nodes.nchan_mod, node_ind, axis = 0)
        sword.nodes.dist_out = np.delete(sword.nodes.dist_out, node_ind, axis = 0)
        sword.nodes.reach_id = np.delete(sword.nodes.reach_id, node_ind, axis = 0)
        sword.nodes.facc = np.delete(sword.nodes.facc, node_ind, axis = 0)
        sword.nodes.lakeflag = np.delete(sword.nodes.lakeflag, node_ind, axis = 0)
        sword.nodes.wth_coef = np.delete(sword.nodes.wth_coef, node_ind, axis = 0)
        sword.nodes.ext_dist_coef = np.delete(sword.nodes.ext_dist_coef, node_ind, axis = 0)
        sword.nodes.max_wth = np.delete(sword.nodes.max_wth, node_ind, axis = 0)
        sword.nodes.meand_len = np.delete(sword.nodes.meand_len, node_ind, axis = 0)
        sword.nodes.river_name = np.delete(sword.nodes.river_name, node_ind, axis = 0)
        sword.nodes.manual_add = np.delete(sword.nodes.manual_add, node_ind, axis = 0)
        sword.nodes.sinuosity = np.delete(sword.nodes.sinuosity, node_ind, axis = 0)
        sword.nodes.edit_flag = np.delete(sword.nodes.edit_flag, node_ind, axis = 0)
        sword.nodes.trib_flag = np.delete(sword.nodes.trib_flag, node_ind, axis = 0)

        sword.reaches.id = np.delete(sword.reaches.id, rch_ind, axis = 0)
        sword.reaches.cl_id = np.delete(sword.reaches.cl_id, rch_ind, axis = 1)
        sword.reaches.x = np.delete(sword.reaches.x, rch_ind, axis = 0)
        sword.reaches.x_min = np.delete(sword.reaches.x_min, rch_ind, axis = 0)
        sword.reaches.x_max = np.delete(sword.reaches.x_max, rch_ind, axis = 0)
        sword.reaches.y = np.delete(sword.reaches.y, rch_ind, axis = 0)
        sword.reaches.y_min = np.delete(sword.reaches.y_min, rch_ind, axis = 0)
        sword.reaches.y_max = np.delete(sword.reaches.y_max, rch_ind, axis = 0)
        sword.reaches.len = np.delete(sword.reaches.len, rch_ind, axis = 0)
        sword.reaches.wse = np.delete(sword.reaches.wse, rch_ind, axis = 0)
        sword.reaches.wse_var = np.delete(sword.reaches.wse_var, rch_ind, axis = 0)
        sword.reaches.wth = np.delete(sword.reaches.wth, rch_ind, axis = 0)
        sword.reaches.wth_var = np.delete(sword.reaches.wth_var, rch_ind, axis = 0)
        sword.reaches.slope = np.delete(sword.reaches.slope, rch_ind, axis = 0)
        sword.reaches.rch_n_nodes = np.delete(sword.reaches.rch_n_nodes, rch_ind, axis = 0)
        sword.reaches.grod = np.delete(sword.reaches.grod, rch_ind, axis = 0)
        sword.reaches.grod_fid = np.delete(sword.reaches.grod_fid, rch_ind, axis = 0)
        sword.reaches.hfalls_fid = np.delete(sword.reaches.hfalls_fid, rch_ind, axis = 0)
        sword.reaches.lakeflag = np.delete(sword.reaches.lakeflag, rch_ind, axis = 0)
        sword.reaches.nchan_max = np.delete(sword.reaches.nchan_max, rch_ind, axis = 0)
        sword.reaches.nchan_mod = np.delete(sword.reaches.nchan_mod, rch_ind, axis = 0)
        sword.reaches.dist_out = np.delete(sword.reaches.dist_out, rch_ind, axis = 0)
        sword.reaches.n_rch_up = np.delete(sword.reaches.n_rch_up, rch_ind, axis = 0)
        sword.reaches.n_rch_down = np.delete(sword.reaches.n_rch_down, rch_ind, axis = 0)
        sword.reaches.rch_id_up = np.delete(sword.reaches.rch_id_up, rch_ind, axis = 1)
        sword.reaches.rch_id_down = np.delete(sword.reaches.rch_id_down, rch_ind, axis = 1)
        sword.reaches.max_obs = np.delete(sword.reaches.max_obs, rch_ind, axis = 0)
        sword.reaches.orbits = np.delete(sword.reaches.orbits, rch_ind, axis = 1)
        sword.reaches.facc = np.delete(sword.reaches.facc, rch_ind, axis = 0)
        sword.reaches.iceflag = np.delete(sword.reaches.iceflag, rch_ind, axis = 1)
        sword.reaches.max_wth = np.delete(sword.reaches.max_wth, rch_ind, axis = 0)
        sword.reaches.river_name = np.delete(sword.reaches.river_name, rch_ind, axis = 0)
        sword.reaches.low_slope = np.delete(sword.reaches.low_slope, rch_ind, axis = 0)
        sword.reaches.edit_flag = np.delete(sword.reaches.edit_flag, rch_ind, axis = 0)
        sword.reaches.trib_flag = np.delete(sword.reaches.trib_flag, rch_ind, axis = 0)

        #removing residual neighbors with deleted reach id in centerline and reach groups. 
        cl_ind1 = np.where(sword.centerlines.reach_id[0,:] == rch_updates[ind])[0]
        cl_ind2 = np.where(sword.centerlines.reach_id[1,:] == rch_updates[ind])[0]
        cl_ind3 = np.where(sword.centerlines.reach_id[2,:] == rch_updates[ind])[0]
        cl_ind4 = np.where(sword.centerlines.reach_id[3,:] == rch_updates[ind])[0]
        if len(cl_ind1) > 0:
            sword.centerlines.reach_id[0,cl_ind1] = 0
        if len(cl_ind2) > 0:
            sword.centerlines.reach_id[1,cl_ind2] = 0
        if len(cl_ind3) > 0:
            sword.centerlines.reach_id[2,cl_ind3] = 0
        if len(cl_ind4) > 0:
            sword.centerlines.reach_id[3,cl_ind4] = 0

        rch_up_ind1 = np.where(sword.reaches.rch_id_up[0,:] == rch_updates[ind])[0]
        rch_up_ind2 = np.where(sword.reaches.rch_id_up[1,:] == rch_updates[ind])[0]
        rch_up_ind3 = np.where(sword.reaches.rch_id_up[2,:] == rch_updates[ind])[0]
        rch_up_ind4 = np.where(sword.reaches.rch_id_up[3,:] == rch_updates[ind])[0]
        if len(rch_up_ind1) > 0:
            sword.reaches.rch_id_up[0,rch_up_ind1] = 0
        if len(rch_up_ind2) > 0:
            sword.reaches.rch_id_up[1,rch_up_ind2] = 0
        if len(rch_up_ind3) > 0:
            sword.reaches.rch_id_up[2,rch_up_ind3] = 0
        if len(rch_up_ind4) > 0:
            sword.reaches.rch_id_up[3,rch_up_ind4] = 0

        rch_dn_ind1 = np.where(sword.reaches.rch_id_down[0,:] == rch_updates[ind])[0]
        rch_dn_ind2 = np.where(sword.reaches.rch_id_down[1,:] == rch_updates[ind])[0]
        rch_dn_ind3 = np.where(sword.reaches.rch_id_down[2,:] == rch_updates[ind])[0]
        rch_dn_ind4 = np.where(sword.reaches.rch_id_down[3,:] == rch_updates[ind])[0]
        if len(rch_dn_ind1) > 0:
            sword.reaches.rch_id_down[0,rch_dn_ind1] = 0
        if len(rch_dn_ind2) > 0:
            sword.reaches.rch_id_down[1,rch_dn_ind2] = 0
        if len(rch_dn_ind3) > 0:
            sword.reaches.rch_id_down[2,rch_dn_ind3] = 0
        if len(rch_dn_ind4) > 0:
            sword.reaches.rch_id_down[3,rch_dn_ind4] = 0
        # skip rest of the loop. 
        continue

    #calculate distance variables for centerline points. 
    facc = calc_rch_facc(nodes, centerlines, cl_rch)
    sort_ids = np.argsort(sword.centerlines.cl_id[cl_rch])
    gdf = gp.GeoDataFrame(geometry=gp.points_from_xy(sword.centerlines.x[cl_rch[sort_ids]], 
                                                     sword.centerlines.y[cl_rch[sort_ids]]),crs="EPSG:4326").to_crs("EPSG:3857")
    diff = gdf.distance(gdf.shift(1)); diff[0] = 0
    dist = np.array(np.cumsum(diff))
    
    #update reach variables:
    diff_len = sword.reaches.len[rch_ind] - np.max(dist)
    sword.reaches.x[rch_ind] = np.median(sword.centerlines.x[cl_rch])
    sword.reaches.y[rch_ind] = np.median(sword.centerlines.y[cl_rch])
    sword.reaches.x_max[rch_ind] = np.max(sword.centerlines.x[cl_rch])
    sword.reaches.x_min[rch_ind] = np.min(sword.centerlines.x[cl_rch])
    sword.reaches.y_max[rch_ind] = np.max(sword.centerlines.y[cl_rch])
    sword.reaches.y_min[rch_ind] = np.min(sword.centerlines.y[cl_rch])
    sword.reaches.cl_id[0,rch_ind] = np.min(sword.centerlines.cl_id[cl_rch])
    sword.reaches.cl_id[1,rch_ind] = np.max(sword.centerlines.cl_id[cl_rch])
    sword.reaches.rch_n_nodes[rch_ind] = len(np.unique(sword.centerlines.node_id[0,cl_rch]))
    sword.reaches.len[rch_ind] = np.max(dist)
    sword.reaches.dist_out[rch_ind] = sword.reaches.dist_out[rch_ind]-diff_len

    nds = node_updates[np.in1d(node_updates, sword.centerlines.node_id[0,cl_rch])]
    for idx in list(range(len(nds))):
        cl_nds = np.where(sword.centerlines.node_id[0,cl_rch] == nds[idx])[0]
        ns = np.where(sword.nodes.id == nds[idx])[0]

        #update reach variables:
        node_len = np.max(dist[cl_nds])-np.min(dist[cl_nds])
        node_diff_len = sword.nodes.len[ns] - node_len
        sword.nodes.x[ns] = np.median(sword.centerlines.x[cl_rch[cl_nds]])
        sword.nodes.y[ns] = np.median(sword.centerlines.y[cl_rch[cl_nds]])
        sword.nodes.cl_id[0,ns] = np.min(sword.centerlines.cl_id[cl_rch[cl_nds]])
        sword.nodes.cl_id[1,ns] = np.max(sword.centerlines.cl_id[cl_rch[cl_nds]])
        sword.nodes.len[ns] = node_len
        sword.nodes.dist_out[ns] = sword.nodes.dist_out[ns]-node_diff_len 

diff = np.in1d(sword.nodes.id, np.unique(sword.centerlines.node_id[0,:]))
rmv2 = np.where(diff == False)[0]
sword.nodes.id = np.delete(sword.nodes.id, rmv2, axis = 0)
sword.nodes.cl_id = np.delete(sword.nodes.cl_id, rmv2, axis = 1)
sword.nodes.x = np.delete(sword.nodes.x, rmv2, axis = 0)
sword.nodes.y = np.delete(sword.nodes.y, rmv2, axis = 0)
sword.nodes.len = np.delete(sword.nodes.len, rmv2, axis = 0)
sword.nodes.wse = np.delete(sword.nodes.wse, rmv2, axis = 0)
sword.nodes.wse_var = np.delete(sword.nodes.wse_var, rmv2, axis = 0)
sword.nodes.wth = np.delete(sword.nodes.wth, rmv2, axis = 0)
sword.nodes.wth_var = np.delete(sword.nodes.wth_var, rmv2, axis = 0)
sword.nodes.grod = np.delete(sword.nodes.grod, rmv2, axis = 0)
sword.nodes.grod_fid = np.delete(sword.nodes.grod_fid, rmv2, axis = 0)
sword.nodes.hfalls_fid = np.delete(sword.nodes.hfalls_fid, rmv2, axis = 0)
sword.nodes.nchan_max = np.delete(sword.nodes.nchan_max, rmv2, axis = 0)
sword.nodes.nchan_mod = np.delete(sword.nodes.nchan_mod, rmv2, axis = 0)
sword.nodes.dist_out = np.delete(sword.nodes.dist_out, rmv2, axis = 0)
sword.nodes.reach_id = np.delete(sword.nodes.reach_id, rmv2, axis = 0)
sword.nodes.facc = np.delete(sword.nodes.facc, rmv2, axis = 0)
sword.nodes.lakeflag = np.delete(sword.nodes.lakeflag, rmv2, axis = 0)
sword.nodes.wth_coef = np.delete(sword.nodes.wth_coef, rmv2, axis = 0)
sword.nodes.ext_dist_coef = np.delete(sword.nodes.ext_dist_coef, rmv2, axis = 0)
sword.nodes.max_wth = np.delete(sword.nodes.max_wth, rmv2, axis = 0)
sword.nodes.meand_len = np.delete(sword.nodes.meand_len, rmv2, axis = 0)
sword.nodes.river_name = np.delete(sword.nodes.river_name, rmv2, axis = 0)
sword.nodes.manual_add = np.delete(sword.nodes.manual_add, rmv2, axis = 0)
sword.nodes.sinuosity = np.delete(sword.nodes.sinuosity, rmv2, axis = 0)
sword.nodes.edit_flag = np.delete(sword.nodes.edit_flag, rmv2, axis = 0)
sword.nodes.trib_flag = np.delete(sword.nodes.trib_flag, rmv2, axis = 0)

#####################################################################

print('Writing NetCDF')
sword.save_nc()

end_all = time.time()
print('Finished '+region+' Updates in: '+str(np.round((end_all-start_all)/3600,2))+' hrs')

end = time.time()
print('Finished Reach and Node Updates in: '+str(np.round((end-start)/60,2))+' min')
print('Cl Dimensions:', len(np.unique(sword.centerlines.cl_id)), len(sword.centerlines.cl_id))
print('Rch Dimensions:', len(np.unique(sword.centerlines.reach_id[0,:])), len(np.unique(sword.nodes.reach_id)), len(sword.reaches.id))
print('Node Dimensions:', len(np.unique(sword.centerlines.node_id[0,:])), len(np.unique(sword.nodes.id)), len(sword.nodes.id))
