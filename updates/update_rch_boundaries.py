from __future__ import division
# import sys
# sys.path.append('../reach_definition/')
# from reach_definition import Reach_Definition_Tools_v11 as rdt
import numpy as np
import time
import netCDF4 as nc
import pandas as pd
from scipy import spatial as sp
import geopy.distance
import utm 

###############################################################################
###############################################################################
###############################################################################

class Object(object):
    """
    FUNCTION:
        Creates class object to assign attributes to.
    """
    pass 

###############################################################################

def read_merge_netcdf(filename):

    """
    FUNCTION:
        Reads in attributes from the merged database and assigns them to an
        object.

    INPUTS
        filename -- Merged database netcdf file.

    OUTPUTS
        data -- Object containing attributes from the merged database.
    """

    data = Object()
    new = nc.Dataset(filename)
    data.lon = new.groups['centerlines'].variables['x'][:]
    data.lat = new.groups['centerlines'].variables['y'][:]
    data.x = new.groups['centerlines'].variables['easting'][:]
    data.y = new.groups['centerlines'].variables['northing'][:]
    data.seg = new.groups['centerlines'].variables['segID'][:]
    data.ind = new.groups['centerlines'].variables['segInd'][:]
    data.id = new.groups['centerlines'].variables['cl_id'][:]
    data.segDist = new.groups['centerlines'].variables['segDist'][:]
    data.wth = new.groups['centerlines'].variables['p_width'][:]
    data.elv = new.groups['centerlines'].variables['p_height'][:]
    data.facc = new.groups['centerlines'].variables['flowacc'][:]
    data.lake = new.groups['centerlines'].variables['lakeflag'][:]
    data.delta = new.groups['centerlines'].variables['deltaflag'][:]
    data.nchan = new.groups['centerlines'].variables['nchan'][:]
    data.grand = new.groups['centerlines'].variables['grand_id'][:]
    data.grod = new.groups['centerlines'].variables['grod_id'][:]
    data.grod_fid = new.groups['centerlines'].variables['grod_fid'][:]
    data.hfalls_fid = new.groups['centerlines'].variables['hfalls_fid'][:]
    data.basins = new.groups['centerlines'].variables['basin_code'][:]
    data.manual = new.groups['centerlines'].variables['manual_add'][:]
    data.num_obs = new.groups['centerlines'].variables['number_obs'][:]
    data.orbits = new.groups['centerlines'].variables['orbits'][:]
    data.tile = new.groups['centerlines'].variables['grwl_tile'][:]
    data.eps = new.groups['centerlines'].variables['endpoints'][:]
    data.lake_id = new.groups['centerlines'].variables['lake_id'][:]
    new.close()

    return data

###############################################################################

def read_merge_netcdf_subset(filename, basin, level):

    """
    FUNCTION:
        Reads in attributes from the merged database and assigns them to an
        object.

    INPUTS
        filename -- Merged database netcdf file.

    OUTPUTS
        data -- Object containing attributes from the merged database.
    """

    data = Object()
    new = nc.Dataset(filename)
    pfaf = np.array([str(b)[0:level] for b in new.groups['centerlines'].variables['basin_code'][:]])
    lvl = np.where(pfaf == str(basin))[0]

    data.lon = new.groups['centerlines'].variables['x'][lvl]
    data.lat = new.groups['centerlines'].variables['y'][lvl]
    data.x = new.groups['centerlines'].variables['easting'][lvl]
    data.y = new.groups['centerlines'].variables['northing'][lvl]
    data.seg = new.groups['centerlines'].variables['segID'][lvl]
    data.ind = new.groups['centerlines'].variables['segInd'][lvl]
    data.id = new.groups['centerlines'].variables['cl_id'][lvl]
    data.segDist = new.groups['centerlines'].variables['segDist'][lvl]
    data.wth = new.groups['centerlines'].variables['p_width'][lvl]
    data.elv = new.groups['centerlines'].variables['p_height'][lvl]
    data.facc = new.groups['centerlines'].variables['flowacc'][lvl]
    data.lake = new.groups['centerlines'].variables['lakeflag'][lvl]
    data.delta = new.groups['centerlines'].variables['deltaflag'][lvl]
    data.nchan = new.groups['centerlines'].variables['nchan'][lvl]
    data.grand = new.groups['centerlines'].variables['grand_id'][lvl]
    data.grod = new.groups['centerlines'].variables['grod_id'][lvl]
    data.grod_fid = new.groups['centerlines'].variables['grod_fid'][lvl]
    data.hfalls_fid = new.groups['centerlines'].variables['hfalls_fid'][lvl]
    data.basins = new.groups['centerlines'].variables['basin_code'][lvl]
    data.manual = new.groups['centerlines'].variables['manual_add'][lvl]
    data.num_obs = new.groups['centerlines'].variables['number_obs'][lvl]
    data.orbits = new.groups['centerlines'].variables['orbits'][lvl]
    data.tile = new.groups['centerlines'].variables['grwl_tile'][lvl]
    data.eps = new.groups['centerlines'].variables['endpoints'][lvl]
    data.lake_id = new.groups['centerlines'].variables['lake_id'][lvl]
    new.close()

    return data

###############################################################################

def read_data(filename):

    centerlines = Object()
    nodes = Object()
    reaches = Object ()
    
    data = nc.Dataset(filename)
    
    centerlines.cl_id = data.groups['centerlines'].variables['cl_id'][:]
    centerlines.x = data.groups['centerlines'].variables['x'][:]
    centerlines.y = data.groups['centerlines'].variables['y'][:]
    centerlines.reach_id = data.groups['centerlines'].variables['reach_id'][:]
    centerlines.node_id = data.groups['centerlines'].variables['node_id'][:]
    
    nodes.id = data.groups['nodes'].variables['node_id'][:]
    nodes.cl_id = data.groups['nodes'].variables['cl_ids'][:]
    nodes.x = data.groups['nodes'].variables['x'][:]
    nodes.y = data.groups['nodes'].variables['y'][:]
    nodes.len = data.groups['nodes'].variables['node_length'][:]
    nodes.wse = data.groups['nodes'].variables['wse'][:]
    nodes.wse_var = data.groups['nodes'].variables['wse_var'][:]
    nodes.wth = data.groups['nodes'].variables['width'][:]
    nodes.wth_var = data.groups['nodes'].variables['width_var'][:]
    nodes.grod = data.groups['nodes'].variables['obstr_type'][:]
    nodes.grod_fid = data.groups['nodes'].variables['grod_id'][:]
    nodes.hfalls_fid = data.groups['nodes'].variables['hfalls_id'][:]
    nodes.nchan_max = data.groups['nodes'].variables['n_chan_max'][:]
    nodes.nchan_mod = data.groups['nodes'].variables['n_chan_mod'][:]
    nodes.dist_out = data.groups['nodes'].variables['dist_out'][:]
    nodes.reach_id = data.groups['nodes'].variables['reach_id'][:]
    nodes.facc = data.groups['nodes'].variables['facc'][:]
    nodes.lakeflag = data.groups['nodes'].variables['lakeflag'][:]
    nodes.wth_coef = data.groups['nodes'].variables['wth_coef'][:]
    nodes.ext_dist_coef = data.groups['nodes'].variables['ext_dist_coef'][:]
    nodes.max_wth = data.groups['nodes'].variables['max_width'][:]
    nodes.meand_len = data.groups['nodes'].variables['meander_length'][:]
    nodes.river_name = data.groups['nodes'].variables['river_name'][:]
    nodes.manual_add = data.groups['nodes'].variables['manual_add'][:]
    nodes.sinuosity = data.groups['nodes'].variables['sinuosity'][:]
    nodes.edit_flag = data.groups['nodes'].variables['edit_flag'][:]

    reaches.id = data.groups['reaches'].variables['reach_id'][:]
    reaches.cl_id = data.groups['reaches'].variables['cl_ids'][:]
    reaches.x = data.groups['reaches'].variables['x'][:]
    reaches.x_min = data.groups['reaches'].variables['x_min'][:]
    reaches.x_max = data.groups['reaches'].variables['x_max'][:]
    reaches.y = data.groups['reaches'].variables['y'][:]
    reaches.y_min = data.groups['reaches'].variables['y_min'][:]
    reaches.y_max = data.groups['reaches'].variables['y_max'][:]
    reaches.len = data.groups['reaches'].variables['reach_length'][:]
    reaches.wse = data.groups['reaches'].variables['wse'][:]
    reaches.wse_var = data.groups['reaches'].variables['wse_var'][:]
    reaches.wth = data.groups['reaches'].variables['width'][:]
    reaches.wth_var = data.groups['reaches'].variables['width_var'][:]
    reaches.slope = data.groups['reaches'].variables['slope'][:]
    reaches.rch_n_nodes = data.groups['reaches'].variables['n_nodes'][:]
    reaches.grod = data.groups['reaches'].variables['obstr_type'][:]
    reaches.grod_fid = data.groups['reaches'].variables['grod_id'][:]
    reaches.hfalls_fid = data.groups['reaches'].variables['hfalls_id'][:]
    reaches.lakeflag = data.groups['reaches'].variables['lakeflag'][:]
    reaches.nchan_max = data.groups['reaches'].variables['n_chan_max'][:]
    reaches.nchan_mod = data.groups['reaches'].variables['n_chan_mod'][:]
    reaches.dist_out = data.groups['reaches'].variables['dist_out'][:]
    reaches.n_rch_up = data.groups['reaches'].variables['n_rch_up'][:]
    reaches.n_rch_down = data.groups['reaches'].variables['n_rch_down'][:]
    reaches.rch_id_up = data.groups['reaches'].variables['rch_id_up'][:]
    reaches.rch_id_down = data.groups['reaches'].variables['rch_id_dn'][:]
    reaches.max_obs = data.groups['reaches'].variables['swot_obs'][:]
    reaches.orbits = data.groups['reaches'].variables['swot_orbits'][:]
    reaches.facc = data.groups['reaches'].variables['facc'][:]
    reaches.iceflag = data.groups['reaches'].variables['iceflag'][:]
    reaches.max_wth = data.groups['reaches'].variables['max_width'][:]
    reaches.river_name = data.groups['reaches'].variables['river_name'][:]
    reaches.low_slope = data.groups['reaches'].variables['low_slope_flag'][:]
    reaches.edit_flag= data.groups['reaches'].variables['edit_flag'][:]

    data.close()    

    return centerlines, nodes, reaches
    
###############################################################################    

def subset_data(centerlines, nodes, reaches, old_rch_ids):
    rch_keep = np.where(np.in1d(reaches.id, old_rch_ids))[0]
    nodes_keep = np.where(np.in1d(nodes.reach_id, old_rch_ids))[0] 
    cl_keep = np.where(np.in1d(centerlines.reach_id[0,:], old_rch_ids))[0] 

    sub_centerlines = Object()
    sub_nodes = Object()
    sub_reaches = Object ()

    sub_centerlines.cl_id = centerlines.cl_id[cl_keep]
    sub_centerlines.x = centerlines.x[cl_keep]
    sub_centerlines.y = centerlines.y[cl_keep]
    sub_centerlines.reach_id = centerlines.reach_id[:,cl_keep]
    sub_centerlines.node_id = centerlines.node_id[:,cl_keep]
    
    sub_nodes.id = nodes.id[nodes_keep]
    sub_nodes.cl_id = nodes.cl_id[:,nodes_keep]
    sub_nodes.x = nodes.x[nodes_keep]
    sub_nodes.y = nodes.y[nodes_keep]
    sub_nodes.len = nodes.len[nodes_keep]
    sub_nodes.wse = nodes.wse[nodes_keep]
    sub_nodes.wse_var = nodes.wse_var[nodes_keep]
    sub_nodes.wth = nodes.wth[nodes_keep]
    sub_nodes.wth_var = nodes.wth_var[nodes_keep]
    sub_nodes.grod = nodes.grod[nodes_keep]
    sub_nodes.grod_fid = nodes.grod_fid[nodes_keep]
    sub_nodes.hfalls_fid = nodes.hfalls_fid[nodes_keep]
    sub_nodes.nchan_max = nodes.nchan_max[nodes_keep]
    sub_nodes.nchan_mod = nodes.nchan_mod[nodes_keep]
    sub_nodes.dist_out = nodes.dist_out[nodes_keep]
    sub_nodes.reach_id = nodes.reach_id[nodes_keep]
    sub_nodes.facc = nodes.facc[nodes_keep]
    sub_nodes.lakeflag = nodes.lakeflag[nodes_keep]
    sub_nodes.wth_coef = nodes.wth_coef[nodes_keep]
    sub_nodes.ext_dist_coef = nodes.ext_dist_coef[nodes_keep]
    sub_nodes.max_wth = nodes.max_wth[nodes_keep]
    sub_nodes.meand_len = nodes.meand_len[nodes_keep]
    sub_nodes.river_name = nodes.river_name[nodes_keep]
    sub_nodes.manual_add = nodes.manual_add[nodes_keep]
    sub_nodes.sinuosity = nodes.sinuosity[nodes_keep]
    sub_nodes.edit_flag = nodes.edit_flag[nodes_keep]

    sub_reaches.id = reaches.id[rch_keep]
    sub_reaches.cl_id = reaches.cl_id[:,rch_keep]
    sub_reaches.x = reaches.x[rch_keep]
    sub_reaches.x_min = reaches.x_min[rch_keep]
    sub_reaches.x_max = reaches.x_max[rch_keep]
    sub_reaches.y = reaches.y[rch_keep]
    sub_reaches.y_min = reaches.y_min[rch_keep]
    sub_reaches.y_max = reaches.y_max[rch_keep]
    sub_reaches.len = reaches.len[rch_keep]
    sub_reaches.wse = reaches.wse[rch_keep]
    sub_reaches.wse_var = reaches.wse_var[rch_keep]
    sub_reaches.wth = reaches.wth[rch_keep]
    sub_reaches.wth_var = reaches.wth_var[rch_keep]
    sub_reaches.slope = reaches.slope[rch_keep]
    sub_reaches.rch_n_nodes = reaches.rch_n_nodes[rch_keep]
    sub_reaches.grod = reaches.grod[rch_keep]
    sub_reaches.grod_fid = reaches.grod_fid[rch_keep]
    sub_reaches.hfalls_fid = reaches.hfalls_fid[rch_keep]
    sub_reaches.lakeflag = reaches.lakeflag[rch_keep]
    sub_reaches.nchan_max = reaches.nchan_max[rch_keep]
    sub_reaches.nchan_mod = reaches.nchan_mod[rch_keep]
    sub_reaches.dist_out = reaches.dist_out[rch_keep]
    sub_reaches.n_rch_up = reaches.n_rch_up[rch_keep]
    sub_reaches.n_rch_down = reaches.n_rch_down[rch_keep]
    sub_reaches.rch_id_up = reaches.rch_id_up[:,rch_keep]
    sub_reaches.rch_id_down = reaches.rch_id_down[:,rch_keep]
    sub_reaches.max_obs = reaches.max_obs[rch_keep]
    sub_reaches.orbits = reaches.orbits[:,rch_keep]
    sub_reaches.facc = reaches.facc[rch_keep]
    sub_reaches.iceflag = reaches.iceflag[:,rch_keep]
    sub_reaches.max_wth = reaches.max_wth[rch_keep]
    sub_reaches.river_name = reaches.river_name[rch_keep]
    sub_reaches.low_slope = reaches.low_slope[rch_keep]
    sub_reaches.edit_flag = reaches.edit_flag [rch_keep]

    return sub_centerlines, sub_nodes, sub_reaches

###############################################################################

def append_data(reaches, subreaches):
    reaches.id = np.insert(reaches.id, len(reaches.id), np.copy(subreaches.id))
    reaches.cl_id = np.append(reaches.cl_id, subreaches.cl_id, axis=1)
    reaches.x = np.insert(reaches.x, len(reaches.x), np.copy(subreaches.x))
    reaches.x_min = np.insert(reaches.x_min, len(reaches.x_min), np.copy(subreaches.x_min))
    reaches.x_max = np.insert(reaches.x_max, len(reaches.x_max), np.copy(subreaches.x_max))
    reaches.y = np.insert(reaches.y, len(reaches.y), np.copy(subreaches.y))
    reaches.y_min = np.insert(reaches.y_min, len(reaches.y_min), np.copy(subreaches.y_min))
    reaches.y_max = np.insert(reaches.y_max, len(reaches.y_max), np.copy(subreaches.y_max))
    reaches.len = np.insert(reaches.len, len(reaches.len), np.copy(subreaches.len))
    reaches.wse = np.insert(reaches.wse, len(reaches.wse), np.copy(subreaches.wse))
    reaches.wse_var = np.insert(reaches.wse_var, len(reaches.wse_var), np.copy(subreaches.wse_var))
    reaches.wth = np.insert(reaches.wth, len(reaches.wth), np.copy(subreaches.wth))
    reaches.wth_var = np.insert(reaches.wth_var, len(reaches.wth_var), np.copy(subreaches.wth_var))
    reaches.slope = np.insert(reaches.slope, len(reaches.slope), np.copy(subreaches.slope))
    reaches.rch_n_nodes = np.insert(reaches.rch_n_nodes, len(reaches.rch_n_nodes), np.copy(subreaches.rch_n_nodes))
    reaches.grod = np.insert(reaches.grod, len(reaches.grod), np.copy(subreaches.grod))
    reaches.grod_fid = np.insert(reaches.grod_fid, len(reaches.grod_fid), np.copy(subreaches.grod_fid))
    reaches.hfalls_fid = np.insert(reaches.hfalls_fid, len(reaches.hfalls_fid), np.copy(subreaches.hfalls_fid))
    reaches.nchan_max = np.insert(reaches.nchan_max, len(reaches.nchan_max), np.copy(subreaches.nchan_max))
    reaches.nchan_mod = np.insert(reaches.nchan_mod, len(reaches.nchan_mod), np.copy(subreaches.nchan_mod))
    reaches.dist_out = np.insert(reaches.dist_out, len(reaches.dist_out), np.copy(subreaches.dist_out))
    reaches.n_rch_up = np.insert(reaches.n_rch_up, len(reaches.n_rch_up), np.copy(subreaches.n_rch_up_filt))
    reaches.n_rch_down = np.insert(reaches.n_rch_down, len(reaches.n_rch_down), np.copy(subreaches.n_rch_down_filt))
    reaches.rch_id_up = np.append(reaches.rch_id_up, subreaches.rch_id_up_filt, axis=1)
    reaches.rch_id_down = np.append(reaches.rch_id_down, subreaches.rch_id_down_filt, axis=1)
    reaches.lakeflag = np.insert(reaches.lakeflag, len(reaches.lakeflag), np.copy(subreaches.lakeflag))
    reaches.facc = np.insert(reaches.facc, len(reaches.facc), np.copy(subreaches.facc))
    reaches.max_obs = np.insert(reaches.max_obs, len(reaches.max_obs), np.copy(subreaches.max_obs))
    reaches.iceflag = np.append(reaches.iceflag, subreaches.iceflag, axis=1)
    reaches.max_wth = np.insert(reaches.max_wth, len(reaches.max_wth), np.copy(subreaches.max_wth))
    reaches.orbits = np.append(reaches.orbits, subreaches.orbits, axis=1)
    reaches.river_name = np.insert(reaches.river_name, len(reaches.river_name), np.copy(subreaches.river_name))
    reaches.low_slope = np.insert(reaches.low_slope, len(reaches.low_slope), np.copy(subreaches.low_slope))
    reaches.edit_flag = np.insert(reaches.edit_flag, len(reaches.edit_flag), np.copy(subreaches.edit_flag))

###############################################################################

def local_topology(subcls, subreaches):

    """
    FUNCTION:
        Identifies the upstream and downstream reach neighbors, and number of
        neighbors for each reach.

    INPUTS
        subreaches -- Object containing attributes for the reach locations.
            [attributes used]:
                id -- Reach ID for each reach.
                dist_out -- Distance from outlet for each reach (m).
                neighbors -- All neighbors, without direction, for each reach.
                wse -- Water surface elevation (m) for each reach.
                facc -- Flow accumulatin (km^2) for each reach.
        subcls -- Object containing attributes for the node locations.
            [attributes used]:
                reach_id -- Reach ID along the high-resolution centerline.
                rch_ind6 -- Point indexes after aggregating short reaches and
                            cutting long reaches.
                lon -- Latitude (wgs84).
                lat -- Logitude (wgs84).

    OUTPUTS
        n_rch_up -- Number of upstream reaches per reach.
        n_rch_down -- Number of downstream reaches per reach.
        rch_id_up -- Reach ids of the upstream reach neighbors per reach.
        rch_id_down -- Reach ids of the downstream reach neighbors per reach.
    """

    n_rch_up = np.zeros(len(np.unique(subcls.reach_id)))
    n_rch_down = np.zeros(len(np.unique(subcls.reach_id)))
    rch_id_up = np.zeros((len(np.unique(subcls.reach_id)),10))
    rch_id_down = np.zeros((len(np.unique(subcls.reach_id)),10))
    #flagged = np.zeros(len(np.unique(subcls.reach_id)))

    for ind in list(range(len(subreaches.id))):
        #print(ind)
        #identifying the current reach neighbors.
        nghs = subreaches.neighbors[ind,np.where(subreaches.neighbors[ind,:] > 0)[0]]

        if len(nghs) == 0:
            continue

        else:
            #determining important reach attributes.
            rch_cls = np.where(subcls.reach_id == subreaches.id[ind])[0]
            rch = np.where(subreaches.id == subreaches.id[ind])[0]
            rch_dist = subreaches.dist_out[rch]
            rch_wse = subreaches.wse[rch]
            rch_facc = subreaches.facc[rch]

            #identifying reach endpoints based on index values.
            end1 = rch_cls[np.where(subcls.rch_ind6[rch_cls] == np.min(subcls.rch_ind6[rch_cls]))[0][0]]
            end2 = rch_cls[np.where(subcls.rch_ind6[rch_cls] == np.max(subcls.rch_ind6[rch_cls]))[0][0]]

            #finding the coordination of the reach endpoints.
            ep1_lon, ep1_lat = subcls.lon[end1], subcls.lat[end1]
            ep2_lon, ep2_lat = subcls.lon[end2], subcls.lat[end2]

            #looping through each reach neighbor and determining if it is
            #upstream or downstream based on flow accumulation or wse.
            ep1_nghs = np.zeros([len(nghs), 4])
            ep2_nghs = np.zeros([len(nghs), 4])
            for idx in list(range(len(nghs))):
                rch2 = np.where(subreaches.id == nghs[idx])[0]
                rch2_cls = np.where(subcls.reach_id == nghs[idx])[0]
                ngh_dist = subreaches.dist_out[rch2]
                ngh_wse = subreaches.wse[rch2]
                ngh_facc = subreaches.facc[rch2]

                # finding high-res centerline endpoints for neighboring reach.
                ngh_end1 = rch2_cls[np.where(subcls.rch_ind6[rch2_cls] == np.min(subcls.rch_ind6[rch2_cls]))[0][0]]
                ngh_end2 = rch2_cls[np.where(subcls.rch_ind6[rch2_cls] == np.max(subcls.rch_ind6[rch2_cls]))[0][0]]
                ngh_end1_lon, ngh_end1_lat = subcls.lon[ngh_end1], subcls.lat[ngh_end1]
                ngh_end2_lon, ngh_end2_lat = subcls.lon[ngh_end2], subcls.lat[ngh_end2]

                # finding distance between neighboring endpoints and current reach.
                coords_1 = (ep1_lat, ep1_lon)
                coords_2 = (ep2_lat, ep2_lon)
                coords_3 = (ngh_end1_lat, ngh_end1_lon)
                coords_4 = (ngh_end2_lat, ngh_end2_lon)
                ep1_dist_ngh1 = geopy.distance.geodesic(coords_1, coords_3).m
                ep1_dist_ngh2 = geopy.distance.geodesic(coords_1, coords_4).m
                ep2_dist_ngh1 = geopy.distance.geodesic(coords_2, coords_3).m
                ep2_dist_ngh2 = geopy.distance.geodesic(coords_2, coords_4).m
                ep1_dist = np.min([ep1_dist_ngh1, ep1_dist_ngh2])
                ep2_dist = np.min([ep2_dist_ngh1, ep2_dist_ngh2])

                if ep1_dist <= ep2_dist: #if equal distance then assign to end1 for now...
                    fill_row = np.min(np.where(ep1_nghs[:,0] == 0)[0])
                    ep1_nghs[fill_row,0] = nghs[idx]
                    ep1_nghs[fill_row,1] = ngh_facc
                    ep1_nghs[fill_row,2] = ngh_wse
                    ep1_nghs[fill_row,3] = ngh_dist
                else:
                    fill_row = np.min(np.where(ep2_nghs[:,0] == 0)[0])
                    ep2_nghs[fill_row,0] = nghs[idx]
                    ep2_nghs[fill_row,1] = ngh_facc
                    ep2_nghs[fill_row,2] = ngh_wse
                    ep2_nghs[fill_row,3] = ngh_dist

            ep1_delete = np.where(ep1_nghs[:,0] == 0)[0]
            ep2_delete = np.where(ep2_nghs[:,0] == 0)[0]
            ep1_nghs = np.delete(ep1_nghs, ep1_delete, axis = 0)
            ep2_nghs = np.delete(ep2_nghs, ep2_delete, axis = 0)

            if len(ep1_nghs) > 0:
                min_dist = np.min(ep1_nghs[:,3])
                min_wse = np.min(ep1_nghs[:,2])
                max_facc = np.max(ep1_nghs[:,1])

                if min_dist < rch_dist:
                    ## assign to downstream end
                    start_col = np.min(np.where(rch_id_down[ind,:] == 0)[0])
                    end_col = start_col + len(ep1_nghs)
                    rch_id_down[ind,start_col:end_col] = ep1_nghs[:,0]
                elif min_dist > rch_dist:
                    ### assign to upstream end
                    start_col = np.min(np.where(rch_id_up[ind,:] == 0)[0])
                    end_col = start_col + len(ep1_nghs)
                    rch_id_up[ind,start_col:end_col] = ep1_nghs[:,0]
                else:
                    if max_facc > rch_facc:
                        ### assign to downstream end
                        start_col = np.min(np.where(rch_id_down[ind,:] == 0)[0])
                        end_col = start_col + len(ep1_nghs)
                        rch_id_down[ind,start_col:end_col] = ep1_nghs[:,0]
                    elif max_facc < rch_facc:
                        ### assign to upstream end
                        start_col = np.min(np.where(rch_id_up[ind,:] == 0)[0])
                        end_col = start_col + len(ep1_nghs)
                        rch_id_up[ind,start_col:end_col] = ep1_nghs[:,0]
                    else:
                        if min_wse > rch_wse:
                            ### assign to upstream end
                            start_col = np.min(np.where(rch_id_up[ind,:] == 0)[0])
                            end_col = start_col + len(ep1_nghs)
                            rch_id_up[ind,start_col:end_col] = ep1_nghs[:,0]
                        else:
                            ### assign to downstream end
                            start_col = np.min(np.where(rch_id_down[ind,:] == 0)[0])
                            end_col = start_col + len(ep1_nghs)
                            rch_id_down[ind,start_col:end_col] = ep1_nghs[:,0]

            if len(ep2_nghs) > 0:
                min_dist = np.min(ep2_nghs[:,3])
                min_wse = np.min(ep2_nghs[:,2])
                max_facc = np.max(ep2_nghs[:,1])

                if min_dist < rch_dist:
                    ## assign to downstream end
                    start_col = np.min(np.where(rch_id_down[ind,:] == 0)[0])
                    end_col = start_col + len(ep2_nghs)
                    rch_id_down[ind,start_col:end_col] = ep2_nghs[:,0]
                elif min_dist > rch_dist:
                    ### assign to upstream end
                    start_col = np.min(np.where(rch_id_up[ind,:] == 0)[0])
                    end_col = start_col + len(ep2_nghs)
                    rch_id_up[ind,start_col:end_col] = ep2_nghs[:,0]
                else:
                    if max_facc > rch_facc:
                        ### assign to downstream end
                        start_col = np.min(np.where(rch_id_down[ind,:] == 0)[0])
                        end_col = start_col + len(ep2_nghs)
                        rch_id_down[ind,start_col:end_col] = ep2_nghs[:,0]
                    elif max_facc < rch_facc:
                        ### assign to upstream end
                        start_col = np.min(np.where(rch_id_up[ind,:] == 0)[0])
                        end_col = start_col + len(ep2_nghs)
                        rch_id_up[ind,start_col:end_col] = ep2_nghs[:,0]
                    else:
                        if min_wse > rch_wse:
                            ### assign to upstream end
                            start_col = np.min(np.where(rch_id_up[ind,:] == 0)[0])
                            end_col = start_col + len(ep2_nghs)
                            rch_id_up[ind,start_col:end_col] = ep2_nghs[:,0]
                        else:
                            ### assign to downstream end
                            start_col = np.min(np.where(rch_id_down[ind,:] == 0)[0])
                            end_col = start_col + len(ep2_nghs)
                            rch_id_down[ind,start_col:end_col] = ep2_nghs[:,0]

        ### count number of upstream and downstream neighbors.
        n_rch_down[ind] = len(np.where(rch_id_down[ind,:] > 0)[0])
        n_rch_up[ind] = len(np.where(rch_id_up[ind,:] > 0)[0])

    return n_rch_up, n_rch_down, rch_id_up, rch_id_down

###############################################################################

def filter_neighbors(subreaches):

    """
        FUNCTION:
            Filters reach neighbors to exclude upstream or downstream
            neighbors that do not belong in the associated category. For example,
            if two tributaries are joining together, one may be miscategorized
            as being downstream of the other when it is just a neighboring
            reach with a common downstream neighbor. This function helps
            remove these misclassifications.

        INPUTS
            subreaches -- Object containing attributes for the reache locations.
                [attributes used]:
                    rch_id_down -- List of reach IDs for the downstream reaches.
                    n_rch_down -- Number of downstream reaches.
                    rch_id_up -- List of reach IDs for the upstream reaches.
                    n_rch_up -- Number of downstream reaches.

        OUTPUTS
            copy_rchs_up -- Filtered list of reach IDs for upstream reaches.
            copy_nrch_up -- Filtered number of upstream reaches.
            copy_rchs_dn -- Filtered list of reach IDs for downstream reaches.
            copy_nrch_dn -- Filtered number of downstream reaches.
    """

    #make copies of existing attributes.
    copy_rchs_dn = np.copy(subreaches.rch_id_down)
    copy_nrch_dn = np.copy(subreaches.n_rch_down)
    copy_rchs_up = np.copy(subreaches.rch_id_up)
    copy_nrch_up = np.copy(subreaches.n_rch_up)

    #Filter downstream reach neighbors.
    multi_downstream_tribs = np.where(subreaches.n_rch_down > 1)[0]
    for ind in list(range(len(multi_downstream_tribs))):
        current_rch_ngh_dn = np.unique(copy_rchs_dn[multi_downstream_tribs[ind]])
        current_rch_ngh_dn = np.delete(current_rch_ngh_dn, np.where(current_rch_ngh_dn == 0)[0])
        for idx in list(range(len(current_rch_ngh_dn))):
            ds_rch_ngh = np.unique(copy_rchs_dn[np.where(subreaches.id == current_rch_ngh_dn[idx])[0]])
            ds_rch_ngh = np.delete(ds_rch_ngh, np.where(ds_rch_ngh == 0)[0])
            matches = np.intersect1d(ds_rch_ngh, current_rch_ngh_dn)
            if len(matches) > 0:
                #print(ind)
                rmv_index = np.where(copy_rchs_dn[multi_downstream_tribs[ind]] == current_rch_ngh_dn[idx])[0]
                copy_rchs_dn[multi_downstream_tribs[ind]][rmv_index] = 0

        copy_rchs_dn[multi_downstream_tribs[ind]][::-1].sort()
        copy_nrch_dn[multi_downstream_tribs[ind]] = len(np.where(copy_rchs_dn[multi_downstream_tribs[ind]]>0)[0])

    #Filter upstream reach neighbors.
    multi_upstream_tribs = np.where(subreaches.n_rch_up > 1)[0]
    for ind in list(range(len(multi_upstream_tribs))):
        current_rch_ngh_up = np.unique(copy_rchs_up[multi_upstream_tribs[ind]])
        current_rch_ngh_up = np.delete(current_rch_ngh_up, np.where(current_rch_ngh_up == 0)[0])
        for idx in list(range(len(current_rch_ngh_up))):
            us_rch_ngh = np.unique(copy_rchs_up[np.where(subreaches.id == current_rch_ngh_up[idx])[0]])
            us_rch_ngh = np.delete(us_rch_ngh, np.where(us_rch_ngh == 0)[0])
            matches = np.intersect1d(us_rch_ngh, current_rch_ngh_up)
            if len(matches) > 0:
                #print(ind)
                rmv_index = np.where(copy_rchs_up[multi_upstream_tribs[ind]] == current_rch_ngh_up[idx])[0]
                copy_rchs_up[multi_upstream_tribs[ind]][rmv_index] = 0

        copy_rchs_up[multi_upstream_tribs[ind]][::-1].sort()
        copy_nrch_up[multi_upstream_tribs[ind]] = len(np.where(copy_rchs_up[multi_upstream_tribs[ind]]>0)[0])

    return copy_rchs_up, copy_nrch_up, copy_rchs_dn, copy_nrch_dn

###############################################################################

def update_rch_ids(centerlines, nodes, new_ids):
    nodes.old_id = np.copy(nodes.id)
    centerlines.old_node_id = np.copy(centerlines.node_id)
    centerlines.old_rch_id = np.copy(centerlines.reach_id)
    unq_rchs = np.unique(new_ids.new_rch_id)
    for ind in list(range(len(unq_rchs))):
        node_rch = np.where(nodes.reach_id == unq_rchs[ind])[0]
        numloc = np.where(new_ids.new_rch_id == unq_rchs[ind])[0]
        
        if nodes.edit_flag[node_rch][0] == 'NaN':
            edit_val = '5'
        else:
            edit_val = str(nodes.edit_flag[node_rch][0]) + ',5' 
        
        node_nums = []
        for nl in list(range(len(numloc))):
            if len(str(new_ids.new_node_num[numloc[nl]])) == 1:
                node_nums.append(str('00')+str(new_ids.new_node_num[numloc[nl]]))
            elif len(str(new_ids.new_node_num[numloc[nl]])) == 2:
                node_nums.append(str('0')+str(new_ids.new_node_num[numloc[nl]]))
            else:
                node_nums.append(str(new_ids.new_node_num[numloc[nl]]))
        
        new_node_ids = [int(str(unq_rchs[ind])[:-1]+node_nums[i]+str(unq_rchs[ind])[-1:]) for i in list(range(len(node_nums)))]
        
        #update node id attributes. 
        nodes.id[node_rch] = new_node_ids
        nodes.edit_flag[node_rch] = np.repeat(edit_val, len(node_rch))
        
        #update centerline id attributes. 
        for nr in list(range(len(node_rch))):           
            mn = np.where(centerlines.cl_id == nodes.cl_id[0,node_rch[nr]])[0]
            mx = np.where(centerlines.cl_id == nodes.cl_id[1,node_rch[nr]])[0]
            cl_vals = np.linspace(centerlines.cl_id[mn], centerlines.cl_id[mx], int((centerlines.cl_id[mx]-centerlines.cl_id[mn]))+1, dtype=int)
            cl_inds = np.where(np.in1d(centerlines.cl_id, cl_vals))[0]
            centerlines.node_id[0,cl_inds] = new_node_ids[nr]
            centerlines.reach_id[0,cl_inds] = unq_rchs[ind]

###############################################################################

def neigboring_reaches(cl_lon, cl_lat, cl_rch, reaches):

    """
    FUNCTION:
        This function finds all reach neighbors within the larger level 2
        Pfafstetter basin. This output is used in "find_pathways" function.

    INPUTS
        subcls -- Object containing attributes for the high-resolution centerlines.
            [attributes used]:
                lat -- Latitude (wgs84) along the high-resolution centerline.
                lon -- Longitude (wgs84) along the high-reasolution centerline.
                reach_id -- Reach IDs along the high-resolution centerline.
        subreaches -- Object containing attributes for the reach locations.
            [attributes used]:
                id -- Reach IDs per reach.

    OUTPUTS
        neighbors -- List of neighboring reach ids for each reach.
    """

    #reproject lat-lon coordinates to utm values and finding 20 closest points
    #to each point.
    # x, y, __, __ = reproject_utm(cl_lat, cl_lon)
    all_pts = np.vstack((cl_lon, cl_lat)).T
    kdt = sp.cKDTree(all_pts)
    pt_dist, pt_ind = kdt.query(all_pts, k = 20, distance_upper_bound = 500)

    #loop through each reach and find the neighboring reaches.
    neighbors = np.zeros((len(reaches),20))
    uniq_rch = np.unique(cl_rch)
    for ind in list(range(len(uniq_rch))):
        #print(ind)
        rch = np.where(cl_rch == uniq_rch[ind])

        id_filter = np.where(pt_ind[rch,:] != len(cl_rch))
        new_ids = pt_ind[rch,:][id_filter]

        all_nghbrs = np.unique(cl_rch[new_ids])
        nghbrs = all_nghbrs[np.where(all_nghbrs != uniq_rch[ind])[0]]

        num_ngh = len(nghbrs)
        rch_loc = np.where(reaches == uniq_rch[ind])[0]
        if num_ngh == 0:
            continue
        else:
            neighbors[rch_loc,0:num_ngh] = nghbrs

    return neighbors

###############################################################################

def calc_segDist(subcls_lon, subcls_lat, subcls_rch_id, subcls_facc,
                 subcls_rch_ind):

    #loop through each reach and calculate flow distance.
    segDist = np.zeros(len(subcls_lon))
    uniq_segs = np.unique(subcls_rch_id)
    for ind in list(range(len(uniq_segs))):
        #for a single reach, reproject lat-lon coordinates to utm.
        #print(ind, uniq_segs[ind])
        seg = np.where(subcls_rch_id == uniq_segs[ind])[0]
        seg_lon = subcls_lon[seg]
        seg_lat = subcls_lat[seg]
        # seg_x, seg_y, __, __ = reproject_utm(seg_lat, seg_lon)
        upa = subcls_facc[seg]

        #order the reach points based on index values, then calculate the
        #eculdean distance bewteen each ordered point.
        order_ids = np.argsort(subcls_rch_ind[seg])
        dist = np.zeros(len(seg))
        dist[order_ids[0]] = 0
  
        for idx in list(range(len(order_ids)-1)):
            coords_1 = (seg_lat[order_ids[idx]], seg_lon[order_ids[idx]])
            coords_2 = (seg_lat[order_ids[idx+1]], seg_lon[order_ids[idx+1]])
            d =  geopy.distance.geodesic(coords_1, coords_2).m
            # d = np.sqrt((seg_x[order_ids[idx]]-seg_x[order_ids[idx+1]])**2 +
            #             (seg_y[order_ids[idx]]-seg_y[order_ids[idx+1]])**2)
            dist[order_ids[idx+1]] = d + dist[order_ids[idx]]

        #format flow distance as array and determine flow direction by flowacc.
        dist = np.array(dist)
        start = upa[np.where(dist == np.min(dist))[0][0]]
        end = upa[np.where(dist == np.max(dist))[0][0]]

        if end > start:
            segDist[seg] = abs(dist-np.max(dist))

        else:
            segDist[seg] = dist

    return segDist

###############################################################################

def calc_rchLen(merge):
    unq_rch = np.unique(merge.reach_id)
    rch_len = np.zeros(len(merge.reach_id))
    for r in list(range(len(unq_rch))):
        rch = np.where(merge.reach_id == unq_rch[r])[0]
        rch_len[rch] = np.max(merge.rch_dist6[rch])
    return rch_len

###############################################################################

def swot_obs_percentage(subcls, subreaches):

    """
    FUNCTION:
        Calculating the SWOT coverage for each overpass along a reach.

    INPUTS
        subcls -- Object containing attributes for the high-resolution centerline.
            [attributes used]:
                reach_id -- Reach IDs for along the high-resolution centerline.
                orbits -- SWOT orbit locations along the high-resolution
                    centerline.

    OUTPUTS
        swot_coverage -- 2-D array of the minimum and maximum swot coverage
            per reach (%). Dimension is [number of reaches, 2] with the first
            column representing the minimum swot coverage for a particular
            reach and the second column representing the maximum swot coverage
            for a particular reach.
    """

    # Set variables.
    uniq_rch = np.unique(subreaches.id)
    swot_coverage = np.zeros((len(uniq_rch), 75))
    swot_orbits = np.zeros((len(uniq_rch), 75))
    max_obs = np.zeros(len(uniq_rch))
    med_obs = np.zeros(len(uniq_rch))
    mean_obs = np.zeros(len(uniq_rch))

    # Loop through each reach and calculate the coverage for each swot overpass.
    for ind in list(range(len(uniq_rch))):
        rch = np.where(subcls.reach_id == uniq_rch[ind])[0]
        orbs = subcls.orbits[rch]
        max_obs[ind] = np.max(subcls.num_obs[rch])
        med_obs[ind] = np.median(subcls.num_obs[rch])
        mean_obs[ind] = np.mean(subcls.num_obs[rch])
        uniq_orbits = np.unique(orbs)
        uniq_orbits = uniq_orbits[np.where(uniq_orbits>0)[0]]

        if len(uniq_orbits) == 0:
            continue

        for idz in list(range(len(uniq_orbits))):
            rows = np.where(orbs == uniq_orbits[idz])[0]
            swot_coverage[ind,idz] = (len(rows)/len(rch))*100
            swot_orbits[ind,idz] = uniq_orbits[idz]

    return swot_coverage, swot_orbits, max_obs, med_obs, mean_obs

###############################################################################

def delete_rchs(reaches, rmv):
    reaches.id = np.delete(reaches.id, rmv, axis = 0)
    reaches.cl_id = np.delete(reaches.cl_id, rmv, axis = 1)
    reaches.x = np.delete(reaches.x, rmv, axis = 0)
    reaches.x_min = np.delete(reaches.x_min, rmv, axis = 0)
    reaches.x_max = np.delete(reaches.x_max, rmv, axis = 0)
    reaches.y = np.delete(reaches.y, rmv, axis = 0)
    reaches.y_min = np.delete(reaches.y_min, rmv, axis = 0)
    reaches.y_max = np.delete(reaches.y_max, rmv, axis = 0)
    reaches.len = np.delete(reaches.len, rmv, axis = 0)
    reaches.wse = np.delete(reaches.wse, rmv, axis = 0)
    reaches.wse_var = np.delete(reaches.wse_var, rmv, axis = 0)
    reaches.wth = np.delete(reaches.wth, rmv, axis = 0)
    reaches.wth_var = np.delete(reaches.wth_var, rmv, axis = 0)
    reaches.slope = np.delete(reaches.slope, rmv, axis = 0)
    reaches.rch_n_nodes = np.delete(reaches.rch_n_nodes, rmv, axis = 0)
    reaches.grod = np.delete(reaches.grod, rmv, axis = 0)
    reaches.grod_fid = np.delete(reaches.grod_fid, rmv, axis = 0)
    reaches.hfalls_fid = np.delete(reaches.hfalls_fid, rmv, axis = 0)
    reaches.lakeflag = np.delete(reaches.lakeflag, rmv, axis = 0)
    reaches.nchan_max = np.delete(reaches.nchan_max, rmv, axis = 0)
    reaches.nchan_mod = np.delete(reaches.nchan_mod, rmv, axis = 0)
    reaches.dist_out = np.delete(reaches.dist_out, rmv, axis = 0)
    reaches.n_rch_up = np.delete(reaches.n_rch_up, rmv, axis = 0)
    reaches.n_rch_down = np.delete(reaches.n_rch_down, rmv, axis = 0)
    reaches.rch_id_up = np.delete(reaches.rch_id_up, rmv, axis = 1)
    reaches.rch_id_down = np.delete(reaches.rch_id_down, rmv, axis = 1)
    reaches.max_obs = np.delete(reaches.max_obs, rmv, axis = 0)
    reaches.orbits = np.delete(reaches.orbits, rmv, axis = 1)
    reaches.facc = np.delete(reaches.facc, rmv, axis = 0)
    reaches.iceflag = np.delete(reaches.iceflag, rmv, axis = 1)
    reaches.max_wth = np.delete(reaches.max_wth, rmv, axis = 0)
    reaches.river_name = np.delete(reaches.river_name, rmv, axis = 0)
    reaches.low_slope = np.delete(reaches.low_slope, rmv, axis = 0)
    reaches.edit_flag = np.delete(reaches.edit_flag, rmv, axis = 0)

###############################################################################

def reach_attributes(subcls):

    """
    FUNCTION:
        Creates reach locations and attributes from the high-resolution
        centerline points for each unique Reach ID.

    INPUTS
        rch_eps -- List of indexes for all reach endpoints.
        eps_ind -- Spatial query array containing closest point indexes
            for each reach endpoint.
        eps_dist -- Spatial query array containing closest point distances
            for each reach endpoint.
        subcls -- Object containing attributes for the high-resolution centerline.
            [attributes used]:
                reach_id -- Reach IDs for along the high-resolution centerline.
                rch_len5 -- Reach lengths along the high-resolution centerline.
                elv -- Elevations along the high-resolution centerline (meters).
                wth -- Widths along the high_resolution centerline (meters).
                nchan -- Number of channels along the high_resolution centerline.
                lon -- Longitude values.
                lat -- Latitude values.
                grod -- GROD IDs along the high-resolution centerline.
                node_id -- Node IDs along the high_resolution centerline.
                rch_dist5 -- Flow distance along the high-resolution centerline
                    (meters).
                type5 -- Type flag for each point in the high-resolution
                    centerline (1 = river, 2 = lake, 3 = lake on river,
                    4 = dam, 5 = no topology).
                facc -- Flow accumulation along the high-resolution ceterline
                    (km^2).

    OUTPUTS
        Reach_ID -- Reach ID respresenting a single node location.
        reach_x -- Average longitude value calculated from the
            high-resolution centerline points associated with a reach.
        reach_y -- Average latitude value calculated from the
            high-resolution centerline points associated with a reach.
        reach_len -- Reach length for a single reach (meters).
        reach_wse -- Average water surface elevation value calculated from the
            high-resolution centerlines points assosicated with a reach (meters).
        reach_wse_var -- Water surface elevation variablity calculated from the
            high-resolution centerlines points assosicated with a reach (meters).
        reach_wth -- Average width value calculated from the high-resolution
            centerlines points assosicated with a reach (meters).
        reach_wth_var -- Width variablity calculated from the high-resolution
            centerlines points assosicated with a reach (meters).
        reach_nchan_max -- Maximum number of channels calculated from
            the high-resolution centerline points associated with a reach.
        reach_nchan_mod -- Mode of the number of channels calculated from the
            high-resolution centerline points associated with a reach.
        reach_grod_id -- GROD ID associated with a reach.
        reach_n_nodes -- Number of nodes associated with a reach.
        reach_slope -- Slope calculated from the high_resolution centerline
            points associated with a reach (m/km).
        rch_id_up -- List of reach IDs for the upstream reaches.
        rch_id_down -- List of reach IDs for the downstream reaches.
        n_rch_up -- Number of upstream reaches.
        n_rch_down -- Number of dowstream reaches.
        rch_lakeflag -- GRWL lakeflag ID associated with a reach.
        rch_grod_id -- GROD dam location associated with a reach.
        rch_grod_fid -- GROD ID associated with a reach.
        rch_hfalls_fid -- HydroFALLS ID associated with a reach.
        rch_lake_id -- Prior Lake Database ID associated with a reach.
        reach_facc -- Flow accumulation associated with a reach.
    """

    # Set variables.
    Reach_ID = np.zeros(len(np.unique(subcls.reach_id)))
    reach_x = np.zeros(len(np.unique(subcls.reach_id)))
    reach_y = np.zeros(len(np.unique(subcls.reach_id)))
    reach_x_max = np.zeros(len(np.unique(subcls.reach_id)))
    reach_x_min = np.zeros(len(np.unique(subcls.reach_id)))
    reach_y_max = np.zeros(len(np.unique(subcls.reach_id)))
    reach_y_min = np.zeros(len(np.unique(subcls.reach_id)))
    reach_wse = np.zeros(len(np.unique(subcls.reach_id)))
    reach_wse_var = np.zeros(len(np.unique(subcls.reach_id)))
    reach_wth = np.zeros(len(np.unique(subcls.reach_id)))
    reach_wth_var = np.zeros(len(np.unique(subcls.reach_id)))
    reach_facc = np.zeros(len(np.unique(subcls.reach_id)))
    reach_len = np.zeros(len(np.unique(subcls.reach_id)))
    reach_nchan_max = np.zeros(len(np.unique(subcls.reach_id)))
    reach_nchan_mod = np.zeros(len(np.unique(subcls.reach_id)))
    reach_n_nodes = np.zeros(len(np.unique(subcls.reach_id)))
    reach_slope = np.zeros(len(np.unique(subcls.reach_id)))
    rch_grod_id = np.zeros(len(np.unique(subcls.reach_id)))
    rch_grod_fid = np.zeros(len(np.unique(subcls.reach_id)))
    rch_hfalls_fid = np.zeros(len(np.unique(subcls.reach_id)))
    rch_lakeflag = np.zeros(len(np.unique(subcls.reach_id)))
    rch_lake_id = np.zeros(len(np.unique(subcls.reach_id)))

    # Loop through and calculate reach locations and attributes for each
    # unique reach ID.
    uniq_rch = np.unique(subcls.reach_id)
    for ind in list(range(len(uniq_rch))):
        #print(ind)
        reach = np.where(subcls.reach_id == uniq_rch[ind])[0]
        Reach_ID[ind] = int(np.unique(subcls.reach_id[reach]))
        reach_x[ind] = np.mean(subcls.lon[reach])
        reach_y[ind] = np.mean(subcls.lat[reach])
        reach_x_max[ind] = np.max(subcls.lon[reach])
        reach_x_min[ind] = np.min(subcls.lon[reach])
        reach_y_max[ind] = np.max(subcls.lat[reach])
        reach_y_min[ind] = np.min(subcls.lat[reach])
        reach_len[ind] = np.unique(subcls.rch_len6[reach])
        reach_wse[ind] = np.median(subcls.elv[reach])
        reach_wse_var[ind] = np.var(subcls.elv[reach])
        reach_wth[ind] = np.median(subcls.wth[reach])
        reach_wth_var[ind] = np.var(subcls.wth[reach])
        reach_facc[ind] = np.max(subcls.facc[reach])
        reach_nchan_max[ind] = np.max(subcls.nchan[reach])
        reach_nchan_mod[ind] = max(set(list(subcls.nchan[reach])), key=list(subcls.nchan[reach]).count)
        reach_n_nodes[ind] = len(np.unique(subcls.node_id[reach]))
        rch_lakeflag[ind] = max(set(list(subcls.lake[reach])), key=list(subcls.lake[reach]).count)
        rch_lake_id[ind] = max(set(list(subcls.lake_id[reach])), key=list(subcls.lake_id[reach]).count)

        # Find grod type per reach.
        GROD = np.copy(subcls.grod[reach])
        GROD[np.where(GROD > 4)] = 0
        rch_grod_id[ind] = np.max(GROD)
        # Assign grod and hydrofalls ids to reach.
        ID = np.where(GROD == np.max(GROD))[0][0]
        if np.max(GROD) == 0:
            rch_grod_fid[ind] = 0
        elif np.max(GROD) == 4:
            rch_hfalls_fid[ind] = subcls.hfalls_fid[reach[ID]]
        else:
            rch_grod_fid[ind] = subcls.grod_fid[reach[ID]]

        # Slope calculation.
        slope_pts = np.vstack([subcls.rch_dist6[reach]/1000, np.ones(len(reach))]).T
        slope, intercept = np.linalg.lstsq(slope_pts, subcls.elv[reach], rcond=None)[0]
        reach_slope[ind] = abs(slope) # m/km

    return(Reach_ID, reach_x, reach_y, reach_x_max, reach_x_min, reach_y_max,
           reach_y_min, reach_len, reach_wse, reach_wse_var, reach_wth,
           reach_wth_var, reach_nchan_max, reach_nchan_mod, reach_n_nodes,
           reach_slope, rch_grod_id, rch_lakeflag, reach_facc, rch_grod_fid,
           rch_hfalls_fid, rch_lake_id)

###############################################################################

def calc_attributes_from_nodes(subreaches, nodes):
    dist_out = np.zeros(len(subreaches.id))
    river_name = np.zeros(len(subreaches.id),dtype=Object)
    max_wth = np.zeros(len(subreaches.id))
    edit_flag = np.zeros(len(subreaches.id),dtype=Object)
    for ind in list(range(len(subreaches.id))):
        rch = np.where(nodes.reach_id == subreaches.id[ind])[0]
        dist_out[ind] = np.max(nodes.dist_out[rch])
        river_name[ind] = np.unique(nodes.river_name[rch])[0]
        max_wth[ind] = np.max(nodes.max_wth[rch])
        vals = np.unique(nodes.edit_flag[rch])
        if len(vals) == 1:
            edit_flag[ind] = vals[0]
        else:
            all_vals = ','.join(vals)
            edit_flag[ind] = all_vals
          
    return dist_out, river_name, max_wth, edit_flag

###############################################################################

def calc_cl_iceflag(reaches, centerlines):
    ice_flag = np.zeros((366, len(centerlines.cl_id)))
    for r in list(range(len(reaches.id))):
        rind = np.where(centerlines.reach_id[0,:] == reaches.id[r])[0]
        row = reaches.iceflag[:,r]
        row_reps = np.repeat(row, len(rind))
        row_arr = row_reps.reshape((366,len(rind)))
        ice_flag[:,rind] = row_arr
    return ice_flag

###############################################################################

def calc_cls_iceflag(subreaches, centerlines):
    ice_flag = np.zeros((366, len(subreaches.id)))
    for r in list(range(len(subreaches.id))):
        rind = np.where(centerlines.reach_id[0,:] == subreaches.id[r])[0]
        row = centerlines.ice_flag[:,rind[0]]
        ice_flag[:,r] = row
    return ice_flag

###############################################################################

def format_cl_node_ids_pt2(nodes, centerlines):

    """
    FUNCTION:
        Assigns and formats the node ids along high_resolution centerline
        at the continental scale. This is a subfunction of "format_cl_node_ids"
        in order to speed up the code.

    INPUTS
        nodes -- Object containing attributes for the nodes.
        centerlines -- Object containing attributes for the high-resolution centerline.

    OUTPUTS
        cl_nodes_id - Node IDs for each high-resolution centerline point.
    """

    nodes_x = np.zeros(len(nodes.id))
    nodes_y = np.zeros(len(nodes.id))
    for idx in list(range(len(nodes.id))):
        nodes_x[idx], nodes_y[idx], __ , __ = utm.from_latlon(nodes.y[idx], nodes.x[idx])

    all_pts = np.vstack((nodes_x, nodes_y)).T
    kdt = sp.cKDTree(all_pts)

    cl_nodes_id= np.zeros([4,len(centerlines.cl_id)])
    for ind in list(range(len(nodes.id))):
        #print(ind)

        # converting coordinates for centerlines points.
        cp1 = np.where(centerlines.cl_id == nodes.cl_id[0,ind])[0]
        cp2 = np.where(centerlines.cl_id == nodes.cl_id[1,ind])[0]
        cp1_x, cp1_y, __, __ = utm.from_latlon(centerlines.y[cp1], centerlines.x[cp1])
        cp2_x, cp2_y, __, __ = utm.from_latlon(centerlines.y[cp2], centerlines.x[cp2])

        cp1_pts = np.vstack((cp1_x, cp1_y)).T
        cp1_dist, cp1_ind = kdt.query(cp1_pts, k = 4, distance_upper_bound = 200)
        cp2_pts = np.vstack((cp2_x, cp2_y)).T
        cp2_dist, cp2_ind = kdt.query(cp2_pts, k = 4, distance_upper_bound = 200)

        rmv1 = np.where(cp1_ind[0] == len(nodes.id))[0]
        rmv2 = np.where(cp2_ind[0] == len(nodes.id))[0]
        cp1_ind = np.delete(cp1_ind[0], rmv1)
        cp2_ind = np.delete(cp2_ind[0], rmv2)

        cp1_nodes = nodes.id[np.where(nodes.id[cp1_ind] != nodes.id[ind])[0]]
        cp2_nodes = nodes.id[np.where(nodes.id[cp2_ind] != nodes.id[ind])[0]]

        cl_nodes_id[0:len(cp1_nodes),cp1] = cp1_nodes.reshape((len(cp1_nodes),1))
        cl_nodes_id[0:len(cp2_nodes),cp2] = cp2_nodes.reshape((len(cp2_nodes),1))

    return cl_nodes_id

###############################################################################

def format_cl_node_ids(nodes, centerlines, verbose):

    """
    FUNCTION:
        Assigns and formats the node ids along high_resolution centerline
        at the continental scale.

    INPUTS
        nodes -- Object containing attributes for the nodes.
        centerlines -- Object containing attributes for the high-resolution centerline.

    OUTPUTS
        cl_nodes_id - Node IDs for each high-resolution centerline point.
    """

    cl_nodes_id = np.zeros([4,len(centerlines.cl_id)])
    # divide up into basin level 6 to see if it makes things faster...
    level_nodes = np.array([int(str(ind)[0:6]) for ind in nodes.id])
    level_cl = np.array([int(str(ind)[0:6]) for ind in centerlines.node_id[0,:]])
    uniq_level = np.unique(level_cl)
    for ind in list(range(len(uniq_level))):
        if verbose == True:
            print(str(ind) + ' of ' + str(len(uniq_level)))

        cl_ind = np.where(level_cl == uniq_level[ind])[0]
        nodes_ind = np.where(level_nodes == uniq_level[ind])[0]

        Subnodes = Object()
        Subcls = Object()

        Subnodes.id = nodes.id[nodes_ind]
        Subnodes.x = nodes.x[nodes_ind]
        Subnodes.y = nodes.y[nodes_ind]
        Subnodes.cl_id = nodes.cl_id[:,nodes_ind]
        Subcls.cl_id = centerlines.cl_id[cl_ind]
        Subcls.x = centerlines.x[cl_ind]
        Subcls.y = centerlines.y[cl_ind]

        cl_nodes_id[:,cl_ind] = format_cl_node_ids_pt2(Subnodes, Subcls)

    return cl_nodes_id

################################################################################

def format_cl_rch_ids(reaches, centerlines, verbose):

    """
    FUNCTION:
        Assigns and formats the reach ids along high_resolution centerline
        at the continental scale.

    INPUTS
        reaches -- Object containing attributes for the reaches.
        centerlines -- Object containing attributes for the high-resolution centerline.

    OUTPUTS
        cl_rch_id - Reach IDs for each high-resolution centerline point.
    """

    cl_rch_id = np.zeros([4,len(centerlines.cl_id)])
    for ind in list(range(len(reaches.id))):
        if verbose == True:
            print(str(ind) + ' of ' + str(len(reaches.id)))
        up = np.where(reaches.rch_id_up[:,ind] > 0)[0]
        down = np.where(reaches.rch_id_down[:,ind] > 0)[0]

        # converting coordinates for centerlines points.
        cp1 = np.where(centerlines.cl_id == reaches.cl_id[0,ind])[0]
        cp2 = np.where(centerlines.cl_id == reaches.cl_id[1,ind])[0]
        cp1_x, cp1_y, __, __ = utm.from_latlon(centerlines.y[cp1], centerlines.x[cp1])
        cp2_x, cp2_y, __, __ = utm.from_latlon(centerlines.y[cp2], centerlines.x[cp2])
        
        vals_up = reaches.rch_id_up[0:4,ind]
        vals_down = reaches.rch_id_down[0:4,ind]

        if len(up) == 0 and len(down) == 0:
            continue

        if len(up) > 0 and len(down) == 0:

            up_lon = reaches.x[np.where(reaches.id == reaches.rch_id_up[:,ind][up[0]])]
            up_lat = reaches.y[np.where(reaches.id == reaches.rch_id_up[:,ind][up[0]])]
            up_x, up_y, __, __ = utm.from_latlon(up_lat, up_lon)

            d1 = np.sqrt(((cp1_x - up_x)**2 + (cp1_y - up_y)**2))
            d2 = np.sqrt(((cp2_x - up_x)**2 + (cp2_y - up_y)**2))

            if d1 > d2:
                cl_rch_id[:,cp2] = vals_up.reshape((4,1))
            if d1 < d2:
                cl_rch_id[:,cp1] = vals_up.reshape((4,1))

        if len(up) == 0 and len(down) > 0:
            dn_lon = reaches.x[np.where(reaches.id == reaches.rch_id_down[:,ind][down[0]])]
            dn_lat = reaches.y[np.where(reaches.id == reaches.rch_id_down[:,ind][down[0]])]
            dn_x, dn_y, __, __ = utm.from_latlon(dn_lat, dn_lon)

            d1 = np.sqrt(((cp1_x - dn_x)**2 + (cp1_y - dn_y)**2))
            d2 = np.sqrt(((cp2_x - dn_x)**2 + (cp2_y - dn_y)**2))

            if d1 > d2:
                cl_rch_id[:,cp2] = vals_down.reshape((4,1))
            if d1 < d2:
                cl_rch_id[:,cp1] = vals_down.reshape((4,1))

        if len(up) > 0 and len(down) > 0:

            up_lon = reaches.x[np.where(reaches.id == reaches.rch_id_up[:,ind][up[0]])] #changed 0 to ind in rch_id_up rows.
            up_lat = reaches.y[np.where(reaches.id == reaches.rch_id_up[:,ind][up[0]])]
            up_x, up_y, __, __ = utm.from_latlon(up_lat, up_lon)

            dn_lon = reaches.x[np.where(reaches.id == reaches.rch_id_down[:,ind][down[0]])]
            dn_lat = reaches.y[np.where(reaches.id == reaches.rch_id_down[:,ind][down[0]])]
            dn_x, dn_y, __, __ = utm.from_latlon(dn_lat, dn_lon)

            d1 = np.sqrt(((cp1_x - up_x)**2 + (cp1_y - up_y)**2))
            d2 = np.sqrt(((cp1_x - dn_x)**2 + (cp1_y - dn_y)**2))

            #### need to check!!!!!!!
            if d1 > d2:
                cl_rch_id[:,cp1] = vals_down.reshape((4,1))
                cl_rch_id[:,cp2] = vals_up.reshape((4,1))
            if d1 < d2:
                cl_rch_id[:,cp1] = vals_up.reshape((4,1))
                cl_rch_id[:,cp2] = vals_down.reshape((4,1))

    return cl_rch_id

###############################################################################

def update_rch_indexes(subcls, new_rch_id):

    """
    FUNCTION:
        Re-orders the point indexes within a reach and defines reach endpoints.

    INPUTS
        subcls -- Object containing attributes for the high-resolution
            centerline.
            [attributes used]:
                lon -- Longitude values along the high-resolution centerline.
                lat -- Latitude values along the high-resolution centerline.
                seg -- GRWL segment values along the high-resolution centerline.
                ind -- Point indexes for each GRWL segment along the
                    high-resolution centerline.
        new_rch_id -- 1-D array of the reach IDs to re-format the point
            indexes.

    OUTPUTS
        new_rch_ind -- Updated reach indexes (1-D array).
        new_rch_eps -- Updated reach endpoints (1-D array).
    """

    # Set variables and find unique reaches.
    uniq_rch = np.unique(new_rch_id)
    new_rch_ind = np.zeros(len(subcls.ind))
    new_rch_eps = np.zeros(len(subcls.ind))

    # Loop through each reach and re-order indexes.
    for ind in list(range(len(uniq_rch))):
        rch = np.where(new_rch_id == uniq_rch[ind])[0]
        rch_lon = subcls.lon[rch]
        rch_lat = subcls.lat[rch]
        # rch_x, rch_y, __, __ = reproject_utm(rch_lat, rch_lon)
        rch_pts = np.vstack((rch_lon, rch_lat)).T
        rch_segs = subcls.seg[rch]
        segs = np.unique(subcls.seg[rch])
        new_ind = np.zeros(len(rch))
        eps = np.zeros(len(rch))

        # Reformat indexes if multiple segments are within a reach.
        if len(segs) > 1:
            # print(ind)
            for idx in list(range(len(segs))):
                s = np.where(subcls.seg[rch] == segs[idx])[0]
                mn = np.where(subcls.ind[rch[s]] == np.min(subcls.ind[rch[s]]))[0]
                mx = np.where(subcls.ind[rch[s]] == np.max(subcls.ind[rch[s]]))[0]
                eps[s[mn]] = 1
                eps[s[mx]] = 1

            # Finding true endpoints from orginal GRWL segment extents within
            # the new reach extent.
            eps_ind = np.where(eps>0)[0]
            ep_pts = np.vstack((rch_lon[eps_ind], rch_lat[eps_ind])).T
            kdt = sp.cKDTree(rch_pts)
            if len(rch_segs) < 10: #use to be 5.
                pt_dist, pt_ind = kdt.query(ep_pts, k = len(rch_segs))
            else:
                pt_dist, pt_ind = kdt.query(ep_pts, k = 10) #use to be 5

            real_eps = []
            for idy in list(range(len(eps_ind))):
                neighbors = len(np.unique(rch_segs[pt_ind[idy,:]]))
                if neighbors == 1:
                    real_eps.append(eps_ind[idy])
            real_eps = np.array(real_eps)

            if len(real_eps) == 1 or len(real_eps) == 2:
                final_eps = real_eps

            else:
                kdt2 = sp.cKDTree(ep_pts)
                if len(ep_pts) < 4:
                    pt_dist2, pt_ind2 = kdt2.query(ep_pts, k = len(ep_pts))
                else:
                    pt_dist2, pt_ind2 = kdt2.query(ep_pts, k = 4)
                real_eps2 = np.where(pt_dist2== np.max(pt_dist2))[0]
                final_eps = real_eps2

            if len(final_eps) == 0 or len(final_eps) > 2:
                print(ind, 'problem with indexes')

            # Re-ordering points based on updated endpoints.
            new_ind[final_eps[0]]=1
            idz = final_eps[0]
            count = 2
            while np.min(new_ind) == 0:
                d = np.array([geopy.distance.geodesic((rch_lat[idz],rch_lon[idz]), (rch_lat[i], rch_lon[i])).m for i in list(range(len(rch_lon)))])
                dzero = np.where(new_ind == 0)[0]
                #vals = np.where(edits_segInd[dzero] eq 0)
                next_pt = dzero[np.where(d[dzero] == np.min(d[dzero]))[0]][0]
                new_ind[next_pt] = count
                count = count+1
                idz = next_pt

            new_rch_ind[rch] = new_ind
            ep1 = np.where(new_ind == np.min(new_ind))[0]
            ep2 = np.where(new_ind == np.max(new_ind))[0]
            new_rch_eps[rch[ep1]] = 1
            new_rch_eps[rch[ep2]] = 1
            #reverse index order to have indexes increasing in the upstream direction.
            if subcls.facc[rch[ep1]] < subcls.facc[rch[ep2]]:
                new_rch_ind[rch] = abs(new_ind - np.max(new_ind))

        # If there are no combined segments within reach keep current indexes.
        else:
            new_rch_ind[rch] = subcls.ind[rch]
            ep1 = np.where(subcls.ind[rch] == np.min(subcls.ind[rch]))[0]
            ep2 = np.where(subcls.ind[rch] == np.max(subcls.ind[rch]))[0]
            new_rch_eps[rch[ep1]] = 1
            new_rch_eps[rch[ep2]] = 1
            #reverse index order to have indexes increasing in the upstream direction.
            if subcls.facc[rch[ep1]] < subcls.facc[rch[ep2]]:
                new_rch_ind[rch] = abs(new_rch_ind[rch] - np.max(new_rch_ind[rch]))

    return new_rch_ind, new_rch_eps

###############################################################################

def update_cl_ids(subreaches, subnodes_ids, subcls):

    ID = np.copy(subcls.id)
    cl_reach_id = np.copy(subcls.reach_id)
    cl_reach_ind = np.copy(subcls.rch_ind6)
    cl_node_id = np.copy(subcls.node_id)
    reach_id = np.copy(subreaches.id)
    node_id = np.copy(subnodes_ids)

    cl_id = np.zeros(len(ID))
    rch_cl_id = np.full((len(reach_id), 2), 0)
    node_cl_id = np.full((len(node_id), 2), 0)
    for ind in list(range(len(reach_id))):
        rch = np.where(cl_reach_id == reach_id[ind])[0]
        cl_id[rch] = cl_reach_ind[rch]
        rch_cl_id[ind, 0] = np.min(cl_id[rch])
        rch_cl_id[ind, 1] = np.max(cl_id[rch])
        uniq_nodes = np.unique(cl_node_id[rch])
        for idx in list(range(len(uniq_nodes))):
            Nodes = np.where(cl_node_id == uniq_nodes[idx])[0]
            index = np.where(node_id == uniq_nodes[idx])[0]
            node_cl_id[index,0] = np.min(cl_id[Nodes])
            node_cl_id[index,1] = np.max(cl_id[Nodes])

    return(cl_id, rch_cl_id, node_cl_id)


###############################################################################

def centerline_ids_pt2(subreaches, subnodes, subcls, cl_basins, rch_basins,
                       node_basins, cnt):

    """
    FUNCTION:
        Creating unique centerline ids that are ordered per reach and per node
        within a level 6 basin.
        This is a subfunction of "centerline_ids."

    INPUTS
        subreaches -- Object containing attributes for the reaches.
        subcls -- Object containing attributes for the high-resolution centerline.
        cnt -- starting number to add to existing indexes.
        cl_basins -- High-resolution centerline point indexes for the current level
            6 basin.
        rch_basins --  Reach indexes for the current level 6 basin.
        node_basins --  Node indexes for the current level 6 basin.

    OUTPUTS
        cl_id - Unique IDs for each high-resolution centerline point that are
            ordered within the reaches of a level 6 basin.
        rch_cl_id - Array containing the max and min high-resolution centerline
            IDs for each reach in a level 6 basin.
        node_cl_id - Array containing the max and min high_resolution centerline
            IDs for each node in a level 6 basin.
    """

    ID = np.copy(subcls.id[cl_basins])
    cl_reach_id = np.copy(subcls.reach_id[cl_basins])
    cl_reach_ind = np.copy(subcls.rch_ind6[cl_basins])
    cl_node_id = np.copy(subcls.node_id[cl_basins])
    reach_id = np.copy(subreaches.id[rch_basins])
    node_id = np.copy(subnodes.id[node_basins])

    cl_id = np.zeros(len(ID))
    rch_cl_id = np.full((len(reach_id), 2), 0)
    node_cl_id = np.full((len(node_id), 2), 0)
    cnt = cnt
    for ind in list(range(len(reach_id))):
        rch = np.where(cl_reach_id == reach_id[ind])[0]
        cl_id[rch] = cl_reach_ind[rch] + cnt
        rch_cl_id[ind, 0] = np.min(cl_id[rch])
        rch_cl_id[ind, 1] = np.max(cl_id[rch])
        cnt = np.max(cl_id)+1
        uniq_nodes = np.unique(cl_node_id[rch])
        for idx in list(range(len(uniq_nodes))):
            Nodes = np.where(cl_node_id == uniq_nodes[idx])[0]
            index = np.where(node_id == uniq_nodes[idx])[0]
            node_cl_id[index,0] = np.min(cl_id[Nodes])
            node_cl_id[index,1] = np.max(cl_id[Nodes])

    return(cl_id, rch_cl_id, node_cl_id)

###############################################################################

def centerline_ids(subreaches, subnodes, subcls, cnt):
    """
    FUNCTION:
        Creating unique centerline ids that are ordered per reach and per node.

    INPUTS
        subreaches -- Object containing attributes for the reaches.
        subcls -- Object containing attributes for the high-resolution centerline.
        cnt -- starting number to add to existing indexes.

    OUTPUTS
        cl_id - Unique IDs for each high-resolution centerline point that are
            ordered within the reaches.
        rch_cl_id - Array containing the max and min high-resolution centerline
            IDs for each reach.
        node_cl_id - Array containing the max and min high_resolution centerline
            IDs for each node.
    """

    cl_id = np.zeros(len(subcls.id))
    rch_cl_id = np.full((len(subreaches.id), 2), 0)
    node_cl_id = np.full((len(subnodes.id), 2), 0)
    uniq_basins = np.unique(subcls.basins)

    reach_basins = np.zeros(len(subreaches.id))
    for ind in list(range(len(subreaches.id))):
        reach_basins[ind] = int(str(subreaches.id[ind])[0:6])

    nbasins = np.zeros(len(subnodes.id))
    for ind in list(range(len(subnodes.id))):
        nbasins[ind] = int(str(subnodes.id[ind])[0:6])

    cnt = cnt
    for ind in list(range(len(uniq_basins))):
        cl_basins = np.where(subcls.basins == uniq_basins[ind])[0]
        rch_basins = np.where(reach_basins == uniq_basins[ind])[0]
        node_basins = np.where(nbasins == uniq_basins[ind])[0]
        new_cl_id, new_rch_cl_id, \
         new_node_cl_id = centerline_ids_pt2(subreaches, subnodes, subcls,
                                             cl_basins, rch_basins, node_basins, cnt)

        cl_id[cl_basins] = new_cl_id
        rch_cl_id[rch_basins,:] = new_rch_cl_id
        node_cl_id[node_basins,:] = new_node_cl_id
        cnt = np.max(cl_id)+1

    return(cl_id, rch_cl_id, node_cl_id)

###############################################################################

def write_database_nc(centerlines, reaches, nodes, region, outfile):

    """
    FUNCTION:
        Outputs the SWOT River Database (SWORD) information in netcdf
        format. The file contains attributes for the high-resolution centerline,
        nodes, and reaches.

    INPUTS
        centerlines -- Object containing lcation and attribute information
            along the high-resolution centerline.
        reaches -- Object containing lcation and attribute information for
            each reach.
        nodes -- Object containing lcation and attribute information for
            each node.
        outfile -- Path for netcdf to be written.

    OUTPUTS
        SWORD NetCDF -- NetCDF file containing attributes for the high-resolution
            centerline, node, and reach locations.
    """

    start = time.time()

    # global attributes
    root_grp = nc.Dataset(outfile, 'w', format='NETCDF4')
    root_grp.x_min = np.min(centerlines.x)
    root_grp.x_max = np.max(centerlines.x)
    root_grp.y_min = np.min(centerlines.y)
    root_grp.y_max = np.max(centerlines.y)
    root_grp.Name = region
    root_grp.production_date = time.strftime("%d-%b-%Y %H:%M:%S", time.gmtime()) #utc time
    #root_grp.history = 'Created ' + time.ctime(time.time())

    # groups
    cl_grp = root_grp.createGroup('centerlines')
    node_grp = root_grp.createGroup('nodes')
    rch_grp = root_grp.createGroup('reaches')
    # subgroups
    sub_grp1 = rch_grp.createGroup('area_fits')
    sub_grp2 = rch_grp.createGroup('discharge_models')
    # discharge subgroups
    qgrp1 = sub_grp2.createGroup('unconstrained')
    qgrp2 = sub_grp2.createGroup('constrained')
    # unconstrained discharge models
    ucmod1 = qgrp1.createGroup('MetroMan')
    ucmod2 = qgrp1.createGroup('BAM')
    ucmod3 = qgrp1.createGroup('HiVDI')
    ucmod4 = qgrp1.createGroup('MOMMA')
    ucmod5 = qgrp1.createGroup('SADS')
    ucmod6 = qgrp1.createGroup('SIC4DVar')
    # constrained discharge models
    cmod1 = qgrp2.createGroup('MetroMan')
    cmod2 = qgrp2.createGroup('BAM')
    cmod3 = qgrp2.createGroup('HiVDI')
    cmod4 = qgrp2.createGroup('MOMMA')
    cmod5 = qgrp2.createGroup('SADS')
    cmod6 = qgrp2.createGroup('SIC4DVar')

    # dimensions
    #root_grp.createDimension('d1', 2)
    cl_grp.createDimension('num_points', len(centerlines.cl_id))
    cl_grp.createDimension('num_domains', 4)

    node_grp.createDimension('num_nodes', len(nodes.id))
    node_grp.createDimension('num_ids', 2)

    rch_grp.createDimension('num_reaches', len(reaches.id))
    rch_grp.createDimension('num_ids', 2)
    rch_grp.createDimension('num_domains', 4)
    rch_grp.createDimension('julian_day', 366)
    rch_grp.createDimension('orbits', 75)
    sub_grp1.createDimension('nCoeffs', 2)
    sub_grp1.createDimension('nReg', 3)
    sub_grp1.createDimension('num_domains', 4)

    # centerline variables
    cl_id = cl_grp.createVariable(
        'cl_id', 'i8', ('num_points',), fill_value=-9999.)
    cl_x = cl_grp.createVariable(
        'x', 'f8', ('num_points',), fill_value=-9999.)
    cl_x.units = 'degrees east'
    cl_y = cl_grp.createVariable(
        'y', 'f8', ('num_points',), fill_value=-9999.)
    cl_y.units = 'degrees north'
    reach_id = cl_grp.createVariable(
        'reach_id', 'i8', ('num_domains','num_points'), fill_value=-9999.)
    reach_id.format = 'CBBBBBRRRRT'
    node_id = cl_grp.createVariable(
        'node_id', 'i8', ('num_domains','num_points'), fill_value=-9999.)
    node_id.format = 'CBBBBBRRRRNNNT'

    # node variables
    Node_ID = node_grp.createVariable(
        'node_id', 'i8', ('num_nodes',), fill_value=-9999.)
    Node_ID.format = 'CBBBBBRRRRNNNT'
    node_cl_id = node_grp.createVariable(
        'cl_ids', 'i8', ('num_ids','num_nodes'), fill_value=-9999.)
    node_x = node_grp.createVariable(
        'x', 'f8', ('num_nodes',), fill_value=-9999.)
    node_x.units = 'degrees east'
    node_y = node_grp.createVariable(
        'y', 'f8', ('num_nodes',), fill_value=-9999.)
    node_y.units = 'degrees north'
    node_len = node_grp.createVariable(
        'node_length', 'f8', ('num_nodes',), fill_value=-9999.)
    node_len.units = 'meters'
    node_rch_id = node_grp.createVariable(
        'reach_id', 'i8', ('num_nodes',), fill_value=-9999.)
    node_rch_id.format = 'CBBBBBRRRRT'
    node_wse = node_grp.createVariable(
        'wse', 'f8', ('num_nodes',), fill_value=-9999.)
    node_wse.units = 'meters'
    node_wse_var = node_grp.createVariable(
        'wse_var', 'f8', ('num_nodes',), fill_value=-9999.)
    node_wse_var.units = 'meters^2'
    node_wth = node_grp.createVariable(
        'width', 'f8', ('num_nodes',), fill_value=-9999.)
    node_wth.units = 'meters'
    node_wth_var = node_grp.createVariable(
        'width_var', 'f8', ('num_nodes',), fill_value=-9999.)
    node_wth_var.units = 'meters^2'
    node_chan_max = node_grp.createVariable(
        'n_chan_max', 'i4', ('num_nodes',), fill_value=-9999.)
    node_chan_mod = node_grp.createVariable(
        'n_chan_mod', 'i4', ('num_nodes',), fill_value=-9999.)
    node_grod_id = node_grp.createVariable(
        'obstr_type', 'i4', ('num_nodes',), fill_value=-9999.)
    node_grod_fid = node_grp.createVariable(
        'grod_id', 'i8', ('num_nodes',), fill_value=-9999.)
    node_hfalls_fid = node_grp.createVariable(
        'hfalls_id', 'i8', ('num_nodes',), fill_value=-9999.)
    node_dist_out = node_grp.createVariable(
        'dist_out', 'f8', ('num_nodes',), fill_value=-9999.)
    node_dist_out.units = 'meters'
    node_wth_coef = node_grp.createVariable(
        'wth_coef', 'f8', ('num_nodes',), fill_value=-9999.)
    node_ext_dist_coef = node_grp.createVariable(
        'ext_dist_coef', 'f8', ('num_nodes',), fill_value=-9999.)
    node_facc = node_grp.createVariable(
        'facc', 'f8', ('num_nodes',), fill_value=-9999.)
    node_facc.units = 'km^2'
    node_lakeflag = node_grp.createVariable(
        'lakeflag', 'i8', ('num_nodes',), fill_value=-9999.)
    #node_lake_id = node_grp.createVariable(
        #'lake_id', 'i8', ('num_nodes',), fill_value=-9999.)
    node_max_wth = node_grp.createVariable(
        'max_width', 'f8', ('num_nodes',), fill_value=-9999.)
    node_max_wth.units = 'meters'
    node_meand_len = node_grp.createVariable(
        'meander_length', 'f8', ('num_nodes',), fill_value=-9999.)
    node_sinuosity = node_grp.createVariable(
        'sinuosity', 'f8', ('num_nodes',), fill_value=-9999.)
    node_manual_add = node_grp.createVariable(
        'manual_add', 'i4', ('num_nodes',), fill_value=-9999.)
    node_river_name = node_grp.createVariable(
        'river_name', 'S50', ('num_nodes',))
    node_river_name._Encoding = 'ascii'
    node_edit_flag = node_grp.createVariable(
        'edit_flag',  'S50', ('num_nodes',))
    node_edit_flag._Encoding = 'ascii'

    # reach variables
    Reach_ID = rch_grp.createVariable(
        'reach_id', 'i8', ('num_reaches',), fill_value=-9999.)
    Reach_ID.format = 'CBBBBBRRRRT'
    rch_cl_id = rch_grp.createVariable(
        'cl_ids', 'i8', ('num_ids','num_reaches'), fill_value=-9999.)
    rch_x = rch_grp.createVariable(
        'x', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_x.units = 'degrees east'
    rch_x_min = rch_grp.createVariable(
        'x_min', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_x_min.units = 'degrees east'
    rch_x_max = rch_grp.createVariable(
        'x_max', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_x_max.units = 'degrees east'
    rch_y = rch_grp.createVariable(
        'y', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_y.units = 'degrees north'
    rch_y_min = rch_grp.createVariable(
        'y_min', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_y_min.units = 'degrees north'
    rch_y_max = rch_grp.createVariable(
        'y_max', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_y_max.units = 'degrees north'
    rch_len = rch_grp.createVariable(
        'reach_length', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_len.units = 'meters'
    num_nodes = rch_grp.createVariable(
        'n_nodes', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_wse = rch_grp.createVariable(
        'wse', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_wse.units = 'meters'
    rch_wse_var = rch_grp.createVariable(
        'wse_var', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_wse_var.units = 'meters^2'
    rch_wth = rch_grp.createVariable(
        'width', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_wth.units = 'meters'
    rch_wth_var = rch_grp.createVariable(
        'width_var', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_wth_var.units = 'meters^2'
    rch_facc = rch_grp.createVariable(
        'facc', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_facc.units = 'km^2'
    rch_chan_max = rch_grp.createVariable(
        'n_chan_max', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_chan_mod = rch_grp.createVariable(
        'n_chan_mod', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_grod_id = rch_grp.createVariable(
        'obstr_type', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_grod_fid = rch_grp.createVariable(
        'grod_id', 'i8', ('num_reaches',), fill_value=-9999.)
    rch_hfalls_fid = rch_grp.createVariable(
        'hfalls_id', 'i8', ('num_reaches',), fill_value=-9999.)
    rch_slope = rch_grp.createVariable(
        'slope', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_slope.units = 'meters/kilometers'
    rch_dist_out = rch_grp.createVariable(
        'dist_out', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_dist_out.units = 'meters'
    n_rch_up = rch_grp.createVariable(
        'n_rch_up', 'i4', ('num_reaches',), fill_value=-9999.)
    n_rch_down= rch_grp.createVariable(
        'n_rch_down', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_id_up = rch_grp.createVariable(
        'rch_id_up', 'i8', ('num_domains','num_reaches'), fill_value=-9999.)
    rch_id_down = rch_grp.createVariable(
        'rch_id_dn', 'i8', ('num_domains','num_reaches'), fill_value=-9999.)
    rch_lakeflag = rch_grp.createVariable(
        'lakeflag', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_iceflag = rch_grp.createVariable(
        'iceflag', 'i4', ('julian_day','num_reaches'), fill_value=-9999.)
    #rch_lake_id = rch_grp.createVariable(
        #'lake_id', 'i8', ('num_reaches',), fill_value=-9999.)
    rch_swot_obs = rch_grp.createVariable(
        'swot_obs', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_orbits = rch_grp.createVariable(
        'swot_orbits', 'i8', ('orbits','num_reaches'), fill_value=-9999.)
    rch_river_name = rch_grp.createVariable(
        'river_name', 'S50', ('num_reaches',))
    rch_river_name._Encoding = 'ascii'
    rch_max_wth = rch_grp.createVariable(
        'max_width', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_max_wth.units = 'meters'
    rch_low_slope = rch_grp.createVariable(
        'low_slope_flag', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_edit_flag = rch_grp.createVariable(
        'edit_flag', 'S50', ('num_reaches',), fill_value=-9999.)
    rch_edit_flag._Encoding = 'ascii'
    # subgroup 1 - 'area_fits'
    h_break = sub_grp1.createVariable(
        'h_break', 'f8', ('num_domains','num_reaches'), fill_value=-9999.)
    h_break.units = 'meters'
    w_break = sub_grp1.createVariable(
        'w_break', 'f8', ('num_domains','num_reaches'), fill_value=-9999.)
    w_break.units = 'meters'
    h_variance = sub_grp1.createVariable(
        'h_variance', 'f8', ('num_reaches',), fill_value=-9999.)
    h_variance.units = 'meters^2'
    w_variance = sub_grp1.createVariable(
        'w_variance', 'f8', ('num_reaches',), fill_value=-9999.)
    w_variance.units = 'meters^2'
    hw_covariance = sub_grp1.createVariable(
        'hw_covariance', 'f8', ('num_reaches',), fill_value=-9999.)
    hw_covariance.units = 'meters^2'
    h_err_stdev = sub_grp1.createVariable(
        'h_err_stdev', 'f8', ('num_reaches',), fill_value=-9999.)
    h_err_stdev.units = 'meters'
    w_err_stdev = sub_grp1.createVariable(
        'w_err_stdev', 'f8', ('num_reaches',), fill_value=-9999.)
    w_err_stdev.units = 'meters'
    h_w_nobs = sub_grp1.createVariable(
        'h_w_nobs', 'f8', ('num_reaches',), fill_value=-9999.)
    fit_coeffs = sub_grp1.createVariable(
        'fit_coeffs', 'f8', ('nCoeffs','nReg','num_reaches'), fill_value=-9999.)
    med_flow_area = sub_grp1.createVariable(
        'med_flow_area', 'f8', ('num_reaches',), fill_value=-9999.)
    # unconstrained discharge subgroups
    # MetroMan (ucmod1)
    uc_metroman_Abar = ucmod1.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_Abar.units = 'meters'
    uc_metroman_Abar_stdev = ucmod1.createVariable(
        'Abar_stdev', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_Abar_stdev.units = 'meters'
    uc_metroman_ninf = ucmod1.createVariable(
        'ninf', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_ninf_stdev = ucmod1.createVariable(
        'ninf_stdev', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_p = ucmod1.createVariable(
        'p', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_p_stdev = ucmod1.createVariable(
        'p_stdev', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_ninf_p_cor = ucmod1.createVariable(
        'ninf_p_cor', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_p_Abar_cor = ucmod1.createVariable(
        'p_Abar_cor', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_ninf_Abar_cor = ucmod1.createVariable(
        'ninf_Abar_cor', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_metroman_sbQ_rel = ucmod1.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # BAM (ucmod2)
    uc_bam_Abar = ucmod2.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_bam_Abar.units = 'meters'
    uc_bam_n = ucmod2.createVariable(
        'n', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_bam_sbQ_rel = ucmod2.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # HiVDI (ucmod3)
    uc_hivdi_Abar = ucmod3.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_hivdi_Abar.units = 'meters'
    uc_hivdi_alpha = ucmod3.createVariable(
        'alpha', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_hivdi_beta = ucmod3.createVariable(
        'beta', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_hivdi_sbQ_rel = ucmod3.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # MOMMA (ucmod4)
    uc_momma_B = ucmod4.createVariable(
        'B', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_momma_B.units = 'meters'
    uc_momma_H = ucmod4.createVariable(
        'H', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_momma_H.units = 'meters'
    uc_momma_Save = ucmod4.createVariable(
        'Save', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_momma_Save.units = 'meters/kilometers'
    uc_momma_sbQ_rel = ucmod4.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # SADS (ucmod5)
    uc_sads_Abar = ucmod5.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_sads_Abar.units = 'meters'
    uc_sads_n = ucmod5.createVariable(
        'n', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_sads_sbQ_rel = ucmod5.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # SIC4DVar (ucmod6)
    uc_sic4d_Abar = ucmod6.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_sic4d_Abar.units = 'meters'
    uc_sic4d_n = ucmod6.createVariable(
        'n', 'f8', ('num_reaches',), fill_value=-9999.)
    uc_sic4d_sbQ_rel = ucmod6.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # constrained discharge subgroups
    # MetroMan (cmod1)
    c_metroman_Abar = cmod1.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_Abar.units = 'meters'
    c_metroman_Abar_stdev = cmod1.createVariable(
        'Abar_stdev', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_Abar_stdev.units = 'meters'
    c_metroman_ninf = cmod1.createVariable(
        'ninf', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_ninf_stdev = cmod1.createVariable(
        'ninf_stdev', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_p = cmod1.createVariable(
        'p', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_p_stdev = cmod1.createVariable(
        'p_stdev', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_ninf_p_cor = cmod1.createVariable(
        'ninf_p_cor', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_p_Abar_cor = cmod1.createVariable(
        'p_Abar_cor', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_ninf_Abar_cor = cmod1.createVariable(
        'ninf_Abar_cor', 'f8', ('num_reaches',), fill_value=-9999.)
    c_metroman_sbQ_rel = cmod1.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # BAM (cmod2)
    c_bam_Abar = cmod2.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    c_bam_Abar.units = 'meters'
    c_bam_n = cmod2.createVariable(
        'n', 'f8', ('num_reaches',), fill_value=-9999.)
    c_bam_sbQ_rel = cmod2.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # HiDVI (cmod3)
    c_hivdi_Abar = cmod3.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    c_hivdi_Abar.units = 'meters'
    c_hivdi_alpha = cmod3.createVariable(
        'alpha', 'f8', ('num_reaches',), fill_value=-9999.)
    c_hivdi_beta = cmod3.createVariable(
        'beta', 'f8', ('num_reaches',), fill_value=-9999.)
    c_hivdi_sbQ_rel = cmod3.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # MOMMA (cmod4)
    c_momma_B = cmod4.createVariable(
        'B', 'f8', ('num_reaches',), fill_value=-9999.)
    c_momma_B.units = 'meters'
    c_momma_H = cmod4.createVariable(
        'H', 'f8', ('num_reaches',), fill_value=-9999.)
    c_momma_H.units = 'meters'
    c_momma_Save = cmod4.createVariable(
        'Save', 'f8', ('num_reaches',), fill_value=-9999.)
    c_momma_Save.units = 'meters/kilometers'
    c_momma_sbQ_rel = cmod4.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # SADS (cmod5)
    c_sads_Abar = cmod5.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    c_sads_Abar.units = 'meters'
    c_sads_n = cmod5.createVariable(
        'n', 'f8', ('num_reaches',), fill_value=-9999.)
    c_sads_sbQ_rel = cmod5.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)
    # SIC4DVar (cmod6)
    c_sic4d_Abar = cmod6.createVariable(
        'Abar', 'f8', ('num_reaches',), fill_value=-9999.)
    c_sic4d_Abar.units = 'meters'
    c_sic4d_n = cmod6.createVariable(
        'n', 'f8', ('num_reaches',), fill_value=-9999.)
    c_sic4d_sbQ_rel = cmod6.createVariable(
        'sbQ_rel', 'f8', ('num_reaches',), fill_value=-9999.)

    # saving data
    print("saving nc")

    # root group data
    #cont_str = nc.stringtochar(np.array(['NA'], 'S2'))
    #Name[:] = cont_str

    # centerline data
    cl_id[:] = centerlines.cl_id
    cl_x[:] = centerlines.x
    cl_y[:] = centerlines.y
    reach_id[:,:] = centerlines.reach_id
    node_id[:,:] = centerlines.node_id

    # node data
    Node_ID[:] = nodes.id
    node_cl_id[:,:] = nodes.cl_id
    node_x[:] = nodes.x
    node_y[:] = nodes.y
    node_len[:] = nodes.len
    node_rch_id[:] = nodes.reach_id
    node_wse[:] = nodes.wse
    node_wse_var[:] = nodes.wse_var
    node_wth[:] = nodes.wth
    node_wth_var[:] = nodes.wth_var
    node_chan_max[:] = nodes.nchan_max
    node_chan_mod[:] = nodes.nchan_mod
    node_grod_id[:] = nodes.grod
    node_grod_fid[:] = nodes.grod_fid
    node_hfalls_fid[:] = nodes.hfalls_fid
    node_dist_out[:] = nodes.dist_out
    node_wth_coef[:] = nodes.wth_coef
    node_ext_dist_coef[:] = nodes.ext_dist_coef
    node_facc[:] = nodes.facc
    node_lakeflag[:] = nodes.lakeflag
    #node_lake_id[:] = nodes.lake_id
    node_max_wth[:] = nodes.max_wth
    node_meand_len[:] = nodes.meand_len
    node_sinuosity[:] = nodes.sinuosity
    node_river_name[:] = nodes.river_name
    node_manual_add[:] = nodes.manual_add
    node_edit_flag[:] = nodes.edit_flag

    # reach data
    Reach_ID[:] = reaches.id
    rch_cl_id[:,:] = reaches.cl_id
    rch_x[:] = reaches.x
    rch_x_min[:] = reaches.x_min
    rch_x_max[:] = reaches.x_max
    rch_y[:] = reaches.y
    rch_y_min[:] = reaches.y_min
    rch_y_max[:] = reaches.y_max
    rch_len[:] = reaches.len
    num_nodes[:] = reaches.rch_n_nodes
    rch_wse[:] = reaches.wse
    rch_wse_var[:] = reaches.wse_var
    rch_wth[:] = reaches.wth
    rch_wth_var[:] = reaches.wth_var
    rch_facc[:] = reaches.facc
    rch_chan_max[:] = reaches.nchan_max
    rch_chan_mod[:] = reaches.nchan_mod
    rch_grod_id[:] = reaches.grod
    rch_grod_fid[:] = reaches.grod_fid
    rch_hfalls_fid[:] = reaches.hfalls_fid
    rch_slope[:] = reaches.slope
    rch_dist_out[:] = reaches.dist_out
    n_rch_up[:] = reaches.n_rch_up
    n_rch_down[:] = reaches.n_rch_down
    rch_id_up[:,:] = reaches.rch_id_up
    rch_id_down[:,:] = reaches.rch_id_down
    rch_lakeflag[:] = reaches.lakeflag
    rch_iceflag[:,:] = reaches.iceflag
    #rch_lake_id[:] = reaches.lake_id
    rch_swot_obs[:] = reaches.max_obs
    rch_orbits[:,:] = reaches.orbits
    rch_river_name[:] = reaches.river_name
    rch_max_wth[:] = reaches.max_wth
    rch_low_slope[:] = reaches.low_slope
    rch_edit_flag[:] = reaches.edit_flag
    # subgroup1 - area fits
    h_break[:,:] = reaches.h_break
    w_break[:,:] = reaches.w_break
    h_variance[:] = reaches.wse_var
    w_variance[:] = reaches.wth_var
    hw_covariance[:] = reaches.hw_covariance
    h_err_stdev[:] = reaches.h_err_stdev
    w_err_stdev[:] = reaches.w_err_stdev
    h_w_nobs[:] = reaches.h_w_nobs
    fit_coeffs[:,:,:] = reaches.fit_coeffs
    med_flow_area[:] = reaches.med_flow_area
    # ucmod1
    uc_metroman_Abar[:] = reaches.metroman_abar
    uc_metroman_ninf[:] = reaches.metroman_ninf
    uc_metroman_p[:] = reaches.metroman_p
    uc_metroman_Abar_stdev[:] = reaches.metroman_abar_stdev
    uc_metroman_ninf_stdev[:] = reaches.metroman_ninf_stdev
    uc_metroman_p_stdev[:] = reaches.metroman_p_stdev
    uc_metroman_ninf_p_cor[:] = reaches.metroman_ninf_p_cor
    uc_metroman_ninf_Abar_cor[:] = reaches.metroman_ninf_abar_cor
    uc_metroman_p_Abar_cor[:] = reaches.metroman_p_abar_cor
    uc_metroman_sbQ_rel[:] = reaches.metroman_sbQ_rel
    # ucmod2
    uc_bam_Abar[:] = reaches.bam_abar
    uc_bam_n[:] = reaches.bam_n
    uc_bam_sbQ_rel[:] = reaches.bam_sbQ_rel
    # ucmod3
    uc_hivdi_Abar[:] = reaches.hivdi_abar
    uc_hivdi_alpha[:] = reaches.hivdi_alpha
    uc_hivdi_beta[:] = reaches.hivdi_beta
    uc_hivdi_sbQ_rel[:] = reaches.hivdi_sbQ_rel
    # ucmod4
    uc_momma_B[:] = reaches.momma_b
    uc_momma_H[:] = reaches.momma_h
    uc_momma_Save[:] = reaches.momma_save
    uc_momma_sbQ_rel[:] = reaches.momma_sbQ_rel
    # ucmod5
    uc_sads_Abar[:] = reaches.sads_abar
    uc_sads_n[:] = reaches.sads_n
    uc_sads_sbQ_rel[:] = reaches.sads_sbQ_rel
    # ucmod6
    uc_sic4d_Abar[:] = reaches.sic4d_abar
    uc_sic4d_n[:] = reaches.sic4d_n
    uc_sic4d_sbQ_rel[:] = reaches.sic4d_sbQ_rel
    # cmod1
    c_metroman_Abar[:] = reaches.metroman_abar
    c_metroman_ninf[:] = reaches.metroman_ninf
    c_metroman_p[:] = reaches.metroman_p
    c_metroman_Abar_stdev[:] = reaches.metroman_abar_stdev
    c_metroman_ninf_stdev[:] = reaches.metroman_ninf_stdev
    c_metroman_p_stdev[:] = reaches.metroman_p_stdev
    c_metroman_ninf_p_cor[:] = reaches.metroman_ninf_p_cor
    c_metroman_ninf_Abar_cor[:] = reaches.metroman_ninf_abar_cor
    c_metroman_p_Abar_cor[:] = reaches.metroman_p_abar_cor
    c_metroman_sbQ_rel[:] = reaches.metroman_sbQ_rel
    # cmod2
    c_bam_Abar[:] = reaches.bam_abar
    c_bam_n[:] = reaches.bam_n
    c_bam_sbQ_rel[:] = reaches.bam_sbQ_rel
    # cmod3
    c_hivdi_Abar[:] = reaches.hivdi_abar
    c_hivdi_alpha[:] = reaches.hivdi_alpha
    c_hivdi_beta[:] = reaches.hivdi_beta
    c_hivdi_sbQ_rel[:] = reaches.hivdi_sbQ_rel
    # cmod4
    c_momma_B[:] = reaches.momma_b
    c_momma_H[:] = reaches.momma_h
    c_momma_Save[:] = reaches.momma_save
    c_momma_sbQ_rel[:] = reaches.momma_sbQ_rel
    # cmod5
    c_sads_Abar[:] = reaches.sads_abar
    c_sads_n[:] = reaches.sads_n
    c_sads_sbQ_rel[:] = reaches.sads_sbQ_rel
    # cmod6
    c_sic4d_Abar[:] = reaches.sic4d_abar
    c_sic4d_n[:] = reaches.sic4d_n
    c_sic4d_sbQ_rel[:] = reaches.sic4d_sbQ_rel

    root_grp.close()

    end = time.time()

    print("Ended Saving Main NetCDF in: ", str(np.round((end-start)/60, 2)), " min")

    return outfile

###############################################################################

def add_fill_vars(reaches):
    reaches.h_break = np.full((4,len(reaches.id)), -9999.0)
    reaches.w_break = np.full((4,len(reaches.id)), -9999.0)
    reaches.hw_covariance = np.repeat(-9999., len(reaches.id))
    reaches.h_err_stdev = np.repeat(-9999., len(reaches.id))
    reaches.w_err_stdev = np.repeat(-9999., len(reaches.id))
    reaches.h_w_nobs = np.repeat(-9999., len(reaches.id))
    reaches.fit_coeffs = np.zeros((2, 3, len(reaches.id)))
    reaches.fit_coeffs[np.where(reaches.fit_coeffs == 0)] = -9999.0
    reaches.med_flow_area = np.repeat(-9999., len(reaches.id))
    #MetroMan
    reaches.metroman_ninf = np.repeat(-9999, len(reaches.id))
    reaches.metroman_p = np.repeat(-9999, len(reaches.id))
    reaches.metroman_abar = np.repeat(-9999, len(reaches.id))
    reaches.metroman_abar_stdev = np.repeat(-9999, len(reaches.id))
    reaches.metroman_ninf_stdev = np.repeat(-9999, len(reaches.id))
    reaches.metroman_p_stdev = np.repeat(-9999, len(reaches.id))
    reaches.metroman_ninf_p_cor = np.repeat(-9999, len(reaches.id))
    reaches.metroman_ninf_abar_cor = np.repeat(-9999, len(reaches.id))
    reaches.metroman_p_abar_cor = np.repeat(-9999, len(reaches.id))
    reaches.metroman_sbQ_rel = np.repeat(-9999, len(reaches.id))
    #HiDVI
    reaches.hivdi_abar = np.repeat(-9999, len(reaches.id))
    reaches.hivdi_alpha = np.repeat(-9999, len(reaches.id))
    reaches.hivdi_beta = np.repeat(-9999, len(reaches.id))
    reaches.hivdi_sbQ_rel = np.repeat(-9999, len(reaches.id))
    #MOMMA
    reaches.momma_b = np.repeat(-9999, len(reaches.id))
    reaches.momma_h = np.repeat(-9999, len(reaches.id))
    reaches.momma_save = np.repeat(-9999, len(reaches.id))
    reaches.momma_sbQ_rel = np.repeat(-9999, len(reaches.id))
    #SADS
    reaches.sads_abar = np.repeat(-9999, len(reaches.id))
    reaches.sads_n = np.repeat(-9999, len(reaches.id))
    reaches.sads_sbQ_rel = np.repeat(-9999, len(reaches.id))
    #BAM
    reaches.bam_abar = np.repeat(-9999, len(reaches.id))
    reaches.bam_n = np.repeat(-9999, len(reaches.id))
    reaches.bam_sbQ_rel = np.repeat(-9999, len(reaches.id))
    #SIC4DVar
    reaches.sic4d_abar = np.repeat(-9999, len(reaches.id))
    reaches.sic4d_n = np.repeat(-9999, len(reaches.id))
    reaches.sic4d_sbQ_rel = np.repeat(-9999, len(reaches.id))

###############################################################################
###############################################################################
###############################################################################
start_all = time.time()

write_custom = True
version = 'v15'
region = 'NA'
sword_dir = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'\
    +version+'/netcdf/before_edits/'+region.lower()+'_sword_'+version+'.nc'
sub_outdir = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'  
fn_merge = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Merged_Data/v12/NA_Merge_v12.nc'
rch_fn = '/Users/ealteanau/Documents/SWORD_Dev/update_requests/v14/basin822810_edits.xlsx'

#read in data. 
new_ids = pd.read_excel(rch_fn)
centerlines, nodes, reaches = read_data(sword_dir)

#create new attribute at node level with updated rch ids. 
nodes.old_rch_id = np.copy(nodes.reach_id)
for ind in list(range(len(new_ids.node_id))): 
    n = np.where(nodes.id == new_ids.node_id[ind])[0]
    nodes.reach_id[n] = new_ids.new_rch_id[ind]

#update node and centerline reach ids.
centerlines.ice_flag = calc_cl_iceflag(reaches, centerlines)
# ice_fn = outfile = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'\
#     'SWOT_Coverage_Ice/v14/netcdf/na_centerline_iceflag.nc'
# ice = nc.Dataset(ice_fn)
# centerlines.ice_flag = ice.groups['centerlines'].variables['iceflag'][:]

print('~~~~~~~~~~~~~ UPDATING REACH BOUNDARIES ~~~~~~~~~~~~~~')
update_rch_ids(centerlines, nodes, new_ids)

### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###

print('~~~~~~~~~ REDOING RCH ATTRIBUTES IN SUB-BASIN ~~~~~~~~') 
print('Reading in merged data')
cl_level6 = np.array([str(r)[0:6] for r in centerlines.reach_id[0,:]])
cl_l6 = np.where(cl_level6 == str(new_ids.basin[0]))[0]
nd_level6 = np.array([str(r)[0:6] for r in nodes.reach_id])
nd_l6 = np.where(nd_level6 == str(new_ids.basin[0]))[0]

merge = read_merge_netcdf_subset(fn_merge, new_ids.basin[0], 6)

cl_pts = np.vstack((centerlines.x[cl_l6], centerlines.y[cl_l6])).T
merge_pts = np.vstack((merge.lon, merge.lat)).T
kdt = sp.cKDTree(cl_pts)
pt_dist, pt_ind = kdt.query(merge_pts, k = 5, distance_upper_bound = 500)
#add reach ids to merge database 
merge.reach_id = centerlines.reach_id[0,cl_l6[pt_ind[:,0]]] ### reach ids are odd....
merge.node_id = centerlines.node_id[0,cl_l6[pt_ind[:,0]]]

print('Updating Reach Indexes')
# Updating reach indexes and type.
merge.rch_ind6 = centerlines.cl_id[cl_l6[pt_ind[:,0]]]
# merge.rch_ind6, merge.rch_eps6 = update_rch_indexes(merge, merge.reach_id) 

print('Calculating Reach Length')
# Updating reach flow distance.
merge.rch_dist6 = calc_segDist(merge.lon, merge.lat, merge.reach_id, merge.facc, merge.rch_ind6)
merge.rch_len6 = calc_rchLen(merge)

print('Creating Reach Attributes')
# Defining reach attributes.
subreaches = Object()
subreaches.id, subreaches.x, subreaches.y, subreaches.x_max,\
    subreaches.x_min, subreaches.y_max, subreaches.y_min, subreaches.len,\
    subreaches.wse, subreaches.wse_var, subreaches.wth, subreaches.wth_var,\
    subreaches.nchan_max, subreaches.nchan_mod, subreaches.rch_n_nodes,\
    subreaches.slope, subreaches.grod, subreaches.lakeflag,\
    subreaches.facc, subreaches.grod_fid,\
    subreaches.hfalls_fid, subreaches.lake_id = reach_attributes(merge)

print('Finding All Neighbors')
subreaches.neighbors = neigboring_reaches(merge.lon, merge.lat, merge.reach_id, subreaches.id)

print('Calculating Attributes from Nodes')
subreaches.dist_out, subreaches.river_name,\
    subreaches.max_wth, subreaches.edit_flag = calc_attributes_from_nodes(subreaches, nodes)

print('Calculating Attributes from Centerlines')
subreaches.iceflag = calc_cls_iceflag(subreaches, centerlines)

print('Defining Local Topology')
# Defining intial topology.
subreaches.n_rch_up, subreaches.n_rch_down, subreaches.rch_id_up,\
    subreaches.rch_id_down = local_topology(merge, subreaches)
# Filtering downstream neighbors
subreaches.rch_id_up_filt, subreaches.n_rch_up_filt,\
    subreaches.rch_id_down_filt, subreaches.n_rch_down_filt = filter_neighbors(subreaches)

print('Calculating SWOT Coverage')
# Calculating swot coverage.
subreaches.coverage, subreaches.orbits, subreaches.max_obs,\
    subreaches.median_obs, subreaches.mean_obs = swot_obs_percentage(merge, subreaches)

print('Finalizing Reach Attributes')
# start_cnt = np.max(centerlines.cl_id)+1
# cls_cl_id, rch_cl_id, node_cl_id = centerline_ids(subreaches, nodes, merge, start_cnt)
__, rch_cl_id, node_cl_id = update_cl_ids(subreaches, nodes.id[nd_l6], merge)
subreaches.cl_id = rch_cl_id.T
# Replace node and centerline cl_id values. 
nodes.cl_id[:,nd_l6] = node_cl_id.T
# centerlines.cl_id[cl_l6[pt_ind[:,0]]] = cls_cl_id

# Reformatting attributes
subreaches.rch_id_up_filt = subreaches.rch_id_up_filt[:,0:4]
subreaches.rch_id_down_filt = subreaches.rch_id_down_filt[:,0:4]
subreaches.n_rch_up_filt[np.where(subreaches.n_rch_up_filt > 4)] = 4
subreaches.n_rch_down_filt[np.where(subreaches.n_rch_down_filt > 4)] = 4
subreaches.nchan_mod[np.where(subreaches.nchan_mod == 0)] = 1
subreaches.nchan_max[np.where(subreaches.nchan_max == 0)] = 1
subreaches.rch_id_up_filt = subreaches.rch_id_up_filt.T
subreaches.rch_id_down_filt = subreaches.rch_id_down_filt.T
subreaches.orbits = subreaches.orbits.T
subreaches.coverage = subreaches.coverage.T
subreaches.low_slope = np.zeros(len(subreaches.id))

### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###

#delete existing reaches in level6 basin.
level6 = np.array([str(r)[0:6] for r in reaches.id])
l6 = np.where(level6 == str(new_ids.basin[0]))[0]
old_reach_num = len(reaches.id)
delete_rchs(reaches, l6)

#append new reaches
append_data(reaches, subreaches)

#add fill variables for reaches
add_fill_vars(reaches)

### OPTIONAL (only if no other updates are made)  
#redo centerline ids for nodes and reaches. 
# cl_nodes_id = format_cl_node_ids(nodes, centerlines, verbose = True)
# cl_rch_id = format_cl_rch_ids(reaches, centerlines, verbose = True)
# centerlines.reach_id = np.insert(cl_rch_id, 0, centerlines.reach_id, axis = 0)
# centerlines.reach_id = centerlines.reach_id[0:4,:]
# centerlines.node_id =  np.insert(cl_nodes_id, 0, centerlines.node_id, axis = 0)
# centerlines.node_id = centerlines.node_id[0:4,:]

# write new netcdf.
# write_database_nc(centerlines, reaches, nodes, region, sword_dir)

######## !!!!!!!!!!!!!!!! SAVING SUBSETS FOR LATER UPDATES !!!!!!!!!!!!!!!!!!!! #########
if write_custom == True:
    print('Saving Custom Basin')
    sub_centerlines, sub_nodes, sub_reaches = subset_data(centerlines, nodes, reaches, subreaches.id)
    add_fill_vars(sub_reaches)
    #redo centerline ids for nodes and reaches. (only if no other updates are made)  
    cl_nodes_id = format_cl_node_ids(sub_nodes, sub_centerlines, verbose = True)
    cl_rch_id = format_cl_rch_ids(sub_reaches, sub_centerlines, verbose = True)
    sub_centerlines.reach_id = np.insert(cl_rch_id, 0, sub_centerlines.reach_id[0,:], axis = 0)
    sub_centerlines.reach_id = sub_centerlines.reach_id[0:4,:]
    sub_centerlines.node_id =  np.insert(cl_nodes_id, 0, sub_centerlines.node_id[0,:], axis = 0)
    sub_centerlines.node_id = sub_centerlines.node_id[0:4,:]
    sub_fn = sub_outdir+region.lower()+'_'+str(new_ids.basin[0])+'_calval_'+version+'.nc'
    write_database_nc(sub_centerlines, sub_reaches, sub_nodes, region, sub_fn)
######## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #########

end_all = time.time()
print('Done in: ' + str(np.round((end_all-start_all)/60, 2)) + ' min')