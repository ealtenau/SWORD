from __future__ import division
import numpy as np
from scipy import spatial as sp
import netCDF4 as nc
from pyproj import Proj
import time
import geopy.distance

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

def update_node_order(subcls, subnodes, subreaches, reach):
    
    nodes_rch = np.where(subnodes.reach_id == reach)[0]
    rch = np.where(subreaches.id == reach)[0]
    
    if '2' in subreaches.edit_flag[rch][0].split(','):
        edit_val = subreaches.edit_flag[rch][0]
    elif 'N' in subreaches.edit_flag[rch][0].split(','):
        edit_val = '2'
    else:
        edit_val = subreaches.edit_flag[rch][0] + ',2'
                    
    #create new variables
    node_ids = subnodes.id[nodes_rch] 
    dist_out = subnodes.dist_out[nodes_rch]
    new_node_ids = node_ids[::-1]
    new_dist_out = dist_out[::-1]  
    #update variables in netcdf
    subnodes.id[nodes_rch] = new_node_ids
    subnodes.dist_out[nodes_rch] = new_dist_out
    subnodes.edit_flag[nodes_rch] = np.repeat(edit_val, len(nodes_rch))
    subreaches.edit_flag[rch] = edit_val

    for n in list(range(len(node_ids))):
        cl_n1 = np.where(subcls.node_id[0,:] == node_ids[n])[0]
        cl_n2 = np.where(subcls.node_id[1,:] == node_ids[n])[0]
        cl_n3 = np.where(subcls.node_id[2,:] == node_ids[n])[0]
        cl_n4 = np.where(subcls.node_id[3,:] == node_ids[n])[0]
        if len(cl_n1) > 0:
            subcls.node_id[0,cl_n1] = new_node_ids[n]
        if len(cl_n2) > 0:
            subcls.node_id[1,cl_n2] = new_node_ids[n]
        if len(cl_n3) > 0:
            subcls.node_id[2,cl_n3] = new_node_ids[n]
        if len(cl_n4) > 0:
            subcls.node_id[3,cl_n4] = new_node_ids[n]

###############################################################################

def local_topology(subcls, subnodes, subreaches, subset = True):

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
    if subset == True:
        rch_list = subreaches.id[np.where(subreaches.edit_flag != 'NaN')[0]]
        n_rch_up = np.copy(subreaches.n_rch_up)
        n_rch_down = np.copy(subreaches.n_rch_down)
        rch_id_up = np.zeros((len(subreaches.id),10))
        rch_id_down = np.zeros((len(subreaches.id),10))
        rch_id_up[:,0:4] = subreaches.rch_id_up.T
        rch_id_down[:,0:4] = subreaches.rch_id_down.T
    else:
        rch_list = np.copy(subreaches.id)
        n_rch_up = np.zeros(len(subreaches.id))
        n_rch_down = np.zeros(len(subreaches.id))
        rch_id_up = np.zeros((len(subreaches.id),10))
        rch_id_down = np.zeros((len(subreaches.id),10))

    for r in list(range(len(rch_list))):
        print(r)
        #identifying the current reach neighbors.
        ind = np.where(subreaches.id == rch_list[r])
        #not sure why the double zero indexes are needed after neighbors...
        nghs = subreaches.neighbors[ind,np.where(subreaches.neighbors[ind,:][0][0] > 0)[0]][0]

        if len(nghs) == 0:
            continue

        else:
            #determining important reach attributes.
            rch_cls = np.where(subcls.reach_id[0,:] == subreaches.id[ind])[0]
            rch = np.where(subreaches.id == subreaches.id[ind])[0]
            rch_dist = subreaches.dist_out[rch]
            rch_wse = subreaches.wse[rch]
            rch_facc = subreaches.facc[rch]

            #identifying reach endpoints based on index values.
            end1 = rch_cls[np.where(subcls.cl_id[rch_cls] == np.min(subcls.cl_id[rch_cls]))[0][0]]
            end2 = rch_cls[np.where(subcls.cl_id[rch_cls] == np.max(subcls.cl_id[rch_cls]))[0][0]]

            #finding the coordination of the reach endpoints.
            ep1_lon, ep1_lat = subcls.x[end1], subcls.y[end1]
            ep2_lon, ep2_lat = subcls.x[end2], subcls.y[end2]
            node1 = subcls.node_id[0,end1]
            node2 = subcls.node_id[0,end2]

            #looping through each reach neighbor and determining if it is
            #upstream or downstream based on flow accumulation or wse.
            ep1_nghs = np.zeros([len(nghs), 4])
            ep2_nghs = np.zeros([len(nghs), 4])
            for idx in list(range(len(nghs))):
                rch2 = np.where(subreaches.id == nghs[idx])[0]
                rch2_cls = np.where(subcls.reach_id[0,:] == nghs[idx])[0]
                ngh_dist = subreaches.dist_out[rch2]
                ngh_wse = subreaches.wse[rch2]
                ngh_facc = subreaches.facc[rch2]

                # finding high-res centerline endpoints for neighboring reach.
                ngh_end1 = rch2_cls[np.where(subcls.cl_id[rch2_cls] == np.min(subcls.cl_id[rch2_cls]))[0][0]]
                ngh_end2 = rch2_cls[np.where(subcls.cl_id[rch2_cls] == np.max(subcls.cl_id[rch2_cls]))[0][0]]
                ngh_end1_lon, ngh_end1_lat = subcls.x[ngh_end1], subcls.y[ngh_end1]
                ngh_end2_lon, ngh_end2_lat = subcls.x[ngh_end2], subcls.y[ngh_end2]

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
            
            ################################# NODE CHECK ################################
            if node1 != node2:
                if len(ep1_nghs[:,0]) > 0 and len(ep2_nghs[:,0]) > 0:
                    ngh1_wse = np.min(ep1_nghs[:,2])
                    ngh1_facc = np.max(ep1_nghs[:,1])
                    ngh2_wse = np.min(ep2_nghs[:,2])
                    ngh2_facc = np.max(ep2_nghs[:,1])
                    if ngh1_facc < ngh2_facc:
                        check = node1 > node2
                        if check == False:
                            update_node_order(subcls, subnodes, subreaches, subreaches.id[ind])
                            print(subreaches.id[ind], 'nodes switched')
                    elif ngh1_facc > ngh2_facc:
                        check = node1 < node2
                        if check == False:
                            update_node_order(subcls, subnodes, subreaches, subreaches.id[ind])
                            print(subreaches.id[ind], 'nodes switched')
                    else:
                        if ngh1_wse > ngh2_wse:
                            check = node1 > node2
                            if check == False:
                                update_node_order(subcls, subnodes, subreaches, subreaches.id[ind])
                                print(subreaches.id[ind], 'nodes switched')
                        else:
                            check = node1 < node2
                            if check == False:
                                update_node_order(subcls, subnodes, subreaches, subreaches.id[ind])
                                print(subreaches.id[ind], 'nodes switched')

                if len(ep1_nghs[:,0]) > 0 and len(ep2_nghs[:,0]) == 0:
                    ngh1_wse = np.min(ep1_nghs[:,2])
                    ngh1_facc = np.max(ep1_nghs[:,1])
                    if ngh1_facc < rch_facc:
                        check = node1 > node2
                        if check == False:
                            update_node_order(subcls, subnodes, subreaches, subreaches.id[ind])
                            print(subreaches.id[ind], 'nodes switched')
                    elif ngh1_facc > rch_facc:
                        check = node1 < node2
                        if check == False:
                            update_node_order(subcls, subnodes, subreaches, subreaches.id[ind])
                            print(subreaches.id[ind], 'nodes switched')
                    else:
                        if ngh1_wse > rch_wse:
                            check = node1 > node2
                            if check == False:
                                update_node_order(subcls, subnodes, subreaches, subreaches.id[ind])
                                print(subreaches.id[ind], 'nodes switched')
                        else:
                            check = node1 < node2
                            if check == False:
                                update_node_order(subcls, subnodes, subreaches, subreaches.id[ind])
                                print(subreaches.id[ind], 'nodes switched')
                
                if len(ep1_nghs[:,0]) == 0 and len(ep2_nghs[:,0]) > 0:
                    ngh2_wse = np.min(ep2_nghs[:,2])
                    ngh2_facc = np.max(ep2_nghs[:,1])
                    if ngh2_facc < rch_facc:
                        check = node2 > node1
                        if check == False:
                            update_node_order(subcls, subnodes, subreaches, subreaches.id[ind])
                            print(subreaches.id[ind], 'nodes switched')
                    elif ngh2_facc > rch_facc:
                        check = node2 < node1
                        if check == False:
                            update_node_order(subcls, subnodes, subreaches, subreaches.id[ind])
                            print(subreaches.id[ind], 'nodes switched')
                    else:
                        if ngh2_wse > rch_wse:
                            check = node2 > node1
                            if check == False:
                                update_node_order(subcls, subnodes, subreaches, subreaches.id[ind])
                                print(subreaches.id[ind], 'nodes switched')
                        else:
                            check = node2 < node1
                            if check == False:
                                update_node_order(subcls, subnodes, subreaches, subreaches.id[ind])
                                print(subreaches.id[ind], 'nodes switched')
                
            ############################### NODE CHECK  END ##############################

            #change dist to last option for updates...
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
###############################################################################
###############################################################################

start_all = time.time()

version = 'v14'
region='NA'
sword_dir = '/Users/ealteanau/Documents/SWORD_Dev/outputs/'+version+'/netcdf/'
sword = nc.Dataset(sword_dir+region.lower()+'_sword_'+version+'.nc', 'r+')

#read in netcdf data. 
centerlines, nodes, reaches = read_data(sword_dir+region.lower()+'_sword_'+version+'.nc')

### need to edit based on level 2 to speed up the process. 
l2_rch = np.array([np.int(np.str(ind)[0:2]) for ind in reaches.id])
l2_cl = np.array([np.int(np.str(ind)[0:2]) for ind in centerlines.reach_id[0,:]])
uniq_level2 = np.unique(l2_rch)
all_nghs = np.zeros((len(reaches.id),20))
for ind in list(range(len(uniq_level2))):
    start = time.time()
    print('STARTING BASIN: ' + str(uniq_level2[ind]))
    cl_ind = np.where(l2_cl == uniq_level2[ind])[0]
    rch_ind = np.where(l2_rch == uniq_level2[ind])[0]
    sub_nghs = neigboring_reaches(
        centerlines.x[cl_ind], 
        centerlines.y[cl_ind], 
        centerlines.reach_id[0,cl_ind], 
        reaches.id[rch_ind])
    end = time.time()
    all_nghs[rch_ind, :] = sub_nghs
    print('Time to Finish Basin ' + str(uniq_level2[ind]) + ': ' + str(np.round((end-start)/60, 2)) + ' min')

#create neighbor object
reaches.neighbors = all_nghs

### topo function
reaches.n_rch_up, reaches.n_rch_down, \
    reaches.rch_id_up, reaches.rch_id_down = local_topology(centerlines, nodes, reaches, subset = True)

#filter neighbors
reaches.rch_id_up_filt, reaches.n_rch_up_filt, \
    reaches.rch_id_down_filt, reaches.n_rch_down_filt = filter_neighbors(reaches)
reaches.rch_id_up_filt = reaches.rch_id_up_filt[:,0:4]
reaches.rch_id_down_filt = reaches.rch_id_down_filt[:,0:4]

#transpose for netcdf
reaches.rch_id_up_filt = reaches.rch_id_up_filt.T
reaches.rch_id_down_filt = reaches.rch_id_down_filt.T

#check node directions - new function...

#update netcdf
# sword.groups['reaches'].variables['n_rch_up'][:] = reaches.n_rch_up_filt
# sword.groups['reaches'].variables['n_rch_down'][:] = reaches.n_rch_down_filt
# sword.groups['reaches'].variables['rch_id_up'][:] = reaches.rch_id_up_filt
# sword.groups['reaches'].variables['rch_id_dn'][:] = reaches.rch_id_down_filt
# sword.groups['nodes'].variables['node_id'][:] = nodes.id
# sword.groups['nodes'].variables['dist_out'][:] = nodes.dist_out
# sword.groups['centerlines'].variables['node_id'][:] = centerlines.node_id[:]
sword.close()

end_all = time.time()
print('Time to Finish Topology: ' + str(np.round((end_all-start_all)/60, 2)) + ' min')
