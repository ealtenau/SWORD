from __future__ import division
import numpy as np
from scipy import spatial as sp
import netCDF4 as nc
from pyproj import Proj
import time
import geopy.distance


#########################################################################################

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

#########################################################################################

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