"""
Attaching Path Variables to SWORD (path_vars_to_main_nc.py)
==============================================================

Script for adding the path variables to the SWORD netCDF file.

The script is run at a Pfafstetter Level 2 basin scale.
Command line arguments required are the two-letter
region identifier (i.e. NA), SWORD version (i.e. v17),
and Pfafstetter Level 2 basin (i.e. 74).

Execution example (terminal):
    python path/to/path_vars_to_main_nc.py NA v17 74

"""

import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import netCDF4 as nc
import time
import argparse

###############################################################################
###########################  FUNCTIONS  #######################################
###############################################################################

class Object(object):
    """
    FUNCTION:
        Creates empty class object to assign attributes to.
    """
    pass 

###############################################################################

def read_data(filename):
    """
    Reads SWORD data from the netCDF file and stores each spatial 
    dimension in a separate class object with associated attributes. 

    Parmeters
    ---------
    filename: str
        The directory to the SWORD netCDF file. 
    
    Returns
    -------
    centerlines: obj 
        Object containing attributes associated with the SWORD 
        centerlines. 
        
        Attributes: type [dimension]
        ----------------------------
        cl_id: numpy.array() [number of points]
            Unique ID associated with each centerline point. 
        x: numpy.array() [number of points]
            Longitude (WGS 84, EPSG:4326). 
        y: numpy.array() [number of points]
            Latitude (WGS 84, EPSG:4326).
        reach_id: numpy.array() [4, number of points]
            Reach ID associated with a centerline point. The first row 
            contains the reach IDs associated with each centerline 
            point. Rows 2-4 are reach IDs of neighboring reaches.  
        node_id: numpy.array() [4, number of points]
            Node ID associated with a centerline point. The first row 
            contains the node IDs associated with each centerline 
            point. Rows 2-4 are node IDs of neighboring nodes.

    nodes: obj
        Object containing attributes associated with the SWORD 
        nodes. 

        Attributes: type [dimension]
        ----------------------------
        id: numpy.array() [number of nodes]
            Unique Node ID. 
        cl_ids: numpy.array() [2, number of nodes]
            Minimum (row 1) and maximum (row 2) centerline 
            point ids associated with each node.
        x: numpy.array() [number of nodes]
            Longitude (WGS 84, EPSG:4326). 
        y: numpy.array() [number of nodes]
            Latitude (WGS 84, EPSG:4326). 
        len: numpy.array() [number of nodes]
            Node length (meters). 
        wse: numpy.array() [number of nodes]
            Water surface elevation (meters). 
        wse_var: numpy.array() [number of nodes]
            Water surface elevation variance (squared meters).
        wth: numpy.array() [number of nodes]
            Width (meters).
        wth_var: numpy.array() [number of nodes]
            Width variance (squared meters).
        grod: numpy.array() [number of nodes]
            Type of obstruction for each node based on GROD and
            HydroFALLS databases. Obstr_type values: 
            0 - No Dam, 
            1 - Dam, 
            2 - Lock, 
            3 - Low Permeable Dam, 
            4 - Waterfall.
        grod_fid: numpy.array() [number of nodes]
            GROD database ID. 
        hfalls_fid: numpy.array() [number of nodes]
            HydroFALLS database ID. 
        nchan_max: numpy.array() [number of nodes]
            Maximum number of channels for each node.
        nchan_mod: numpy.array() [number of nodes]
            Mode of the number of channels for each node.
        dist_out: numpy.array() [number of nodes]
            Distance from the river outlet (meters).
        reach_id: numpy.array() [number of nodes]
            Reach ID the node is associated with.
        facc: numpy.array() [number of nodes]
            Flow accumulation (squared kilometers).
        lakeflag: numpy.array() [number of nodes]
            GRWL water body identifier for each node: 
            0 - river, 
            1 - lake/reservoir, 
            2 - canal , 
            3 - tidally influenced river.
        wth_coef: numpy.array() [number of nodes]
            Coefficient that is multiplied by the width to
            inform the search window for SWOT data.
        ext_dist_coef: numpy.array() [number of nodes]
            Coefficient that informs the maximum search
            window for SWOT data.
        max_wth: numpy.array() [number of nodes]
            Maximum width value across the channel that
            includes any island and bar areas (meters).
        meand_len: numpy.array() [number of nodes]
            Length of the meander that a node belongs to, 
            measured from beginning of the meander to 
            its end (meters).
        river_name: numpy.array() [number of nodes]
            All river names associated with a node. If there are 
            multiple names they are listed in alphabetical order 
            and separated by a semicolon.
        manual_add: numpy.array() [number of nodes]
            Binary flag indicating whether the node was manually added. 
            0 - Not manually added. 
            1 - Manually added. 
        sinuosity: numpy.array() [number of nodes]
            The total reach length the node belongs to divided by the 
            Euclidean distance between the reach end points (meters).
        edit_flag: numpy.array() [number of nodes]
            Numerical flag indicating the type of update applied to
            SWORD nodes from the previous version. Flag descriptions:
            1 - reach type change,
            2 - node order change,
            3 - reach neighbor change,
            41 - flow accumulation update,
            42 - elevation update,
            43 - width update,
            44 - slope update,
            45 - river name update,
            5 - reach id change,
            6 - reach boundary change,
            7 - reach/node addition
            Multiple updates will be separated by a comma (i.e. "41,2")
        trib_flag: numpy.array() [number of nodes]
            Binary flag indicating if a large tributary not represented in
            SWORD is entering a node. 
            0 - no tributary, 
            1 - tributary.
        path_freq: numpy.array() [number of nodes]
            The number of times a node is traveled along get to any 
            given headwater point.
        path_order: numpy.array() [number of nodes]
            Unique values representing continuous paths from the
            river outlet to the headwaters ordered from the longest
            path (1) to the shortest path (N).
        path_segs: numpy.array() [number of nodes]
            Unique values indicating continuous river segments
            between river junctions.
        strm_order: numpy.array() [number of nodes]
            Stream order based on the log scale of the path frequency.
            Stream order is calculated for the main network only (see
            “main_side” description).
        main_side: numpy.array() [number of nodes]
            Value indicating whether a node is on the:
            0 - main network
            1 - side network 
            2 - secondary outlet
        end_rch: numpy.array() [number of nodes]
            Value indicating whether a node is:
            0 - main stem 
            1 - headwater
            2 - outlet
            3 - junction
        network: numpy.array() [number of nodes]
            Unique value for each connected river network.
        add_flag: numpy.array() [number of nodes]
            Binary flag indicating if the node was added
            to the current SWORD version based on the 
            MERIT Hydro Vector database. 
            0 - not added, 
            1 - added. 

    reaches: obj
        Object containing attributes associated with the SWORD 
        reaches. 

        Attributes: type [dimension]
        ----------------------------
        id: numpy.array() [number of reaches]
            Unique Reach ID. 
        cl_ids: numpy.array() [2, number of reaches]
            Minimum (row 1) and maximum (row 2) centerline 
            point ids associated with each reach.
        x: numpy.array() [number of reaches]
            Longitude (WGS 84, EPSG:4326). 
        x_min: numpy.array() [number of reaches]
            Minimum longitude of a reach (WGS 84, EPSG:4326). 
        x_max: numpy.array() [number of reaches]
            Maximum longitude of a reach (WGS 84, EPSG:4326). 
        y: numpy.array() [number of reaches]
            Latitude (WGS 84, EPSG:4326). 
        y_min: numpy.array() [number of reaches]
            Minimum latitude of a reach (WGS 84, EPSG:4326).
        y_max: numpy.array() [number of reaches]
            Maximum latitude of a reach (WGS 84, EPSG:4326).
        len: numpy.array() [number of reaches]
            Reach length (meters). 
        wse: numpy.array() [number of reaches]
            Water surface elevation (meters). 
        wse_var: numpy.array() [number of reaches]
            Water surface elevation variance (squared meters).
        wth: numpy.array() [number of reaches]
            Width (meters).
        wth_var: numpy.array() [number of reaches]
            Width variance (squared meters).
        grod: numpy.array() [number of reaches]
            Type of obstruction for each reach based on GROD and
            HydroFALLS databases. Obstr_type values: 
            0 - No Dam, 
            1 - Dam, 
            2 - Lock, 
            3 - Low Permeable Dam, 
            4 - Waterfall.
        grod_fid: numpy.array() [number of reaches]
            GROD database ID. 
        hfalls_fid: numpy.array() [number of reaches]
            HydroFALLS database ID. 
        nchan_max: numpy.array() [number of reaches]
            Maximum number of channels for each reach.
        nchan_mod: numpy.array() [number of reaches]
            Mode of the number of channels for each reach.
        dist_out: numpy.array() [number of reaches]
            Distance from the river outlet (meters).
        rch_n_nodes: numpy.array() [number of reaches]
            Number of nodes in a reach.
        slope: [number of reaches]
            Reach slope (meters per kilometer).
        n_rch_up: [number of reaches]
            Number of upstream neighbors.
        n_rch_down: [number of reaches]
            Number of downstream neighbors.
        rch_id_up: [4, number of reaches]
            Reach IDs of upstream neighbors (4 maximum).
        rch_id_down: [4, number of reaches]
            Reach IDs of downstream neighbors (4 maximum).
        max_obs: [number of reaches]
            Maximum number of SWOT passes to intersect each 
            reach during the ~21 day orbit cycle.
        orbits: [75, number of reaches]
            SWOT orbit pass_tile IDs that intersect each reach 
            during the 21 day orbit cycle. One ID per row (75 maximum).
        iceflag: [number of reaches]
        low_slope: [number of reaches]
            Binary flag where a value of 1 indicates the reach 
            slope is too low for effective discharge estimation
            for SWOT discharge algorithms.
        facc: numpy.array() [number of reaches]
            Flow accumulation (squared kilometers).
        lakeflag: numpy.array() [number of reaches]
            GRWL water body identifier for each reach: 
            0 - river, 
            1 - lake/reservoir, 
            2 - canal , 
            3 - tidally influenced river.
        max_wth: numpy.array() [number of reaches]
            Maximum width value across the channel that
            includes any island and bar areas (meters).
        river_name: numpy.array() [number of reaches]
            All river names associated with a reach. If there are 
            multiple names they are listed in alphabetical order 
            and separated by a semicolon.
        edit_flag: numpy.array() [number of reaches]
            Numerical flag indicating the type of update applied to
            SWORD reaches from the previous version. Flag descriptions:
            1 - reach type change,
            2 - node order change,
            3 - reach neighbor change,
            41 - flow accumulation update,
            42 - elevation update,
            43 - width update,
            44 - slope update,
            45 - river name update,
            5 - reach id change,
            6 - reach boundary change,
            7 - reach/node addition
            Multiple updates will be separated by a comma (i.e. "41,2")
        trib_flag: numpy.array() [number of reaches]
            Binary flag indicating if a large tributary not represented in
            SWORD is entering a reach. 
            0 - no tributary, 
            1 - tributary.
        path_freq: numpy.array() [number of reaches]
            The number of times a reach is traveled along get to any 
            given headwater point.
        path_order: numpy.array() [number of reaches]
            Unique values representing continuous paths from the
            river outlet to the headwaters ordered from the longest
            path (1) to the shortest path (N).
        path_segs: numpy.array() [number of reaches]
            Unique values indicating continuous river segments
            between river junctions.
        strm_order: numpy.array() [number of reaches]
            Stream order based on the log scale of the path frequency.
            Stream order is calculated for the main network only (see
            “main_side” description).
        main_side: numpy.array() [number of reaches]
            Value indicating whether a reach is on the:
            0 - main network
            1 - side network 
            2 - secondary outlet
        end_rch: numpy.array() [number of reaches]
            Value indicating whether a reach is:
            0 - main stem 
            1 - headwater
            2 - outlet
            3 - junction
        network: numpy.array() [number of reaches]
            Unique value for each connected river network.
        add_flag: numpy.array() [number of reaches]
            Binary flag indicating if the reach was added
            to the current SWORD version based on the 
            MERIT Hydro Vector database. 
            0 - not added, 
            1 - added.
    
    """

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
    nodes.trib_flag = data.groups['nodes'].variables['trib_flag'][:]
    if 'path_freq' in data.groups['nodes'].variables.keys():
        nodes.path_freq = data.groups['nodes'].variables['path_freq'][:]
        nodes.path_order = data.groups['nodes'].variables['path_order'][:]
        nodes.path_segs = data.groups['nodes'].variables['path_segs'][:]
        nodes.main_side = data.groups['nodes'].variables['main_side'][:]
        nodes.strm_order = data.groups['nodes'].variables['stream_order'][:]
    else:
        nodes.path_freq = np.zeros(len(nodes.id))
        nodes.path_order = np.zeros(len(nodes.id))
        nodes.path_segs = np.zeros(len(nodes.id))
        nodes.main_side = np.zeros(len(nodes.id))
        nodes.strm_order = np.zeros(len(nodes.id))

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
    reaches.trib_flag = data.groups['reaches'].variables['trib_flag'][:]
    if 'path_freq' in data.groups['reaches'].variables.keys():
        reaches.path_freq = data.groups['reaches'].variables['path_freq'][:]
        reaches.path_order = data.groups['reaches'].variables['path_order'][:]
        reaches.path_segs = data.groups['reaches'].variables['path_segs'][:]
        reaches.main_side = data.groups['reaches'].variables['main_side'][:]
        reaches.strm_order = data.groups['reaches'].variables['stream_order'][:]
    else:
        reaches.path_freq = np.zeros(len(reaches.id))
        reaches.path_order = np.zeros(len(reaches.id))
        reaches.path_segs = np.zeros(len(reaches.id))
        reaches.main_side = np.zeros(len(reaches.id))
        reaches.strm_order = np.zeros(len(reaches.id))

    data.close()    

    return centerlines, nodes, reaches
    
###############################################################################

def reorder_cl_ids(path_cl_rch_ids, path_cl_ids, path_cl_dist_out, centerlines):
    """
    Orders SWORD centerline IDs based on the shortest paths 
    direction information.

    Parameters
    ----------
    path_cl_rch_ids: numpy.array()
        Reach IDs associated with the shortest paths.
    path_cl_ids: numpy.array()
        Centerline IDs associated with the shortest paths. 
    path_cl_dist_out: numpy.array()
        Distance from outlet along the shortest paths. 
    centerlines: obj
        Object containing lcation and attribute information
        along the high-resolution centerline.
        
    Returns
    -------
    cl_id_new: numpy.array()
        Updated centerline IDs. 

    """

    unq_rchs = np.unique(path_cl_rch_ids)
    cl_id_new = np.copy(centerlines.cl_id)
    cl_rch_ids = np.copy(centerlines.reach_id[0,:])
    for ind in list(range(len(unq_rchs))):
        rch = np.where(path_cl_rch_ids == unq_rchs[ind])[0]
        rch_main = np.where(cl_rch_ids == unq_rchs[ind])[0]

        ### Did not account for nan values
        # mn = np.where(path_cl_ids[rch] == min(path_cl_ids[rch]))[0]
        # mx = np.where(path_cl_ids[rch] == max(path_cl_ids[rch]))[0]
        # mn_dist = path_cl_dist_out[rch[mn]]
        # mx_dist = path_cl_dist_out[rch[mx]]
        
        keep = np.where(path_cl_dist_out[rch] >= 0)[0]
        mn = np.where(path_cl_dist_out[rch[keep]] == min(path_cl_dist_out[rch[keep]]))[0][0]
        mx = np.where(path_cl_dist_out[rch[keep]] == max(path_cl_dist_out[rch[keep]]))[0][0]
        mn_id = path_cl_ids[rch[keep[mn]]]
        mx_id = path_cl_ids[rch[keep[mx]]]

        if mn_id > mx_id:
            sort_inds = np.argsort(cl_id_new[rch_main])
            cl_id_new[rch_main[sort_inds]] = cl_id_new[rch_main[sort_inds]][::-1]

    return cl_id_new

###############################################################################

def update_headwaters_outlets(path_order, path_cl_dist_out, path_cl_ids, 
                              con_cl_ids, con_end_ids):
    """
    Adds path variable information to the SWORD reach and 
    node objects.

    Parameters
    ----------
    path_cl_ids: numpy.array()
        Centerline IDs associated with the shortest paths. 
    path_order: numpy.array()
        Path order: Unique values representing continuous paths from the
        river outlet to the headwaters ordered from the longest path (1) 
        to the shortest path (N).
    path_cl_dist_out: numpy.array()
        Distance from outlet along the shortest paths. 
    con_cl_ids: numpy.array()
        SWORD centerline IDs. 
    con_end_ids: numpy.array()
        Binary flag indicating whether a SWORD point is a headwater
        or outlet point.  
        
    Returns
    -------
    None.
        
    """
    
    unq_paths = np.unique(path_order)
    unq_paths = unq_paths[unq_paths>0]
    for ind in list(range(len(unq_paths))):
        path = np.where(path_order == unq_paths[ind])[0]
        path_min_dist = min(path_cl_dist_out[path])
        if path_min_dist < 0:
            path_min_dist = 0
        path_max_dist = max(path_cl_dist_out[path])
        
        clids = path_cl_ids[path]
        con_inds = np.where(np.in1d(con_cl_ids, clids)==True)[0]
        hw = np.where(con_end_ids[con_inds]==1)[0]
        out = np.where(con_end_ids[con_inds]==2)[0]
        if len(hw) > 0:
            for h in list(range(len(hw))):
                path_rch = path_cl_rch_ids[np.where(path_cl_ids == con_cl_ids[con_inds[hw[h]]])]
                path_dist = path_cl_dist_out[np.where(path_cl_rch_ids == path_rch)]
                if path_min_dist in path_dist:
                    # print(ind)
                    con_end_ids[con_inds[hw[h]]] = 2
        if len(out) > 0:
            for o in list(range(len(out))):
                path_rch = path_cl_rch_ids[np.where(path_cl_ids == con_cl_ids[con_inds[out[o]]])]
                path_dist = path_cl_dist_out[np.where(path_cl_rch_ids == path_rch)]
                if path_max_dist in path_dist:
                    # print(ind)
                    con_end_ids[con_inds[out[o]]] = 1

    side_clids = path_cl_ids[np.where(path_order == 0)[0]]
    side_cl_ind = np.where(np.in1d(con_cl_ids, side_clids) == True)[0]
    side_update = np.where(con_end_ids[side_cl_ind] != 3)[0]
    con_end_ids[side_cl_ind[side_update]] = 0

###############################################################################
                    
def add_rch_node_path_vars(reaches, nodes, path_cl_rch_ids, path_cl_dist_out, 
                      path_freq, path_order, path_main_side, path_cl_node_ids):
    """
    Adds path variable information to the SWORD reach and 
    node objects.

    Parameters
    ----------
    reaches: obj
        Object containing lcation and attribute information for
        each reach.
    nodes: obj
        Object containing lcation and attribute information for
        each node.
    path_cl_rch_ids: numpy.array()
        Reach IDs associated with the shortest paths. 
    path_cl_dist_out: numpy.array()
        Distance from outlet along the shortest paths. 
    path_freq: numpy.array()
        Path frequency: The number of times a point is traveled along get to any 
        given headwater point.
    path_order: numpy.array()
        Path order: Unique values representing continuous paths from the
        river outlet to the headwaters ordered from the longest path (1) 
        to the shortest path (N).
    path_main_side: numpy.array()
        Main-Side network flag.
    path_cl_node_ids: numpy.array()
        Node IDs associated with the shortest paths.

    Returns
    -------
    None.
        
    """

    unq_rch = np.unique([path_cl_rch_ids])
    for ind in list(range(len(unq_rch))):
        # print(ind, len(unq_rch)-1)
        rch = np.where(reaches.id == unq_rch[ind])[0]
        # if len(rch) == 0:
        #     continue
        pts = np.where(path_cl_rch_ids == unq_rch[ind])[0]
        reaches.path_freq[rch] = np.max(path_freq[pts])
        reaches.path_order[rch] = np.max(path_order[pts])
        reaches.path_segs[rch] = np.max(path_segs[pts])
        reaches.dist_out[rch] = np.max(path_cl_dist_out[pts])
        reaches.main_side[rch] = np.max(path_main_side[pts])
        reaches.strm_order[rch] = np.max(path_strm_order[pts])

        unq_nodes = np.unique(path_cl_node_ids[np.where(path_cl_rch_ids == unq_rch[ind])])
        for n in list(range(len(unq_nodes))):
            # print(ind)
            nds = np.where(nodes.id == unq_nodes[n])[0]
            if len(nds) == 0:
                continue
            pts2 = np.where(path_cl_node_ids == unq_nodes[n])[0]
            nodes.path_freq[nds] = np.max(path_freq[pts2])
            nodes.path_order[nds] = np.max(path_order[pts2])
            nodes.path_segs[nds] = np.max(path_segs[pts2])
            nodes.dist_out[nds] = np.max(path_cl_dist_out[pts2])
            nodes.main_side[nds] = np.max(path_main_side[pts2])
            nodes.strm_order[nds] = np.max(path_strm_order[pts2])

###############################################################################
            
def number_rchs_nodes(reaches, nodes):
    """
    Numbers SWORD reaches and nodes based on the shortest paths 
    direction information.

    Parameters
    ----------
    reaches: obj
        Object containing lcation and attribute information for
        each reach.
    nodes: obj
        Object containing lcation and attribute information for
        each node.

    Returns
    -------
    None.
        
    """

    unq_paths = np.unique(reaches.path_order)
    unq_paths = unq_paths[unq_paths>0]
    unq_paths = np.append(unq_paths,0)
    reaches.rch_num = np.zeros(len(reaches.id), dtype=int)
    nodes.node_num = np.zeros(len(nodes.id), dtype=int)
    rch_cnt = 1
    node_cnt = 1
    for ind in list(range(len(unq_paths))):
        pth = np.where(reaches.path_order == unq_paths[ind])[0]
        pth_order = np.argsort(reaches.dist_out[pth])
        rch_nums = np.array(list(range(len(pth))))+rch_cnt
        reaches.rch_num[pth[pth_order]] = rch_nums
        rch_cnt = max(rch_nums)+1

        nds_pth = np.where(nodes.path_order == unq_paths[ind])[0]
        nds_pth_order = np.argsort(nodes.dist_out[nds_pth])
        node_nums = np.array(list(range(len(nds_pth))))+node_cnt
        nodes.node_num[nds_pth[nds_pth_order]] = node_nums
        node_cnt = max(node_nums)+1

###############################################################################
        
def new_sword_ids(centerlines, nodes, reaches, path_cl_rch_ids):
    """
    Updates SWORD reach and node IDs based on the shortest paths 
    direction information.

    Parameters
    ----------
    centerlines: obj
        Object containing lcation and attribute information
        along the high-resolution centerline.
    reaches: obj
        Object containing lcation and attribute information for
        each reach.
    nodes: obj
        Object containing lcation and attribute information for
        each node.
    path_cl_rch_ids: numpy.array()
        Reach IDs associated with the path variable netCDF data. 

    Returns
    -------
    reach_ids: numpy.array()
        Updated reach IDs. 
    node_ids: numpy.array()
        Updated node IDs. 
    cl_rch_ids: numpy.array()
        Updated centerline reach IDs. 
    cl_node_ids: numpy.array()
        Updated centerline reach IDs. 
    node_rch_ids: numpy.array()
        Updated node reach IDs. 
        
    """

    reach_ids = np.copy(reaches.id)
    node_ids = np.copy(nodes.id)
    cl_rch_ids = np.copy(centerlines.reach_id)
    cl_node_ids = np.copy(centerlines.node_id)
    node_rch_ids = np.copy(nodes.reach_id)

    #loop through level 6 basin reaches and number based on the 
    #path indexing. 
    rch_l6 = np.array([str(rch)[0:6] for rch in reaches.id])
    paths_l6 = np.array([str(rch)[0:6] for rch in np.unique(path_cl_rch_ids)])
    unq_basins = np.unique(paths_l6)
    for ind in list(range(len(unq_basins))):
        # print(ind, len(unq_basins)-1)
        rch_ind = np.where(rch_l6 == unq_basins[ind])[0]
        rch_order_ids = rch_ind[np.argsort(reaches.rch_num[rch_ind])]
        basin_rch_nums = np.array(list(range(len(rch_ind))))+1
        for r in list(range(len(rch_order_ids))):
            cl_rch = np.where(centerlines.reach_id[0,:] == reaches.id[rch_order_ids[r]])[0]
            rch_type = np.unique(centerlines.new_type[cl_rch])[0]
            if len(str(basin_rch_nums[r])) == 1:
                fill = '000'
                new_rch_id = int(unq_basins[ind]+fill+str(basin_rch_nums[r])+rch_type)
            if len(str(basin_rch_nums[r])) == 2:
                fill = '00'
                new_rch_id = int(unq_basins[ind]+fill+str(basin_rch_nums[r])+rch_type)
            if len(str(basin_rch_nums[r])) == 3:
                fill = '0'
                new_rch_id = int(unq_basins[ind]+fill+str(basin_rch_nums[r])+rch_type)
            if len(str(basin_rch_nums[r])) == 4:
                new_rch_id = int(unq_basins[ind]+str(basin_rch_nums[r])+rch_type)
            #update centerline and reach level ids. 
            cl_rch_ids[0,cl_rch] = new_rch_id
            reach_ids[rch_order_ids[r]] = new_rch_id

            #update node reach ids. 
            node_ind = np.where(nodes.reach_id == reaches.id[rch_order_ids[r]])[0]
            node_rch_ids[node_ind] = new_rch_id
            # set up the loop variables for new node ids. 
            nds_order_ids = node_ind[np.argsort(nodes.node_num[node_ind])]
            basin_node_nums = np.array(list(range(len(node_ind))))+1
            for n in list(range(len(nds_order_ids))):
                cl_nodes = np.where(centerlines.node_id[0,:] == nodes.id[nds_order_ids[n]])[0]
                if len(str(basin_node_nums[n])) == 1:
                    fill = '00'
                    new_node_id = int(str(new_rch_id)[0:-1]+fill+str(basin_node_nums[n])+str(new_rch_id)[-1])
                if len(str(basin_node_nums[n])) == 2:
                    fill = '0'
                    new_node_id = int(str(new_rch_id)[0:-1]+fill+str(basin_node_nums[n])+str(new_rch_id)[-1])
                if len(str(basin_node_nums[n])) == 3:
                    new_node_id = int(str(new_rch_id)[0:-1]+str(basin_node_nums[n])+str(new_rch_id)[-1])
                #update centerline and node level ids. 
                cl_node_ids[0,cl_nodes] = new_node_id
                node_ids[nds_order_ids[n]] = new_node_id        

    return reach_ids, node_ids, cl_rch_ids, cl_node_ids, node_rch_ids

###############################################################################

def filter_river_names(reaches, nodes, path_segs, path_cl_rch_ids):
    """
    Corrects river name outliers based on path segment values.

    Parameters
    ----------
    reaches: obj
        Object containing lcation and attribute information for
        each reach.
    nodes: obj
        Object containing lcation and attribute information for
        each node.
    path_segs: numpy.array()
        Unique IDs given to river segments between junctions.
    path_cl_rch_ids: numpy.array()
        Reach IDs associated with the path variable netCDF data. 

    Returns
    -------
    None.

    """

    unq_segs = np.unique(path_segs)
    unq_segs = unq_segs[unq_segs>0]
    for ind in list(range(len(unq_segs))):
        # print(ind)
        seg = np.where(path_segs == unq_segs[ind])[0]
        seg_rchs = np.unique(path_cl_rch_ids[seg])
        rchs = np.where(np.in1d(reaches.id, seg_rchs)==True)[0]
        # if len(rchs) == 0:
        #     continue
        node_rchs = np.where(np.in1d(nodes.reach_id, seg_rchs)==True)[0]
        names, cnt = np.unique(reaches.river_name[rchs], return_counts=True)
        name_ind = np.where(cnt == max(cnt))[0][0]
        reaches.river_name[rchs] = names[name_ind]
        nodes.river_name[node_rchs] = names[name_ind]

###############################################################################    

def write_database_nc(centerlines, reaches, nodes, region, outfile):
    """
    Outputs the SWOT River Database (SWORD) information in netcdf
    format. The file contains attributes for the high-resolution centerline,
    nodes, and reaches.

    Parameters
    ----------
    centerlines: obj
        Object containing lcation and attribute information
        along the high-resolution centerline.
    reaches: obj
        Object containing lcation and attribute information for
        each reach.
    nodes: obj
        Object containing lcation and attribute information for
        each node.
    outfile -- Path for netcdf to be written.

    Returns
    -------
    None.

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
    node_trib_flag = node_grp.createVariable(
        'trib_flag', 'i4', ('num_nodes',), fill_value=-9999.)
    node_path_freq = node_grp.createVariable(
        'path_freq', 'i8', ('num_nodes',), fill_value=-9999.)
    node_path_order = node_grp.createVariable(
        'path_order', 'i8', ('num_nodes',), fill_value=-9999.)
    node_path_seg = node_grp.createVariable(
        'path_segs', 'i8', ('num_nodes',), fill_value=-9999.)
    node_strm_order = node_grp.createVariable(
        'stream_order', 'i4', ('num_nodes',), fill_value=-9999.)
    node_main_side = node_grp.createVariable(
        'main_side', 'i4', ('num_nodes',), fill_value=-9999.)
    node_end_rch = node_grp.createVariable(
        'end_reach', 'i4', ('num_nodes',), fill_value=-9999.)

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
    rch_trib_flag = rch_grp.createVariable(
        'trib_flag', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_path_freq = rch_grp.createVariable(
        'path_freq', 'i8', ('num_reaches',), fill_value=-9999.)
    rch_path_order = rch_grp.createVariable(
        'path_order', 'i8', ('num_reaches',), fill_value=-9999.)
    rch_path_seg = rch_grp.createVariable(
        'path_segs', 'i8', ('num_reaches',), fill_value=-9999.)
    rch_strm_order = rch_grp.createVariable(
        'stream_order', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_main_side = rch_grp.createVariable(
        'main_side', 'i4', ('num_reaches',), fill_value=-9999.)
    rch_end_rch = rch_grp.createVariable(
        'end_reach', 'i4', ('num_reaches',), fill_value=-9999.)
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
    cl_id[:] = centerlines.new_cl_id
    cl_x[:] = centerlines.x
    cl_y[:] = centerlines.y
    reach_id[:,:] = centerlines.new_reach_id
    node_id[:,:] = centerlines.new_node_id

    # node data
    Node_ID[:] = nodes.new_id
    node_cl_id[:,:] = nodes.cl_id
    node_x[:] = nodes.x
    node_y[:] = nodes.y
    node_len[:] = nodes.len
    node_rch_id[:] = nodes.new_reach_id
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
    node_trib_flag[:] = nodes.trib_flag
    node_path_freq[:] = nodes.path_freq
    node_path_order[:] = nodes.path_order
    node_path_seg[:] = nodes.path_segs
    node_strm_order[:] = nodes.strm_order
    node_main_side[:] = nodes.main_side
    node_end_rch[:] = nodes.end_rch

    # reach data
    Reach_ID[:] = reaches.new_id
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
    rch_trib_flag[:] = reaches.trib_flag
    rch_path_freq[:] = reaches.path_freq
    rch_path_order[:] = reaches.path_order
    rch_path_seg[:] = reaches.path_segs
    rch_strm_order[:] = reaches.strm_order
    rch_main_side[:] = reaches.main_side
    rch_end_rch[:] = reaches.end_rch
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
################################  MAIN  #######################################
###############################################################################
        
start_all = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version (i.e. v17)", type = str)
parser.add_argument("basin", help="Pfafstetter Level 2 Basin Number (i.e. 74)", type = str)
args = parser.parse_args()

region = args.region
version = args.version
basin = args.basin

print('Starting Basin: ', basin)
sword_dir = main_dir+'/data/outputs/Reaches_Nodes/'\
    +version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
con_dir = main_dir+'/data/outputs/Reaches_Nodes/'\
    +version+'/reach_geometry/'+region.lower()+'_sword_'+version+'_connectivity.nc'
path_dir = main_dir+'/data/outputs/Reaches_Nodes/'\
    +version+'/network_building/pathway_netcdfs/'+region+'/hb'+basin+'_path_vars.nc'

#read sword data, connectivity data, and path data. 
centerlines, nodes, reaches = read_data(sword_dir)
conn = nc.Dataset(con_dir)
paths = nc.Dataset(path_dir,'r+')

#assign data to arrays. 
con_cl_ids = np.array(conn.groups['centerlines'].variables['cl_id'][:])
con_rch_ids = np.array(conn.groups['centerlines'].variables['reach_id'][:])
con_end_ids = np.array(conn.groups['centerlines'].variables['end_reach'][:])
conn.close()

path_cl_ids = np.array(paths.groups['centerlines'].variables['cl_id'][:])
path_cl_rch_ids = np.array(paths.groups['centerlines'].variables['reach_id'][0,:])
path_cl_node_ids = np.array(paths.groups['centerlines'].variables['node_id'][0,:])
path_cl_dist_out = np.array(paths.groups['centerlines'].variables['dist_out_all'][:])
path_freq = np.array(paths.groups['centerlines'].variables['path_travel_frequency'][:])
path_order = np.array(paths.groups['centerlines'].variables['path_order_by_length'][:])
path_main_side = np.array(paths.groups['centerlines'].variables['main_side_chan'][:])
path_strm_order = np.array(paths.groups['centerlines'].variables['stream_order'][:])
path_segs = np.array(paths.groups['centerlines'].variables['path_segments'][:])

# Filtering River Names.
print('Filtering River Names') #basin-scale
start = time.time()
filter_river_names(reaches, nodes, path_segs, path_cl_rch_ids)
end = time.time()
print(str(np.round((end-start)/60,2))+' mins')

# Re-ordering centerline ids based on new dist_out. 
print('Re-formatting Centerline IDs') #basin-scale
start = time.time()
centerlines.new_cl_id = reorder_cl_ids(path_cl_rch_ids, path_cl_ids, 
                                       path_cl_dist_out, centerlines)
end = time.time()
print(str(np.round((end-start)/60,2))+' mins')

# update Type info for ghost reaches. 
print('Fixing Incorrect Ghost Reaches') #cont-scale
start = time.time()
centerlines.type = np.array([str(rch)[-1] for rch in centerlines.reach_id[0,:]])
centerlines.new_type = np.copy(centerlines.type)
end = time.time()
print(str(np.round((end-start)/60,2))+' mins')

# update end_reaches variable (i.e.) correct headwaters and outlets 
print('Editing Headwaters and Outlets') #basin-scale
start = time.time()
update_headwaters_outlets(path_order, path_cl_dist_out, path_cl_ids, 
                              con_cl_ids, con_end_ids) 
end = time.time()
print(str(np.round((end-start)/60,2))+' mins')

# add new variables to reaches and nodes. 
print('Adding Reach and Node Path Variables') #basin-scale
start = time.time()
add_rch_node_path_vars(reaches, nodes, path_cl_rch_ids, path_cl_dist_out, 
                      path_freq, path_order, path_main_side, path_cl_node_ids)
end = time.time()
print(str(np.round((end-start)/60,2))+' mins')

#re-number reaches and nodes. 
print('Creating New Reach and Node IDs') #basin-scale
start = time.time()
number_rchs_nodes(reaches, nodes) 
reaches.new_id, nodes.new_id, centerlines.new_reach_id, \
    centerlines.new_node_id, nodes.new_reach_id = new_sword_ids(centerlines, nodes, reaches, path_cl_rch_ids)
# zero-out topology variables for now. 
centerlines.new_node_id[1,:] = 0; centerlines.new_node_id[2,:] = 0; centerlines.new_node_id[3,:] = 0
centerlines.new_reach_id[1,:] = 0; centerlines.new_reach_id[2,:] = 0; centerlines.new_reach_id[3,:] = 0
reaches.n_rch_up[:] = 0; reaches.n_rch_down[:] = 0 
reaches.rch_id_up[:] = 0; reaches.rch_id_down[:] = 0 
end = time.time()
print(str(np.round((end-start)/60,2))+' mins')

#adding end reach attribute. 
print('Adding End Reach Attribute to Reaches and Nodes') #basin-scale
rch_hw = centerlines.new_reach_id[0,np.where(con_end_ids == 1)[0]]
rch_out = centerlines.new_reach_id[0,np.where(con_end_ids == 2)[0]]
rch_junc = centerlines.new_reach_id[0,np.where(con_end_ids == 3)[0]]
node_hw = centerlines.new_node_id[0,np.where(con_end_ids == 1)[0]]
node_out = centerlines.new_node_id[0,np.where(con_end_ids == 2)[0]]
node_junc = centerlines.new_node_id[0,np.where(con_end_ids == 3)[0]]
reaches.end_rch = np.zeros(len(reaches.new_id))
nodes.end_rch = np.zeros(len(nodes.new_id))
reaches.end_rch[np.where(np.in1d(reaches.new_id, rch_hw))] = 1
reaches.end_rch[np.where(np.in1d(reaches.new_id, rch_out))] = 2
reaches.end_rch[np.where(np.in1d(reaches.new_id, rch_junc))] = 3
nodes.end_rch[np.where(np.in1d(nodes.new_id, node_hw))] = 1
nodes.end_rch[np.where(np.in1d(nodes.new_id, node_out))] = 2
nodes.end_rch[np.where(np.in1d(nodes.new_id, node_junc))] = 3

#updating path variable netCDF. 
print('Updating Path Variable NetCDF with New SWORD IDs')
path_inds = np.where(np.in1d(centerlines.cl_id, path_cl_ids) == True)[0]
paths_new_rch_ids = centerlines.new_reach_id[:,path_inds]
paths_new_node_ids = centerlines.new_node_id[:,path_inds]
if 'new_reach_id' in paths.groups['centerlines'].variables.keys():
    paths.groups['centerlines'].variables['new_reach_id'][:] = paths_new_rch_ids
    paths.groups['centerlines'].variables['new_node_id'][:] = paths_new_node_ids
    paths.close()
else:
    paths.groups['centerlines'].createVariable('new_reach_id', 'i8', ('num_domains','num_points'), fill_value=-9999.)
    paths.groups['centerlines'].variables['new_reach_id'][:] = paths_new_rch_ids
    paths.groups['centerlines'].createVariable('new_node_id', 'i8', ('num_domains','num_points'), fill_value=-9999.)
    paths.groups['centerlines'].variables['new_node_id'][:] = paths_new_node_ids
    paths.close()

###############################################################################
### Filler variables for SWORD netCDF. 
# discharge subgroup 1
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

# print('Writing New NetCDF')
# write_database_nc(centerlines, reaches, nodes, region, sword_dir)

print('Cl Dimensions:', len(np.unique(centerlines.cl_id)), len(centerlines.cl_id))
print('Rch Dimensions:', len(np.unique(centerlines.reach_id[0,:])), len(np.unique(nodes.reach_id)), len(reaches.id))
print('New Rch Dimensions:', len(np.unique(centerlines.new_reach_id[0,:])), len(np.unique(nodes.new_reach_id)), len(reaches.new_id))
print('Node Dimensions:', len(np.unique(centerlines.node_id[0,:])), len(np.unique(nodes.id)), len(nodes.id))
print('New Node Dimensions:', len(np.unique(centerlines.new_node_id[0,:])), len(np.unique(nodes.new_id)), len(nodes.new_id))
end_all = time.time()
print('Finished ALL Updates in: '+str(np.round((end_all-start_all)/60,2))+' mins')

####################################################################################
####################################################################################
####################################################################################

print('Updating NetCDF')
sword = nc.Dataset(sword_dir, 'r+')
sword.groups['reaches'].variables['reach_id'][:] = reaches.new_id
sword.groups['nodes'].variables['node_id'][:] = nodes.new_id
sword.groups['nodes'].variables['reach_id'][:] = nodes.new_reach_id
sword.groups['centerlines'].variables['reach_id'][:] = centerlines.new_reach_id
sword.groups['centerlines'].variables['node_id'][:] = centerlines.new_node_id
sword.groups['centerlines'].variables['cl_id'][:] = centerlines.new_cl_id

# #other attributes
sword.groups['reaches'].variables['dist_out'][:] = reaches.dist_out
sword.groups['nodes'].variables['dist_out'][:] = nodes.dist_out
sword.groups['reaches'].variables['path_freq'][:] = reaches.path_freq
sword.groups['nodes'].variables['path_freq'][:] = nodes.path_freq
sword.groups['reaches'].variables['path_order'][:] = reaches.path_order
sword.groups['nodes'].variables['path_order'][:] = nodes.path_order

rch_side = np.where(sword.groups['reaches'].variables['main_side'][:] == 1)[0]
node_side = np.where(sword.groups['nodes'].variables['main_side'][:] == 1)[0]
sword.groups['reaches'].variables['path_freq'][rch_side] = -9999
sword.groups['reaches'].variables['path_order'][rch_side] = -9999
sword.groups['nodes'].variables['path_freq'][node_side] = -9999
sword.groups['nodes'].variables['path_order'][node_side] = -9999

sword.close()

####################################################################################
####################################################################################
####################################################################################
# import matplotlib.pyplot as plt

#PLOTS
# basin_ind = np.where(np.in1d(reaches.id, path_cl_rch_ids) == True)[0]
# plt.scatter(reaches.x[basin_ind], reaches.y[basin_ind], c=reaches.dist_out[basin_ind], cmap='rainbow', s=3)
# plt.show()
