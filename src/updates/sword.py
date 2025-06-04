# -*- coding: utf-8 -*-
"""
SWORD (sword.py)
=====================
Class for reading, writing, and running processing commands
on the SWOT River Database (SWORD).

also add useful plots... ?
        - plotting reach as polyline and nodes in order 
        - plot centerline points for specific reach in order
        - plot node wse?

"""
from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import src.updates.sword_utils as swd


class SWORD:
    """
    The SWORD class organizes data and processing methods for the 
    three SWORD spatial dimensions: centerlines, nodes, reaches.  

    """
    
    ###############################################################################

    def __init__(self, main_dir, region, version):
        """
        Initializes the SWORD class.

        Parameters
        ----------
        main_dir: str
            The directory where SWORD data is stored and exported. This will be
            the main directory where all data sub-directories are contained or 
            written.
        region: str
            Two-letter acronymn for a SWORD region (i.e. NA).
        version: str
            SWORD version (i.e. v18).
        
        Attributes
        ----------
        region: str
            Two-letter acronymn for a SWORD region (i.e. NA).
        version: str
            SWORD version (i.e. v18).
        paths: dict
            Contains the import and export paths for SWORD.
        centerlines: obj
            Stores the centerline dimension of SWORD and associated attributes.
            For attribute details see read_nc in swot_utils.py.
        nodes: obj
            Stores the node dimension of SWORD and associated attributes.
            For attribute details see read_nc in swot_utils.py.
        reaches: obj
            Stores the reach dimension of SWORD and associated attributes.
            For attribute details see read_nc in swot_utils.py.
        
        """
        
        self.region = region
        self.version = version
        #defining filnames and relative input and output paths below the specified directory. 
        self.paths = swd.prepare_paths(main_dir, region, version)
        #populating centerline, node, and reach attribute data. 
        self.centerlines, self.nodes, self.reaches = swd.read_nc(self.paths['nc_dir']+self.paths['nc_fn'])

    ###############################################################################

    def save_nc(self):
        """
        Save SWORD data as a netCDF file. File is saved at self.paths['nc_dir'].

        Parameters
        ----------
        None.
        
        Returns
        -------
        None.
        
        """

        #add discharge place holder attributes if missing. 
        if 'h_break' not in self.reaches.__dict__.keys():
            swd.discharge_attr_nc(self.reaches)
        #write netcdf file.
        swd.write_nc(self.centerlines,
                     self.reaches, 
                     self.nodes, 
                     self.region, 
                     self.paths['nc_dir']+self.paths['nc_fn'])

    ###############################################################################

    def save_vectors(self, export):
        """
        Saves SWORD data in vector formats (shapfile and geopackage). 
        Files are saved at self.paths['shp_dir'] and self.paths['gpkg_dir'].

        Parameters
        ----------
        export: str
            'All' - writes both reach and node files.
            'nodes' - writes node files only.
            'reaches' - writes reach files only. 
        
        Returns
        -------
        None.
        
        """
        #saving reach polylines. 
        if export == 'All' or export == 'reaches':
            #finding common points at reach junctions. 
            print('STARTING REACHES')
            print('Determining Reach Connectivity')
            common = swd.find_common_points(self.centerlines, self.reaches)
            #creating reach geometry.  
            print('Creating Geometry') 
            threshold = 500 #meters
            geom, rm_ind = swd.define_geometry(np.unique(self.reaches.id),
                                                self.centerlines.reach_id, 
                                                self.centerlines.x, 
                                                self.centerlines.y, 
                                                self.centerlines.cl_id, 
                                                common, 
                                                threshold, 
                                                self.region)
            #writing reach geopackage and shapefiles.
            print('Writing Reaches')      
            swd.write_rchs(self.reaches, geom, rm_ind, self.paths)
        #saving node points. 
        if export == 'All' or export == 'nodes':
            print('STARTING NODES')
            swd.write_nodes(self.nodes, self.paths)

    ###############################################################################

    def delete_rchs(self, rm_rch):
        """
        Deletes reaches from the SWORD reaches object.

        Parameters
        ----------
        rm_rch: list
            List of reach IDs to be deleted. 
        
        Returns
        -------
        None.
        
        """

        for ind in list(range(len(rm_rch))):
            # print(ind, len(rm_rch))
            rch_ind = np.where(self.reaches.id == rm_rch[ind])[0]
            self.reaches.id = np.delete(self.reaches.id, rch_ind, axis = 0)
            self.reaches.cl_id = np.delete(self.reaches.cl_id, rch_ind, axis = 1)
            self.reaches.x = np.delete(self.reaches.x, rch_ind, axis = 0)
            self.reaches.x_min = np.delete(self.reaches.x_min, rch_ind, axis = 0)
            self.reaches.x_max = np.delete(self.reaches.x_max, rch_ind, axis = 0)
            self.reaches.y = np.delete(self.reaches.y, rch_ind, axis = 0)
            self.reaches.y_min = np.delete(self.reaches.y_min, rch_ind, axis = 0)
            self.reaches.y_max = np.delete(self.reaches.y_max, rch_ind, axis = 0)
            self.reaches.len = np.delete(self.reaches.len, rch_ind, axis = 0)
            self.reaches.wse = np.delete(self.reaches.wse, rch_ind, axis = 0)
            self.reaches.wse_var = np.delete(self.reaches.wse_var, rch_ind, axis = 0)
            self.reaches.wth = np.delete(self.reaches.wth, rch_ind, axis = 0)
            self.reaches.wth_var = np.delete(self.reaches.wth_var, rch_ind, axis = 0)
            self.reaches.slope = np.delete(self.reaches.slope, rch_ind, axis = 0)
            self.reaches.rch_n_nodes = np.delete(self.reaches.rch_n_nodes, rch_ind, axis = 0)
            self.reaches.grod = np.delete(self.reaches.grod, rch_ind, axis = 0)
            self.reaches.grod_fid = np.delete(self.reaches.grod_fid, rch_ind, axis = 0)
            self.reaches.hfalls_fid = np.delete(self.reaches.hfalls_fid, rch_ind, axis = 0)
            self.reaches.lakeflag = np.delete(self.reaches.lakeflag, rch_ind, axis = 0)
            self.reaches.nchan_max = np.delete(self.reaches.nchan_max, rch_ind, axis = 0)
            self.reaches.nchan_mod = np.delete(self.reaches.nchan_mod, rch_ind, axis = 0)
            self.reaches.dist_out = np.delete(self.reaches.dist_out, rch_ind, axis = 0)
            self.reaches.n_rch_up = np.delete(self.reaches.n_rch_up, rch_ind, axis = 0)
            self.reaches.n_rch_down = np.delete(self.reaches.n_rch_down, rch_ind, axis = 0)
            self.reaches.rch_id_up = np.delete(self.reaches.rch_id_up, rch_ind, axis = 1)
            self.reaches.rch_id_down = np.delete(self.reaches.rch_id_down, rch_ind, axis = 1)
            self.reaches.max_obs = np.delete(self.reaches.max_obs, rch_ind, axis = 0)
            self.reaches.orbits = np.delete(self.reaches.orbits, rch_ind, axis = 1)
            self.reaches.facc = np.delete(self.reaches.facc, rch_ind, axis = 0)
            self.reaches.iceflag = np.delete(self.reaches.iceflag, rch_ind, axis = 1)
            self.reaches.max_wth = np.delete(self.reaches.max_wth, rch_ind, axis = 0)
            self.reaches.river_name = np.delete(self.reaches.river_name, rch_ind, axis = 0)
            self.reaches.low_slope = np.delete(self.reaches.low_slope, rch_ind, axis = 0)
            self.reaches.edit_flag = np.delete(self.reaches.edit_flag, rch_ind, axis = 0)
            self.reaches.trib_flag = np.delete(self.reaches.trib_flag, rch_ind, axis = 0)
            self.reaches.path_freq = np.delete(self.reaches.path_freq, rch_ind, axis = 0)
            self.reaches.path_order = np.delete(self.reaches.path_order, rch_ind, axis = 0)
            self.reaches.path_segs = np.delete(self.reaches.path_segs, rch_ind, axis = 0)
            self.reaches.main_side = np.delete(self.reaches.main_side, rch_ind, axis = 0)
            self.reaches.strm_order = np.delete(self.reaches.strm_order, rch_ind, axis = 0)
            self.reaches.end_rch = np.delete(self.reaches.end_rch, rch_ind, axis = 0)
            self.reaches.network = np.delete(self.reaches.network, rch_ind, axis = 0)
            self.reaches.add_flag = np.delete(self.reaches.add_flag, rch_ind, axis = 0)

    ###############################################################################

    def delete_nodes(self, node_ind):
        """
        Deletes nodes from the SWORD nodes object.

        Parameters
        ----------
        node_ind: list
            List containing node IDs to be deleted. 
        
        Returns
        -------
        None.
        
        """

        self.nodes.id = np.delete(self.nodes.id, node_ind, axis = 0)
        self.nodes.cl_id = np.delete(self.nodes.cl_id, node_ind, axis = 1)
        self.nodes.x = np.delete(self.nodes.x, node_ind, axis = 0)
        self.nodes.y = np.delete(self.nodes.y, node_ind, axis = 0)
        self.nodes.len = np.delete(self.nodes.len, node_ind, axis = 0)
        self.nodes.wse = np.delete(self.nodes.wse, node_ind, axis = 0)
        self.nodes.wse_var = np.delete(self.nodes.wse_var, node_ind, axis = 0)
        self.nodes.wth = np.delete(self.nodes.wth, node_ind, axis = 0)
        self.nodes.wth_var = np.delete(self.nodes.wth_var, node_ind, axis = 0)
        self.nodes.grod = np.delete(self.nodes.grod, node_ind, axis = 0)
        self.nodes.grod_fid = np.delete(self.nodes.grod_fid, node_ind, axis = 0)
        self.nodes.hfalls_fid = np.delete(self.nodes.hfalls_fid, node_ind, axis = 0)
        self.nodes.nchan_max = np.delete(self.nodes.nchan_max, node_ind, axis = 0)
        self.nodes.nchan_mod = np.delete(self.nodes.nchan_mod, node_ind, axis = 0)
        self.nodes.dist_out = np.delete(self.nodes.dist_out, node_ind, axis = 0)
        self.nodes.reach_id = np.delete(self.nodes.reach_id, node_ind, axis = 0)
        self.nodes.facc = np.delete(self.nodes.facc, node_ind, axis = 0)
        self.nodes.lakeflag = np.delete(self.nodes.lakeflag, node_ind, axis = 0)
        self.nodes.wth_coef = np.delete(self.nodes.wth_coef, node_ind, axis = 0)
        self.nodes.ext_dist_coef = np.delete(self.nodes.ext_dist_coef, node_ind, axis = 0)
        self.nodes.max_wth = np.delete(self.nodes.max_wth, node_ind, axis = 0)
        self.nodes.meand_len = np.delete(self.nodes.meand_len, node_ind, axis = 0)
        self.nodes.river_name = np.delete(self.nodes.river_name, node_ind, axis = 0)
        self.nodes.manual_add = np.delete(self.nodes.manual_add, node_ind, axis = 0)
        self.nodes.sinuosity = np.delete(self.nodes.sinuosity, node_ind, axis = 0)
        self.nodes.edit_flag = np.delete(self.nodes.edit_flag, node_ind, axis = 0)
        self.nodes.trib_flag = np.delete(self.nodes.trib_flag, node_ind, axis = 0)
        self.nodes.path_freq = np.delete(self.nodes.path_freq, node_ind, axis = 0)
        self.nodes.path_order = np.delete(self.nodes.path_order, node_ind, axis = 0)
        self.nodes.path_segs = np.delete(self.nodes.path_segs, node_ind, axis = 0)
        self.nodes.main_side = np.delete(self.nodes.main_side, node_ind, axis = 0)
        self.nodes.strm_order = np.delete(self.nodes.strm_order, node_ind, axis = 0)
        self.nodes.end_rch = np.delete(self.nodes.end_rch, node_ind, axis = 0)
        self.nodes.network = np.delete(self.nodes.network, node_ind, axis = 0)
        self.nodes.add_flag = np.delete(self.nodes.add_flag, node_ind, axis = 0)

    ###############################################################################

    def append_nodes(self, subnodes):
        """
        Appends nodes and associated attributes to existing SWORD class.

        Parameters
        ----------
        subnodes: obj
            Object containing nodes and associated attributes in the same format 
            as the SWORD class nodes object.
        
        Returns
        -------
        None.
        
        """

        self.nodes.id = np.append(self.nodes.id, subnodes.id)
        self.nodes.cl_id = np.append(self.nodes.cl_id, subnodes.cl_id, axis=1)
        self.nodes.x = np.append(self.nodes.x, subnodes.x)
        self.nodes.y = np.append(self.nodes.y, subnodes.y)
        self.nodes.len = np.append(self.nodes.len, subnodes.len)
        self.nodes.wse = np.append(self.nodes.wse, subnodes.wse)
        self.nodes.wse_var = np.append(self.nodes.wse_var, subnodes.wse_var)
        self.nodes.wth = np.append(self.nodes.wth, subnodes.wth)
        self.nodes.wth_var = np.append(self.nodes.wth_var, subnodes.wth_var)
        self.nodes.grod = np.append(self.nodes.grod, subnodes.grod)
        self.nodes.grod_fid = np.append(self.nodes.grod_fid, subnodes.grod_fid)
        self.nodes.hfalls_fid = np.append(self.nodes.hfalls_fid, subnodes.hfalls_fid)
        self.nodes.nchan_max = np.append(self.nodes.nchan_max, subnodes.nchan_max)
        self.nodes.nchan_mod = np.append(self.nodes.nchan_mod, subnodes.nchan_mod)
        self.nodes.dist_out = np.append(self.nodes.dist_out, subnodes.dist_out)
        self.nodes.reach_id = np.append(self.nodes.reach_id, subnodes.reach_id)
        self.nodes.facc = np.append(self.nodes.facc, subnodes.facc)
        self.nodes.lakeflag = np.append(self.nodes.lakeflag, subnodes.lakeflag)
        self.nodes.wth_coef = np.append(self.nodes.wth_coef, subnodes.wth_coef)
        self.nodes.ext_dist_coef = np.append(self.nodes.ext_dist_coef, subnodes.ext_dist_coef)
        self.nodes.max_wth = np.append(self.nodes.max_wth, subnodes.max_wth)
        self.nodes.meand_len = np.append(self.nodes.meand_len, subnodes.meand_len)
        self.nodes.river_name = np.append(self.nodes.river_name, subnodes.river_name)
        self.nodes.manual_add = np.append(self.nodes.manual_add, subnodes.manual_add)
        self.nodes.sinuosity = np.append(self.nodes.sinuosity, subnodes.sinuosity)
        self.nodes.edit_flag = np.append(self.nodes.edit_flag, subnodes.edit_flag)
        self.nodes.trib_flag = np.append(self.nodes.trib_flag, subnodes.trib_flag)
        self.nodes.path_freq = np.append(self.nodes.path_freq, subnodes.path_freq)
        self.nodes.path_order = np.append(self.nodes.path_order, subnodes.path_order)
        self.nodes.path_segs = np.append(self.nodes.path_segs, subnodes.path_segs)
        self.nodes.main_side = np.append(self.nodes.main_side, subnodes.main_side)
        self.nodes.strm_order = np.append(self.nodes.strm_order, subnodes.strm_order)
        self.nodes.end_rch = np.append(self.nodes.end_rch, subnodes.end_rch)
        self.nodes.network = np.append(self.nodes.network, subnodes.network)
        self.nodes.add_flag = np.append(self.nodes.add_flag, subnodes.add_flag)

    ###############################################################################

    def delete_data(self, rm_rch):
        """
        Deletes attributes associated with specific reach IDs across 
        all objects in the SWORD class.

        Parameters
        ----------
        rm_rch: list
            List containing reach IDs to be deleted. 
        
        Returns
        -------
        None.
        
        """

        for ind in list(range(len(rm_rch))):
            print(ind, len(rm_rch)-1)
            rch_ind = np.where(self.reaches.id == rm_rch[ind])[0]
            node_ind = np.where(self.nodes.reach_id == rm_rch[ind])[0]
            cl_ind = np.where(self.centerlines.reach_id[0,:] == rm_rch[ind])[0]

            if len(rch_ind) == 0:
                print(rm_rch[ind], 'not in database')

            self.centerlines.cl_id = np.delete(self.centerlines.cl_id, cl_ind, axis=0)
            self.centerlines.x = np.delete(self.centerlines.x, cl_ind, axis=0)
            self.centerlines.y = np.delete(self.centerlines.y, cl_ind, axis=0)
            self.centerlines.reach_id = np.delete(self.centerlines.reach_id, cl_ind, axis=1)
            self.centerlines.node_id = np.delete(self.centerlines.node_id, cl_ind, axis=1)

            self.nodes.id = np.delete(self.nodes.id, node_ind, axis = 0)
            self.nodes.cl_id = np.delete(self.nodes.cl_id, node_ind, axis = 1)
            self.nodes.x = np.delete(self.nodes.x, node_ind, axis = 0)
            self.nodes.y = np.delete(self.nodes.y, node_ind, axis = 0)
            self.nodes.len = np.delete(self.nodes.len, node_ind, axis = 0)
            self.nodes.wse = np.delete(self.nodes.wse, node_ind, axis = 0)
            self.nodes.wse_var = np.delete(self.nodes.wse_var, node_ind, axis = 0)
            self.nodes.wth = np.delete(self.nodes.wth, node_ind, axis = 0)
            self.nodes.wth_var = np.delete(self.nodes.wth_var, node_ind, axis = 0)
            self.nodes.grod = np.delete(self.nodes.grod, node_ind, axis = 0)
            self.nodes.grod_fid = np.delete(self.nodes.grod_fid, node_ind, axis = 0)
            self.nodes.hfalls_fid = np.delete(self.nodes.hfalls_fid, node_ind, axis = 0)
            self.nodes.nchan_max = np.delete(self.nodes.nchan_max, node_ind, axis = 0)
            self.nodes.nchan_mod = np.delete(self.nodes.nchan_mod, node_ind, axis = 0)
            self.nodes.dist_out = np.delete(self.nodes.dist_out, node_ind, axis = 0)
            self.nodes.reach_id = np.delete(self.nodes.reach_id, node_ind, axis = 0)
            self.nodes.facc = np.delete(self.nodes.facc, node_ind, axis = 0)
            self.nodes.lakeflag = np.delete(self.nodes.lakeflag, node_ind, axis = 0)
            self.nodes.wth_coef = np.delete(self.nodes.wth_coef, node_ind, axis = 0)
            self.nodes.ext_dist_coef = np.delete(self.nodes.ext_dist_coef, node_ind, axis = 0)
            self.nodes.max_wth = np.delete(self.nodes.max_wth, node_ind, axis = 0)
            self.nodes.meand_len = np.delete(self.nodes.meand_len, node_ind, axis = 0)
            self.nodes.river_name = np.delete(self.nodes.river_name, node_ind, axis = 0)
            self.nodes.manual_add = np.delete(self.nodes.manual_add, node_ind, axis = 0)
            self.nodes.sinuosity = np.delete(self.nodes.sinuosity, node_ind, axis = 0)
            self.nodes.edit_flag = np.delete(self.nodes.edit_flag, node_ind, axis = 0)
            self.nodes.trib_flag = np.delete(self.nodes.trib_flag, node_ind, axis = 0)
            self.nodes.path_freq = np.delete(self.nodes.path_freq, node_ind, axis = 0)
            self.nodes.path_order = np.delete(self.nodes.path_order, node_ind, axis = 0)
            self.nodes.path_segs = np.delete(self.nodes.path_segs, node_ind, axis = 0)
            self.nodes.main_side = np.delete(self.nodes.main_side, node_ind, axis = 0)
            self.nodes.strm_order = np.delete(self.nodes.strm_order, node_ind, axis = 0)
            self.nodes.end_rch = np.delete(self.nodes.end_rch, node_ind, axis = 0)
            self.nodes.network = np.delete(self.nodes.network, node_ind, axis = 0)
            self.nodes.add_flag = np.delete(self.nodes.add_flag, node_ind, axis = 0)

            self.reaches.id = np.delete(self.reaches.id, rch_ind, axis = 0)
            self.reaches.cl_id = np.delete(self.reaches.cl_id, rch_ind, axis = 1)
            self.reaches.x = np.delete(self.reaches.x, rch_ind, axis = 0)
            self.reaches.x_min = np.delete(self.reaches.x_min, rch_ind, axis = 0)
            self.reaches.x_max = np.delete(self.reaches.x_max, rch_ind, axis = 0)
            self.reaches.y = np.delete(self.reaches.y, rch_ind, axis = 0)
            self.reaches.y_min = np.delete(self.reaches.y_min, rch_ind, axis = 0)
            self.reaches.y_max = np.delete(self.reaches.y_max, rch_ind, axis = 0)
            self.reaches.len = np.delete(self.reaches.len, rch_ind, axis = 0)
            self.reaches.wse = np.delete(self.reaches.wse, rch_ind, axis = 0)
            self.reaches.wse_var = np.delete(self.reaches.wse_var, rch_ind, axis = 0)
            self.reaches.wth = np.delete(self.reaches.wth, rch_ind, axis = 0)
            self.reaches.wth_var = np.delete(self.reaches.wth_var, rch_ind, axis = 0)
            self.reaches.slope = np.delete(self.reaches.slope, rch_ind, axis = 0)
            self.reaches.rch_n_nodes = np.delete(self.reaches.rch_n_nodes, rch_ind, axis = 0)
            self.reaches.grod = np.delete(self.reaches.grod, rch_ind, axis = 0)
            self.reaches.grod_fid = np.delete(self.reaches.grod_fid, rch_ind, axis = 0)
            self.reaches.hfalls_fid = np.delete(self.reaches.hfalls_fid, rch_ind, axis = 0)
            self.reaches.lakeflag = np.delete(self.reaches.lakeflag, rch_ind, axis = 0)
            self.reaches.nchan_max = np.delete(self.reaches.nchan_max, rch_ind, axis = 0)
            self.reaches.nchan_mod = np.delete(self.reaches.nchan_mod, rch_ind, axis = 0)
            self.reaches.dist_out = np.delete(self.reaches.dist_out, rch_ind, axis = 0)
            self.reaches.n_rch_up = np.delete(self.reaches.n_rch_up, rch_ind, axis = 0)
            self.reaches.n_rch_down = np.delete(self.reaches.n_rch_down, rch_ind, axis = 0)
            self.reaches.rch_id_up = np.delete(self.reaches.rch_id_up, rch_ind, axis = 1)
            self.reaches.rch_id_down = np.delete(self.reaches.rch_id_down, rch_ind, axis = 1)
            self.reaches.max_obs = np.delete(self.reaches.max_obs, rch_ind, axis = 0)
            self.reaches.orbits = np.delete(self.reaches.orbits, rch_ind, axis = 1)
            self.reaches.facc = np.delete(self.reaches.facc, rch_ind, axis = 0)
            self.reaches.iceflag = np.delete(self.reaches.iceflag, rch_ind, axis = 1)
            self.reaches.max_wth = np.delete(self.reaches.max_wth, rch_ind, axis = 0)
            self.reaches.river_name = np.delete(self.reaches.river_name, rch_ind, axis = 0)
            self.reaches.low_slope = np.delete(self.reaches.low_slope, rch_ind, axis = 0)
            self.reaches.edit_flag = np.delete(self.reaches.edit_flag, rch_ind, axis = 0)
            self.reaches.trib_flag = np.delete(self.reaches.trib_flag, rch_ind, axis = 0)
            self.reaches.path_freq = np.delete(self.reaches.path_freq, rch_ind, axis = 0)
            self.reaches.path_order = np.delete(self.reaches.path_order, rch_ind, axis = 0)
            self.reaches.path_segs = np.delete(self.reaches.path_segs, rch_ind, axis = 0)
            self.reaches.main_side = np.delete(self.reaches.main_side, rch_ind, axis = 0)
            self.reaches.strm_order = np.delete(self.reaches.strm_order, rch_ind, axis = 0)
            self.reaches.end_rch = np.delete(self.reaches.end_rch, rch_ind, axis = 0)
            self.reaches.network = np.delete(self.reaches.network, rch_ind, axis = 0)
            self.reaches.add_flag = np.delete(self.reaches.add_flag, rch_ind, axis = 0)

            #removing residual neighbors with deleted reach id in centerline and reach groups. 
            cl_ind1 = np.where(self.centerlines.reach_id[0,:] == rm_rch[ind])[0]
            cl_ind2 = np.where(self.centerlines.reach_id[1,:] == rm_rch[ind])[0]
            cl_ind3 = np.where(self.centerlines.reach_id[2,:] == rm_rch[ind])[0]
            cl_ind4 = np.where(self.centerlines.reach_id[3,:] == rm_rch[ind])[0]
            if len(cl_ind1) > 0:
                self.centerlines.reach_id[0,cl_ind1] = 0
            if len(cl_ind2) > 0:
                self.centerlines.reach_id[1,cl_ind2] = 0
            if len(cl_ind3) > 0:
                self.centerlines.reach_id[2,cl_ind3] = 0
            if len(cl_ind4) > 0:
                self.centerlines.reach_id[3,cl_ind4] = 0

            rch_up_ind1 = np.where(self.reaches.rch_id_up[0,:] == rm_rch[ind])[0]
            rch_up_ind2 = np.where(self.reaches.rch_id_up[1,:] == rm_rch[ind])[0]
            rch_up_ind3 = np.where(self.reaches.rch_id_up[2,:] == rm_rch[ind])[0]
            rch_up_ind4 = np.where(self.reaches.rch_id_up[3,:] == rm_rch[ind])[0]
            if len(rch_up_ind1) > 0:
                self.reaches.rch_id_up[0,rch_up_ind1] = 0
                self.reaches.rch_id_up[:,rch_up_ind1] = np.sort(self.reaches.rch_id_up[:,rch_up_ind1], axis = 0)[::-1]
                up1 = np.unique(self.reaches.rch_id_up[:,rch_up_ind1]); up1 = up1[up1>0]
                self.reaches.n_rch_up[rch_up_ind1] = len(up1)
            if len(rch_up_ind2) > 0:
                self.reaches.rch_id_up[1,rch_up_ind2] = 0
                self.reaches.rch_id_up[:,rch_up_ind2] = np.sort(self.reaches.rch_id_up[:,rch_up_ind2], axis = 0)[::-1]
                up2 = np.unique(self.reaches.rch_id_up[:,rch_up_ind2]); up2 = up2[up2>0]
                self.reaches.n_rch_up[rch_up_ind2] = len(up2)
            if len(rch_up_ind3) > 0:
                self.reaches.rch_id_up[2,rch_up_ind3] = 0
                self.reaches.rch_id_up[:,rch_up_ind3] = np.sort(self.reaches.rch_id_up[:,rch_up_ind3], axis = 0)[::-1]
                up3 = np.unique(self.reaches.rch_id_up[:,rch_up_ind3]); up3 = up3[up3>0]
                self.reaches.n_rch_up[rch_up_ind3] = len(up3)
            if len(rch_up_ind4) > 0:
                self.reaches.rch_id_up[3,rch_up_ind4] = 0
                self.reaches.rch_id_up[:,rch_up_ind4] = np.sort(self.reaches.rch_id_up[:,rch_up_ind4], axis = 0)[::-1]
                up4 = np.unique(self.reaches.rch_id_up[:,rch_up_ind4]); up4 = up4[up4>0]
                self.reaches.n_rch_up[rch_up_ind4] = len(up4)

            rch_dn_ind1 = np.where(self.reaches.rch_id_down[0,:] == rm_rch[ind])[0]
            rch_dn_ind2 = np.where(self.reaches.rch_id_down[1,:] == rm_rch[ind])[0]
            rch_dn_ind3 = np.where(self.reaches.rch_id_down[2,:] == rm_rch[ind])[0]
            rch_dn_ind4 = np.where(self.reaches.rch_id_down[3,:] == rm_rch[ind])[0]
            if len(rch_dn_ind1) > 0:
                self.reaches.rch_id_down[0,rch_dn_ind1] = 0
                self.reaches.rch_id_down[:,rch_dn_ind1] = np.sort(self.reaches.rch_id_down[:,rch_dn_ind1], axis = 0)[::-1]
                dn1 = np.unique(self.reaches.rch_id_down[:,rch_dn_ind1]); dn1 = dn1[dn1>0]
                self.reaches.n_rch_down[rch_dn_ind1] = len(dn1)
            if len(rch_dn_ind2) > 0:
                self.reaches.rch_id_down[1,rch_dn_ind2] = 0
                self.reaches.rch_id_down[:,rch_dn_ind2] = np.sort(self.reaches.rch_id_down[:,rch_dn_ind2], axis = 0)[::-1]
                dn2 = np.unique(self.reaches.rch_id_down[:,rch_dn_ind2]); dn2 = dn2[dn2>0]
                self.reaches.n_rch_down[rch_dn_ind2] = len(dn2)
            if len(rch_dn_ind3) > 0:
                self.reaches.rch_id_down[2,rch_dn_ind3] = 0
                self.reaches.rch_id_down[:,rch_dn_ind3] = np.sort(self.reaches.rch_id_down[:,rch_dn_ind3], axis = 0)[::-1]
                dn3 = np.unique(self.reaches.rch_id_down[:,rch_dn_ind3]); dn3 = dn3[dn3>0]
                self.reaches.n_rch_down[rch_dn_ind3] = len(dn3)
            if len(rch_dn_ind4) > 0:
                self.reaches.rch_id_down[3,rch_dn_ind4] = 0
                self.reaches.rch_id_down[:,rch_dn_ind4] = np.sort(self.reaches.rch_id_down[:,rch_dn_ind4], axis = 0)[::-1]
                dn4 = np.unique(self.reaches.rch_id_down[:,rch_dn_ind4]); dn4 = dn4[dn4>0]
                self.reaches.n_rch_down[rch_dn_ind4] = len(dn4)

    ###############################################################################

    def append_data(self, subcls, subnodes, subreaches):
        """
        Appends objects and associated attributes in existing SWORD class.

        Parameters
        ----------
        subcls: obj
            Object containing centerlines and associated attributes in the same format 
            as the SWORD class centerlines object.
        subnodes: obj
            Object containing nodes and associated attributes in the same format 
            as the SWORD class nodes object.
        subreaches: obj
            Object containing reaches and associated attributes in the same format 
            as the SWORD class reaches object.
        
        Returns
        -------
        None.
        
        """

        self.centerlines.cl_id = np.append(self.centerlines.cl_id, subcls.new_cl_id)
        self.centerlines.x = np.append(self.centerlines.x, subcls.lon)
        self.centerlines.y = np.append(self.centerlines.y, subcls.lat)
        self.centerlines.reach_id = np.append(self.centerlines.reach_id, subcls.new_reach_id, axis=1)
        self.centerlines.node_id = np.append(self.centerlines.node_id, subcls.new_node_id, axis=1)
        
        self.nodes.id = np.append(self.nodes.id, subnodes.id)
        self.nodes.cl_id = np.append(self.nodes.cl_id, subnodes.cl_id, axis=1)
        self.nodes.x = np.append(self.nodes.x, subnodes.x)
        self.nodes.y = np.append(self.nodes.y, subnodes.y)
        self.nodes.len = np.append(self.nodes.len, subnodes.len)
        self.nodes.wse = np.append(self.nodes.wse, subnodes.wse)
        self.nodes.wse_var = np.append(self.nodes.wse_var, subnodes.wse_var)
        self.nodes.wth = np.append(self.nodes.wth, subnodes.wth)
        self.nodes.wth_var = np.append(self.nodes.wth_var, subnodes.wth_var)
        self.nodes.grod = np.append(self.nodes.grod, subnodes.grod)
        self.nodes.grod_fid = np.append(self.nodes.grod_fid, subnodes.grod_fid)
        self.nodes.hfalls_fid = np.append(self.nodes.hfalls_fid, subnodes.hfalls_fid)
        self.nodes.nchan_max = np.append(self.nodes.nchan_max, subnodes.nchan_max)
        self.nodes.nchan_mod = np.append(self.nodes.nchan_mod, subnodes.nchan_mod)
        self.nodes.dist_out = np.append(self.nodes.dist_out, subnodes.dist_out)
        self.nodes.reach_id = np.append(self.nodes.reach_id, subnodes.reach_id)
        self.nodes.facc = np.append(self.nodes.facc, subnodes.facc)
        self.nodes.lakeflag = np.append(self.nodes.lakeflag, subnodes.lakeflag)
        self.nodes.wth_coef = np.append(self.nodes.wth_coef, subnodes.wth_coef)
        self.nodes.ext_dist_coef = np.append(self.nodes.ext_dist_coef, subnodes.ext_dist_coef)
        self.nodes.max_wth = np.append(self.nodes.max_wth, subnodes.max_wth)
        self.nodes.meand_len = np.append(self.nodes.meand_len, subnodes.meand_len)
        self.nodes.river_name = np.append(self.nodes.river_name, subnodes.river_name)
        self.nodes.manual_add = np.append(self.nodes.manual_add, subnodes.manual_add)
        self.nodes.sinuosity = np.append(self.nodes.sinuosity, subnodes.sinuosity)
        self.nodes.edit_flag = np.append(self.nodes.edit_flag, subnodes.edit_flag)
        self.nodes.trib_flag = np.append(self.nodes.trib_flag, subnodes.trib_flag)
        self.nodes.path_freq = np.append(self.nodes.path_freq, subnodes.path_freq)
        self.nodes.path_order = np.append(self.nodes.path_order, subnodes.path_order)
        self.nodes.path_segs = np.append(self.nodes.path_segs, subnodes.path_segs)
        self.nodes.main_side = np.append(self.nodes.main_side, subnodes.main_side)
        self.nodes.strm_order = np.append(self.nodes.strm_order, subnodes.strm_order)
        self.nodes.end_rch = np.append(self.nodes.end_rch, subnodes.end_rch)
        self.nodes.network = np.append(self.nodes.network, subnodes.network)
        self.nodes.add_flag = np.append(self.nodes.add_flag, subnodes.add_flag)

        self.reaches.id = np.append(self.reaches.id, subreaches.id)
        self.reaches.cl_id = np.append(self.reaches.cl_id, subreaches.cl_id, axis=1)
        self.reaches.x = np.append(self.reaches.x, subreaches.x)
        self.reaches.x_min = np.append(self.reaches.x_min, subreaches.x_min)
        self.reaches.x_max = np.append(self.reaches.x_max, subreaches.x_max)
        self.reaches.y = np.append(self.reaches.y, subreaches.y)
        self.reaches.y_min = np.append(self.reaches.y_min, subreaches.y_min)
        self.reaches.y_max = np.append(self.reaches.y_max, subreaches.y_max)
        self.reaches.len = np.append(self.reaches.len, subreaches.len)
        self.reaches.wse = np.append(self.reaches.wse, subreaches.wse)
        self.reaches.wse_var = np.append(self.reaches.wse_var, subreaches.wse_var)
        self.reaches.wth = np.append(self.reaches.wth, subreaches.wth)
        self.reaches.wth_var = np.append(self.reaches.wth_var, subreaches.wth_var)
        self.reaches.slope = np.append(self.reaches.slope, subreaches.slope)
        self.reaches.rch_n_nodes = np.append(self.reaches.rch_n_nodes, subreaches.rch_n_nodes)
        self.reaches.grod = np.append(self.reaches.grod, subreaches.grod)
        self.reaches.grod_fid = np.append(self.reaches.grod_fid, subreaches.grod_fid)
        self.reaches.hfalls_fid = np.append(self.reaches.hfalls_fid, subreaches.hfalls_fid)
        self.reaches.lakeflag = np.append(self.reaches.lakeflag, subreaches.lakeflag)
        self.reaches.nchan_max = np.append(self.reaches.nchan_max, subreaches.nchan_max)
        self.reaches.nchan_mod = np.append(self.reaches.nchan_mod, subreaches.nchan_mod)
        self.reaches.dist_out = np.append(self.reaches.dist_out, subreaches.dist_out)
        self.reaches.n_rch_up = np.append(self.reaches.n_rch_up, subreaches.n_rch_up)
        self.reaches.n_rch_down = np.append(self.reaches.n_rch_down, subreaches.n_rch_down)
        self.reaches.rch_id_up = np.append(self.reaches.rch_id_up, subreaches.rch_id_up, axis=1)
        self.reaches.rch_id_down = np.append(self.reaches.rch_id_down, subreaches.rch_id_down, axis=1)
        self.reaches.max_obs = np.append(self.reaches.max_obs, subreaches.max_obs)
        self.reaches.orbits = np.append(self.reaches.orbits, subreaches.orbits, axis=1)
        self.reaches.facc = np.append(self.reaches.facc, subreaches.facc)
        self.reaches.iceflag = np.append(self.reaches.iceflag, subreaches.iceflag, axis=1)
        self.reaches.max_wth = np.append(self.reaches.max_wth, subreaches.max_wth)
        self.reaches.river_name = np.append(self.reaches.river_name, subreaches.river_name)
        self.reaches.low_slope = np.append(self.reaches.low_slope, subreaches.low_slope)
        self.reaches.edit_flag = np.append(self.reaches.edit_flag, subreaches.edit_flag)
        self.reaches.trib_flag = np.append(self.reaches.trib_flag, subreaches.trib_flag)
        self.reaches.path_freq = np.append(self.reaches.path_freq, subreaches.path_freq)
        self.reaches.path_order = np.append(self.reaches.path_order, subreaches.path_order)
        self.reaches.path_segs = np.append(self.reaches.path_segs, subreaches.path_segs)
        self.reaches.main_side = np.append(self.reaches.main_side, subreaches.main_side)
        self.reaches.strm_order = np.append(self.reaches.strm_order, subreaches.strm_order)
        self.reaches.end_rch = np.append(self.reaches.end_rch, subreaches.end_rch)
        self.reaches.network = np.append(self.reaches.network, subreaches.network)
        self.reaches.add_flag = np.append(self.reaches.add_flag, subreaches.add_flag)
    