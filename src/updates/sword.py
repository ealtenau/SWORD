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
from datetime import datetime
import shutil
import src.updates.sword_utils as swd
import src.updates.geo_utils as geo 

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

    def copy(self):
        """
        Saves a copy of SWORD data in same file directory.

        Parameters
        ----------
        None.
        
        Returns
        -------
        None.
        
        """
        current_datetime = datetime.now()
        copy_fn = self.paths['nc_dir']+\
            self.paths['nc_fn'][:-3]+\
                '_copy_'+\
                    current_datetime.strftime("%Y-%m-%d_%H-%M-%S")+\
                        '.nc'
        
        shutil.copy2(self.paths['nc_dir']+self.paths['nc_fn'], copy_fn)

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
            geom, rm_ind = swd.define_geometry(self.centerlines.reach_id, 
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
            # print(ind, len(rm_rch)-1)
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

        self.centerlines.cl_id = np.append(self.centerlines.cl_id, subcls.cl_id)
        self.centerlines.x = np.append(self.centerlines.x, subcls.lon)
        self.centerlines.y = np.append(self.centerlines.y, subcls.lat)
        self.centerlines.reach_id = np.append(self.centerlines.reach_id, subcls.reach_id, axis=1)
        self.centerlines.node_id = np.append(self.centerlines.node_id, subcls.node_id, axis=1)
        
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

    ###############################################################################

    def break_reaches(self, reach_id, break_cl_id, verbose=False):
        """
        Breaks and creates new SWORD reaches at specified locations.

        Parameters
        ----------
        reach_id: numpy.array() or list
            Reach IDs of SWORD reaches to break.
        breack_cl_id: numpy.array() or list
            Centerline IDs along the reach indicating where to break 
            the reach.
            
        Returns
        -------
        None.
            
        """

        #isolate type, basin, and node numbers from Node IDs. 
        cl_level6 = np.array([str(ind)[0:6] for ind in self.centerlines.node_id[0,:]])
        cl_node_num_int = np.array([int(str(ind)[10:13]) for ind in self.centerlines.node_id[0,:]])
        cl_rch_type = np.array([str(ind)[-1] for ind in self.centerlines.node_id[0,:]])

        #format input break variables.
        reach = np.array(reach_id)
        break_id = np.array(break_cl_id)

        #loop through and break reaches. 
        unq_rchs = np.unique(reach)
        for r in list(range(len(unq_rchs))):
            if verbose == True:
                print(r, unq_rchs[r], len(unq_rchs)-1)
            
            #find associated centerline point with reach and sort indexes. 
            cl_r = np.where(self.centerlines.reach_id[0,:] == unq_rchs[r])[0]
            order_ids = np.argsort(self.centerlines.cl_id[cl_r])
            old_dist = self.reaches.dist_out[np.where(self.reaches.id == unq_rchs[r])[0]]
            old_len = self.reaches.len[np.where(self.reaches.id == unq_rchs[r])[0]]
            base_val = old_dist - old_len

            #find break points. 
            breaks = break_id[np.where(reach == unq_rchs[r])[0]]
            break_pts = np.array([np.where(self.centerlines.cl_id[cl_r[order_ids]] == b)[0][0] 
                                  for b in breaks])

            #append start and end points. 
            bounds = np.append(0,break_pts)
            bounds = np.append(bounds, len(cl_r))
            bounds = np.sort(bounds) #added 4/26/24
            bounds = np.unique(bounds) #added 6/7/24

            #creating temperary array with new reach divisions 
            #at the centerline spatial scale.     
            new_divs = np.zeros(len(cl_r))
            count = 1
            for b in list(range(len(bounds)-1)):
                update_nds = cl_r[order_ids[bounds[b]:bounds[b+1]]]
                nds = np.unique(self.centerlines.node_id[0,update_nds])
                fill = np.where(np.in1d(self.centerlines.node_id[0,cl_r[order_ids]], nds) == True)[0]
                if np.max(new_divs[fill])==0:
                    new_divs[fill] = count 
                    count = count+1
                else:
                    z = np.where(new_divs[fill] == 0)[0]
                    new_divs[fill[z]] = count
                    count = count+1

            #updating attribute info for new reach divisions.  
            unq_divs = np.unique(new_divs)
            if len(unq_divs) == 1:
                continue
            else:
                for d in list(range(len(unq_divs))):
                    # print('b', b)
                    if d == 0:
                        # print('1')
                        update_ids = cl_r[order_ids[np.where(new_divs == unq_divs[d])]]
                        new_cl_rch_id = self.centerlines.reach_id[0,update_ids]
                        new_cl_node_ids = self.centerlines.node_id[0,update_ids]
                        new_rch_id = np.unique(self.centerlines.reach_id[0,update_ids])[0]
                    else:
                        # print('2')
                        #Create New Reach ID
                        update_ids = cl_r[order_ids[np.where(new_divs == unq_divs[d])]]
                        old_nodes = np.unique(self.centerlines.node_id[0,update_ids])
                        old_rch = np.unique(self.centerlines.reach_id[0,update_ids])[0]
                        l6_basin = np.where(cl_level6 == np.unique(cl_level6[update_ids]))[0]
                        cl_rch_num_int = np.array([int(str(ind)[6:10]) 
                                                   for ind in self.centerlines.node_id[0,l6_basin]])
                        new_rch_num = np.max(cl_rch_num_int)+1
                        if len(str(new_rch_num)) == 1:
                            fill = '000'
                            new_rch_id = int(str(np.unique(cl_level6[update_ids])[0])+fill+
                                             str(new_rch_num)+str(np.unique(cl_rch_type[update_ids])[0]))
                        if len(str(new_rch_num)) == 2:
                            fill = '00'
                            new_rch_id = int(str(np.unique(cl_level6[update_ids])[0])+fill+
                                             str(new_rch_num)+str(np.unique(cl_rch_type[update_ids])[0]))
                        if len(str(new_rch_num)) == 3:
                            fill = '0'
                            new_rch_id = int(str(np.unique(cl_level6[update_ids])[0])+fill+
                                             str(new_rch_num)+str(np.unique(cl_rch_type[update_ids])[0]))
                        if len(str(new_rch_num)) == 4:
                            new_rch_id = int(str(np.unique(cl_level6[update_ids])[0])+
                                             str(new_rch_num)+str(np.unique(cl_rch_type[update_ids])[0]))
                        new_cl_rch_id = np.repeat(new_rch_id, len(update_ids))

                        # print('3')
                        #Create New Node IDs 
                        new_cl_node_ids = np.zeros(len(update_ids),dtype=int)
                        new_cl_node_nums = cl_node_num_int[update_ids] - np.min(cl_node_num_int[update_ids]) + 1
                        for n in list(range(len(new_cl_node_nums))):
                            if len(str(new_cl_node_nums[n])) == 1:
                                fill = '00'
                                new_cl_node_ids[n] = int(str(new_rch_id)[0:-1]+fill+
                                                         str(new_cl_node_nums[n])+str(new_rch_id)[-1])
                            if len(str(new_cl_node_nums[n])) == 2:
                                fill = '0'
                                new_cl_node_ids[n] = int(str(new_rch_id)[0:-1]+fill+
                                                         str(new_cl_node_nums[n])+str(new_rch_id)[-1])
                            if len(str(new_cl_node_nums[n])) == 3:
                                new_cl_node_ids[n] = int(str(new_rch_id)[0:-1]+fill+
                                                         str(new_cl_node_nums[n])+str(new_rch_id)[-1])

                    #updating x-y and length information for new reach definitions. 
                    x_coords = self.centerlines.x[update_ids]
                    y_coords = self.centerlines.y[update_ids]
                    diff = geo.get_distances(x_coords,y_coords)
                    dist = np.cumsum(diff)
                    
                    # print('5')
                    new_rch_len = np.max(dist)
                    new_rch_x = np.median(self.centerlines.x[update_ids])
                    new_rch_y = np.median(self.centerlines.y[update_ids])
                    new_rch_x_max = np.max(self.centerlines.x[update_ids])
                    new_rch_x_min = np.min(self.centerlines.x[update_ids])
                    new_rch_y_max = np.max(self.centerlines.y[update_ids])
                    new_rch_y_min = np.min(self.centerlines.y[update_ids])

                    # print('6')
                    unq_nodes = np.unique(new_cl_node_ids)
                    new_node_len = np.zeros(len(unq_nodes))
                    new_node_x = np.zeros(len(unq_nodes))
                    new_node_y = np.zeros(len(unq_nodes))
                    new_node_id = np.zeros(len(unq_nodes))
                    new_node_cl_ids = np.zeros((2,len(unq_nodes)))
                    for n2 in list(range(len(unq_nodes))):
                        pts = np.where(new_cl_node_ids == unq_nodes[n2])[0]
                        new_node_x[n2] = np.median(self.centerlines.x[update_ids[pts]])
                        new_node_y[n2] = np.median(self.centerlines.y[update_ids[pts]])
                        new_node_len[n2] = max(np.cumsum(diff[pts]))
                        new_node_id[n2] = unq_nodes[n2]
                        new_node_cl_ids[0,n2] = np.min(self.centerlines.cl_id[update_ids[pts]])
                        new_node_cl_ids[1,n2] = np.max(self.centerlines.cl_id[update_ids[pts]])
                        if len(pts) == 1:
                            new_node_len[n2] = 30
                    
                    if new_rch_id in self.reaches.id:
                        # print('7')
                        node_ind = np.where(np.in1d(self.nodes.id, new_node_id)==True)[0]
                        self.nodes.len[node_ind] = new_node_len
                        self.nodes.cl_id[:,node_ind] = new_node_cl_ids
                        self.nodes.x[node_ind] = new_node_x
                        self.nodes.y[node_ind] = new_node_y
                        
                        rch = np.where(self.reaches.id == new_rch_id)[0]
                        self.reaches.cl_id[0,rch] = np.min(self.centerlines.cl_id[update_ids])
                        self.reaches.cl_id[1,rch] = np.max(self.centerlines.cl_id[update_ids])
                        self.reaches.x[rch] = new_rch_x
                        self.reaches.x_min[rch] = new_rch_x_min
                        self.reaches.x_max[rch] = new_rch_x_max
                        self.reaches.y[rch] = new_rch_y
                        self.reaches.y_min[rch] = new_rch_y_min
                        self.reaches.y_max[rch] = new_rch_y_max
                        self.reaches.len[rch] = new_rch_len
                        self.reaches.rch_n_nodes[rch] = len(new_node_id)
                        if self.reaches.edit_flag[rch] == 'NaN':
                            edit_val = '6'
                        elif '6' not in self.reaches.edit_flag[rch][0].split(','):
                            edit_val = self.reaches.edit_flag[rch] + ',6'
                        else:
                            edit_val = self.reaches.edit_flag[rch]
                        self.reaches.edit_flag[rch] = edit_val
                        self.nodes.edit_flag[node_ind] = edit_val

                    else:
                        # print('8')
                        self.centerlines.reach_id[0,update_ids] = new_cl_rch_id
                        self.centerlines.node_id[0,update_ids] = new_cl_node_ids
                        
                        old_ind = np.where(np.in1d(self.nodes.id, old_nodes) == True)[0]
                        self.nodes.id[old_ind] = new_node_id
                        self.nodes.len[old_ind] = new_node_len
                        self.nodes.cl_id[:,old_ind] = new_node_cl_ids
                        self.nodes.x[old_ind] = new_node_x
                        self.nodes.y[old_ind] = new_node_y
                        self.nodes.reach_id[old_ind] = np.repeat(new_rch_id, len(new_node_id))
                        
                        rch = np.where(self.reaches.id == old_rch)[0]
                        self.reaches.id = np.append(self.reaches.id, new_rch_id)
                        new_cl_ids = np.array([np.min(self.centerlines.cl_id[update_ids]), 
                                            np.max(self.centerlines.cl_id[update_ids])]).reshape(2,1)
                        self.reaches.cl_id = np.append(self.reaches.cl_id, new_cl_ids, axis=1)
                        self.reaches.x = np.append(self.reaches.x, new_rch_x)
                        self.reaches.x_min = np.append(self.reaches.x_min, new_rch_x_min)
                        self.reaches.x_max = np.append(self.reaches.x_max, new_rch_x_max)
                        self.reaches.y = np.append(self.reaches.y, new_rch_y)
                        self.reaches.y_min = np.append(self.reaches.y_min, new_rch_y_min)
                        self.reaches.y_max = np.append(self.reaches.y_max, new_rch_y_max)
                        self.reaches.len = np.append(self.reaches.len, new_rch_len)
                        self.reaches.rch_n_nodes = np.append(self.reaches.rch_n_nodes, len(new_node_id))
                        #fill attribute with current values. 
                        self.reaches.wse = np.append(self.reaches.wse, self.reaches.wse[rch])
                        self.reaches.wse_var = np.append(self.reaches.wse_var, self.reaches.wse_var[rch])
                        self.reaches.wth = np.append(self.reaches.wth, self.reaches.wth[rch])
                        self.reaches.wth_var = np.append(self.reaches.wth_var, self.reaches.wth_var[rch])
                        self.reaches.slope = np.append(self.reaches.slope, self.reaches.slope[rch])
                        self.reaches.grod = np.append(self.reaches.grod, self.reaches.grod[rch])
                        self.reaches.grod_fid = np.append(self.reaches.grod_fid, self.reaches.grod_fid[rch])
                        self.reaches.hfalls_fid = np.append(self.reaches.hfalls_fid, self.reaches.hfalls_fid[rch])
                        self.reaches.lakeflag = np.append(self.reaches.lakeflag, self.reaches.lakeflag[rch])
                        self.reaches.nchan_max = np.append(self.reaches.nchan_max, self.reaches.nchan_max[rch])
                        self.reaches.nchan_mod = np.append(self.reaches.nchan_mod, self.reaches.nchan_mod[rch])
                        self.reaches.dist_out = np.append(self.reaches.dist_out, self.reaches.dist_out[rch])
                        self.reaches.n_rch_up = np.append(self.reaches.n_rch_up, self.reaches.n_rch_up[rch])
                        self.reaches.n_rch_down = np.append(self.reaches.n_rch_down, self.reaches.n_rch_down[rch])
                        self.reaches.rch_id_up = np.append(self.reaches.rch_id_up, self.reaches.rch_id_up[:,rch], axis=1)
                        self.reaches.rch_id_down = np.append(self.reaches.rch_id_down, self.reaches.rch_id_down[:,rch], axis=1)
                        self.reaches.max_obs = np.append(self.reaches.max_obs, self.reaches.max_obs[rch])
                        self.reaches.orbits = np.append(self.reaches.orbits, self.reaches.orbits[:,rch], axis=1)
                        self.reaches.facc = np.append(self.reaches.facc, self.reaches.facc[rch])
                        self.reaches.iceflag = np.append(self.reaches.iceflag, self.reaches.iceflag[:,rch], axis=1)
                        self.reaches.max_wth = np.append(self.reaches.max_wth, self.reaches.max_wth[rch])
                        self.reaches.river_name = np.append(self.reaches.river_name, self.reaches.river_name[rch])
                        self.reaches.low_slope = np.append(self.reaches.low_slope, self.reaches.low_slope[rch])
                        self.reaches.trib_flag = np.append(self.reaches.trib_flag, self.reaches.trib_flag[rch])
                        self.reaches.path_freq = np.append(self.reaches.path_freq, self.reaches.path_freq[rch])
                        self.reaches.path_order = np.append(self.reaches.path_order, self.reaches.path_order[rch])
                        self.reaches.main_side = np.append(self.reaches.main_side, self.reaches.main_side[rch])
                        self.reaches.path_segs = np.append(self.reaches.path_segs, self.reaches.path_segs[rch])
                        self.reaches.strm_order = np.append(self.reaches.strm_order, self.reaches.strm_order[rch])
                        self.reaches.end_rch = np.append(self.reaches.end_rch, self.reaches.end_rch[rch])
                        self.reaches.network = np.append(self.reaches.network, self.reaches.network[rch])
                        self.reaches.add_flag = np.append(self.reaches.add_flag, self.reaches.add_flag[rch])
                        if self.reaches.edit_flag[rch] == 'NaN':
                            edit_val = '6'
                        elif '6' not in self.reaches.edit_flag[rch][0].split(','):
                            edit_val = self.reaches.edit_flag[rch] + ',6'
                        else:
                            edit_val = self.reaches.edit_flag[rch]
                        self.reaches.edit_flag = np.append(self.reaches.edit_flag, edit_val)
                        self.nodes.edit_flag[old_ind] = edit_val
                
                ### TOPOLOGY Updates 
                nrchs = np.unique(self.centerlines.reach_id[0,cl_r[order_ids]])
                max_id = [max(self.centerlines.cl_id[cl_r[order_ids[np.where(self.centerlines.reach_id[0,cl_r[order_ids]] 
                                                                             == n)[0]]]]) for n in nrchs]
                id_sort = np.argsort(max_id)
                nrchs = nrchs[id_sort]
                #need to order nrchs in terms of indexes can update dist out easier? 
                for idx in list(range(len(nrchs))):
                    pts = np.where(self.centerlines.reach_id[0,cl_r[order_ids]] == nrchs[idx])[0]
                    binary = np.copy(self.centerlines.reach_id[1:,cl_r[order_ids[pts]]])
                    binary[np.where(binary > 0)] = 1
                    binary_sum = np.sum(binary, axis = 0)
                    existing_nghs = np.where(binary_sum > 0)[0]
                    if len(existing_nghs) > 0:
                        mn = np.where(self.centerlines.cl_id[cl_r[order_ids[pts]]] == 
                                      min(self.centerlines.cl_id[cl_r[order_ids[pts]]]))[0]
                        mx = np.where(self.centerlines.cl_id[cl_r[order_ids[pts]]] == 
                                      max(self.centerlines.cl_id[cl_r[order_ids[pts]]]))[0]
                        if mn in existing_nghs and mx not in existing_nghs:
                            #updating new neighbors at the centerline level. 
                            self.centerlines.reach_id[1:,cl_r[order_ids[pts[mx]]]] = 0
                            self.centerlines.reach_id[1:,cl_r[order_ids[pts[mx]+1]]] = 0 
                            self.centerlines.reach_id[1,cl_r[order_ids[pts[mx]]]] = self.centerlines.reach_id[0,cl_r[order_ids[pts[mx]+1]]][0] #self.centerlines.reach_id[:,cl_r[order_ids[pts[mx]]]]
                            self.centerlines.reach_id[1,cl_r[order_ids[pts[mx]+1]]] = self.centerlines.reach_id[0,cl_r[order_ids[pts[mx]]]][0] #self.centerlines.reach_id[:,cl_r[order_ids[pts[mx]+1]]]
                            #updating new neighbors at the reach level.
                            ridx = np.where(self.reaches.id == self.centerlines.reach_id[0,cl_r[order_ids[pts[mx]]]][0])[0]
                            self.reaches.n_rch_up[ridx] = 1
                            self.reaches.rch_id_up[:,ridx] = 0
                            self.reaches.rch_id_up[0,ridx] = self.centerlines.reach_id[0,cl_r[order_ids[pts[mx]+1]]][0]
                            if idx > 0:
                                #upstream neighor
                                ridx2 = np.where(self.reaches.id == self.centerlines.reach_id[0,cl_r[order_ids[pts[mx]+1]]][0])[0]
                                self.reaches.n_rch_down[ridx2] = 1
                                self.reaches.rch_id_down[:,ridx2] = 0
                                self.reaches.rch_id_down[0,ridx2] = self.centerlines.reach_id[0,cl_r[order_ids[pts[mx]]]][0]
                                #current reach 
                                self.reaches.n_rch_down[ridx] = 1
                                self.reaches.rch_id_down[:,ridx] = 0
                                self.reaches.rch_id_down[0,ridx] = self.centerlines.reach_id[0,cl_r[order_ids[pts[mn]-1]]][0]

                        elif mx in existing_nghs and mn not in existing_nghs:
                            self.centerlines.reach_id[1:,cl_r[order_ids[pts[mn]]]] = 0
                            self.centerlines.reach_id[1:,cl_r[order_ids[pts[mn]-1]]] = 0
                            self.centerlines.reach_id[1,cl_r[order_ids[pts[mn]]]] = self.centerlines.reach_id[0,cl_r[order_ids[pts[mn]-1]]][0] #self.centerlines.reach_id[:,cl_r[order_ids[pts[mx]]]]
                            self.centerlines.reach_id[1,cl_r[order_ids[pts[mn]-1]]] = self.centerlines.reach_id[0,cl_r[order_ids[pts[mn]]]][0] #self.centerlines.reach_id[:,cl_r[order_ids[pts[mx]+1]]]
                            #updating new neighbors at the reach level.
                            ridx = np.where(self.reaches.id == self.centerlines.reach_id[0,cl_r[order_ids[pts[mn]]]][0])[0]
                            self.reaches.n_rch_down[ridx] = 1
                            self.reaches.rch_id_down[:,ridx] = 0
                            self.reaches.rch_id_down[0,ridx] = self.centerlines.reach_id[0,cl_r[order_ids[pts[mn]-1]]][0]
                            if idx > 0:
                                #upstream neighbor
                                ridx2 = np.where(self.reaches.id == self.centerlines.reach_id[0,cl_r[order_ids[pts[mn]-1]]][0])[0]
                                self.reaches.n_rch_up[ridx2] = 1
                                self.reaches.rch_id_up[:,ridx2] = 0
                                self.reaches.rch_id_up[0,ridx2] = self.centerlines.reach_id[0,cl_r[order_ids[pts[mn]]]][0]
                                #current reach 
                                self.reaches.n_rch_up[ridx] = 1
                                self.reaches.rch_id_up[:,ridx] = 0
                                self.reaches.rch_id_up[0,ridx] = self.centerlines.reach_id[0,cl_r[order_ids[pts[mx]+1]]][0]
                        
                        else:
                            #update downstream end for reach level. 
                            ridx = np.where(self.reaches.id == self.centerlines.reach_id[0,cl_r[order_ids[pts[mn]]]][0])[0] 
                            self.reaches.n_rch_down[ridx] = 1
                            self.reaches.rch_id_down[:,ridx] = 0
                            self.reaches.rch_id_down[0,ridx] = self.centerlines.reach_id[0,cl_r[order_ids[pts[mn]-1]]][0]
                            #find the max id and change that reaches values to current reach...
                            up_nghs = np.unique(self.centerlines.reach_id[1:,cl_r[order_ids[pts[mx]]]])
                            up_nghs = up_nghs[up_nghs>0]
                            for up in list(range(len(up_nghs))):
                                #updating upstream most neighbor of original reach's neighbors at the centerline level.
                                ngh_rch = np.where(self.centerlines.reach_id[0,:] == up_nghs[up])[0]
                                vals = np.where(self.centerlines.reach_id[1:,ngh_rch] == nrchs[0])
                                self.centerlines.reach_id[vals[0]+1,ngh_rch[vals[1]]] = self.centerlines.reach_id[0,cl_r[order_ids[pts[mx]]]][0]
                                #updating upstream most neighbor of original reach's neighbors at the reach level. 
                                ridx = np.where(self.reaches.id == up_nghs[up])[0]
                                nridx = np.where(self.reaches.rch_id_down[:,ridx] == nrchs[0])[0]
                                self.reaches.rch_id_down[nridx,ridx] = self.centerlines.reach_id[0,cl_r[order_ids[pts[mx]]]][0]
                #Distance from Outlet 
                rch_indx = np.where(np.in1d(self.reaches.id,nrchs)==True)[0]
                rch_cs = np.cumsum(self.reaches.len[rch_indx])
                self.reaches.dist_out[rch_indx] = rch_cs+base_val

        ### Print Dimensions.
        if verbose == True: 
            print('Cl Dimensions:', 
                len(np.unique(self.centerlines.cl_id)), 
                len(self.centerlines.cl_id))
            print('Rch Dimensions:', 
                len(np.unique(self.centerlines.reach_id[0,:])), 
                len(np.unique(self.nodes.reach_id)), 
                len(np.unique(self.reaches.id)),
                len(self.reaches.id))
            print('Node Dimensions:', 
                len(np.unique(self.centerlines.node_id[0,:])), 
                len(np.unique(self.nodes.id)), 
                len(self.nodes.id))

    ###############################################################################