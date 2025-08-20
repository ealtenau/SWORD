# -*- coding: utf-8 -*-
"""
SWORD Reach Class (sword_reach.py)
=====================================
This module defines the `Reach` class, which serves as a high-level API 
for interacting with a single river reach from the SWORD database.

The class connects to the consolidated DuckDB database and, upon 
initialization with a `reach_id`, loads all associated data for that 
reach. It acts as an abstraction layer, querying the normalized database
schema but presenting the data in objects that mirror the logical structure
of the original NetCDF files for downstream compatibility.
"""

import duckdb
import pandas as pd
import numpy as np
from typing import Optional

from src.updates.sword_models import ReachData, NodeData, CenterlineData

class Reach:
    """
    A class to represent and interact with a single SWORD reach.
    """
    def __init__(self, reach_id: int, db_path: str):
        """
        Initializes the Reach object by connecting to the database and loading
        all data associated with the given reach_id.
        """
        self.reach_id = reach_id
        self.db_path = db_path
        
        self.reach: Optional[ReachData] = None
        self.nodes: Optional[NodeData] = None
        self.centerline: Optional[CenterlineData] = None

        self._load_data()

    def _load_data(self):
        """
        Connects to the DuckDB database and executes queries to fetch all
        data for the reach, its nodes, and its centerline. It then assembles
        the normalized data into objects that match the original NetCDF structure.
        """
        try:
            con = duckdb.connect(database=self.db_path, read_only=True)

            # 1. Fetch and assemble reach data
            self._load_reach_data(con)

            # 2. Fetch and assemble node data
            self._load_node_data(con)
            
            # 3. Fetch and assemble centerline data
            self._load_centerline_data(con)

        finally:
            if 'con' in locals() and con:
                con.close()
    
    def _load_reach_data(self, con):
        """Loads and assembles the core reach attributes."""
        reach_query = f"SELECT * FROM reaches WHERE reach_id = {self.reach_id}"
        reach_df = con.execute(reach_query).fetchdf()
        if reach_df.empty:
            raise ValueError(f"No reach found with reach_id: {self.reach_id}")
        
        reach_record = reach_df.to_dict('records')[0]
        
        # Assemble upstream and downstream reach IDs from normalized columns
        rch_id_up = [reach_record[f'rch_id_up_{i}'] for i in range(1, 5) if reach_record.get(f'rch_id_up_{i}')]
        rch_id_down = [reach_record[f'rch_id_dn_{i}'] for i in range(1, 5) if reach_record.get(f'rch_id_dn_{i}')]
        
        self.reach = ReachData(
            reach_id=reach_record['reach_id'],
            grod_id=reach_record['grod_id'],
            hfalls_id=reach_record['hfalls_id'],
            n_nodes=reach_record['n_nodes'],
            width=reach_record['width'],
            wse=reach_record['wse'],
            wse_var=reach_record['wse_var'],
            reach_length=reach_record['reach_length'],
            facc=reach_record['facc'],
            dist_out=reach_record['dist_out'],
            x=reach_record['x'],
            y=reach_record['y'],
            river_name=reach_record['river_name'],
            continent=reach_record['continent'],
            x_min=reach_record['x_min'],
            x_max=reach_record['x_max'],
            y_min=reach_record['y_min'],
            y_max=reach_record['y_max'],
            stream_order=reach_record['stream_order'],
            rch_id_up=rch_id_up,
            rch_id_down=rch_id_down
        )
        # For compatibility, try to populate older field names if they exist
        if 'type' in reach_record: self.reach.type = reach_record['type']
        if 'slope' in reach_record: self.reach.slope = reach_record['slope']
        if 'slope_var' in reach_record: self.reach.slope_var = reach_record['slope_var']
        if 'lake_id' in reach_record: self.reach.lake_id = reach_record['lake_id']

    def _load_node_data(self, con):
        """Loads and assembles node data."""
        node_query = f"SELECT * FROM nodes WHERE reach_id = {self.reach_id} ORDER BY node_id"
        node_df = con.execute(node_query).fetchdf()
        
        self.nodes = NodeData(
            node_id=node_df['node_id'].to_numpy(),
            reach_id=node_df['reach_id'].to_numpy(),
            cl_id_min=node_df['cl_id_min'].to_numpy(),
            cl_id_max=node_df['cl_id_max'].to_numpy(),
            wse=node_df['wse'].to_numpy(),
            wse_var=node_df['wse_var'].to_numpy(),
            width=node_df['width'].to_numpy(),
            width_var=node_df['width_var'].to_numpy(),
            len=node_df['node_length'].to_numpy(), # Map from new to old name
            facc=node_df['facc'].to_numpy(),
            x=node_df['x'].to_numpy(),
            y=node_df['y'].to_numpy(),
            continent=node_df['continent'].to_numpy(),
        )
        # For compatibility, populate optional fields if they exist in the data
        if 'iceflag' in node_df.columns:
            self.nodes.iceflag = node_df['iceflag'].to_numpy()
            
    def _load_centerline_data(self, con):
        """Loads and assembles centerline data."""
        # A reach can be associated with up to 4 centerline sets, check all
        cl_dfs = []
        for i in range(1, 5):
            cl_query = f"SELECT * FROM centerlines WHERE reach_id_{i} = {self.reach_id}"
            cl_df = con.execute(cl_query).fetchdf()
            if not cl_df.empty:
                cl_dfs.append(cl_df)
        
        if not cl_dfs:
            self.centerline = CenterlineData()
            return
            
        full_cl_df = pd.concat(cl_dfs, ignore_index=True).sort_values(by='cl_id').reset_index(drop=True)
        
        # Assemble node_id and rch_id from the four possible columns
        node_ids = full_cl_df[['node_id_1', 'node_id_2', 'node_id_3', 'node_id_4']].bfill(axis=1).iloc[:, 0]
        rch_ids = full_cl_df[['reach_id_1', 'reach_id_2', 'reach_id_3', 'reach_id_4']].bfill(axis=1).iloc[:, 0]
        
        self.centerline = CenterlineData(
            cl_id=full_cl_df['cl_id'].to_numpy(),
            x=full_cl_df['x'].to_numpy(),
            y=full_cl_df['y'].to_numpy(),
            node_id=node_ids.to_numpy(),
            rch_id=rch_ids.to_numpy(),
            continent=full_cl_df['continent'].to_numpy()
        ) 