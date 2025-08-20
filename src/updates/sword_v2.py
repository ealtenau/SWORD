# -*- coding: utf-8 -*-
"""
SWORD_v2 (sword_v2.py)
=====================
A modernized class for interacting with the SWOT River Database (SWORD),
designed to work with a centralized DuckDB database instead of direct
NetCDF file manipulation.

"""
from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import src.updates.sword_utils_v2 as swd_v2

class SWORD_v2:
    """
    The SWORD_v2 class provides a high-level API for SWORD, orchestrating
    workflows like vector file generation directly from the database.
    """

    def __init__(self, main_dir, region, version, db_path='data/duckdb/sword_global.duckdb'):
        """
        Initializes the SWORD_v2 class.

        This is a lightweight constructor that sets up paths and database
        connections without loading data into memory.

        Parameters
        ----------
        main_dir: str
            The main project directory.
        region: str
            Two-letter acronym for a SWORD region (e.g., 'NA').
        version: str
            SWORD version string (e.g., 'v19').
        db_path: str
            Path to the DuckDB database file.
        """
        self.region = region
        self.version = version
        self.db_path = db_path
        self.paths = swd_v2.prepare_paths(main_dir, region, version)

    def save_vectors(self, export='All', max_dist=150):
        """
        Saves SWORD data in vector formats (shapefile and geopackage).

        This method calls the modernized, database-aware utility functions.

        Parameters
        ----------
        export: str
            'All' - writes both reach and node files.
            'nodes' - writes node files only.
            'reaches' - writes reach files only.
        max_dist: int, optional
            Distance threshold in meters for geometry stitching logic.
        """
        if export in ('All', 'reaches'):
            print('--- EXPORTING REACHES ---')
            swd_v2.write_rchs(
                self.paths, self.db_path, max_dist, self.region
            )

        if export in ('All', 'nodes'):
            print('--- EXPORTING NODES ---')
            swd_v2.write_nodes(self.paths, self.db_path) 