# -*- coding: utf-8 -*-
"""
Writing SWORD Vector Files (sword_vectors.py)
=====================
Script for creating the reach and node geopackage and
shapefile vectors from the DuckDB database of
the SWOT River Database (SWORD).

The script is run at a regional/continental scale.
Command line arguments required are the two-letter
region identifier (i.e. NA), SWORD version (i.e. v17b),
and the data you would like to export ('All', 'reaches',
or 'nodes').

Note: This script uses the DuckDB backend. The database
file is expected at: data/duckdb/sword_{version}.duckdb

Execution example (terminal):
    python path/to/sword_vectors.py NA v17b reaches

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import argparse
import time
import numpy as np
from src.updates.sword_duckdb import SWORD

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("export", help="Keyword for the vector data to write: 'All', 'reaches', or 'nodes'", type = str)
args = parser.parse_args()

region = args.region
version = args.version
export = args.export

# Reading DuckDB and writing vector data.
start = time.time()
db_path = os.path.join(main_dir, f'data/duckdb/sword_{version}.duckdb')
sword = SWORD(db_path, region, version)
sword.save_vectors(export)
end = time.time()
print('Time to Write Vectors:', str(np.round((end-start)/60, 2)), 'mins')
