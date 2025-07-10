# -*- coding: utf-8 -*-
"""
Writing SWORD Vector Files (sword_vectors.py)
=====================
Script for creating the reach and node geopackage and 
shapefile vectors from the netCDF master file of
the SWOT River Database (SWORD). 

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA), SWORD version (i.e. v18), 
and the data you would like to export ('All', 'reaches', 
or 'nodes').

Execution example (terminal):
    python sword_vectors.py NA v18 reaches

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import argparse
import time
import numpy as np
from src.updates.sword import SWORD

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("export", help="Keyword for the vector data to write: 'All', 'reaches', or 'nodes'", type = str)
args = parser.parse_args()

region = args.region
version = args.version
export = args.export

#reading netCDF and writing vector data.
start = time.time()
sword = SWORD(main_dir, region, version)
sword.save_vectors(export)
end = time.time()
print('Time to Write Vectors:', str(np.round((end-start)/60, 2)), 'mins')
