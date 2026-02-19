# -*- coding: utf-8 -*-
"""
Correct Node X-Y locations (correct_node_xy.py)
===============================================

This script re-calculates node x-y values 
in the SWOT River Database (SWORD).

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17)
and a csv file containing reaches to update and new basin
codes.

Execution example (terminal):
    python path/to/correct_node_xy.py NA v17

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import time
import argparse
import src.updates.sword_utils as swd
from src.updates.sword_duckdb import SWORD

###############################################################################
###############################################################################
###############################################################################

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="continental region", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

#read data
db_path = os.path.join(main_dir, f'data/duckdb/sword_{version}.duckdb')
sword = SWORD(db_path, region, version)

#recalculating node x-y values. 
for n in list(range(len(sword.nodes.id))):
    print(n, len(sword.nodes.id)-1)
    pts = np.where(sword.centerlines.node_id[0,:] == sword.nodes.id[n])[0]
    sword.nodes.x[n] = np.median(sword.centerlines.x[pts])
    sword.nodes.y[n] = np.median(sword.centerlines.y[pts])

#write data
sword.save_nc()

end = time.time()
print('DONE IN:', str(np.round((end-start)/60, 2)), 'mins')