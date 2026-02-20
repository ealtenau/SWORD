# -*- coding: utf-8 -*-
"""
Delete Reaches (delete_reaches_post_topo.py)
==============================================

This script deletes data associated with specifed 
Reach IDs in the SWOT River Database (SWORD).

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA), SWORD version (i.e. v17), 
and a csv file containing Reach IDs to delete.

Execution example (terminal):
    python path/to/delete_reaches_post_topo.py NA v17 path/to/reach_deletions.csv

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import pandas as pd
import argparse
from src.updates.sword_duckdb import SWORD

parser = argparse.ArgumentParser()
parser.add_argument("region", help="continental region", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("csv", help="csv file of reaches to delete", type = str)
args = parser.parse_args()

region = args.region
version = args.version

#read sword data.
db_path = os.path.join(main_dir, f'data/duckdb/sword_{version}.duckdb')
sword = SWORD(db_path, region, version)
sword.copy() #copies original file for version control.
rch_check = sword.reaches.id

#read csv data. 
rch_dir = args.csv
rm_rch_df = pd.read_csv(rch_dir)
rm_rch = np.array(rm_rch_df['reach_id']) #csv file
rm_rch = np.unique(rm_rch)

#manually specify reaches if desired. 
# rm_rch = np.array([34100026796]) #manual
# rm_rch = np.unique(rm_rch)

#delete reaches. 
sword.delete_data(rm_rch)

#write data. 
new_rch_num = len(rch_check) - len(rm_rch)
if len(sword.reaches.id) == new_rch_num:
    sword.save_nc()