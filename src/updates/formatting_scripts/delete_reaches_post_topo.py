# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import pandas as pd
import argparse
from src.updates.sword import SWORD

parser = argparse.ArgumentParser()
parser.add_argument("region", help="continental region", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("csv", help="csv file of reaches to delete", type = str)
args = parser.parse_args()

region = args.region
version = args.version

#Line-by-line debugging.
# region = 'OC'
# version = 'v18'

sword = SWORD(main_dir, region, version)
rch_check = sword.reaches.id

rch_dir = args.csv
# rch_dir = paths['update_dir']+'solo_rch_deletions.csv' #manual 
rm_rch_df = pd.read_csv(rch_dir)
rm_rch = np.array(rm_rch_df['reach_id']) #csv file
rm_rch = np.unique(rm_rch)

# rm_rch = np.array([11600200243, 11600201666, 11600200293, 11600200303, 11600201656, 11710500031, 11710500011, 11710500286, 11710600011, 11710600416]) #manual
# rm_rch = np.unique(rm_rch)

#delete reaches. 
sword.delete_data(rm_rch)

#write data. 
new_rch_num = len(rch_check) - len(rm_rch)
if len(sword.reaches.id) == new_rch_num:
    sword.save_nc()