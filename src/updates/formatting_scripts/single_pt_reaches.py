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
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

sword = SWORD(main_dir, region, version)
outpath = sword.paths['update_dir']+region.lower()+'_'+version+'_single_pt_rchs.csv'

single_pt_rchs = []
for ind in list(range(len(sword.reaches.id))):
    print(ind, len(sword.reaches.id)-1)
    pts = np.where(sword.centerlines.reach_id[0,:] == sword.reaches.id[ind])[0]
    if len(pts) == 1:
        single_pt_rchs.append(ind)

#export reaches to delete.
rch_list = sword.reaches.id[single_pt_rchs]
df = pd.DataFrame(rch_list)
df.to_csv(outpath)
print(len(rch_list)) 