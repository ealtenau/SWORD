"""
Attach Shift Flag to Reach Vector Files
(points_to_line_flag.py)
=========================================================

This script attaches the shift flag outputs from the 
point files generated in 'channel_shifting_flag_jrc.py'
or 'channel_shifting_flag_mhv.py' to the SWORD reach 
vector files. It also assigns different shift flag 
priority values based on width:
    0: No shift flag
    1: Flagged rivers > 100 m wide.
    2: Flagged rivers 50-100 m wide. 
    3: Flagged rivers < 50 m wide. 

Outputs are located at:
main_dir+'/data/inputs/JRC_Water_Occurance/'+region+'/'

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python path/to/points_to_line_flag.py NA v17

"""

import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import argparse
import geopandas as gp

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Region", type = str)
parser.add_argument("version", help="<Required> Version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

#data filenames. 
sword_dir = main_dir+'/data/outputs/Reaches_Nodes/'+version+'/gpkg/'\
    +region.lower()+'_sword_reaches_'+version+'.gpkg'
jrc_dir = main_dir+'/data/inputs/JRC_Water_Occurance/'+region+'/'
jrc_files = os.listdir(jrc_dir)
jrc_files = np.array([f for f in jrc_files if 'sword' in f])

#read data and attach to SWORD reach vector file. 
sword = gp.read_file(sword_dir)
flag = np.zeros(len(sword))
for f in list(range(len(jrc_files))):
    print(f,len(jrc_files)-1)
    subset = gp.read_file(jrc_dir+jrc_files[f])
    shift = np.where(subset['shift_flag'] == 1)[0]
    unq_rchs = np.unique(subset['reach_id'].iloc[shift])
    for r in list(range(len(unq_rchs))):
        # print(r)
        rch = np.where(sword['reach_id'] == unq_rchs[r])[0]
        if len(rch) == 0:
            continue
        wth = sword['width'].iloc[rch]
        # print(int(wth))
        if int(wth) >= 100:
            flag[rch] = 3
        elif 50 <= int(wth) < 100: 
            flag[rch] = 2
        else:
            flag[rch] = 1

#write data. 
sword['shift_flag'] = flag
sword.to_file(jrc_dir+region+'_shift_flag_'+version+'.gpkg', driver='GPKG', layer='reaches')

All = np.where(flag > 0)[0]
small = np.where(flag == 1)[0]
medium = np.where(flag == 2)[0]
large = np.where(flag == 3)[0]
print('shift flag %:',np.round(len(All)/len(flag)*100))
print('small rivers %:',np.round(len(small)/len(All)*100))
print('medium rivers %:',np.round(len(medium)/len(All)*100))
print('large rivers %:',np.round(len(large)/len(All)*100))
