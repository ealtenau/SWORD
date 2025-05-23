import os
main_dir = os.getcwd()
import numpy as np
import geopandas as gp

region = 'EU'
sword_dir = main_dir+'/data/outputs/Reaches_Nodes/v17b/gpkg/'+region.lower()+'_sword_reaches_v17b.gpkg'
jrc_dir = main_dir+'/data/inputs/JRC_Water_Occurance/'+region+'/'
jrc_files = os.listdir(jrc_dir)
jrc_files = np.array([f for f in jrc_files if 'sword' in f])

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

sword['shift_flag'] = flag
sword.to_file(jrc_dir+region+'_shift_flag_v17b.gpkg', driver='GPKG', layer='reaches')

All = np.where(flag > 0)[0]
small = np.where(flag == 1)[0]
medium = np.where(flag == 2)[0]
large = np.where(flag == 3)[0]
print('shift flag %:',np.round(len(All)/len(flag)*100))
print('small rivers %:',np.round(len(small)/len(All)*100))
print('medium rivers %:',np.round(len(medium)/len(All)*100))
print('large rivers %:',np.round(len(large)/len(All)*100))
