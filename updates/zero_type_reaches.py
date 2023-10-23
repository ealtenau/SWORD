from __future__ import division
import numpy as np
import time
import netCDF4 as nc
import pandas as pd

region = 'OC'
version = 'v16'
sword_dir = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'

sword = nc.Dataset(sword_dir)
reaches = sword.groups['reaches'].variables['reach_id'][:]
type = np.array([int(str(rch)[-1]) for rch in reaches])
basin = np.array([int(str(rch)[0:6]) for rch in reaches])
number = np.array([int(str(rch)[6:10]) for rch in reaches])

z = np.where(type == 0)[0]
new_rch = []
for ind in list(range(len(z))):
    check_num = number[np.where(basin == basin[z[ind]])[0]]
    if 10 in check_num:
        new_num=np.max(check_num)+1
    else:
        new_num=10

    if len(str(new_num)) == 1:
        new_num = str('000')+str(new_num)
    if len(str(new_num)) == 2:
        new_num = str('00')+str(new_num)
    if len(str(new_num)) == 3:
        new_num = str('0')+str(new_num)
    if len(str(new_num)) == 4:
        new_num = str(new_num)
    type = str(1)
    new_rch.append(int(str(basin[z[ind]])+str(new_num)+str(type)))


df = pd.DataFrame(np.array([reaches[z], new_rch]).T)
df.rename(
    columns={
        0:"old_rch_id",
        1:"new_rch_id",
        },inplace=True)

df.to_csv('/Users/ealteanau/Documents/SWORD_Dev/update_requests/v16/'+region+'_rch_id_changes.csv', index=False)
print('Done')
