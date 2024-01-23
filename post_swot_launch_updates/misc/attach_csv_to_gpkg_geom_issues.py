import numpy as np
import pandas as pd
import geopandas as gp

gpkg_fn = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16/gpkg/na_sword_reaches_v16.gpkg'
csv_fn = '/Users/ealteanau/Documents/SWORD_Dev/testing_files/na_sword_reaches_v16_geom_test2_overlaps.csv'

gpkg = gp.read_file(gpkg_fn)
csv = pd.read_csv(csv_fn)

rchs = np.array(gpkg['reach_id'])

issues = np.where(csv['typ_int'] != 'Point')[0]
con = np.array(csv['typ_int'][issues])
r1 = np.array(csv['r1'][issues])
r2 = np.array(csv['r2'][issues])

matches = np.where(np.in1d(rchs, r1) == True)[0]
flag = np.zeros(len(rchs))
flag_fixed = np.zeros(len(rchs))
# flag_r2 = np.zeros(len(rchs))
# flag_type = np.zeros(len(rchs))

flag[matches] = 1
gpkg['geom_flag'] = flag
gpkg['geom_fix'] = flag_fixed

outfile = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16/gpkg/na_sword_reaches_v16_geom_flag.gpkg'
gpkg.to_file(outfile, driver='GPKG', layer='reaches')

