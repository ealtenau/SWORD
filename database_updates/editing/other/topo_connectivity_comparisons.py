from __future__ import division
import numpy as np
import time
import netCDF4 as nc
import pandas as pd
from scipy import spatial as sp
import utm 
import argparse
from pyproj import Proj
import geopy.distance

region = 'SA'
version = 'v17'

topo_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Topology/'+version+'/'+region+'/'
con_csv= topo_dir + region.lower()+'_sword_reaches_'+version+'_routing_con.csv'
compare_csv = '/Users/ealtenau/Documents/SWORD_Dev/update_requests/con_62.csv'

df1 = pd.read_csv(con_csv, header = None)
df2 = pd.read_csv(compare_csv, header = None)
update = []
for r in list(range(len(df2[0]))):
    rch = np.where(df1[0] == df2[0][r])[0]
    if len(rch) == 0:
        continue
    row2 = df2.iloc[r,:]
    row1 = df1.iloc[rch[0],:]
    identical = row1.equals(row2)
    if identical != True:
        update.append(r)


df3 = df2.iloc[update]
df3.to_csv('/Users/ealtenau/Documents/SWORD_Dev/update_requests/con_62_diff.csv', header=False, index=False)



