import os 
main_dir = os.getcwd() 
import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("basin", help="<Required> Level Two Pfafstetter Basin (i.e. 74)", type = str)
args = parser.parse_args()

b = args.basin
region = args.region
file_fn = main_dir+'/data/outputs/Topology/'+region+'/b'+b+'/intermediate/gap_rchs.csv'
sword_fn = main_dir+'/data/outputs/Reaches_Nodes/v17/netcdf/'+region.lower()+'_sword_v17.nc'

gaps = pd.read_csv(file_fn)
sword = nc.Dataset(sword_fn)

end_rch = np.array(sword['/reaches/end_reach'][:])
rchs = np.array(sword['/reaches/reach_id'][:])

ends = rchs[np.where((end_rch > 0) & (end_rch < 3))[0]]
rmv = np.where(np.in1d(gaps['reach_id'], ends)== True)[0]
gaps = gaps.drop(rmv)

gaps.to_csv(file_fn,index=False)
print('gap reaches removed:', len(rmv))