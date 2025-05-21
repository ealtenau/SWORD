import pandas as pd
import numpy as np
import netCDF4 as nc
from geopy import distance

region = 'OC'
version='v18'
nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
outdir = '/Users/ealtenau/Documents/SWORD_Dev/update_requests/'+version+'/'+region+'/'
sword = nc.Dataset(nc_fn)

n_rch_up = np.array(sword['/reaches/n_rch_up'][:])
n_rch_down = np.array(sword['/reaches/n_rch_down'][:])
rch_len = np.array(sword['/reaches/reach_length'][:])
rch_id = np.array(sword['/reaches/reach_id'][:])

solo = np.where((n_rch_up == 0)&(n_rch_down == 0))[0]
rmv = rch_id[solo]

rch_csv = pd.DataFrame({"reach_id": rmv})
rch_csv.to_csv(outdir+'solo_rch_deletions.csv', index = False)
print('Done')
