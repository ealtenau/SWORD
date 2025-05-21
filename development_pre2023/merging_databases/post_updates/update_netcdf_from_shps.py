# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:05:24 2021

@author: ealtenau
"""

from __future__ import division
import os
os.chdir('/Users/ealteanau/Documents/SWORD_Dev/src/SWORD/merging_databases/')
import Merge_Tools_v06 as mgt
import time
#import geopandas as gp
#import numpy as np
#import glob
#import matplotlib.pyplot as plt 

###############################################################################
###############################################################################
###############################################################################

start_all = time.time()

region = 'OC'
grwl_dir = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Merged_Data/v13/' + region + '/'
grwl_paths = [file for file in mgt.getListOfFiles(grwl_dir) if '.shp' in file]
nc_file = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Merged_Data/v13/' + region + '_Merge_v13.nc'
out_dir = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Merged_Data/v13/' + region + '/'
    
merged = mgt.Object()
cnt = 0
for ind in list(range(len(grwl_paths))):
    
    start = time.time()
    
    fn_grwl = grwl_paths[ind]
    outpath =  out_dir + fn_grwl[-17:-10] + '_merge.shp'
    grwl = mgt.open_merge_shp(fn_grwl)        
    
    if len(grwl.lon) == 0:
        print(fn_grwl[-17:-10] + ": No GRWL Data - Skipped")
        continue         
      
    # Combining current data with previous data.
    mgt.combine_vals(merged, grwl, cnt)
    cnt = cnt+1
    
    # Writing individual shapefiles.
    #mgt.save_merge_shp(grwl, outpath)
    end = time.time()
    print(ind, 'Runtime: ' + str((end-start)/60) + ' min: ' + outpath)

###############################################################################

# merged_copy = merged

# Filter Data.           
start = time.time()
mgt.format_data(merged) 
merged.lake_id = merged.lake_id.astype(int)
end = time.time()
print('Time to Filter Combined Data: ' + str((end-start)/60) + ' min')

# Save filtered data as netcdf file. 
start = time.time()
mgt.save_merged_nc(merged, nc_file) 
end = time.time()
print('Time to Write NetCDF: ' + str((end-start)/60) + ' min')

end_all = time.time()
print('Total Runtime: ' + str((end_all-start_all)/60) + ' min: ' + nc_file)

# test = np.array([int(float(str(ind)[0:6])) for ind in merged.basins])
# len(np.unique(test))
# len(np.unique(merged.new_basins))
# test2 = merged.new_basins - test
