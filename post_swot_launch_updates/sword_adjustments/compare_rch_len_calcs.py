import numpy as np
import netCDF4 as nc
import geopandas as gp

region = 'SA'
version = 'v17'

fn_utm = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/length_comparisons/'\
    +region.lower()+'_sword_'+version+'_utm.nc'
fn_albers = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/length_comparisons/'\
    +region.lower()+'_sword_'+version+'_albers.nc'
fn_molli = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/length_comparisons/'\
    +region.lower()+'_sword_'+version+'_molliwald.nc'
fn_wink = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/length_comparisons/'\
    +region.lower()+'_sword_'+version+'_wink.nc'
fn_green = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/length_comparisons/'\
    +region.lower()+'_sword_'+version+'_green.nc'
fn_gc = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/length_comparisons/'\
    +region.lower()+'_sword_'+version+'_gc.nc'

utm_data = nc.Dataset(fn_utm)
albers_data = nc.Dataset(fn_albers)
molli_data = nc.Dataset(fn_molli)
wink_data = nc.Dataset(fn_wink)
green_data = nc.Dataset(fn_green)
gc_data = nc.Dataset(fn_gc)

utm = np.array(utm_data.groups['reaches'].variables['reach_length'][:])
albers = np.array(albers_data.groups['reaches'].variables['reach_length'][:])
molli = np.array(molli_data.groups['reaches'].variables['reach_length'][:])
wink = np.array(wink_data.groups['reaches'].variables['reach_length'][:])
green = np.array(green_data.groups['reaches'].variables['reach_length'][:])
gc = np.array(gc_data.groups['reaches'].variables['reach_length'][:])

diff_albers = (utm - albers)
abs_diff_albers = abs(diff_albers)
print(np.mean(abs_diff_albers), np.mean(diff_albers)) # absolute difference and difference metrics
print(np.median(abs_diff_albers), np.median(diff_albers))
print(np.std(abs_diff_albers), np.std(diff_albers))

diff_wink = utm - wink
abs_diff_wink = abs(diff_wink)
print(np.mean(abs_diff_wink), np.mean(diff_wink))
print(np.median(abs_diff_wink), np.median(diff_wink))
print(np.std(abs_diff_wink), np.std(diff_wink))

diff_molli = utm - molli
abs_diff_molli = abs(diff_molli)
print(np.mean(abs_diff_molli), np.mean(diff_molli))
print(np.median(abs_diff_molli), np.median(diff_molli))
print(np.std(abs_diff_molli), np.std(diff_molli))

diff_green = utm - green
abs_diff_green = abs(diff_green)
print(np.mean(abs_diff_green), np.mean(diff_green))
print(np.median(abs_diff_green), np.median(diff_green))
print(np.std(abs_diff_green), np.std(diff_green))

diff_gc = utm - gc
abs_diff_gc = abs(diff_gc)
print(np.mean(abs_diff_gc), np.mean(diff_gc))
print(np.median(abs_diff_gc), np.median(diff_gc))
print(np.std(abs_diff_gc), np.std(diff_gc))

'''

gpkg_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/gpkg/'+region.lower()+\
    '_sword_reaches_'+version+'.gpkg'
outgpkg = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/length_comparisons/'+region.lower()+\
    '_sword_reaches_'+version+'_projdiff.gpkg'
gpkg = gp.read_file(gpkg_fn)
max(np.array(gpkg['reach_id'])-np.array(green_data.groups['reaches'].variables['reach_id'][:]))

# keep_logic = np.in1d(np.array(green_data.groups['reaches'].variables['reach_id'][:]),np.array(gpkg['reach_id']))
# keep = np.where(keep_logic == True)[0]

gpkg['utm_gc_dif'] = diff_gc
gpkg['utm_gc_abs'] = abs_diff_gc
gpkg.to_file(outgpkg, driver='GPKG', layer='reaches')


print(np.mean(utm),np.mean(gc))
print(np.median(utm), np.median(gc))
print(max(utm),max(gc))
print(min(utm),min(gc))


'''