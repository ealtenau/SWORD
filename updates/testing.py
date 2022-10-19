import netCDF4 as nc
import numpy as np
import time

start = time.time()
regions=['NA','AS','SA','EU','AF','OC']

for ind in list(range(len(regions))):
    print('Starting Region: ' + regions[ind])
    sword13 = nc.Dataset('/Users/ealteanau/Documents/SWORD_Dev/outputs/v13/netcdf/'+regions[ind].lower()+'_sword_v13.nc')
    sword14 = nc.Dataset('/Users/ealteanau/Documents/SWORD_Dev/outputs/v14/netcdf/'+regions[ind].lower()+'_sword_v14.nc', 'r+')

    high_rch = np.where(sword13.groups['reaches'].variables['width_var'][:] > 100000)[0]
    high_nodes = np.where(sword13.groups['nodes'].variables['width_var'][:] > 100000)[0]

    rch_diff = np.unique(sword14.groups['reaches'].variables['reach_id'][high_rch] - sword13.groups['reaches'].variables['reach_id'][high_rch])
    node_diff = np.unique(sword14.groups['nodes'].variables['node_id'][high_nodes] - sword13.groups['nodes'].variables['node_id'][high_nodes])

    sword14.groups['reaches'].variables['width_var'][high_rch] = sword13.groups['reaches'].variables['width_var'][high_rch]
    sword14.groups['nodes'].variables['width_var'][high_nodes] = sword13.groups['nodes'].variables['width_var'][high_nodes]

    sword13.close()
    sword14.close()

end = time.time()
print('Finished Width Variability Updates in: ' + str(np.round((end-start)/60, 2)) + ' min')

sword13.groups['reaches'].variables['reach_id'][:].shape
sword13.groups['reaches'].variables['width_var'][:].shape

sword14.groups['reaches'].variables['reach_id'][:].shape