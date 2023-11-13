import netCDF4 as nc
import numpy as np
import time

start = time.time()
version = 'v14'
regions=['NA','AS','SA','EU','AF','OC']
sword_dir = '/Users/ealteanau/Documents/SWORD_Dev/outputs/'+version+'/netcdf/'

for ind in list(range(len(regions))):
    print('Starting Region: ' + regions[ind])
    sword = nc.Dataset(sword_dir+regions[ind].lower()+'_sword_'+version+'.nc', 'r+')

    high_rch = np.where(sword.groups['reaches'].variables['width_var'][:] > 100000)[0]
    high_nodes = np.where(sword.groups['nodes'].variables['width_var'][:] > 100000)[0]

    new_rch_wth_var = sword.groups['reaches'].variables['max_width'][high_rch]*2
    new_rch_wth_var[np.where(new_rch_wth_var > 100000)[0]] = 100000

    new_node_wth_var = sword.groups['nodes'].variables['max_width'][high_nodes]*2
    new_node_wth_var[np.where(new_node_wth_var > 100000)[0]] = 100000

    sword.groups['reaches'].variables['width_var'][high_rch] = new_rch_wth_var
    sword.groups['nodes'].variables['width_var'][high_nodes] = new_node_wth_var

    sword.close()

    # np.max(sword.groups['reaches'].variables['width_var'][high_rch][:])
    # np.max(sword.groups['nodes'].variables['width_var'][high_rch][:])

end = time.time()
print('Finished Width Variability Updates in: ' + str(np.round((end-start)/60, 2)) + ' min')