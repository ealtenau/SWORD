import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import geopandas as gp
import time
import matplotlib.pyplot as plt
import argparse 

start_all = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("continent", help="sword continental region", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("region", help="glows region", type = str)
parser.add_argument("local_processing", help="'True' for local machine, 'False' for server", type = str)
parser.add_argument("subset", help="'True' for subsetting data, 'False' for all data", type = str)
parser.add_argument("basin", help="basin for subsetting", type = str)
args = parser.parse_args()

region = args.continent
version = args.version
glows_region = args.region
crop = args.subset
basin_id = args.basin

region = 'NA'
version = 'v17_glows'
glows_region = '7'
crop = 'True'
basin_id = '76'

if args.local_processing == 'True':
    sworddir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'
    glows_data_dir = '/Users/ealtenau/Documents/SWORD_Dev/inputs/GLOW-S/'
else:
    sworddir = '/afs/cas.unc.edu/depts/geological_sciences/pavelsky/students/ealtenau/SWORD_dev/outputs/Reaches_Nodes/'
    glows_data_dir = '/afs/cas.unc.edu/depts/geological_sciences/pavelsky/students/ealtenau/SWORD_dev/inputs/GLOW-S/'

swordpath = sworddir+version+'/'
sword_dir = swordpath+'netcdf/'+region.lower()+'_sword_'+version+'.nc'
wth_data_dir = glows_data_dir + 'GLOW-S_regions_merged/GLOW-S_region_'+glows_region+'_daywidth.parquet'
# wth_node_dir = glows_data_dir + 'GLOW-S_crosssection_points/GLOW-S_region_'+glows_region+'.shp'

print('Reading in SWORD Data')
start = time.time()
sword = nc.Dataset(sword_dir,'r+')
# nlon = np.array(sword.groups['nodes'].variables['x'][:])
# nlat = np.array(sword.groups['nodes'].variables['y'][:])
nid = np.array(sword.groups['nodes'].variables['node_id'][:])
rid = np.array(sword.groups['reaches'].variables['reach_id'][:])
rlon = np.array(sword.groups['reaches'].variables['x'][:])
rlat = np.array(sword.groups['reaches'].variables['y'][:])
nrid = np.array(sword.groups['nodes'].variables['reach_id'][:])
glows_ids = np.array(sword.groups['nodes'].variables['glows_river_id'][:]) #np.unique(glows_ids)
if 'glows_wth_1sig' in sword.groups['nodes'].variables.keys():
    node_glows_sig = np.array(sword.groups['nodes'].variables['glows_wth_1sig'][:])
    node_glows_wth_min = np.array(sword.groups['nodes'].variables['glows_wth_min'][:])
    node_glows_wth_med = np.array(sword.groups['nodes'].variables['glows_wth_med'][:])
    node_glow_wth_max = np.array(sword.groups['nodes'].variables['glows_wth_max'][:])
    node_glow_wth_nobs = np.array(sword.groups['nodes'].variables['glows_wth_nobs'][:])
    rch_glows_sig = np.array(sword.groups['reaches'].variables['glows_wth_1sig'][:])
    rch_glows_wth_min = np.array(sword.groups['reaches'].variables['glows_wth_min'][:])
    rch_glows_wth_med = np.array(sword.groups['reaches'].variables['glows_wth_med'][:])
    rch_glow_wth_max = np.array(sword.groups['reaches'].variables['glows_wth_max'][:])
    rch_glow_wth_nobs = np.array(sword.groups['reaches'].variables['glows_wth_nobs'][:])
else:
    node_glows_sig = np.repeat(-9999, len(nid))
    node_glows_wth_min = np.repeat(-9999, len(nid))
    node_glows_wth_med = np.repeat(-9999, len(nid))
    node_glow_wth_max = np.repeat(-9999, len(nid))
    node_glow_wth_nobs = np.repeat(-9999, len(nid))
    rch_glows_sig = np.repeat(-9999, len(rid))
    rch_glows_wth_min = np.repeat(-9999, len(rid))
    rch_glows_wth_med = np.repeat(-9999, len(rid))
    rch_glow_wth_max = np.repeat(-9999, len(rid))
    rch_glow_wth_nobs = np.repeat(-9999, len(rid))
end = time.time()
print(str(np.round((end-start),2))+' sec')

print('Reading in Width Parquet File')
start = time.time()
df = pd.read_parquet(wth_data_dir)
node = np.array(df['crossSxnID'])
wth = np.array(df['width'])
end = time.time()
print(str(np.round((end-start),2))+' sec')

# print('Reading in Width Cross Section Shp File')
# start = time.time()
# shp = gp.read_file(wth_node_dir)
# wlon = np.array(shp['lon'])
# wlat = np.array(shp['lat'])
# wid = np.array(shp['riverID'])
# end = time.time()
# print(str(np.round((end-start)/60,2))+' min')

if crop == 'True':
    print('Dividing into L2 Basins')
    start = time.time()
    sword_basin = np.array([int(str(ind)[0:len(basin_id)]) for ind in nid])
    wth_l2 = np.array([int(ind[1:3]) for ind in node])
    end = time.time()
    print(str(np.round((end-start),2))+' sec')

    #### basin loop???
    basin = np.where(sword_basin == int(basin_id))[0]
    null = np.where((glows_ids[basin] != 'NaN') & (node_glows_wth_med[basin] == -9999))[0]
    unq_ids = np.unique(glows_ids[basin[null]])
    # unq_ids = unq_ids[unq_ids != 'NaN']

    subset = np.where(wth_l2 == int(basin_id[0:2]))[0]
    node_sub = node[subset]
    wth_sub = wth[subset]
    wth_min = np.copy(node_glows_wth_min[basin])
    wth_max = np.copy(node_glow_wth_max[basin])
    wth_median = np.copy(node_glows_wth_med[basin])
    wth_1sigma = np.copy(node_glows_sig[basin])
    wth_nobs = np.copy(node_glow_wth_nobs[basin])
    glows_ids_sub = np.copy(glows_ids[basin])

else:
    node_sub = np.copy(node)
    wth_sub = np.copy(wth)
    null = np.where((glows_ids != 'NaN') & (node_glows_wth_med == -9999))[0]
    unq_ids = np.unique(glows_ids[null])
    wth_min = np.copy(node_glows_wth_min)
    wth_max = np.copy(node_glow_wth_max)
    wth_median = np.copy(node_glows_wth_med)
    wth_1sigma = np.copy(node_glows_sig)
    wth_nobs = np.copy(node_glow_wth_nobs)
    glows_ids_sub = np.copy(glows_ids)
   
print('Starting Nodes')
start = time.time()
for n in list(range(len(unq_ids))):
    print(n, len(unq_ids)-1)
    inds = np.where(node_sub == unq_ids[n])[0]
    if len(inds) == 0:
        continue
    fill = np.where(glows_ids_sub == unq_ids[n])[0]
    wth_min[fill] = np.min(wth_sub[inds])
    wth_max[fill] = np.max(wth_sub[inds])
    wth_median[fill] = np.median(wth_sub[inds])
    wth_1sigma[fill] = np.std(wth_sub[inds])
    wth_nobs[fill] = len(fill)
end = time.time()
print(str(np.round((end-start)/3600,2))+' hrs')

### end basin loop 

print('Starting Reaches')
start = time.time()
rch_wth_min = np.copy(rch_glows_wth_min)
rch_wth_max = np.copy(rch_glow_wth_max)
rch_wth_median = np.copy(rch_glows_wth_med)
rch_wth_1sigma = np.copy(rch_glows_sig)
data = np.where(wth_median > -9999)[0]
unq_rchs = np.unique(nrid[data])
for r in list(range(len(unq_rchs))):
    print(r, len(unq_rchs)-1)
    nind = np.where(nrid == unq_rchs[r])[0]
    rind = np.where(rid == unq_rchs[r])[0]
    good_data = np.where(wth_median[nind]>-9999)[0]
    prc = len(good_data)/len(wth_median[nind])*100
    if prc >= 50:
        rch_wth_min[rind] = np.min(wth_min[nind[good_data]])
        rch_wth_max[rind] = np.max(wth_max[nind[good_data]])
        rch_wth_median[rind] = np.median(wth_median[nind[good_data]])
        rch_wth_1sigma[rind] = np.std(wth_1sigma[nind[good_data]])
end = time.time()
print(str(np.round((end-start)/60,2))+' min')

print('Updating NetCDF')
start = time.time()
if 'glows_wth_med' in sword.groups['nodes'].variables.keys():
    #nodes
    node_keep = np.where(wth_median > -9999)[0]
    sword.groups['nodes'].variables['glows_wth_med'][node_keep] = wth_median[node_keep]
    sword.groups['nodes'].variables['glows_wth_min'][node_keep] = wth_min[node_keep]
    sword.groups['nodes'].variables['glows_wth_max'][node_keep] = wth_max[node_keep]
    sword.groups['nodes'].variables['glows_wth_1sig'][node_keep] = wth_1sigma[node_keep]
    #reaches
    rch_keep = np.where(rch_wth_median > -9999)[0]
    sword.groups['reaches'].variables['glows_wth_med'][rch_keep] = rch_wth_median[rch_keep]
    sword.groups['reaches'].variables['glows_wth_min'][rch_keep] = rch_wth_min[rch_keep]
    sword.groups['reaches'].variables['glows_wth_max'][rch_keep] = rch_wth_max[rch_keep]
    sword.groups['reaches'].variables['glows_wth_1sig'][rch_keep] = rch_wth_1sigma[rch_keep]
    sword.close()
else:
    #nodes
    sword.groups['nodes'].createVariable('glows_wth_med', 'f8', ('num_nodes',), fill_value=-9999.)
    sword.groups['nodes'].createVariable('glows_wth_min', 'f8', ('num_nodes',), fill_value=-9999.)
    sword.groups['nodes'].createVariable('glows_wth_max', 'f8', ('num_nodes',), fill_value=-9999.)
    sword.groups['nodes'].createVariable('glows_wth_1sig', 'f8', ('num_nodes',), fill_value=-9999.)
    node_keep = np.where(wth_median > -9999)[0]
    sword.groups['nodes'].variables['glows_wth_med'][node_keep] = wth_median[node_keep]
    sword.groups['nodes'].variables['glows_wth_min'][node_keep] = wth_min[node_keep]
    sword.groups['nodes'].variables['glows_wth_max'][node_keep] = wth_max[node_keep]
    sword.groups['nodes'].variables['glows_wth_1sig'][node_keep] = wth_1sigma[node_keep]
    #reaches
    sword.groups['reaches'].createVariable('glows_wth_med', 'f8', ('num_reaches',), fill_value=-9999.)
    sword.groups['reaches'].createVariable('glows_wth_min', 'f8', ('num_reaches',), fill_value=-9999.)
    sword.groups['reaches'].createVariable('glows_wth_max', 'f8', ('num_reaches',), fill_value=-9999.)
    sword.groups['reaches'].createVariable('glows_wth_1sig', 'f8', ('num_reaches',), fill_value=-9999.)
    rch_keep = np.where(rch_wth_median > -9999)[0]
    sword.groups['reaches'].variables['glows_wth_med'][rch_keep] = rch_wth_median[rch_keep]
    sword.groups['reaches'].variables['glows_wth_min'][rch_keep] = rch_wth_min[rch_keep]
    sword.groups['reaches'].variables['glows_wth_max'][rch_keep] = rch_wth_max[rch_keep]
    sword.groups['reaches'].variables['glows_wth_1sig'][rch_keep] = rch_wth_1sigma[rch_keep]
    sword.close()
end = time.time()
print(str(np.round((end-start)/60,2))+' min')

print('DONE in: ')
end_all = time.time()
print(str(np.round((end_all-start_all)/3600,2))+' hrs')

### stats and plotting 
data = np.where(wth_median[basin]>-9999)[0]
rch_basins = np.array([int(str(ind)[0:len(basin_id)]) for ind in rid])
rbasin = np.where(rch_basins == int(basin_id))[0]
rdata = np.where(rch_wth_median[rbasin]>-9999)[0]

print(len(data)/len(nid[basin])*100, '% Node Coverage') #89% node coverage in Ohio Basin. 
print(len(rdata)/len(nid[rbasin])*100, '% Reach Coverage')

# plt.scatter(nlon[basin], nlat[basin], c='blue', s=5)
# plt.scatter(nlon[basin[data]], nlat[basin[data]], c='red', s=5)
# plt.show()

# plt.scatter(rlon[rbasin], rlat[rbasin], c='black', s=5)
# plt.scatter(rlon[rbasin[rdata]], rlat[rbasin[rdata]], c=np.log(rch_wth_median[rbasin[rdata]]), s=5, cmap='rainbow')
# plt.show()

# ndata = np.where(glows_ids != 'NaN')[0]
# plt.scatter(nlon, nlat, c='black', s=5)
# plt.scatter(nlon[ndata], nlat[ndata], c='red', s=5)
# plt.show()